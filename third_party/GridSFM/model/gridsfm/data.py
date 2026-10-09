"""Data loading and pre-forward preprocessing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

import torch
from torch_geometric.data import Batch, HeteroData

from .blocks import DEFAULT_INPUT_DIMS
from .cycle_basis import prepare_for_grid_transformer_
from .pe_features import attach_pe_features_


_CANONICAL_EDGE_TYPES = {
    ("bus", "ac_line",     "bus"):                    9,
    ("bus", "transformer", "bus"):                   11,
    ("generator", "generator_link", "bus"):        None,
    ("bus",       "generator_link", "generator"):  None,
    ("load",      "load_link",      "bus"):        None,
    ("bus",       "load_link",      "load"):       None,
    ("shunt",     "shunt_link",     "bus"):        None,
    ("bus",       "shunt_link",     "shunt"):      None,
    ("bus",       "endpoint_of", "branch_ac"):       1,
    ("branch_ac", "endpoint_of", "bus"):              1,
    ("bus",       "endpoint_of", "branch_tr"):       1,
    ("branch_tr", "endpoint_of", "bus"):              1,
    ("cycle",     "in_cycle",    "branch_ac"):       1,
    ("branch_ac", "in_cycle",    "cycle"):            1,
    ("cycle",     "in_cycle",    "branch_tr"):       1,
    ("branch_tr", "in_cycle",    "cycle"):            1,
}

_BUS_COLS = 4
_GEN_COLS = 11
_AC_LINE_COLS = 9
_TRANSFORMER_COLS = 11


def _to_tensor(rows) -> torch.Tensor:
    if not rows:
        raise ValueError(
            "_to_tensor called with empty rows; the loader should skip empty "
            "node/edge types instead of producing a zero-width tensor."
        )
    return torch.tensor(rows, dtype=torch.float32)


def _check_sender_receiver(e: dict, path, et: str) -> None:
    if len(e["senders"]) != len(e["receivers"]):
        raise ValueError(
            f"{path}: edges.{et} senders ({len(e['senders'])}) and "
            f"receivers ({len(e['receivers'])}) length mismatch."
        )


def _wire_edges_into(d: HeteroData, edges: dict, path) -> None:
    for et in ("ac_line", "transformer"):
        if et in edges and edges[et].get("senders"):
            e = edges[et]
            _check_sender_receiver(e, path, et)
            if not e.get("features"):
                raise ValueError(
                    f"{path}: edges.{et} has senders/receivers but no features; "
                    f"branch edge_attr is required (used by branch-node promotion)."
                )
            d["bus", et, "bus"].edge_index = torch.tensor(
                [e["senders"], e["receivers"]], dtype=torch.long
            )
            d["bus", et, "bus"].edge_attr = _to_tensor(e["features"])
    for et in ("generator_link", "load_link", "shunt_link"):
        if et in edges and edges[et].get("senders"):
            e = edges[et]
            _check_sender_receiver(e, path, et)
            src = et.split("_")[0]
            senders = torch.tensor(e["senders"], dtype=torch.long)
            receivers = torch.tensor(e["receivers"], dtype=torch.long)
            if (src, et, "bus") not in d.edge_types:
                d[src, et, "bus"].edge_index = torch.stack([senders, receivers], dim=0)
            if ("bus", et, src) not in d.edge_types:
                d["bus", et, src].edge_index = torch.stack([receivers, senders], dim=0)


def _fill_missing_node_tensors(d: HeteroData) -> None:
    for nt, dim in DEFAULT_INPUT_DIMS.items():
        if nt not in d.node_types or not hasattr(d[nt], "x") or d[nt].x is None:
            d[nt].x = torch.zeros((0, dim), dtype=torch.float32)


def load_pyg_json(path: Union[str, Path]) -> HeteroData:
    """Load a `.pyg.json` scenario into a HeteroData (no schema validation).

    Reads the GridSFM PyG-flavored JSON: `{grid: {nodes: {bus, generator,
    load, shunt}, edges: {ac_line, transformer, ...}}}`. Returns a raw
    HeteroData; the caller is responsible for calling `prepare_for_inference`
    before the model forward.
    """
    with open(path) as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse {path}: {e}") from e
    if "grid" not in obj or "nodes" not in obj.get("grid", {}):
        raise ValueError(f"{path}: expected top-level 'grid' with 'nodes' (and optionally 'edges').")
    g = obj["grid"]
    nodes = g["nodes"]
    if not nodes.get("bus"):
        raise ValueError(f"{path}: missing or empty 'bus' nodes.")

    d = HeteroData()
    for nt in ("bus", "generator", "load", "shunt"):
        if nt in nodes and nodes[nt]:
            d[nt].x = _to_tensor(nodes[nt])

    _wire_edges_into(d, g.get("edges", {}), path)
    return d


def load_opfdata(path: Union[str, Path]) -> HeteroData:
    """Load an OPFData JSON scenario into a HeteroData WITH schema validation.

    Same envelope shape as `load_pyg_json` but enforces OPFData's exact
    column counts on `bus`, `generator`, `ac_line.features`, and
    `transformer.features`. Use this for the OPFData benchmark format
    (https://arxiv.org/abs/2406.07234); use `load_pyg_json` for the
    GridSFM `.pyg.json` variant.
    """
    with open(path) as f:
        try:
            scn = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse {path}: {e}") from e
    if "grid" not in scn or "nodes" not in scn.get("grid", {}):
        raise ValueError(
            f"{path}: expected top-level 'grid' with 'nodes' (and optionally 'edges')."
        )
    grid = scn["grid"]
    nodes = grid["nodes"]
    edges = grid.get("edges", {})
    if not nodes.get("bus"):
        raise ValueError(f"{path}: missing or empty 'bus' nodes.")

    bus_rows = nodes["bus"]
    if bus_rows and len(bus_rows[0]) != _BUS_COLS:
        raise ValueError(
            f"{path}: bus rows have {len(bus_rows[0])} cols, expected exactly "
            f"{_BUS_COLS}; see gridsfm/schema.py."
        )
    if nodes.get("generator"):
        gen_rows = nodes["generator"]
        if gen_rows and len(gen_rows[0]) != _GEN_COLS:
            raise ValueError(
                f"{path}: generator rows have {len(gen_rows[0])} cols, "
                f"expected exactly {_GEN_COLS}; see gridsfm/schema.py."
            )
    for et, want in (("ac_line", _AC_LINE_COLS), ("transformer", _TRANSFORMER_COLS)):
        ef = edges.get(et, {}).get("features")
        if ef and len(ef[0]) != want:
            raise ValueError(
                f"{path}: edges.{et}.features rows have {len(ef[0])} cols, "
                f"expected exactly {want}; see gridsfm/schema.py."
            )

    d = HeteroData()
    d["bus"].x = _to_tensor(nodes["bus"])
    for nt in ("generator", "load", "shunt"):
        if nt in nodes and nodes[nt]:
            d[nt].x = _to_tensor(nodes[nt])

    _wire_edges_into(d, edges, path)
    return d


def prepare_for_inference(data: HeteroData) -> HeteroData:
    """Mutate `data` in place to add cycle-basis + Hodge PE features and
    fill in any missing canonical node tensors.

    Required before the model forward (HodgePE reads `data['cycle']`).
    Idempotent: both transforms check for prior application and short-
    circuit. Returns the same `data` object for chaining.
    """
    prepare_for_grid_transformer_(data)
    attach_pe_features_(data)
    _fill_missing_node_tensors(data)
    return data


def batch_data_list(data_list: List[HeteroData], copy: bool = True) -> Batch:
    """Collate a list of prepared HeteroData scenarios into a PyG `Batch`.

    Fills missing canonical edge types with empty tensors so a
    heterogeneous list (e.g. some grids without transformers) still
    batches cleanly. `copy=True` (default) deep-copies inputs first to
    avoid mutating caller-held HeteroData.
    """
    if copy:
        # `.clone()` is ~5000x faster than `copy.deepcopy` on case6470-scale
        # HeteroData (PyG's tuned per-store tensor-clone vs Python's memo
        # machinery).
        data_list = [d.clone() for d in data_list]
    for d in data_list:
        _fill_missing_node_tensors(d)
        for et, ea_dim in _CANONICAL_EDGE_TYPES.items():
            if et not in d.edge_types:
                d[et].edge_index = torch.zeros((2, 0), dtype=torch.long)
                if ea_dim is not None:
                    d[et].edge_attr = torch.zeros((0, ea_dim), dtype=torch.float32)
        bus_x = d["bus"].x
        n_bus = int(bus_x.size(0))
        if not hasattr(d["bus"], "v_setpoint") or d["bus"].v_setpoint is None:
            d["bus"].v_setpoint = torch.zeros((n_bus, 1), dtype=bus_x.dtype)
    batch = Batch.from_data_list(data_list)
    missing = [i for i, d in enumerate(data_list)
               if not (hasattr(d, "g_ctx") and isinstance(d.g_ctx, torch.Tensor))]
    if missing:
        raise ValueError(
            f"batch_data_list: {len(missing)} of {len(data_list)} graphs are missing "
            f"`g_ctx` (indices {missing[:5]}{'...' if len(missing) > 5 else ''}). "
            f"All graphs must go through `prepare_for_inference(data)` before batching."
        )
    g_ctxs = [d.g_ctx for d in data_list]
    batch.g_ctx = torch.cat(
        [g.reshape(1, -1) if g.dim() == 1 else g for g in g_ctxs], dim=0,
    )
    return batch
