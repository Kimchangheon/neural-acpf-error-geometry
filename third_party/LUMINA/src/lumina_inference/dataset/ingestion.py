"""Flexible ingestion layer for LUMINA inference.

Provides multiple entry points for loading data into HeteroData format
compatible with LUMINA models:

- :func:`load_from_json_file` — Load from a JSON file path
- :func:`load_from_json_string` — Load from a JSON string
- :func:`load_from_dict` — Load from a Python dictionary
- :func:`load_from_matpower` — Load from a MATPOWER ``.m`` file
- :func:`build_hetero_data` — Build HeteroData from raw node/edge dicts
- :func:`process_opf_dict` — Core OPF dict → HeteroData converter

Copyright 2026 UChicago Argonne, LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing
permissions and limitations under the License.
"""

import json
import re
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import HeteroData

from lumina_inference.dataset.utils import extract_edge_index, extract_edge_index_rev
from lumina_inference.dataset.validation import (
    MissingDataError,
    SchemaValidationError,
)


# ---------------------------------------------------------------------------
# Core OPF dict → HeteroData converter
# ---------------------------------------------------------------------------


def process_opf_dict(
    obj: Dict[str, Any],
    require_solution: bool = False,
) -> HeteroData:
    """Convert an OPF data dictionary to a HeteroData object.

    This is the core conversion function extracted from
    ``process_json_file``. It handles the standard OPF JSON schema
    with ``grid``, ``solution``, and ``metadata`` top-level keys.

    When ``require_solution`` is False (the default for inference),
    the ``solution`` key is optional. If absent, ``.y`` and
    ``.edge_label`` attributes are not set on the returned HeteroData.

    Args:
        obj: Dictionary with OPF data. Must contain ``'grid'`` key.
            Optionally contains ``'solution'`` and ``'metadata'``.
        require_solution: If True, raise :class:`MissingDataError`
            when ``solution`` is absent. Defaults to False.

    Returns:
        HeteroData: Processed heterogeneous graph data object.

    Raises:
        MissingDataError: If ``'grid'`` key is missing, or if
            ``require_solution`` is True and ``'solution'`` is absent.
        SchemaValidationError: If required grid structure is invalid.
    """
    if "grid" not in obj:
        raise MissingDataError(
            "Input dictionary must contain a 'grid' key.",
            missing_field="grid",
            context="OPF data dictionary",
        )

    grid = obj["grid"]
    solution = obj.get("solution", None)
    metadata = obj.get("metadata", {})

    if require_solution and solution is None:
        raise MissingDataError(
            "Solution data is required but missing from input.",
            missing_field="solution",
            context="OPF data dictionary",
        )

    # Validate grid structure
    if "nodes" not in grid:
        raise SchemaValidationError(
            "Grid dictionary must contain a 'nodes' key.",
            expected="grid.nodes",
            actual=str(list(grid.keys())),
        )
    if "edges" not in grid:
        raise SchemaValidationError(
            "Grid dictionary must contain an 'edges' key.",
            expected="grid.edges",
            actual=str(list(grid.keys())),
        )

    # Graph-level properties
    hdata = HeteroData()
    if "context" in grid:
        hdata.baseMVA = torch.tensor(grid["context"]).view(-1).item()

    if "objective" in metadata:
        hdata.objective = torch.tensor(metadata["objective"])

    # --- Node features ---

    # Bus: x = [base_kv, bus_type, vmin, vmax] → one-hot encode bus_type
    if "bus" in grid["nodes"]:
        bus_x = np.array(grid["nodes"]["bus"])
        bus_type = bus_x[:, 1].astype(int)
        # One-hot encode bus_type (4 types: 1=PQ, 2=PV, 3=ref, 4=isolated)
        bus_type_onehot = np.eye(4)[bus_type - 1]
        bus_x_wo_type = np.delete(bus_x, 1, axis=1)
        # Final: [base_kv, vmin, vmax, pq, pv, ref, isolated]
        bus_x_final = np.concatenate([bus_x_wo_type, bus_type_onehot], axis=1)
        hdata["bus"].x = torch.tensor(bus_x_final, dtype=torch.float32)

    # Generator: x = [mbase, pg, pmin, pmax, qg, qmin, qmax, vg, cost_sq, cost_lin, cost_off]
    if "generator" in grid["nodes"]:
        hdata["generator"].x = torch.tensor(
            grid["nodes"]["generator"], dtype=torch.float32
        )

    # Load: x = [pd, qd]
    if "load" in grid["nodes"]:
        hdata["load"].x = torch.tensor(
            grid["nodes"]["load"], dtype=torch.float32
        )

    # Shunt: x = [bs, gs]
    if "shunt" in grid["nodes"]:
        hdata["shunt"].x = torch.tensor(
            grid["nodes"]["shunt"], dtype=torch.float32
        )

    # --- Solution targets (optional) ---

    if solution is not None:
        if "nodes" in solution:
            if "bus" in solution["nodes"]:
                hdata["bus"].y = torch.tensor(
                    solution["nodes"]["bus"], dtype=torch.float32
                )
            if "generator" in solution["nodes"]:
                hdata["generator"].y = torch.tensor(
                    solution["nodes"]["generator"], dtype=torch.float32
                )

        if "edges" in solution:
            if "ac_line" in solution["edges"]:
                hdata["bus", "ac_line", "bus"].edge_label = torch.tensor(
                    solution["edges"]["ac_line"]["features"],
                    dtype=torch.float32,
                )
            if "transformer" in solution["edges"]:
                hdata["bus", "transformer", "bus"].edge_label = torch.tensor(
                    solution["edges"]["transformer"]["features"],
                    dtype=torch.float32,
                )

    # --- Edge indices and attributes ---

    # AC line
    if "ac_line" in grid["edges"]:
        hdata["bus", "ac_line", "bus"].edge_index = extract_edge_index(
            obj, "ac_line"
        )
        if "features" in grid["edges"]["ac_line"]:
            hdata["bus", "ac_line", "bus"].edge_attr = torch.tensor(
                grid["edges"]["ac_line"]["features"], dtype=torch.float32
            )

    # Transformer
    if "transformer" in grid["edges"]:
        hdata["bus", "transformer", "bus"].edge_index = extract_edge_index(
            obj, "transformer"
        )
        if "features" in grid["edges"]["transformer"]:
            hdata["bus", "transformer", "bus"].edge_attr = torch.tensor(
                grid["edges"]["transformer"]["features"],
                dtype=torch.float32,
            )

    # Virtual links (bidirectional)
    if "generator_link" in grid["edges"]:
        hdata[
            "generator", "generator_link", "bus"
        ].edge_index = extract_edge_index(obj, "generator_link")
        hdata[
            "bus", "generator_link", "generator"
        ].edge_index = extract_edge_index_rev(obj, "generator_link")

    if "load_link" in grid["edges"]:
        hdata["load", "load_link", "bus"].edge_index = extract_edge_index(
            obj, "load_link"
        )
        hdata[
            "bus", "load_link", "load"
        ].edge_index = extract_edge_index_rev(obj, "load_link")

    if "shunt_link" in grid["edges"]:
        hdata["shunt", "shunt_link", "bus"].edge_index = extract_edge_index(
            obj, "shunt_link"
        )
        hdata[
            "bus", "shunt_link", "shunt"
        ].edge_index = extract_edge_index_rev(obj, "shunt_link")

    return hdata


# ---------------------------------------------------------------------------
# Public ingestion functions
# ---------------------------------------------------------------------------


def load_from_json_file(
    path: Union[str, Path],
    require_solution: bool = False,
) -> HeteroData:
    """Load OPF data from a JSON file and convert to HeteroData.

    Args:
        path: Path to the JSON file.
        require_solution: If True, raise when solution data is missing.

    Returns:
        HeteroData: Processed heterogeneous graph data.

    Raises:
        FileNotFoundError: If the file does not exist.
        MissingDataError: On JSON decode errors or missing required fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path) as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError as e:
            raise MissingDataError(
                f"Failed to decode JSON from file: {path}. Error: {e}",
                missing_field="valid JSON",
                context=str(path),
            ) from e

    return process_opf_dict(obj, require_solution=require_solution)


def load_from_json_string(
    json_str: str,
    require_solution: bool = False,
) -> HeteroData:
    """Load OPF data from a JSON string and convert to HeteroData.

    Args:
        json_str: JSON string containing OPF data.
        require_solution: If True, raise when solution data is missing.

    Returns:
        HeteroData: Processed heterogeneous graph data.

    Raises:
        MissingDataError: On JSON decode errors or missing required fields.
    """
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise MissingDataError(
            f"Failed to decode JSON string. Error: {e}",
            missing_field="valid JSON",
            context="JSON string input",
        ) from e

    return process_opf_dict(obj, require_solution=require_solution)


def load_from_dict(
    data_dict: Dict[str, Any],
    require_solution: bool = False,
) -> HeteroData:
    """Load OPF data from a Python dictionary and convert to HeteroData.

    The dictionary must follow the OPF JSON schema with at minimum
    a ``'grid'`` key containing ``'nodes'`` and ``'edges'``.

    Args:
        data_dict: Python dictionary with OPF data.
        require_solution: If True, raise when solution data is missing.

    Returns:
        HeteroData: Processed heterogeneous graph data.

    Raises:
        MissingDataError: If required fields are missing.
    """
    return process_opf_dict(data_dict, require_solution=require_solution)


def load_from_matpower(
    path: Union[str, Path],
) -> HeteroData:
    """Load a MATPOWER ``.m`` case file and convert to HeteroData.

    Parses the MATPOWER file using regex to extract raw matrix data,
    then converts to pandapower format for bus ID mapping, and finally
    builds the OPF dict structure that is converted to HeteroData.

    This function requires ``pandapower`` to be installed.

    Args:
        path: Path to the MATPOWER ``.m`` file.

    Returns:
        HeteroData: Processed heterogeneous graph data (inference-only,
        no solution targets).

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If pandapower is not installed.
        ValueError: If the file cannot be parsed.
    """
    try:
        import pandapower as pp
        from pandapower.pypower.idx_brch import (
            BR_R,
            BR_X,
            F_BUS,
            RATE_A,
            RATE_B,
            RATE_C,
            SHIFT,
            T_BUS,
            TAP,
        )
        from pandapower.pypower.idx_bus import (
            BASE_KV,
            BS,
            BUS_I,
            BUS_TYPE,
            GS,
            VMAX,
            VMIN,
        )
        from pandapower.pypower.idx_gen import (
            GEN_BUS,
            MBASE,
            PG,
            PMAX,
            PMIN,
            QG,
            QMAX,
            QMIN,
            VG,
        )
    except ImportError as e:
        raise ImportError(
            "pandapower is required for MATPOWER file loading. "
            "Install it with: pip install pandapower"
        ) from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MATPOWER file not found: {path}")

    # Parse raw .m file to get original values
    raw_mpc = _parse_mfile_raw(path)
    raw_bus = raw_mpc["bus"]
    raw_gen = raw_mpc["gen"]
    raw_branch = raw_mpc["branch"]
    raw_gencost = raw_mpc["gencost"]
    baseMVA = raw_mpc["baseMVA"]

    # Load via pandapower for bus ID mapping
    content = path.read_text(encoding="utf-8")
    patched_content = _patch_matpower_content_for_pandapower(content)

    fd, tmp_path = tempfile.mkstemp(suffix=".m")
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as tmp:
            tmp.write(patched_content)
        net = pp.converter.from_mpc(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    ppc = pp.converter.pypower.to_ppc(net, init="flat")

    # Ensure branch data has required columns
    branch = ppc["branch"]
    if branch.shape[1] < 24:
        missing_cols = 24 - branch.shape[1]
        padding = np.zeros((branch.shape[0], missing_cols))
        branch = np.hstack([branch, padding])
        ppc["branch"] = branch

    # Build bus ID mapping
    ppc_bus_ids = ppc["bus"][:, BUS_I].astype(int)
    bus_lookup = getattr(net, "_pd2ppc_lookups", {}).get("bus")
    net_bus_index = net.bus.index.to_numpy()

    if len(net_bus_index) == raw_bus.shape[0]:
        bus_id_to_net_idx = {
            int(bus_id): int(net_idx)
            for bus_id, net_idx in zip(raw_bus[:, 0], net_bus_index)
        }
    else:
        net_index_set = set(int(x) for x in net_bus_index)
        bus_id_to_net_idx = {}
        for bus_id in raw_bus[:, 0]:
            candidate = int(bus_id) - 1
            if candidate not in net_index_set:
                raise ValueError(
                    f"Cannot map bus ID to net index: {int(bus_id)}"
                )
            bus_id_to_net_idx[int(bus_id)] = candidate

    def _ppc_idx_for_bus(bus_id: int) -> int:
        if bus_lookup is None:
            raise ValueError(
                "pandapower bus lookup missing; cannot map bus IDs"
            )
        net_idx = bus_id_to_net_idx.get(int(bus_id))
        if net_idx is None:
            raise ValueError(f"Unknown bus ID (no net index): {bus_id}")
        if net_idx < 0 or net_idx >= len(bus_lookup):
            raise ValueError(
                f"Unknown bus ID (net index out of range): {bus_id}"
            )
        idx = int(bus_lookup[net_idx])
        if idx < 0:
            raise ValueError(f"Unknown bus ID (no ppc index): {bus_id}")
        return idx

    def _map_bus_ids(ids: np.ndarray, *, context: str) -> np.ndarray:
        mapped = []
        missing = []
        for bid in ids:
            b = int(bid)
            try:
                mapped.append(_ppc_idx_for_bus(b))
            except ValueError:
                missing.append(b)
        if missing:
            missing_sorted = sorted(set(missing))
            raise ValueError(
                f"Unknown bus IDs in {context}: {missing_sorted[:10]}"
            )
        return np.array(mapped, dtype=int)

    # Convert generator powers to per-unit
    raw_gen_pu = np.array(raw_gen, copy=True)
    raw_gen_pu[:, [PG, QG, QMAX, QMIN, PMAX, PMIN]] = (
        raw_gen_pu[:, [PG, QG, QMAX, QMIN, PMAX, PMIN]] / baseMVA
    )

    # Zero out limits for offline generators
    gen_status = raw_gen[:, 7]  # GEN_STATUS
    offline_mask = gen_status == 0
    if offline_mask.any():
        raw_gen_pu[offline_mask, PMIN] = 0.0
        raw_gen_pu[offline_mask, PMAX] = 0.0
        raw_gen_pu[offline_mask, QMIN] = 0.0
        raw_gen_pu[offline_mask, QMAX] = 0.0

    # Process gencost
    gencost = np.array(raw_gencost, copy=True)
    if np.any(gencost[:, 0] != 2):
        bad_rows = np.where(gencost[:, 0] != 2)[0].tolist()
        raise ValueError(
            f"Unsupported gencost model in rows {bad_rows}; expected model=2"
        )
    if np.any(gencost[:, 3] < 3):
        bad_rows = np.where(gencost[:, 3] < 3)[0].tolist()
        raise ValueError(
            f"Unsupported gencost order in rows {bad_rows}; expected n>=3"
        )
    for row_idx in np.where(gencost[:, 3] > 3)[0]:
        ncost = int(gencost[row_idx, 3])
        if np.any(gencost[row_idx, 4: 4 + ncost - 3] != 0):
            raise ValueError(
                f"Unsupported non-zero higher-order gencost terms in row "
                f"{int(row_idx)}"
            )
        gencost[row_idx, 4:7] = gencost[row_idx, 4 + ncost - 3: 4 + ncost]
        gencost[row_idx, 3] = 3

    # Convert to per-unit power basis
    gencost[:, 4] = gencost[:, 4] * (baseMVA**2)  # c2
    gencost[:, 5] = gencost[:, 5] * baseMVA  # c1
    gencost = gencost.astype(np.float32)

    cost_squared = gencost[:, 4]
    cost_linear = gencost[:, 5]
    cost_offset = gencost[:, 6]

    # Build generator features
    mbase = raw_gen[:, 6]
    pg = raw_gen_pu[:, 1]
    pmin = raw_gen_pu[:, 9]
    pmax = raw_gen_pu[:, 8]
    qg = raw_gen_pu[:, 2]
    qmin = raw_gen_pu[:, 4]
    qmax = raw_gen_pu[:, 3]
    vg = raw_gen[:, 5]

    gen_features = np.column_stack(
        [mbase, pg, pmin, pmax, qg, qmin, qmax, vg,
         cost_squared, cost_linear, cost_offset]
    ).astype(np.float32)

    # Identify transformers vs AC lines
    branch_in_service_mask = raw_branch[:, 10] > 0
    xfmr_mask = (raw_branch[:, 8] != 0) & branch_in_service_mask
    ac_line_mask = (raw_branch[:, 8] == 0) & branch_in_service_mask

    raw_ac_branches = raw_branch[ac_line_mask]
    raw_xfmr_branches = raw_branch[xfmr_mask]

    # Build AC line features
    if len(raw_ac_branches) > 0:
        ac_br_b = raw_ac_branches[:, 4]
        ac_b_fr = ac_br_b / 2.0
        ac_b_to = ac_br_b / 2.0
        ac_line_features = np.column_stack(
            [
                np.radians(raw_ac_branches[:, 11]),  # angmin
                np.radians(raw_ac_branches[:, 12]),  # angmax
                ac_b_fr,
                ac_b_to,
                raw_ac_branches[:, 2],  # r
                raw_ac_branches[:, 3],  # x
                raw_ac_branches[:, 5] / baseMVA,  # rateA
                raw_ac_branches[:, 6] / baseMVA,  # rateB
                raw_ac_branches[:, 7] / baseMVA,  # rateC
            ]
        ).astype(np.float32)
        ac_senders = _map_bus_ids(
            raw_ac_branches[:, 0], context="ac_line_from"
        )
        ac_receivers = _map_bus_ids(
            raw_ac_branches[:, 1], context="ac_line_to"
        )
    else:
        ac_line_features = np.zeros((0, 9), dtype=np.float32)
        ac_senders = np.array([], dtype=int)
        ac_receivers = np.array([], dtype=int)

    # Build transformer features
    if len(raw_xfmr_branches) > 0:
        xfmr_b_fr = raw_xfmr_branches[:, 4] / 2.0
        xfmr_b_to = raw_xfmr_branches[:, 4] / 2.0
        transformer_features = np.column_stack(
            [
                np.radians(raw_xfmr_branches[:, 11]),  # angmin
                np.radians(raw_xfmr_branches[:, 12]),  # angmax
                raw_xfmr_branches[:, 2],  # r
                raw_xfmr_branches[:, 3],  # x
                raw_xfmr_branches[:, 5] / baseMVA,  # rateA
                raw_xfmr_branches[:, 6] / baseMVA,  # rateB
                raw_xfmr_branches[:, 7] / baseMVA,  # rateC
                raw_xfmr_branches[:, 8],  # tap
                np.radians(raw_xfmr_branches[:, 9]),  # shift
                xfmr_b_fr,
                xfmr_b_to,
            ]
        ).astype(np.float32)
        xfmr_senders = _map_bus_ids(
            raw_xfmr_branches[:, 0], context="transformer_from"
        )
        xfmr_receivers = _map_bus_ids(
            raw_xfmr_branches[:, 1], context="transformer_to"
        )
    else:
        transformer_features = np.zeros((0, 11), dtype=np.float32)
        xfmr_senders = np.array([], dtype=int)
        xfmr_receivers = np.array([], dtype=int)

    # Build bus features
    n_bus = len(ppc_bus_ids)
    vmin = np.zeros(n_bus, dtype=np.float32)
    vmax = np.zeros(n_bus, dtype=np.float32)
    for row in raw_bus:
        bus_id = int(row[0])
        idx = _ppc_idx_for_bus(bus_id)
        vmin[idx] = float(row[12])  # Vmin
        vmax[idx] = float(row[11])  # Vmax

    bus_features = np.column_stack(
        [
            ppc["bus"][:, BASE_KV],
            ppc["bus"][:, BUS_TYPE],
            vmin,
            vmax,
        ]
    ).astype(np.float32)

    # Build load features
    load_features_list = []
    load_receivers_list = []
    for row in raw_bus:
        pd = float(row[2])
        qd = float(row[3])
        if pd == 0.0 and qd == 0.0:
            continue
        bus_id = int(row[0])
        idx = _ppc_idx_for_bus(bus_id)
        load_features_list.append([pd / baseMVA, qd / baseMVA])
        load_receivers_list.append(idx)

    if load_features_list:
        load_features = np.array(load_features_list, dtype=np.float32)
        load_senders = np.arange(len(load_receivers_list))
        load_receivers = np.array(load_receivers_list, dtype=int)
    else:
        load_features = np.zeros((0, 2), dtype=np.float32)
        load_senders = np.array([], dtype=int)
        load_receivers = np.array([], dtype=int)

    # Build shunt features
    shunt_features_list = []
    shunt_receivers_list = []
    for row in raw_bus:
        gs = float(row[4])
        bs = float(row[5])
        if gs == 0.0 and bs == 0.0:
            continue
        bus_id = int(row[0])
        idx = _ppc_idx_for_bus(bus_id)
        shunt_features_list.append([bs / baseMVA, gs / baseMVA])
        shunt_receivers_list.append(idx)

    if shunt_features_list:
        shunt_features = np.array(shunt_features_list, dtype=np.float32)
        shunt_senders = np.arange(len(shunt_receivers_list))
        shunt_receivers = np.array(shunt_receivers_list, dtype=int)
    else:
        shunt_features = np.zeros((0, 2), dtype=np.float32)
        shunt_senders = np.array([], dtype=int)
        shunt_receivers = np.array([], dtype=int)

    # Build generator link
    gen_senders = np.arange(len(raw_gen))
    gen_receivers = _map_bus_ids(raw_gen[:, 0], context="generator_link")

    # Assemble OPF dict
    opf_dict = {
        "grid": {
            "context": [[[float(baseMVA)]]],
            "nodes": {
                "bus": bus_features.tolist(),
                "generator": gen_features.tolist(),
                "load": load_features.tolist(),
                "shunt": shunt_features.tolist(),
            },
            "edges": {
                "ac_line": {
                    "senders": ac_senders.tolist(),
                    "receivers": ac_receivers.tolist(),
                    "features": ac_line_features.tolist(),
                },
                "transformer": {
                    "senders": xfmr_senders.tolist(),
                    "receivers": xfmr_receivers.tolist(),
                    "features": transformer_features.tolist(),
                },
                "generator_link": {
                    "senders": gen_senders.tolist(),
                    "receivers": gen_receivers.tolist(),
                },
                "load_link": {
                    "senders": load_senders.tolist(),
                    "receivers": load_receivers.tolist(),
                },
                "shunt_link": {
                    "senders": shunt_senders.tolist(),
                    "receivers": shunt_receivers.tolist(),
                },
            },
        },
        "metadata": {"objective": 0.0},
    }

    return process_opf_dict(opf_dict, require_solution=False)


# ---------------------------------------------------------------------------
# Generic HeteroData builder
# ---------------------------------------------------------------------------


def build_hetero_data(
    nodes: Dict[str, Dict[str, Any]],
    edges: Dict[Tuple[str, str, str], Dict[str, Any]],
    graph_attrs: Optional[Dict[str, Any]] = None,
) -> HeteroData:
    """Build a HeteroData object from raw node and edge dictionaries.

    This is the generic entry point for non-OPF data. It constructs
    a HeteroData object from arbitrary node types and edge types
    without assuming any specific schema.

    Args:
        nodes: Dictionary mapping node type names to their data.
            Each value is a dict that may contain:

            - ``'x'``: Feature array (required), shape ``[num_nodes, num_features]``
            - ``'y'``: Target array (optional), shape ``[num_nodes, num_targets]``

        edges: Dictionary mapping edge type tuples ``(src, rel, dst)``
            to their data. Each value is a dict that may contain:

            - ``'edge_index'``: Edge index array (required), shape ``[2, num_edges]``
            - ``'edge_attr'``: Edge attribute array (optional)
            - ``'edge_label'``: Edge label array (optional)

        graph_attrs: Optional dictionary of graph-level attributes
            to attach to the HeteroData object.

    Returns:
        HeteroData: Constructed heterogeneous graph data.

    Raises:
        MissingDataError: If required fields (``x``, ``edge_index``)
            are missing.

    Example::

        data = build_hetero_data(
            nodes={
                'bus': {'x': bus_features},
                'generator': {'x': gen_features},
            },
            edges={
                ('bus', 'ac_line', 'bus'): {
                    'edge_index': [[0, 1], [1, 0]],
                    'edge_attr': line_features,
                },
            },
            graph_attrs={'baseMVA': 100.0},
        )
    """
    hdata = HeteroData()

    # Set graph-level attributes
    if graph_attrs is not None:
        for key, value in graph_attrs.items():
            setattr(hdata, key, value)

    # Set node data
    for node_type, node_data in nodes.items():
        if "x" not in node_data:
            raise MissingDataError(
                f"Node type '{node_type}' must have an 'x' (features) entry.",
                missing_field="x",
                context=f"nodes['{node_type}']",
            )
        x = node_data["x"]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.array(x))
        hdata[node_type].x = x

        if "y" in node_data:
            y = node_data["y"]
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(np.array(y))
            hdata[node_type].y = y

    # Set edge data
    for edge_type, edge_data in edges.items():
        if "edge_index" not in edge_data:
            raise MissingDataError(
                f"Edge type {edge_type} must have an 'edge_index' entry.",
                missing_field="edge_index",
                context=f"edges[{edge_type}]",
            )
        edge_index = edge_data["edge_index"]
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(np.array(edge_index), dtype=torch.long)
        hdata[edge_type].edge_index = edge_index

        if "edge_attr" in edge_data:
            edge_attr = edge_data["edge_attr"]
            if not isinstance(edge_attr, torch.Tensor):
                edge_attr = torch.tensor(np.array(edge_attr))
            hdata[edge_type].edge_attr = edge_attr

        if "edge_label" in edge_data:
            edge_label = edge_data["edge_label"]
            if not isinstance(edge_label, torch.Tensor):
                edge_label = torch.tensor(np.array(edge_label))
            hdata[edge_type].edge_label = edge_label

    return hdata


# ---------------------------------------------------------------------------
# Internal MATPOWER parsing helpers
# ---------------------------------------------------------------------------


def _parse_mfile_raw(filepath: Path) -> Dict[str, Any]:
    """Parse a MATPOWER .m file using regex to extract raw matrix data.

    Bypasses pandapower's modifications to retrieve original values,
    particularly generator pg/qg setpoints and bus vmin/vmax.

    Args:
        filepath: Path to the MATPOWER .m file.

    Returns:
        Dictionary with keys: ``baseMVA``, ``bus``, ``gen``, ``branch``,
        ``gencost`` (all as numpy arrays except baseMVA which is float).
    """
    content = filepath.read_text(encoding="utf-8")

    # Extract baseMVA
    basemva_match = re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)", content)
    if not basemva_match:
        raise ValueError("Could not find baseMVA in MATPOWER file")
    baseMVA = float(basemva_match.group(1))

    def extract_matrix(name: str) -> np.ndarray:
        pattern = rf"mpc\.{name}\s*=\s*\[(.*?)\];"
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            raise ValueError(f"Could not find mpc.{name} in MATPOWER file")

        matrix_str = match.group(1)
        lines = []
        for line in matrix_str.split("\n"):
            line = re.sub(r"%.*$", "", line)
            line = line.strip()
            if line:
                line = line.rstrip(";")
                lines.append(line)

        rows = []
        for line in lines:
            values = line.split()
            if values:
                row = [float(v) for v in values]
                rows.append(row)

        return np.array(rows, dtype=np.float64)

    return {
        "baseMVA": baseMVA,
        "bus": extract_matrix("bus"),
        "gen": extract_matrix("gen"),
        "branch": extract_matrix("branch"),
        "gencost": extract_matrix("gencost"),
    }


def _patch_matpower_content_for_pandapower(content: str) -> str:
    """Patch MATPOWER content to avoid pandapower gencost conversion issues.

    Handles 'fake' cubic costs (NCOST=4 with 0 coefficient for cubic term),
    converting them to quadratic (NCOST=3).
    """
    lines = content.splitlines()
    new_lines: List[str] = []
    in_gencost = False

    for line in lines:
        if "mpc.gencost" in line and "=" in line and "[" in line:
            in_gencost = True
            new_lines.append(line)
            continue
        if in_gencost and "];" in line:
            in_gencost = False
            new_lines.append(line)
            continue

        if in_gencost:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "2" and parts[3] == "4":
                try:
                    c3 = float(parts[4])
                    if abs(c3) < 1e-9:
                        parts[3] = "3"
                        parts.pop(4)
                        new_lines.append("\t" + "\t".join(parts))
                        continue
                except ValueError:
                    pass

        new_lines.append(line)

    return "\n".join(new_lines)
