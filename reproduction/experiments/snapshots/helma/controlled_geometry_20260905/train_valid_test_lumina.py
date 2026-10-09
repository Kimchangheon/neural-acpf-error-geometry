#!/usr/bin/env python3
"""
Train/evaluate the released LUMINA-2M surrogate on the PPC LVN parquet pipeline.

This keeps the existing PPC dataloader, split logic, supervised Newton target,
and complex128 AC-PF residual evaluation. Only the neural surrogate is swapped:
LVN branch-row parquet batches are adapted into LUMINA's HeteroData schema
(the PGLib-OPF layout used by argonne/LUMINA-2M).

LUMINA node/edge schema (from lumina_inference.dataset.validation):
  bus        7  [base_kv, vmin, vmax, onehot(PQ, PV, ref, isolated)]
  generator 11  [mbase, pg, pmin, pmax, qg, qmin, qmax, vg, c2, c1, c0]
  load       2  [pd, qd]
  shunt      2  [bs, gs]
  ac_line    9  [angmin, angmax, b_fr, b_to, r, x, rate_a, rate_b, rate_c]
  transformer 11 [angmin, angmax, r, x, rate_a, rate_b, rate_c, tap, shift, b_fr, b_to]

The HGT head emits 2 channels per bus, ordered [theta, |V|], matching the
OPFDataset bus solution layout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch, HeteroData

from Dataset_optimized_complex_columns import ChanghunDataset
from grad_accum import GradAccumulator, add_grad_accum_args, resolve_accum_steps
from preload_dataset import add_preload_args, preload_split
from ddp_utils import (add_ddp_args, barrier, cleanup, make_loader,
                       quiet_non_main, reduce_sums, setup_ddp, unwrap, wrap_model)
from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from train_valid_test_gridfm import (
    Tee,
    angle_diff,
    cap_subset,
    compute_power_flow_residual_metrics,
    ensure_dense_y_for_metrics,
    format_residual_distribution_compact,
    ppc_physics_loss,
    residual_distribution,
)
from train_valid_test_gridsfm import (
    _make_branch_family,
    _map_bus_types,
)
from opf_task import (
    OPF_EXTRA_COLUMNS,
    format_opf_metrics,
    has_opf_columns,
    opf_decision_space,
    opf_loss,
    opf_metrics,
    dispatch_loss,
    gen_head_loss,
    target_dispatch,
)
from known_operator_pf import (
    add_known_operator_args,
    build_known_operator,
    format_known_operator_diagnostics,
    known_operator_tag,
)


LUMINA_NODE_DIMS = {"bus": 7, "generator": 11, "load": 2, "shunt": 2}
LUMINA_EDGE_TYPES = [
    ("bus", "ac_line", "bus"),
    ("bus", "transformer", "bus"),
    ("generator", "generator_link", "bus"),
    ("bus", "generator_link", "generator"),
    ("load", "load_link", "bus"),
    ("bus", "load_link", "load"),
    ("shunt", "shunt_link", "bus"),
    ("bus", "shunt_link", "shunt"),
]
BUS_TYPE_PQ = 1
BUS_TYPE_PV = 2
BUS_TYPE_REF = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="LUMINA-2M released backbone on PPC branch-row parquet data",
    )
    parser.add_argument("--PARQUET", type=str, required=True)
    parser.add_argument(
        "--task",
        choices=("pf", "opf"),
        default="pf",
        help="pf: predict voltages for a given injection. opf: predict the OPF "
             "operating point from loads and the decision space only.",
    )
    parser.add_argument("--opf_limit_weight", type=float, default=1.0)
    parser.add_argument("--opf_band_weight", type=float, default=0.0)
    parser.add_argument("--dispatch_weight", type=float, default=0.0,
                        help="Supervise the dispatch implied by the predicted voltages "
                             "against the reference dispatch. Applies to every model.")
    parser.add_argument("--gen_head_weight", type=float, default=0.0,
                        help="Supervise the model's own generator head. Needed for "
                             "GridSFM, whose angle prior is built from head_Pg through "
                             "a detached path, so that head is otherwise never trained.")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--log_to_file", action="store_true")
    parser.add_argument("--log_dir", type=str, default="./results/logs/lumina")
    parser.add_argument("--ckpt_dir", type=str, default="./results/ckpt/lumina")

    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="",
        help="Released LUMINA weights, e.g. model.safetensors or model.pt.",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="",
        help="LUMINA config.json describing metadata/input_channels/HGT sizes.",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default="argonne/LUMINA-2M",
        help="Hugging Face repo used when --model_config/--pretrained_checkpoint are unset.",
    )
    parser.add_argument(
        "--init_mode",
        choices=("pretrained", "scratch"),
        default="pretrained",
        help="Use released LUMINA weights or a random LUMINA-shaped HGT init.",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default="",
        help="Warm-start from a checkpoint saved by this script (state_dict).",
    )
    parser.add_argument(
        "--retain_init_as_best",
        action="store_true",
        help=(
            "Retain the epoch-0 warm-start state as an early-stopping candidate "
            "after validation, so continuation cannot be selected over a better "
            "pre-continuation checkpoint merely by construction."
        ),
    )

    parser.add_argument("--PER_UNIT", action="store_true")
    parser.add_argument("--target_S_base", type=float, default=None)
    parser.add_argument(
        "--dataset_complex_dtype",
        choices=("complex64", "complex128"),
        default="complex128",
    )
    parser.add_argument("--share_grid", action="store_true")
    parser.add_argument("--share_ybus", action="store_true")
    parser.add_argument("--lazy_parquet", action="store_true")
    parser.add_argument("--row_group_cache_size", type=int, default=2)

    parser.add_argument("--BATCH", type=int, default=4)
    parser.add_argument("--EPOCHS", type=int, default=40)
    parser.add_argument("--LR", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed_value", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.3333)
    parser.add_argument("--valid_ratio", type=float, default=0.3333)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_valid_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--VAL_EVERY", type=int, default=1)

    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--physics_weight", type=float, default=1e-2)
    parser.add_argument(
        "--physics_loss_form",
        choices=("mse", "huber", "logcosh"),
        default="logcosh",
    )
    parser.add_argument("--physics_huber_delta", type=float, default=1.0)
    parser.add_argument(
        "--pf_injection", choices=("signed", "legacy"), default="signed",
        help="signed: carry the net PQ-bus injection with its sign (correct). "
             "legacy: clamp it at zero, reproducing the first campaign.")
    parser.add_argument("--residual_tol_pu", type=float, default=1e-6)
    parser.add_argument("--convergence_tol_pu", type=float, default=1e-6)
    parser.add_argument("--vmin", type=float, default=0.5)
    parser.add_argument("--vmax", type=float, default=1.5)

    parser.add_argument(
        "--treat_voltage_mismatch_as_transformer",
        action="store_true",
        help="Classify branch rows with different endpoint vn_kv as LUMINA transformers.",
    )
    parser.add_argument(
        "--rate_a",
        type=float,
        default=0.0,
        help="Fallback LUMINA branch rate_a feature when the parquet has no rate.",
    )
    parser.add_argument(
        "--no_minmax_scaling",
        action="store_true",
        help="Disable LUMINA's sigmoid min/max rescaling of the voltage head.",
    )
    parser.add_argument(
        "--gen_setpoint_mode",
        choices=("known", "zero"),
        default="known",
        help="Fill generator pg/qg with the known parquet injection or leave zero.",
    )
    add_known_operator_args(parser)
    add_grad_accum_args(parser)
    add_ddp_args(parser)
    add_preload_args(parser)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, train_ratio, valid_ratio, seed):
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test = n - n_train - n_valid
    if min(n_train, n_valid, n_test) <= 0:
        raise ValueError(
            f"Bad split sizes for n={n}: train={n_train}, valid={n_valid}, test={n_test}",
        )
    gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_valid, n_test], generator=gen)


def _bus_features(bus_type: torch.Tensor, vn_kv: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    """LUMINA bus row: [base_kv, vmin, vmax, onehot(PQ, PV, ref, isolated)]."""
    n = bus_type.numel()
    x = torch.zeros((n, LUMINA_NODE_DIMS["bus"]), dtype=torch.float32)
    x[:, 0] = vn_kv
    x[:, 1] = float(vmin)
    x[:, 2] = float(vmax)
    onehot_col = 3 + (bus_type.long().clamp(1, 4) - 1)
    x[torch.arange(n), onehot_col] = 1.0
    return x


def _bus_features_band(bus_type: torch.Tensor, vn_kv: torch.Tensor,
                       vmin: torch.Tensor, vmax: torch.Tensor) -> torch.Tensor:
    """LUMINA bus row with a per-bus voltage band."""
    n = bus_type.numel()
    x = torch.zeros((n, LUMINA_NODE_DIMS["bus"]), dtype=torch.float32)
    x[:, 0] = vn_kv
    x[:, 1] = vmin
    x[:, 2] = vmax
    onehot_col = 3 + (bus_type.long().clamp(1, 4) - 1)
    x[torch.arange(n), onehot_col] = 1.0
    return x


def make_lumina_graphs(batch_cpu: Dict[str, torch.Tensor], args,
                       opf_space: Dict[str, torch.Tensor] = None) -> Tuple[List[HeteroData], torch.Tensor]:
    sizes = batch_cpu["sizes"].long()
    branch_sizes = batch_cpu["branch_sizes"].long()
    offsets = batch_cpu["offsets"].long()
    branch_offsets = torch.cat((branch_sizes.new_zeros(1), torch.cumsum(branch_sizes, 0)[:-1]))

    graphs: List[HeteroData] = []
    target = batch_cpu["V_newton"].squeeze(0).float()

    for n, nl, off, boff in zip(sizes, branch_sizes, offsets, branch_offsets):
        n = int(n.item())
        nl = int(nl.item())
        off = int(off.item())
        boff = int(boff.item())
        bs = slice(off, off + n)
        es = slice(boff, boff + nl)

        bus_type_ppc = batch_cpu["bus_type"].squeeze(0)[bs].long()
        bus_type = _map_bus_types(bus_type_ppc)
        vn_kv = batch_cpu["vn_kv"].squeeze(0)[bs].float()
        Vstart = batch_cpu["V_start"].squeeze(0)[bs].float()
        S = batch_cpu["S_start"].squeeze(0)[bs].to(torch.complex128)
        # Reactive power at slack/PV buses is not a specified quantity: the
        # solver fixes |V| there and solves for Q, so whatever sits in that slot
        # is arbitrary. CGMES exports park reactive-capability figures there --
        # LVN heo1 carries Q = -2507 pu at one 380 kV PV bus against a true
        # maximum of 8.4 pu. Left in, it reaches the generator node (as absurd
        # q_min/q_max) and, because |Q| > 0 passes the load selector below, also
        # invents a 2507 pu reactive load that does not exist. Zero it here so
        # the node features match how the physics loss already treats it.
        _q_not_specified = (bus_type_ppc == 1) | (bus_type_ppc == 2)
        S = torch.complex(S.real, S.imag.masked_fill(_q_not_specified, 0.0))
        Ysh = batch_cpu["Y_shunt_bus"].squeeze(0)[bs].to(torch.complex128)

        d = HeteroData()
        if opf_space is not None and "Bus_vmin" in opf_space:
            # The OPF dataset carries the real voltage band, and LUMINA's sigmoid
            # output scaling maps the magnitude head into exactly this interval.
            # Feeding the CLI default instead would make it predict into a band
            # several times too wide.
            d["bus"].x = _bus_features_band(
                bus_type, vn_kv,
                opf_space["Bus_vmin"][bs].float(),
                opf_space["Bus_vmax"][bs].float(),
            )
        else:
            d["bus"].x = _bus_features(bus_type, vn_kv, args.vmin, args.vmax)

        if opf_space is not None:
            # OPF: the generators are exactly the controllable buses, and their
            # feature row is the real decision space -- limits and cost slope --
            # rather than a band synthesised around a known dispatch. pg/qg stay
            # at zero because the dispatch is the answer, not an input.
            ctrl = opf_space["Gen_controllable"][bs]
            gen_bus = torch.where(ctrl)[0]
            if gen_bus.numel() == 0:
                gen_bus = torch.tensor([0], dtype=torch.long)
            xg = torch.zeros((gen_bus.numel(), LUMINA_NODE_DIMS["generator"]), dtype=torch.float32)
            xg[:, 0] = 1.0
            xg[:, 2] = opf_space["Gen_p_min"][bs][gen_bus].float()
            xg[:, 3] = opf_space["Gen_p_max"][bs][gen_bus].float()
            xg[:, 5] = opf_space["Gen_q_min"][bs][gen_bus].float()
            xg[:, 6] = opf_space["Gen_q_max"][bs][gen_bus].float()
            xg[:, 7] = Vstart[gen_bus, 0]
            if "Gen_cost_c1" in opf_space:
                xg[:, 9] = opf_space["Gen_cost_c1"][bs][gen_bus].float()
        else:
            gen_bus = torch.where((bus_type_ppc == 1) | (bus_type_ppc == 2))[0]
            if gen_bus.numel() == 0:
                gen_bus = torch.tensor([0], dtype=torch.long)
            xg = torch.zeros((gen_bus.numel(), LUMINA_NODE_DIMS["generator"]), dtype=torch.float32)
            Pg0 = S.real[gen_bus].float()
            Qg0 = S.imag[gen_bus].float()
            p_margin = torch.maximum(Pg0.abs() * 0.5, torch.ones_like(Pg0) * 10.0)
            q_margin = torch.maximum(Qg0.abs() * 0.5, torch.ones_like(Qg0) * 10.0)
            xg[:, 0] = 1.0
            if args.gen_setpoint_mode == "known":
                xg[:, 1] = Pg0
                xg[:, 4] = Qg0
            xg[:, 2] = Pg0 - p_margin
            xg[:, 3] = Pg0 + p_margin
            xg[:, 5] = Qg0 - q_margin
            xg[:, 6] = Qg0 + q_margin
            xg[:, 7] = Vstart[gen_bus, 0]
        d["generator"].x = xg
        gen_idx = torch.arange(gen_bus.numel(), dtype=torch.long)
        d["generator", "generator_link", "bus"].edge_index = torch.stack([gen_idx, gen_bus], dim=0)
        d["bus", "generator_link", "generator"].edge_index = torch.stack([gen_bus, gen_idx], dim=0)

        if opf_space is not None or args.pf_injection == "legacy":
            load_bus = torch.where(((-S.real).abs() > 0) | ((-S.imag).abs() > 0))[0]
            pd_ = torch.clamp(-S.real[load_bus].float(), min=0.0)
            qd_ = torch.clamp(-S.imag[load_bus].float(), min=0.0)
        else:
        # PF: S is the net injection and the only statement of what the scenario
        # asks for. Clamping it at zero deleted every bus whose net injection is
        # positive -- 55.5% of SimBench's PQ buses, carrying 99.7% of the
        # active-power magnitude on them -- while the physics loss still scored
        # the prediction against the true S. The model was being asked for a
        # voltage state determined by inputs it never saw. Carry the injection
        # signed, and only on PQ buses: at PV and slack the generator node
        # already encodes it, so a load node there would double count.
            is_pq = (bus_type_ppc != 1) & (bus_type_ppc != 2)
            load_bus = torch.where(is_pq & ((S.real.abs() > 0) | (S.imag.abs() > 0)))[0]
            pd_ = -S.real[load_bus].float()
            qd_ = -S.imag[load_bus].float()
        xl = torch.zeros((load_bus.numel(), LUMINA_NODE_DIMS["load"]), dtype=torch.float32)
        if load_bus.numel() > 0:
            xl[:, 0] = pd_
            xl[:, 1] = qd_
        d["load"].x = xl
        li = torch.arange(load_bus.numel(), dtype=torch.long)
        d["load", "load_link", "bus"].edge_index = torch.stack([li, load_bus], dim=0)
        d["bus", "load_link", "load"].edge_index = torch.stack([load_bus, li], dim=0)

        sh_bus = torch.where(Ysh.abs() > 0)[0]
        xs = torch.zeros((sh_bus.numel(), LUMINA_NODE_DIMS["shunt"]), dtype=torch.float32)
        if sh_bus.numel() > 0:
            xs[:, 0] = Ysh.imag[sh_bus].float()
            xs[:, 1] = Ysh.real[sh_bus].float()
        d["shunt"].x = xs
        si = torch.arange(sh_bus.numel(), dtype=torch.long)
        d["shunt", "shunt_link", "bus"].edge_index = torch.stack([si, sh_bus], dim=0)
        d["bus", "shunt_link", "shunt"].edge_index = torch.stack([sh_bus, si], dim=0)

        f = batch_cpu["Branch_f_bus"].squeeze(0)[es].long() - off
        t = batch_cpu["Branch_t_bus"].squeeze(0)[es].long() - off
        status = batch_cpu["Branch_status"].squeeze(0)[es].bool()
        tau = batch_cpu["Branch_tau"].squeeze(0)[es].float()
        shift = batch_cpu["Branch_shift_deg"].squeeze(0)[es].float()
        yft = batch_cpu["Branch_y_series_ft"].squeeze(0)[es].to(torch.complex128)
        yf = batch_cpu["Branch_y_series_from"].squeeze(0)[es].to(torch.complex128)
        y_series = torch.where(yft.abs() > 1e-12, yft, yf)
        ysh_f = batch_cpu["Branch_y_shunt_from"].squeeze(0)[es].to(torch.complex128)
        ysh_t = batch_cpu["Branch_y_shunt_to"].squeeze(0)[es].to(torch.complex128)

        if "Is_trafo" in batch_cpu:
            is_tr = batch_cpu["Is_trafo"].squeeze(0)[es].bool()
        else:
            is_tr = torch.zeros(nl, dtype=torch.bool)
        is_tr = is_tr | ((tau - 1.0).abs() > 1e-8) | (shift.abs() > 1e-8)
        if args.treat_voltage_mismatch_as_transformer and nl > 0:
            valid_ft = (f >= 0) & (f < n) & (t >= 0) & (t < n)
            vn_mismatch = torch.zeros(nl, dtype=torch.bool)
            vn_mismatch[valid_ft] = (vn_kv[f[valid_ft]] - vn_kv[t[valid_ft]]).abs() > 1e-6
            is_tr = is_tr | vn_mismatch

        ac_ei, ac_attr = _make_branch_family(
            f, t, y_series, ysh_f, ysh_t, tau, shift, status & (~is_tr),
            is_transformer=False, rate_a=args.rate_a,
        )
        tr_ei, tr_attr = _make_branch_family(
            f, t, y_series, ysh_f, ysh_t, tau, shift, status & is_tr,
            is_transformer=True, rate_a=args.rate_a,
        )
        d["bus", "ac_line", "bus"].edge_index = ac_ei
        d["bus", "ac_line", "bus"].edge_attr = ac_attr
        d["bus", "transformer", "bus"].edge_index = tr_ei
        d["bus", "transformer", "bus"].edge_attr = tr_attr

        graphs.append(d)

    return graphs, target


def _dicts_from_batch(gbatch, device):
    """x_dict/edge_index_dict with every LUMINA node and edge type present."""
    x_dict = {}
    for nt, dim in LUMINA_NODE_DIMS.items():
        store = gbatch[nt] if nt in gbatch.node_types else None
        x = getattr(store, "x", None) if store is not None else None
        if x is None:
            x = torch.zeros((0, dim), dtype=torch.float32)
        x_dict[nt] = x.to(device).float()

    edge_index_dict = {}
    edge_attr_dict = {}
    for et in LUMINA_EDGE_TYPES:
        store = gbatch[et] if et in gbatch.edge_types else None
        ei = getattr(store, "edge_index", None) if store is not None else None
        if ei is None:
            ei = torch.zeros((2, 0), dtype=torch.long)
        edge_index_dict[et] = ei.to(device)
        ea = getattr(store, "edge_attr", None) if store is not None else None
        if ea is not None:
            edge_attr_dict[et] = ea.to(device).float()
    return x_dict, edge_index_dict, (edge_attr_dict or None)


def lumina_forward(model, batch_cpu: Dict[str, torch.Tensor], args, device,
                   opf_space: Dict[str, torch.Tensor] = None, return_gen: bool = False):
    graphs, _ = make_lumina_graphs(batch_cpu, args, opf_space=opf_space)
    gbatch = Batch.from_data_list(graphs)
    x_dict, edge_index_dict, edge_attr_dict = _dicts_from_batch(gbatch, device)
    out = model(
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        minmax_scaling=not args.no_minmax_scaling,
    )
    pred = out["bus"]
    theta = torch.atan2(torch.sin(pred[:, 0]), torch.cos(pred[:, 0]))
    mag = pred[:, 1]
    V = torch.stack([mag, theta], dim=-1).unsqueeze(0)
    if not return_gen:
        return V
    # LUMINA's generator head emits (Pg, Qg) per generator node; the adapter
    # places one generator node on each controllable bus, so gen_bus recovers
    # the mapping. Unused under voltage-only supervision.
    gen_pred = out.get("generator")
    gen_bus = None
    if ("generator", "generator_link", "bus") in gbatch.edge_types:
        gen_bus = gbatch["generator", "generator_link", "bus"].edge_index[1].to(device)
    return V, gen_pred, gen_bus


def _resolve_lumina_files(args) -> Tuple[str, str]:
    """Return (config_path, weight_path), downloading from the Hub if needed."""
    config_path = args.model_config
    weight_path = args.pretrained_checkpoint
    if config_path and weight_path:
        return config_path, weight_path

    from huggingface_hub import hf_hub_download

    if not config_path:
        config_path = hf_hub_download(repo_id=args.hf_repo_id, filename="config.json")
    if not weight_path and args.init_mode == "pretrained":
        weight_path = hf_hub_download(repo_id=args.hf_repo_id, filename="model.safetensors")
    return config_path, weight_path


def _load_state_dict(weight_path: str) -> dict:
    if weight_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(weight_path)
    obj = torch.load(weight_path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def make_model(args, device):
    from lumina_inference import Modeler
    from lumina_inference.model.hetero_model import HGT

    config_path, weight_path = _resolve_lumina_files(args)
    with open(config_path, "r") as fh:
        config_data = json.load(fh)
    hgt_cfg = config_data["config"]["models"]["HGT"]
    print(
        f"[model] LUMINA HGT layers={hgt_cfg['num_layers']} hidden={hgt_cfg['hidden_channels']} "
        f"heads={hgt_cfg.get('num_heads', 1)} dropout={hgt_cfg.get('dropout', 0.0)}"
    )

    if args.init_mode == "pretrained":
        if not weight_path:
            raise ValueError("--init_mode pretrained requires released LUMINA weights")
        modeler = Modeler(device, verbose=True)
        model, _ = modeler.load_model(config_data, _load_state_dict(weight_path))
    else:
        model = HGT(
            metadata=config_data["metadata"],
            input_channels=config_data["input_channels"],
            hidden_channels=hgt_cfg["hidden_channels"],
            num_layers=hgt_cfg["num_layers"],
            num_heads=hgt_cfg.get("num_heads", 1),
            dropout=hgt_cfg.get("dropout", 0.0),
        ).to(device)

    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[model] warm start from {args.init_checkpoint} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    model.train()
    return model


def run():
    args = parse_args()
    set_seed(args.seed_value)

    if args.task != "pf" and args.kol_pf_mode != "off":
        raise ValueError(
            "--kol_pf_mode solves specified-injection AC power flow and cannot "
            "be used for --task opf, where dispatch is an optimization decision."
        )

    if not args.run_name:
        mode = "pre" if args.init_mode == "pretrained" else "scratch"
        args.run_name = (
            f"lumina_{mode}_{known_operator_tag(args)}_"
            f"{Path(args.PARQUET).stem}"
        )
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    ddp = setup_ddp(args.DDP, timeout_hours=args.ddp_timeout_hours)
    import atexit
    atexit.register(cleanup, ddp)

    # One tee per file: eight ranks appending to the same log interleaves it.
    if args.log_to_file and ddp.is_main:
        log_path = os.path.join(args.log_dir, f"{args.run_name}_training_log.txt")
        log_f = open(log_path, "a", buffering=1)
        sys.stdout = Tee(sys.__stdout__, log_f)
        sys.stderr = Tee(sys.__stderr__, log_f)
        print(f"[log] tee -> {log_path}")

    quiet_non_main(ddp)
    device = ddp.device
    print(f"[env] device={device} torch={torch.__version__}")
    print(f"[run] {args.run_name}")
    print(f"[task] {args.task}"
          + (f" (limit_weight={args.opf_limit_weight} band_weight={args.opf_band_weight})"
             if args.task == "opf" else ""))
    print(f"[data] PARQUET={args.PARQUET}")
    print(
        f"[data] PER_UNIT={args.PER_UNIT} target_S_base={args.target_S_base} "
        f"dtype={args.dataset_complex_dtype} share_grid={args.share_grid}"
    )
    print(
        f"[model] LUMINA init_mode={args.init_mode} "
        f"pretrained_checkpoint={args.pretrained_checkpoint or '<hub>'} "
        f"minmax_scaling={not args.no_minmax_scaling} gen_setpoint={args.gen_setpoint_mode}"
    )
    print(
        f"[loss] mse_weight={args.mse_weight} physics_weight={args.physics_weight} "
        f"physics_form={args.physics_loss_form}"
    )
    print(f"[known-operator] {known_operator_tag(args)} raw_loss_weight={args.kol_raw_loss_weight:g}")

    dataset = ChanghunDataset(
        args.PARQUET,
        per_unit=args.PER_UNIT,
        target_S_base=args.target_S_base,
        share_grid=args.share_grid,
        share_ybus=args.share_ybus or args.share_grid,
        lazy_row_groups=args.lazy_parquet,
        row_group_cache_size=args.row_group_cache_size,
        complex_dtype=args.dataset_complex_dtype,
        extra_binary_columns=OPF_EXTRA_COLUMNS if args.task == "opf" else None,
    )
    train_ds, val_ds, test_ds = split_dataset(
        dataset,
        args.train_ratio,
        args.valid_ratio,
        args.seed_value,
    )
    train_ds = cap_subset(train_ds, args.max_train_samples)
    val_ds = cap_subset(val_ds, args.max_valid_samples)
    test_ds = cap_subset(test_ds, args.max_test_samples)
    print(f"[split] train={len(train_ds)} valid={len(val_ds)} test={len(test_ds)}")

    if args.preload_ram:
        # One decode pass now, then every epoch is pure indexing.
        train_ds = preload_split(train_ds, label='train')
        val_ds = preload_split(val_ds, label='valid')
        if args.preload_test:
            test_ds = preload_split(test_ds, label='test')

    loader_kwargs = {
        "batch_size": args.BATCH,
        "collate_fn": collate_blockdiag,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader, train_sampler = make_loader(train_ds, ddp, shuffle=True, **loader_kwargs)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **loader_kwargs)
    val_loader, _ = make_loader(val_ds, ddp, shuffle=False, **loader_kwargs)
    # Undistributed on purpose: every rank walks the whole test set, so the
    # reported figure is computed exactly as in the single-GPU runs and stays
    # comparable with the existing tables. Summing identical replicas and
    # dividing by the summed count returns the same mean.
    test_loader, _ = make_loader(test_ds, ddp, shuffle=False, distributed=False,
                                 **loader_kwargs)

    model = make_model(args, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable_params={n_params:,}")
    known_operator = build_known_operator(args)
    if known_operator is not None:
        known_operator = known_operator.to(device)
        print(f"[known-operator] {known_operator}")

    model = wrap_model(model, ddp,
                       find_unused_parameters=args.ddp_find_unused_parameters,
                       static_graph=args.ddp_static_graph)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.LR,
        weight_decay=args.weight_decay,
    )
    accum_steps = resolve_accum_steps(args, args.BATCH, ddp.world_size)
    accum = GradAccumulator(optim, model, accum_steps, max_grad_norm=1.0,
                            ddp_model=model if ddp.distributed else None)
    print(
        f"[optim] micro-batch {args.BATCH} x world {ddp.world_size} "
        f"x grad_accum_steps {accum_steps} "
        f"= effective batch {args.BATCH * ddp.world_size * accum_steps}"
    )
    best_val = float("inf")
    best_path = os.path.join(args.ckpt_dir, f"{args.run_name}_best.pt")

    is_opf = args.task == "opf"

    def run_epoch(loader, train):
        model.train(train)
        sum_loss = 0.0
        sum_mse = 0.0
        sum_mse_mag = 0.0
        sum_mse_ang = 0.0
        sum_phys = 0.0
        sum_max_dp = 0.0
        sum_max_dq = 0.0
        sum_max_dp_mva = 0.0
        sum_max_dq_mva = 0.0
        n_graphs = 0
        n_conv = 0
        dp_values = []
        dq_values = []
        opf_sums: Dict[str, float] = {}
        kol_initial_values = []
        kol_final_values = []
        kol_converged_values = []
        kol_iteration_values = []

        if train:
            accum.zero()

        with torch.set_grad_enabled(train):
            for batch_cpu in loader:
                B_eff = int(batch_cpu["sizes"].numel())
                n_graphs += B_eff
                n_nodes = batch_cpu["sizes"].to(device)
                batch_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_cpu.items()
                }
                bus_type = batch_dev["bus_type"]
                Sstart_metric = batch_dev["S_start"]
                S_base_metric = batch_dev.get("S_base", None)
                if S_base_metric is None and args.target_S_base is not None:
                    S_base_metric = torch.full(
                        (B_eff,),
                        float(args.target_S_base),
                        dtype=torch.float64,
                        device=device,
                    )
                Y = batch_dev.get("Ybus", None)
                Y_metric = ensure_dense_y_for_metrics(
                    Y,
                    bus_type,
                    batch_dev["Branch_f_bus"],
                    batch_dev["Branch_t_bus"],
                    batch_dev["Branch_status"],
                    batch_dev["Branch_tau"],
                    batch_dev["Branch_shift_deg"],
                    batch_dev["Branch_y_series_from"],
                    batch_dev["Branch_y_series_to"],
                    batch_dev["Branch_y_series_ft"],
                    batch_dev["Branch_y_shunt_from"],
                    batch_dev["Branch_y_shunt_to"],
                    batch_dev["Y_shunt_bus"],
                )

                target = batch_dev["V_newton"].float()
                if is_opf:
                    space_cpu = opf_decision_space(batch_cpu, batch_cpu["sizes"], torch.device("cpu"))
                    space_dev = opf_decision_space(batch_dev, batch_cpu["sizes"], device)
                else:
                    space_cpu = space_dev = None
                want_gen = is_opf and (args.gen_head_weight > 0.0)
                if want_gen:
                    Vraw, gen_pred, gen_bus = lumina_forward(
                        model, batch_cpu, args, device, opf_space=space_cpu, return_gen=True)
                else:
                    Vraw = lumina_forward(model, batch_cpu, args, device, opf_space=space_cpu)
                    gen_pred = gen_bus = None

                kol_diag = None
                if known_operator is not None:
                    Vpred, kol_diag = known_operator(
                        Vraw,
                        Y_metric,
                        Sstart_metric,
                        bus_type,
                        V_fixed=batch_dev["V_start"],
                        sizes=n_nodes,
                        differentiable=train,
                    )
                else:
                    Vpred = Vraw

                dmag = Vpred[..., 0] - target[..., 0]
                dang = angle_diff(Vpred[..., 1], target[..., 1])
                mse_mag = torch.mean(dmag * dmag)
                mse_ang = torch.mean(dang * dang)
                mse = mse_mag + mse_ang
                if is_opf:
                    phys = opf_loss(
                        Y_metric, Vpred, Sstart_metric, space_dev,
                        balance_weight=1.0,
                        limit_weight=args.opf_limit_weight,
                        band_weight=args.opf_band_weight,
                        form=args.physics_loss_form,
                    )
                else:
                    phys = ppc_physics_loss(
                        Y_metric.to(torch.complex64),
                        Vpred,
                        Sstart_metric.to(torch.complex64),
                        bus_type,
                        form=args.physics_loss_form,
                        huber_delta=args.physics_huber_delta,
                    )
                loss = args.mse_weight * mse + args.physics_weight * phys
                if known_operator is not None and args.kol_raw_loss_weight > 0.0:
                    raw_dmag = Vraw[..., 0] - target[..., 0]
                    raw_dang = angle_diff(Vraw[..., 1], target[..., 1])
                    raw_mse = torch.mean(raw_dmag * raw_dmag) + torch.mean(raw_dang * raw_dang)
                    raw_phys = ppc_physics_loss(
                        Y_metric.to(torch.complex64),
                        Vraw,
                        Sstart_metric.to(torch.complex64),
                        bus_type,
                        form=args.physics_loss_form,
                        huber_delta=args.physics_huber_delta,
                    )
                    loss = loss + args.kol_raw_loss_weight * (
                        args.mse_weight * raw_mse + args.physics_weight * raw_phys
                    )
                if is_opf and (args.dispatch_weight > 0.0 or args.gen_head_weight > 0.0):
                    tgt_gen = target_dispatch(batch_dev, space_dev, device)
                    if tgt_gen is not None:
                        if args.dispatch_weight > 0.0:
                            loss = loss + args.dispatch_weight * dispatch_loss(
                                Y_metric, Vpred, Sstart_metric, space_dev, tgt_gen)
                        if args.gen_head_weight > 0.0 and gen_pred is not None:
                            loss = loss + args.gen_head_weight * gen_head_loss(
                                gen_pred, tgt_gen, gen_bus)

                if train:
                    # Clip and step once per accumulation group; with
                    # accum_steps == 1 this is the original per-batch update.
                    accum.backward(loss)

                if is_opf:
                    om = opf_metrics(Y_metric, Vpred.detach(), target, Sstart_metric,
                                     space_dev, batch_cpu["sizes"])
                    for key, value in om.items():
                        opf_sums[key] = opf_sums.get(key, 0.0) + float(value) * B_eff
                residual_metrics = compute_power_flow_residual_metrics(
                    Y_metric,
                    Vpred.detach(),
                    Sstart_metric,
                    bus_type,
                    n_nodes_per_graph=n_nodes,
                    S_base=S_base_metric,
                )
                sum_loss += float(loss.item()) * B_eff
                sum_mse += float(mse.item()) * B_eff
                sum_mse_mag += float(mse_mag.item()) * B_eff
                sum_mse_ang += float(mse_ang.item()) * B_eff
                sum_phys += float(phys.item()) * B_eff
                sum_max_dp += residual_metrics["max_dp_pu"].sum().item()
                sum_max_dq += residual_metrics["max_dq_pu"].sum().item()
                if "max_dp_mva" in residual_metrics:
                    sum_max_dp_mva += residual_metrics["max_dp_mva"].sum().item()
                    sum_max_dq_mva += residual_metrics["max_dq_mva"].sum().item()
                dp_values.append(residual_metrics["dp_abs_valid"].cpu())
                dq_values.append(residual_metrics["dq_abs_valid"].cpu())
                conv = (
                    (residual_metrics["max_dp_pu"] < args.convergence_tol_pu)
                    & (residual_metrics["max_dq_pu"] < args.convergence_tol_pu)
                )
                n_conv += int(conv.sum().item())
                if kol_diag is not None:
                    kol_initial_values.append(kol_diag["initial_max_mismatch"].detach().cpu())
                    kol_final_values.append(kol_diag["final_max_mismatch"].detach().cpu())
                    kol_converged_values.append(kol_diag["converged"].detach().cpu())
                    kol_iteration_values.append(kol_diag["iterations"].detach().cpu())

        if train:
            # A trailing partial group would otherwise be dropped.
            accum.flush()

        # Each rank holds only its shard's totals. Sum them, and the counts
        # they divide by, before any mean is taken.
        (sum_loss, sum_mse, sum_mse_mag, sum_mse_ang, sum_phys,
         sum_max_dp, sum_max_dq, sum_max_dp_mva, sum_max_dq_mva,
         _nc, _ng) = reduce_sums(
            [sum_loss, sum_mse, sum_mse_mag, sum_mse_ang, sum_phys,
             sum_max_dp, sum_max_dq, sum_max_dp_mva, sum_max_dq_mva,
             n_conv, n_graphs], ddp)
        n_conv, n_graphs = int(round(_nc)), int(round(_ng))

        denom = max(n_graphs, 1)
        dist = residual_distribution(dp_values, dq_values, args.residual_tol_pu)
        dist["convergence_rate"] = n_conv / denom
        dist["n_converged"] = n_conv
        dist["n_cases"] = n_graphs
        kol_summary = None
        if kol_final_values:
            kol_summary = {
                "initial_max_mismatch": torch.cat(kol_initial_values),
                "final_max_mismatch": torch.cat(kol_final_values),
                "converged": torch.cat(kol_converged_values),
                "iterations": torch.cat(kol_iteration_values),
            }
        return {
            "loss": sum_loss / denom,
            "mse": sum_mse / denom,
            "mse_mag": sum_mse_mag / denom,
            "mse_ang": sum_mse_ang / denom,
            "phys": sum_phys / denom,
            "max_dp_pu": sum_max_dp / denom,
            "max_dq_pu": sum_max_dq / denom,
            "max_dp_mva": sum_max_dp_mva / denom,
            "max_dq_mva": sum_max_dq_mva / denom,
            "dist": dist,
            "opf": {k: v / denom for k, v in opf_sums.items()} if opf_sums else None,
            "kol": kol_summary,
        }

    def fmt(prefix, m):
        rmse = math.sqrt(max(m["mse"], 0.0))
        rmse_mag = math.sqrt(max(m["mse_mag"], 0.0))
        rmse_ang_deg = math.sqrt(max(m["mse_ang"], 0.0)) * 180.0 / math.pi
        if m.get("opf"):
            return (
                f"{prefix} loss {m['loss']:.4e} mse {m['mse']:.4e} phys {m['phys']:.4e} "
                f"rmse {rmse:.4e} (mag {rmse_mag:.4e}, ang {rmse_ang_deg:.4e}deg) "
                f"{format_opf_metrics(m['opf'])}"
            )
        return (
            f"{prefix} loss {m['loss']:.4e} mse {m['mse']:.4e} phys {m['phys']:.4e} "
            f"rmse {rmse:.4e} (mag {rmse_mag:.4e}, ang {rmse_ang_deg:.4e}deg) "
            f"(dPinf {m['max_dp_pu']:.3e} pu, dQinf {m['max_dq_pu']:.3e} pu; "
            f"{m['max_dp_mva']:.3e} MW, {m['max_dq_mva']:.3e} MVAr) "
            f"{format_residual_distribution_compact(m['dist'])} "
            f"{format_known_operator_diagnostics(m.get('kol'))}"
        )

    print("Initial metrics before training:")
    if args.EPOCHS <= 0:
        tr0 = run_epoch(train_eval_loader, train=False)
        va0 = run_epoch(val_loader, train=False)
        te0 = run_epoch(test_loader, train=False)
        print("Epoch   0 | " + fmt("train", tr0))
        print("Epoch   0 | " + fmt("valid", va0))
        print("Epoch   0 | " + fmt("test", te0))
        return

    val0 = run_epoch(val_loader, train=False)
    print("Epoch   0 | " + fmt("valid", val0))
    if args.retain_init_as_best:
        best_val = val0["loss"]
        if ddp.is_main:
            torch.save(unwrap(model).state_dict(), best_path)
        print(f"  retained epoch-0 warm start as early-stopping candidate at {best_path}")

    for epoch in range(1, args.EPOCHS + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)   # or every epoch reuses one shuffle
        t0 = time.time()
        tr = run_epoch(train_loader, train=True)
        if epoch % args.VAL_EVERY == 0 or epoch == args.EPOCHS:
            va = run_epoch(val_loader, train=False)
            print(
                f"Epoch {epoch:3d} | {fmt('train', tr)} | {fmt('valid', va)} "
                f"| time {time.time() - t0:.2f}s"
            )
            if va["loss"] < best_val:
                best_val = va["loss"]
                if ddp.is_main:
                    torch.save(unwrap(model).state_dict(), best_path)
                print(f"  checkpoint saved to {best_path}")
        else:
            print(f"Epoch {epoch:3d} | {fmt('train', tr)} | time {time.time() - t0:.2f}s")

    barrier(ddp)
    if os.path.exists(best_path):
        unwrap(model).load_state_dict(torch.load(best_path, map_location=device))
        print(f"[test] loaded best checkpoint: {best_path}")
    te = run_epoch(test_loader, train=False)
    rmse = math.sqrt(max(te["mse"], 0.0))
    rmse_mag = math.sqrt(max(te["mse_mag"], 0.0))
    rmse_ang_deg = math.sqrt(max(te["mse_ang"], 0.0)) * 180.0 / math.pi
    print(
        f"\nFinal test-set RMSE : {rmse:.4e}"
        f" (|V|: {rmse_mag:.4e}, theta: {rmse_ang_deg:.4e}deg)"
        f" | dPinf : {te['max_dp_pu']:.4e} pu ({te['max_dp_mva']:.4e} MW)"
        f" | dQinf : {te['max_dq_pu']:.4e} pu ({te['max_dq_mva']:.4e} MVAr)"
    )
    print(format_residual_distribution_compact(te["dist"]))
    if te.get("opf"):
        # The PF-convention residual above does not mask controllable buses, so
        # on OPF data it charges correct redispatch as error. These are the
        # metrics to read for the OPF task.
        print("Final test-set OPF : " + format_opf_metrics(te["opf"]))


if __name__ == "__main__":
    run()
