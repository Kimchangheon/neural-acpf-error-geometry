#!/usr/bin/env python3
"""
Train/evaluate the released GridSFM surrogate on the PPC LVN parquet pipeline.

This keeps the existing PPC dataloader, split logic, supervised Newton target,
and complex128 AC-PF residual evaluation. Only the neural surrogate is swapped:
LVN branch-row parquet batches are adapted into GridSFM's HeteroData schema.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torch_geometric.data import HeteroData

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
from opf_task import (
    OPF_EXTRA_COLUMNS,
    format_opf_metrics,
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
from supervised_voltage_loss import (
    add_correction_target_norm_args,
    correction_scales_for_batch,
    estimate_correction_target_scales,
    normalized_voltage_mse,
)


BUS_TYPE_PQ = 1
BUS_TYPE_PV = 2
BUS_TYPE_REF = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="GridSFM released backbone on PPC branch-row parquet data",
    )
    parser.add_argument("--ang_mse_weight", type=float, default=1.0,
        help="Weight on the angle term of the supervised MSE. 0 trains the "
             "voltage magnitude alone, which separates a magnitude-path defect "
             "from the channel imbalance in the unweighted sum.")
    parser.add_argument("--PARQUET", type=str, required=True)
    from pf_anchors import add_anchor_args
    add_anchor_args(parser)
    parser.add_argument("--mask_known_v", action="store_true",
        help="Hold |V| at PV/slack and theta at slack at their setpoints.")
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
    parser.add_argument("--log_dir", type=str, default="./results/logs/gridsfm")
    parser.add_argument("--ckpt_dir", type=str, default="./results/ckpt/gridsfm")

    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="",
        help="Released GridSFM checkpoint, e.g. gridsfm_open_v1.1.pt.",
    )
    parser.add_argument(
        "--init_checkpoint", type=str, default="",
        help="Load a state_dict saved by this driver (rescoring, warm start).",
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
    parser.add_argument(
        "--init_strict", action="store_true",
        help="Require an exact key/shape match for --init_checkpoint.",
    )
    parser.add_argument(
        "--init_mode",
        choices=("pretrained", "scratch"),
        default="pretrained",
        help="Use released GridSFM weights or random GridSFM backbone init.",
    )
    parser.add_argument(
        "--resume_state_dict",
        type=str,
        default="",
        # --init_mode pretrained goes through GridSFM's release loader, which
        # rejects the plain state_dict this script saves as <run>_best.pt. And
        # --EPOCHS 0 returns before the post-training checkpoint load. This flag
        # is the only way to score a checkpoint this script produced.
        help="Load a training-format state_dict into the freshly built model.",
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
    parser.add_argument(
        "--init_recipe", choices=("default", "sd002"), default="default",
        help="'sd002' completes GridSFM's own normal_(0, 0.02) recipe across "
             "the whole backbone (it currently reaches 44%% of weight matrices) "
             "while preserving head_V's bias of 1.0.")
    parser.add_argument("--split_seed", type=int, default=None,
                        help="Data-partition seed; defaults to --seed_value for legacy runs.")
    parser.add_argument("--train_ratio", type=float, default=0.3333)
    parser.add_argument("--valid_ratio", type=float, default=0.3333)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_valid_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--VAL_EVERY", type=int, default=1)
    parser.add_argument(
        "--terminal_projection_k",
        type=int,
        default=0,
        help=(
            "For PF only, project the final GridSFM voltage output onto the "
            "rank-k subspace fitted once from training-split reference states "
            "(0 disables it). The fixed projection is part of the forward "
            "graph, not post-hoc scoring."
        ),
    )

    parser.add_argument("--mse_weight", type=float, default=1.0)
    add_correction_target_norm_args(parser)
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
        help="Classify branch rows with different endpoint vn_kv as GridSFM transformers.",
    )
    parser.add_argument(
        "--rate_a",
        type=float,
        default=0.0,
        help="Fallback GridSFM branch rate_a feature when the parquet has no rate.",
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


def _cpu(x):
    return x.detach().cpu() if isinstance(x, torch.Tensor) else x


def _map_bus_types(ppc_bus_type: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(ppc_bus_type.long(), BUS_TYPE_PQ)
    out[ppc_bus_type.long() == 2] = BUS_TYPE_PV
    out[ppc_bus_type.long() == 1] = BUS_TYPE_REF
    return out


def _safe_inverse_admittance(y: torch.Tensor) -> torch.Tensor:
    tiny = torch.tensor(1e-12, dtype=y.real.dtype)
    mask = y.abs() > tiny
    z = torch.zeros_like(y)
    z[mask] = 1.0 / y[mask]
    return z


def _make_branch_family(
    f: torch.Tensor,
    t: torch.Tensor,
    y_series: torch.Tensor,
    ysh_f: torch.Tensor,
    ysh_t: torch.Tensor,
    tau: torch.Tensor,
    shift_deg: torch.Tensor,
    mask: torch.Tensor,
    *,
    is_transformer: bool,
    rate_a: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        cols = 11 if is_transformer else 9
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, cols), dtype=torch.float32)

    ff = f[idx].long()
    tt = t[idx].long()
    yy = y_series[idx].to(torch.complex128)
    zz = _safe_inverse_admittance(yy)
    r = zz.real.float()
    x = zz.imag.float()
    bfr = (ysh_f[idx].to(torch.complex128).imag.float()) / 2.0
    bto = (ysh_t[idx].to(torch.complex128).imag.float()) / 2.0
    shift_rad = torch.deg2rad(shift_deg[idx].float())
    tap = tau[idx].float()
    tap = torch.where(tap.abs() < 1e-9, torch.ones_like(tap), tap)
    edge_index = torch.stack([ff, tt], dim=0).contiguous()

    if is_transformer:
        attr = torch.zeros((idx.numel(), 11), dtype=torch.float32)
        attr[:, 0] = -math.pi
        attr[:, 1] = math.pi
        attr[:, 2] = r
        attr[:, 3] = x
        attr[:, 4] = float(rate_a)
        attr[:, 7] = tap
        attr[:, 8] = shift_rad
        attr[:, 9] = bfr
        attr[:, 10] = bto
    else:
        attr = torch.zeros((idx.numel(), 9), dtype=torch.float32)
        attr[:, 0] = -math.pi
        attr[:, 1] = math.pi
        attr[:, 2] = bfr
        attr[:, 3] = bto
        attr[:, 4] = r
        attr[:, 5] = x
        attr[:, 6] = float(rate_a)
    return edge_index, attr


def make_gridsfm_graphs(batch_cpu: Dict[str, torch.Tensor], args,
                        opf_space: Dict[str, torch.Tensor] = None) -> Tuple[List[HeteroData], torch.Tensor]:
    from gridsfm import prepare_for_inference

    sizes = batch_cpu["sizes"].long()
    branch_sizes = batch_cpu["branch_sizes"].long()
    offsets = batch_cpu["offsets"].long()
    branch_offsets = torch.cat((branch_sizes.new_zeros(1), torch.cumsum(branch_sizes, 0)[:-1]))

    graphs: List[HeteroData] = []
    target = batch_cpu["V_newton"].squeeze(0).float()

    for gi, (n, nl, off, boff) in enumerate(zip(sizes, branch_sizes, offsets, branch_offsets)):
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
        bus_x = torch.zeros((n, 4), dtype=torch.float32)
        bus_x[:, 0] = vn_kv
        bus_x[:, 1] = bus_type.float()
        if opf_space is not None and "Bus_vmin" in opf_space:
            bus_x[:, 2] = opf_space["Bus_vmin"][bs].float()
            bus_x[:, 3] = opf_space["Bus_vmax"][bs].float()
        else:
            bus_x[:, 2] = float(args.vmin)
            bus_x[:, 3] = float(args.vmax)
        d["bus"].x = bus_x
        d["bus"].v_setpoint = Vstart[:, 0:1].clone()

        if opf_space is not None:
            # OPF: generators are the controllable buses, described by their real
            # limits and cost slope. The dispatch itself is the answer, so pg/qg
            # stay at zero.
            ctrl = opf_space["Gen_controllable"][bs]
            gen_bus = torch.where(ctrl)[0]
            if gen_bus.numel() == 0:
                gen_bus = torch.tensor([0], dtype=torch.long)
            xg = torch.zeros((gen_bus.numel(), 11), dtype=torch.float32)
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
            xg = torch.zeros((gen_bus.numel(), 11), dtype=torch.float32)
            Pg0 = S.real[gen_bus].float()
            Qg0 = S.imag[gen_bus].float()
            p_margin = torch.maximum(Pg0.abs() * 0.5, torch.ones_like(Pg0) * 10.0)
            q_margin = torch.maximum(Qg0.abs() * 0.5, torch.ones_like(Qg0) * 10.0)
            xg[:, 0] = 1.0
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
        if load_bus.numel() > 0:
            xl = torch.zeros((load_bus.numel(), 2), dtype=torch.float32)
            xl[:, 0] = pd_
            xl[:, 1] = qd_
            d["load"].x = xl
            li = torch.arange(load_bus.numel(), dtype=torch.long)
            d["load", "load_link", "bus"].edge_index = torch.stack([li, load_bus], dim=0)
            d["bus", "load_link", "load"].edge_index = torch.stack([load_bus, li], dim=0)

        sh_mask = Ysh.abs() > 0
        sh_bus = torch.where(sh_mask)[0]
        if sh_bus.numel() > 0:
            xs = torch.zeros((sh_bus.numel(), 2), dtype=torch.float32)
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

        ac_mask = status & (~is_tr)
        tr_mask = status & is_tr
        ac_ei, ac_attr = _make_branch_family(
            f, t, y_series, ysh_f, ysh_t, tau, shift, ac_mask,
            is_transformer=False, rate_a=args.rate_a,
        )
        tr_ei, tr_attr = _make_branch_family(
            f, t, y_series, ysh_f, ysh_t, tau, shift, tr_mask,
            is_transformer=True, rate_a=args.rate_a,
        )
        if ac_ei.size(1) > 0:
            d["bus", "ac_line", "bus"].edge_index = ac_ei
            d["bus", "ac_line", "bus"].edge_attr = ac_attr
        if tr_ei.size(1) > 0:
            d["bus", "transformer", "bus"].edge_index = tr_ei
            d["bus", "transformer", "bus"].edge_attr = tr_attr

        d.feasible = torch.tensor(1, dtype=torch.long)
        d = prepare_for_inference(d)
        graphs.append(d)

    return graphs, target


def gridsfm_forward(model, batch_cpu: Dict[str, torch.Tensor], args, device,
                    opf_space: Dict[str, torch.Tensor] = None, return_gen: bool = False):
    from gridsfm import batch_data_list

    graphs, _ = make_gridsfm_graphs(batch_cpu, args, opf_space=opf_space)
    gbatch = batch_data_list(graphs, copy=False).to(device)
    out = model(gbatch)
    pred = out["bus"].pred
    theta = torch.atan2(torch.sin(pred[:, 0]), torch.cos(pred[:, 0]))
    mag = pred[:, 1]
    V = torch.stack([mag, theta], dim=-1).unsqueeze(0)
    # Anchors first, masking second: masking enforces the quantities the problem
    # specifies and must have the last word over any reparametrisation.
    from pf_anchors import apply_anchors
    V = apply_anchors(V, batch_cpu, args)
    if not return_gen:
        if getattr(args, "mask_known_v", False):
            from pf_known_mask import apply_known_v
            V = apply_known_v(V, batch_cpu["bus_type"].reshape(-1), batch_cpu["V_start"])
        return V
    # out['generator'].pred is [n_gen, 2] = (Pg, Qg). GridSFM's angle prior is
    # built from head_Pg through a detached path, so this head only receives a
    # gradient if it is supervised directly.
    gen_pred = out["generator"].pred if "generator" in out.node_types else None
    gen_bus = None
    if ("generator", "generator_link", "bus") in out.edge_types:
        gen_bus = out["generator", "generator_link", "bus"].edge_index[1].to(device)
    return V, gen_pred, gen_bus


def _apply_gridsfm_init_recipe(model, recipe: str = "default"):
    """Extend GridSFM's own small-init recipe to the whole backbone.

    GridSFM already writes normal_(0, 0.02) into part of itself and gives
    head_V a bias of 1.0 so it starts at flat voltage.  Auditing the built model
    shows the coverage is partial: 265 of 599 weight matrices (44%) carry the
    0.02 init and the remaining 334 -- the per-block attention W_q/W_k/W_v, the
    HeteroConv lin_l/lin_r, the first FFN layer of each block, and the fusion
    projections -- are left at PyTorch's default, about std 0.05.

    Completing the coverage is the change that took GraphKit on GBnetwork from
    R^2 -1.95 to +0.68 at the same 40 epochs.

    The head biases are re-applied afterwards: a blanket zeroing would erase
    head_V's 1.0 and with it the flat-voltage start, which is the part of
    GridSFM's initialisation that already works.
    """
    import torch
    import torch.nn as nn

    if recipe == "default":
        return model
    if recipe != "sd002":
        raise SystemExit(f"unknown --init_recipe {recipe!r}")

    n = 0
    for m in model.modules():
        w = getattr(m, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() >= 2:
            nn.init.normal_(w, mean=0.0, std=0.02)
            b = getattr(m, "bias", None)
            if isinstance(b, torch.Tensor):
                nn.init.zeros_(b)
            n += 1

    restored = []
    for name, val in (("head_theta", 0.0), ("head_V", 1.0),
                      ("head_Pg", 0.0), ("head_Qg", 0.0), ("head_feas", 0.0)):
        head = getattr(model, name, None)
        if head is None:
            continue
        last = head[-1] if hasattr(head, "__getitem__") else None
        if last is None or not isinstance(getattr(last, "bias", None), torch.Tensor):
            continue
        with torch.no_grad():
            last.bias.fill_(float(val))
        restored.append(f"{name}={val}")
    print(f"[init] gridsfm normal_(0, 0.02) on {n} weight matrices; "
          f"head biases restored: {', '.join(restored)}", flush=True)
    return model


def make_model(args, device):
    if args.init_mode == "pretrained":
        if not args.pretrained_checkpoint:
            raise ValueError("--init_mode pretrained requires --pretrained_checkpoint")
        from gridsfm import load_model

        model = load_model(args.pretrained_checkpoint, device=device)
        model.train()
        return model

    from gridsfm import GridTransformerBackbone

    model = GridTransformerBackbone().to(device)
    _apply_gridsfm_init_recipe(model, getattr(args, "init_recipe", "default"))
    if args.init_checkpoint:
        # A plain state_dict written by this driver, unlike
        # --pretrained_checkpoint which goes through GridSFM's release loader.
        state = torch.load(args.init_checkpoint, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        result = model.load_state_dict(state, strict=args.init_strict)
        print(
            f"[init] loaded checkpoint {args.init_checkpoint} "
            f"strict={args.init_strict} missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)}",
            flush=True,
        )
    return model


def fit_terminal_projection_basis(train_subset, batch_size, k, device):
    """Fit a fixed training-only |V|/angle basis for terminal projection."""
    from manifold_projection import fit_basis

    loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_blockdiag,
    )
    vm_chunks, theta_chunks = [], []
    with torch.no_grad():
        for batch in loader:
            sizes = batch["sizes"].to(torch.int64)
            if int(sizes.min()) != int(sizes.max()):
                raise ValueError("--terminal_projection_k requires one grid per campaign")
            n_bus = int(sizes[0])
            reference = batch["V_newton"].reshape(-1, n_bus, 2).to(torch.float64)
            vm_chunks.append(reference[..., 0])
            theta_chunks.append(reference[..., 1])
    vm = torch.cat(vm_chunks, dim=0)
    theta = torch.cat(theta_chunks, dim=0)
    basis, explained_vm, explained_theta = fit_basis(vm, theta, k)
    basis = tuple(x.to(device=device, dtype=torch.float32) for x in basis)
    print(
        f"[terminal-projection] training-only basis: {vm.shape[0]} scenarios, "
        f"{vm.shape[1]} buses, k={k}; explained |V|={100 * explained_vm:.3f}% "
        f"theta={100 * explained_theta:.3f}%",
        flush=True,
    )
    return basis


def run():
    args = parse_args()
    set_seed(args.seed_value)

    if args.task != "pf" and args.kol_pf_mode != "off":
        raise ValueError(
            "--kol_pf_mode solves a specified-injection PF and cannot be used "
            "for --task opf, where dispatch remains a decision variable."
        )

    if not args.run_name:
        mode = "pre" if args.init_mode == "pretrained" else "scratch"
        args.run_name = (
            f"gridsfm_{mode}_{known_operator_tag(args)}_"
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
        f"[model] GridSFM init_mode={args.init_mode} "
        f"pretrained_checkpoint={args.pretrained_checkpoint or '<none>'}"
    )
    print(
        f"[loss] mse_weight={args.mse_weight} physics_weight={args.physics_weight} "
        f"correction_target_norm={args.correction_target_norm} "
        f"physics_form={args.physics_loss_form}"
    )
    if args.terminal_projection_k > 0:
        print(f"[terminal-projection] enabled, k={args.terminal_projection_k}")
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
    split_seed = args.seed_value if args.split_seed is None else args.split_seed
    train_ds, val_ds, test_ds = split_dataset(
        dataset,
        args.train_ratio,
        args.valid_ratio,
        split_seed,
    )
    train_ds = cap_subset(train_ds, args.max_train_samples)
    val_ds = cap_subset(val_ds, args.max_valid_samples)
    test_ds = cap_subset(test_ds, args.max_test_samples)
    print(f"[split] seed={split_seed} train={len(train_ds)} valid={len(val_ds)} test={len(test_ds)}")

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
    correction_target_scales = estimate_correction_target_scales(
        train_loader, args.correction_target_norm, args.correction_target_norm_eps
    )

    if args.terminal_projection_k > 0 and args.task != "pf":
        raise ValueError("--terminal_projection_k is currently defined only for PF")
    terminal_basis = (
        fit_terminal_projection_basis(
            train_ds, args.BATCH, args.terminal_projection_k, device
        )
        if args.terminal_projection_k > 0
        else None
    )

    model = make_model(args, device)
    if getattr(args, "resume_state_dict", ""):
        model.load_state_dict(torch.load(args.resume_state_dict, map_location=device))
        print(f"[init] loaded state_dict: {args.resume_state_dict}")
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
                    Vraw, gen_pred, gen_bus = gridsfm_forward(
                        model, batch_cpu, args, device, opf_space=space_cpu, return_gen=True)
                else:
                    Vraw = gridsfm_forward(model, batch_cpu, args, device, opf_space=space_cpu)
                    gen_pred = gen_bus = None

                if terminal_basis is not None:
                    from manifold_projection import project_packed
                    Vraw = project_packed(Vraw, terminal_basis)

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
                # Weighted, because the two channels are not commensurate: on
                # case14 sigma(|V|)=0.0236 pu against an angle error near 0.30
                # rad, so mse_ang outweighs mse_mag about 168:1 and the
                # magnitude term contributes 0.6% of the supervised loss.
                target_scale = correction_scales_for_batch(
                    correction_target_scales, Vpred, n_nodes
                )
                mse = normalized_voltage_mse(
                    dmag, dang, target_scale, args.ang_mse_weight
                )
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
                sum_mse += float((mse_mag + mse_ang).item()) * B_eff
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
