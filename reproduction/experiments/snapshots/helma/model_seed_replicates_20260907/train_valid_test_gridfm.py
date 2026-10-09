#!/usr/bin/env python3
"""
Train a GridFM-GraphKit-style heterogeneous Transformer surrogate on the PPC
LVN parquet pipeline.

This script deliberately keeps the PPC dataloader, split logic, and residual
evaluation path, but replaces the PIGNN/Armijo surrogate with the GridFM HGNS
core idea: bus/gen heterogeneous message passing with PyG TransformerConv.
"""

import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torch_geometric.nn import HeteroConv, TransformerConv

from Dataset_optimized_complex_columns import ChanghunDataset
from grad_accum import GradAccumulator, add_grad_accum_args, resolve_accum_steps
from preload_dataset import add_preload_args, preload_split
from ddp_utils import (add_ddp_args, barrier, cleanup, make_loader,
                       quiet_non_main, reduce_sums, setup_ddp, unwrap, wrap_model)
from collate_blockdiag_optimized_complex_columns import collate_blockdiag, ybus_matvec
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
    power_flow_residual_loss,
)


# GridFM-GraphKit feature column layout.
PD_H = 0
QD_H = 1
QG_H = 2
VM_H = 3
VA_H = 4
PQ_H = 5
PV_H = 6
REF_H = 7
MIN_VM_H = 8
MAX_VM_H = 9
MIN_QG_H = 10
MAX_QG_H = 11
GS = 12
BS = 13
VN_KV = 14

PG_H = 0
MIN_PG = 1
MAX_PG = 2
C0_H = 3
C1_H = 4
C2_H = 5
G_ON = 6

P_E = 0
Q_E = 1
YFF_TT_R = 2
YFF_TT_I = 3
YFT_TF_R = 4
YFT_TF_I = 5
TAP = 6
ANG_MIN = 7
ANG_MAX = 8
RATE_A = 9


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="GridFM HGNS surrogate on PPC branch-row parquet data",
    )
    parser.add_argument("--PARQUET", type=str, required=True)
    parser.add_argument(
        "--zero_unspecified_q", action="store_true",
        help="Zero the reactive injection at slack/PV, where it is not a "
             "specified quantity, instead of feeding the parquet's arbitrary "
             "value as a bus feature.")
    parser.add_argument(
        "--pin_known",
        action="store_true",
        help="Power flow only: hold |V| at PV/slack and the angle at the slack "
             "to their given input values instead of predicting a correction "
             "there. Physically correct for PF; off by default so the first "
             "campaign stays reproducible.",
    )
    parser.add_argument(
        "--armijo_max_backtracks",
        type=int,
        default=6,
        help="graphkit_ls only: backtracking depth. With rho=0.5 the smallest "
             "step is 2^-(n-1), so 6 reaches 0.031.",
    )
    parser.add_argument(
        "--gridfm_impl",
        choices=("mirror", "graphkit", "graphkit_ls", "graphkit_inc", "graphkit_projected"),
        default="mirror",
        help="mirror: the local GridFMHeteroSurrogate. graphkit: the released "
             "gridfm_graphkit GNS_heterogeneous, which adds per-layer physics "
             "feedback and a PhysicsDecoderPF. graphkit_ls: the same backbone "
             "with the per-layer decoder read as a state-space increment and "
             "scaled by an Armijo line search. graphkit_inc: the increment "
             "without the line search, which separates the two changes. "
             "graphkit_projected fuses each layer's learned increment with an "
             "exact DPF/Adam direction, descent projection, and per-grid Armijo. None "
             "has published weights, so all are scratch-only.",
    )
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
    parser.add_argument("--log_dir", type=str, default="./results/logs/gridfm")
    parser.add_argument("--ckpt_dir", type=str, default="./results/ckpt/gridfm")
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default="",
        help="Optional GridFM surrogate state_dict checkpoint for fine-tuning/warm-start.",
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
        "--init_strict",
        action="store_true",
        help="Require exact checkpoint key/shape match when --init_checkpoint is used.",
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

    parser.add_argument("--BATCH", type=int, default=8)
    parser.add_argument("--EPOCHS", type=int, default=40)
    parser.add_argument("--LR", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed_value", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=None,
                        help="Data-partition seed; defaults to --seed_value for legacy runs.")
    parser.add_argument("--train_ratio", type=float, default=0.3333)
    parser.add_argument("--valid_ratio", type=float, default=0.3333)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_valid_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--VAL_EVERY", type=int, default=1)

    parser.add_argument("--hidden_size", type=int, default=48)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--zero_init_head", action="store_true")
    parser.add_argument(
        "--vn_feature_mode",
        choices=("none", "log", "norm"),
        default="log",
        help="How to put LVN nominal voltage into the GridFM VN_KV feature.",
    )
    parser.add_argument(
        "--feature_transform",
        choices=("raw", "signed_log"),
        default="signed_log",
        help=(
            "Transform large PPC power/admittance input features before the "
            "TransformerConv MLPs. Physics loss/eval still use exact tensors."
        ),
    )
    parser.add_argument(
        "--graphkit_neural_edge_transform",
        choices=("raw", "signed_log"),
        default="raw",
        help=(
            "Released GraphKit only: transform admittance columns entering "
            "input_proj_edge while retaining raw edge_attr for the model's "
            "ComputeBranchFlow physics path. This is an adapter-boundary "
            "preprocessing hook; it does not change the public backbone."
        ),
    )

    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--physics_weight", type=float, default=1e-2)
    parser.add_argument(
        "--physics_loss_form",
        choices=("mse", "huber", "logcosh"),
        default="logcosh",
    )
    parser.add_argument("--physics_huber_delta", type=float, default=1.0)
    parser.add_argument(
        "--physics_compute_dtype",
        choices=("complex64", "complex128"),
        default="complex64",
        help="Precision of the outer AC-PF loss; neural inference stays float32.",
    )
    parser.add_argument(
        "--physics_residual_norm",
        choices=("none", "local_ybus"),
        default="none",
        help=(
            "Optional equation-wise scaling of the outer physics loss by "
            "|Y_ii|*|V_i|^2. Does not alter model inputs or GraphKit's "
            "per-layer physical decoder."
        ),
    )
    parser.add_argument("--residual_tol_pu", type=float, default=1e-6)
    parser.add_argument("--convergence_tol_pu", type=float, default=1e-6)
    parser.add_argument("--vmin", type=float, default=0.5)
    parser.add_argument("--vmax", type=float, default=1.5)
    parser.add_argument("--projected_dpf_lr", type=float, default=0.003377)
    parser.add_argument("--projected_dpf_beta1", type=float, default=0.979681)
    parser.add_argument("--projected_dpf_beta2", type=float, default=0.963442)
    parser.add_argument("--projected_dpf_mu", type=float, default=1e-4)
    parser.add_argument("--projected_dpf_max_vm_fraction", type=float, default=0.10)
    parser.add_argument("--projected_dpf_max_angle_step", type=float, default=0.30)
    parser.add_argument(
        "--projected_dpf_first_order", action="store_true",
        help="Detach inner mismatch gradients to avoid second derivatives; ablation only.",
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


def angle_diff(a, b):
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def signed_log1p(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _real_dtype(dtype):
    return torch.float64 if dtype == torch.complex128 else torch.float32


def build_dense_y_from_branchrows_single(
    N,
    Branch_f_bus,
    Branch_t_bus,
    Branch_status,
    Branch_tau,
    Branch_shift_deg,
    Branch_y_series_from,
    Branch_y_series_to,
    Branch_y_series_ft,
    Branch_y_shunt_from,
    Branch_y_shunt_to,
    Y_shunt_bus,
):
    dtype = Branch_y_series_from.dtype
    Y = torch.zeros((N, N), dtype=dtype, device=Branch_y_series_from.device)
    Y = Y + torch.diag(Y_shunt_bus.to(dtype))

    mask = Branch_status.to(torch.bool)
    if not torch.any(mask):
        return Y

    f = Branch_f_bus[mask].long()
    t = Branch_t_bus[mask].long()
    real_dtype = _real_dtype(dtype)
    tau = Branch_tau[mask].to(real_dtype)
    theta = torch.deg2rad(Branch_shift_deg[mask].to(real_dtype))
    a = tau.to(dtype) * torch.exp(1j * theta.to(dtype))

    y_from = Branch_y_series_from[mask].to(dtype)
    y_to = Branch_y_series_to[mask].to(dtype)
    y_ft = Branch_y_series_ft[mask].to(dtype)
    y_tf = y_ft
    ysh_f = Branch_y_shunt_from[mask].to(dtype)
    ysh_t = Branch_y_shunt_to[mask].to(dtype)

    Yff = (y_from + ysh_f / 2.0) / (a * torch.conj(a))
    Ytt = y_to + ysh_t / 2.0
    Yft = -y_ft / torch.conj(a)
    Ytf = -y_tf / a

    Y.index_put_((f, f), Yff, accumulate=True)
    Y.index_put_((t, t), Ytt, accumulate=True)
    Y.index_put_((f, t), Yft, accumulate=True)
    Y.index_put_((t, f), Ytf, accumulate=True)
    return Y


def ensure_dense_y_for_metrics(
    Y,
    bus_type,
    Branch_f_bus,
    Branch_t_bus,
    Branch_status,
    Branch_tau,
    Branch_shift_deg,
    Branch_y_series_from,
    Branch_y_series_to,
    Branch_y_series_ft,
    Branch_y_shunt_from,
    Branch_y_shunt_to,
    Y_shunt_bus,
):
    if Y is not None:
        # A sparse block-diagonal Y-bus stays exactly as the collate produced it:
        # unsqueeze would make it 3-D sparse, and several ops (as_strided, hence
        # squeeze) are unimplemented for the sparse CUDA backend. ybus_matvec
        # consumes a 2-D sparse Y directly.
        if Y.is_sparse:
            return Y
        return Y.unsqueeze(0) if Y.dim() == 2 else Y
    if Y_shunt_bus is None:
        raise ValueError("Y is None and Y_shunt_bus is None.")

    B, N = bus_type.shape
    if B == 1:
        return build_dense_y_from_branchrows_single(
            N,
            Branch_f_bus.squeeze(0),
            Branch_t_bus.squeeze(0),
            Branch_status.squeeze(0),
            Branch_tau.squeeze(0),
            Branch_shift_deg.squeeze(0),
            Branch_y_series_from.squeeze(0),
            Branch_y_series_to.squeeze(0),
            Branch_y_series_ft.squeeze(0),
            Branch_y_shunt_from.squeeze(0),
            Branch_y_shunt_to.squeeze(0),
            Y_shunt_bus.squeeze(0),
        ).unsqueeze(0)

    Ys = []
    for b in range(B):
        Ys.append(build_dense_y_from_branchrows_single(
            N,
            Branch_f_bus[b],
            Branch_t_bus[b],
            Branch_status[b],
            Branch_tau[b],
            Branch_shift_deg[b],
            Branch_y_series_from[b],
            Branch_y_series_to[b],
            Branch_y_series_ft[b],
            Branch_y_shunt_from[b],
            Branch_y_shunt_to[b],
            Y_shunt_bus[b],
        ))
    return torch.stack(Ys, dim=0)


def compute_power_flow_residual_metrics(
    Y,
    Vpred,
    Sset,
    bus_type,
    *,
    n_nodes_per_graph=None,
    S_base=None,
):
    if Y.dim() == 2:
        Y = Y.unsqueeze(0)

    if Y.is_complex():
        complex_dtype = Y.dtype
        real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    else:
        real_dtype = Y.dtype
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64

    v = Vpred[..., 0].to(dtype=real_dtype)
    th = Vpred[..., 1].to(dtype=real_dtype)
    Sset = Sset.to(device=Y.device, dtype=complex_dtype)

    Vc = v * torch.exp(1j * th)
    Ic = ybus_matvec(Y, Vc)
    Sc = Vc * Ic.conj()

    dp_abs = (Sset.real - Sc.real).abs()
    dq_abs = (Sset.imag - Sc.imag).abs()
    slack_mask = bus_type == 1
    pv_mask = bus_type == 2
    p_mask = ~slack_mask
    q_mask = ~(slack_mask | pv_mask)

    if n_nodes_per_graph is not None:
        max_dp_pu = []
        max_dq_pu = []
        offset = 0
        for size in n_nodes_per_graph.tolist():
            size = int(size)
            sl = slice(offset, offset + size)
            dp_g = dp_abs[0, sl]
            dq_g = dq_abs[0, sl]
            p_mask_g = p_mask[0, sl]
            q_mask_g = q_mask[0, sl]
            max_dp_pu.append(dp_g[p_mask_g].max() if p_mask_g.any() else dp_g.new_zeros(()))
            max_dq_pu.append(dq_g[q_mask_g].max() if q_mask_g.any() else dq_g.new_zeros(()))
            offset += size
        max_dp_pu = torch.stack(max_dp_pu)
        max_dq_pu = torch.stack(max_dq_pu)
    else:
        max_dp_pu = dp_abs.masked_fill(~p_mask, 0.0).amax(dim=-1)
        max_dq_pu = dq_abs.masked_fill(~q_mask, 0.0).amax(dim=-1)
        max_dq_pu = torch.where(q_mask.any(dim=-1), max_dq_pu, torch.zeros_like(max_dq_pu))

    metrics = {
        "max_dp_pu": max_dp_pu,
        "max_dq_pu": max_dq_pu,
        "dp_abs_valid": dp_abs[p_mask].detach(),
        "dq_abs_valid": dq_abs[q_mask].detach(),
    }
    if S_base is not None:
        S_base = S_base.to(max_dp_pu.device, dtype=max_dp_pu.dtype).reshape(-1)
        if S_base.numel() == 1 and max_dp_pu.numel() != 1:
            S_base = S_base.expand(max_dp_pu.numel())
        metrics["max_dp_mva"] = max_dp_pu * S_base / 1e6
        metrics["max_dq_mva"] = max_dq_pu * S_base / 1e6
    return metrics


def ppc_physics_loss(Y, Vpred, Sset, bus_type, form="logcosh", huber_delta=1.0):
    if Y.dim() == 2:
        Y = Y.unsqueeze(0)
    complex_dtype = torch.complex128 if Y.dtype == torch.complex128 else torch.complex64
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32

    v = Vpred[..., 0].to(real_dtype)
    th = Vpred[..., 1].to(real_dtype)
    Sset = Sset.to(Y.device, dtype=complex_dtype)
    Vc = v * torch.exp(1j * th)
    Sc = Vc * ybus_matvec(Y if Y.is_sparse else Y.to(complex_dtype), Vc).conj()
    dp = Sset.real - Sc.real
    dq = Sset.imag - Sc.imag
    slack_mask = bus_type == 1
    pv_mask = bus_type == 2
    parts = [dp[~slack_mask], dq[~(slack_mask | pv_mask)]]
    r = torch.cat([p.reshape(-1) for p in parts if p.numel() > 0], dim=0)
    if r.numel() == 0:
        return Vpred.sum() * 0.0
    if form == "mse":
        return torch.mean(r * r)
    if form == "huber":
        return F.huber_loss(r, torch.zeros_like(r), delta=huber_delta)
    return torch.mean(torch.log(torch.cosh(torch.clamp(r, min=-30.0, max=30.0))))


# Tolerances reported alongside the configured --residual_tol_pu. 1e-6 pu is
# 100 W on a 100 MVA base -- two orders below anything a learned surrogate
# reaches, so it reads 0% on every grid above 300 buses and separates nothing.
# 1e-3 (100 kW) and 1e-2 (1 MW) are where the models actually differ.
REPORT_TOLERANCES = (1e-2, 1e-3)


def _tol_key(channel, tol):
    return f"frac_{channel}_le_{tol:.0e}"


def residual_distribution(dp_values, dq_values, tol_pu):
    max_quantile_points = 2_000_000

    def summarize(chunks):
        if not chunks:
            return {
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "rmse": 0.0,
                "frac": 0.0,
                "frac_at": {t: 0.0 for t in REPORT_TOLERANCES},
                "n": 0,
            }
        x = torch.cat([c.reshape(-1).float() for c in chunks if c.numel() > 0], dim=0)
        if x.numel() == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "rmse": 0.0,
                "frac": 0.0,
                "frac_at": {t: 0.0 for t in REPORT_TOLERANCES},
                "n": 0,
            }
        qx = x
        if x.numel() > max_quantile_points:
            idx = torch.linspace(
                0,
                x.numel() - 1,
                steps=max_quantile_points,
                device=x.device,
            ).long().clamp_max(x.numel() - 1)
            qx = x[idx]
        return {
            "mean": float(x.mean().item()),
            "median": float(torch.quantile(qx, 0.50).item()),
            "p95": float(torch.quantile(qx, 0.95).item()),
            "p99": float(torch.quantile(qx, 0.99).item()),
            "rmse": float(torch.sqrt(torch.mean(x * x)).item()),
            "frac": float((x <= tol_pu).float().mean().item()),
            "frac_at": {
                t: float((x <= t).float().mean().item()) for t in REPORT_TOLERANCES
            },
            "n": int(x.numel()),
        }

    p = summarize(dp_values)
    q = summarize(dq_values)
    return {
        "mean_dp_pu": p["mean"],
        "mean_dq_pu": q["mean"],
        "median_dp_pu": p["median"],
        "median_dq_pu": q["median"],
        "p95_dp_pu": p["p95"],
        "p95_dq_pu": q["p95"],
        "p99_dp_pu": p["p99"],
        "p99_dq_pu": q["p99"],
        "rmse_dp_pu": p["rmse"],
        "rmse_dq_pu": q["rmse"],
        "frac_dp_below_tol": p["frac"],
        "frac_dq_below_tol": q["frac"],
        **{_tol_key("dp", t): p["frac_at"][t] for t in REPORT_TOLERANCES},
        **{_tol_key("dq", t): q["frac_at"][t] for t in REPORT_TOLERANCES},
        "n_dp": p["n"],
        "n_dq": q["n"],
    }


def format_residual_distribution_compact(d):
    # median and frac-below-tol are appended rather than interleaved so log
    # parsers written against the older line keep matching.
    return (
        f"mean|dP| {d['mean_dp_pu']:.3e}, mean|dQ| {d['mean_dq_pu']:.3e}, "
        f"p95|dP| {d['p95_dp_pu']:.3e}, p95|dQ| {d['p95_dq_pu']:.3e}, "
        f"rmse|dP| {d['rmse_dp_pu']:.3e}, rmse|dQ| {d['rmse_dq_pu']:.3e}, "
        f"median|dP| {d['median_dp_pu']:.3e}, median|dQ| {d['median_dq_pu']:.3e}, "
        f"frac|dP| {d['frac_dp_below_tol']:.4f}, frac|dQ| {d['frac_dq_below_tol']:.4f}, "
        + ", ".join(
            f"frac|{c[0]}{c[1].upper()}|@{t:.0e} {d.get(_tol_key(c, t), 0.0):.4f}"
            for c in ("dp", "dq")
            for t in REPORT_TOLERANCES
        )
    )


def cap_subset(ds, cap):
    if cap is None or int(cap) <= 0 or int(cap) >= len(ds):
        return ds
    return Subset(ds, list(range(int(cap))))


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


def make_branch_edge_features(batch, dtype=torch.float32, feature_transform="signed_log"):
    f = batch["Branch_f_bus"].squeeze(0).long()
    t = batch["Branch_t_bus"].squeeze(0).long()
    status = batch["Branch_status"].squeeze(0).to(torch.bool)
    tau = batch["Branch_tau"].squeeze(0).to(dtype)
    shift_deg = batch["Branch_shift_deg"].squeeze(0).to(dtype)
    y_from = batch["Branch_y_series_from"].squeeze(0).to(torch.complex64)
    y_to = batch["Branch_y_series_to"].squeeze(0).to(torch.complex64)
    y_ft = batch["Branch_y_series_ft"].squeeze(0).to(torch.complex64)
    y_tf = y_ft
    ysh_f = batch["Branch_y_shunt_from"].squeeze(0).to(torch.complex64)
    ysh_t = batch["Branch_y_shunt_to"].squeeze(0).to(torch.complex64)

    f = f[status]
    t = t[status]
    tau = tau[status]
    shift_deg = shift_deg[status]
    y_from = y_from[status]
    y_to = y_to[status]
    y_ft = y_ft[status]
    y_tf = y_tf[status]
    ysh_f = ysh_f[status]
    ysh_t = ysh_t[status]

    theta = torch.deg2rad(shift_deg)
    a = tau.to(torch.complex64) * torch.exp(1j * theta.to(torch.complex64))
    Yff = (y_from + ysh_f / 2.0) / (a * torch.conj(a))
    Ytt = y_to + ysh_t / 2.0
    Yft = -y_ft / torch.conj(a)
    Ytf = -y_tf / a

    edge_index = torch.cat(
        [torch.stack([f, t], dim=0), torch.stack([t, f], dim=0)],
        dim=1,
    )

    def attrs(y_self, y_cross):
        out = torch.zeros((y_self.numel(), 10), dtype=dtype, device=y_self.device)
        out[:, YFF_TT_R] = y_self.real.to(dtype)
        out[:, YFF_TT_I] = y_self.imag.to(dtype)
        out[:, YFT_TF_R] = y_cross.real.to(dtype)
        out[:, YFT_TF_I] = y_cross.imag.to(dtype)
        if feature_transform == "signed_log":
            out[:, YFF_TT_R : YFT_TF_I + 1] = signed_log1p(
                out[:, YFF_TT_R : YFT_TF_I + 1],
            )
        out[:, TAP] = tau.to(dtype)
        out[:, ANG_MIN] = -math.pi
        out[:, ANG_MAX] = math.pi
        out[:, RATE_A] = 0.0
        return out

    edge_attr = torch.cat([attrs(Yff, Yft), attrs(Ytt, Ytf)], dim=0)
    return edge_index, edge_attr


def make_gridfm_inputs(batch, device, vn_feature_mode, feature_transform,
                       opf_space=None, zero_unspecified_q=False):
    bus_type = batch["bus_type"].to(device)
    Vstart = batch["V_start"].to(device).float()
    Vnewton = batch["V_newton"].to(device).float()
    Sstart = batch["S_start"].to(device)
    Y_shunt_bus = batch["Y_shunt_bus"].to(device)
    vn_kv = batch["vn_kv"].to(device).float()

    if bus_type.dim() != 2 or bus_type.shape[0] != 1:
        raise ValueError("GridFM adapter expects blockdiag collate with shape [1, N].")

    bt = bus_type.squeeze(0)
    S = Sstart.squeeze(0).to(torch.complex64)
    N = bt.numel()
    x_bus = torch.zeros((N, 15), dtype=torch.float32, device=device)

    # The PPC parquet stores net complex injection. For GridFM input features,
    # keep both signed demand-like channels so the network can learn the PPC
    # convention; the actual training/eval physics still uses S_start directly.
    p_in = -S.real
    q_in = -S.imag
    if zero_unspecified_q:
        # Reactive power is not a specified quantity at slack or PV: the solver
        # fixes |V| there and solves for Q, so whatever the parquet carries in
        # that slot is arbitrary. LVN_heo1 parks CGMES reactive-capability
        # figures there -- up to 2626 pu against a true maximum of about 8 --
        # and the GridSFM and LUMINA drivers already zero it for exactly this
        # reason. Off by default so the first campaign stays reproducible.
        _q_free = (bt == 1) | (bt == 2)
        q_in = q_in.masked_fill(_q_free, 0.0)
    if feature_transform == "signed_log":
        p_in = signed_log1p(p_in)
        q_in = signed_log1p(q_in)
    x_bus[:, PD_H] = p_in
    x_bus[:, QD_H] = q_in
    x_bus[:, QG_H] = 0.0
    x_bus[:, VM_H] = Vstart.squeeze(0)[:, 0]
    x_bus[:, VA_H] = Vstart.squeeze(0)[:, 1]
    x_bus[:, PQ_H] = ((bt != 1) & (bt != 2)).float()
    x_bus[:, PV_H] = (bt == 2).float()
    x_bus[:, REF_H] = (bt == 1).float()
    if opf_space is not None:
        # OPF: real voltage band and reactive limits, not placeholders.
        x_bus[:, MIN_VM_H] = opf_space["Bus_vmin"].float()
        x_bus[:, MAX_VM_H] = opf_space["Bus_vmax"].float()
        x_bus[:, MIN_QG_H] = opf_space["Gen_q_min"].float()
        x_bus[:, MAX_QG_H] = opf_space["Gen_q_max"].float()
    else:
        x_bus[:, MIN_VM_H] = 0.5
        x_bus[:, MAX_VM_H] = 1.5
        x_bus[:, MIN_QG_H] = -10.0
        x_bus[:, MAX_QG_H] = 10.0
    gs = Y_shunt_bus.squeeze(0).real.float()
    bs = Y_shunt_bus.squeeze(0).imag.float()
    if feature_transform == "signed_log":
        gs = signed_log1p(gs)
        bs = signed_log1p(bs)
    x_bus[:, GS] = gs
    x_bus[:, BS] = bs
    if vn_feature_mode == "none":
        x_bus[:, VN_KV] = 0.0
    elif vn_feature_mode == "norm":
        vmax = torch.clamp(vn_kv.max(), min=1.0)
        x_bus[:, VN_KV] = vn_kv.squeeze(0) / vmax
    else:
        x_bus[:, VN_KV] = torch.log10(vn_kv.squeeze(0) + 1e-9)

    if opf_space is not None:
        # OPF: the generator set is the controllable buses, described by their
        # real active limits and cost slope. PG stays zero: it is the answer.
        gen_bus = torch.where(opf_space["Gen_controllable"])[0]
        if gen_bus.numel() == 0:
            gen_bus = torch.tensor([0], dtype=torch.long, device=device)
        G = gen_bus.numel()
        x_gen = torch.zeros((G, 7), dtype=torch.float32, device=device)
        x_gen[:, MIN_PG] = opf_space["Gen_p_min"][gen_bus].float()
        x_gen[:, MAX_PG] = opf_space["Gen_p_max"][gen_bus].float()
        if "Gen_cost_c1" in opf_space:
            x_gen[:, C1_H] = opf_space["Gen_cost_c1"][gen_bus].float()
        x_gen[:, G_ON] = 1.0
    else:
        gen_bus = torch.where((bt == 1) | (bt == 2))[0]
        if gen_bus.numel() == 0:
            gen_bus = torch.tensor([0], dtype=torch.long, device=device)
        G = gen_bus.numel()
        x_gen = torch.zeros((G, 7), dtype=torch.float32, device=device)
        pg = torch.clamp(S.real[gen_bus], min=0.0)
        x_gen[:, PG_H] = signed_log1p(pg) if feature_transform == "signed_log" else pg
        x_gen[:, MIN_PG] = -10.0
        x_gen[:, MAX_PG] = 10.0
        x_gen[:, G_ON] = 1.0

    edge_bus, edge_attr_bus = make_branch_edge_features(
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
        dtype=torch.float32,
        feature_transform=feature_transform,
    )
    gen_idx = torch.arange(G, dtype=torch.long, device=device)
    gen_to_bus = torch.stack([gen_idx, gen_bus], dim=0)
    bus_to_gen = torch.stack([gen_bus, gen_idx], dim=0)

    x_dict = {"bus": x_bus, "gen": x_gen}
    edge_index_dict = {
        ("bus", "connects", "bus"): edge_bus,
        ("gen", "connected_to", "bus"): gen_to_bus,
        ("bus", "connected_to", "gen"): bus_to_gen,
    }
    edge_attr_dict = {
        ("bus", "connects", "bus"): edge_attr_bus,
        ("gen", "connected_to", "bus"): None,
        ("bus", "connected_to", "gen"): None,
    }
    mask_dict = {
        "bus": torch.zeros_like(x_bus, dtype=torch.bool),
        "gen": torch.zeros((G, 7), dtype=torch.bool, device=device),
        "PQ": x_bus[:, PQ_H].bool(),
        "PV": x_bus[:, PV_H].bool(),
        "REF": x_bus[:, REF_H].bool(),
    }
    mask_dict["bus"][:, VM_H] = True
    mask_dict["bus"][:, VA_H] = True
    mask_dict["gen"][:, PG_H] = True
    target = Vnewton.squeeze(0)
    start = Vstart.squeeze(0)
    return x_dict, edge_index_dict, edge_attr_dict, mask_dict, start, target


class GridFMHeteroSurrogate(nn.Module):
    """GridFM HGNS-style hetero TransformerConv surrogate for Vm/theta."""

    def __init__(
        self,
        input_bus_dim=15,
        input_gen_dim=7,
        edge_dim=10,
        hidden_size=48,
        heads=8,
        num_layers=12,
        dropout=0.0,
        zero_init_head=False,
        vmin=0.5,
        vmax=1.5,
        out_bus_dim=2,
        pin_known=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.num_layers = num_layers
        self.vmin = vmin
        self.vmax = vmax
        # 2 = (dVM, dVA), the voltage-only head. 6 additionally emits
        # (PG, QG, PD, QD) so GridFM's own MaskedReconstructionMSE, which
        # supervises all six quantities, can be used -- see native_loss.py.
        self.out_bus_dim = out_bus_dim
        # In AC power flow |V| is an input at PV and slack buses and the angle
        # is an input at the slack; only the remaining quantities are unknown.
        # With pin_known the head's correction is masked off at those buses so
        # the given values pass through exactly. Left False by default because
        # the first PF campaign ran without it and its numbers must stay
        # reproducible; measured on case118, not pinning moved |V| away from
        # the given value by 2.3e-3 pu on average (max 1.6e-2) at the 46% of
        # buses where it was already known, and the slack angle by 0.15 deg.
        self.pin_known = pin_known

        self.input_proj_bus = nn.Sequential(
            nn.Linear(input_bus_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.input_proj_gen = nn.Sequential(
            nn.Linear(input_gen_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.input_proj_edge = nn.Sequential(
            nn.Linear(edge_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.layers = nn.ModuleList()
        self.norms_bus = nn.ModuleList()
        self.norms_gen = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_size if i == 0 else hidden_size * heads
            conv = HeteroConv(
                {
                    ("bus", "connects", "bus"): TransformerConv(
                        in_dim,
                        hidden_size,
                        heads=heads,
                        edge_dim=hidden_size,
                        dropout=dropout,
                        beta=True,
                    ),
                    ("gen", "connected_to", "bus"): TransformerConv(
                        in_dim,
                        hidden_size,
                        heads=heads,
                        dropout=dropout,
                        beta=True,
                    ),
                    ("bus", "connected_to", "gen"): TransformerConv(
                        in_dim,
                        hidden_size,
                        heads=heads,
                        dropout=dropout,
                        beta=True,
                    ),
                },
                aggr="sum",
            )
            self.layers.append(conv)
            self.norms_bus.append(nn.LayerNorm(hidden_size * heads))
            self.norms_gen.append(nn.LayerNorm(hidden_size * heads))

        self.activation = nn.LeakyReLU()
        self.mlp_bus = nn.Sequential(
            nn.Linear(hidden_size * heads, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_bus_dim),
        )
        if zero_init_head:
            nn.init.zeros_(self.mlp_bus[-1].weight)
            nn.init.zeros_(self.mlp_bus[-1].bias)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        h_bus = self.input_proj_bus(x_dict["bus"])
        h_gen = self.input_proj_gen(x_dict["gen"])
        edge_attr_proj = {
            key: self.input_proj_edge(val) if val is not None else None
            for key, val in edge_attr_dict.items()
        }

        for conv, norm_bus, norm_gen in zip(self.layers, self.norms_bus, self.norms_gen):
            out = conv(
                {"bus": h_bus, "gen": h_gen},
                edge_index_dict,
                edge_attr_proj,
            )
            out_bus = self.activation(norm_bus(out["bus"]))
            out_gen = self.activation(norm_gen(out["gen"]))
            h_bus = h_bus + out_bus if h_bus.shape == out_bus.shape else out_bus
            h_gen = h_gen + out_gen if h_gen.shape == out_gen.shape else out_gen

        delta = self.mlp_bus(h_bus)
        dv, dth = delta[:, 0], delta[:, 1]
        if self.pin_known:
            # PQ_H / REF_H are one-hots the feature builder already sets, so no
            # extra input is needed. |V| is free only at PQ; the angle is free
            # everywhere except the reference bus.
            is_pq = x_dict["bus"][:, PQ_H] > 0.5
            is_ref = x_dict["bus"][:, REF_H] > 0.5
            dv = torch.where(is_pq, dv, torch.zeros_like(dv))
            dth = torch.where(is_ref, torch.zeros_like(dth), dth)
        v = torch.clamp(x_dict["bus"][:, VM_H] + dv, self.vmin, self.vmax)
        th = x_dict["bus"][:, VA_H] + dth
        th = torch.atan2(torch.sin(th), torch.cos(th))
        V = torch.stack([v, th], dim=-1).unsqueeze(0)
        if self.out_bus_dim <= 2:
            return V
        # [VM, VA, PG, QG, PD, QD] in GridFM's output-column order. VM/VA reuse
        # the residual-corrected values so the two heads cannot disagree; the
        # remaining four are emitted directly.
        pred6 = torch.cat([torch.stack([v, th], dim=-1), delta[:, 2:]], dim=-1)
        self._last_pred6 = pred6
        return V


def run():
    args = parse_args()
    set_seed(args.seed_value)

    if args.task != "pf" and args.kol_pf_mode != "off":
        raise ValueError(
            "--kol_pf_mode is an AC power-flow feasibility operator and cannot "
            "be applied to --task opf, where generator dispatch is a decision."
        )
    if args.gridfm_impl == "graphkit_projected" and args.kol_pf_mode != "off":
        raise ValueError(
            "graphkit_projected already contains the DPF known operator inside "
            "every layer; appending --kol_pf_mode dpf would attach a second solver."
        )

    if not args.run_name:
        args.run_name = (
            f"gridfm_{args.gridfm_impl}_{known_operator_tag(args)}_"
            f"{Path(args.PARQUET).stem}"
        )
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs("./results/plots", exist_ok=True)

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
        f"[model] GridFM-HGNS hidden={args.hidden_size} layers={args.num_layers} "
        f"heads={args.n_heads} dropout={args.dropout} vn={args.vn_feature_mode} "
        f"feature_transform={args.feature_transform} "
        f"graphkit_neural_edge_transform={args.graphkit_neural_edge_transform}"
    )
    print(
        f"[loss] mse_weight={args.mse_weight} physics_weight={args.physics_weight} "
        f"physics_form={args.physics_loss_form} "
        f"physics_dtype={args.physics_compute_dtype} "
        f"physics_residual_norm={args.physics_residual_norm}"
    )
    print(f"[known-operator] {known_operator_tag(args)} raw_loss_weight={args.kol_raw_loss_weight:g}")

    is_opf = args.task == "opf"
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
    val_loader, _ = make_loader(val_ds, ddp, shuffle=False, **loader_kwargs)
    # Undistributed on purpose: every rank walks the whole test set, so the
    # reported figure is computed exactly as in the single-GPU runs and stays
    # comparable with the existing tables. Summing identical replicas and
    # dividing by the summed count returns the same mean.
    test_loader, _ = make_loader(test_ds, ddp, shuffle=False, distributed=False,
                                 **loader_kwargs)

    if args.gridfm_impl.startswith("graphkit"):
        from gridfm_graphkit_adapter import (
            build_graphkit_model,
            configure_neural_edge_transform,
            make_args,
        )
        # Bus/gen feature layouts are byte-identical to gridfm_graphkit's own
        # globals, so only capacity is ours to choose.
        if args.gridfm_impl == "graphkit":
            model = build_graphkit_model(
                task_name=("OptimalPowerFlow" if is_opf else "PowerFlow"),
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                attention_head=args.n_heads,
            ).to(device)
        elif args.gridfm_impl == "graphkit_projected":
            if is_opf:
                raise SystemExit("--gridfm_impl graphkit_projected is power-flow only")
            from gridfm_projected_dpf_hybrid import GNS_heterogeneous_ProjectedDPF
            model = GNS_heterogeneous_ProjectedDPF(
                make_args("PowerFlow", args.hidden_size, args.num_layers,
                          args.n_heads, args.dropout),
                dpf_lr=args.projected_dpf_lr,
                beta1=args.projected_dpf_beta1,
                beta2=args.projected_dpf_beta2,
                descent_mu=args.projected_dpf_mu,
                armijo_max_backtracks=args.armijo_max_backtracks,
                vmin=args.vmin,
                vmax=args.vmax,
                max_vm_fraction=args.projected_dpf_max_vm_fraction,
                max_angle_step=args.projected_dpf_max_angle_step,
                first_order=args.projected_dpf_first_order,
            ).to(device)
        else:
            if is_opf:
                raise SystemExit(
                    f"--gridfm_impl {args.gridfm_impl} is power-flow only; its "
                    f"line-search objective is the power-balance residual.")
            from genco_ls import GNS_heterogeneous_LS
            model = GNS_heterogeneous_LS(
                make_args("PowerFlow", args.hidden_size, args.num_layers,
                          args.n_heads, 0.0),
                line_search=(args.gridfm_impl == "graphkit_ls"),
                armijo_max_backtracks=args.armijo_max_backtracks,
            ).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        configure_neural_edge_transform(
            model, mode=args.graphkit_neural_edge_transform
        )
        print(f"[model] gridfm_impl={args.gridfm_impl} trainable_params={n_params:,}")
    else:
        model = GridFMHeteroSurrogate(
            hidden_size=args.hidden_size,
            heads=args.n_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            zero_init_head=args.zero_init_head,
            vmin=args.vmin,
            vmax=args.vmax,
            pin_known=(args.pin_known and not is_opf),
        ).to(device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        load_result = model.load_state_dict(ckpt, strict=args.init_strict)
        print(
            f"[init] loaded checkpoint {args.init_checkpoint} "
            f"strict={args.init_strict} missing={len(load_result.missing_keys)} "
            f"unexpected={len(load_result.unexpected_keys)}"
        )
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
        opf_sums = {}
        dq_values = []
        kol_initial_values = []
        kol_final_values = []
        kol_converged_values = []
        kol_iteration_values = []
        projected_accepted = 0
        projected_steps = 0
        projected_monotonic_violations = 0
        projected_max_cost_increase = 0.0
        projected_initial_cost = 0.0
        projected_final_cost = 0.0
        projected_graphs = 0

        if train:
            accum.zero()

        with torch.set_grad_enabled(train):
            for batch in loader:
                B_eff = int(batch["sizes"].numel()) if "sizes" in batch else int(batch["bus_type"].size(0))
                n_graphs += B_eff
                n_nodes = batch["sizes"].to(device) if "sizes" in batch else None

                batch_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
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

                space_dev = (opf_decision_space(batch_dev, batch["sizes"], device)
                             if is_opf else None)
                x_dict, edge_index_dict, edge_attr_dict, _, _, target = make_gridfm_inputs(
                    batch_dev,
                    device,
                    args.vn_feature_mode,
                    args.feature_transform,
                    opf_space=space_dev,
                    zero_unspecified_q=args.zero_unspecified_q,
                )
                if args.gridfm_impl.startswith("graphkit"):
                    from gridfm_graphkit_adapter import forward_graphkit_parquet
                    Vraw = forward_graphkit_parquet(
                        model, batch_dev, device,
                        task_name=("OptimalPowerFlow" if is_opf else "PowerFlow"),
                        feature_transform=args.feature_transform,
                        vn_feature_mode=args.vn_feature_mode,
                        opf_space=space_dev,
                        zero_unspecified_q=args.zero_unspecified_q,
                    )
                else:
                    Vraw = model(x_dict, edge_index_dict, edge_attr_dict)

                if args.gridfm_impl == "graphkit_projected" and model.cost_history:
                    costs = [model.initial_cost] + model.cost_history
                    projected_accepted += sum(
                        int((alpha > 0).sum().item()) for alpha in model.alpha_history
                    )
                    projected_steps += sum(alpha.numel() for alpha in model.alpha_history)
                    for before, after in zip(costs[:-1], costs[1:]):
                        increase = after - before
                        tolerance = 1e-7 * torch.maximum(
                            before.abs(), torch.ones_like(before)
                        )
                        projected_monotonic_violations += int(
                            (increase > tolerance).sum().item()
                        )
                        projected_max_cost_increase = max(
                            projected_max_cost_increase,
                            float(increase.max().item()),
                        )
                    projected_initial_cost += float(costs[0].sum().item())
                    projected_final_cost += float(costs[-1].sum().item())
                    projected_graphs += costs[0].numel()

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

                target_b = target.unsqueeze(0)
                dmag = Vpred[..., 0] - target_b[..., 0]
                dang = angle_diff(Vpred[..., 1], target_b[..., 1])
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
                    physics_dtype = (
                        torch.complex128
                        if args.physics_compute_dtype == "complex128"
                        else torch.complex64
                    )
                    if args.physics_residual_norm == "local_ybus":
                        # The diagnostic local-normalized loss is deliberately
                        # MSE: after Jacobi scaling its residual is already
                        # dimensionless and should retain a quadratic basin.
                        phys = power_flow_residual_loss(
                            Y_metric.to(physics_dtype),
                            Vpred,
                            Sstart_metric.to(physics_dtype),
                            bus_type,
                            sizes=n_nodes,
                            residual_norm="local_ybus",
                        )
                    else:
                        phys = ppc_physics_loss(
                            Y_metric.to(physics_dtype),
                            Vpred,
                            Sstart_metric.to(physics_dtype),
                            bus_type,
                            form=args.physics_loss_form,
                            huber_delta=args.physics_huber_delta,
                        )
                loss = args.mse_weight * mse + args.physics_weight * phys
                if known_operator is not None and args.kol_raw_loss_weight > 0.0:
                    raw_dmag = Vraw[..., 0] - target_b[..., 0]
                    raw_dang = angle_diff(Vraw[..., 1], target_b[..., 1])
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
                if is_opf and args.dispatch_weight > 0.0:
                    # GridFM has no generator head, so only the implied dispatch
                    # can be supervised; --gen_head_weight is a no-op here.
                    tgt_gen = target_dispatch(batch_dev, space_dev, device)
                    if tgt_gen is not None:
                        loss = loss + args.dispatch_weight * dispatch_loss(
                            Y_metric, Vpred, Sstart_metric, space_dev, tgt_gen)

                if train:
                    # Clip and step once per accumulation group; with
                    # accum_steps == 1 this is the original per-batch update.
                    accum.backward(loss)

                if is_opf:
                    om = opf_metrics(Y_metric, Vpred.detach(), target_b, Sstart_metric,
                                     space_dev, batch["sizes"])
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
        projected_summary = None
        if projected_steps:
            projected_summary = {
                "accepted_rate": projected_accepted / projected_steps,
                "monotonic_violations": projected_monotonic_violations,
                "max_cost_increase": projected_max_cost_increase,
                "initial_cost": projected_initial_cost / max(projected_graphs, 1),
                "final_cost": projected_final_cost / max(projected_graphs, 1),
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
            "projected": projected_summary,
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
        projected_text = ""
        if m.get("projected"):
            p = m["projected"]
            projected_text = (
                f"projected(accept {p['accepted_rate']:.1%}, "
                f"cost {p['initial_cost']:.3e}->{p['final_cost']:.3e}, "
                f"viol {p['monotonic_violations']}, "
                f"max_inc {p['max_cost_increase']:.1e}) "
            )
        return (
            f"{prefix} loss {m['loss']:.4e} mse {m['mse']:.4e} phys {m['phys']:.4e} "
            f"rmse {rmse:.4e} (mag {rmse_mag:.4e}, ang {rmse_ang_deg:.4e}deg) "
            f"(dPinf {m['max_dp_pu']:.3e} pu, dQinf {m['max_dq_pu']:.3e} pu; "
            f"{m['max_dp_mva']:.3e} MW, {m['max_dq_mva']:.3e} MVAr) "
            f"{format_residual_distribution_compact(m['dist'])} "
            f"{projected_text}"
            f"{format_known_operator_diagnostics(m.get('kol'))}"
        )

    print("Initial metrics before training:")
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
