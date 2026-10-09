"""Differentiable HELM-style known operator for AC power flow.

This is not a wrapper around the NumPy/SciPy HELMpy package.  HELMpy has no
autograd path and no learned initial-guess interface.  Instead, this module
implements an endpoint-preserving rectangular holomorphic embedding directly
in PyTorch:

    h(x(s)) = h(x0) + sum_m c_m s**m,
    sum_m c_m = b - h(x0).

``h`` contains the standard AC equations (P/Q at PQ buses, P/|V|^2 at PV
buses, and rectangular voltage at the slack).  A graph network predicts only
how the fixed endpoint displacement is distributed over the embedding powers.
The endpoint at s=1 therefore remains the original AC-PF problem.  Series
coefficients are obtained with a constant real linear operator and evaluated
with diagonal Pade approximants.

Finite truncation is not a feasibility proof.  The returned state must still
be certified with the actual masked P/Q residual, exactly as for any numerical
power-flow solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from collate_blockdiag_optimized_complex_columns import ybus_matvec
from GNSMsg_SelfAttention_armijo import EdgeSelfAttnBlock, _build_directed_edges_single


@dataclass(frozen=True)
class HELMKOLConfig:
    series_order: int = 8
    path_order: int = 3
    pade_regularization: float = 1e-7
    linear_regularization: float = 1e-7
    select_best_eval_order: bool = True

    def validate(self) -> None:
        if self.series_order < 4 or self.series_order % 2:
            raise ValueError("series_order must be an even integer >= 4")
        if self.path_order < 1 or self.path_order > self.series_order:
            raise ValueError("path_order must lie in [1, series_order]")
        if self.pade_regularization < 0 or self.linear_regularization < 0:
            raise ValueError("HELM regularization values must be non-negative")


def add_helm_kol_args(parser) -> None:
    group = parser.add_argument_group("PIGNN + HELM known-operator learning")
    group.add_argument("--helm_series_order", type=int, default=8)
    group.add_argument("--helm_path_order", type=int, default=3)
    group.add_argument("--helm_pade_regularization", type=float, default=1e-7)
    group.add_argument("--helm_linear_regularization", type=float, default=1e-7)
    group.add_argument(
        "--helm_uniform_path",
        action="store_true",
        help=(
            "Bypass PIGNN and use fixed uniform embedding weights at every bus. "
            "Use with --mode valid_test for the untrained HELM-path ablation."
        ),
    )
    group.add_argument(
        "--helm_no_best_eval_order",
        action="store_true",
        help="Use only the maximum Pade order at evaluation instead of selecting the lowest-residual candidate.",
    )


def _split_blockdiag_y(Y: torch.Tensor, n: int) -> torch.Tensor:
    """Return the first n-by-n grid block without densifying the mega-graph."""
    if Y.dim() != 2:
        if Y.dim() == 3 and Y.shape[0] == 1:
            Y = Y[0]
        else:
            raise ValueError(f"Expected a 2-D block-diagonal Ybus, got {tuple(Y.shape)}")
    if Y.is_sparse:
        Y = Y.coalesce()
        idx = Y.indices()
        keep = (idx[0] < n) & (idx[1] < n)
        block = torch.sparse_coo_tensor(
            idx[:, keep], Y.values()[keep], (n, n), dtype=Y.dtype, device=Y.device
        ).coalesce()
        return block.to_dense()
    return Y[:n, :n]


def _reshape_equal_graphs(
    value: torch.Tensor, sizes: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, int, int]:
    if sizes is None:
        return value, value.shape[0], value.shape[1]
    sizes_list = [int(x) for x in sizes.detach().cpu().tolist()]
    if not sizes_list or len(set(sizes_list)) != 1:
        raise ValueError("HELM KOL currently requires equal-size graphs in each batch")
    n = sizes_list[0]
    graphs = len(sizes_list)
    if value.shape[0] != 1 or value.shape[1] != graphs * n:
        raise ValueError("sizes do not match the block-diagonal tensor layout")
    return value.reshape(graphs, n, *value.shape[2:]), graphs, n


def _flat_germ(Vfixed: torch.Tensor, bus_type: torch.Tensor) -> torch.Tensor:
    """Canonical feasible germ for fixed voltage constraints."""
    vm = torch.ones_like(Vfixed[..., 0])
    va = torch.zeros_like(Vfixed[..., 1])
    slack = bus_type == 1
    pv = bus_type == 2
    vm = torch.where(slack | pv, Vfixed[..., 0], vm)
    va = torch.where(slack, Vfixed[..., 1], va)
    return torch.stack([vm, va], dim=-1)


def _polar_to_rect(V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vm, va = V.unbind(dim=-1)
    vr = vm * torch.cos(va)
    vi = vm * torch.sin(va)
    return vr, vi, torch.complex(vr, vi)


def _equation_value(
    Y: torch.Tensor, vr: torch.Tensor, vi: torch.Tensor, bus_type: torch.Tensor
) -> torch.Tensor:
    Vc = torch.complex(vr, vi)
    current = Vc @ Y.transpose(0, 1)
    Scalc = Vc * current.conj()
    slack = bus_type == 1
    pv = bus_type == 2
    first = torch.where(slack, vr, Scalc.real)
    second = torch.where(slack, vi, torch.where(pv, vr.square() + vi.square(), Scalc.imag))
    return torch.cat([first, second], dim=-1)


def _equation_target(
    Sset: torch.Tensor, Vfixed: torch.Tensor, bus_type: torch.Tensor
) -> torch.Tensor:
    vr, vi, _ = _polar_to_rect(Vfixed)
    slack = bus_type == 1
    pv = bus_type == 2
    first = torch.where(slack, vr, Sset.real)
    second = torch.where(slack, vi, torch.where(pv, Vfixed[..., 0].square(), Sset.imag))
    return torch.cat([first, second], dim=-1)


def _rectangular_jacobian(
    Y: torch.Tensor, vr: torch.Tensor, vi: torch.Tensor, bus_type: torch.Tensor
) -> torch.Tensor:
    """Jacobian of [P/Q, P/Vm2, Vr/Vi] equations at one germ."""
    n = vr.numel()
    G, B = Y.real, Y.imag
    Ir = G @ vr - B @ vi
    Ii = B @ vr + G @ vi
    eye = torch.eye(n, dtype=vr.dtype, device=vr.device)

    dP_dvr = eye * Ir[:, None] + vr[:, None] * G + vi[:, None] * B
    dP_dvi = eye * Ii[:, None] - vr[:, None] * B + vi[:, None] * G
    dQ_dvr = -eye * Ii[:, None] + vi[:, None] * G - vr[:, None] * B
    dQ_dvi = eye * Ir[:, None] - vi[:, None] * B - vr[:, None] * G

    A = torch.empty((2 * n, 2 * n), dtype=vr.dtype, device=vr.device)
    A[:n, :n] = dP_dvr
    A[:n, n:] = dP_dvi
    A[n:, :n] = dQ_dvr
    A[n:, n:] = dQ_dvi

    slack = bus_type == 1
    pv = bus_type == 2
    for i in torch.nonzero(slack, as_tuple=False).flatten().tolist():
        A[i].zero_(); A[i, i] = 1.0
        A[n + i].zero_(); A[n + i, n + i] = 1.0
    for i in torch.nonzero(pv, as_tuple=False).flatten().tolist():
        A[n + i].zero_()
        A[n + i, i] = 2.0 * vr[i]
        A[n + i, n + i] = 2.0 * vi[i]
    return A


def _pade_at_one(coefficients: torch.Tensor, order: int, regularization: float) -> torch.Tensor:
    """Evaluate a batched diagonal [L/L] Pade approximant at s=1."""
    if order < 2 or order % 2:
        raise ValueError("Pade order must be positive and even")
    series = coefficients[..., : order + 1]
    L = order // 2
    rows = []
    for r in range(L):
        k = L + 1 + r
        rows.append(torch.stack([series[..., k - j] for j in range(1, L + 1)], dim=-1))
    matrix = torch.stack(rows, dim=-2)
    rhs = -series[..., L + 1 : 2 * L + 1].unsqueeze(-1)
    if regularization:
        eye = torch.eye(L, dtype=matrix.dtype, device=matrix.device)
        matrix = matrix + regularization * eye
    denominator_tail = torch.linalg.solve(matrix, rhs).squeeze(-1)
    denominator = torch.cat([torch.ones_like(denominator_tail[..., :1]), denominator_tail], dim=-1)

    numerator_terms = []
    for k in range(L + 1):
        term = torch.zeros_like(series[..., 0])
        for j in range(k + 1):
            term = term + denominator[..., j] * series[..., k - j]
        numerator_terms.append(term)
    numerator_at_one = torch.stack(numerator_terms, dim=-1).sum(dim=-1)
    denominator_at_one = denominator.sum(dim=-1)
    eps = torch.finfo(denominator_at_one.real.dtype).eps
    safe = torch.where(
        denominator_at_one.abs() < eps,
        denominator_at_one + eps,
        denominator_at_one,
    )
    return numerator_at_one / safe


class DifferentiableHELMPath(nn.Module):
    """Power-series and Pade known operator with a learned endpoint-safe path."""

    def __init__(self, config: HELMKOLConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.last_diagnostics: Dict[str, torch.Tensor] = {}
        # Factorization is grid/germ-only and must not be repeated for every
        # minibatch. These are deliberately ordinary attributes rather than
        # checkpoint buffers; they are rebuilt after loading or moving device.
        self._cached_signature: Optional[torch.Tensor] = None
        self._cached_lu: Optional[torch.Tensor] = None
        self._cached_pivots: Optional[torch.Tensor] = None

    def _solve_coefficients(
        self,
        Y: torch.Tensor,
        Sset: torch.Tensor,
        Vfixed: torch.Tensor,
        bus_type: torch.Tensor,
        path_logits: torch.Tensor,
    ) -> Sequence[torch.Tensor]:
        batch, n = bus_type.shape
        germ = _flat_germ(Vfixed, bus_type)
        vr0, vi0, Vc0 = _polar_to_rect(germ)
        h0 = _equation_value(Y, vr0, vi0, bus_type)
        target = _equation_target(Sset, Vfixed, bus_type)
        delta = target - h0

        # P and Q/PV-equation path weights are learned independently. Fixed
        # slack equations have zero endpoint displacement by construction.
        weights = torch.softmax(path_logits, dim=-1)
        if weights.shape != (batch, 2 * n, self.config.path_order):
            raise ValueError(
                f"path_logits must have shape {(batch, 2*n, self.config.path_order)}, "
                f"got {tuple(weights.shape)}"
            )
        path_coeff = delta.unsqueeze(-1) * weights

        coefficients = [Vc0]
        rectangular = [torch.cat([vr0, vi0], dim=-1)]

        # The canonical germ is identical across scenarios for the corpus.
        # If fixed voltage setpoints differ, factor and solve each sample to
        # preserve correctness rather than silently sharing the wrong matrix.
        reg = self.config.linear_regularization
        shared_germ = bool(
            torch.equal(bus_type, bus_type[:1].expand_as(bus_type))
            and torch.allclose(vr0, vr0[:1].expand_as(vr0), atol=1e-7, rtol=1e-7)
            and torch.allclose(vi0, vi0[:1].expand_as(vi0), atol=1e-7, rtol=1e-7)
        )
        shared_factor = None
        individual_factors = None
        if shared_germ:
            signature = torch.cat([
                vr0[0].detach(), vi0[0].detach(), bus_type[0].to(vr0.dtype)
            ])
            cache_valid = (
                self._cached_signature is not None
                and self._cached_signature.device == signature.device
                and self._cached_signature.dtype == signature.dtype
                and torch.equal(self._cached_signature, signature)
                and self._cached_lu is not None
                and self._cached_lu.shape == (2 * n, 2 * n)
            )
            if not cache_valid:
                A = _rectangular_jacobian(Y, vr0[0], vi0[0], bus_type[0])
                if reg:
                    A = A + reg * torch.eye(2 * n, dtype=A.dtype, device=A.device)
                lu, pivots = torch.linalg.lu_factor(A)
                self._cached_signature = signature
                self._cached_lu = lu.detach()
                self._cached_pivots = pivots.detach()
            shared_factor = (self._cached_lu, self._cached_pivots)
        else:
            individual_factors = []
            eye = torch.eye(2 * n, dtype=vr0.dtype, device=vr0.device)
            for b in range(batch):
                A = _rectangular_jacobian(Y, vr0[b], vi0[b], bus_type[b])
                if reg:
                    A = A + reg * eye
                individual_factors.append(torch.linalg.lu_factor(A))

        current_coefficients = [Vc0 @ Y.transpose(0, 1)]
        slack = bus_type == 1
        pv = bus_type == 2
        for order in range(1, self.config.series_order + 1):
            remainder_p = torch.zeros_like(vr0)
            remainder_q = torch.zeros_like(vi0)
            remainder_vm2 = torch.zeros_like(vr0)
            for k in range(1, order):
                Vk = coefficients[k]
                Ik_other = current_coefficients[order - k]
                product = Vk * Ik_other.conj()
                remainder_p = remainder_p + product.real
                remainder_q = remainder_q + product.imag
                remainder_vm2 = remainder_vm2 + (
                    Vk.real * coefficients[order - k].real
                    + Vk.imag * coefficients[order - k].imag
                )

            first_remainder = torch.where(slack, torch.zeros_like(remainder_p), remainder_p)
            second_remainder = torch.where(
                slack,
                torch.zeros_like(remainder_q),
                torch.where(pv, remainder_vm2, remainder_q),
            )
            remainder = torch.cat([first_remainder, second_remainder], dim=-1)
            forcing = (
                path_coeff[..., order - 1]
                if order <= self.config.path_order
                else torch.zeros_like(delta)
            )
            rhs = forcing - remainder
            if shared_factor is not None:
                lu, pivots = shared_factor
                zn = torch.linalg.lu_solve(lu, pivots, rhs.transpose(0, 1)).transpose(0, 1)
            else:
                solved = []
                for b, (lu, pivots) in enumerate(individual_factors):
                    solved.append(
                        torch.linalg.lu_solve(lu, pivots, rhs[b].unsqueeze(-1)).squeeze(-1)
                    )
                zn = torch.stack(solved, dim=0)
            Vn = torch.complex(zn[:, :n], zn[:, n:])
            rectangular.append(zn)
            coefficients.append(Vn)
            current_coefficients.append(Vn @ Y.transpose(0, 1))
        return coefficients

    @staticmethod
    def _project_fixed(Vc: torch.Tensor, Vfixed: torch.Tensor, bus_type: torch.Tensor) -> torch.Tensor:
        slack = bus_type == 1
        pv = bus_type == 2
        _, _, Vfixed_c = _polar_to_rect(Vfixed)
        angle = torch.angle(Vc)
        pv_voltage = Vfixed[..., 0].to(Vc.real.dtype) * torch.exp(1j * angle)
        Vc = torch.where(pv, pv_voltage, Vc)
        return torch.where(slack, Vfixed_c, Vc)

    @staticmethod
    def _max_residual(
        Y: torch.Tensor, Vc: torch.Tensor, Sset: torch.Tensor, bus_type: torch.Tensor
    ) -> torch.Tensor:
        Scalc = Vc * (Vc @ Y.transpose(0, 1)).conj()
        slack = bus_type == 1
        pv = bus_type == 2
        dP = (Scalc.real - Sset.real).masked_fill(slack, 0.0)
        dQ = (Scalc.imag - Sset.imag).masked_fill(slack | pv, 0.0)
        return torch.maximum(dP.abs().amax(dim=-1), dQ.abs().amax(dim=-1))

    def forward(
        self,
        Y: torch.Tensor,
        Sset: torch.Tensor,
        Vfixed: torch.Tensor,
        bus_type: torch.Tensor,
        path_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        coefficients = self._solve_coefficients(Y, Sset, Vfixed, bus_type, path_logits)
        stacked = torch.stack(coefficients, dim=-1)
        orders = (
            [self.config.series_order]
            if self.training
            else list(range(4, self.config.series_order + 1, 2))
        )
        candidates = [
            self._project_fixed(
                _pade_at_one(stacked, order, self.config.pade_regularization),
                Vfixed,
                bus_type,
            )
            for order in orders
        ]
        residuals = torch.stack(
            [self._max_residual(Y, candidate, Sset, bus_type) for candidate in candidates],
            dim=-1,
        )
        if self.training or not self.config.select_best_eval_order:
            selected = candidates[-1]
            selected_index = torch.full(
                (Sset.shape[0],), len(candidates) - 1, dtype=torch.long, device=Sset.device
            )
        else:
            selected_index = residuals.argmin(dim=-1)
            all_candidates = torch.stack(candidates, dim=1)
            selected = all_candidates[
                torch.arange(Sset.shape[0], device=Sset.device), selected_index
            ]
        final_residual = residuals.gather(1, selected_index[:, None]).squeeze(1)
        diagnostics = {
            "max_mismatch": final_residual.detach(),
            "selected_order": torch.tensor(orders, device=Sset.device)[selected_index].detach(),
            "candidate_residuals": residuals.detach(),
        }
        self.last_diagnostics = diagnostics
        return selected, diagnostics


class PIGNNHELMKOL(nn.Module):
    """Edge-aware PIGNN controller inside an endpoint-preserving HELM path."""

    def __init__(
        self,
        *,
        d_model: int = 24,
        n_heads: int = 8,
        num_attn_layers: int = 8,
        attn_dropout: float = 0.0,
        series_order: int = 8,
        path_order: int = 3,
        pade_regularization: float = 1e-7,
        linear_regularization: float = 1e-7,
        select_best_eval_order: bool = True,
        physics_loss_form: str = "logcosh",
        physics_huber_delta: float = 1.0,
        uniform_path: bool = False,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.path_order = path_order
        self.physics_loss_form = physics_loss_form
        self.physics_huber_delta = physics_huber_delta
        self.uniform_path = bool(uniform_path)
        self.in_proj = nn.Linear(7, d_model)
        self.blocks = nn.ModuleList([
            EdgeSelfAttnBlock(d_model, n_heads, 9, 4 * d_model, attn_dropout)
            for _ in range(num_attn_layers)
        ])
        self.path_head = nn.Linear(d_model, 2 * path_order)
        nn.init.zeros_(self.path_head.weight)
        nn.init.zeros_(self.path_head.bias)
        self.helm = DifferentiableHELMPath(HELMKOLConfig(
            series_order=series_order,
            path_order=path_order,
            pade_regularization=pade_regularization,
            linear_regularization=linear_regularization,
            select_best_eval_order=select_best_eval_order,
        ))

    def _physics_loss(
        self, Y: torch.Tensor, Vc: torch.Tensor, Sset: torch.Tensor, bus_type: torch.Tensor
    ) -> torch.Tensor:
        Scalc = Vc * (Vc @ Y.transpose(0, 1)).conj()
        slack = bus_type == 1
        pv = bus_type == 2
        residual = torch.cat([
            (Scalc.real - Sset.real)[~slack],
            (Scalc.imag - Sset.imag)[~(slack | pv)],
        ])
        if residual.numel() == 0:
            return Vc.real.sum() * 0.0
        if self.physics_loss_form == "mse":
            return residual.square().mean()
        if self.physics_loss_form == "huber":
            return F.huber_loss(
                residual, torch.zeros_like(residual), delta=self.physics_huber_delta
            )
        if self.physics_loss_form == "logcosh":
            return (residual + F.softplus(-2.0 * residual) - math.log(2.0)).mean()
        raise ValueError(f"Unknown physics_loss_form={self.physics_loss_form!r}")

    def forward(
        self,
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
        Is_trafo,
        Y,
        S,
        V0,
        n_nodes_per_graph=None,
        Y_shunt_bus=None,
        vn_log=None,
    ):
        if Y is None:
            raise ValueError("PIGNNHELMKOL requires the sparse Ybus")
        Vbatch, graphs, n = _reshape_equal_graphs(V0, n_nodes_per_graph)
        Sbatch, _, _ = _reshape_equal_graphs(S.unsqueeze(-1), n_nodes_per_graph)
        Sbatch = Sbatch.squeeze(-1)
        type_batch, _, _ = _reshape_equal_graphs(bus_type.unsqueeze(-1), n_nodes_per_graph)
        type_batch = type_batch.squeeze(-1)
        Ygrid = _split_blockdiag_y(Y, n)

        if self.uniform_path:
            # Exactly uniform after softmax and independent of all PIGNN
            # parameters. This is the deterministic, untrained HELM ablation.
            logits = Vbatch.new_zeros((graphs, 2 * n, self.path_order))
        else:
            germ = _flat_germ(Vbatch, type_batch)
            _, _, Vgerm = _polar_to_rect(germ)
            Scalc0 = Vgerm * (Vgerm @ Ygrid.transpose(0, 1)).conj()
            dP = Sbatch.real - Scalc0.real
            dQ = Sbatch.imag - Scalc0.imag
            slack = type_batch == 1
            pv = type_batch == 2
            dP = dP.masked_fill(slack, 0.0)
            dQ = dQ.masked_fill(slack | pv, 0.0)
            onehot = torch.stack([slack, pv, ~(slack | pv)], dim=-1).to(Vbatch.dtype)
            features = torch.cat([Vbatch, dP.unsqueeze(-1), dQ.unsqueeze(-1), onehot], dim=-1)
            features = features.to(dtype=self.in_proj.weight.dtype)
            x = self.in_proj(features.reshape(1, graphs * n, -1))

            edge_index, edge_features = _build_directed_edges_single(
                Branch_f_bus.squeeze(0), Branch_t_bus.squeeze(0), Branch_status.squeeze(0),
                Branch_tau.squeeze(0), Branch_shift_deg.squeeze(0),
                Branch_y_series_from.squeeze(0), Branch_y_series_to.squeeze(0),
                Branch_y_series_ft.squeeze(0), Branch_y_shunt_from.squeeze(0),
                Branch_y_shunt_to.squeeze(0), Is_trafo.squeeze(0),
            )
            for block in self.blocks:
                x = block(x, edge_index, edge_features)
            logits = self.path_head(x).reshape(graphs, n, 2, self.path_order)
            logits = torch.cat([logits[:, :, 0], logits[:, :, 1]], dim=1)

        helm_real_dtype = torch.float64 if Ygrid.dtype == torch.complex128 else torch.float32
        logits_helm = logits.to(helm_real_dtype)
        Vc, _ = self.helm(Ygrid, Sbatch, Vbatch, type_batch, logits_helm)
        output = torch.stack([Vc.abs(), torch.angle(Vc)], dim=-1).reshape(1, graphs * n, 2)
        physics_loss = self._physics_loss(Ygrid, Vc, Sbatch, type_batch)
        return output, physics_loss
