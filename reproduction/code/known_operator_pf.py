"""Differentiable known-operator correction for AC power-flow surrogates.

The neural network supplies a warm start ``V0 = (|V|, theta)``.  This module
then minimizes the *actual* bus-type-aware AC power-flow mismatch

    F(V) = [P_calc(V) - P_set]_(PV,PQ) ||
           [Q_calc(V) - Q_set]_(PQ)

with gradients obtained from the exact differentiable power-flow equations.
It is deliberately backbone agnostic: PIGNN-Attn-LS, either GridFM
implementation, GridSFM, and LUMINA can all use the same operator.

There are two distinct regimes and the distinction matters scientifically:

* training uses a fixed number of unrolled steps so gradients can reach the
  surrogate;
* evaluation uses at most ``eval_steps`` and stops graphs independently once
  their infinity-norm mismatch is below ``tol``.

Only evaluation samples with ``diagnostics["converged"] == True`` have an
inference-time residual certificate at the requested tolerance.  A finite
unrolled block is a physics-based correction, not an unconditional guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from collate_blockdiag_optimized_complex_columns import ybus_matvec


def add_known_operator_args(parser):
    """Add the shared PF known-operator flags to an argparse parser."""
    group = parser.add_argument_group("known-operator AC power-flow correction")
    group.add_argument(
        "--kol_pf_mode",
        choices=("off", "dpf"),
        default="off",
        help=(
            "off: return the raw surrogate. dpf: append a differentiable "
            "AC power-flow mismatch optimizer. The operator is defined for "
            "the PF task, not the OPF task."
        ),
    )
    group.add_argument(
        "--kol_train_steps",
        type=int,
        default=3,
        help="Fixed number of differentiable DPF steps unrolled during training.",
    )
    group.add_argument(
        "--kol_eval_steps",
        type=int,
        default=1000,
        help="Maximum DPF steps at validation/test time; graphs stop independently.",
    )
    group.add_argument(
        "--kol_lr",
        type=float,
        default=0.003377,
        help=(
            "Voltage-state learning rate inside the known operator. The "
            "default matches the author's reference Adam configuration; "
            "their very-large-grid experiments use 1e-4 instead."
        ),
    )
    group.add_argument(
        "--kol_optimizer",
        choices=("adam", "gd"),
        default="adam",
        help="First-order optimizer used by the differentiable PF operator.",
    )
    group.add_argument(
        "--kol_tol",
        type=float,
        default=1e-6,
        help="Per-graph max masked P/Q mismatch required for certification (pu).",
    )
    group.add_argument("--kol_beta1", type=float, default=0.979681)
    group.add_argument("--kol_beta2", type=float, default=0.963442)
    group.add_argument("--kol_eps", type=float, default=1e-8)
    group.add_argument(
        "--kol_scheduler",
        choices=("author_plateau", "off"),
        default="author_plateau",
        help=(
            "Learning-rate scheduler for the adaptive evaluation solve. "
            "author_plateau matches the reference DPF Adam configuration."
        ),
    )
    group.add_argument("--kol_scheduler_factor", type=float, default=0.547191)
    group.add_argument("--kol_scheduler_patience", type=int, default=41)
    group.add_argument("--kol_scheduler_threshold", type=float, default=0.067321)
    group.add_argument("--kol_scheduler_cooldown", type=int, default=97)
    group.add_argument(
        "--kol_vmin",
        type=float,
        default=0.5,
        help="Lower numerical guard for optimized PQ-bus voltage magnitudes.",
    )
    group.add_argument(
        "--kol_vmax",
        type=float,
        default=1.5,
        help="Upper numerical guard for optimized PQ-bus voltage magnitudes.",
    )
    group.add_argument(
        "--kol_grad_clip",
        type=float,
        default=0.0,
        help="Optional elementwise clip on inner PF gradients; 0 disables it.",
    )
    group.add_argument(
        "--kol_first_order",
        action="store_true",
        help=(
            "Do not construct second derivatives through inner gradients. "
            "This lowers memory, but is only a first-order approximation to "
            "the fully differentiable unrolled operator."
        ),
    )
    group.add_argument(
        "--kol_raw_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Optional auxiliary weight on the raw surrogate loss. This is "
            "useful because an exactly converged PF solution is independent "
            "of its warm start, so gradients through a fully converged solver "
            "can vanish."
        ),
    )
    group.add_argument(
        "--kol_final_physics_weight",
        type=float,
        default=1.0,
        help=(
            "Weight for the corrected-output PF mismatch added by the PIGNN "
            "pipeline. Other pipelines already apply --physics_weight to the "
            "corrected output."
        ),
    )
    return parser


@dataclass(frozen=True)
class DifferentiablePFConfig:
    train_steps: int = 3
    eval_steps: int = 1000
    lr: float = 0.003377
    optimizer: str = "adam"
    tol: float = 1e-6
    beta1: float = 0.979681
    beta2: float = 0.963442
    eps: float = 1e-8
    scheduler: str = "author_plateau"
    scheduler_factor: float = 0.547191
    scheduler_patience: int = 41
    scheduler_threshold: float = 0.067321
    scheduler_cooldown: int = 97
    vmin: float = 0.5
    vmax: float = 1.5
    grad_clip: float = 0.0
    first_order: bool = False

    def validate(self) -> None:
        if self.train_steps < 0 or self.eval_steps < 0:
            raise ValueError("KOL step counts must be non-negative")
        if self.lr <= 0.0:
            raise ValueError("--kol_lr must be positive")
        if self.optimizer not in ("adam", "gd"):
            raise ValueError(f"Unsupported KOL optimizer: {self.optimizer!r}")
        if self.tol <= 0.0:
            raise ValueError("--kol_tol must be positive")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("KOL Adam beta values must lie in [0, 1)")
        if self.eps <= 0.0:
            raise ValueError("--kol_eps must be positive")
        if self.scheduler not in ("author_plateau", "off"):
            raise ValueError(f"Unsupported KOL scheduler: {self.scheduler!r}")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("KOL scheduler factor must lie in (0, 1)")
        if self.scheduler_patience < 0 or self.scheduler_cooldown < 0:
            raise ValueError("KOL scheduler patience/cooldown cannot be negative")
        if not 0.0 <= self.scheduler_threshold < 1.0:
            raise ValueError("KOL scheduler threshold must lie in [0, 1)")
        if self.vmin <= 0.0 or self.vmin >= self.vmax:
            raise ValueError("KOL voltage guards require 0 < vmin < vmax")
        if self.grad_clip < 0.0:
            raise ValueError("--kol_grad_clip cannot be negative")


def config_from_args(args) -> DifferentiablePFConfig:
    cfg = DifferentiablePFConfig(
        train_steps=int(args.kol_train_steps),
        eval_steps=int(args.kol_eval_steps),
        lr=float(args.kol_lr),
        optimizer=str(args.kol_optimizer),
        tol=float(args.kol_tol),
        beta1=float(args.kol_beta1),
        beta2=float(args.kol_beta2),
        eps=float(args.kol_eps),
        scheduler=str(args.kol_scheduler),
        scheduler_factor=float(args.kol_scheduler_factor),
        scheduler_patience=int(args.kol_scheduler_patience),
        scheduler_threshold=float(args.kol_scheduler_threshold),
        scheduler_cooldown=int(args.kol_scheduler_cooldown),
        vmin=float(args.kol_vmin),
        vmax=float(args.kol_vmax),
        grad_clip=float(args.kol_grad_clip),
        first_order=bool(args.kol_first_order),
    )
    cfg.validate()
    return cfg


def build_known_operator(args) -> Optional["DifferentiablePowerFlow"]:
    if getattr(args, "kol_pf_mode", "off") == "off":
        return None
    if args.kol_pf_mode != "dpf":
        raise ValueError(f"Unknown --kol_pf_mode={args.kol_pf_mode!r}")
    return DifferentiablePowerFlow(config_from_args(args))


def known_operator_tag(args) -> str:
    if getattr(args, "kol_pf_mode", "off") == "off":
        return "kol-off"
    return (
        f"kol-dpf-tr{args.kol_train_steps}-ev{args.kol_eval_steps}"
        f"-lr{args.kol_lr:g}-tol{args.kol_tol:g}-{args.kol_optimizer}"
        f"-{args.kol_scheduler}"
    )


def _complex_and_real_dtype(Y: torch.Tensor) -> Tuple[torch.dtype, torch.dtype]:
    if Y.dtype == torch.complex128:
        return torch.complex128, torch.float64
    if Y.dtype == torch.complex64:
        return torch.complex64, torch.float32
    if Y.dtype == torch.float64:
        return torch.complex128, torch.float64
    return torch.complex64, torch.float32


def _validate_layout(
    Y: torch.Tensor,
    V: torch.Tensor,
    Sset: torch.Tensor,
    bus_type: torch.Tensor,
    sizes: Optional[torch.Tensor],
) -> None:
    if V.ndim != 3 or V.shape[-1] != 2:
        raise ValueError(f"V must have shape [B, N, 2], got {tuple(V.shape)}")
    if Sset.shape != V.shape[:-1] or bus_type.shape != V.shape[:-1]:
        raise ValueError(
            "Sset and bus_type must match V's [B, N] dimensions; got "
            f"V={tuple(V.shape)}, S={tuple(Sset.shape)}, bus_type={tuple(bus_type.shape)}"
        )
    if Y.shape[-2:] != (V.shape[-2], V.shape[-2]):
        raise ValueError(f"Y/V size mismatch: Y={tuple(Y.shape)}, V={tuple(V.shape)}")
    if sizes is not None:
        if V.shape[0] != 1:
            raise ValueError("sizes denotes a block-diagonal mega-graph and requires B=1")
        if int(sizes.sum().item()) != V.shape[1]:
            raise ValueError(
                f"sum(sizes)={int(sizes.sum().item())} != number of nodes {V.shape[1]}"
            )


def power_flow_mismatch(
    Y: torch.Tensor,
    V: torch.Tensor,
    Sset: torch.Tensor,
    bus_type: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return signed mismatch and the masks of equations specified by PF."""
    complex_dtype, real_dtype = _complex_and_real_dtype(Y)
    vm = V[..., 0].to(real_dtype)
    va = V[..., 1].to(real_dtype)
    Sset = Sset.to(device=Y.device, dtype=complex_dtype)
    Vc = vm.to(complex_dtype) * torch.exp(1j * va.to(complex_dtype))
    Scalc = Vc * ybus_matvec(Y, Vc).conj()

    slack = bus_type == 1
    pv = bus_type == 2
    p_mask = ~slack
    q_mask = ~(slack | pv)
    dP = Scalc.real - Sset.real
    dQ = Scalc.imag - Sset.imag
    return dP, dQ, p_mask, q_mask


def _graph_slices(V: torch.Tensor, sizes: Optional[torch.Tensor]):
    if sizes is None:
        return [(b, slice(0, V.shape[1])) for b in range(V.shape[0])]
    out = []
    offset = 0
    for size in sizes.detach().cpu().tolist():
        size = int(size)
        out.append((0, slice(offset, offset + size)))
        offset += size
    return out


def power_flow_residual_loss(
    Y: torch.Tensor,
    V: torch.Tensor,
    Sset: torch.Tensor,
    bus_type: torch.Tensor,
    *,
    sizes: Optional[torch.Tensor] = None,
    residual_norm: str = "none",
    norm_eps: float = 1e-12,
) -> torch.Tensor:
    """Mean of per-graph masked AC-PF squared residuals.

    With ``residual_norm='none'`` this is exactly the ``torch.nn.MSELoss``
    objective used by the reference DPF implementation, averaged per
    independent grid before the outer batch mean. ``local_ybus`` changes only
    the training residual coordinates, not the AC equations themselves.
    """
    if residual_norm not in ("none", "local_ybus"):
        raise ValueError(
            f"unknown residual_norm={residual_norm!r}; expected 'none' or "
            "'local_ybus'"
        )

    dP, dQ, p_mask, q_mask = power_flow_mismatch(Y, V, Sset, bus_type)
    if residual_norm == "local_ybus":
        # Diagonal Jacobi scaling: |Y_ii| |V_i|^2 is the local power scale
        # induced by a unit relative voltage perturbation.  Keep the exact Y
        # in the residual itself; only the residual handed to the loss is
        # scaled.  The sparse path avoids densifying large block-diagonal
        # batches.
        if Y.is_sparse:
            c = Y.coalesce()
            idx = c.indices()
            on_diag = idx[-2] == idx[-1]
            if Y.dim() == 2:
                diag = torch.zeros(
                    Y.shape[-1], dtype=c.values().dtype, device=Y.device
                )
                diag.index_add_(0, idx[-2, on_diag], c.values()[on_diag])
                diag_abs = diag.abs().unsqueeze(0)
            elif Y.dim() == 3:
                diag = torch.zeros(
                    Y.shape[0], Y.shape[-1], dtype=c.values().dtype,
                    device=Y.device,
                )
                diag.index_put_(
                    (idx[0, on_diag], idx[-2, on_diag]),
                    c.values()[on_diag],
                    accumulate=True,
                )
                diag_abs = diag.abs()
            else:
                raise ValueError(f"unsupported sparse Y rank {Y.dim()}")
        else:
            diag_abs = torch.diagonal(Y, dim1=-2, dim2=-1).abs()
            if diag_abs.dim() == 1:
                diag_abs = diag_abs.unsqueeze(0)
        vm = V[..., 0].to(diag_abs.dtype)
        scale = (diag_abs * vm.square()).clamp_min(float(norm_eps))
        dP = dP / scale
        dQ = dQ / scale
    graph_losses = []
    for b, sl in _graph_slices(V, sizes):
        parts = []
        pm = p_mask[b, sl]
        qm = q_mask[b, sl]
        if pm.any():
            parts.append(dP[b, sl][pm])
        if qm.any():
            parts.append(dQ[b, sl][qm])
        if parts:
            residual = torch.cat(parts)
            graph_losses.append(torch.mean(residual.square()))
    if not graph_losses:
        return V.sum() * 0.0
    return torch.stack(graph_losses).mean()


def adjoint_ybus_matvec(Y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute ``Y.conj().T @ x`` without densifying a sparse Y-bus."""
    if Y.is_sparse:
        coalesced = Y.coalesce()
        indices = coalesced.indices().clone()
        indices[[-2, -1]] = indices[[-1, -2]]
        shape = list(Y.shape)
        shape[-2], shape[-1] = shape[-1], shape[-2]
        Yh = torch.sparse_coo_tensor(
            indices,
            coalesced.values().conj(),
            tuple(shape),
            device=Y.device,
            dtype=Y.dtype,
        ).coalesce()
    else:
        Yh = Y.transpose(-2, -1).conj()
    return ybus_matvec(Yh, x)


def _graph_loss_scales(
    p_mask: torch.Tensor,
    q_mask: torch.Tensor,
    sizes: Optional[torch.Tensor],
) -> torch.Tensor:
    """Node-wise derivative scale for the reference per-graph MSE."""
    if sizes is None:
        n_graphs = p_mask.shape[0]
        n_equations = (p_mask.sum(dim=-1) + q_mask.sum(dim=-1)).clamp_min(1)
        return (2.0 / (n_graphs * n_equations)).to(dtype=torch.float64).view(-1, 1)

    scales = []
    offset = 0
    n_graphs = int(sizes.numel())
    for size in sizes.detach().cpu().tolist():
        size = int(size)
        sl = slice(offset, offset + size)
        n_equations = (p_mask[0, sl].sum() + q_mask[0, sl].sum()).clamp_min(1)
        scale = 2.0 / (n_graphs * n_equations)
        scales.append(scale.expand(size))
        offset += size
    return torch.cat(scales).view(1, -1).to(dtype=torch.float64)


def power_flow_loss_and_gradient(
    Y: torch.Tensor,
    V: torch.Tensor,
    Sset: torch.Tensor,
    bus_type: torch.Tensor,
    *,
    sizes: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the exact loss and its sparse-adjoint gradients in polar form.

    For ``S(V) = V * conj(YV)`` and a complex residual weight ``w``, the
    voltage-space adjoint is

        g_V = w * (YV) + Y^H (conj(w) * V).

    Projecting ``g_V`` onto ``dV/d|V|`` and ``dV/dtheta`` yields the two real
    update directions.  All operations remain in the ordinary autograd graph,
    so the unrolled optimizer is differentiable without differentiating
    through a nested ``torch.autograd.grad`` call or materializing ``J``.
    """
    complex_dtype, real_dtype = _complex_and_real_dtype(Y)
    vm = V[..., 0].to(real_dtype)
    va = V[..., 1].to(real_dtype)
    phase = torch.exp(1j * va.to(complex_dtype))
    Vc = vm.to(complex_dtype) * phase
    current = ybus_matvec(Y, Vc)
    Scalc = Vc * current.conj()
    Sset = Sset.to(device=Y.device, dtype=complex_dtype)

    slack = bus_type == 1
    pv = bus_type == 2
    p_mask = ~slack
    q_mask = ~(slack | pv)
    dP = Scalc.real - Sset.real
    dQ = Scalc.imag - Sset.imag

    graph_losses = []
    for b, sl in _graph_slices(V, sizes):
        parts = []
        pm = p_mask[b, sl]
        qm = q_mask[b, sl]
        if pm.any():
            parts.append(dP[b, sl][pm])
        if qm.any():
            parts.append(dQ[b, sl][qm])
        if parts:
            r = torch.cat(parts)
            graph_losses.append(r.square().mean())
    loss = torch.stack(graph_losses).mean() if graph_losses else V.sum() * 0.0

    scale = _graph_loss_scales(p_mask, q_mask, sizes).to(device=V.device, dtype=real_dtype)
    w = (dP * p_mask * scale).to(complex_dtype)
    w = w + 1j * (dQ * q_mask * scale).to(complex_dtype)
    g_complex = w * current + adjoint_ybus_matvec(Y, w.conj() * Vc)
    grad_vm = (g_complex.conj() * phase).real.to(real_dtype)
    grad_va = (g_complex.conj() * (1j * Vc)).real.to(real_dtype)
    return loss, grad_vm, grad_va


def power_flow_max_mismatch(
    Y: torch.Tensor,
    V: torch.Tensor,
    Sset: torch.Tensor,
    bus_type: torch.Tensor,
    *,
    sizes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-graph infinity norm of the bus-type-aware AC-PF mismatch."""
    dP, dQ, p_mask, q_mask = power_flow_mismatch(Y, V, Sset, bus_type)
    maxima = []
    for b, sl in _graph_slices(V, sizes):
        vals = []
        pm = p_mask[b, sl]
        qm = q_mask[b, sl]
        if pm.any():
            vals.append(dP[b, sl][pm].abs().amax())
        if qm.any():
            vals.append(dQ[b, sl][qm].abs().amax())
        maxima.append(torch.stack(vals).amax() if vals else dP.new_zeros(()))
    return torch.stack(maxima)


def _node_active_mask(
    active_graphs: torch.Tensor,
    V: torch.Tensor,
    sizes: Optional[torch.Tensor],
) -> torch.Tensor:
    if sizes is None:
        return active_graphs.view(-1, 1).expand(V.shape[0], V.shape[1])
    return torch.repeat_interleave(active_graphs, sizes.to(active_graphs.device)).view(1, -1)


class DifferentiablePowerFlow(nn.Module):
    """Fixed-step differentiable training layer and adaptive certified evaluator."""

    def __init__(self, config: DifferentiablePFConfig):
        super().__init__()
        config.validate()
        self.config = config

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"optimizer={c.optimizer}, train_steps={c.train_steps}, "
            f"eval_steps={c.eval_steps}, lr={c.lr:g}, tol={c.tol:g}, "
            f"scheduler={c.scheduler}"
        )

    def forward(
        self,
        V0: torch.Tensor,
        Y: torch.Tensor,
        Sset: torch.Tensor,
        bus_type: torch.Tensor,
        *,
        V_fixed: Optional[torch.Tensor] = None,
        sizes: Optional[torch.Tensor] = None,
        differentiable: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Correct ``V0`` and return the state plus certificate diagnostics.

        ``V_fixed`` supplies the specified slack voltage and PV magnitudes.  It
        should normally be the dataset's ``V_start`` rather than the surrogate
        output, otherwise a model could alter specified PF boundary values.
        """
        if differentiable is None:
            differentiable = bool(self.training)
        if V_fixed is None:
            V_fixed = V0
        _validate_layout(Y, V0, Sset, bus_type, sizes)
        _validate_layout(Y, V_fixed, Sset, bus_type, sizes)

        c = self.config
        steps = c.train_steps if differentiable else c.eval_steps
        _, real_dtype = _complex_and_real_dtype(Y)
        V0_work = V0.to(real_dtype)
        Vfixed_work = V_fixed.to(real_dtype)

        slack = bus_type == 1
        pv = bus_type == 2
        vm_free = ~(slack | pv)
        va_free = ~slack

        # Enforce specified PF boundary values before the first residual is
        # evaluated.  This is a hard architectural constraint, not a loss.
        vm = torch.where(vm_free, V0_work[..., 0], Vfixed_work[..., 0])
        va = torch.where(va_free, V0_work[..., 1], Vfixed_work[..., 1])
        vm = torch.where(vm_free, vm.clamp(c.vmin, c.vmax), vm)
        va = torch.where(va_free, torch.atan2(torch.sin(va), torch.cos(va)), va)

        if not differentiable:
            vm = vm.detach()
            va = va.detach()

        def state() -> torch.Tensor:
            return torch.stack((vm, va), dim=-1)

        with torch.enable_grad():
            initial_max = power_flow_max_mismatch(
                Y, state(), Sset, bus_type, sizes=sizes
            ).detach()
            n_graphs = int(initial_max.numel())
            iterations = torch.zeros(n_graphs, dtype=torch.long, device=V0.device)
            active = initial_max > c.tol

            m_vm = torch.zeros_like(vm)
            m_va = torch.zeros_like(va)
            v_vm = torch.zeros_like(vm)
            v_va = torch.zeros_like(va)
            current_lr = c.lr
            scheduler_best = float("inf")
            scheduler_bad_epochs = 0
            scheduler_cooldown = 0

            for step in range(1, steps + 1):
                if not differentiable:
                    current_max = power_flow_max_mismatch(
                        Y, state(), Sset, bus_type, sizes=sizes
                    ).detach()
                    active = current_max > c.tol
                    if not bool(active.any()):
                        break

                current_loss, grad_vm, grad_va = power_flow_loss_and_gradient(
                    Y, state(), Sset, bus_type, sizes=sizes
                )
                if c.first_order:
                    grad_vm = grad_vm.detach()
                    grad_va = grad_va.detach()
                grad_vm = grad_vm.masked_fill(~vm_free, 0.0)
                grad_va = grad_va.masked_fill(~va_free, 0.0)
                if c.grad_clip > 0.0:
                    grad_vm = grad_vm.clamp(-c.grad_clip, c.grad_clip)
                    grad_va = grad_va.clamp(-c.grad_clip, c.grad_clip)

                node_active = _node_active_mask(active, V0_work, sizes)
                grad_vm = grad_vm.masked_fill(~node_active, 0.0)
                grad_va = grad_va.masked_fill(~node_active, 0.0)

                if c.optimizer == "adam":
                    m_vm = c.beta1 * m_vm + (1.0 - c.beta1) * grad_vm
                    m_va = c.beta1 * m_va + (1.0 - c.beta1) * grad_va
                    v_vm = c.beta2 * v_vm + (1.0 - c.beta2) * grad_vm.square()
                    v_va = c.beta2 * v_va + (1.0 - c.beta2) * grad_va.square()
                    bias1 = 1.0 - c.beta1 ** step
                    bias2 = 1.0 - c.beta2 ** step
                    delta_vm = current_lr * (m_vm / bias1) / (torch.sqrt(v_vm / bias2) + c.eps)
                    delta_va = current_lr * (m_va / bias1) / (torch.sqrt(v_va / bias2) + c.eps)
                else:
                    delta_vm = current_lr * grad_vm
                    delta_va = current_lr * grad_va

                # A graph that has met the residual certificate must remain
                # frozen. Masking only the current gradient is insufficient
                # for Adam because old first/second moments otherwise keep
                # moving the voltage after convergence.
                delta_vm = delta_vm.masked_fill(~node_active, 0.0)
                delta_va = delta_va.masked_fill(~node_active, 0.0)

                vm = torch.where(vm_free, (vm - delta_vm).clamp(c.vmin, c.vmax), Vfixed_work[..., 0])
                va_candidate = va - delta_va
                va_candidate = torch.atan2(torch.sin(va_candidate), torch.cos(va_candidate))
                va = torch.where(va_free, va_candidate, Vfixed_work[..., 1])
                iterations = iterations + active.to(torch.long)

                # Evaluation is an inner numerical solve, not an outer graph
                # to retain.  Detaching each step keeps its memory O(N).
                if not differentiable:
                    vm = vm.detach()
                    va = va.detach()

                    if c.scheduler == "author_plateau":
                        metric = float(current_loss.detach().item())
                        better = metric < scheduler_best * (1.0 - c.scheduler_threshold)
                        if better:
                            scheduler_best = metric
                            scheduler_bad_epochs = 0
                        else:
                            scheduler_bad_epochs += 1
                        if scheduler_cooldown > 0:
                            scheduler_cooldown -= 1
                            scheduler_bad_epochs = 0
                        if scheduler_bad_epochs > c.scheduler_patience:
                            next_lr = current_lr * c.scheduler_factor
                            if current_lr - next_lr > 1e-8:
                                current_lr = next_lr
                            scheduler_cooldown = c.scheduler_cooldown
                            scheduler_bad_epochs = 0

            output = state()
            final_max = power_flow_max_mismatch(
                Y, output, Sset, bus_type, sizes=sizes
            ).detach()

        if not differentiable:
            output = output.detach()
        diagnostics = {
            "initial_max_mismatch": initial_max,
            "final_max_mismatch": final_max,
            "converged": final_max <= c.tol,
            "iterations": iterations.detach(),
            "tolerance": final_max.new_full(final_max.shape, c.tol),
        }
        return output, diagnostics


def format_known_operator_diagnostics(diag: Optional[Dict[str, torch.Tensor]]) -> str:
    if diag is None or diag["final_max_mismatch"].numel() == 0:
        return "KOL off"
    initial = diag["initial_max_mismatch"].float()
    final = diag["final_max_mismatch"].float()
    converged = diag["converged"]
    iterations = diag["iterations"].float()
    return (
        f"KOL F_inf {initial.mean().item():.3e}->{final.mean().item():.3e}, "
        f"cert {int(converged.sum().item())}/{converged.numel()} "
        f"({converged.float().mean().item():.1%}), "
        f"steps {iterations.mean().item():.1f}"
    )
