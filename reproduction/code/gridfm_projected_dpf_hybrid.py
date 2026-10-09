"""Projected DPF hybrid built inside the released GridFM-GraphKit processor.

Each GraphKit layer proposes a state-space increment.  The proposal is fused
with an exact differentiable AC-mismatch/Adam direction, projected per packed
grid onto a strict descent half-space, and accepted by an exact per-grid Armijo
test.  This differs from ``known_operator_pf``: there is no surrogate followed
by a separate solver; the known operator is the state transition of every GNN
layer.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch_scatter import scatter_add, scatter_min

from collate_blockdiag_optimized_complex_columns import ybus_matvec
from gridfm_graphkit.datasets.globals import VM_H, VA_H, PG_H
from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous


def _graph_index_from_sizes(sizes: torch.Tensor, device: torch.device) -> torch.Tensor:
    sizes = sizes.to(device=device, dtype=torch.long).reshape(-1)
    return torch.repeat_interleave(torch.arange(sizes.numel(), device=device), sizes)


def exact_pf_mismatch(
    Y: torch.Tensor,
    state: torch.Tensor,
    Sset: torch.Tensor,
    bus_type: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact bus-type-aware AC mismatch on the parquet's per-unit tensors."""
    complex_dtype = Y.dtype if Y.is_complex() else (
        torch.complex128 if state.dtype == torch.float64 else torch.complex64
    )
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    vm = state[:, 0].to(real_dtype)
    va = state[:, 1].to(real_dtype)
    Vc = vm.to(complex_dtype) * torch.exp(1j * va.to(complex_dtype))
    Scalc = Vc * ybus_matvec(Y, Vc).conj()
    Sset = Sset.reshape(-1).to(device=state.device, dtype=complex_dtype)
    bus_type = bus_type.reshape(-1)
    p_mask = bus_type != 1
    q_mask = (bus_type != 1) & (bus_type != 2)
    return Scalc.real - Sset.real, Scalc.imag - Sset.imag, p_mask, q_mask


def per_graph_pf_cost(
    dP: torch.Tensor,
    dQ: torch.Tensor,
    p_mask: torch.Tensor,
    q_mask: torch.Tensor,
    graph_index: torch.Tensor,
    n_graphs: int,
) -> torch.Tensor:
    """Mean squared specified-equation residual, independently per graph."""
    values = dP.square() * p_mask.to(dP.dtype) + dQ.square() * q_mask.to(dQ.dtype)
    counts = scatter_add(
        p_mask.to(dP.dtype) + q_mask.to(dP.dtype),
        graph_index,
        dim=0,
        dim_size=n_graphs,
    ).clamp_min(1.0)
    return scatter_add(values, graph_index, dim=0, dim_size=n_graphs) / counts


def project_to_descent_halfspace(
    proposal: torch.Tensor,
    gradient: torch.Tensor,
    graph_index: torch.Tensor,
    n_graphs: int,
    mu: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Euclidean projection onto g^T d <= -mu ||g||^2, per packed graph."""
    dot = scatter_add(
        (proposal * gradient).sum(dim=-1), graph_index, dim=0, dim_size=n_graphs
    )
    norm2 = scatter_add(
        gradient.square().sum(dim=-1), graph_index, dim=0, dim_size=n_graphs
    )
    coefficient = torch.relu(dot + float(mu) * norm2) / norm2.clamp_min(eps)
    return proposal - coefficient[graph_index, None] * gradient


def scale_to_voltage_bounds(
    state: torch.Tensor,
    direction: torch.Tensor,
    free_vm: torch.Tensor,
    graph_index: torch.Tensor,
    n_graphs: int,
    vmin: float,
    vmax: float,
) -> torch.Tensor:
    """One positive scale per graph keeps every free voltage magnitude feasible."""
    vm = state[:, 0]
    dvm = direction[:, 0]
    one = torch.ones_like(vm)
    ratio = torch.where(
        free_vm & (dvm > 0),
        (float(vmax) - vm) / dvm.clamp_min(torch.finfo(dvm.dtype).eps),
        one,
    )
    ratio = torch.where(
        free_vm & (dvm < 0),
        (float(vmin) - vm) / dvm.clamp_max(-torch.finfo(dvm.dtype).eps),
        ratio,
    )
    ratio = ratio.clamp(min=0.0, max=1.0)
    graph_scale = scatter_min(ratio, graph_index, dim=0, dim_size=n_graphs)[0]
    return direction * graph_scale[graph_index, None]


def scale_to_local_step_limits(
    state: torch.Tensor,
    direction: torch.Tensor,
    graph_index: torch.Tensor,
    n_graphs: int,
    max_vm_fraction: float,
    max_angle_step: float,
) -> torch.Tensor:
    """Preserve descent while limiting the nonlinear trial displacement."""
    eps = torch.finfo(direction.dtype).eps
    vm_limit = float(max_vm_fraction) * state[:, 0].abs().clamp_min(eps)
    vm_ratio = torch.where(
        direction[:, 0].abs() > vm_limit,
        vm_limit / direction[:, 0].abs().clamp_min(eps),
        torch.ones_like(vm_limit),
    )
    va_ratio = torch.where(
        direction[:, 1].abs() > float(max_angle_step),
        float(max_angle_step) / direction[:, 1].abs().clamp_min(eps),
        torch.ones_like(vm_limit),
    )
    node_ratio = torch.minimum(vm_ratio, va_ratio).clamp(max=1.0)
    graph_scale = scatter_min(node_ratio, graph_index, dim=0, dim_size=n_graphs)[0]
    return direction * graph_scale[graph_index, None]


class GNS_heterogeneous_ProjectedDPF(GNS_heterogeneous):
    """Released GraphKit PF backbone with a projected DPF state transition."""

    def __init__(
        self,
        args,
        *,
        dpf_lr: float = 0.003377,
        beta1: float = 0.979681,
        beta2: float = 0.963442,
        adam_eps: float = 1e-8,
        descent_mu: float = 1e-4,
        armijo_c1: float = 1e-4,
        armijo_rho: float = 0.5,
        armijo_max_backtracks: int = 8,
        vmin: float = 0.5,
        vmax: float = 1.5,
        max_vm_fraction: float = 0.10,
        max_angle_step: float = 0.30,
        first_order: bool = False,
    ) -> None:
        super().__init__(args)
        if self.task != "PowerFlow":
            raise ValueError("The projected GraphKit-DPF hybrid is power-flow only")
        if dpf_lr <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Invalid projected-DPF Adam hyperparameters")
        if not 0 < armijo_rho < 1 or armijo_max_backtracks < 1:
            raise ValueError("Invalid projected-DPF Armijo configuration")
        self.dpf_lr = float(dpf_lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.adam_eps = float(adam_eps)
        self.descent_mu = float(descent_mu)
        self.armijo_c1 = float(armijo_c1)
        self.armijo_rho = float(armijo_rho)
        self.armijo_max_backtracks = int(armijo_max_backtracks)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.max_vm_fraction = float(max_vm_fraction)
        self.max_angle_step = float(max_angle_step)
        self.first_order = bool(first_order)
        self.alpha_history = []
        self.cost_history = []
        self.initial_cost = None

    def _cost_and_residual(
        self, Y, state, Sset, bus_type, graph_index, n_graphs
    ):
        dP, dQ, p_mask, q_mask = exact_pf_mismatch(Y, state, Sset, bus_type)
        cost = per_graph_pf_cost(dP, dQ, p_mask, q_mask, graph_index, n_graphs)
        residual = torch.stack(
            [dP * p_mask.to(dP.dtype), dQ * q_mask.to(dQ.dtype)], dim=-1
        )
        return cost, residual

    def _feasible_state(self, candidate, bus_mask, bus_fixed):
        """Use one identical normalization path for Armijo trials and commits."""
        candidate = torch.stack(
            [candidate[:, 0].clamp(self.vmin, self.vmax),
             torch.atan2(torch.sin(candidate[:, 1]), torch.cos(candidate[:, 1]))],
            dim=-1,
        )
        # Re-pin after normalization so fixed PF quantities are bit-identical
        # to the inputs as well as mathematically equivalent.
        return torch.where(bus_mask, candidate, bus_fixed)

    def forward(self, batch):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict
        mask_dict = batch.mask_dict

        try:
            Y = batch.kol_Ybus
            Sset = batch.kol_Sset
            bus_type = batch.kol_bus_type
            sizes = batch.kol_sizes
        except AttributeError as exc:
            raise ValueError(
                "Projected GraphKit-DPF requires exact kol_Ybus/kol_Sset/"
                "kol_bus_type/kol_sizes metadata from the parquet adapter"
            ) from exc

        num_bus = x_dict["bus"].shape[0]
        graph_index = _graph_index_from_sizes(sizes, x_dict["bus"].device)
        if graph_index.numel() != num_bus:
            raise ValueError("Projected-DPF sizes do not match the number of bus nodes")
        n_graphs = int(sizes.numel())

        h_bus = self.input_proj_bus(x_dict["bus"])
        h_gen = self.input_proj_gen(x_dict["gen"])
        edge_attr_proj = {
            k: self.input_proj_edge(v) if v is not None else None
            for k, v in edge_attr_dict.items()
        }
        bus_mask = mask_dict["bus"][:, VM_H:VA_H + 1]
        bus_fixed = x_dict["bus"][:, VM_H:VA_H + 1]
        gen_mask = mask_dict["gen"][:, :(PG_H + 1)]
        gen_fixed = x_dict["gen"][:, :(PG_H + 1)]

        state = bus_fixed
        gen_state = gen_fixed
        adam_m = torch.zeros_like(state)
        adam_v = torch.zeros_like(state)
        self.alpha_history = []
        self.cost_history = []
        self.initial_cost = None
        self.layer_residuals = {}

        for layer_idx, conv in enumerate(self.layers):
            out = conv({"bus": h_bus, "gen": h_gen}, edge_index_dict, edge_attr_proj)
            out_bus = self.activation(self.norms_bus[layer_idx](out["bus"]))
            out_gen = self.activation(self.norms_gen[layer_idx](out["gen"]))
            h_bus = h_bus + out_bus if h_bus.shape == out_bus.shape else out_bus
            h_gen = h_gen + out_gen if h_gen.shape == out_gen.shape else out_gen

            learned = torch.where(bus_mask, self.mlp_bus(h_bus), torch.zeros_like(state))
            gen_delta = torch.where(gen_mask, self.mlp_gen(h_gen), torch.zeros_like(gen_state))
            gen_state = gen_state + gen_delta

            # Validation is normally under no_grad; the known operator still
            # needs the exact state gradient. Training retains second-order
            # derivatives unless the explicit memory-saving ablation is set.
            with torch.enable_grad():
                if not state.requires_grad:
                    state = state.detach().requires_grad_(True)
                cost0, _ = self._cost_and_residual(
                    Y, state, Sset, bus_type, graph_index, n_graphs
                )
                if layer_idx == 0:
                    self.initial_cost = cost0.detach()
                gradient = torch.autograd.grad(
                    cost0.sum(),
                    state,
                    create_graph=self.training and not self.first_order,
                    retain_graph=True,
                )[0]
            gradient = torch.where(bus_mask, gradient, torch.zeros_like(gradient))
            if self.first_order:
                gradient = gradient.detach()

            adam_m = self.beta1 * adam_m + (1.0 - self.beta1) * gradient
            adam_v = self.beta2 * adam_v + (1.0 - self.beta2) * gradient.square()
            t = layer_idx + 1
            m_hat = adam_m / (1.0 - self.beta1 ** t)
            v_hat = adam_v / (1.0 - self.beta2 ** t)
            dpf = -self.dpf_lr * m_hat / (v_hat.sqrt() + self.adam_eps)

            direction = project_to_descent_halfspace(
                learned + dpf, gradient, graph_index, n_graphs, self.descent_mu
            )
            direction = torch.where(bus_mask, direction, torch.zeros_like(direction))
            direction = scale_to_local_step_limits(
                state, direction, graph_index, n_graphs,
                self.max_vm_fraction, self.max_angle_step,
            )
            direction = scale_to_voltage_bounds(
                state, direction, bus_mask[:, 0], graph_index, n_graphs,
                self.vmin, self.vmax,
            )
            slope = scatter_add(
                (gradient * direction).sum(dim=-1),
                graph_index,
                dim=0,
                dim_size=n_graphs,
            )

            # Step selection is deliberately detached; the accepted update
            # remains differentiable through both the GNN and DPF direction.
            with torch.no_grad():
                alpha = torch.zeros(n_graphs, device=state.device, dtype=state.dtype)
                remaining = torch.ones(n_graphs, device=state.device, dtype=torch.bool)
                trial_alpha = 1.0
                for _ in range(self.armijo_max_backtracks):
                    candidate = self._feasible_state(
                        state + trial_alpha * direction, bus_mask, bus_fixed
                    )
                    trial_cost, _ = self._cost_and_residual(
                        Y, candidate, Sset, bus_type, graph_index, n_graphs
                    )
                    accepted = remaining & (
                        trial_cost <= cost0.detach() + self.armijo_c1 * trial_alpha * slope.detach()
                    )
                    alpha = torch.where(
                        accepted, torch.full_like(alpha, trial_alpha), alpha
                    )
                    remaining &= ~accepted
                    if not bool(remaining.any()):
                        break
                    trial_alpha *= self.armijo_rho

            state = state + alpha[graph_index, None] * direction
            state = self._feasible_state(state, bus_mask, bus_fixed)
            cost, residual = self._cost_and_residual(
                Y, state, Sset, bus_type, graph_index, n_graphs
            )
            self.alpha_history.append(alpha.detach())
            self.cost_history.append(cost.detach())
            self.layer_residuals[layer_idx] = cost.mean()
            h_bus = h_bus + self.physics_mlp(residual.to(h_bus.dtype))

        # PF training/evaluation consumes Vm/Va. Pg/Qg remain derived quantities
        # from the exact balance and are intentionally not part of this state.
        return {"bus": state, "gen": gen_state}
