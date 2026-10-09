"""GNS_heterogeneous with a state-space incremental decoder and an Armijo line search.

Motivation
----------
The released GridFM-PF model (``GNS_heterogeneous``) already runs an iterative
loop with physics feedback: at every layer it decodes an *absolute* voltage
state from the latent, computes the power-balance residual, and injects that
residual back into the latent,

    h_bus <- h_bus + physics_mlp(residual)          (latent-space update)

GENCO (arXiv:2608.09921) keeps exactly this shape -- its Eq. (17) is the same
update -- and adds an HGT backbone, discounted deep supervision, and task
decoders. In neither model is there a step in the space the residual is a
function of, so there is nothing for a line search to scale.

PIGNN-Attn-LS instead updates the *state*,

    V <- clamp(V + alpha * dV)                      (state-space update)

with alpha chosen by Armijo backtracking on the physics residual. This module
is the minimal change that gives the GridFM loop the same property: the
per-layer decoder output is read as an increment to a running state rather
than as a fresh absolute state, and the increment is scaled by a line search
before it is applied.

What this is not
----------------
This is not GENCO. It is the GridFM-PF backbone with GENCO's feedback loop
(already present upstream) and a state-space line search added. It isolates
the line search, which is the variable of interest, on a backbone we have
already trained and measured on all 31 grids.

Copied from gridfm_graphkit v0.8.1
``gridfm_graphkit/models/gnn_heterogeneous_gns.py``; the forward is
reimplemented rather than wrapped because the loop body has to change.
"""

from __future__ import annotations

import torch
from torch_scatter import scatter_add

from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous
from gridfm_graphkit.datasets.globals import VM_H, VA_H, PG_H


class GNS_heterogeneous_LS(GNS_heterogeneous):
    """PF-only. The OPF/SE branches of the parent are deliberately not carried
    over: they decode different variables and would need their own line-search
    objective, and the comparison this exists for is power flow."""

    def __init__(self, args, armijo_max_backtracks: int = 6,
                 armijo_rho: float = 0.5, armijo_c1: float = 1e-4,
                 line_search: bool = True) -> None:
        super().__init__(args)
        if self.task != "PowerFlow":
            raise ValueError(
                f"GNS_heterogeneous_LS is power-flow only, got task={self.task!r}")
        # 6, not 4: with rho=0.5 four backtracks bottom out at alpha=0.125, and
        # a probe of a smoke-trained checkpoint found alpha pinned to the two
        # smallest values with 35% of graphs rejected outright and alpha=1 never
        # accepted -- the signature of a floor that is too high rather than of
        # no acceptable step. Six reaches 0.03125. The extra evaluations only
        # run for graphs that have not yet accepted.
        self.armijo_max_backtracks = int(armijo_max_backtracks)
        self.armijo_rho = float(armijo_rho)
        self.armijo_c1 = float(armijo_c1)
        # line_search=False keeps the incremental decoder but always takes a
        # full step. It is the ablation that separates "increment" from
        # "line-searched increment"; without it a win could be either.
        self.line_search = bool(line_search)
        self.alpha_history = []

    # ------------------------------------------------------------------ #
    # residual objective
    # ------------------------------------------------------------------ #
    def _residuals(self, bus_state, x_bus, bus_edge_index, bus_edge_attr,
                   gen_temp, gen_to_bus_index, num_bus, mask_dict):
        """Per-bus (residual_P, residual_Q) at a candidate state.

        Same call chain the parent uses, factored out so a candidate step can
        be evaluated without duplicating it.
        """
        Pft, Qft = self.branch_flow_layer(bus_state, bus_edge_index, bus_edge_attr)
        P_in, Q_in = self.node_injection_layer(Pft, Qft, bus_edge_index, num_bus)
        agg_bus = scatter_add(gen_temp.squeeze(-1), gen_to_bus_index,
                              dim=0, dim_size=num_bus)
        out = self.physics_decoder(P_in, Q_in, bus_state, x_bus, agg_bus, mask_dict)
        rP, rQ = self.node_residuals_layer(P_in, Q_in, out, x_bus)
        return rP, rQ, out

    @staticmethod
    def _per_graph_cost(rP, rQ, bus_batch, n_graphs):
        """Sum of squared residuals per graph.

        Per graph, not per bus: a line search picks one step length for a
        direction, and a per-bus alpha would change the direction itself
        rather than scale it.
        """
        sq = rP.pow(2) + rQ.pow(2)
        return scatter_add(sq, bus_batch, dim=0, dim_size=n_graphs)

    # ------------------------------------------------------------------ #
    def forward(self, batch):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict
        mask_dict = batch.mask_dict

        self.layer_residuals = {}
        self.alpha_history = []

        h_bus = self.input_proj_bus(x_dict["bus"])
        h_gen = self.input_proj_gen(x_dict["gen"])

        num_bus = x_dict["bus"].size(0)
        _, gen_to_bus_index = edge_index_dict[("gen", "connected_to", "bus")]
        bus_edge_index = edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr = edge_attr_dict[("bus", "connects", "bus")]

        edge_attr_proj_dict = {
            k: (self.input_proj_edge(v) if v is not None else None)
            for k, v in edge_attr_dict.items()
        }

        bus_mask = mask_dict["bus"][:, VM_H:VA_H + 1]
        gen_mask = mask_dict["gen"][:, :(PG_H + 1)]
        bus_fixed = x_dict["bus"][:, VM_H:VA_H + 1]
        gen_fixed = x_dict["gen"][:, :(PG_H + 1)]

        # Graph assignment for the per-graph step length. A single unbatched
        # graph has no `batch` attribute, so fall back to one group.
        bus_batch = getattr(batch["bus"], "batch", None)
        if bus_batch is None:
            bus_batch = torch.zeros(num_bus, dtype=torch.long,
                                    device=x_dict["bus"].device)
        n_graphs = int(bus_batch.max().item()) + 1

        # The running state starts at the values the parquet carries -- the DC
        # initialisation -- with known entries already in place. The parent
        # never keeps a state across layers; this is the whole change.
        bus_state = torch.where(bus_mask, bus_fixed, bus_fixed)
        gen_state = gen_fixed

        for i, conv in enumerate(self.layers):
            out_dict = conv({"bus": h_bus, "gen": h_gen},
                            edge_index_dict, edge_attr_proj_dict)
            out_bus = self.activation(self.norms_bus[i](out_dict["bus"]))
            out_gen = self.activation(self.norms_gen[i](out_dict["gen"]))
            h_bus = h_bus + out_bus if out_bus.shape == h_bus.shape else out_bus
            h_gen = h_gen + out_gen if out_gen.shape == h_gen.shape else out_gen

            # The decoder output is an INCREMENT now, not an absolute state.
            d_bus = self.mlp_bus(h_bus)
            d_gen = self.mlp_gen(h_gen)

            # Never move a quantity that was given: at PV/slack the input is
            # the answer, so the increment is masked to zero there.
            d_bus = torch.where(bus_mask, d_bus, torch.zeros_like(d_bus))
            d_gen = torch.where(gen_mask, d_gen, torch.zeros_like(d_gen))

            gen_cand = gen_state + d_gen

            if self.line_search:
                with torch.no_grad():
                    rP0, rQ0, _ = self._residuals(
                        bus_state, x_dict["bus"], bus_edge_index, bus_edge_attr,
                        gen_cand, gen_to_bus_index, num_bus, mask_dict)
                    f0 = self._per_graph_cost(rP0, rQ0, bus_batch, n_graphs)
                    # Armijo reference slope. The direction is not guaranteed to
                    # be a descent direction here -- it comes from a learned
                    # head, not from a gradient -- so the sufficient-decrease
                    # test is used as an acceptance filter and alpha=0 is the
                    # fallback. That yields a monotone non-increase of the
                    # residual, which is the property we actually want; it is
                    # not the convergence guarantee textbook Armijo carries.
                    slope = self._per_graph_cost(
                        d_bus[:, 0].pow(2), d_bus[:, 1].pow(2), bus_batch, n_graphs)

                    alpha = torch.zeros(n_graphs, device=f0.device, dtype=f0.dtype)
                    remaining = torch.ones(n_graphs, dtype=torch.bool, device=f0.device)
                    a = 1.0
                    for _ in range(self.armijo_max_backtracks):
                        a_node = torch.full_like(bus_state[:, :1], a)
                        cand = bus_state + a_node * d_bus
                        cand = torch.where(bus_mask, cand, bus_fixed)
                        rP, rQ, _ = self._residuals(
                            cand, x_dict["bus"], bus_edge_index, bus_edge_attr,
                            gen_cand, gen_to_bus_index, num_bus, mask_dict)
                        f = self._per_graph_cost(rP, rQ, bus_batch, n_graphs)
                        ok = remaining & (f <= f0 - self.armijo_c1 * a * slope)
                        alpha = torch.where(ok, torch.full_like(alpha, a), alpha)
                        remaining = remaining & ~ok
                        if not bool(remaining.any()):
                            break
                        a *= self.armijo_rho
                    self.alpha_history.append(alpha.detach())
                alpha_node = alpha[bus_batch].unsqueeze(-1)
            else:
                alpha_node = torch.ones_like(bus_state[:, :1])

            # The step itself stays in the graph: alpha is a detached scalar,
            # d_bus is not, so gradients flow through the increment.
            bus_state = bus_state + alpha_node * d_bus
            bus_state = torch.where(bus_mask, bus_state, bus_fixed)
            gen_state = gen_cand

            rP, rQ, output_temp = self._residuals(
                bus_state, x_dict["bus"], bus_edge_index, bus_edge_attr,
                gen_state, gen_to_bus_index, num_bus, mask_dict)

            bus_residuals = torch.stack([rP, rQ], dim=-1)
            self.layer_residuals[i] = torch.linalg.norm(bus_residuals, dim=-1).mean()
            h_bus = h_bus + self.physics_mlp(bus_residuals)

        return {"bus": output_temp, "gen": gen_state}
