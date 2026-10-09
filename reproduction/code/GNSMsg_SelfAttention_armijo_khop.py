import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from GNSMsg_SelfAttention_armijo import (
    EdgeSelfAttnBlock,
    _batched_mismatch_inf_norm,
    _build_dense_Y_from_branchrows_single,
    _build_directed_edges_single,
)


class GNSMsg_EdgeSelfAttnKHop(nn.Module):
    """
    Edge-aware self-attention solver with K-hop Gaussian diffusion features.

    The original attention model consumes local state [|V|, theta, dP, dQ, memory].
    This variant additionally computes D_K X, where D_K is a Gaussian-weighted
    multi-hop diffusion operator built from an electrical transition matrix.
    """

    def __init__(
        self,
        d: int = 10,
        d_hi: int = 32,
        K: int = 30,
        pinn: bool = True,
        gamma: float = 0.9,
        v_limit: bool = True,
        use_armijo: bool = True,
        d_model: int = None,
        n_heads: int = 4,
        num_attn_layers: int = 1,
        attn_dropout: float = 0.0,
        armijo_mode: str = "fixed",
        armijo_rho: float = 0.5,
        armijo_c1: float = 1e-4,
        armijo_max_backtracks: int = 5,
        armijo_min_alpha: float = 0.0625,
        dtheta_max: float = 0.30,
        dvm_frac: float = 0.10,
        physics_loss_form: str = "mse",
        physics_residual_norm: str = "none",
        physics_norm_eps: float = 1e-6,
        physics_huber_delta: float = 1.0,
        physics_final_weight: float = 0.0,
        khop_K: int = 3,
        khop_sigma: float = 1.5,
        khop_norm: str = "row",
        khop_source: str = "yabs",
    ):
        super().__init__()
        self.K = K
        self.d = d
        self.d_hi = d_hi
        self.pinn = pinn
        self.gamma = gamma
        self.v_limit = v_limit
        self.use_armijo = use_armijo
        self.armijo_mode = armijo_mode
        self.armijo_rho = armijo_rho
        self.armijo_c1 = armijo_c1
        self.armijo_max_backtracks = armijo_max_backtracks
        self.armijo_min_alpha = armijo_min_alpha
        self.dtheta_max = dtheta_max
        self.dvm_frac = dvm_frac
        self.physics_loss_form = physics_loss_form
        self.physics_residual_norm = physics_residual_norm
        self.physics_norm_eps = physics_norm_eps
        self.physics_huber_delta = physics_huber_delta
        self.physics_final_weight = physics_final_weight
        self.khop_K = khop_K
        self.khop_sigma = khop_sigma
        self.khop_norm = khop_norm
        self.khop_source = khop_source

        self.d_model = d_model if d_model is not None else d_hi
        self.n_heads = n_heads
        assert self.d_model % self.n_heads == 0
        self.num_attn_layers = num_attn_layers

        self.node_state_dim = 4 + d
        self.edge_feat_dim = 9
        self.in_proj = nn.Linear(2 * self.node_state_dim, self.d_model)
        self.blocks = nn.ModuleList([
            EdgeSelfAttnBlock(
                self.d_model,
                self.n_heads,
                self.edge_feat_dim,
                ffn_hidden=4 * self.d_model,
                dropout=attn_dropout,
            )
            for _ in range(self.num_attn_layers)
        ])

        self.theta_head = nn.ModuleList([nn.Linear(self.d_model, 1) for _ in range(K)])
        self.v_head = nn.ModuleList([nn.Linear(self.d_model, 1) for _ in range(K)])
        self.m_head = nn.ModuleList([nn.Linear(self.d_model, d) for _ in range(K)])

        for k in range(K):
            nn.init.zeros_(self.theta_head[k].weight)
            nn.init.zeros_(self.theta_head[k].bias)
            nn.init.zeros_(self.v_head[k].weight)
            nn.init.zeros_(self.v_head[k].bias)
            nn.init.zeros_(self.m_head[k].weight)
            nn.init.zeros_(self.m_head[k].bias)

    def _transition_from_y(self, Y):
        A = Y.abs()
        eye = torch.eye(A.size(-1), device=A.device, dtype=torch.bool).unsqueeze(0)
        A = A.masked_fill(eye, 0.0)

        if self.khop_source == "adj":
            A = (A > 0).to(A.dtype)
        elif self.khop_source != "yabs":
            raise ValueError(f"Unknown khop_source={self.khop_source!r}; expected 'yabs' or 'adj'.")

        eps = 1e-12
        if self.khop_norm == "row":
            return A / A.sum(dim=-1, keepdim=True).clamp_min(eps)
        if self.khop_norm == "sym":
            deg = A.sum(dim=-1).clamp_min(eps)
            dinv_sqrt = deg.rsqrt()
            return dinv_sqrt.unsqueeze(-1) * A * dinv_sqrt.unsqueeze(-2)
        if self.khop_norm == "col":
            return A / A.sum(dim=-2, keepdim=True).clamp_min(eps)

        raise ValueError(f"Unknown khop_norm={self.khop_norm!r}; expected 'row', 'sym', or 'col'.")

    def _khop_diffuse(self, Y, X):
        if self.khop_K <= 0:
            return torch.zeros_like(X)

        T = self._transition_from_y(Y).to(dtype=X.dtype)
        H = X
        acc = torch.zeros_like(X)
        z = X.new_zeros(())
        sigma = max(float(self.khop_sigma), 1e-12)

        for hop in range(1, int(self.khop_K) + 1):
            H = torch.matmul(T, H)
            w = math.exp(-(hop ** 2) / (2.0 * sigma ** 2))
            acc = acc + w * H
            z = z + X.new_tensor(w)

        return acc / z.clamp_min(1e-12)

    def _graph_scale_from_s(self, S_abs, n_nodes_per_graph):
        if n_nodes_per_graph is None:
            return S_abs.amax(dim=-1, keepdim=True).clamp_min(self.physics_norm_eps)

        scales = torch.empty_like(S_abs)
        offset = 0
        for size in n_nodes_per_graph.tolist():
            size = int(size)
            sl = slice(offset, offset + size)
            scale = S_abs[:, sl].amax(dim=-1, keepdim=True).clamp_min(self.physics_norm_eps)
            scales[:, sl] = scale
            offset += size
        return scales

    def _physics_residual_loss(self, DP, DQ, P_set, Q_set, p_mask, q_mask, n_nodes_per_graph):
        if self.physics_residual_norm == "none":
            if self.physics_loss_form == "mse":
                return (DP ** 2 + DQ ** 2).mean()
            residual = torch.cat([DP.reshape(-1), DQ.reshape(-1)])
        else:
            # Scale on enforced setpoints only -- Q at slack/PV buses is not a
            # constraint, so an arbitrary value parked there (LVN heo1 carries
            # Q_set = -2507 pu at one 380 kV PV bus) would otherwise divide the
            # residual at every node once "graph" takes amax over the grid.
            S_abs = torch.sqrt(
                (P_set * p_mask.to(P_set.dtype)) ** 2
                + (Q_set * q_mask.to(Q_set.dtype)) ** 2
            )
            if self.physics_residual_norm == "setpoint":
                scale = S_abs.clamp_min(self.physics_norm_eps)
            elif self.physics_residual_norm == "graph":
                scale = self._graph_scale_from_s(S_abs, n_nodes_per_graph)
            else:
                raise ValueError(
                    f"Unknown physics_residual_norm={self.physics_residual_norm!r}; "
                    "expected 'none', 'setpoint', or 'graph'."
                )

            residual = torch.cat([(DP / scale)[p_mask], (DQ / scale)[q_mask]])
            if residual.numel() == 0:
                return DP.sum() * 0.0

        if self.physics_loss_form == "mse":
            return (residual ** 2).mean()
        if self.physics_loss_form == "huber":
            return F.huber_loss(
                residual,
                torch.zeros_like(residual),
                delta=self.physics_huber_delta,
                reduction="mean",
            )
        if self.physics_loss_form == "logcosh":
            return (residual + F.softplus(-2.0 * residual) - math.log(2.0)).mean()

        raise ValueError(
            f"Unknown physics_loss_form={self.physics_loss_form!r}; "
            "expected 'mse', 'huber', or 'logcosh'."
        )

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
    ):
        device = bus_type.device
        B, N = bus_type.shape

        if Y is None:
            if Y_shunt_bus is None:
                raise ValueError("Y is None and Y_shunt_bus is also None; cannot reconstruct Y.")
            if B == 1:
                Y = _build_dense_Y_from_branchrows_single(
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
            else:
                Ys = []
                for b in range(B):
                    Ys.append(_build_dense_Y_from_branchrows_single(
                        N,
                        Branch_f_bus[b], Branch_t_bus[b], Branch_status[b],
                        Branch_tau[b], Branch_shift_deg[b],
                        Branch_y_series_from[b], Branch_y_series_to[b], Branch_y_series_ft[b],
                        Branch_y_shunt_from[b], Branch_y_shunt_to[b],
                        Y_shunt_bus[b],
                    ))
                Y = torch.stack(Ys, dim=0)
        elif Y.dim() == 2:
            Y = Y.unsqueeze(0)

        if B == 1:
            edge_index_dir, edge_feat_dir = _build_directed_edges_single(
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
                Is_trafo.squeeze(0),
            )
            edge_index_dir_list = None
            edge_feat_dir_list = None
        else:
            edge_index_dir = None
            edge_feat_dir = None
            edge_index_dir_list = []
            edge_feat_dir_list = []
            for b in range(B):
                ei, ef = _build_directed_edges_single(
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
                    Is_trafo[b],
                )
                edge_index_dir_list.append(ei)
                edge_feat_dir_list.append(ef)

        P_set, Q_set = S.real, S.imag
        v = V0[..., 0].clone()
        th = V0[..., 1].clone()
        m = torch.zeros(B, N, self.d, device=device, dtype=V0.dtype)

        slack_mask = (bus_type == 1)
        pv_mask = (bus_type == 2)
        p_mask = ~slack_mask
        q_mask = ~(slack_mask | pv_mask)

        phys_terms = []

        for k in range(self.K):
            Vc = v * torch.exp(1j * th)
            Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
            Sc = Vc * Ic.conj()

            DP = (P_set - Sc.real).masked_fill(slack_mask, 0.0)
            DQ = (Q_set - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)

            bus_feat = torch.stack([v, th, DP, DQ], dim=-1)
            node_state = torch.cat([bus_feat, m], dim=-1)
            node_state_diff = self._khop_diffuse(Y, node_state)
            x = self.in_proj(torch.cat([node_state, node_state_diff], dim=-1))

            if B == 1:
                for blk in self.blocks:
                    x = blk(x, edge_index_dir, edge_feat_dir)
            else:
                x_new = x.clone()
                for b in range(B):
                    xb = x[b:b + 1]
                    ei = edge_index_dir_list[b]
                    ef = edge_feat_dir_list[b]
                    if ei.numel() == 0:
                        x_new[b:b + 1] = xb
                        continue
                    for blk in self.blocks:
                        xb = blk(xb, ei, ef)
                    x_new[b:b + 1] = xb
                x = x_new

            dth = self.theta_head[k](x).squeeze(-1)
            dv = self.v_head[k](x).squeeze(-1)
            dm = torch.tanh(self.m_head[k](x))
            dm = F.layer_norm(dm, dm.shape[-1:])

            dth = dth.masked_fill(slack_mask, 0.0)
            dv = dv.masked_fill(slack_mask | pv_mask, 0.0)

            if self.v_limit:
                v_abs = v.abs()
                dth = torch.clamp(dth, -self.dtheta_max, self.dtheta_max)
                dv = torch.clamp(dv, -self.dvm_frac * v_abs, self.dvm_frac * v_abs)

            if self.use_armijo:
                v_min, v_max = 0.75, 1.20
                with torch.no_grad():
                    F0 = _batched_mismatch_inf_norm(Y, v, th, P_set, Q_set, slack_mask, pv_mask)

                max_backtracks = max(1, int(self.armijo_max_backtracks))
                rho = min(max(float(self.armijo_rho), 1e-12), 1.0 - 1e-12)
                c1 = float(self.armijo_c1)
                min_alpha = max(0.0, float(self.armijo_min_alpha))

                if self.armijo_mode == "fixed":
                    alphas = v.new_tensor([rho ** i for i in range(max_backtracks)])
                    if min_alpha > 0.0:
                        alphas = alphas[alphas >= min_alpha]
                    if alphas.numel() == 0:
                        alphas = v.new_tensor([min_alpha])
                elif self.armijo_mode in ("geometric", "geometric_safe", "reject"):
                    alphas = []
                    a_tmp = 1.0
                    for _ in range(max_backtracks):
                        alphas.append(a_tmp)
                        a_tmp *= rho
                        if min_alpha > 0.0 and a_tmp < min_alpha:
                            break
                    alphas = v.new_tensor(alphas)
                else:
                    raise ValueError(f"Unknown armijo_mode={self.armijo_mode!r}; expected 'fixed', 'geometric', 'geometric_safe', or 'reject'.")

                def armijo_candidate(a):
                    v_try = torch.clamp(v + a * dv, v_min, v_max)
                    th_try = (th + a * dth + math.pi) % (2 * math.pi) - math.pi
                    with torch.no_grad():
                        F_try = _batched_mismatch_inf_norm(Y, v_try, th_try, P_set, Q_set, slack_mask, pv_mask)
                        ok = bool(F_try <= (1.0 - c1 * a) * F0)
                    return v_try, th_try, ok

                if self.armijo_mode == "fixed":
                    accepted = False
                    for a_tensor in alphas:
                        a = float(a_tensor)
                        v_try, th_try, ok = armijo_candidate(a)
                        if ok:
                            v = v_try
                            th = th_try
                            m = m + a * dm
                            accepted = True
                            break

                    if not accepted:
                        a = float(alphas[-1])
                        v2 = torch.clamp(v + a * dv, v_min, v_max)
                        th2 = (th + a * dth + math.pi) % (2 * math.pi) - math.pi
                        with torch.no_grad():
                            ok = _batched_mismatch_inf_norm(Y, v2, th2, P_set, Q_set, slack_mask, pv_mask) < F0
                        if ok:
                            v, th, m = v2, th2, m + a * dm
                    continue

                accepted = False
                a_sel = float(alphas[-1])
                for a_tensor in alphas:
                    a = float(a_tensor)
                    _, _, ok = armijo_candidate(a)
                    if ok:
                        accepted = True
                        a_sel = a
                        break

                if self.armijo_mode == "geometric_safe" and not accepted:
                    a_sel = min_alpha

                if self.armijo_mode == "reject" and not accepted:
                    continue

                # For geometric, match the LVN/student Armijo semantics: the
                # line search only selects a scalar step size. In geometric_safe,
                # rejected searches use a tiny configured fallback to keep
                # gradients alive while staying near old PPC. In reject mode,
                # rejected searches discard the update entirely.
                v = torch.clamp(v + a_sel * dv, v_min, v_max)
                th = (th + a_sel * dth + math.pi) % (2 * math.pi) - math.pi
                m = m + a_sel * dm
            else:
                th = (th + dth + math.pi) % (2 * math.pi) - math.pi
                v = torch.clamp(v + dv, 0.75, 1.20)
                m = m + dm

            if self.pinn:
                term = (self.gamma ** (self.K - 1 - k)) * self._physics_residual_loss(
                    DP, DQ, P_set, Q_set, p_mask, q_mask, n_nodes_per_graph
                )
                phys_terms.append(term)

        out = torch.stack([v, th], dim=-1)

        if self.pinn:
            if self.physics_final_weight != 0.0:
                Vc = v * torch.exp(1j * th)
                Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
                Sc = Vc * Ic.conj()
                DP = (P_set - Sc.real).masked_fill(slack_mask, 0.0)
                DQ = (Q_set - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)
                final_term = self._physics_residual_loss(
                    DP, DQ, P_set, Q_set, p_mask, q_mask, n_nodes_per_graph
                )
                phys_terms.append(self.physics_final_weight * final_term)
            phys_loss = torch.sum(torch.stack(phys_terms))
            return out, phys_loss

        return out
