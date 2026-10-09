import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


# -----------------------------------------------------------------------
# Re-usable MLP block
# -----------------------------------------------------------------------

class LearningBlock(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.lin1 = nn.Linear(dim_in, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, dim_out)
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.act(self.norm1(self.lin1(x)))
        x = self.act(self.norm2(self.lin2(x)))
        return self.lin3(x)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    return dtype

def _batched_mismatch_inf_norm(Y, v, th, P_set, Q_set, slack_mask, pv_mask):
    """
    Y:   (B,N,N) or (N,N)
    v:   (B,N)
    th:  (B,N)
    """
    if Y.dim() == 2:
        Y = Y.unsqueeze(0)

    Vc = v * torch.exp(1j * th)
    Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
    Sc = Vc * Ic.conj()

    DP = (P_set - Sc.real).masked_fill(slack_mask, 0.0)
    DQ = (Q_set - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)

    return torch.maximum(DP.abs().amax(dim=-1), DQ.abs().amax(dim=-1)).amax()


def _build_dense_Y_from_branchrows_single(
    N: int,
    Branch_f_bus: torch.Tensor,
    Branch_t_bus: torch.Tensor,
    Branch_status: torch.Tensor,
    Branch_tau: torch.Tensor,
    Branch_shift_deg: torch.Tensor,
    Branch_y_series_from: torch.Tensor,
    Branch_y_series_to: torch.Tensor,
    Branch_y_series_ft: torch.Tensor,
    Branch_y_shunt_from: torch.Tensor,
    Branch_y_shunt_to: torch.Tensor,
    Y_shunt_bus: torch.Tensor,
) -> torch.Tensor:
    """
    Dense Ybus reconstruction from one metadata row per PPC branch row.
    """
    device = Branch_f_bus.device
    dtype = Branch_y_series_from.dtype

    Y = torch.zeros(N, N, dtype=dtype, device=device)
    Y.diagonal().add_(Y_shunt_bus.to(dtype))

    mask = (Branch_status != 0)
    if mask.sum() == 0:
        return Y

    f = Branch_f_bus[mask].long()
    t = Branch_t_bus[mask].long()

    real_dtype = _real_dtype(dtype)
    tau = Branch_tau[mask].to(real_dtype)
    theta = torch.deg2rad(Branch_shift_deg[mask].to(real_dtype))
    a = tau.to(dtype) * torch.exp(1j * theta.to(dtype))

    y_from = Branch_y_series_from[mask].to(dtype)
    y_to   = Branch_y_series_to[mask].to(dtype)
    ysh_f  = Branch_y_shunt_from[mask].to(dtype)
    ysh_t  = Branch_y_shunt_to[mask].to(dtype)

    Yff = (y_from + ysh_f / 2.0) / (a * torch.conj(a))
    Ytt = (y_to   + ysh_t / 2.0)
    Yft = -y_from / torch.conj(a)
    Ytf = -y_to / a

    Y.index_put_((f, f), Yff, accumulate=True)
    Y.index_put_((t, t), Ytt, accumulate=True)
    Y.index_put_((f, t), Yft, accumulate=True)
    Y.index_put_((t, f), Ytf, accumulate=True)

    return Y


def _build_directed_edges_single(
    Branch_f_bus: torch.Tensor,
    Branch_t_bus: torch.Tensor,
    Branch_status: torch.Tensor,
    Branch_tau: torch.Tensor,
    Branch_shift_deg: torch.Tensor,
    Branch_y_series_from: torch.Tensor,
    Branch_y_series_to: torch.Tensor,
    Branch_y_series_ft: torch.Tensor,
    Branch_y_shunt_from: torch.Tensor,
    Branch_y_shunt_to: torch.Tensor,
    Is_trafo: torch.Tensor,
):
    """
    Build directed edge index + directed edge features from branch-row metadata.

    Edge feature layout:
      [Re(Ydir), Im(Ydir),
       Re(ysh_src), Im(ysh_src),
       Re(ysh_dst), Im(ysh_dst),
       tau, theta_rad, is_trafo]
    """
    device = Branch_f_bus.device
    ctype = Branch_y_series_from.dtype
    rtype = Branch_tau.dtype

    mask = (Branch_status != 0)
    if mask.sum() == 0:
        return (
            torch.empty(0, 2, dtype=torch.long, device=device),
            torch.empty(0, 9, dtype=rtype, device=device)
        )

    f = Branch_f_bus[mask].long()
    t = Branch_t_bus[mask].long()

    tau = Branch_tau[mask].to(rtype)
    theta = torch.deg2rad(Branch_shift_deg[mask].to(rtype))
    is_tr = Is_trafo[mask].to(rtype)

    a = tau.to(ctype) * torch.exp(1j * theta.to(ctype))

    y_from = Branch_y_series_from[mask].to(ctype)
    y_to = Branch_y_series_to[mask].to(ctype)
    ysh_f = Branch_y_shunt_from[mask].to(ctype)
    ysh_t = Branch_y_shunt_to[mask].to(ctype)

    ydir_ft = -y_from / torch.conj(a)   # f -> t
    ydir_tf = -y_to / a                 # t -> f

    feat_ft = torch.stack([
        ydir_ft.real.to(rtype), ydir_ft.imag.to(rtype),
        ysh_f.real.to(rtype),   ysh_f.imag.to(rtype),
        ysh_t.real.to(rtype),   ysh_t.imag.to(rtype),
        tau, theta, is_tr
    ], dim=-1)

    feat_tf = torch.stack([
        ydir_tf.real.to(rtype), ydir_tf.imag.to(rtype),
        ysh_t.real.to(rtype),   ysh_t.imag.to(rtype),
        ysh_f.real.to(rtype),   ysh_f.imag.to(rtype),
        tau, -theta, is_tr
    ], dim=-1)

    edge_index = torch.cat([
        torch.stack([f, t], dim=1),
        torch.stack([t, f], dim=1),
    ], dim=0)

    edge_feat = torch.cat([feat_ft, feat_tf], dim=0)
    return edge_index, edge_feat


# -----------------------------------------------------------------------
# GNSMsg adapted to branch-row metadata
# -----------------------------------------------------------------------

class GNSMsg(nn.Module):
    def __init__(
        self,
        d: int = 10,
        d_hi: int = 32,
        K: int = 30,
        pinn: bool = True,
        gamma: float = 0.9,
        v_limit: bool = True,
        use_armijo: bool = True,
    ):
        super().__init__()
        self.K = K
        self.d = d
        self.d_hi = d_hi
        self.pinn = pinn
        self.gamma = gamma
        self.v_limit = v_limit
        self.use_armijo = use_armijo

        # edge feature is now richer than old [Ysr, Ysi, Yc]
        # [Re(Ydir), Im(Ydir), Re(ysh_src), Im(ysh_src), Re(ysh_dst), Im(ysh_dst), tau, theta, is_trafo]
        self.edge_feat_dim = 9

        # per-iteration edge MLP
        self.edge_mlp = nn.ModuleList([
            LearningBlock(d + self.edge_feat_dim, d_hi, d) for _ in range(K)
        ])

        # node update heads
        in_dim = 4 + d + d   # [v, θ, ΔP, ΔQ] + m_i + aggregated message
        self.theta_upd = nn.ModuleList([LearningBlock(in_dim, d_hi, 1) for _ in range(K)])
        self.v_upd     = nn.ModuleList([LearningBlock(in_dim, d_hi, 1) for _ in range(K)])
        self.m_upd     = nn.ModuleList([LearningBlock(in_dim, d_hi, d) for _ in range(K)])

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
        """
        Same interface as adapted GNSMsg_EdgeSelfAttn.
        """
        device = bus_type.device
        B, N = bus_type.shape

        # If Y is not provided, reconstruct it
        if Y is None:
            if Y_shunt_bus is None:
                raise ValueError("Y is None and Y_shunt_bus is None; cannot reconstruct Y.")
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
                Y_list = []
                for b in range(B):
                    Y_list.append(_build_dense_Y_from_branchrows_single(
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
                Y = torch.stack(Y_list, dim=0)
        else:
            if Y.dim() == 2:
                Y = Y.unsqueeze(0)

        # Build directed sparse graph
        if B == 1:
            edge_index, edge_feat = _build_directed_edges_single(
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
            edge_index_list = None
            edge_feat_list = None
        else:
            edge_index = None
            edge_feat = None
            edge_index_list = []
            edge_feat_list = []
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
                edge_index_list.append(ei)
                edge_feat_list.append(ef)

        P_set, Q_set = S.real, S.imag

        v = V0[..., 0].clone()
        th = V0[..., 1].clone()
        m = torch.zeros(B, N, self.d, device=device, dtype=V0.dtype)

        slack_mask = (bus_type == 1)
        pv_mask = (bus_type == 2)

        phys_loss = torch.zeros(1, device=device) if self.pinn else None

        for k in range(self.K):
            # --- mismatches ---
            Vc = v * torch.exp(1j * th)
            Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
            Sc = Vc * Ic.conj()

            DP = (P_set - Sc.real).masked_fill(slack_mask, 0.0)
            DQ = (Q_set - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)

            # --- message passing ---
            if B == 1:
                M_neigh = torch.zeros(N, self.d, device=device, dtype=v.dtype)

                if edge_index.numel() > 0:
                    src = edge_index[:, 0]
                    dst = edge_index[:, 1]

                    m_src = m[0, src, :]                      # (E,d)
                    phi_in = torch.cat([m_src, edge_feat], dim=-1)   # (E, d+F)
                    phi = self.edge_mlp[k](phi_in)                  # (E,d)

                    M_neigh.index_add_(0, dst, phi)

                    deg = torch.zeros(N, device=device, dtype=v.dtype)
                    deg.index_add_(0, dst, torch.ones(dst.numel(), device=device, dtype=v.dtype))
                    M_neigh = M_neigh * deg.clamp_min(1.0).reciprocal().unsqueeze(-1)

                M_neigh = M_neigh.unsqueeze(0)   # (1,N,d)

            else:
                M_neigh = torch.zeros(B, N, self.d, device=device, dtype=v.dtype)

                for b in range(B):
                    ei = edge_index_list[b]
                    ef = edge_feat_list[b]
                    if ei.numel() == 0:
                        continue

                    src = ei[:, 0]
                    dst = ei[:, 1]

                    m_src = m[b, src, :]
                    phi_in = torch.cat([m_src, ef], dim=-1)
                    phi = self.edge_mlp[k](phi_in)

                    M_neigh[b].index_add_(0, dst, phi)

                    deg = torch.zeros(N, device=device, dtype=v.dtype)
                    deg.index_add_(0, dst, torch.ones(dst.numel(), device=device, dtype=v.dtype))
                    M_neigh[b] = M_neigh[b] * deg.clamp_min(1.0).reciprocal().unsqueeze(-1)

            # --- node updates ---
            bus_feat = torch.stack([v, th, DP, DQ], dim=-1)
            feats = torch.cat([bus_feat, m, M_neigh], dim=-1)

            dth = self.theta_upd[k](feats).squeeze(-1)
            dv  = self.v_upd[k](feats).squeeze(-1)
            dm  = torch.tanh(self.m_upd[k](feats))
            dm  = F.layer_norm(dm, dm.shape[-1:])

            dth = dth.masked_fill(slack_mask, 0.0)
            dv  = dv.masked_fill(slack_mask | pv_mask, 0.0)

            if self.v_limit:
                dtheta_max = 0.30
                dvm_frac = 0.10
                v_abs = v.abs()
                dth = torch.clamp(dth, -dtheta_max, dtheta_max)
                dv = torch.clamp(dv, -dvm_frac * v_abs, dvm_frac * v_abs)

            # --- Armijo ---
            if self.use_armijo:
                v_min, v_max = 0.8, 1.2
                with torch.no_grad():
                    F0 = _batched_mismatch_inf_norm(Y, v, th, P_set, Q_set, slack_mask, pv_mask)

                alphas = v.new_tensor([1.0, 0.5, 0.25, 0.125, 0.0625])
                accepted = False

                for a in alphas:
                    v_try = torch.clamp(v + a * dv, v_min, v_max)
                    th_try = (th + a * dth + math.pi) % (2 * math.pi) - math.pi

                    with torch.no_grad():
                        F1 = _batched_mismatch_inf_norm(Y, v_try, th_try, P_set, Q_set, slack_mask, pv_mask)

                    if F1 <= (1.0 - 1e-4 * a) * F0:
                        v, th, m = v_try, th_try, m + a * dm
                        accepted = True
                        break

                if not accepted:
                    a = alphas[-1]
                    v2 = torch.clamp(v + a * dv, v_min, v_max)
                    th2 = (th + a * dth + math.pi) % (2 * math.pi) - math.pi
                    with torch.no_grad():
                        ok = _batched_mismatch_inf_norm(Y, v2, th2, P_set, Q_set, slack_mask, pv_mask) < F0
                    if ok:
                        v, th, m = v2, th2, m + a * dm

            else:
                th = (th + dth + math.pi) % (2 * math.pi) - math.pi
                v = torch.clamp(v + dv, 0.8, 1.2)
                m = m + dm

            if self.pinn:
                phys_loss = phys_loss + (self.gamma ** (self.K - 1 - k)) * ((DP ** 2 + DQ ** 2).mean())

        out = torch.stack([v, th], dim=-1)
        return (out, phys_loss) if self.pinn else out
