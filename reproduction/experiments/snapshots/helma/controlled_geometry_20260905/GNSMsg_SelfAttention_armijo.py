import math
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max
from collate_blockdiag_optimized_complex_columns import ybus_matvec
from known_operator_pf import adjoint_ybus_matvec


# --------------------------- utils ---------------------------

def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    return dtype

def _segmented_softmax(logits_b_e_h: torch.Tensor, dst_e: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Softmax over incoming edges per (batch, head, destination-node).
    logits_b_e_h: (B, E, H)
    dst_e:        (E,)
    """
    B, E, H = logits_b_e_h.shape
    device = logits_b_e_h.device
    N = num_nodes

    b_ids = torch.arange(B, device=device).view(B, 1, 1)
    h_ids = torch.arange(H, device=device).view(1, 1, H)
    dst = dst_e.view(1, E, 1)

    seg = (b_ids * (N * H)) + (h_ids * N) + dst
    seg = seg.reshape(-1)
    src = logits_b_e_h.reshape(-1)

    max_per_seg, _ = scatter_max(src, seg, dim=0, dim_size=B * N * H)
    max_g = max_per_seg.index_select(0, seg)
    x = torch.exp(src - max_g)

    denom = scatter_add(x, seg, dim=0, dim_size=B * N * H)
    denom_g = denom.index_select(0, seg)

    alpha = (x / (denom_g + 1e-12)).reshape(B, E, H)
    return alpha


def _batched_mismatch_inf_norm(Y, v, th, P_set, Q_set, slack_mask, pv_mask,
                               norm="inf"):
    """Scalarise the AC mismatch over the whole packed batch for Armijo.

    'inf' is the worst bus anywhere in the batch. On a large grid that maximum
    is set by one pathological bus, so almost no step decreases it and the line
    search rejects everything -- and because the reduction spans the batch, a
    single bad bus in one scenario vetoes the step for all of them. 'rms'
    averages instead, so a step that improves most buses is accepted. The
    slack/PV zeros dilute the mean, but both sides of the Armijo inequality
    carry the same factor, so it cancels.
    """
    Vc = v * torch.exp(1j * th)
    Ic = ybus_matvec(Y, Vc)
    Sc = Vc * Ic.conj()

    DP = (P_set - Sc.real).masked_fill(slack_mask, 0.0)
    DQ = (Q_set - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)

    if norm == "rms":
        return torch.sqrt(DP.square().mean() + DQ.square().mean())
    return torch.maximum(DP.abs().amax(dim=-1), DQ.abs().amax(dim=-1)).amax()


def _per_graph_mismatch_inf_norm(
    Y, v, th, P_set, Q_set, slack_mask, pv_mask, n_nodes_per_graph=None,
    norm="inf"
):
    """Return one masked AC mismatch norm per independent grid.

    See `_batched_mismatch_inf_norm` for why 'rms' exists.
    """
    Vc = v * torch.exp(1j * th)
    Sc = Vc * ybus_matvec(Y, Vc).conj()
    DP = (P_set - Sc.real).masked_fill(slack_mask, 0.0)
    DQ = (Q_set - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)

    if norm == "rms":
        if n_nodes_per_graph is None:
            return torch.sqrt(DP.square().mean(dim=-1) + DQ.square().mean(dim=-1))
        vals = []
        offset = 0
        for size in n_nodes_per_graph.detach().cpu().tolist():
            size = int(size)
            sl = slice(offset, offset + size)
            vals.append(torch.sqrt(DP[0, sl].square().mean() + DQ[0, sl].square().mean()))
            offset += size
        return torch.stack(vals)

    if n_nodes_per_graph is None:
        return torch.maximum(DP.abs().amax(dim=-1), DQ.abs().amax(dim=-1))

    maxima = []
    offset = 0
    for size in n_nodes_per_graph.detach().cpu().tolist():
        size = int(size)
        sl = slice(offset, offset + size)
        maxima.append(torch.maximum(DP[0, sl].abs().amax(), DQ[0, sl].abs().amax()))
        offset += size
    return torch.stack(maxima)


def _ybus_diag_abs(Y, n):
    """$|Y_{ii}|$ as a real vector of length n, for dense or sparse block Y."""
    Ym = Y
    if Ym.dim() > 2:
        if Ym.shape[0] != 1:
            raise ValueError(f"Y-bus with leading dim {Ym.shape[0]} != 1")
        if Ym.is_sparse:
            c = Ym.coalesce()
            Ym = torch.sparse_coo_tensor(c.indices()[1:], c.values(), c.shape[1:])
        else:
            Ym = Ym[0]
    if Ym.is_sparse:
        c = Ym.coalesce()
        idx = c.indices()
        on_diag = idx[0] == idx[1]
        d = torch.zeros(n, dtype=c.values().dtype, device=c.values().device)
        d.index_add_(0, idx[0][on_diag], c.values()[on_diag])
        return d.abs()
    return torch.diagonal(Ym).abs()


def _expand_graph_values(values, reference, n_nodes_per_graph=None):
    """Expand one scalar per grid to the packed node layout."""
    if n_nodes_per_graph is None:
        return values.view(-1, 1).expand_as(reference)
    return torch.repeat_interleave(
        values, n_nodes_per_graph.to(device=values.device)
    ).view(1, -1)


def _reference_mse_gradient_scale(p_mask, q_mask, n_nodes_per_graph=None):
    """Derivative scale of the author's masked ``torch.nn.MSELoss``."""
    if n_nodes_per_graph is None:
        n_eq = (p_mask.sum(dim=-1) + q_mask.sum(dim=-1)).clamp_min(1)
        return (2.0 / n_eq).to(dtype=torch.float64).view(-1, 1)

    values = []
    offset = 0
    for size in n_nodes_per_graph.detach().cpu().tolist():
        size = int(size)
        sl = slice(offset, offset + size)
        n_eq = (p_mask[0, sl].sum() + q_mask[0, sl].sum()).clamp_min(1)
        values.append((2.0 / n_eq).expand(size))
        offset += size
    return torch.cat(values).view(1, -1).to(dtype=torch.float64)


def _normalize_per_graph(grad_vm, grad_va, n_nodes_per_graph=None):
    """Normalize gradient features independently for each packed grid."""
    if n_nodes_per_graph is None:
        scale = torch.maximum(
            grad_vm.abs().amax(dim=-1, keepdim=True),
            grad_va.abs().amax(dim=-1, keepdim=True),
        ).clamp_min(1e-12)
        return grad_vm / scale, grad_va / scale

    vm_parts = []
    va_parts = []
    offset = 0
    for size in n_nodes_per_graph.detach().cpu().tolist():
        size = int(size)
        sl = slice(offset, offset + size)
        scale = torch.maximum(
            grad_vm[0, sl].abs().amax(), grad_va[0, sl].abs().amax()
        ).clamp_min(1e-12)
        vm_parts.append(grad_vm[:, sl] / scale)
        va_parts.append(grad_va[:, sl] / scale)
        offset += size
    return torch.cat(vm_parts, dim=1), torch.cat(va_parts, dim=1)


def _project_to_descent_halfspace(
    dv, dth, grad_vm, grad_va, n_nodes_per_graph=None, margin=1e-4, eps=1e-12
):
    """Project a learned proposal onto ``g^T d <= -margin * ||g||^2``.

    This keeps the direct PIGNN correction expressive while making the
    unprojected hybrid direction a descent direction for the exact DPF MSE.
    The subsequent physical line search protects against voltage clipping and
    nonlinear finite-step effects.
    """
    if n_nodes_per_graph is None:
        dot = (grad_vm * dv + grad_va * dth).sum(dim=-1, keepdim=True)
        norm2 = (grad_vm.square() + grad_va.square()).sum(dim=-1, keepdim=True)
        coefficient = torch.relu(dot + margin * norm2) / norm2.clamp_min(eps)
        return dv - coefficient * grad_vm, dth - coefficient * grad_va

    projected_vm = []
    projected_va = []
    offset = 0
    for size in n_nodes_per_graph.detach().cpu().tolist():
        size = int(size)
        sl = slice(offset, offset + size)
        gvm = grad_vm[:, sl]
        gva = grad_va[:, sl]
        dvm = dv[:, sl]
        dva = dth[:, sl]
        dot = (gvm * dvm + gva * dva).sum(dim=-1, keepdim=True)
        norm2 = (gvm.square() + gva.square()).sum(dim=-1, keepdim=True)
        coefficient = torch.relu(dot + margin * norm2) / norm2.clamp_min(eps)
        projected_vm.append(dvm - coefficient * gvm)
        projected_va.append(dva - coefficient * gva)
        offset += size
    return torch.cat(projected_vm, dim=1), torch.cat(projected_va, dim=1)


def _scale_direction_to_limits(
    dv, dth, v, dtheta_max, dvm_frac, n_nodes_per_graph=None, eps=1e-12
):
    """Apply one positive scale per graph so projection remains a descent step."""
    v_bound = dvm_frac * v.abs()
    theta_bound = dth.new_full(dth.shape, dtheta_max)
    v_ratio = torch.where(
        dv.abs() > v_bound,
        v_bound / dv.abs().clamp_min(eps),
        torch.ones_like(dv),
    )
    theta_ratio = torch.where(
        dth.abs() > theta_bound,
        theta_bound / dth.abs().clamp_min(eps),
        torch.ones_like(dth),
    )

    if n_nodes_per_graph is None:
        scale = torch.minimum(
            v_ratio.amin(dim=-1, keepdim=True),
            theta_ratio.amin(dim=-1, keepdim=True),
        ).clamp(max=1.0)
        return scale * dv, scale * dth

    scales = []
    offset = 0
    for size in n_nodes_per_graph.detach().cpu().tolist():
        size = int(size)
        sl = slice(offset, offset + size)
        graph_scale = torch.minimum(v_ratio[:, sl].amin(), theta_ratio[:, sl].amin())
        scales.append(graph_scale.clamp(max=1.0).expand(size))
        offset += size
    scale = torch.cat(scales).view(1, -1)
    return scale * dv, scale * dth


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
    feature_norm: str = "none",
    ydiag_abs: torch.Tensor = None,
):
    """
    Build directed sparse edge list and direction-aware edge features from branch rows.

    Edge feature for f->t:
      [Re(Yft_dir), Im(Yft_dir),
       Re(ysh_f),   Im(ysh_f),
       Re(ysh_t),   Im(ysh_t),
       tau, theta_rad, is_trafo]

    Edge feature for t->f swaps the shunts and flips theta sign.
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

    ydir_ft = -y_from / torch.conj(a)  # f -> t
    ydir_tf = -y_to / a                # t -> f

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
    if feature_norm == "signed_log":
        # Only admittance-derived columns are compressed. Tap, phase shift,
        # and transformer identity retain their physical scale. The exact raw
        # branch tensors and Y-bus used by the AC residual are untouched.
        admittance = edge_feat[:, :6]
        edge_feat[:, :6] = torch.sign(admittance) * torch.log1p(admittance.abs())
    elif feature_norm in ("diagonal", "dual"):
        if ydiag_abs is None:
            raise ValueError(
                f"edge feature norm {feature_norm!r} requires |diag(Ybus)|"
            )
        diag = ydiag_abs.to(device=device, dtype=rtype).clamp_min(1e-12)
        diag_f = diag[f]
        diag_t = diag[t]
        cross = torch.sqrt(diag_f * diag_t).clamp_min(1e-12)

        normalized_ft = feat_ft.clone()
        normalized_tf = feat_tf.clone()
        normalized_ft[:, :2] /= cross.unsqueeze(-1)
        normalized_tf[:, :2] /= cross.unsqueeze(-1)
        normalized_ft[:, 2:4] /= diag_f.unsqueeze(-1)
        normalized_ft[:, 4:6] /= diag_t.unsqueeze(-1)
        normalized_tf[:, 2:4] /= diag_t.unsqueeze(-1)
        normalized_tf[:, 4:6] /= diag_f.unsqueeze(-1)
        normalized = torch.cat([normalized_ft, normalized_tf], dim=0)

        if feature_norm == "diagonal":
            edge_feat = normalized
        else:
            # Dual-scale edges expose both robust heavy-tail compression and
            # a Jacobi-like, base-invariant operator coordinate. Structural
            # channels are appended once because they are identical in both
            # views.
            signed = edge_feat[:, :6]
            signed = torch.sign(signed) * torch.log1p(signed.abs())
            edge_feat = torch.cat(
                [signed, normalized[:, :6], edge_feat[:, 6:]], dim=-1
            )
    elif feature_norm != "none":
        raise ValueError(
            f"unknown edge feature norm {feature_norm!r}; expected 'none', "
            "'signed_log', 'diagonal', or 'dual'"
        )
    return edge_index, edge_feat


# --------------------- attention block -----------------------

class EdgeSelfAttnBlock(nn.Module):
    """
    Sparse graph self-attention with edge bias.
    """
    def __init__(self, d_model: int, n_heads: int, edge_feat_dim: int, ffn_hidden: int = None, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.h = n_heads
        self.dh = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.edge_bias = nn.Sequential(
            nn.Linear(edge_feat_dim, max(16, 2 * edge_feat_dim)),
            nn.LeakyReLU(0.1),
            nn.Linear(max(16, 2 * edge_feat_dim), self.h)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        hid = ffn_hidden or (4 * d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index_dir: torch.Tensor, edge_feat_dir: torch.Tensor):
        """
        x:             (B, N, D)
        edge_index_dir:(E, 2)
        edge_feat_dir: (E, F)
        """
        B, N, D = x.shape
        device = x.device
        src = edge_index_dir[:, 0]
        dst = edge_index_dir[:, 1]

        y = self.ln1(x)
        Q = self.q(y).view(B, N, self.h, self.dh)
        K = self.k(y).view(B, N, self.h, self.dh)
        V = self.v(y).view(B, N, self.h, self.dh)

        Qi = Q[:, dst, :, :]
        Kj = K[:, src, :, :]
        Vj = V[:, src, :, :]

        logits = (Qi * Kj).sum(dim=-1) / math.sqrt(self.dh)
        bias = self.edge_bias(edge_feat_dir).unsqueeze(0)
        logits = logits + bias

        alpha = _segmented_softmax(logits, dst, N)
        attn_msg = alpha.unsqueeze(-1) * Vj

        out = torch.zeros(B, N, self.h, self.dh, device=device, dtype=x.dtype)
        out.index_add_(1, dst, attn_msg)
        out = self.drop(self.out(out.reshape(B, N, D)))
        x = x + out

        z = self.ln2(x)
        z = self.drop(self.ffn(z))
        return x + z


def _build_exact_two_hop_edges_single(edge_index_dir, n_nodes):
    """Build sparse directed distance-exactly-two edges and path counts."""
    n_nodes = int(n_nodes)
    outgoing = [set() for _ in range(n_nodes)]
    direct = set()
    for src, dst in edge_index_dir.detach().to("cpu").tolist():
        src, dst = int(src), int(dst)
        if src == dst or not (0 <= src < n_nodes and 0 <= dst < n_nodes):
            continue
        outgoing[src].add(dst)
        direct.add((src, dst))

    path_counts = {}
    for src in range(n_nodes):
        for mid in outgoing[src]:
            for dst in outgoing[mid]:
                pair = (src, dst)
                if src == dst or pair in direct:
                    continue
                path_counts[pair] = path_counts.get(pair, 0) + 1

    if not path_counts:
        return (
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((0, 1), dtype=torch.float32),
        )
    pairs = sorted(path_counts)
    edge_index = torch.tensor(pairs, dtype=torch.long)
    edge_feat = torch.tensor(
        [[math.log1p(path_counts[pair])] for pair in pairs],
        dtype=torch.float32,
    )
    return edge_index, edge_feat


class PerGraphGlobalContext(nn.Module):
    """Linear-cost graph context sidecar for the local edge attention.

    The PPC loader uses block-diagonal batches, so a single tensor row can
    contain several independent graphs.  This module deliberately receives
    ``n_nodes_per_graph`` and computes one token per graph; it never pools
    across scenarios.  ``meanmax`` is a cheap statistics-only context, while
    ``attn`` is a one-query cross-attention readout.  Both are O(N D), not
    dense O(N^2) self-attention.
    """

    def __init__(
        self,
        d_model: int,
        mode: str,
        n_heads: int = 1,
        inner_gate_mode: str = "none",
    ):
        super().__init__()
        if mode not in ("meanmax", "attn"):
            raise ValueError(f"unknown global context mode {mode!r}")
        if inner_gate_mode not in ("none", "sdpa_sigmoid"):
            raise ValueError(f"unknown inner global gate {inner_gate_mode!r}")
        if inner_gate_mode != "none" and mode != "attn":
            raise ValueError("post-SDPA sigmoid gating requires attentive context")
        if d_model % n_heads != 0:
            raise ValueError(
                f"global d_model={d_model} is not divisible by n_heads={n_heads}"
            )
        self.mode = mode
        self.d_model = int(d_model)
        self.inner_gate_mode = inner_gate_mode
        # The historical G2/G3 path remains one-query, one-head attention.
        # The paper-inspired gate uses the model's actual head partition so
        # every head and every value channel receives its own dynamic gate.
        self.h = int(n_heads) if inner_gate_mode == "sdpa_sigmoid" else 1
        self.dh = self.d_model // self.h
        if mode == "meanmax":
            self.pool = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.query = self.key = self.value = self.out = self.inner_gate = None
        else:
            self.pool = None
            self.query = nn.Linear(d_model, d_model, bias=False)
            self.key = nn.Linear(d_model, d_model, bias=False)
            self.value = nn.Linear(d_model, d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.inner_gate = (
                nn.Linear(d_model, d_model, bias=True)
                if inner_gate_mode == "sdpa_sigmoid"
                else None
            )

    @staticmethod
    def _segments(x: torch.Tensor, n_nodes_per_graph):
        """Yield ``(batch_row, slice)`` for each independent graph."""
        B, N, _ = x.shape
        if n_nodes_per_graph is None:
            for b in range(B):
                yield b, slice(0, N)
            return
        if B != 1:
            raise ValueError(
                "block-diagonal graph sizes require B=1 in the global context"
            )
        offset = 0
        for size in n_nodes_per_graph.detach().cpu().tolist():
            size = int(size)
            if size <= 0 or offset + size > N:
                raise ValueError(
                    f"invalid graph segment size={size} for packed N={N}"
                )
            yield 0, slice(offset, offset + size)
            offset += size
        if offset != N:
            raise ValueError(
                f"graph segments cover {offset} nodes but packed tensor has {N}"
            )

    def _read_one(self, h: torch.Tensor):
        """Return broadcast context, entropy, and inner-gate diagnostics."""
        # Layer-normalize node states before pooling so graph size and latent
        # scale do not turn the token into a proxy for the number of buses.
        hn = F.layer_norm(h, (h.shape[-1],))
        if self.mode == "meanmax":
            token = self.pool(torch.cat([hn.mean(dim=0), hn.amax(dim=0)], dim=-1))
            return token.unsqueeze(0).expand_as(h), None, None, None

        graph_state = hn.mean(dim=0)
        q = self.query(graph_state).view(self.h, self.dh)
        k = self.key(hn).view(h.shape[0], self.h, self.dh)
        value = self.value(hn).view(h.shape[0], self.h, self.dh)
        logits = (k * q.unsqueeze(0)).sum(dim=-1) / math.sqrt(self.dh)
        weights = torch.softmax(logits, dim=0)
        token = (weights.unsqueeze(-1) * value).sum(dim=0)
        gate_mean = gate_low_fraction = None
        if self.inner_gate is not None:
            # Paper-faithful position: gate the attention-weighted value
            # independently by head and channel, then apply W_O.
            inner_gate = torch.sigmoid(
                self.inner_gate(graph_state).view(self.h, self.dh)
            )
            token = token * inner_gate
            gate_mean = inner_gate.mean()
            gate_low_fraction = (inner_gate < 0.1).to(inner_gate.dtype).mean()
        context = self.out(token.reshape(self.d_model)).unsqueeze(0).expand_as(h)
        if h.shape[0] > 1:
            entropy = -torch.sum(
                weights * torch.log(weights.clamp_min(1e-12)), dim=0
            )
            entropy = (entropy / math.log(float(h.shape[0]))).mean()
        else:
            entropy = weights.new_zeros(())
        return context, entropy, gate_mean, gate_low_fraction

    def forward(self, x: torch.Tensor, n_nodes_per_graph=None):
        contexts = []
        entropies = []
        gate_means = []
        gate_low_fractions = []
        if n_nodes_per_graph is None:
            for b in range(x.shape[0]):
                context, entropy, gate_mean, gate_low = self._read_one(x[b])
                contexts.append(context)
                if entropy is not None:
                    entropies.append(entropy)
                if gate_mean is not None:
                    gate_means.append(gate_mean)
                    gate_low_fractions.append(gate_low)
            context = torch.stack(contexts, dim=0)
        else:
            packed = []
            for b, sl in self._segments(x, n_nodes_per_graph):
                ctx, entropy, gate_mean, gate_low = self._read_one(x[b, sl])
                packed.append(ctx)
                if entropy is not None:
                    entropies.append(entropy)
                if gate_mean is not None:
                    gate_means.append(gate_mean)
                    gate_low_fractions.append(gate_low)
            context = torch.cat(packed, dim=0).unsqueeze(0)

        if entropies:
            entropy_mean = torch.stack(entropies).mean()
        else:
            entropy_mean = x.new_zeros(())
        if gate_means:
            inner_gate_mean = torch.stack(gate_means).mean()
            inner_gate_low_fraction = torch.stack(gate_low_fractions).mean()
        else:
            inner_gate_mean = x.new_zeros(())
            inner_gate_low_fraction = x.new_zeros(())
        return context, entropy_mean, inner_gate_mean, inner_gate_low_fraction


class RangeMasterContext(nn.Module):
    """RANGE-inspired bidirectional node/master attention in pure PyTorch.

    This is a power-grid adaptation of the aggregation and broadcast blocks in
    the MIT-licensed official RANGE implementation (Clementi Group, 2025):
    https://github.com/ClementiGroup/RANGE .  It intentionally has no dependency
    on ``rangemp``, ``mlcg`` or ``e3nn``.  Every graph is processed separately,
    so block-diagonal batches cannot exchange information.

    The master state is *not* committed here.  The PIGNN solver returns a
    candidate state and commits it with the same Armijo alpha used for voltage,
    angle and recurrent-memory updates.
    """

    def __init__(
        self,
        node_dim: int,
        master_dim: int,
        num_heads: int,
        num_masters: int,
        pe_dim: int,
    ):
        super().__init__()
        if node_dim <= 0 or master_dim <= 0 or num_masters <= 0 or pe_dim <= 0:
            raise ValueError("RANGE dimensions and number of masters must be positive")
        if master_dim % num_heads != 0:
            raise ValueError(
                f"range_master_dim={master_dim} is not divisible by "
                f"range_num_heads={num_heads}"
            )
        self.node_dim = int(node_dim)
        self.master_dim = int(master_dim)
        self.num_heads = int(num_heads)
        self.num_masters = int(num_masters)
        self.pe_dim = int(pe_dim)
        self.head_dim = self.master_dim // self.num_heads

        self.master_init = nn.Parameter(
            torch.empty(self.num_masters, self.master_dim)
        )

        # Node -> master aggregation.  The additive GATv2-style score matches
        # RANGE: a^T LeakyReLU(W_Q u_m + W_K x_i + W_E p_i).
        self.agg_q = nn.Linear(self.master_dim, self.master_dim, bias=False)
        self.agg_k = nn.Linear(self.node_dim, self.master_dim, bias=False)
        self.agg_v = nn.Linear(self.node_dim, self.master_dim, bias=False)
        self.agg_e = nn.Linear(self.pe_dim, self.master_dim, bias=False)
        self.agg_attention = nn.Parameter(
            torch.empty(1, self.num_heads, self.head_dim)
        )
        self.agg_norm = nn.LayerNorm(self.master_dim)

        # Master -> node broadcast.  Each bus attends to all masters plus its
        # own self-loop.  Master and self-loop values use separate transforms,
        # as in the official RANGE broadcast block.
        self.bcast_q = nn.Linear(self.node_dim, self.master_dim, bias=False)
        self.bcast_master_k = nn.Linear(
            self.master_dim, self.master_dim, bias=False
        )
        self.bcast_master_v = nn.Linear(
            self.master_dim, self.master_dim, bias=False
        )
        self.bcast_self_k = nn.Linear(self.node_dim, self.master_dim, bias=False)
        self.bcast_self_v = nn.Linear(self.node_dim, self.master_dim, bias=False)
        self.bcast_e = nn.Linear(self.pe_dim, self.master_dim, bias=False)
        self.bcast_attention = nn.Parameter(
            torch.empty(1, self.num_heads, self.head_dim)
        )
        self.bcast_out = nn.Sequential(
            nn.Linear(self.master_dim, self.master_dim, bias=False),
            nn.LayerNorm(self.master_dim),
            nn.GELU(),
            nn.Linear(self.master_dim, self.node_dim, bias=False),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.master_init, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.agg_attention, mean=0.0, std=0.02)
        nn.init.normal_(self.bcast_attention, mean=0.0, std=0.02)

    def initial_state(self, num_graphs: int, *, device, dtype):
        return self.master_init.to(device=device, dtype=dtype).unsqueeze(0).expand(
            int(num_graphs), -1, -1
        )

    @staticmethod
    def _offdiag_cosine(vectors: torch.Tensor) -> torch.Tensor:
        count = int(vectors.shape[0])
        if count <= 1:
            return vectors.new_zeros(())
        flat = F.normalize(vectors.reshape(count, -1), dim=-1, eps=1e-12)
        similarity = flat @ flat.transpose(0, 1)
        return (similarity.sum() - similarity.diagonal().sum()) / (
            count * (count - 1)
        )

    @staticmethod
    def _normalized_entropy(weights: torch.Tensor, candidate_dim: int) -> torch.Tensor:
        if candidate_dim <= 1:
            return weights.new_zeros(())
        entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=1)
        return entropy.mean() / math.log(float(candidate_dim))

    def _one_graph(
        self,
        nodes: torch.Tensor,
        masters: torch.Tensor,
        pe: torch.Tensor,
    ):
        n_nodes = int(nodes.shape[0])
        if n_nodes <= 0:
            raise ValueError("RANGE received an empty graph")

        q_master = self.agg_q(masters).view(
            self.num_masters, self.num_heads, self.head_dim
        )
        k_node = self.agg_k(nodes).view(n_nodes, self.num_heads, self.head_dim)
        v_node = self.agg_v(nodes).view(n_nodes, self.num_heads, self.head_dim)
        e_node = self.agg_e(pe).view(n_nodes, self.num_heads, self.head_dim)
        agg_logits = (
            self.agg_attention
            * F.leaky_relu(
                q_master.unsqueeze(0)
                + k_node.unsqueeze(1)
                + e_node.unsqueeze(1),
                negative_slope=0.2,
            )
        ).sum(dim=-1)
        # [node, master, head], normalized over all buses for each master/head.
        agg_weights = torch.softmax(agg_logits, dim=0)
        master_candidate = torch.einsum("nmh,nhd->mhd", agg_weights, v_node)
        master_candidate = master_candidate.reshape(
            self.num_masters, self.master_dim
        )
        master_candidate = F.gelu(self.agg_norm(master_candidate))

        q_node = self.bcast_q(nodes).view(n_nodes, self.num_heads, self.head_dim)
        k_master = self.bcast_master_k(master_candidate).view(
            self.num_masters, self.num_heads, self.head_dim
        )
        v_master = self.bcast_master_v(master_candidate).view(
            self.num_masters, self.num_heads, self.head_dim
        )
        k_self = self.bcast_self_k(nodes).view(
            n_nodes, self.num_heads, self.head_dim
        )
        v_self = self.bcast_self_v(nodes).view(
            n_nodes, self.num_heads, self.head_dim
        )
        e_master = self.bcast_e(pe).view(
            n_nodes, self.num_heads, self.head_dim
        )
        master_logits = (
            self.bcast_attention
            * F.leaky_relu(
                q_node.unsqueeze(1)
                + k_master.unsqueeze(0)
                + e_master.unsqueeze(1),
                negative_slope=0.2,
            )
        ).sum(dim=-1)
        # RANGE gives broadcast self-loops zero positional edge attributes.
        self_logits = (
            self.bcast_attention
            * F.leaky_relu(q_node + k_self, negative_slope=0.2)
        ).sum(dim=-1).unsqueeze(1)
        bcast_logits = torch.cat([master_logits, self_logits], dim=1)
        bcast_weights = torch.softmax(bcast_logits, dim=1)
        master_message = torch.einsum(
            "nmh,mhd->nhd", bcast_weights[:, : self.num_masters], v_master
        )
        self_message = bcast_weights[:, self.num_masters].unsqueeze(-1) * v_self
        node_output = self.bcast_out(
            (master_message + self_message).reshape(n_nodes, self.master_dim)
        )

        usage = bcast_weights[:, : self.num_masters].sum(dim=(0, 2))
        usage_prob = usage / usage.sum().clamp_min(1e-12)
        effective_masters = 1.0 / usage_prob.square().sum().clamp_min(1e-12)
        diagnostics = {
            "aggregation_entropy": self._normalized_entropy(
                agg_weights.permute(1, 0, 2), n_nodes
            ),
            "broadcast_entropy": self._normalized_entropy(
                bcast_weights, self.num_masters + 1
            ),
            "master_attention_cosine": self._offdiag_cosine(
                agg_weights.permute(1, 0, 2)
            ),
            "master_embedding_similarity": self._offdiag_cosine(master_candidate),
            "effective_masters": effective_masters,
            "self_loop_fraction": bcast_weights[:, self.num_masters].mean(),
            "token_utilization": usage_prob,
        }
        return node_output, master_candidate, diagnostics

    def forward(
        self,
        x: torch.Tensor,
        masters: torch.Tensor,
        pe: torch.Tensor,
        n_nodes_per_graph=None,
    ):
        outputs = []
        candidates = []
        diagnostics = []
        segments = list(PerGraphGlobalContext._segments(x, n_nodes_per_graph))
        if masters.shape[0] != len(segments):
            raise ValueError(
                f"RANGE has {masters.shape[0]} master groups for "
                f"{len(segments)} graph segments"
            )
        for graph_index, (batch_row, sl) in enumerate(segments):
            node_output, master_candidate, diag = self._one_graph(
                x[batch_row, sl], masters[graph_index], pe[batch_row, sl]
            )
            outputs.append(node_output)
            candidates.append(master_candidate)
            diagnostics.append(diag)

        if n_nodes_per_graph is None:
            node_output = torch.stack(outputs, dim=0)
        else:
            node_output = torch.cat(outputs, dim=0).unsqueeze(0)
        master_candidate = torch.stack(candidates, dim=0)
        merged = {}
        for key in diagnostics[0]:
            values = [diag[key] for diag in diagnostics]
            merged[key] = torch.stack(values, dim=0).mean(dim=0)
        return node_output, master_candidate, merged


# ------------------ main model ------------------

class GNSMsg_EdgeSelfAttn(nn.Module):
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
        armijo_norm: str = "inf",
        bus_feat_extra_dim: int = 0,
        bus_type_features: bool = False,
        heterogeneous_injection_features: bool = False,
        two_hop_attention_mode: str = "none",
        global_context_mode: str = "none",
        global_context_gate_mode: str = "scalar_tanh",
        range_num_masters: int = 1,
        range_master_dim: int = 24,
        range_num_heads: int = 8,
        range_positional_encoding: str = "hop_slack_rbf",
        range_pe_dim: int = 10,
        range_share_grid: bool = False,
        dtheta_max: float = 0.30,
        dvm_frac: float = 0.10,
        physics_loss_form: str = "mse",
        physics_residual_norm: str = "none",
        residual_feature_norm: str = "none",
        edge_feature_norm: str = "none",
        relative_stiffness_feature: bool = False,
        physics_norm_eps: float = 1e-6,
        physics_huber_delta: float = 1.0,
        physics_final_weight: float = 0.0,
        solver_update_mode: str = "direct",
        preconditioner_vm_step: float = 0.003377,
        preconditioner_va_step: float = 0.003377,
        preconditioner_log_clip: float = 5.0,
        preconditioner_optimizer: str = "author_adam",
        preconditioner_beta1: float = 0.979681,
        preconditioner_beta2: float = 0.963442,
        preconditioner_eps: float = 1e-8,
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
        if armijo_norm not in ("inf", "rms"):
            raise ValueError(
                f"Unknown armijo_norm={armijo_norm!r}; expected 'inf' or 'rms'."
            )
        self.armijo_norm = armijo_norm
        self.dtheta_max = dtheta_max
        self.dvm_frac = dvm_frac
        self.physics_loss_form = physics_loss_form
        self.physics_residual_norm = physics_residual_norm
        # Manifold projection basis, installed by the driver before epoch 0 and
        # frozen thereafter.  None disables the projection entirely.
        self.manifold_basis = None
        # "terminal" projects only the emitted state, leaving the K solver
        # steps and the Armijo line search exactly as they were tuned;
        # "per_step" projects every iterate, so the residual features fed back
        # at the next step and every physics term are computed from a state
        # inside the manifold.  per_step is the principled version -- it makes
        # physics_final_weight unnecessary -- but it changes the trajectory the
        # step clamp and the line search were tuned against, so a failure there
        # cannot be attributed between the projection and that interaction.
        self.manifold_mode = "terminal"
        if residual_feature_norm not in ("none", "signed_log", "ybus", "dual"):
            raise ValueError(
                f"Unknown residual_feature_norm={residual_feature_norm!r}; "
                "expected 'none', 'signed_log', 'ybus', or 'dual'."
            )
        self.residual_feature_norm = residual_feature_norm
        if edge_feature_norm not in ("none", "signed_log", "diagonal", "dual"):
            raise ValueError(
                f"Unknown edge_feature_norm={edge_feature_norm!r}; expected "
                "'none', 'signed_log', 'diagonal', or 'dual'."
            )
        self.edge_feature_norm = edge_feature_norm
        self.relative_stiffness_feature = bool(relative_stiffness_feature)
        if residual_feature_norm == "dual" and not self.relative_stiffness_feature:
            raise ValueError(
                "residual_feature_norm='dual' requires "
                "relative_stiffness_feature=True so its gate sees local scale."
            )
        self.physics_norm_eps = physics_norm_eps
        self.physics_huber_delta = physics_huber_delta
        self.physics_final_weight = physics_final_weight
        if solver_update_mode not in (
            "direct", "physics_preconditioner", "physics_projected_hybrid"
        ):
            raise ValueError(
                f"Unknown solver_update_mode={solver_update_mode!r}; expected "
                "'direct', 'physics_preconditioner', or "
                "'physics_projected_hybrid'."
            )
        if preconditioner_vm_step <= 0.0 or preconditioner_va_step <= 0.0:
            raise ValueError("Preconditioner initial step scales must be positive.")
        if preconditioner_log_clip <= 0.0:
            raise ValueError("preconditioner_log_clip must be positive.")
        if preconditioner_optimizer not in ("author_adam", "gd"):
            raise ValueError("preconditioner_optimizer must be 'author_adam' or 'gd'.")
        if not 0.0 <= preconditioner_beta1 < 1.0:
            raise ValueError("preconditioner_beta1 must lie in [0, 1).")
        if not 0.0 <= preconditioner_beta2 < 1.0:
            raise ValueError("preconditioner_beta2 must lie in [0, 1).")
        if preconditioner_eps <= 0.0:
            raise ValueError("preconditioner_eps must be positive.")
        if solver_update_mode == "physics_projected_hybrid":
            if not use_armijo:
                raise ValueError(
                    "physics_projected_hybrid requires use_armijo=True for its "
                    "finite-step physical safeguard."
                )
            if armijo_mode != "reject":
                raise ValueError(
                    "physics_projected_hybrid requires armijo_mode='reject'; "
                    "fallback steps do not preserve residual monotonicity."
                )
        self.solver_update_mode = solver_update_mode
        self.preconditioner_vm_step = preconditioner_vm_step
        self.preconditioner_va_step = preconditioner_va_step
        self.preconditioner_log_clip = preconditioner_log_clip
        self.preconditioner_optimizer = preconditioner_optimizer
        self.preconditioner_beta1 = preconditioner_beta1
        self.preconditioner_beta2 = preconditioner_beta2
        self.preconditioner_eps = preconditioner_eps
        self.last_solver_diagnostics = None

        self.d_model = d_model if d_model is not None else d_hi
        self.n_heads = n_heads
        assert self.d_model % self.n_heads == 0
        self.num_attn_layers = num_attn_layers

        self.bus_feat_extra_dim = int(bus_feat_extra_dim)
        self.bus_type_features = bool(bus_type_features)
        self.heterogeneous_injection_features = bool(heterogeneous_injection_features)
        if two_hop_attention_mode not in ("none", "pre", "post"):
            raise ValueError(
                f"Unknown two_hop_attention_mode={two_hop_attention_mode!r}; "
                "expected 'none', 'pre', or 'post'."
            )
        self.two_hop_attention_mode = two_hop_attention_mode
        self._two_hop_cache = {}
        self.last_two_hop_edge_count = 0
        if global_context_mode not in (
            "none", "meanmax_pre", "attn_pre", "attn_post", "range_post"
        ):
            raise ValueError(
                f"Unknown global_context_mode={global_context_mode!r}; expected "
                "'none', 'meanmax_pre', 'attn_pre', 'attn_post', or "
                "'range_post'."
            )
        self.global_context_mode = global_context_mode
        if global_context_gate_mode not in ("scalar_tanh", "sdpa_sigmoid"):
            raise ValueError(
                f"Unknown global_context_gate_mode={global_context_gate_mode!r}; "
                "expected 'scalar_tanh' or 'sdpa_sigmoid'."
            )
        if global_context_gate_mode == "sdpa_sigmoid" and global_context_mode not in (
            "attn_pre", "attn_post"
        ):
            raise ValueError(
                "global_context_gate_mode='sdpa_sigmoid' requires "
                "global_context_mode='attn_pre' or 'attn_post'."
            )
        self.global_context_gate_mode = global_context_gate_mode
        if range_positional_encoding != "hop_slack_rbf":
            raise ValueError(
                "range_positional_encoding must be 'hop_slack_rbf'"
            )
        self.range_num_masters = int(range_num_masters)
        self.range_master_dim = int(range_master_dim)
        self.range_num_heads = int(range_num_heads)
        self.range_positional_encoding = range_positional_encoding
        self.range_pe_dim = int(range_pe_dim)
        self.range_share_grid = bool(range_share_grid)
        self._range_pe_cache_cpu = None
        self._range_pe_cache_n = None
        # The integrated solver exposes the exact polar AC-PF gradient to the
        # graph network.  The direct predictor retains the historical feature
        # contract for checkpoint compatibility.
        physics_gradient_dim = 2 if solver_update_mode != "direct" else 0
        # 4 = [v, theta, DP, DQ]. signed_log divides the mismatch by a per-graph
        # scale, so the scale itself is handed to the network as one more
        # channel -- without it the transform is not invertible from the input.
        self.bus_feat_dim = (
            4 + physics_gradient_dim + self.bus_feat_extra_dim + d
            + (4 if self.bus_type_features else 0)
            + (1 if residual_feature_norm in ("signed_log", "dual") else 0)
            + (1 if self.relative_stiffness_feature else 0)
        )
        self.edge_feat_dim = 15 if edge_feature_norm == "dual" else 9

        self.in_proj = nn.Linear(self.bus_feat_dim, self.d_model)
        if self.heterogeneous_injection_features:
            # GridSFM-inspired typed one-hop fusion.  Generator and load proxy
            # nodes are incident to exactly one bus in the PPC adapter, so
            # relation-specific encoders followed by addition reproduce that
            # association without changing PIGNN's bus/branch attention graph.
            self.generator_role_proj = nn.Linear(2, self.d_model, bias=False)
            self.load_role_proj = nn.Linear(2, self.d_model, bias=False)
        else:
            self.generator_role_proj = None
            self.load_role_proj = None
        if global_context_mode == "meanmax_pre":
            self.global_context = PerGraphGlobalContext(self.d_model, "meanmax")
        elif global_context_mode in ("attn_pre", "attn_post"):
            self.global_context = PerGraphGlobalContext(
                self.d_model,
                "attn",
                n_heads=self.n_heads,
                inner_gate_mode=(
                    "sdpa_sigmoid"
                    if global_context_gate_mode == "sdpa_sigmoid"
                    else "none"
                ),
            )
        else:
            self.global_context = None
        self.range_context = (
            RangeMasterContext(
                node_dim=self.d_model,
                master_dim=self.range_master_dim,
                num_heads=self.range_num_heads,
                num_masters=self.range_num_masters,
                pe_dim=self.range_pe_dim,
            )
            if global_context_mode == "range_post"
            else None
        )
        # Zero gate keeps the initial function exactly equal to the G0 local
        # PIGNN.  The shared scalar can grow only if graph context helps the
        # supervised/physics objective.
        self.global_context_gate = (
            nn.Parameter(torch.zeros(())) if self.global_context is not None else None
        )
        self.last_global_gate = 0.0
        self.last_global_attention_entropy = 0.0
        self.last_global_inner_gate_mean = 0.0
        self.last_global_inner_gate_low_fraction = 0.0
        self.last_range_trace = []
        self.last_range_diagnostics = None
        self.record_step_states = False
        self.last_step_states = []
        if residual_feature_norm == "dual":
            self.dual_residual_gate = nn.Sequential(
                nn.Linear(2, 8),
                nn.GELU(),
                nn.Linear(8, 1),
                nn.Sigmoid(),
            )
        else:
            self.dual_residual_gate = None
        self.blocks = nn.ModuleList([
            EdgeSelfAttnBlock(
                self.d_model,
                self.n_heads,
                self.edge_feat_dim,
                ffn_hidden=4 * self.d_model,
                dropout=attn_dropout
            )
            for _ in range(self.num_attn_layers)
        ])
        self.two_hop_block = (
            EdgeSelfAttnBlock(
                self.d_model,
                self.n_heads,
                1,
                ffn_hidden=4 * self.d_model,
                dropout=attn_dropout,
            )
            if self.two_hop_attention_mode != "none"
            else None
        )

        self.theta_head = nn.ModuleList([nn.Linear(self.d_model, 1) for _ in range(K)])
        self.v_head     = nn.ModuleList([nn.Linear(self.d_model, 1) for _ in range(K)])
        self.m_head     = nn.ModuleList([nn.Linear(self.d_model, d) for _ in range(K)])
        if solver_update_mode == "physics_projected_hybrid":
            self.theta_preconditioner_head = nn.ModuleList(
                [nn.Linear(self.d_model, 1) for _ in range(K)]
            )
            self.v_preconditioner_head = nn.ModuleList(
                [nn.Linear(self.d_model, 1) for _ in range(K)]
            )
        else:
            self.theta_preconditioner_head = None
            self.v_preconditioner_head = None

        # Output heads: zero-init so the model starts with identity corrections
        # (Vpred = V_start at epoch 0), matching original behaviour.
        for k in range(K):
            nn.init.zeros_(self.theta_head[k].weight); nn.init.zeros_(self.theta_head[k].bias)
            nn.init.zeros_(self.v_head[k].weight);     nn.init.zeros_(self.v_head[k].bias)
            nn.init.zeros_(self.m_head[k].weight);     nn.init.zeros_(self.m_head[k].bias)
            if self.theta_preconditioner_head is not None:
                nn.init.zeros_(self.theta_preconditioner_head[k].weight)
                nn.init.zeros_(self.theta_preconditioner_head[k].bias)
                nn.init.zeros_(self.v_preconditioner_head[k].weight)
                nn.init.zeros_(self.v_preconditioner_head[k].bias)

        # Feature extraction layers (in_proj + attention blocks): sd=0.02 normal
        # init, matching the student's weight_init="sd0.02" recipe. This gives
        # the attention layers a meaningful starting point so gradient flow is
        # better conditioned from the first update.
        init_modules = [self.in_proj] + list(self.blocks.modules())
        if self.two_hop_block is not None:
            init_modules += list(self.two_hop_block.modules())
        if self.global_context is not None:
            init_modules += list(self.global_context.modules())
        if self.range_context is not None:
            init_modules += list(self.range_context.modules())
        if self.dual_residual_gate is not None:
            init_modules += list(self.dual_residual_gate.modules())
        for module in init_modules:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _two_hop_for_packed(self, edge_index, n_nodes_per_graph, device, dtype):
        sizes = (
            (int(edge_index.max().item()) + 1,)
            if n_nodes_per_graph is None
            else tuple(int(v) for v in n_nodes_per_graph.tolist())
        )
        key = (sizes, int(edge_index.shape[0]))
        cached = self._two_hop_cache.get(key)
        if cached is None:
            all_edges, all_features = [], []
            offset = 0
            for size in sizes:
                mask = (
                    (edge_index[:, 0] >= offset)
                    & (edge_index[:, 0] < offset + size)
                    & (edge_index[:, 1] >= offset)
                    & (edge_index[:, 1] < offset + size)
                )
                local_edges = edge_index[mask] - offset
                hop_edges, hop_features = _build_exact_two_hop_edges_single(
                    local_edges, size
                )
                all_edges.append(hop_edges + offset)
                all_features.append(hop_features)
                offset += size
            cached = (
                torch.cat(all_edges, dim=0),
                torch.cat(all_features, dim=0),
            )
            self._two_hop_cache[key] = cached
        hop_edges, hop_features = cached
        self.last_two_hop_edge_count = int(hop_edges.shape[0])
        return (
            hop_edges.to(device=device),
            hop_features.to(device=device, dtype=dtype),
        )

    def _apply_two_hop(self, x, edge_index, n_nodes_per_graph=None):
        hop_edges, hop_features = self._two_hop_for_packed(
            edge_index, n_nodes_per_graph, x.device, x.dtype
        )
        if hop_edges.numel() == 0:
            return x
        return self.two_hop_block(x, hop_edges, hop_features)

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

    def _relative_stiffness(self, ydiag_abs, reference, n_nodes_per_graph):
        """Log diagonal admittance relative to each graph's geometric mean."""
        logdiag = torch.log(
            ydiag_abs.to(device=reference.device, dtype=reference.dtype)
            .clamp_min(self.physics_norm_eps)
        ).view(1, -1)
        if n_nodes_per_graph is None:
            centered = logdiag - logdiag.mean(dim=-1, keepdim=True)
            return centered.expand_as(reference)

        centered = torch.empty_like(logdiag)
        offset = 0
        for size in n_nodes_per_graph.detach().cpu().tolist():
            size = int(size)
            sl = slice(offset, offset + size)
            centered[:, sl] = logdiag[:, sl] - logdiag[:, sl].mean(
                dim=-1, keepdim=True
            )
            offset += size
        return centered.expand_as(reference)

    def _residual_features(self, DP, DQ, P_set, Q_set, p_mask, q_mask,
                           ydiag_abs, v, n_nodes_per_graph):
        """Rescale the mismatch on its way into the input projection.

        `physics_residual_norm` rescales the *loss* only; DP and DQ still reach
        `in_proj` raw, and on LVN_heo1 they run to 1e6. A linear layer cannot
        absorb that, so the step clamp ends up carrying the update and the
        clipped coordinates stop receiving gradient. Returns the two features
        plus an optional extra channel to append.
        """
        if self.residual_feature_norm == "none":
            return DP, DQ, None, None, None

        if self.residual_feature_norm == "ybus":
            # |Y_ii| |V|^2 converts a power mismatch into the voltage correction
            # that would remove it, so the ratio is a per-unit voltage error and
            # is comparable across grids whatever their impedance base.
            scale = (ydiag_abs * v.square()).clamp_min(self.physics_norm_eps)
            return DP / scale, DQ / scale, None, None, None

        S_abs = torch.sqrt(
            (P_set * p_mask.to(P_set.dtype)) ** 2
            + (Q_set * q_mask.to(Q_set.dtype)) ** 2
        )
        scale = self._graph_scale_from_s(S_abs, n_nodes_per_graph)
        dp = torch.sign(DP) * torch.log1p(DP.abs() / scale)
        dq = torch.sign(DQ) * torch.log1p(DQ.abs() / scale)
        scale_feature = torch.log(scale).expand_as(DP)
        if self.residual_feature_norm == "signed_log":
            return dp, dq, scale_feature, None, None

        local_scale = (ydiag_abs * v.square()).clamp_min(self.physics_norm_eps)
        return dp, dq, scale_feature, DP / local_scale, DQ / local_scale

    def _apply_global_context(self, x, n_nodes_per_graph):
        if self.global_context is None:
            return x
        context, entropy, inner_gate_mean, inner_gate_low = self.global_context(
            x, n_nodes_per_graph
        )
        gate = torch.tanh(self.global_context_gate)
        self.last_global_gate = float(gate.detach().cpu().item())
        self.last_global_attention_entropy = float(entropy.detach().cpu().item())
        self.last_global_inner_gate_mean = float(
            inner_gate_mean.detach().cpu().item()
        )
        self.last_global_inner_gate_low_fraction = float(
            inner_gate_low.detach().cpu().item()
        )
        return x + gate * context

    def _hop_slack_rbf_one(self, bus_type, f_bus, t_bus, status):
        """Return fixed RBF channels of normalized nearest-slack hop distance."""
        n_nodes = int(bus_type.numel())
        bus_type_cpu = bus_type.detach().to(device="cpu", dtype=torch.long)
        f_cpu = f_bus.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        t_cpu = t_bus.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        status_cpu = status.detach().to(device="cpu").reshape(-1).bool()

        slack_nodes = torch.nonzero(bus_type_cpu == 1, as_tuple=False).reshape(-1)
        if slack_nodes.numel() == 0:
            raise ValueError("hop_slack_rbf requires at least one slack bus per graph")
        adjacency = [[] for _ in range(n_nodes)]
        for f, t, enabled in zip(
            f_cpu.tolist(), t_cpu.tolist(), status_cpu.tolist()
        ):
            if enabled and 0 <= f < n_nodes and 0 <= t < n_nodes and f != t:
                adjacency[f].append(t)
                adjacency[t].append(f)

        distances = [-1] * n_nodes
        queue = deque()
        for slack in slack_nodes.tolist():
            distances[int(slack)] = 0
            queue.append(int(slack))
        while queue:
            current = queue.popleft()
            next_distance = distances[current] + 1
            for neighbor in adjacency[current]:
                if distances[neighbor] < 0:
                    distances[neighbor] = next_distance
                    queue.append(neighbor)

        finite = [distance for distance in distances if distance >= 0]
        max_finite = max(finite) if finite else 0
        unreachable_distance = max_finite + 1
        distances = [
            distance if distance >= 0 else unreachable_distance
            for distance in distances
        ]
        denominator = max(max(distances), 1)
        normalized = torch.tensor(distances, dtype=torch.float64) / float(denominator)
        centers = torch.linspace(0.0, 1.0, self.range_pe_dim, dtype=torch.float64)
        sigma = 1.0 / max(self.range_pe_dim - 1, 1)
        return torch.exp(
            -0.5 * ((normalized.unsqueeze(-1) - centers) / sigma) ** 2
        )

    def _range_positional_features(
        self,
        bus_type,
        Branch_f_bus,
        Branch_t_bus,
        Branch_status,
        n_nodes_per_graph,
        *,
        dtype,
        device,
    ):
        """Build/cache hop-to-slack RBF features without crossing graph blocks."""
        B, N = bus_type.shape
        if n_nodes_per_graph is None:
            graph_sizes = [N] * B
        else:
            if B != 1:
                raise ValueError("packed RANGE batches require B=1")
            graph_sizes = [
                int(size) for size in n_nodes_per_graph.detach().cpu().tolist()
            ]

        # The campaign uses --share_grid.  Cache the first shared topology once
        # and repeat it over scenarios, avoiding a CPU BFS in every K/batch.
        if self.range_share_grid:
            if len(set(graph_sizes)) != 1:
                raise ValueError("range_share_grid requires identical graph sizes")
            n0 = graph_sizes[0]
            if self._range_pe_cache_cpu is None or self._range_pe_cache_n != n0:
                if n_nodes_per_graph is None:
                    bt = bus_type[0]
                    f = Branch_f_bus[0]
                    t = Branch_t_bus[0]
                    status = Branch_status[0]
                else:
                    bt = bus_type[0, :n0]
                    f_all = Branch_f_bus.reshape(-1)
                    t_all = Branch_t_bus.reshape(-1)
                    status_all = Branch_status.reshape(-1)
                    first_graph = (
                        (f_all >= 0) & (f_all < n0)
                        & (t_all >= 0) & (t_all < n0)
                    )
                    f = f_all[first_graph]
                    t = t_all[first_graph]
                    status = status_all[first_graph]
                self._range_pe_cache_cpu = self._hop_slack_rbf_one(
                    bt, f, t, status
                ).cpu()
                self._range_pe_cache_n = n0
            base = self._range_pe_cache_cpu.to(device=device, dtype=dtype)
            if n_nodes_per_graph is None:
                return base.unsqueeze(0).expand(B, -1, -1)
            return base.repeat(len(graph_sizes), 1).unsqueeze(0)

        features = []
        if n_nodes_per_graph is None:
            for b in range(B):
                features.append(
                    self._hop_slack_rbf_one(
                        bus_type[b], Branch_f_bus[b], Branch_t_bus[b],
                        Branch_status[b]
                    )
                )
            return torch.stack(features, dim=0).to(device=device, dtype=dtype)

        offset = 0
        f_all = Branch_f_bus.reshape(-1)
        t_all = Branch_t_bus.reshape(-1)
        status_all = Branch_status.reshape(-1)
        for size in graph_sizes:
            sl = slice(offset, offset + size)
            in_graph = (
                (f_all >= offset) & (f_all < offset + size)
                & (t_all >= offset) & (t_all < offset + size)
            )
            features.append(
                self._hop_slack_rbf_one(
                    bus_type[0, sl],
                    f_all[in_graph] - offset,
                    t_all[in_graph] - offset,
                    status_all[in_graph],
                )
            )
            offset += size
        return torch.cat(features, dim=0).unsqueeze(0).to(
            device=device, dtype=dtype
        )

    @staticmethod
    def _blend_range_masters(current, candidate, alpha):
        if current is None or candidate is None:
            return current
        if torch.is_tensor(alpha):
            alpha_graph = alpha.to(device=current.device, dtype=current.dtype).reshape(-1)
            if alpha_graph.numel() == 1:
                alpha_graph = alpha_graph.expand(current.shape[0])
        else:
            alpha_graph = current.new_full((current.shape[0],), float(alpha))
        if alpha_graph.numel() != current.shape[0]:
            raise ValueError(
                f"Armijo produced {alpha_graph.numel()} graph alphas for "
                f"{current.shape[0]} RANGE master groups"
            )
        return current + alpha_graph[:, None, None] * (candidate - current)

    def _record_range_diagnostics(self, diagnostics):
        self.last_range_trace.append(
            {key: value.detach() for key, value in diagnostics.items()}
        )

    def _finalize_range_diagnostics(self):
        if not self.last_range_trace:
            self.last_range_diagnostics = None
            return
        scalar_keys = (
            "aggregation_entropy",
            "broadcast_entropy",
            "master_attention_cosine",
            "master_embedding_similarity",
            "effective_masters",
            "self_loop_fraction",
        )
        result = {}
        for key in scalar_keys:
            steps = torch.stack([entry[key] for entry in self.last_range_trace])
            result[f"{key}_mean"] = float(steps.mean().cpu().item())
            result[f"step_{key}"] = steps.cpu().tolist()
        utilization = torch.stack(
            [entry["token_utilization"] for entry in self.last_range_trace]
        )
        result["token_utilization_mean"] = utilization.mean(dim=0).cpu().tolist()
        result["step_token_utilization"] = utilization.cpu().tolist()
        self.last_range_diagnostics = result

    def _record_step_state(self, v, th):
        if self.record_step_states:
            self.last_step_states.append(
                torch.stack([v, th], dim=-1).detach()
            )


    def _project_manifold(self, v, th, n_nodes_per_graph):
        """Constrain the state to the frozen training-split voltage manifold.

        The formula lives in manifold_projection.project_state, shared with the
        scorer, because the post-hoc measurement made with that code is the
        acceptance check for this one and two copies could drift apart.  The
        basis is installed by the driver before epoch 0 and never updated:
        refreshing it during training would let the constraint chase the
        model's own drift instead of the physics.
        """
        if self.manifold_basis is None:
            return v, th
        from manifold_projection import project_state

        return project_state(v, th, self.manifold_basis)

    def _physics_residual_loss(self, DP, DQ, P_set, Q_set, p_mask, q_mask, n_nodes_per_graph,
                               ydiag_abs=None, v=None):
        if self.physics_residual_norm == "local_ybus":
            # Per-bus normalisation by d_i = |Y_ii| |V_i|^2.
            #
            # "graph" divides every bus by one scalar, so on a grid whose
            # diagonal spans five decades the residuals being compared are not
            # commensurate: a stiff bus and a weak one with the same normalised
            # value are physically nothing alike.  Worse, with a saturating
            # loss (log-cosh has gradient tanh, which is +-1 well before the
            # numbers reached here) every large residual then contributes the
            # same gradient, so the objective cannot tell 26 pu from 200 pu.
            # d_i is the natural local scale: it is what dS_i/d|V_i| is
            # proportional to, so a normalised residual of 1 means the same
            # voltage-space error everywhere and the saturation point becomes
            # physically meaningful.  This is the loss-side counterpart of
            # residual_feature_norm="ybus".
            if ydiag_abs is None or v is None:
                raise ValueError(
                    "physics_residual_norm='local_ybus' needs ydiag_abs and v; "
                    "the caller did not pass them."
                )
            scale = (ydiag_abs * v.square()).clamp_min(self.physics_norm_eps)
            residual = torch.cat([
                (DP / scale)[p_mask],
                (DQ / scale)[q_mask],
            ])
            if residual.numel() == 0:
                return DP.sum() * 0.0
            return self._reduce_physics_residual(residual)
        if self.physics_residual_norm == "none":
            if self.physics_loss_form == "mse":
                return (DP ** 2 + DQ ** 2).mean()
            residual = torch.cat([DP.reshape(-1), DQ.reshape(-1)])
        else:
            # Scale on enforced setpoints only. Q at a slack/PV bus is not a
            # constraint -- the solver fixes |V| and solves for Q -- so whatever
            # sits in that slot is arbitrary. CGMES exports park reactive
            # capability figures there: LVN heo1 carries Q_set = -2507 pu at one
            # 380 kV PV bus against a true maximum of 8.4 pu. Unmasked, the
            # "graph" scaler takes amax over the grid, so that single entry
            # would divide the residual at every node in the graph.
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
                    "expected 'none', 'setpoint', 'graph', or 'local_ybus'."
                )

            residual = torch.cat([
                (DP / scale)[p_mask],
                (DQ / scale)[q_mask],
            ])
            if residual.numel() == 0:
                return DP.sum() * 0.0

        return self._reduce_physics_residual(residual)

    def _reduce_physics_residual(self, residual):
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
        vn_log=None,
    ):
        """
        Works for:
          - blockdiag batching (recommended): B=1, global node indices in branch rows
          - plain batching fallback: B>1, each batch item has local branch rows
        """
        device = bus_type.device
        B, N = bus_type.shape

        # reconstruct Y if needed
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
        else:
            if Y.dim() == 2 and not Y.is_sparse:
                Y = Y.unsqueeze(0)

        needs_ydiag = (
            self.residual_feature_norm in ("ybus", "dual")
            or self.edge_feature_norm in ("diagonal", "dual")
            or self.relative_stiffness_feature
            or self.physics_residual_norm == "local_ybus"
        )
        ydiag_abs = None
        ydiag_abs_list = None
        if needs_ydiag:
            with torch.no_grad():
                if B == 1:
                    ydiag_abs = _ybus_diag_abs(Y, N).to(
                        dtype=V0.dtype, device=V0.device
                    )
                else:
                    ydiag_abs_list = [
                        _ybus_diag_abs(Y[b], N).to(
                            dtype=V0.dtype, device=V0.device
                        )
                        for b in range(B)
                    ]

        # -------- build sparse attention graph --------
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
                feature_norm=self.edge_feature_norm,
                ydiag_abs=ydiag_abs,
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
                    feature_norm=self.edge_feature_norm,
                    ydiag_abs=(
                        None if ydiag_abs_list is None else ydiag_abs_list[b]
                    ),
                )
                edge_index_dir_list.append(ei)
                edge_feat_dir_list.append(ef)

        # -------- initialize states --------
        P_set, Q_set = S.real, S.imag
        v = V0[..., 0].clone()
        th = V0[..., 1].clone()
        m = torch.zeros(B, N, self.d, device=device, dtype=V0.dtype)

        slack_mask = (bus_type == 1)
        pv_mask = (bus_type == 2)
        p_mask = ~slack_mask
        q_mask = ~(slack_mask | pv_mask)

        range_masters = None
        range_pe = None
        self.last_range_trace = []
        self.last_range_diagnostics = None
        self.last_step_states = []
        if self.range_context is not None:
            num_graphs = (
                B if n_nodes_per_graph is None else int(n_nodes_per_graph.numel())
            )
            range_masters = self.range_context.initial_state(
                num_graphs, device=device, dtype=V0.dtype
            )
            range_pe = self._range_positional_features(
                bus_type,
                Branch_f_bus,
                Branch_t_bus,
                Branch_status,
                n_nodes_per_graph,
                dtype=V0.dtype,
                device=device,
            )

        phys_terms = []
        dpf_m_vm = torch.zeros_like(v)
        dpf_m_va = torch.zeros_like(th)
        dpf_v_vm = torch.zeros_like(v)
        dpf_v_va = torch.zeros_like(th)
        # Behaviour-neutral bookkeeping: one entry per solver iteration
        # recording what the line search did. The direct path never filled
        # accepted_history, so there was no way to tell an accepted step from a
        # discarded one after the fact.
        self.last_armijo_trace = []
        accepted_history = []
        alpha_history = []
        theta_scale_history = []
        v_scale_history = []
        if self.solver_update_mode != "direct":
            with torch.no_grad():
                integrated_initial_mismatch = _per_graph_mismatch_inf_norm(
                    Y, v, th, P_set, Q_set, slack_mask, pv_mask,
                    n_nodes_per_graph,
                ).detach()

        stiffness_feat = None
        if self.relative_stiffness_feature:
            if B != 1:
                stiffness_feat = torch.stack([
                    self._relative_stiffness(diag, v[b:b + 1], None)[0]
                    for b, diag in enumerate(ydiag_abs_list)
                ], dim=0)
            else:
                stiffness_feat = self._relative_stiffness(
                    ydiag_abs, v, n_nodes_per_graph
                )

        # ------------------------- K iterations -------------------------
        for k in range(self.K):
            if self.manifold_basis is not None and self.manifold_mode == "per_step":
                # Project before the mismatch is formed, so DP/DQ -- which feed
                # both the physics term for this step and the residual features
                # for the next one -- describe a state inside the manifold.
                v, th = self._project_manifold(v, th, n_nodes_per_graph)
            Vc = v * torch.exp(1j * th)
            Ic = ybus_matvec(Y, Vc)
            Sc = Vc * Ic.conj()

            DP = (P_set - Sc.real).masked_fill(slack_mask, 0.0)
            DQ = (Q_set - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)

            grad_vm = None
            grad_va = None
            exact_grad_vm = None
            exact_grad_va = None
            dpf_direction_vm = None
            dpf_direction_va = None
            if self.solver_update_mode != "direct":
                # Exact gradient of the author's masked torch.nn.MSELoss:
                # P at PV/PQ buses and Q at PQ buses. Each block-diagonal grid
                # is scaled independently so batching cannot couple its solve.
                rP = (-DP).to(Vc.dtype)
                rQ = (-DQ).to(Vc.dtype)
                mse_scale = _reference_mse_gradient_scale(
                    p_mask, q_mask, n_nodes_per_graph
                ).to(device=v.device, dtype=v.dtype)
                w = rP * mse_scale + 1j * (rQ * mse_scale)
                g_complex = w * Ic + adjoint_ybus_matvec(Y, w.conj() * Vc)
                phase = torch.exp(1j * th.to(Vc.dtype))
                grad_vm = (g_complex.conj() * phase).real.to(v.dtype)
                grad_va = (g_complex.conj() * (1j * Vc)).real.to(th.dtype)
                grad_vm = grad_vm.masked_fill(slack_mask | pv_mask, 0.0)
                grad_va = grad_va.masked_fill(slack_mask, 0.0)
                exact_grad_vm = grad_vm
                exact_grad_va = grad_va

                if self.preconditioner_optimizer == "author_adam":
                    beta1 = self.preconditioner_beta1
                    beta2 = self.preconditioner_beta2
                    dpf_m_vm = beta1 * dpf_m_vm + (1.0 - beta1) * grad_vm
                    dpf_m_va = beta1 * dpf_m_va + (1.0 - beta1) * grad_va
                    dpf_v_vm = beta2 * dpf_v_vm + (1.0 - beta2) * grad_vm.square()
                    dpf_v_va = beta2 * dpf_v_va + (1.0 - beta2) * grad_va.square()
                    bias1 = 1.0 - beta1 ** (k + 1)
                    bias2 = 1.0 - beta2 ** (k + 1)
                    dpf_direction_vm = (dpf_m_vm / bias1) / (
                        torch.sqrt(dpf_v_vm / bias2) + self.preconditioner_eps
                    )
                    dpf_direction_va = (dpf_m_va / bias1) / (
                        torch.sqrt(dpf_v_va / bias2) + self.preconditioner_eps
                    )
                else:
                    dpf_direction_vm = grad_vm
                    dpf_direction_va = grad_va

                # Only the neural features are normalized; the actual update
                # below remains the reference optimizer direction.
                grad_vm, grad_va = _normalize_per_graph(
                    grad_vm, grad_va, n_nodes_per_graph
                )

            dp_feat, dq_feat, scale_feat, local_dp_feat, local_dq_feat = self._residual_features(
                DP, DQ, P_set, Q_set, p_mask, q_mask, ydiag_abs, v,
                n_nodes_per_graph,
            )
            if self.residual_feature_norm == "dual":
                gate_input = torch.stack([stiffness_feat, scale_feat], dim=-1)
                gate = self.dual_residual_gate(gate_input).squeeze(-1)
                dp_feat = gate * dp_feat + (1.0 - gate) * local_dp_feat
                dq_feat = gate * dq_feat + (1.0 - gate) * local_dq_feat
                self.last_dual_gate_mean = gate.detach().mean()
            bus_parts = [v, th, dp_feat, dq_feat]
            if scale_feat is not None:
                bus_parts.append(scale_feat)
            if stiffness_feat is not None:
                bus_parts.append(stiffness_feat)
            if grad_vm is not None:
                bus_parts.extend([grad_vm, grad_va])
            bus_feat = torch.stack(bus_parts, dim=-1)
            semantic_parts = []
            if self.bus_type_features:
                semantic_parts.append(
                    F.one_hot(
                        (bus_type.long() - 1).clamp(0, 2), num_classes=3
                    ).to(dtype=bus_feat.dtype)
                )
                zero_injection = (
                    (P_set.abs() <= 1e-12) & (Q_set.abs() <= 1e-12)
                ).to(dtype=bus_feat.dtype)
                semantic_parts.append(zero_injection.unsqueeze(-1))
            if self.bus_feat_extra_dim > 0:
                if vn_log is None:
                    extra = bus_feat.new_zeros(bus_feat.shape[:-1] + (self.bus_feat_extra_dim,))
                else:
                    extra = vn_log.unsqueeze(-1)
                semantic_parts.append(extra)
            if semantic_parts:
                bus_feat = torch.cat([bus_feat, *semantic_parts], dim=-1)
            x = self.in_proj(torch.cat([bus_feat, m], dim=-1))
            if self.heterogeneous_injection_features:
                role_scale = self._graph_scale_from_s(
                    torch.sqrt(P_set.square() + Q_set.square()),
                    n_nodes_per_graph,
                )
                gen_role = torch.stack([
                    torch.where(slack_mask | pv_mask, P_set, torch.zeros_like(P_set)),
                    torch.zeros_like(Q_set),
                ], dim=-1)
                # On PQ buses the GridSFM adapter creates a signed load proxy:
                # negative values therefore retain embedded generation instead
                # of silently clipping it away.
                load_role = torch.stack([
                    torch.where(q_mask, -P_set, torch.zeros_like(P_set)),
                    torch.where(q_mask, -Q_set, torch.zeros_like(Q_set)),
                ], dim=-1)
                gen_role = torch.sign(gen_role) * torch.log1p(
                    gen_role.abs() / role_scale.unsqueeze(-1)
                )
                load_role = torch.sign(load_role) * torch.log1p(
                    load_role.abs() / role_scale.unsqueeze(-1)
                )
                x = x + self.generator_role_proj(gen_role) + self.load_role_proj(load_role)

            if self.global_context_mode in ("meanmax_pre", "attn_pre"):
                x = self._apply_global_context(x, n_nodes_per_graph)

            if self.two_hop_attention_mode == "pre":
                if B == 1:
                    x = self._apply_two_hop(x, edge_index_dir, n_nodes_per_graph)
                else:
                    x = torch.cat([
                        self._apply_two_hop(x[b:b + 1], edge_index_dir_list[b])
                        for b in range(B)
                    ], dim=0)

            # sparse attention message passing
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

            if self.two_hop_attention_mode == "post":
                if B == 1:
                    x = self._apply_two_hop(x, edge_index_dir, n_nodes_per_graph)
                else:
                    x = torch.cat([
                        self._apply_two_hop(x[b:b + 1], edge_index_dir_list[b])
                        for b in range(B)
                    ], dim=0)

            if self.global_context_mode == "attn_post":
                x = self._apply_global_context(x, n_nodes_per_graph)

            range_master_candidate = None
            if self.global_context_mode == "range_post":
                x, range_master_candidate, range_diagnostics = self.range_context(
                    x, range_masters, range_pe, n_nodes_per_graph
                )
                self._record_range_diagnostics(range_diagnostics)

            theta_raw = self.theta_head[k](x).squeeze(-1)
            v_raw = self.v_head[k](x).squeeze(-1)
            if self.solver_update_mode == "physics_preconditioner":
                clip = self.preconditioner_log_clip
                theta_scale = self.preconditioner_va_step * torch.exp(
                    torch.clamp(theta_raw, -clip, clip)
                )
                v_scale = self.preconditioner_vm_step * torch.exp(
                    torch.clamp(v_raw, -clip, clip)
                )
                theta_scale_history.append(theta_scale.detach())
                v_scale_history.append(v_scale.detach())
                # PIGNN learns positive coordinate-wise multipliers around the
                # author's DPF optimizer direction. The residual safeguard
                # below accepts or rejects this proposal independently per grid.
                dth = -theta_scale * dpf_direction_va
                dv = -v_scale * dpf_direction_vm
            elif self.solver_update_mode == "physics_projected_hybrid":
                clip = self.preconditioner_log_clip
                theta_preconditioner_raw = self.theta_preconditioner_head[k](x).squeeze(-1)
                v_preconditioner_raw = self.v_preconditioner_head[k](x).squeeze(-1)
                theta_scale = self.preconditioner_va_step * torch.exp(
                    torch.clamp(theta_preconditioner_raw, -clip, clip)
                )
                v_scale = self.preconditioner_vm_step * torch.exp(
                    torch.clamp(v_preconditioner_raw, -clip, clip)
                )
                theta_scale_history.append(theta_scale.detach())
                v_scale_history.append(v_scale.detach())

                # Full-strength PIGNN proposal plus an author-compatible DPF
                # base direction. Unlike the preconditioner-only baseline,
                # gradients into the direct heads are not multiplied by the
                # small 0.003377 DPF learning rate.
                dth = theta_raw - theta_scale * dpf_direction_va
                dv = v_raw - v_scale * dpf_direction_vm
            else:
                dth = theta_raw
                dv = v_raw
            dm  = torch.tanh(self.m_head[k](x))
            dm = F.layer_norm(dm, dm.shape[-1:])

            dth = dth.masked_fill(slack_mask, 0.0)
            dv  = dv.masked_fill(slack_mask | pv_mask, 0.0)

            if self.v_limit:
                v_abs = v.abs()
                dth = torch.clamp(dth, -self.dtheta_max, self.dtheta_max)
                dv = torch.clamp(dv, -self.dvm_frac * v_abs, self.dvm_frac * v_abs)

            if self.solver_update_mode == "physics_projected_hybrid":
                # Clipping individual coordinates can invalidate g^T d < 0.
                # Re-project the feasible proposal, then use one positive
                # scale per packed graph to restore the step limits without
                # changing the direction's descent property.
                dv, dth = _project_to_descent_halfspace(
                    dv, dth, exact_grad_vm, exact_grad_va, n_nodes_per_graph
                )
                if self.v_limit:
                    dv, dth = _scale_direction_to_limits(
                        dv,
                        dth,
                        v,
                        self.dtheta_max,
                        self.dvm_frac,
                        n_nodes_per_graph,
                    )
                dth = dth.masked_fill(slack_mask, 0.0)
                dv = dv.masked_fill(slack_mask | pv_mask, 0.0)

            if self.use_armijo:
                v_min, v_max = 0.75, 1.20

                if self.solver_update_mode != "direct":
                    with torch.no_grad():
                        F0_graph = _per_graph_mismatch_inf_norm(
                            Y, v, th, P_set, Q_set, slack_mask, pv_mask,
                            n_nodes_per_graph, norm=self.armijo_norm,
                        )

                    max_backtracks = max(1, int(self.armijo_max_backtracks))
                    rho = min(max(float(self.armijo_rho), 1e-12), 1.0 - 1e-12)
                    c1 = float(self.armijo_c1)
                    min_alpha = max(0.0, float(self.armijo_min_alpha))
                    alpha_values = []
                    a_tmp = 1.0
                    for _ in range(max_backtracks):
                        alpha_values.append(a_tmp)
                        a_tmp *= rho
                        if min_alpha > 0.0 and a_tmp < min_alpha:
                            break

                    selected = F0_graph.new_zeros(F0_graph.shape)
                    accepted = torch.zeros_like(F0_graph, dtype=torch.bool)
                    for a in alpha_values:
                        v_try = torch.clamp(v + a * dv, v_min, v_max)
                        th_try = (th + a * dth + math.pi) % (2 * math.pi) - math.pi
                        with torch.no_grad():
                            F_try_graph = _per_graph_mismatch_inf_norm(
                                Y, v_try, th_try, P_set, Q_set,
                                slack_mask, pv_mask, n_nodes_per_graph,
                                norm=self.armijo_norm,
                            )
                            ok = F_try_graph <= (1.0 - c1 * a) * F0_graph
                            take = (~accepted) & ok
                            selected = torch.where(
                                take, selected.new_full(selected.shape, a), selected
                            )
                            accepted = accepted | take

                    if self.armijo_mode == "geometric_safe":
                        selected = torch.where(
                            accepted, selected, selected.new_full(selected.shape, min_alpha)
                        )
                    # For the integrated solver, fixed/geometric/reject all
                    # reject a grid if no candidate decreases its physical
                    # residual. This is the inference-time safety property.
                    alpha_node = _expand_graph_values(
                        selected, v, n_nodes_per_graph
                    )
                    accepted_history.append(accepted.detach())
                    alpha_history.append(selected.detach())
                    v = torch.clamp(v + alpha_node * dv, v_min, v_max)
                    th = (th + alpha_node * dth + math.pi) % (2 * math.pi) - math.pi
                    m = m + alpha_node.unsqueeze(-1) * dm
                    range_masters = self._blend_range_masters(
                        range_masters, range_master_candidate, selected
                    )
                    self._record_step_state(v, th)
                    if self.pinn:
                        term = (self.gamma ** (self.K - 1 - k)) * self._physics_residual_loss(
                            DP, DQ, P_set, Q_set, p_mask, q_mask, n_nodes_per_graph,
                            ydiag_abs=ydiag_abs, v=v
                        )
                        phys_terms.append(term)
                    continue

                with torch.no_grad():
                    F0 = _batched_mismatch_inf_norm(Y, v, th, P_set, Q_set, slack_mask, pv_mask,
                                                    norm=self.armijo_norm)

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
                        F_try = _batched_mismatch_inf_norm(Y, v_try, th_try, P_set, Q_set, slack_mask, pv_mask,
                                                           norm=self.armijo_norm)
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
                            range_masters = self._blend_range_masters(
                                range_masters, range_master_candidate, a
                            )
                            accepted = True
                            self.last_armijo_trace.append(("accept", a))
                            break

                    if not accepted:
                        a = float(alphas[-1])
                        v2 = torch.clamp(v + a * dv, v_min, v_max)
                        th2 = (th + a * dth + math.pi) % (2 * math.pi) - math.pi
                        with torch.no_grad():
                            ok = _batched_mismatch_inf_norm(Y, v2, th2, P_set, Q_set, slack_mask, pv_mask,
                                                            norm=self.armijo_norm) < F0
                        if ok:
                            v, th, m = v2, th2, m + a * dm
                            range_masters = self._blend_range_masters(
                                range_masters, range_master_candidate, a
                            )
                            self.last_armijo_trace.append(("fallback", a))
                        else:
                            # Step discarded outright: v, th and m are unchanged,
                            # so this iteration contributes nothing and carries no
                            # gradient.
                            self.last_armijo_trace.append(("discard", 0.0))
                else:
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

                    if accepted:
                        self.last_armijo_trace.append(("accept", a_sel))
                    elif self.armijo_mode == "geometric_safe":
                        self.last_armijo_trace.append(("safe", a_sel))
                    else:
                        self.last_armijo_trace.append(("reject", 0.0))

                    if self.armijo_mode == "reject" and not accepted:
                        pass
                    else:
                        # For geometric, match the LVN/student Armijo semantics:
                        # the line search only selects a scalar step size. In
                        # geometric_safe, rejected searches use a tiny configured
                        # fallback to keep gradients alive while staying near old
                        # PPC. In reject mode, rejected searches discard the
                        # update entirely.
                        v = torch.clamp(v + a_sel * dv, v_min, v_max)
                        th = (th + a_sel * dth + math.pi) % (2 * math.pi) - math.pi
                        m = m + a_sel * dm
                        range_masters = self._blend_range_masters(
                            range_masters, range_master_candidate, a_sel
                        )
            else:
                th = (th + dth + math.pi) % (2 * math.pi) - math.pi
                v = torch.clamp(v + dv, 0.75, 1.20)
                m = m + dm
                range_masters = self._blend_range_masters(
                    range_masters, range_master_candidate, 1.0
                )

            self._record_step_state(v, th)

            if self.pinn:
                term = (self.gamma ** (self.K - 1 - k)) * self._physics_residual_loss(
                    DP, DQ, P_set, Q_set, p_mask, q_mask, n_nodes_per_graph,
                    ydiag_abs=ydiag_abs, v=v
                )
                phys_terms.append(term)

        self._finalize_range_diagnostics()
        # Emit a projected state in both modes.  Under "terminal" this is the
        # only projection; under "per_step" it also catches the final update,
        # which happens after the last in-loop projection.  The operator is
        # idempotent, so applying it twice to an already-projected state is a
        # no-op rather than a distortion.
        v, th = self._project_manifold(v, th, n_nodes_per_graph)
        out = torch.stack([v, th], dim=-1)

        if self.solver_update_mode != "direct":
            with torch.no_grad():
                integrated_final_mismatch = _per_graph_mismatch_inf_norm(
                    Y, v, th, P_set, Q_set, slack_mask, pv_mask,
                    n_nodes_per_graph,
                ).detach()
                accepted_tensor = torch.stack(accepted_history) if accepted_history else None
                alpha_tensor = torch.stack(alpha_history) if alpha_history else None
                theta_scales = torch.stack(theta_scale_history)
                v_scales = torch.stack(v_scale_history)
                self.last_solver_diagnostics = {
                    "initial_max_mismatch": integrated_initial_mismatch,
                    "final_max_mismatch": integrated_final_mismatch,
                    "accepted_fraction": (
                        accepted_tensor.float().mean(dim=0)
                        if accepted_tensor is not None else integrated_final_mismatch.new_zeros(
                            integrated_final_mismatch.shape
                        )
                    ),
                    "mean_alpha": (
                        alpha_tensor.mean(dim=0)
                        if alpha_tensor is not None else integrated_final_mismatch.new_zeros(
                            integrated_final_mismatch.shape
                        )
                    ),
                    "theta_scale_mean": theta_scales.mean(),
                    "theta_scale_min": theta_scales.amin(),
                    "theta_scale_max": theta_scales.amax(),
                    "v_scale_mean": v_scales.mean(),
                    "v_scale_min": v_scales.amin(),
                    "v_scale_max": v_scales.amax(),
                }
        else:
            self.last_solver_diagnostics = None

        if self.pinn:
            if self.physics_final_weight != 0.0:
                Vc = v * torch.exp(1j * th)
                Ic = ybus_matvec(Y, Vc)
                Sc = Vc * Ic.conj()
                DP = (P_set - Sc.real).masked_fill(slack_mask, 0.0)
                DQ = (Q_set - Sc.imag).masked_fill(slack_mask | pv_mask, 0.0)
                final_term = self._physics_residual_loss(
                    DP, DQ, P_set, Q_set, p_mask, q_mask, n_nodes_per_graph,
                    ydiag_abs=ydiag_abs, v=v
                )
                phys_terms.append(self.physics_final_weight * final_term)
            phys_loss = torch.sum(torch.stack(phys_terms))
            return out, phys_loss
        else:
            return out
