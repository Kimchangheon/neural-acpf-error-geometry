from typing import Dict, List, Union
import torch


# --------------------------------------------------------------------------- #
# Fields
# --------------------------------------------------------------------------- #

_VECTOR_FIELDS = [
    "bus_type",
    "vn_kv",
    "vn_log",
    "V_base_bus",
    "Y_shunt_bus",
    "U_start",
    "U_newton",
    "S_start",
    "S_newton",
    # AC-OPF decision space (per bus). Absent from the power-flow parquets;
    # _concat_present skips any field a sample does not carry.
    "S_load",
    "S_gen_opf",
    "Gen_p_min",
    "Gen_p_max",
    "Gen_q_min",
    "Gen_q_max",
    "Gen_p_avail",
    "Gen_cost_c1",
    "Gen_controllable",
    "Gen_box_injection",
    "Bus_vmin",
    "Bus_vmax",
]

_SCALAR_FIELDS = [
    "S_base",
    "U_base",
]

_TWO_CHANNEL_FIELDS = [
    "V_start",
    "V_newton",
]

_BRANCH_FIELDS = [
    "Branch_status",
    "Branch_tau",
    "Branch_shift_deg",
    "Branch_y_series_from",
    "Branch_y_series_to",
    "Branch_y_series_ft",
    "Branch_y_shunt_from",
    "Branch_y_shunt_to",
    "Is_trafo",
    "Branch_hv_is_f",
    "Branch_n",
    "Y_Lines",
    "Y_C_Lines",
    "Branch_rate_a",
]

_BRANCH_INDEX_FIELDS = [
    "Branch_f_bus",
    "Branch_t_bus",
]

_MATRIX_FIELDS = [
    "Ybus",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _concat_present(samples, fields, dim=0):
    out = {}
    for f in fields:
        vals = [s.get(f, None) for s in samples]
        if all(isinstance(v, torch.Tensor) for v in vals):
            out[f] = torch.cat(vals, dim=dim)
    return out


# --------------------------------------------------------------------------- #
# Y-bus: sparse block diagonal
# --------------------------------------------------------------------------- #

# The block-diagonal Y-bus is the memory wall in this pipeline. Dense, it costs
# (B*N)^2 * 16 bytes -- quadratic in batch*buses -- so case9241pegase at batch 8
# asks for 81 GB and is OOM-killed, while the matrix is 99.98% zeros: a 9241-bus
# grid has ~16k branches, i.e. ~41k nonzeros against 85M entries.
#
# Y is only ever used for the matvec Y @ V (opf_task.implied_injection and the
# two physics-loss sites in train_valid_test_gridfm), so a sparse COO layout is
# numerically identical and turns the cost into O(nnz * B).

def block_diag_sparse(mats: List[torch.Tensor]) -> torch.Tensor:
    """Sparse-COO block diagonal of dense or sparse square matrices."""
    idx_chunks, val_chunks = [], []
    off = 0
    for m in mats:
        n = m.shape[0]
        # PyTorch <1.12 exposes the same COO conversion as ``to_sparse``.
        sp = m if m.is_sparse else (
            m.to_sparse_coo() if hasattr(m, "to_sparse_coo") else m.to_sparse()
        )
        sp = sp.coalesce()
        idx_chunks.append(sp.indices() + off)
        val_chunks.append(sp.values())
        off += n
    if not idx_chunks:
        empty = torch.zeros((0, 0), dtype=torch.complex128)
        return empty.to_sparse_coo() if hasattr(empty, "to_sparse_coo") else empty.to_sparse()
    indices = torch.cat(idx_chunks, dim=1)
    values = torch.cat(val_chunks, dim=0)
    return torch.sparse_coo_tensor(indices, values, (off, off),
                                   dtype=values.dtype).coalesce()


def ybus_matvec(Y: torch.Tensor, Vc: torch.Tensor) -> torch.Tensor:
    """Y @ Vc for dense or sparse Y, accepting Vc as [N] or [1, N].

    Returns the same leading shape it was given, so call sites do not have to
    know which layout the collate produced.
    """
    squeeze_batch = False
    if Vc.dim() == 2 and Vc.shape[0] == 1:
        Vc = Vc.squeeze(0)
        squeeze_batch = True
    Ym = Y
    if Ym.is_sparse:
        if Ym.dim() > 2:
            # Rebuild rather than squeeze: as_strided is unimplemented for the
            # sparse CUDA backend, so squeeze() raises there.
            if Ym.shape[0] != 1:
                raise ValueError(f"sparse Y-bus with leading dim {Ym.shape[0]} != 1")
            c = Ym.coalesce()
            Ym = torch.sparse_coo_tensor(c.indices()[1:], c.values(),
                                         tuple(Ym.shape[1:])).coalesce()
        out = torch.sparse.mm(Ym, Vc.unsqueeze(-1)).squeeze(-1)
    else:
        if Ym.dim() > 2:                  # dense collate adds a batch axis
            Ym = Ym.squeeze(0)
        out = torch.matmul(Ym, Vc.unsqueeze(-1)).squeeze(-1)
    return out.unsqueeze(0) if squeeze_batch else out


# --------------------------------------------------------------------------- #
# main API
# --------------------------------------------------------------------------- #

def collate_blockdiag(samples: List[Dict[str, torch.Tensor]],
                      sparse_ybus: bool = True) -> Dict[str, Union[torch.Tensor, None]]:
    """
    Collate per-sample graphs into one block-diagonal mega-graph.

    Important:
      - bus-index branch fields are offset to global node numbering
      - Ybus is block-diagonalized
      - everything gets an explicit batch dimension B=1
    """
    out: Dict[str, Union[torch.Tensor, None]] = {}

    device_for_meta = None
    for s in samples:
        for v in s.values():
            if isinstance(v, torch.Tensor):
                device_for_meta = v.device
                break
        if device_for_meta is not None:
            break
    if device_for_meta is None:
        device_for_meta = torch.device("cpu")

    sizes = torch.tensor([int(s["N"]) for s in samples], device=device_for_meta, dtype=torch.long)
    branch_sizes = torch.tensor([int(s["nl"]) for s in samples], device=device_for_meta, dtype=torch.long)
    offsets = torch.cat((sizes.new_zeros(1), torch.cumsum(sizes, 0)[:-1]))

    out.update(_concat_present(samples, _VECTOR_FIELDS, dim=0))
    out.update(_concat_present(samples, _TWO_CHANNEL_FIELDS, dim=0))
    out.update(_concat_present(samples, _BRANCH_FIELDS, dim=0))

    # Variable-length Newton trajectories are packed along the same block-
    # diagonal node axis as the ordinary voltage tensors.  They deliberately
    # retain shape [T, total_nodes, ...] (no synthetic leading batch axis), and
    # the transition mask selects graphs that actually own step t -> t+1.
    have_trajectories = all(
        isinstance(s.get("V_nr_trajectory", None), torch.Tensor)
        and isinstance(s.get("U_nr_trajectory", None), torch.Tensor)
        for s in samples
    ) and len(samples) > 0
    if have_trajectories:
        lengths = torch.tensor(
            [int(s["V_nr_trajectory"].shape[0]) for s in samples],
            device=device_for_meta,
            dtype=torch.long,
        )
        max_states = int(lengths.max().item())
        total_nodes = int(sizes.sum().item())
        v_ref = samples[0]["V_nr_trajectory"]
        u_ref = samples[0]["U_nr_trajectory"]
        v_path = torch.zeros(
            max_states, total_nodes, 2, dtype=v_ref.dtype, device=v_ref.device
        )
        u_path = torch.zeros(
            max_states, total_nodes, dtype=u_ref.dtype, device=u_ref.device
        )
        state_mask = torch.zeros(
            max_states, total_nodes, dtype=torch.bool, device=device_for_meta
        )
        for sample, off, size, length in zip(samples, offsets, sizes, lengths):
            begin = int(off.item())
            end = begin + int(size.item())
            steps = int(length.item())
            v_path[:steps, begin:end] = sample["V_nr_trajectory"]
            u_path[:steps, begin:end] = sample["U_nr_trajectory"]
            state_mask[:steps, begin:end] = True
            # Pad inactive timesteps with the converged state. This prevents
            # meaningless residual spikes if a caller evaluates all packed
            # graphs before applying nr_transition_mask.
            if steps < max_states:
                v_path[steps:, begin:end] = sample["V_nr_trajectory"][-1]
                u_path[steps:, begin:end] = sample["U_nr_trajectory"][-1]
        out["V_nr_trajectory"] = v_path
        out["U_nr_trajectory"] = u_path
        out["nr_trajectory_lengths"] = lengths
        out["nr_trajectory_state_mask"] = state_mask
        out["nr_transition_mask"] = state_mask[1:] & state_mask[:-1]

    for name in _SCALAR_FIELDS:
        vals = [s.get(name, None) for s in samples]
        if all(isinstance(v, torch.Tensor) for v in vals):
            out[name] = torch.stack([v.reshape(()) for v in vals], dim=0)

    for name in _BRANCH_INDEX_FIELDS:
        vals = []
        for s, off in zip(samples, offsets):
            vals.append(s[name] + off)
        out[name] = torch.cat(vals, dim=0)

    have_all_y = all(isinstance(s.get("Ybus", None), torch.Tensor) for s in samples)
    if have_all_y and len(samples) > 0:
        mats = [s["Ybus"] for s in samples]
        out["Ybus"] = block_diag_sparse(mats) if sparse_ybus else torch.block_diag(*mats)
    else:
        out["Ybus"] = None

    for k, v in list(out.items()):
        # A sparse Y-bus keeps its plain [N, N] shape: unsqueeze on sparse COO
        # is awkward and every consumer goes through ybus_matvec, which accepts
        # either layout.
        if (
            isinstance(v, torch.Tensor)
            and k not in _SCALAR_FIELDS
            and k not in {
                "V_nr_trajectory", "U_nr_trajectory",
                "nr_trajectory_lengths", "nr_trajectory_state_mask",
                "nr_transition_mask",
            }
            and not v.is_sparse
        ):
            out[k] = v.unsqueeze(0)

    out["sizes"] = sizes
    out["branch_sizes"] = branch_sizes
    out["offsets"] = offsets

    return out
