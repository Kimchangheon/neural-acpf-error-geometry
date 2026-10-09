from typing import Any, Optional
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

try:
    from torch_sparse import SparseTensor
except ImportError:
    SparseTensor = None


def add_node_attr(data: Data, value: Any, attr_name: Optional[str] = None) -> Data:
    if attr_name is None:
        if "x" in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data


@torch.no_grad()
def add_full_rrwp(
    data,
    walk_length=8,
    attr_name_abs="rrwp",  # name: 'rrwp'
    attr_name_rel="rrwp",  # name: ('rrwp_idx', 'rrwp_val')
    add_identity=True,
    spd=False,
    **kwargs,
):
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    if SparseTensor is None:
        raise ImportError(
            "torch-sparse is required for RRWP positional encodings. "
            "Install it with: pip install torch-sparse",
        )

    adj = SparseTensor.from_edge_index(
        edge_index,
        edge_weight,
        sparse_sizes=(num_nodes, num_nodes),
    )

    # Compute D^{-1} A:
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / deg
    deg_inv[deg_inv == float("inf")] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float))
        i = i + 1

    out = adj
    pe_list.append(adj)

    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1)  # n x n x k

    abs_pe = pe.diagonal().transpose(0, 1)  # n x k

    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    # rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)
    rel_pe_idx = torch.stack([rel_pe_col, rel_pe_row], dim=0)
    # the framework of GRIT performing right-mul while adj is row-normalized,
    #                 need to switch the order or row and col.
    #    note: both can work but the current version is more reasonable.

    if spd:
        spd_idx = walk_length - torch.arange(walk_length)
        val = (rel_pe_val > 0).type(torch.float) * spd_idx.unsqueeze(0)
        val = torch.argmax(val, dim=-1)
        rel_pe_val = F.one_hot(val, walk_length).type(torch.float)
        abs_pe = torch.zeros_like(abs_pe)

    data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
    data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_rel}_val")
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)

    return data


@torch.no_grad()
def add_topk_rrwp(
    data,
    walk_length=8,
    topk=10,
    attr_name_abs="rrwp",
    attr_name_rel="rrwp",
    add_identity=True,
    spd=False,
    **kwargs,
):
    """Compute RRWP positional encodings with Top-K sparsification.

    Instead of retaining the full N×N relative PE matrix, this function
    keeps only the `topk` highest-magnitude neighbors per node (based on
    the L2 norm of the multi-step random walk probability vector).  The
    original graph edges are always retained regardless of their rank.

    This provides a smooth interpolation between:
      - topk=0 (or topk >= N): equivalent to full RRWP (all pairs)
      - topk=1: nearly equivalent to RWSE (mostly self-loops / local)

    Args:
        data: PyG Data object with edge_index.
        walk_length: Number of random walk steps (k).
        topk: Number of highest-ranked neighbors to retain per node.
            If 0 or >= num_nodes, all edges are kept (full RRWP).
        attr_name_abs: Attribute name for the absolute (diagonal) PE.
        attr_name_rel: Prefix for relative PE index/val attributes.
        add_identity: Whether to include the identity (step 0) in PE.
        spd: If True, encode shortest-path distance instead of probabilities.

    Returns:
        Data object with rrwp, rrwp_index, rrwp_val, log_deg, deg attributes.
    """
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    if SparseTensor is None:
        raise ImportError(
            "torch-sparse is required for RRWP positional encodings. "
            "Install it with: pip install torch-sparse",
        )

    adj = SparseTensor.from_edge_index(
        edge_index,
        edge_weight,
        sparse_sizes=(num_nodes, num_nodes),
    )

    # Compute D^{-1} A:
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / deg
    deg_inv[deg_inv == float("inf")] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float))
        i = i + 1

    out = adj
    pe_list.append(adj)

    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1)  # n x n x k

    abs_pe = pe.diagonal().transpose(0, 1)  # n x k

    if spd:
        spd_idx = walk_length - torch.arange(walk_length)
        val = (pe > 0).type(torch.float) * spd_idx.unsqueeze(0).unsqueeze(0)
        val = torch.argmax(val, dim=-1)
        pe = F.one_hot(val, walk_length).type(torch.float)
        abs_pe = torch.zeros_like(abs_pe)

    # --- Top-K sparsification ---
    # If topk <= 0 or topk >= num_nodes, keep everything (full RRWP)
    if topk <= 0 or topk >= num_nodes:
        rel_pe = SparseTensor.from_dense(pe, has_value=True)
        rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
        rel_pe_idx = torch.stack([rel_pe_col, rel_pe_row], dim=0)
    else:
        # Score each (i,j) pair by L2 norm of the k-step probability vector
        # pe shape: [n, n, k] — pe[i, j, :] is the walk vector from i to j
        scores = pe.norm(dim=-1)  # [n, n]

        # Always include original graph edges (set their scores to infinity)
        edge_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
        edge_mask[edge_index[0], edge_index[1]] = True
        # Also always include self-loops
        diag_idx = torch.arange(num_nodes)
        edge_mask[diag_idx, diag_idx] = True

        # For Top-K selection: get top-k scores per row (per source node)
        # Clamp topk to at most num_nodes
        k = min(topk, num_nodes)
        _, topk_indices = scores.topk(k, dim=1)  # [n, k]

        # Build a mask of selected entries
        topk_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
        row_idx = torch.arange(num_nodes).unsqueeze(1).expand_as(topk_indices)
        topk_mask[row_idx, topk_indices] = True

        # Union of top-k and original edges
        keep_mask = topk_mask | edge_mask

        # Extract sparse entries from the masked pe tensor
        # Zero out entries we don't want, then sparsify
        pe_sparse = pe.clone()
        pe_sparse[~keep_mask] = 0

        rel_pe = SparseTensor.from_dense(pe_sparse, has_value=True)
        rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
        rel_pe_idx = torch.stack([rel_pe_col, rel_pe_row], dim=0)

    data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
    data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_rel}_val")
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)

    return data
