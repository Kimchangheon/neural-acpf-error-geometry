"""
The RRWP encoder for GRIT (ours)
"""

import torch
from torch import nn

try:
    import torch_sparse
except ImportError:
    torch_sparse = None

from torch_geometric.utils import (
    add_self_loops,
)

try:
    from torch_scatter import scatter
except ImportError:
    scatter = None

import warnings

_MISSING_MSG = (
    "{pkg} is required for the RRWP / GRIT modules but is not installed. "
    "Install it with: pip install {pkg}"
)


def _check_sparse():
    if torch_sparse is None:
        raise ImportError(_MISSING_MSG.format(pkg="torch-sparse"))


def _check_scatter():
    if scatter is None:
        raise ImportError(_MISSING_MSG.format(pkg="torch-scatter"))


def full_edge_index(edge_index, batch=None):
    """
    Return the Full batched sparse adjacency matrices given by edge indices.
    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.
    Implementation inspired by `torch_geometric.utils.to_dense_adj`
    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.
    Returns:
        Complementary edge index.
    """

    _check_scatter()

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce="add")
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short, device=edge_index.device)

        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        # _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_full = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_full


class RRWPLinearNodeEncoder(torch.nn.Module):
    """
    FC_1(RRWP) + FC_2 (Node-attr)
    note: FC_2 is given by the Typedict encoder of node-attr in some cases
    Parameters:
    num_classes - the number of classes for the embedding mapping to learn
    """

    def __init__(
        self,
        emb_dim,
        out_dim,
        use_bias=False,
        batchnorm=False,
        layernorm=False,
        pe_name="rrwp",
    ):
        super().__init__()
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.name = pe_name

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        rrwp = batch[f"{self.name}"]
        rrwp = self.fc(rrwp)

        if self.batchnorm:
            rrwp = self.bn(rrwp)

        if self.layernorm:
            rrwp = self.ln(rrwp)

        if "x" in batch:
            batch.x = batch.x + rrwp
        else:
            batch.x = rrwp

        return batch


class RRWPLinearEdgeEncoder(torch.nn.Module):
    """
    Merge RRWP with given edge-attr and Zero-padding to all pairs of node
    FC_1(RRWP) + FC_2(edge-attr)
    - FC_2 given by the TypedictEncoder in same cases
    - Zero-padding for non-existing edges in fully-connected graph
    - (optional) add node-attr as the E_{i,i}'s attr
        note: assuming  node-attr and edge-attr is with the same dimension after Encoders
    """

    def __init__(
        self,
        emb_dim,
        out_dim,
        batchnorm=False,
        layernorm=False,
        use_bias=False,
        pad_to_full_graph=True,
        fill_value=0.0,
        add_node_attr_as_self_loop=False,
        overwrite_old_attr=False,
    ):
        super().__init__()
        # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
        self.overwrite_old_attr = overwrite_old_attr  # remove the old edge-attr

        self.batchnorm = batchnorm
        self.layernorm = layernorm
        if self.batchnorm or self.layernorm:
            warnings.warn(
                "batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info ",
            )

        # print('--------fc in and out:', emb_dim, out_dim)
        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.pad_to_full_graph = pad_to_full_graph
        self.fill_value = fill_value

        padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
        self.register_buffer("padding", padding)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        rrwp_idx = batch.rrwp_index
        rrwp_val = batch.rrwp_val
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        rrwp_val = self.fc(rrwp_val)

        if edge_attr is None:
            edge_attr = edge_index.new_zeros(edge_index.size(1), rrwp_val.size(1))
            # zero padding for non-existing edges

        if self.overwrite_old_attr:
            out_idx, out_val = rrwp_idx, rrwp_val
        else:
            edge_index, edge_attr = add_self_loops(
                edge_index,
                edge_attr,
                num_nodes=batch.num_nodes,
                fill_value=0.0,
            )
            _check_sparse()
            out_idx, out_val = torch_sparse.coalesce(
                torch.cat([edge_index, rrwp_idx], dim=1),
                torch.cat([edge_attr, rrwp_val], dim=0),
                batch.num_nodes,
                batch.num_nodes,
                op="add",
            )

        if self.pad_to_full_graph:
            edge_index_full = full_edge_index(out_idx, batch=batch.batch)
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            # zero padding to fully-connected graphs
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            _check_sparse()
            out_idx, out_val = torch_sparse.coalesce(
                out_idx,
                out_val,
                batch.num_nodes,
                batch.num_nodes,
                op="add",
            )

        if self.batchnorm:
            out_val = self.bn(out_val)

        if self.layernorm:
            out_val = self.ln(out_val)

        batch.edge_index, batch.edge_attr = out_idx, out_val
        return batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(pad_to_full_graph={self.pad_to_full_graph},"
            f"fill_value={self.fill_value},"
            f"{self.fc.__repr__()})"
        )
