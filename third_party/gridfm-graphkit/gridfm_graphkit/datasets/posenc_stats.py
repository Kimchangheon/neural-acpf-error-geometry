import torch

from torch_geometric.utils import (
    to_dense_adj,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes

try:
    from torch_scatter import scatter_add
except ImportError:
    scatter_add = None

from functools import partial
from gridfm_graphkit.datasets.rrwp import add_full_rrwp, add_topk_rrwp

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData
from typing import Any


def compute_posenc_stats(data, pe_types, cfg):
    """Precompute positional encodings for the given graph.
    Supported PE statistics to precompute in original implementation,
    selected by `pe_types`:
    'LapPE': Laplacian eigen-decomposition.
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    'HKfullPE': Full heat kernels and their diagonals. (NOT IMPLEMENTED)
    'HKdiagSE': Diagonals of heat kernel diffusion.
    'ElstaticSE': Kernel based on the electrostatic interaction between nodes.
    'RRWP': Relative Random Walk Probabilities PE (Ours, for GRIT)
    Args:
        data: PyG graph
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination, e.g. 'eigen+rw_landing'
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    # Verify PE types.
    for t in pe_types:
        if t not in [
            "LapPE",
            "EquivStableLapPE",
            "SignNet",
            "RWSE",
            "HKdiagSE",
            "HKfullPE",
            "ElstaticSE",
            "RRWP",
        ]:
            raise ValueError(f"Unexpected PE stats selection {t} in {pe_types}")

    if "RRWP" in pe_types:
        param = cfg.posenc_RRWP
        topk = getattr(param, "topk", 0)
        if topk and topk > 0:
            transform = partial(
                add_topk_rrwp,
                walk_length=param.ksteps,
                topk=topk,
                attr_name_abs="rrwp",
                attr_name_rel="rrwp",
                add_identity=True,
            )
        else:
            transform = partial(
                add_full_rrwp,
                walk_length=param.ksteps,
                attr_name_abs="rrwp",
                attr_name_rel="rrwp",
                add_identity=True,
            )
        data = transform(data)

    # Random Walks.
    if "RWSE" in pe_types:
        kernel_param = cfg.posenc_RWSE.kernel
        if hasattr(data, "num_nodes"):
            N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
        else:
            N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
        if kernel_param.times == 0:
            raise ValueError("List of kernel times required for RWSE")
        rw_landing = get_rw_landing_probs(
            ksteps=[xx + 1 for xx in range(kernel_param.times)],
            edge_index=data.edge_index,
            num_nodes=N,
        )
        data.pestat_RWSE = rw_landing

    return data


def get_rw_landing_probs(
    ksteps,
    edge_index,
    edge_weight=None,
    num_nodes=None,
    space_dim=0,
):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, _ = edge_index[0], edge_index[1]
    if scatter_add is None:
        raise ImportError(
            "torch-scatter is required for RWSE positional encodings. "
            "Install it with: pip install torch-scatter",
        )
    deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)  # Out degrees.
    deg_inv = deg.pow(-1.0)
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(
            edge_index,
            max_num_nodes=num_nodes,
        )  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(
                torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1)
                * (k ** (space_dim / 2)),
            )
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


class ComputePosencStat(BaseTransform):
    def __init__(self, pe_types, cfg):
        self.pe_types = pe_types
        self.cfg = cfg

    def forward(self, data: Any) -> Any:
        pass

    def __call__(self, data) -> Data:
        if isinstance(data, HeteroData):
            return self._call_hetero(data)

        data = compute_posenc_stats(data, pe_types=self.pe_types, cfg=self.cfg)
        return data

    def _call_hetero(self, data: HeteroData) -> HeteroData:
        """Compute PE on the bus-only subgraph and store results on data['bus']."""
        bus_data = Data(
            x=data["bus"].x,
            edge_index=data["bus", "connects", "bus"].edge_index,
            num_nodes=data["bus"].num_nodes,
        )
        if hasattr(data["bus", "connects", "bus"], "edge_weight"):
            bus_data.edge_weight = data["bus", "connects", "bus"].edge_weight

        bus_data = compute_posenc_stats(
            bus_data,
            pe_types=self.pe_types,
            cfg=self.cfg,
        )

        # Copy computed PE attributes back onto the HeteroData.
        # Node-level attrs go on the bus store; the sparse RRWP index/val
        # are stored as a dedicated edge type so PyG batches them correctly
        # (edge_index needs cat_dim=1 + node-count incrementing).
        node_pe_attrs = [
            "pestat_RWSE",  # RWSE
            "rrwp",  # RRWP absolute (node-level diagonal)
            "log_deg",
            "deg",  # degree info from RRWP
        ]
        for attr in node_pe_attrs:
            if hasattr(bus_data, attr):
                data["bus"][attr] = getattr(bus_data, attr)

        # RRWP relative PE: sparse [2, E] index + [E, K] values → edge type
        if hasattr(bus_data, "rrwp_index") and hasattr(bus_data, "rrwp_val"):
            data["bus", "rrwp", "bus"].edge_index = bus_data.rrwp_index
            data["bus", "rrwp", "bus"].edge_attr = bus_data.rrwp_val

        return data
