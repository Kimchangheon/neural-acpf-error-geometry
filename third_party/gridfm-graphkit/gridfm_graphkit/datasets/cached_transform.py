import os
import tempfile
import hashlib

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import remove_self_loops


def _topology_fingerprint(
    data,
    cached_edge_type=None,
    use_admittance=False,
    admittance_remove_self_loops=True,
):
    """Compute a short hash that uniquely identifies the graph topology.

    The fingerprint captures everything that determines the positional
    encoding output: edge connectivity, number of nodes, and (when
    admittance-weighted) the edge weights derived from admittance values.

    This allows samples sharing the same topology (and admittance values)
    to reuse a single cached PE file instead of storing redundant copies.

    Args:
        data: HeteroData or Data object.
        cached_edge_type: Edge type tuple for hetero data, or None.
        use_admittance: Whether admittance weights affect the PE.
        admittance_remove_self_loops: Whether self-loops are removed
            when computing admittance weights.

    Returns:
        A 16-character hex string uniquely identifying the topology.
    """
    if isinstance(data, HeteroData):
        edge_index = data["bus", "connects", "bus"].edge_index
        num_nodes = data["bus"].num_nodes
        edge_attr = data["bus", "connects", "bus"].edge_attr
    else:
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        edge_attr = getattr(data, "edge_attr", None)

    h = hashlib.sha256()
    h.update(
        num_nodes.to_bytes(4, "little")
        if isinstance(num_nodes, int)
        else int(num_nodes).to_bytes(4, "little"),
    )
    h.update(edge_index.cpu().numpy().tobytes())

    # Include admittance values in the fingerprint when they affect the PE
    if use_admittance and edge_attr is not None:
        if edge_attr.size(1) == 2:
            g, b = edge_attr[:, 0], edge_attr[:, 1]
        else:
            # Yft at indices 4, 5
            g, b = edge_attr[:, 4], edge_attr[:, 5]
        edge_weight = torch.sqrt(g**2 + b**2)

        if admittance_remove_self_loops:
            _, edge_weight = remove_self_loops(edge_index, edge_weight)

        # Quantize to 6 decimal places to avoid floating-point noise
        # causing spurious cache misses
        quantized = (edge_weight * 1e6).round().to(torch.int64)
        h.update(quantized.cpu().numpy().tobytes())

    return h.hexdigest()[:16]


class CachedPosencTransform:
    """Disk-caching wrapper for positional encoding transforms.

    Computes the PE on the first access and caches the result to disk.
    Subsequent accesses with the same graph topology load from cache,
    avoiding redundant computation across epochs, splits, and jobs.

    Cache keying strategy:
        PE depends only on graph topology (edge_index, num_nodes) and
        optionally on admittance values — NOT on node features like
        loads/generation.  The cache is keyed by a hash of these
        topology-determining inputs so that all samples sharing the same
        network structure reuse a single cache file.

    Thread/process safety:
        - Uses atomic write (write to temp file, then os.replace) so
          concurrent DataLoader workers or separate jobs cannot produce
          corrupt cache files.
        - Different train/val/test splits across jobs safely share the
          cache since PE depends only on topology, not on split
          membership.

    Args:
        transform: The inner PE transform (e.g. ComputePosencStat).
        cache_dir: Directory to store cached PE tensors.
        cached_attrs: List of attribute names to cache on the bus node store
            (e.g. ["pestat_RWSE"]).
        cached_edge_type: Optional edge type tuple (e.g. ("bus", "rrwp", "bus"))
            whose edge_index and edge_attr should also be cached.
        key_attr: Attribute on the data object used as the cache key.
            If "topology" (default), uses a hash of the graph structure
            so samples with identical topology share one cache file.
            Otherwise, uses the named scalar attribute (e.g. "scenario_id")
            for per-sample caching.
        use_admittance: Whether admittance weights are used in the PE
            computation (affects the topology fingerprint).
        admittance_remove_self_loops: Whether self-loops are removed
            for admittance weighting (affects the topology fingerprint).
    """

    def __init__(
        self,
        transform,
        cache_dir: str,
        cached_attrs: list[str],
        cached_edge_type: tuple[str, str, str] | None = None,
        key_attr: str = "topology",
        use_admittance: bool = False,
        admittance_remove_self_loops: bool = True,
    ):
        self.transform = transform
        self.cache_dir = cache_dir
        self.cached_attrs = cached_attrs
        self.cached_edge_type = cached_edge_type
        self.key_attr = key_attr
        self.use_admittance = use_admittance
        self.admittance_remove_self_loops = admittance_remove_self_loops
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, data) -> str:
        if self.key_attr == "topology":
            key = _topology_fingerprint(
                data,
                cached_edge_type=self.cached_edge_type,
                use_admittance=self.use_admittance,
                admittance_remove_self_loops=self.admittance_remove_self_loops,
            )
        else:
            key = data[self.key_attr].item()
        return os.path.join(self.cache_dir, f"pe_cache_{key}.pt")

    def _load_cache(self, cache_path, data):
        """Load cached PE attributes and attach them to data."""
        cached = torch.load(cache_path, weights_only=True)
        if isinstance(data, HeteroData):
            for attr, val in cached.items():
                if attr == "_edge_type_index":
                    data[self.cached_edge_type].edge_index = val
                elif attr == "_edge_type_attr":
                    data[self.cached_edge_type].edge_attr = val
                else:
                    data["bus"][attr] = val
        else:
            for attr, val in cached.items():
                setattr(data, attr, val)

    def _save_cache(self, cache_path, data):
        """Atomically save PE attributes to the cache file.

        Uses a temporary file in the same directory followed by
        os.replace, which is atomic on both Linux and Windows.
        This ensures concurrent workers/jobs never see a partially
        written file.
        """
        if isinstance(data, HeteroData):
            target = data["bus"]
        else:
            target = data

        to_cache = {}
        for attr in self.cached_attrs:
            if hasattr(target, attr):
                to_cache[attr] = getattr(target, attr)

        # Cache edge-type data (RRWP sparse index + values)
        if (
            self.cached_edge_type is not None
            and isinstance(data, HeteroData)
            and self.cached_edge_type in data.edge_types
        ):
            edge_store = data[self.cached_edge_type]
            if hasattr(edge_store, "edge_index"):
                to_cache["_edge_type_index"] = edge_store.edge_index
            if hasattr(edge_store, "edge_attr"):
                to_cache["_edge_type_attr"] = edge_store.edge_attr

        if not to_cache:
            return

        # Write to a temporary file in the same directory (same filesystem)
        # to guarantee os.replace is atomic.
        fd, tmp_path = tempfile.mkstemp(
            dir=self.cache_dir,
            prefix=f".pe_cache_tmp_{os.getpid()}_",
            suffix=".pt",
        )
        try:
            os.close(fd)
            torch.save(to_cache, tmp_path)
            os.replace(tmp_path, cache_path)
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def __call__(self, data):
        cache_path = self._cache_path(data)

        # Fast path: load from cache if available
        if os.path.exists(cache_path):
            self._load_cache(cache_path, data)
            return data

        # Slow path: compute, then cache
        data = self.transform(data)
        self._save_cache(cache_path, data)
        return data


def make_pe_cache_dir(processed_dir: str, pe_type: str, cfg) -> str:
    """Build a cache directory path that includes a config fingerprint.

    The fingerprint ensures that changing PE parameters (e.g. kernel.times)
    invalidates the cache automatically by using a different directory.

    Args:
        processed_dir: The dataset's processed directory.
        pe_type: "RWSE" or "RRWP".
        cfg: The data config namespace containing PE parameters.

    Returns:
        Path to the cache directory.
    """
    if pe_type == "RWSE":
        kernel_times = cfg.posenc_RWSE.kernel.times
        fingerprint = f"k{kernel_times}"
    elif pe_type == "RRWP":
        ksteps = cfg.posenc_RRWP.ksteps
        topk = getattr(cfg.posenc_RRWP, "topk", 0)
        if topk and topk > 0:
            fingerprint = f"k{ksteps}_topk{topk}"
        else:
            fingerprint = f"k{ksteps}"
    else:
        fingerprint = "default"

    cache_dir_name = f"pe_cache_{pe_type.lower()}_{fingerprint}"
    return os.path.join(processed_dir, cache_dir_name)
