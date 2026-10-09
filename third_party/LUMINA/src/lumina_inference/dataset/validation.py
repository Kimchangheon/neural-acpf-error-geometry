"""Schema detection, validation, and exception hierarchy for flexible ingestion.

Provides validation utilities to check that HeteroData objects are compatible
with loaded LUMINA models, and to detect whether data follows the standard
OPF schema or a generic heterogeneous graph structure.

Copyright 2026 UChicago Argonne, LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing
permissions and limitations under the License.
"""

from typing import Dict, Optional, Set, Tuple

from torch_geometric.data import HeteroData


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class LuminaIngestionError(Exception):
    """Base exception for all ingestion and validation errors."""


class SchemaValidationError(LuminaIngestionError):
    """Raised when HeteroData doesn't match the required schema.

    Attributes:
        expected: Description of expected schema elements.
        actual: Description of actual schema elements found.
    """

    def __init__(self, message: str, expected=None, actual=None):
        self.expected = expected
        self.actual = actual
        super().__init__(message)


class FeatureDimensionError(LuminaIngestionError):
    """Raised when feature dimensions don't match model expectations.

    Attributes:
        node_type: The node type with mismatched dimensions.
        expected_dim: Expected feature dimension.
        actual_dim: Actual feature dimension found.
    """

    def __init__(
        self,
        message: str,
        node_type: str = "",
        expected_dim: int = 0,
        actual_dim: int = 0,
    ):
        self.node_type = node_type
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim
        super().__init__(message)


class MissingDataError(LuminaIngestionError):
    """Raised when required data fields are missing.

    Attributes:
        missing_field: Name of the missing field.
        context: Additional context about where the field was expected.
    """

    def __init__(
        self, message: str, missing_field: str = "", context: str = ""
    ):
        self.missing_field = missing_field
        self.context = context
        super().__init__(message)


# ---------------------------------------------------------------------------
# OPF schema constants
# ---------------------------------------------------------------------------

OPF_NODE_TYPES: Set[str] = {"bus", "generator", "load", "shunt"}

OPF_EDGE_TYPES: Set[Tuple[str, str, str]] = {
    ("bus", "ac_line", "bus"),
    ("bus", "transformer", "bus"),
    ("generator", "generator_link", "bus"),
    ("bus", "generator_link", "generator"),
    ("load", "load_link", "bus"),
    ("bus", "load_link", "load"),
    ("shunt", "shunt_link", "bus"),
    ("bus", "shunt_link", "shunt"),
}

# Expected feature dimensions for OPF node types (after one-hot encoding of bus_type)
OPF_NODE_FEATURE_DIMS: Dict[str, int] = {
    "bus": 7,  # base_kv, vmin, vmax, pq, pv, ref, isolated
    "generator": 11,  # mbase, pg, pmin, pmax, qg, qmin, qmax, vg, cost_sq, cost_lin, cost_off
    "load": 2,  # pd, qd
    "shunt": 2,  # bs, gs
}

# Expected edge attribute dimensions for OPF edge types
OPF_EDGE_ATTR_DIMS: Dict[Tuple[str, str, str], int] = {
    ("bus", "ac_line", "bus"): 9,
    ("bus", "transformer", "bus"): 11,
}


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------


def detect_schema(data: HeteroData) -> str:
    """Detect whether data follows the OPF schema or is generic.

    Checks if all required OPF node types are present in the data.
    Edge types are not strictly required for detection since some
    grids may lack transformers or shunts.

    Args:
        data: A HeteroData object to inspect.

    Returns:
        ``'opf'`` if all OPF node types (bus, generator, load, shunt)
        are present, ``'generic'`` otherwise.
    """
    node_types = set(data.node_types)
    if OPF_NODE_TYPES.issubset(node_types):
        return "opf"
    return "generic"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_hetero_data(
    data: HeteroData,
    model_metadata: Optional[dict] = None,
    model_input_channels: Optional[Dict[str, int]] = None,
) -> None:
    """Validate that a HeteroData object is compatible with a LUMINA model.

    Performs the following checks:

    1. Data has at least one node type and one edge type.
    2. All node types have a non-empty ``x`` (feature) tensor.
    3. Edge index tensors are 2D with shape ``[2, num_edges]``.
    4. If ``model_metadata`` is provided, all node types required by the
       model exist in the data.
    5. If ``model_input_channels`` is provided, feature dimensions match.

    Args:
        data: The HeteroData object to validate.
        model_metadata: Optional model metadata dict with ``'nodes'`` and
            ``'edges'`` keys describing expected types.
        model_input_channels: Optional dict mapping node type names to
            expected input feature dimensions.

    Raises:
        SchemaValidationError: If structural requirements are not met
            (missing node/edge types, wrong edge index shape).
        FeatureDimensionError: If feature dimensions don't match model
            expectations.
        MissingDataError: If required feature tensors are missing.
    """
    # Basic structural checks
    if not data.node_types:
        raise SchemaValidationError(
            "HeteroData has no node types.",
            expected="at least one node type",
            actual="none",
        )

    if not data.edge_types:
        raise SchemaValidationError(
            "HeteroData has no edge types.",
            expected="at least one edge type",
            actual="none",
        )

    # Check that all node types have features
    for node_type in data.node_types:
        x = getattr(data[node_type], "x", None)
        if x is None:
            raise MissingDataError(
                f"Node type '{node_type}' has no feature tensor (.x).",
                missing_field="x",
                context=f"node type '{node_type}'",
            )
        if x.ndim != 2:
            raise SchemaValidationError(
                f"Node type '{node_type}' feature tensor has {x.ndim} "
                f"dimensions, expected 2.",
                expected="2D tensor [num_nodes, num_features]",
                actual=f"{x.ndim}D tensor with shape {list(x.shape)}",
            )

    # Check edge index shapes
    for edge_type in data.edge_types:
        edge_store = data[edge_type]
        edge_index = getattr(edge_store, "edge_index", None)
        if edge_index is None:
            raise MissingDataError(
                f"Edge type {edge_type} has no edge_index tensor.",
                missing_field="edge_index",
                context=f"edge type {edge_type}",
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise SchemaValidationError(
                f"Edge type {edge_type} edge_index has shape "
                f"{list(edge_index.shape)}, expected [2, num_edges].",
                expected="[2, num_edges]",
                actual=str(list(edge_index.shape)),
            )

    # Model-specific validation
    if model_metadata is not None:
        if isinstance(model_metadata, dict) and "nodes" in model_metadata:
            expected_node_types = set(model_metadata["nodes"].keys())
        elif isinstance(model_metadata, (list, tuple)) and len(model_metadata) >= 1:
            expected_node_types = set(model_metadata[0])
        else:
            expected_node_types = set()

        actual_node_types = set(data.node_types)
        missing = expected_node_types - actual_node_types
        if missing:
            raise SchemaValidationError(
                f"Data is missing node types required by the model: {missing}. "
                f"Model expects: {expected_node_types}, "
                f"data has: {actual_node_types}.",
                expected=expected_node_types,
                actual=actual_node_types,
            )

    if model_input_channels is not None:
        for node_type, expected_dim in model_input_channels.items():
            if node_type not in data.node_types:
                continue  # Already caught by model_metadata check above
            actual_dim = data[node_type].x.shape[-1]
            if actual_dim != expected_dim:
                raise FeatureDimensionError(
                    f"Feature dimension mismatch for node type '{node_type}': "
                    f"model expects {expected_dim}, data has {actual_dim}.",
                    node_type=node_type,
                    expected_dim=expected_dim,
                    actual_dim=actual_dim,
                )


def validate_opf_schema(data: HeteroData) -> None:
    """Validate that a HeteroData object follows the OPF schema.

    Checks for all required OPF node types, edge types, and expected
    feature dimensions.

    Args:
        data: The HeteroData object to validate.

    Raises:
        SchemaValidationError: If OPF-specific structural requirements
            are not met.
        FeatureDimensionError: If feature dimensions don't match OPF
            expectations.
        MissingDataError: If required feature tensors are missing.
    """
    # First run generic validation
    validate_hetero_data(data)

    # Check OPF node types
    actual_node_types = set(data.node_types)
    missing_nodes = OPF_NODE_TYPES - actual_node_types
    if missing_nodes:
        raise SchemaValidationError(
            f"Data is missing required OPF node types: {missing_nodes}. "
            f"OPF schema requires: {OPF_NODE_TYPES}.",
            expected=OPF_NODE_TYPES,
            actual=actual_node_types,
        )

    # Check OPF feature dimensions
    for node_type, expected_dim in OPF_NODE_FEATURE_DIMS.items():
        actual_dim = data[node_type].x.shape[-1]
        if actual_dim != expected_dim:
            raise FeatureDimensionError(
                f"OPF feature dimension mismatch for '{node_type}': "
                f"expected {expected_dim}, got {actual_dim}.",
                node_type=node_type,
                expected_dim=expected_dim,
                actual_dim=actual_dim,
            )

    # Check required edge types (at minimum ac_line and virtual links)
    actual_edge_types = set(data.edge_types)
    # ac_line is always required; transformer may be absent for some grids
    required_edges = {
        ("bus", "ac_line", "bus"),
        ("generator", "generator_link", "bus"),
        ("bus", "generator_link", "generator"),
        ("load", "load_link", "bus"),
        ("bus", "load_link", "load"),
        ("shunt", "shunt_link", "bus"),
        ("bus", "shunt_link", "shunt"),
    }
    missing_edges = required_edges - actual_edge_types
    if missing_edges:
        raise SchemaValidationError(
            f"Data is missing required OPF edge types: {missing_edges}.",
            expected=required_edges,
            actual=actual_edge_types,
        )
