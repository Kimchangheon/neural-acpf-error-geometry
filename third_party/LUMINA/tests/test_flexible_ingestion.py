"""Tests for flexible ingestion capabilities.

Tests cover:
- process_opf_dict with and without solution
- load_from_dict, load_from_json_file, load_from_json_string
- build_hetero_data for generic heterogeneous graphs
- Schema detection (detect_schema)
- Validation (validate_hetero_data, validate_opf_schema)
- predict_single and _validate_batch on Modeler
- Error cases for missing data, wrong dimensions, etc.

These tests do NOT require network access or HuggingFace downloads.
They use synthetic data to validate the ingestion pipeline.

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

import json
import os
import tempfile

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from lumina_inference.dataset.ingestion import (
    build_hetero_data,
    load_from_dict,
    load_from_json_file,
    load_from_json_string,
    process_opf_dict,
)
from lumina_inference.dataset.opf_dataset import process_json_file
from lumina_inference.dataset.validation import (
    FeatureDimensionError,
    LuminaIngestionError,
    MissingDataError,
    SchemaValidationError,
    detect_schema,
    validate_hetero_data,
    validate_opf_schema,
)


# ---------------------------------------------------------------------------
# Fixtures: sample OPF data
# ---------------------------------------------------------------------------


def _make_opf_dict(with_solution=False):
    """Create a minimal valid OPF data dictionary."""
    opf_dict = {
        "grid": {
            "context": [[[100.0]]],
            "nodes": {
                "bus": [
                    [69.0, 3, 0.94, 1.06],
                    [69.0, 2, 0.94, 1.06],
                    [69.0, 1, 0.94, 1.06],
                    [69.0, 1, 0.94, 1.06],
                ],
                "generator": [
                    [100.0, 2.324, 0.0, 3.324, 0.0, -0.1, 0.1, 1.06,
                     0.043, 20.0, 0.0],
                    [100.0, 0.40, 0.0, 1.40, 0.0, -0.4, 0.5, 1.045,
                     0.25, 20.0, 0.0],
                ],
                "load": [
                    [0.217, 0.127],
                    [0.942, 0.190],
                    [0.478, -0.039],
                ],
                "shunt": [
                    [0.19, 0.0],
                ],
            },
            "edges": {
                "ac_line": {
                    "senders": [0, 1, 1, 2],
                    "receivers": [1, 2, 3, 3],
                    "features": [
                        [-0.52, 0.52, 0.026, 0.026, 0.019, 0.059, 4.72, 4.72, 4.72],
                        [-0.52, 0.52, 0.022, 0.022, 0.054, 0.223, 1.28, 1.28, 1.28],
                        [-0.52, 0.52, 0.017, 0.017, 0.047, 0.198, 1.28, 1.28, 1.28],
                        [-0.52, 0.52, 0.013, 0.013, 0.067, 0.171, 0.65, 0.65, 0.65],
                    ],
                },
                "transformer": {
                    "senders": [0],
                    "receivers": [3],
                    "features": [
                        [-0.52, 0.52, 0.0, 0.209, 0.978, 0.978, 0.978, 0.969, 0.0, 0.0, 0.0],
                    ],
                },
                "generator_link": {
                    "senders": [0, 1],
                    "receivers": [0, 1],
                },
                "load_link": {
                    "senders": [0, 1, 2],
                    "receivers": [1, 2, 3],
                },
                "shunt_link": {
                    "senders": [0],
                    "receivers": [3],
                },
            },
        },
        "metadata": {"objective": 0.0},
    }

    if with_solution:
        opf_dict["solution"] = {
            "nodes": {
                "bus": [
                    [0.0, 1.06],
                    [-0.087, 1.045],
                    [-0.222, 1.010],
                    [-0.180, 1.018],
                ],
                "generator": [
                    [2.324, -0.169],
                    [0.40, 0.424],
                ],
            },
            "edges": {
                "ac_line": {
                    "features": [
                        [1.57, 0.0, -1.57, 0.0],
                        [0.73, 0.0, -0.73, 0.0],
                        [0.24, 0.0, -0.24, 0.0],
                        [0.28, 0.0, -0.28, 0.0],
                    ],
                },
                "transformer": {
                    "features": [
                        [0.76, 0.0, -0.76, 0.0],
                    ],
                },
            },
        }

    return opf_dict


@pytest.fixture
def opf_dict_no_solution():
    return _make_opf_dict(with_solution=False)


@pytest.fixture
def opf_dict_with_solution():
    return _make_opf_dict(with_solution=True)


# ============================================================================
# Tests: process_opf_dict
# ============================================================================


class TestProcessOpfDict:
    """Test the core OPF dict → HeteroData converter."""

    def test_basic_conversion_no_solution(self, opf_dict_no_solution):
        """Convert OPF dict without solution data."""
        data = process_opf_dict(opf_dict_no_solution, require_solution=False)

        assert isinstance(data, HeteroData)
        assert "bus" in data.node_types
        assert "generator" in data.node_types
        assert "load" in data.node_types
        assert "shunt" in data.node_types

    def test_basic_conversion_with_solution(self, opf_dict_with_solution):
        """Convert OPF dict with solution data."""
        data = process_opf_dict(opf_dict_with_solution, require_solution=True)

        assert isinstance(data, HeteroData)
        assert data["bus"].y is not None
        assert data["generator"].y is not None

    def test_bus_features_shape(self, opf_dict_no_solution):
        """Bus features should be 7-dimensional after one-hot encoding."""
        data = process_opf_dict(opf_dict_no_solution)
        # [base_kv, vmin, vmax, pq, pv, ref, isolated] = 7
        assert data["bus"].x.shape == (4, 7)

    def test_generator_features_shape(self, opf_dict_no_solution):
        """Generator features should be 11-dimensional."""
        data = process_opf_dict(opf_dict_no_solution)
        assert data["generator"].x.shape == (2, 11)

    def test_load_features_shape(self, opf_dict_no_solution):
        """Load features should be 2-dimensional."""
        data = process_opf_dict(opf_dict_no_solution)
        assert data["load"].x.shape == (3, 2)

    def test_shunt_features_shape(self, opf_dict_no_solution):
        """Shunt features should be 2-dimensional."""
        data = process_opf_dict(opf_dict_no_solution)
        assert data["shunt"].x.shape == (1, 2)

    def test_edge_indices_present(self, opf_dict_no_solution):
        """All edge types should have edge_index tensors."""
        data = process_opf_dict(opf_dict_no_solution)

        assert data["bus", "ac_line", "bus"].edge_index.shape[0] == 2
        assert data["bus", "transformer", "bus"].edge_index.shape[0] == 2
        assert data["generator", "generator_link", "bus"].edge_index.shape[0] == 2
        assert data["bus", "generator_link", "generator"].edge_index.shape[0] == 2
        assert data["load", "load_link", "bus"].edge_index.shape[0] == 2
        assert data["bus", "load_link", "load"].edge_index.shape[0] == 2
        assert data["shunt", "shunt_link", "bus"].edge_index.shape[0] == 2
        assert data["bus", "shunt_link", "shunt"].edge_index.shape[0] == 2

    def test_edge_attr_present(self, opf_dict_no_solution):
        """AC line and transformer should have edge attributes."""
        data = process_opf_dict(opf_dict_no_solution)

        assert data["bus", "ac_line", "bus"].edge_attr.shape == (4, 9)
        assert data["bus", "transformer", "bus"].edge_attr.shape == (1, 11)

    def test_no_y_without_solution(self, opf_dict_no_solution):
        """Without solution, .y should not be set."""
        data = process_opf_dict(opf_dict_no_solution, require_solution=False)

        assert not hasattr(data["bus"], "y") or data["bus"].y is None
        assert not hasattr(data["generator"], "y") or data["generator"].y is None

    def test_y_with_solution(self, opf_dict_with_solution):
        """With solution, .y should be set for bus and generator."""
        data = process_opf_dict(opf_dict_with_solution, require_solution=True)

        assert data["bus"].y.shape == (4, 2)
        assert data["generator"].y.shape == (2, 2)

    def test_edge_label_with_solution(self, opf_dict_with_solution):
        """With solution, edge_label should be set for ac_line and transformer."""
        data = process_opf_dict(opf_dict_with_solution, require_solution=True)

        assert data["bus", "ac_line", "bus"].edge_label.shape == (4, 4)
        assert data["bus", "transformer", "bus"].edge_label.shape == (1, 4)

    def test_basemva_stored(self, opf_dict_no_solution):
        """baseMVA should be stored as a graph-level attribute."""
        data = process_opf_dict(opf_dict_no_solution)
        assert hasattr(data, "baseMVA")
        assert data.baseMVA == 100.0

    def test_require_solution_raises_when_missing(self, opf_dict_no_solution):
        """Should raise MissingDataError when solution is required but absent."""
        with pytest.raises(MissingDataError, match="Solution data is required"):
            process_opf_dict(opf_dict_no_solution, require_solution=True)

    def test_missing_grid_key_raises(self):
        """Should raise MissingDataError when 'grid' key is missing."""
        with pytest.raises(MissingDataError, match="'grid' key"):
            process_opf_dict({"metadata": {}})

    def test_missing_nodes_key_raises(self):
        """Should raise SchemaValidationError when 'nodes' key is missing."""
        with pytest.raises(SchemaValidationError, match="'nodes' key"):
            process_opf_dict({"grid": {"edges": {}}})

    def test_missing_edges_key_raises(self):
        """Should raise SchemaValidationError when 'edges' key is missing."""
        with pytest.raises(SchemaValidationError, match="'edges' key"):
            process_opf_dict({"grid": {"nodes": {}}})

    def test_bus_type_one_hot_encoding(self, opf_dict_no_solution):
        """Bus type should be one-hot encoded correctly."""
        data = process_opf_dict(opf_dict_no_solution)
        bus_x = data["bus"].x

        # Bus 0: type 3 (ref) → [0, 0, 1, 0]
        assert bus_x[0, 3:7].tolist() == pytest.approx([0, 0, 1, 0])
        # Bus 1: type 2 (PV) → [0, 1, 0, 0]
        assert bus_x[1, 3:7].tolist() == pytest.approx([0, 1, 0, 0])
        # Bus 2: type 1 (PQ) → [1, 0, 0, 0]
        assert bus_x[2, 3:7].tolist() == pytest.approx([1, 0, 0, 0])


# ============================================================================
# Tests: load_from_* functions
# ============================================================================


class TestLoadFromDict:
    """Test load_from_dict."""

    def test_load_from_dict_no_solution(self, opf_dict_no_solution):
        data = load_from_dict(opf_dict_no_solution, require_solution=False)
        assert isinstance(data, HeteroData)
        assert "bus" in data.node_types

    def test_load_from_dict_with_solution(self, opf_dict_with_solution):
        data = load_from_dict(opf_dict_with_solution, require_solution=True)
        assert data["bus"].y is not None


class TestLoadFromJsonString:
    """Test load_from_json_string."""

    def test_load_from_json_string(self, opf_dict_no_solution):
        json_str = json.dumps(opf_dict_no_solution)
        data = load_from_json_string(json_str, require_solution=False)
        assert isinstance(data, HeteroData)
        assert "bus" in data.node_types

    def test_invalid_json_raises(self):
        with pytest.raises(MissingDataError, match="Failed to decode JSON"):
            load_from_json_string("{invalid json", require_solution=False)


class TestLoadFromJsonFile:
    """Test load_from_json_file."""

    def test_load_from_json_file(self, opf_dict_no_solution):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(opf_dict_no_solution, f)
            tmp_path = f.name

        try:
            data = load_from_json_file(tmp_path, require_solution=False)
            assert isinstance(data, HeteroData)
            assert "bus" in data.node_types
        finally:
            os.unlink(tmp_path)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_from_json_file("/nonexistent/path.json")

    def test_invalid_json_file_raises(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{invalid json")
            tmp_path = f.name

        try:
            with pytest.raises(MissingDataError, match="Failed to decode JSON"):
                load_from_json_file(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestProcessJsonFile:
    """Test the refactored process_json_file function."""

    def test_backward_compatible_with_solution(self, opf_dict_with_solution):
        """process_json_file should work identically to before with solution."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(opf_dict_with_solution, f)
            tmp_path = f.name

        try:
            data = process_json_file(tmp_path, require_solution=True)
            assert isinstance(data, HeteroData)
            assert data["bus"].y is not None
            assert data["generator"].y is not None
            assert data["bus"].x.shape == (4, 7)
        finally:
            os.unlink(tmp_path)

    def test_inference_mode_no_solution(self, opf_dict_no_solution):
        """process_json_file should work without solution when not required."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(opf_dict_no_solution, f)
            tmp_path = f.name

        try:
            data = process_json_file(tmp_path, require_solution=False)
            assert isinstance(data, HeteroData)
            assert "bus" in data.node_types
        finally:
            os.unlink(tmp_path)

    def test_invalid_json_returns_none(self):
        """process_json_file should return None for invalid JSON."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{invalid json")
            tmp_path = f.name

        try:
            result = process_json_file(tmp_path)
            assert result is None
        finally:
            os.unlink(tmp_path)


# ============================================================================
# Tests: build_hetero_data
# ============================================================================


class TestBuildHeteroData:
    """Test the generic HeteroData builder."""

    def test_basic_build(self):
        """Build a simple heterogeneous graph."""
        data = build_hetero_data(
            nodes={
                "A": {"x": np.random.randn(5, 3).astype(np.float32)},
                "B": {"x": np.random.randn(3, 4).astype(np.float32)},
            },
            edges={
                ("A", "connects", "B"): {
                    "edge_index": torch.tensor([[0, 1, 2], [0, 1, 2]]),
                },
            },
        )

        assert isinstance(data, HeteroData)
        assert "A" in data.node_types
        assert "B" in data.node_types
        assert data["A"].x.shape == (5, 3)
        assert data["B"].x.shape == (3, 4)

    def test_build_with_edge_attr(self):
        """Build with edge attributes."""
        data = build_hetero_data(
            nodes={
                "node": {"x": torch.randn(4, 2)},
            },
            edges={
                ("node", "edge", "node"): {
                    "edge_index": torch.tensor([[0, 1], [1, 2]]),
                    "edge_attr": torch.randn(2, 5),
                },
            },
        )

        assert data["node", "edge", "node"].edge_attr.shape == (2, 5)

    def test_build_with_targets(self):
        """Build with target labels."""
        data = build_hetero_data(
            nodes={
                "node": {
                    "x": torch.randn(4, 2),
                    "y": torch.randn(4, 1),
                },
            },
            edges={
                ("node", "edge", "node"): {
                    "edge_index": torch.tensor([[0, 1], [1, 2]]),
                    "edge_label": torch.randn(2, 3),
                },
            },
        )

        assert data["node"].y.shape == (4, 1)
        assert data["node", "edge", "node"].edge_label.shape == (2, 3)

    def test_build_with_graph_attrs(self):
        """Build with graph-level attributes."""
        data = build_hetero_data(
            nodes={"node": {"x": torch.randn(2, 3)}},
            edges={
                ("node", "self_loop", "node"): {
                    "edge_index": torch.tensor([[0, 1], [1, 0]]),
                },
            },
            graph_attrs={"name": "test_graph", "value": 42.0},
        )

        assert data.name == "test_graph"
        assert data.value == 42.0

    def test_build_with_numpy_arrays(self):
        """Build from numpy arrays (auto-converted to tensors)."""
        data = build_hetero_data(
            nodes={
                "node": {"x": np.array([[1.0, 2.0], [3.0, 4.0]])},
            },
            edges={
                ("node", "edge", "node"): {
                    "edge_index": np.array([[0], [1]]),
                },
            },
        )

        assert isinstance(data["node"].x, torch.Tensor)
        assert isinstance(data["node", "edge", "node"].edge_index, torch.Tensor)

    def test_missing_x_raises(self):
        """Should raise MissingDataError when 'x' is missing."""
        with pytest.raises(MissingDataError, match="must have an 'x'"):
            build_hetero_data(
                nodes={"node": {"y": torch.randn(2, 3)}},
                edges={},
            )

    def test_missing_edge_index_raises(self):
        """Should raise MissingDataError when 'edge_index' is missing."""
        with pytest.raises(MissingDataError, match="must have an 'edge_index'"):
            build_hetero_data(
                nodes={"node": {"x": torch.randn(2, 3)}},
                edges={
                    ("node", "edge", "node"): {
                        "edge_attr": torch.randn(2, 5),
                    },
                },
            )


# ============================================================================
# Tests: Schema detection
# ============================================================================


class TestDetectSchema:
    """Test schema detection."""

    def test_detect_opf_schema(self, opf_dict_no_solution):
        """OPF data should be detected as 'opf'."""
        data = process_opf_dict(opf_dict_no_solution)
        assert detect_schema(data) == "opf"

    def test_detect_generic_schema(self):
        """Non-OPF data should be detected as 'generic'."""
        data = build_hetero_data(
            nodes={
                "station": {"x": torch.randn(5, 3)},
                "sensor": {"x": torch.randn(10, 2)},
            },
            edges={
                ("station", "connects", "station"): {
                    "edge_index": torch.tensor([[0, 1], [1, 2]]),
                },
            },
        )
        assert detect_schema(data) == "generic"

    def test_partial_opf_is_generic(self):
        """Data with only some OPF node types should be 'generic'."""
        data = build_hetero_data(
            nodes={
                "bus": {"x": torch.randn(4, 7)},
                "generator": {"x": torch.randn(2, 11)},
                # Missing 'load' and 'shunt'
            },
            edges={
                ("bus", "ac_line", "bus"): {
                    "edge_index": torch.tensor([[0, 1], [1, 2]]),
                },
            },
        )
        assert detect_schema(data) == "generic"


# ============================================================================
# Tests: Validation
# ============================================================================


class TestValidateHeteroData:
    """Test generic HeteroData validation."""

    def test_valid_data_passes(self, opf_dict_no_solution):
        """Valid OPF data should pass validation."""
        data = process_opf_dict(opf_dict_no_solution)
        validate_hetero_data(data)  # Should not raise

    def test_empty_node_types_raises(self):
        """HeteroData with no node types should fail."""
        data = HeteroData()
        with pytest.raises(SchemaValidationError, match="no node types"):
            validate_hetero_data(data)

    def test_missing_x_raises(self):
        """Node type without features should fail."""
        data = HeteroData()
        data["node"].edge_index = torch.tensor([[0], [1]])
        # node has no .x
        with pytest.raises((SchemaValidationError, MissingDataError)):
            validate_hetero_data(data)

    def test_model_metadata_validation(self, opf_dict_no_solution):
        """Validation against model metadata should catch missing types."""
        data = build_hetero_data(
            nodes={"bus": {"x": torch.randn(4, 7)}},
            edges={
                ("bus", "self", "bus"): {
                    "edge_index": torch.tensor([[0], [1]]),
                },
            },
        )

        with pytest.raises(SchemaValidationError, match="missing node types"):
            validate_hetero_data(
                data,
                model_metadata={
                    "nodes": {"bus": 7, "generator": 11},
                    "edges": {},
                },
            )

    def test_feature_dimension_validation(self, opf_dict_no_solution):
        """Validation should catch wrong feature dimensions."""
        data = build_hetero_data(
            nodes={
                "bus": {"x": torch.randn(4, 5)},  # Wrong: should be 7
                "generator": {"x": torch.randn(2, 11)},
            },
            edges={
                ("bus", "ac_line", "bus"): {
                    "edge_index": torch.tensor([[0], [1]]),
                },
            },
        )

        with pytest.raises(FeatureDimensionError, match="dimension mismatch"):
            validate_hetero_data(
                data,
                model_input_channels={"bus": 7, "generator": 11},
            )


class TestValidateOpfSchema:
    """Test OPF-specific schema validation."""

    def test_valid_opf_passes(self, opf_dict_no_solution):
        """Valid OPF data should pass OPF validation."""
        data = process_opf_dict(opf_dict_no_solution)
        validate_opf_schema(data)  # Should not raise

    def test_missing_opf_node_type_raises(self):
        """Missing OPF node types should fail."""
        data = build_hetero_data(
            nodes={
                "bus": {"x": torch.randn(4, 7)},
                "generator": {"x": torch.randn(2, 11)},
                # Missing 'load' and 'shunt'
            },
            edges={
                ("bus", "ac_line", "bus"): {
                    "edge_index": torch.tensor([[0], [1]]),
                },
            },
        )

        with pytest.raises(SchemaValidationError, match="missing required OPF node types"):
            validate_opf_schema(data)

    def test_wrong_feature_dims_raises(self):
        """Wrong OPF feature dimensions should fail."""
        data = build_hetero_data(
            nodes={
                "bus": {"x": torch.randn(4, 5)},  # Wrong: should be 7
                "generator": {"x": torch.randn(2, 11)},
                "load": {"x": torch.randn(3, 2)},
                "shunt": {"x": torch.randn(1, 2)},
            },
            edges={
                ("bus", "ac_line", "bus"): {
                    "edge_index": torch.tensor([[0], [1]]),
                    "edge_attr": torch.randn(1, 9),
                },
                ("generator", "generator_link", "bus"): {
                    "edge_index": torch.tensor([[0], [0]]),
                },
                ("bus", "generator_link", "generator"): {
                    "edge_index": torch.tensor([[0], [0]]),
                },
                ("load", "load_link", "bus"): {
                    "edge_index": torch.tensor([[0], [0]]),
                },
                ("bus", "load_link", "load"): {
                    "edge_index": torch.tensor([[0], [0]]),
                },
                ("shunt", "shunt_link", "bus"): {
                    "edge_index": torch.tensor([[0], [0]]),
                },
                ("bus", "shunt_link", "shunt"): {
                    "edge_index": torch.tensor([[0], [0]]),
                },
            },
        )

        with pytest.raises(FeatureDimensionError, match="OPF feature dimension mismatch"):
            validate_opf_schema(data)


# ============================================================================
# Tests: Exception hierarchy
# ============================================================================


class TestExceptionHierarchy:
    """Test that exception classes have proper hierarchy."""

    def test_schema_validation_is_ingestion_error(self):
        assert issubclass(SchemaValidationError, LuminaIngestionError)

    def test_feature_dimension_is_ingestion_error(self):
        assert issubclass(FeatureDimensionError, LuminaIngestionError)

    def test_missing_data_is_ingestion_error(self):
        assert issubclass(MissingDataError, LuminaIngestionError)

    def test_schema_validation_attributes(self):
        err = SchemaValidationError("test", expected="a", actual="b")
        assert err.expected == "a"
        assert err.actual == "b"

    def test_feature_dimension_attributes(self):
        err = FeatureDimensionError(
            "test", node_type="bus", expected_dim=7, actual_dim=5
        )
        assert err.node_type == "bus"
        assert err.expected_dim == 7
        assert err.actual_dim == 5

    def test_missing_data_attributes(self):
        err = MissingDataError("test", missing_field="x", context="bus")
        assert err.missing_field == "x"
        assert err.context == "bus"


# ============================================================================
# Tests: Modeler.predict_single and _validate_batch
# ============================================================================


class TestModelerPredictSingle:
    """Test predict_single and _validate_batch on Modeler."""

    def test_predict_single_raises_without_model(self, opf_dict_no_solution):
        """predict_single should raise RuntimeError if model not loaded."""
        modeler = Modeler(torch.device("cpu"))
        data = process_opf_dict(opf_dict_no_solution)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            modeler.predict_single(data)

    def test_validate_batch_raises_without_model(self, opf_dict_no_solution):
        """_validate_batch should raise RuntimeError if model not loaded."""
        modeler = Modeler(torch.device("cpu"))
        data = process_opf_dict(opf_dict_no_solution)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            modeler._validate_batch(data)


# Need to import Modeler here since it's used in the test class above
from lumina_inference.modeler import Modeler


# ============================================================================
# Tests: Top-level imports
# ============================================================================


class TestTopLevelImports:
    """Test that ingestion functions are importable from top-level."""

    def test_import_from_lumina_inference(self):
        """All ingestion functions should be importable from lumina_inference."""
        from lumina_inference import (
            build_hetero_data,
            load_from_dict,
            load_from_json_file,
            load_from_json_string,
            load_from_matpower,
            process_json_file,
            process_opf_dict,
        )

        # Verify they are callable
        assert callable(build_hetero_data)
        assert callable(load_from_dict)
        assert callable(load_from_json_file)
        assert callable(load_from_json_string)
        assert callable(load_from_matpower)
        assert callable(process_json_file)
        assert callable(process_opf_dict)

    def test_import_from_dataset_package(self):
        """All ingestion functions should be importable from dataset package."""
        from lumina_inference.dataset import (
            build_hetero_data,
            detect_schema,
            load_from_dict,
            load_from_json_file,
            load_from_json_string,
            load_from_matpower,
            process_json_file,
            process_opf_dict,
            validate_hetero_data,
            validate_opf_schema,
            FeatureDimensionError,
            LuminaIngestionError,
            MissingDataError,
            SchemaValidationError,
        )

        assert callable(build_hetero_data)
        assert callable(detect_schema)


# ============================================================================
# Tests: Model forward pass with conditional routing
# ============================================================================


class TestModelConditionalRouting:
    """Test HGT forward pass on OPF data."""

    def _make_opf_model(self):
        """Create a small HGT model for testing."""
        from lumina_inference.model.hetero_model import HGT

        metadata = {
            "nodes": {
                "bus": 7,
                "generator": 11,
                "load": 2,
                "shunt": 2,
            },
            "edges": {
                ("bus", "ac_line", "bus"): 9,
                ("bus", "transformer", "bus"): 11,
                ("generator", "generator_link", "bus"): 0,
                ("bus", "generator_link", "generator"): 0,
                ("load", "load_link", "bus"): 0,
                ("bus", "load_link", "load"): 0,
                ("shunt", "shunt_link", "bus"): 0,
                ("bus", "shunt_link", "shunt"): 0,
            },
        }
        input_channels = {"bus": 7, "generator": 11, "load": 2, "shunt": 2}

        model = HGT(
            metadata=metadata,
            input_channels=input_channels,
            hidden_channels=16,
            out_channels=2,
            num_layers=2,
            num_heads=1,
        )
        model.eval()
        return model

    def test_opf_forward_without_scaling(self, opf_dict_no_solution):
        """Forward pass on OPF data without scaling."""
        model = self._make_opf_model()
        data = process_opf_dict(opf_dict_no_solution)

        with torch.no_grad():
            out = model(
                data.x_dict,
                data.edge_index_dict,
                minmax_scaling=False,
            )

        assert "bus" in out
        assert "generator" in out
        assert out["bus"].shape == (4, 2)
        assert out["generator"].shape == (2, 2)

    def test_opf_forward_with_scaling(self, opf_dict_no_solution):
        """Forward pass on OPF data with minmax scaling."""
        model = self._make_opf_model()
        data = process_opf_dict(opf_dict_no_solution)

        with torch.no_grad():
            out = model(
                data.x_dict,
                data.edge_index_dict,
                minmax_scaling=True,
            )

        assert "bus" in out
        assert "generator" in out
        # Voltage magnitude (column 1) should be bounded by vmin/vmax
        vmin = data["bus"].x[:, 1]
        vmax = data["bus"].x[:, 2]
        vm_pred = out["bus"][:, 1]
        assert (vm_pred >= vmin - 1e-5).all(), "Voltage below vmin"
        assert (vm_pred <= vmax + 1e-5).all(), "Voltage above vmax"

    def test_opf_forward_outputs_finite(self, opf_dict_no_solution):
        """All outputs should be finite."""
        model = self._make_opf_model()
        data = process_opf_dict(opf_dict_no_solution)

        with torch.no_grad():
            out = model(
                data.x_dict,
                data.edge_index_dict,
                minmax_scaling=True,
            )

        for key, tensor in out.items():
            assert torch.isfinite(tensor).all(), (
                f"Non-finite values in output['{key}']"
            )
