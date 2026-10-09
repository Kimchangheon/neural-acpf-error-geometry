"""
Session-scoped fixture that ensures the processed test data directory is
populated before any test that needs it runs.

Specifically it:
  1. Runs LitGridHeteroDataModule.setup("fit") which triggers
     HeteroGridDatasetDisk to write the ``processed/`` .pt files.
  2. Persists the fitted normalizer stats as
     ``tests/data/case14_ieee/processed/data_stats_HeteroDataMVANormalizer.pt``
     so that test_edge_flows.py and test_simulate_measurements.py can load
     them directly without needing a full DM setup.
"""

import os

import pytest
import torch
import yaml

from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
from gridfm_graphkit.io.param_handler import NestedNamespace

_STATS_PATH = "tests/data/case14_ieee/processed/data_stats_HeteroDataMVANormalizer.pt"
_CONFIG_PATH = "tests/config/datamodule_test_base_config.yaml"


class _DummyTrainer:
    """Minimal stand-in for a Lightning Trainer used only during test setup."""

    is_global_zero = True
    logger = None  # prevents AttributeError in hetero_powergrid_datamodule.setup()


@pytest.fixture(scope="session", autouse=True)
def generate_processed_test_data():
    """
    Generate processed test data files that are needed by tests which load
    them directly (test_edge_flows, test_simulate_measurements).

    Skipped silently if the stats file already exists (e.g., second pytest run
    in the same environment without cleaning the processed/ directory).
    """
    if os.path.exists(_STATS_PATH):
        return

    with open(_CONFIG_PATH) as f:
        config_dict = yaml.safe_load(f)

    args = NestedNamespace(**config_dict)
    dm = LitGridHeteroDataModule(args, data_dir="tests/data")
    dm.trainer = _DummyTrainer()
    dm.setup("fit")

    # Persist the fitted normalizer stats under the name used by the tests.
    normalizer = dm.data_normalizers[0]
    assert isinstance(normalizer, HeteroDataMVANormalizer), (
        f"Expected HeteroDataMVANormalizer, got {type(normalizer).__name__}"
    )
    torch.save(normalizer.get_stats(), _STATS_PATH)
