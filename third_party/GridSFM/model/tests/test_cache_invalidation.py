"""Cache fingerprints must invalidate on any state that affects cached output."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from gridsfm import load_pyg_json
from gridsfm.cycle_basis import CycleBasisCache
from gridsfm.dc_prior import DCPriorCache
from gridsfm.pe_features import _topology_fingerprint
from gridsfm.schema import BUS_TYPE_IDX, AC_LINE_KEY

ROOT = Path(__file__).resolve().parent.parent


SAMPLE = ROOT / "samples" / "case500_goc.pyg.json"


@pytest.fixture(scope="module")
def data():
    return load_pyg_json(str(SAMPLE))


def test_laplacian_fingerprint_invalidates_on_impedance(data):
    d2 = copy.deepcopy(data)
    d2[AC_LINE_KEY].edge_attr[:, 5] *= 1.05  # 5% derate of x
    assert _topology_fingerprint(data) != _topology_fingerprint(d2)


def test_laplacian_fingerprint_invalidates_on_slack(data):
    d2 = copy.deepcopy(data)
    slack = (d2['bus'].x[:, BUS_TYPE_IDX] == 3).nonzero().flatten()
    pv = (d2['bus'].x[:, BUS_TYPE_IDX] == 2).nonzero().flatten()
    assert slack.numel() > 0 and pv.numel() > 0, "sample must have slack and PV buses"
    d2['bus'].x[slack[0], BUS_TYPE_IDX] = 2
    d2['bus'].x[pv[0], BUS_TYPE_IDX] = 3
    assert _topology_fingerprint(data) != _topology_fingerprint(d2)


def test_laplacian_fingerprint_invalidates_on_gen_placement(data):
    d2 = copy.deepcopy(data)
    gei = d2[('generator', 'generator_link', 'bus')].edge_index
    d2[('generator', 'generator_link', 'bus')].edge_index = torch.stack(
        [gei[0], gei[1].roll(1)]
    )
    assert _topology_fingerprint(data) != _topology_fingerprint(d2)


def test_cycle_fingerprint_invalidates_on_impedance(data):
    cbc = CycleBasisCache(cache_dir=None)
    d2 = copy.deepcopy(data)
    d2[AC_LINE_KEY].edge_attr[:, 5] *= 1.05
    assert cbc._fingerprint(data) != cbc._fingerprint(d2)


def test_dc_prior_fingerprint_invalidates_on_slack():
    cache = DCPriorCache()
    n_bus = 4
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    b_ij = np.array([10.0, 10.0, 10.0])
    k_slack0 = cache.topo_key(n_bus, ei, b_ij, slack_idx=0)
    k_slack3 = cache.topo_key(n_bus, ei, b_ij, slack_idx=3)
    assert k_slack0 != k_slack3


def test_dc_prior_fingerprint_invalidates_on_impedance():
    cache = DCPriorCache()
    n_bus = 4
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    b1 = np.array([10.0, 10.0, 10.0])
    b2 = np.array([10.0, 5.0, 10.0])
    assert cache.topo_key(n_bus, ei, b1, 0) != cache.topo_key(n_bus, ei, b2, 0)
