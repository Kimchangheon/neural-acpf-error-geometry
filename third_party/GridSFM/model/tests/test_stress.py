"""Shape + invariant checks for compute_physics_stress."""
from __future__ import annotations

from pathlib import Path

import torch

from gridsfm import batch_data_list, load_pyg_json, prepare_for_inference
from gridsfm.stress_features import STRESS_DIM, compute_physics_stress

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "case500_goc.pyg.json"


def _zero_pred(data):
    n_bus = data['bus'].x.size(0)
    n_gen = data['generator'].x.size(0)
    V = torch.ones(n_bus)
    theta = torch.zeros(n_bus)
    Pg = torch.zeros(n_gen)
    Qg = torch.zeros(n_gen)
    return V, theta, Pg, Qg


def test_shape_single_graph():
    data = prepare_for_inference(load_pyg_json(str(SAMPLE)))
    V, th, Pg, Qg = _zero_pred(data)
    s = compute_physics_stress(data, V, th, Pg, Qg, n_graphs=1)
    assert s.shape == (1, STRESS_DIM)
    assert torch.isfinite(s).all()


def test_shape_batched():
    prepared = [prepare_for_inference(load_pyg_json(str(SAMPLE))) for _ in range(3)]
    batch = batch_data_list(prepared)
    V = torch.ones(batch['bus'].x.size(0))
    th = torch.zeros(batch['bus'].x.size(0))
    Pg = torch.zeros(batch['generator'].x.size(0))
    Qg = torch.zeros(batch['generator'].x.size(0))
    s = compute_physics_stress(batch, V, th, Pg, Qg, n_graphs=3)
    assert s.shape == (3, STRESS_DIM)
    assert torch.isfinite(s).all()


def test_per_graph_pooling_aggregates_independently():
    """Same scenario stacked 3x must give identical stress rows (deterministic pooling)."""
    prepared = [prepare_for_inference(load_pyg_json(str(SAMPLE))) for _ in range(3)]
    batch = batch_data_list(prepared)
    V = torch.ones(batch['bus'].x.size(0))
    th = torch.zeros(batch['bus'].x.size(0))
    Pg = torch.zeros(batch['generator'].x.size(0))
    Qg = torch.zeros(batch['generator'].x.size(0))
    s = compute_physics_stress(batch, V, th, Pg, Qg, n_graphs=3)
    torch.testing.assert_close(s[0], s[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(s[1], s[2], atol=1e-6, rtol=1e-6)
