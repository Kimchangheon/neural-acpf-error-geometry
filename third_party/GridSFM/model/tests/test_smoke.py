"""Smoke test: load checkpoint, run predict() on a shipped sample."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from gridsfm import (
    load_model, predict, prepare_for_inference, load_pyg_json, batch_data_list,
)

ROOT = Path(__file__).resolve().parent.parent


CKPT = ROOT / "checkpoints" / "gridsfm_open_v1.1.pt"
SAMPLE = ROOT / "samples" / "case500_goc.pyg.json"
DEVICE = os.environ.get("GRIDSFM_TEST_DEVICE", "cpu")


@pytest.fixture(scope="module")
def model():
    if not CKPT.exists():
        pytest.skip(f"Checkpoint not found at {CKPT}; "
                    f"run hf download to fetch it.")
    return load_model(str(CKPT), device=DEVICE)


def test_single_predict_shapes(model):
    # Derive expected shapes from the raw scenario rather than hard-coding,
    # so the test stays valid if SAMPLE is swapped for a different grid.
    data = load_pyg_json(str(SAMPLE))
    n_bus = int(data["bus"].x.size(0))
    n_gen = int(data["generator"].x.size(0))
    out = predict(model, str(SAMPLE))
    assert out["V"].shape[0] == n_bus
    assert out["theta"].shape[0] == n_bus
    assert out["Pg"].shape[0] == n_gen
    assert out["Qg"].shape[0] == n_gen
    assert torch.isfinite(out["V"]).all()
    assert torch.isfinite(out["theta"]).all()
    assert torch.isfinite(out["Pg"]).all()
    assert torch.isfinite(out["Qg"]).all()
    assert 0.0 <= out["feas"] <= 1.0


def test_batched_matches_single(model):
    out_single = predict(model, str(SAMPLE))
    data = prepare_for_inference(load_pyg_json(str(SAMPLE)))
    batch = batch_data_list([data]).to(DEVICE)
    with torch.no_grad():
        bout = model(batch)
    bus_pred = bout["bus"].pred.cpu()
    gen_pred = bout["generator"].pred.cpu()
    # 2e-3 covers GPU FP32 reduction-order non-determinism between single
    # and batched forward (Qg via scatter_reduce has the largest spread).
    torch.testing.assert_close(out_single["theta"], bus_pred[:, 0], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(out_single["V"],     bus_pred[:, 1], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(out_single["Pg"],    gen_pred[:, 0], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(out_single["Qg"],    gen_pred[:, 1], atol=2e-3, rtol=2e-3)


def test_predict_rejects_multi_graph_batch(model):
    d1 = prepare_for_inference(load_pyg_json(str(SAMPLE)))
    d2 = prepare_for_inference(load_pyg_json(str(SAMPLE)))
    batch = batch_data_list([d1, d2])
    with pytest.raises(ValueError, match=r"single scenario"):
        predict(model, batch)


def test_predict_flow_edge_types_align(model):
    out = predict(model, str(SAMPLE))
    total = sum(out["flow_edge_counts"])
    assert total == out["Pij"].shape[0]
    assert len(out["flow_edge_types"]) == len(out["flow_edge_counts"])
    assert "ac_line" in out["flow_edge_types"]
    assert "transformer" in out["flow_edge_types"]
