"""OPFData loader: bus-column validation and round-trip predict()."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch

from gridsfm import load_pyg_json, load_opfdata, predict, load_model

ROOT = Path(__file__).resolve().parent.parent


SAMPLE = ROOT / "samples" / "case500_goc.pyg.json"
CKPT = ROOT / "checkpoints" / "gridsfm_open_v1.1.pt"
DEVICE = os.environ.get("GRIDSFM_TEST_DEVICE", "cpu")


def _read_sample_as_opfdata_json() -> dict:
    with open(SAMPLE) as f:
        return json.load(f)  # already has {grid: {nodes, edges}}


def test_load_opfdata_accepts_canonical_layout():
    obj = _read_sample_as_opfdata_json()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(obj, f)
        path = f.name
    try:
        d = load_opfdata(path)
        assert d["bus"].x.size(0) > 0
        assert d["generator"].x.size(0) > 0
    finally:
        os.unlink(path)


def test_load_opfdata_rejects_flat_layout():
    obj = _read_sample_as_opfdata_json()["grid"]  # drop the 'grid' wrapper
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(obj, f)
        path = f.name
    try:
        with pytest.raises(ValueError, match=r"top-level 'grid'"):
            load_opfdata(path)
    finally:
        os.unlink(path)


def test_load_opfdata_rejects_too_narrow_bus():
    obj = _read_sample_as_opfdata_json()
    obj["grid"]["nodes"]["bus"] = [row[:2] for row in obj["grid"]["nodes"]["bus"]]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(obj, f)
        path = f.name
    try:
        with pytest.raises(ValueError, match=r"bus rows have"):
            load_opfdata(path)
    finally:
        os.unlink(path)


def test_load_opfdata_rejects_too_wide_bus():
    obj = _read_sample_as_opfdata_json()
    obj["grid"]["nodes"]["bus"] = [row + [0.0, 0.0] for row in obj["grid"]["nodes"]["bus"]]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(obj, f)
        path = f.name
    try:
        with pytest.raises(ValueError, match=r"bus rows have"):
            load_opfdata(path)
    finally:
        os.unlink(path)


def test_load_opfdata_rejects_wider_ac_line():
    obj = _read_sample_as_opfdata_json()
    ac = obj["grid"]["edges"].get("ac_line", {}).get("features")
    if not ac:
        pytest.skip("sample has no ac_line edges")
    obj["grid"]["edges"]["ac_line"]["features"] = [row + [0.0, 0.0] for row in ac]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(obj, f)
        path = f.name
    try:
        with pytest.raises(ValueError, match=r"edges.ac_line.features rows have"):
            load_opfdata(path)
    finally:
        os.unlink(path)


@pytest.fixture(scope="module")
def model():
    if not CKPT.exists():
        pytest.skip(f"Checkpoint not found at {CKPT}")
    return load_model(str(CKPT), device=DEVICE)


def test_predict_opfdata_roundtrip(model):
    out_pyg = predict(model, str(SAMPLE))
    obj = _read_sample_as_opfdata_json()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(obj, f)
        path = f.name
    try:
        out_opf = predict(model, path, fmt="opfdata")
    finally:
        os.unlink(path)
    torch.testing.assert_close(out_pyg["theta"], out_opf["theta"], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(out_pyg["V"],     out_opf["V"],     atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(out_pyg["Pg"],    out_opf["Pg"],    atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(out_pyg["Qg"],    out_opf["Qg"],    atol=2e-3, rtol=2e-3)
