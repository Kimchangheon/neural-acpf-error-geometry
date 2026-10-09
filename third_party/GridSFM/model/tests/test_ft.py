"""Fine-tuning + cross-version checkpoint load coverage."""
from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch

from gridsfm import (
    OPFDataAdapterDataset, SyntheticMixedDataset,
    compute_loss, eval_pass, finetune_opfdata, load_model,
)
from gridsfm.checkpoint import _hash_state_dict


ROOT = Path(__file__).resolve().parent.parent
DEVICE = os.environ.get("GRIDSFM_TEST_DEVICE", "cpu")

CKPT_V11 = ROOT / "checkpoints" / "gridsfm_open_v1.1.pt"


@pytest.fixture(scope="module")
def model():
    if not CKPT_V11.exists():
        pytest.skip(f"Checkpoint not found at {CKPT_V11}.")
    return load_model(CKPT_V11, device=DEVICE)


def test_load_v1_1(model):
    """v1.1 release ckpt loads strict, hash check passes, param count is correct."""
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 15_148_227, f"expected 15.15M params, got {n_params}"
    # W_global mean+max wiring: 8*hidden_dim = 1024 input cols
    assert tuple(model.fusion.W_global.weight.shape) == (128, 1024)


def _make_synth_v10_ckpt(tmp_path):
    """Synthesize a v1.0-shape ckpt from the v1.1 release: keep only the
    mean columns of W_global (cols 0..d, 2d..3d, 4d..5d, 6d..7d) and
    re-hash. Returns (synth_path, v10_w, d).
    """
    blob = torch.load(CKPT_V11, weights_only=True, map_location="cpu")
    sd = dict(blob["state_dict"])
    w11 = sd["fusion.W_global.weight"]
    # v1.1 W_global has shape (hidden_dim, 8 * hidden_dim); per-type block
    # width is in_dim / 8 (not out_dim) so the helper works at any hidden_dim.
    d = w11.size(1) // 8
    v10_w = torch.cat(
        [w11[:, t * 2 * d:t * 2 * d + d] for t in range(4)], dim=-1,
    ).contiguous()
    sd["fusion.W_global.weight"] = v10_w
    synth_path = tmp_path / "synth_v1.0.pt"
    torch.save({
        "state_dict": sd,
        "metadata": {"name": "GridSFM-open-v1.0-synth", "hash": _hash_state_dict(sd)},
    }, synth_path)
    return synth_path, v10_w, d


def test_load_v1_0_emits_deprecation_warning(tmp_path):
    """The v1.0 → v1.1 cross-version load must emit DeprecationWarning so
    downstream code can detect the deprecated checkpoint format."""
    if not CKPT_V11.exists():
        pytest.skip(f"Checkpoint not found at {CKPT_V11}.")
    synth_path, _, _ = _make_synth_v10_ckpt(tmp_path)
    with pytest.warns(DeprecationWarning, match="v1.0-shape"):
        _ = load_model(synth_path, device=DEVICE)


def test_load_v1_0_cross_version_adapt(tmp_path):
    """Synthesize a v1.0-shape ckpt by cropping W_global to 4d width and
    verify load_model permutes the v1.0 type-mean columns into the v1.1
    interleaved (mean, max) slots.

    v1.0 layout (4d wide): [mean_bus | mean_ac | mean_tr | mean_cyc]
    v1.1 layout (8d wide): [mean_bus | max_bus | mean_ac | max_ac |
                            mean_tr  | max_tr  | mean_cyc| max_cyc]
    """
    if not CKPT_V11.exists():
        pytest.skip(f"Checkpoint not found at {CKPT_V11}.")
    synth_path, v10_w, d = _make_synth_v10_ckpt(tmp_path)
    m = load_model(synth_path, device=DEVICE)
    w = m.fusion.W_global.weight.detach().cpu()
    assert tuple(w.shape) == (128, 8 * d), "model arch is v1.1 (8d wide)"
    # Each type's mean-slot in v1.1 should carry the v1.0 type-mean column;
    # each max-slot should be zero (init contract).
    for type_idx in range(4):
        mean_lo = type_idx * 2 * d
        max_lo = mean_lo + d
        assert torch.allclose(w[:, mean_lo:mean_lo + d], v10_w[:, type_idx * d:(type_idx + 1) * d]), \
            f"type {type_idx} mean-slot did not receive the v1.0 mean column"
        assert w[:, max_lo:max_lo + d].abs().max().item() == 0.0, \
            f"type {type_idx} max-slot must be zeroed by the adapter"

    # Smoke-test forward: a typo in the column permutation (off-by-one
    # block, swapped src/dst) wouldn't fail the slice assertions if the
    # ranges happen to align, but would produce NaN / garbage from a
    # broken fusion. Run one batch and assert all outputs finite.
    if not Path("/data/OPF/opfdata_pyg_train").exists():
        pytest.skip("OPFData cache not found; skipping cross-version forward smoke-test")
    from torch_geometric.loader import DataLoader as PygLoader

    from gridsfm.cycle_basis import CycleBasisCache, prepare_for_grid_transformer_
    from gridsfm.pe_features import LaplacianFactorizationCache, attach_pe_features_

    cb = CycleBasisCache()
    pe = LaplacianFactorizationCache()

    def tfm(d_):
        prepare_for_grid_transformer_(d_, cache=cb)
        attach_pe_features_(d_, cache=pe)
        return d_

    base = OPFDataAdapterDataset(
        root="/data/OPF/opfdata_pyg_train",
        case_name="pglib_opf_case500_goc",
        variant="fulltop", split="train", n_graphs=2,
        transform=tfm,
    )
    batch = next(iter(PygLoader(base, batch_size=2, shuffle=False))).to(DEVICE)
    m(batch)
    bp = batch["bus"].pred
    gp = batch["generator"].pred
    assert torch.isfinite(bp).all(), "cross-version-adapted model produced non-finite bus pred"
    assert torch.isfinite(gp).all(), "cross-version-adapted model produced non-finite gen pred"


def test_load_hash_mismatch_raises(tmp_path):
    """Corrupted bytes must fail the hash check before any shape adapt."""
    if not CKPT_V11.exists():
        pytest.skip(f"Checkpoint not found at {CKPT_V11}.")
    blob = torch.load(CKPT_V11, weights_only=True, map_location="cpu")
    sd = dict(blob["state_dict"])
    sd["fusion.W_global.weight"] = sd["fusion.W_global.weight"] + 1e-3
    bad = {
        "state_dict": sd,
        "metadata": dict(blob["metadata"]),
    }
    bad_path = tmp_path / "corrupted.pt"
    torch.save(bad, bad_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_model(bad_path, device=DEVICE)


@pytest.fixture(scope="module")
def small_loader():
    """Tiny train loader: 4 graphs from case500_goc, batch_size=2, synth-wrapped."""
    root = "/data/OPF/opfdata_pyg_train"
    if not Path(root).exists():
        pytest.skip(f"OPFData cache not found at {root}")
    from torch_geometric.loader import DataLoader

    from gridsfm.cycle_basis import CycleBasisCache, prepare_for_grid_transformer_
    from gridsfm.pe_features import LaplacianFactorizationCache, attach_pe_features_

    cb = CycleBasisCache()
    pe = LaplacianFactorizationCache()

    def tfm(d):
        prepare_for_grid_transformer_(d, cache=cb)
        attach_pe_features_(d, cache=pe)
        return d

    base = OPFDataAdapterDataset(
        root=root, case_name="pglib_opf_case500_goc",
        variant="fulltop", split="train", n_graphs=4,
    )
    ds = SyntheticMixedDataset(
        base, seed=42, transform=tfm,
        mode_weights=(0.25, 0.20, 0.20, 0.35), infeas_prob=0.5,
    )
    return DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)


def test_compute_loss_forward_backward(model, small_loader):
    """One-batch FT step end-to-end: forward, all parts finite, backward, gradient finite."""
    torch.manual_seed(42)
    batch = next(iter(small_loader)).to(DEVICE)
    model.train()
    _ = model(batch)
    loss, parts = compute_loss(batch)
    assert torch.isfinite(loss), f"loss is non-finite: {loss}"
    expected_keys = {
        "L_total",
        "L_theta", "L_V", "L_Pg", "L_Qg",
        "L_feas", "L_cost", "L_stress_feas",
        "L_kcl_p", "L_kcl_q", "L_br_p", "L_br_q", "L_therm", "L_therm_lim",
    }
    assert set(parts.keys()) == expected_keys, \
        f"parts keys mismatch:\n  missing: {expected_keys - set(parts.keys())}\n  extra: {set(parts.keys()) - expected_keys}"
    for k, v in parts.items():
        assert math.isfinite(v), f"non-finite parts[{k}] = {v}"
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradient flow"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"


def test_synthetic_mode_dispatch():
    """Each of the 4 modes fires correctly when its weight is 1.0."""
    if not Path("/data/OPF/opfdata_pyg_train").exists():
        pytest.skip("OPFData cache not found")
    base = OPFDataAdapterDataset(
        root="/data/OPF/opfdata_pyg_train",
        case_name="pglib_opf_case500_goc",
        variant="fulltop", split="train", n_graphs=4,
    )
    weights = [(1.0, 0.0, 0.0, 0.0),
               (0.0, 1.0, 0.0, 0.0),
               (0.0, 0.0, 1.0, 0.0),
               (0.0, 0.0, 0.0, 1.0)]
    for idx, w in enumerate(weights):
        ds = SyntheticMixedDataset(base, seed=42, mode_weights=w, infeas_prob=1.0)
        modes_seen = set()
        for i in range(len(ds)):
            g = ds[i]
            # _perturb_* may early-return mutated=False (e.g. n_edges<3) so
            # not every graph flips to infeasible. But if any graph did flip,
            # its mode must match the slot.
            if int(g.feasible.item()) == 0:
                modes_seen.add(int(g.perturb_mode.item()))
        assert modes_seen <= {idx}, \
            f"weight slot {idx} produced modes {modes_seen}, expected {{{idx}}}"
        assert modes_seen, f"no perturbation fired for mode {idx} despite infeas_prob=1.0"


def test_eval_pass_shapes(model, small_loader):
    """eval_pass returns a finite metrics dict with the documented keys."""
    res = eval_pass(model, small_loader, device=DEVICE)
    for k in ("loss", "cost_mape", "pg_mae", "qg_mae", "V_mae", "theta_mae",
              "feas_acc", "n_graphs"):
        assert k in res, f"missing key {k} in eval_pass result"
        if k == "n_graphs":
            assert isinstance(res[k], int)
        else:
            assert isinstance(res[k], float) and res[k] == res[k], f"{k} not a finite float"
    assert 0.0 <= res["feas_acc"] <= 1.0, f"feas_acc out of [0,1]: {res['feas_acc']}"
    # Untrained-on-this-grid model should have non-zero regression error;
    # a stuck-at-zero head would slip through a shape-only check.
    assert res["pg_mae"] > 0.0, f"pg_mae stuck at 0; head likely degenerate"
    assert res["V_mae"] > 0.0, f"V_mae stuck at 0; head likely degenerate"
    assert res["n_graphs"] > 0, "no feasible graphs in eval loader; fixture broken"


def test_eval_pass_loss_kwargs_passthrough(model, small_loader):
    """eval_pass(..., loss_kwargs={...}) must use the overridden lambdas
    so val_loss stays comparable to train_loss when lambdas are tuned."""
    base = eval_pass(model, small_loader, device=DEVICE)
    bumped = eval_pass(model, small_loader, device=DEVICE,
                       loss_kwargs={"lambda_thermal_limit": 100.0})
    assert bumped["loss"] > base["loss"], \
        "lambda_thermal_limit=100 should increase loss vs the default 5.0"
    # Non-loss metrics depend only on the model, not on lambdas. Allow
    # ~1e-4 numerical noise from non-deterministic scatter_add on GPU.
    for k in ("cost_mape", "pg_mae", "qg_mae", "V_mae", "theta_mae"):
        assert abs(bumped[k] - base[k]) < 1e-4, f"{k} drifted: {base[k]} -> {bumped[k]}"


def test_finetune_opfdata_one_epoch(model, small_loader):
    """One full FT epoch: optimizer step actually changes weights and
    finetune_opfdata returns a log entry with the expected keys."""
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    log = finetune_opfdata(
        model, small_loader, val_loader=small_loader,
        epochs=1, lr=1e-3, weight_decay=0.0,
    )
    assert len(log) == 1
    entry = log[0]
    # Static keys + every eval_pass key prefixed with `val_` (single source of
    # truth: derive expectations from a real eval_pass run on the same loader).
    expected_val_keys = {f"val_{k}"
                         for k in eval_pass(model, small_loader, device=DEVICE)}
    for k in (
        {"epoch", "train_loss", "n_train_iters", "n_train_skipped",
         "epoch_s", "elapsed_s"}
        | expected_val_keys
    ):
        assert k in entry, f"missing key {k} in finetune_opfdata log entry"
    assert entry["epoch_s"] <= entry["elapsed_s"], \
        "per-epoch wall must not exceed cumulative elapsed"
    # Real FT step should move some transformer-block weights by at
    # least 1e-4 relative; a stuck-grad / LR=0 / detached-graph regression
    # would slip past a default `allclose` (rtol=1e-5).
    max_rel = 0.0
    for k, v_new in model.state_dict().items():
        if v_new.dtype not in (torch.float16, torch.float32, torch.float64,
                                torch.bfloat16):
            continue
        old = snapshot[k].cpu()
        new = v_new.detach().cpu()
        denom = old.abs().max().clamp_min(1e-8)
        rel = ((new - old).abs() / denom).max().item()
        if rel > max_rel:
            max_rel = rel
    assert max_rel > 1e-4, \
        f"weights barely moved after a full FT epoch (max_rel={max_rel:.2e}); " \
        f"check LR / grad path / detached graph"
    # Restore the original weights for downstream tests.
    target_device = next(model.parameters()).device
    model.load_state_dict({k: v.to(target_device) for k, v in snapshot.items()})


def test_load_model_missing_arch_silent_when_default(tmp_path, caplog):
    """v1.1 release ships without metadata.arch by design; load must
    succeed silently when shapes already match the default backbone
    (i.e. nothing actually needed adaptation)."""
    if not CKPT_V11.exists():
        pytest.skip(f"Checkpoint not found at {CKPT_V11}.")
    blob = torch.load(CKPT_V11, weights_only=True, map_location="cpu")
    sd = dict(blob["state_dict"])
    meta_no_arch = {"name": blob["metadata"]["name"],
                    "hash": _hash_state_dict(sd)}
    no_arch_path = tmp_path / "no_arch.pt"
    torch.save({"state_dict": sd, "metadata": meta_no_arch}, no_arch_path)
    import logging as _logging
    with caplog.at_level(_logging.WARNING, logger="gridsfm.checkpoint"):
        _ = load_model(no_arch_path, device=DEVICE)
    noise = [r for r in caplog.records
             if "shape-adapted" in r.message or "metadata.arch" in r.message
             or "skipped" in r.message]
    assert not noise, \
        f"expected silent load on v1.1 default arch, got warnings: {[r.message for r in noise]}"


def test_load_model_arch_mismatch_raises(tmp_path):
    """A shape mismatch outside the v1.0 W_global signature must raise
    `RuntimeError` (not silently zero-pad/crop). Guards against an
    unintended ckpt/arch drift slipping through the hash check."""
    if not CKPT_V11.exists():
        pytest.skip(f"Checkpoint not found at {CKPT_V11}.")
    blob = torch.load(CKPT_V11, weights_only=True, map_location="cpu")
    sd = dict(blob["state_dict"])
    target_key = next(
        (k for k, v in sd.items()
         if k != "fusion.W_global.weight" and v.dim() == 2 and min(v.shape) > 1),
        None,
    )
    assert target_key is not None, "no 2D non-W_global tensor found to perturb"
    sd[target_key] = sd[target_key][:-1, :-1].contiguous()
    bad = {
        "state_dict": sd,
        "metadata": {"name": blob["metadata"]["name"], "hash": _hash_state_dict(sd)},
    }
    bad_path = tmp_path / "arch_mismatch.pt"
    torch.save(bad, bad_path)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        load_model(bad_path, device=DEVICE)


def test_load_model_unknown_ckpt_key_raises(tmp_path):
    """A ckpt key not present in the v1.1 backbone must raise
    `RuntimeError` (not be silently appended to `skipped` and dropped).
    Guards against a subset-overlap wrong-arch ckpt loading silently."""
    if not CKPT_V11.exists():
        pytest.skip(f"Checkpoint not found at {CKPT_V11}.")
    blob = torch.load(CKPT_V11, weights_only=True, map_location="cpu")
    sd = dict(blob["state_dict"])
    sd["fake.extra.weight"] = torch.zeros(2, 3)
    bad = {
        "state_dict": sd,
        "metadata": {"name": blob["metadata"]["name"], "hash": _hash_state_dict(sd)},
    }
    bad_path = tmp_path / "extra_key.pt"
    torch.save(bad, bad_path)
    with pytest.raises(RuntimeError, match="not present in the v1.1 backbone"):
        load_model(bad_path, device=DEVICE)


def test_load_model_missing_ckpt_key_raises(tmp_path):
    """A model parameter not present in the ckpt must raise
    `RuntimeError`. Otherwise the parameter ships at Kaiming init."""
    if not CKPT_V11.exists():
        pytest.skip(f"Checkpoint not found at {CKPT_V11}.")
    blob = torch.load(CKPT_V11, weights_only=True, map_location="cpu")
    sd = dict(blob["state_dict"])
    target_key = next((k for k in sd if "weight" in k), None)
    assert target_key is not None, "no weight key found to remove"
    del sd[target_key]
    bad = {
        "state_dict": sd,
        "metadata": {"name": blob["metadata"]["name"], "hash": _hash_state_dict(sd)},
    }
    bad_path = tmp_path / "missing_key.pt"
    torch.save(bad, bad_path)
    with pytest.raises(RuntimeError, match="not present in the ckpt"):
        load_model(bad_path, device=DEVICE)
