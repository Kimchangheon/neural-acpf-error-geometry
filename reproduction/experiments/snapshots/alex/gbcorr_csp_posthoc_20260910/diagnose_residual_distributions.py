#!/usr/bin/env python3
"""Detailed, paired AC-residual distribution diagnostics.

This evaluator intentionally scores every held-out bus in the same test split,
then reports pooled and per-scenario-tail statistics.  It is independent of
the training reducer: in particular, ``mean_case_max`` is computed explicitly
and all residuals are recomputed with complex128 Y-bus arithmetic.
"""
from __future__ import annotations

import argparse
import json
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag, ybus_matvec


def angle_diff(a, b):
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def pct(x):
    return float(100.0 * x)


def summary(values: np.ndarray, tol: float = 0.01):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    ordered = np.sort(values)
    total = max(float(values.sum()), 1e-300)
    def top_mass(frac):
        count = max(1, math.ceil(frac * values.size))
        return float(ordered[-count:].sum() / total)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.quantile(values, .50)),
        "p95": float(np.quantile(values, .95)),
        "p99": float(np.quantile(values, .99)),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "global_max": float(values.max()),
        "frac_le_0.01": pct(np.mean(values <= tol)),
        "frac_gt_0.1": pct(np.mean(values > .1)),
        "frac_gt_1": pct(np.mean(values > 1.0)),
        "frac_gt_10": pct(np.mean(values > 10.0)),
        "frac_gt_100": pct(np.mean(values > 100.0)),
        "top_mass_0.1pct": top_mass(.001),
        "top_mass_1pct": top_mass(.01),
        "top_mass_5pct": top_mass(.05),
    }


def scenario_summary(values, scenario):
    vals = []
    for g in np.unique(scenario):
        x = values[scenario == g]
        vals.append(float(x.max()) if x.size else 0.0)
    vals = np.asarray(vals, dtype=np.float64)
    out = summary(vals)
    out["mean_per_scenario_max"] = float(vals.mean()) if vals.size else 0.0
    out["scenario_frac_gt_0.1"] = pct(np.mean(vals > .1)) if vals.size else 0.0
    out["scenario_frac_gt_1"] = pct(np.mean(vals > 1.0)) if vals.size else 0.0
    out["scenario_frac_gt_10"] = pct(np.mean(vals > 10.0)) if vals.size else 0.0
    out["scenario_frac_gt_100"] = pct(np.mean(vals > 100.0)) if vals.size else 0.0
    return out


def grouped(values, key, name):
    key = np.asarray(key)
    out = {}
    for k in np.unique(key):
        x = values[key == k]
        out[str(k)] = summary(x)
    return {name: out}


def add_group(out, values, key, name):
    out[name] = {str(k): summary(values[key == k]) for k in np.unique(key)}


def add_yii_bins(out, values, yii):
    """Log-spaced |Yii| groups; exact floating values would create thousands of rows."""
    yii = np.asarray(yii, dtype=np.float64)
    positive = yii[yii > 0]
    if positive.size == 0:
        out["abs_Yii_log10_bins"] = {"zero": summary(values)}
        return
    lo = math.floor(float(np.log10(positive.min())))
    hi = math.ceil(float(np.log10(positive.max())))
    edges = np.arange(lo, hi + 1, dtype=np.float64)
    if edges.size < 2:
        edges = np.asarray([lo, lo + 1.0])
    labels = np.digitize(np.log10(np.maximum(yii, 10.0 ** lo)), edges[1:-1], right=False)
    groups = {}
    for index in np.unique(labels):
        left, right = edges[index], edges[index + 1]
        groups[f"1e{left:g}_to_1e{right:g}"] = summary(values[labels == index])
    out["abs_Yii_log10_bins"] = groups


def fit_per_bus_baseline(train_subset, batch_size):
    """Per-bus constant predictor fitted on the training split only.

    |V| gets the arithmetic per-bus mean; theta gets the per-bus circular mean
    via the resultant of the unit phasors, so a bus sitting near +/-pi does not
    average to zero.  This is the "learned nothing" reference the R^2 tables in
    the reports measure against, so it must never see the test scenarios.
    """
    loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_blockdiag)
    sum_vm = sum_sin = sum_cos = None
    n_scen = 0
    for batch in loader:
        sizes = batch["sizes"].numpy().astype(int)
        if len(set(sizes.tolist())) != 1:
            raise RuntimeError("per-bus baseline needs one grid per campaign")
        N = int(sizes[0])
        V = batch["V_newton"][0].to(torch.float64).reshape(len(sizes), N, 2)
        vm = V[..., 0].sum(0).numpy()
        sn = torch.sin(V[..., 1]).sum(0).numpy()
        cs = torch.cos(V[..., 1]).sum(0).numpy()
        if sum_vm is None:
            sum_vm, sum_sin, sum_cos = vm, sn, cs
        else:
            sum_vm += vm; sum_sin += sn; sum_cos += cs
        n_scen += len(sizes)
    vm_bar = sum_vm / n_scen
    th_bar = np.arctan2(sum_sin / n_scen, sum_cos / n_scen)
    print(f"[baseline] fitted on {n_scen} training scenarios, {len(vm_bar)} buses",
          flush=True)
    return vm_bar, th_bar


def manifold_basis(train_subset, batch_size, k=32):
    """Training-split bases for |V| and theta, via the shared fitter."""
    from manifold_projection import fit_basis

    loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_blockdiag)
    vm, th = [], []
    for batch in loader:
        sizes = batch["sizes"].numpy().astype(int); N = int(sizes[0])
        V = batch["V_newton"][0].to(torch.float64).reshape(len(sizes), N, 2)
        vm.append(V[..., 0]); th.append(V[..., 1])
    VM, TH = torch.cat(vm, 0), torch.cat(th, 0)
    (Uv, Ut, vbar, tbar), ev, et = fit_basis(VM, TH, k)
    print(f"[manifold] {VM.shape[0]} train scenarios, {VM.shape[1]} buses; "
          f"k={Uv.shape[1]} captures {100*ev:.3f}% of |V| and {100*et:.3f}% "
          f"of theta variance", flush=True)
    return Uv, Ut, vbar, tbar, ev, et


def project_to_manifold(V, basis):
    """Scorer-side wrapper over the shared projector; see manifold_projection."""
    from manifold_projection import project_packed

    return project_packed(V, basis)


def reference_spreads(loader):
    """sigma of the reference |V| and theta under both baselines.

    "per-bus" subtracts each bus's own mean (the convention used by the report
    tables); "global" subtracts a single mean over every bus-scenario entry
    (what train_valid_test.py used to do).  Angles are wrapped about the
    per-bus circular mean.
    """
    vm_chunks = []; th_chunks = []
    for batch in loader:
        sizes = batch["sizes"].numpy().astype(int); N = int(sizes[0])
        V = batch["V_newton"][0].to(torch.float64).reshape(len(sizes), N, 2)
        vm_chunks.append(V[..., 0].numpy()); th_chunks.append(V[..., 1].numpy())
    VM = np.concatenate(vm_chunks); TH = np.concatenate(th_chunks)
    th_bar = np.arctan2(np.sin(TH).mean(0), np.cos(TH).mean(0))
    dth = np.arctan2(np.sin(TH - th_bar), np.cos(TH - th_bar))
    return {
        "vm_sigma_per_bus": float((VM - VM.mean(0)).std()),
        "vm_sigma_global": float(VM.std()),
        "theta_sigma_per_bus_deg": float(np.sqrt((dth ** 2).mean()) * 180 / math.pi),
        "theta_sigma_global_deg": float(
            np.sqrt((np.arctan2(np.sin(TH - np.arctan2(np.sin(TH).mean(), np.cos(TH).mean())),
                                np.cos(TH - np.arctan2(np.sin(TH).mean(), np.cos(TH).mean()))) ** 2).mean())
            * 180 / math.pi),
        "theta_sigma_per_bus_circvar_deg": float(
            np.sqrt(2 * (1 - np.hypot(np.sin(TH).mean(0), np.cos(TH).mean(0))).mean())
            * 180 / math.pi),
    }


def angle_stats(err_rad):
    """Summary of a wrapped angle error array, in degrees."""
    e = np.abs(err_rad) * 180.0 / math.pi
    return {"mae": float(e.mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "median": float(np.median(e)), "p95": float(np.percentile(e, 95)),
            "max": float(e.max()), "n": int(e.size)}


def branch_angle_diagnostics(th_pred, th_ref, f, t, status, scen):
    """Per-edge angle-difference error, and the gauge decomposition.

    Branch flows depend on theta_i - theta_j, not on the absolute bus angle, so
    a model whose angles are all shifted by the same scenario-dependent amount
    posts a large per-bus theta RMSE while predicting every flow correctly.
    GridSFM's paper makes exactly this point about gridfm-graphkit (per-bus
    theta MAE inflated by "uncontrolled angle-gauge drift"; per-edge dtheta_ij
    MAE 0.146 deg on the same checkpoint).  Reporting per-bus RMSE alone cannot
    tell the two apart, so this returns:

      per_bus      : the raw wrapped bus-angle error
      per_bus_degauged : the same after removing, per scenario, the mean
                     pred-minus-ref offset -- what is left once a pure gauge
                     shift is forgiven
      gauge        : the removed offset itself
      per_edge     : the wrapped error on theta_f - theta_t over in-service
                     branches, which is what the flows actually see
    """
    wrap = lambda x: np.arctan2(np.sin(x), np.cos(x))  # noqa: E731
    d_bus = wrap(th_pred - th_ref)
    st = status.astype(bool)
    d_edge = wrap(wrap(th_pred[f] - th_pred[t]) - wrap(th_ref[f] - th_ref[t]))[st]

    # Per-scenario gauge: the circular mean of the bus-angle error.
    gauge = np.zeros_like(d_bus)
    for sid in np.unique(scen):
        m = scen == sid
        g = math.atan2(float(np.sin(d_bus[m]).mean()), float(np.cos(d_bus[m]).mean()))
        gauge[m] = g
    return {
        "per_bus": angle_stats(d_bus),
        "per_bus_degauged": angle_stats(wrap(d_bus - gauge)),
        "gauge": angle_stats(gauge[np.unique(scen, return_index=True)[1]]),
        "per_edge": angle_stats(d_edge),
    }


def fit_per_bus_offset(model, forward, train_subset, batch_size, device):
    """Per-bus mean prediction offset, measured on the TRAINING split.

    The error decomposition says most of the |V| error of every model here is a
    fixed per-bus offset rather than a failure to track the scenario.  That is
    a claim about a correction that must be learnable without touching the test
    set, so the offset is fitted here on the training third only -- exactly
    like the per-bus baseline predictor -- and then subtracted at test time.
    Fitting it on the test split would make the correction unfalsifiable.

    Returns (dvm[n_bus], dtheta[n_bus]); dtheta is a circular mean.
    """
    loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_blockdiag)
    sum_dvm = sum_sin = sum_cos = None
    n = 0
    with torch.no_grad():
        for batch in loader:
            sizes = batch["sizes"].numpy().astype(int)
            N = int(sizes[0])
            Vp = forward(batch).detach().to(torch.float64)
            Vr = batch["V_newton"].to(device=Vp.device, dtype=torch.float64)
            dvm = (Vp[0, :, 0] - Vr[0, :, 0]).reshape(-1, N).sum(0)
            dth = torch.atan2(torch.sin(Vp[0, :, 1] - Vr[0, :, 1]),
                              torch.cos(Vp[0, :, 1] - Vr[0, :, 1])).reshape(-1, N)
            sn, cs = torch.sin(dth).sum(0), torch.cos(dth).sum(0)
            if sum_dvm is None:
                sum_dvm, sum_sin, sum_cos = dvm, sn, cs
            else:
                sum_dvm += dvm; sum_sin += sn; sum_cos += cs
            n += len(sizes)
    print(f"[debias] per-bus offset fitted on {n} training scenarios; "
          f"rms dvm {float((sum_dvm / n).square().mean() ** 0.5):.5e} pu", flush=True)
    return sum_dvm / n, torch.atan2(sum_sin / n, sum_cos / n)


def voltage_roughness(vm_pred, vm_ref, f, t, status, yij=None):
    """Is the voltage error a smooth field or per-bus noise?

    The AC residual is a difference operator: a voltage error shared by two
    neighbouring buses largely cancels in dS_i, while an error that differs
    between them is multiplied by the branch admittance.  So what matters is
    not the size of eps_i = |V|_pred - |V|_ref but its variation across a
    branch.

    The diagnostic statistic is
        ratio = rms_edge / rms_bus,
    with rms_edge over |eps_i - eps_j| on in-service branches.  For a field
    that is white noise across buses this tends to sqrt(2) ~ 1.414; for a
    perfectly smooth field it tends to 0.  A model can therefore have a small
    rms_bus and still produce large residuals, if its ratio is near sqrt(2).
    """
    eps = vm_pred - vm_ref
    st = status.astype(bool)
    d = (eps[f] - eps[t])[st]
    rms_bus = float(np.sqrt((eps ** 2).mean()))
    rms_edge = float(np.sqrt((d ** 2).mean()))
    out = {"rms_bus": rms_bus, "rms_edge": rms_edge,
           "ratio": rms_edge / max(rms_bus, 1e-30),
           "mean_edge": float(np.abs(d).mean()),
           "p95_edge": float(np.percentile(np.abs(d), 95)),
           "max_edge": float(np.abs(d).max()), "n_edge": int(d.size)}
    if yij is not None:
        # What the residual actually sees: the branch-admittance-weighted
        # roughness, in pu of power rather than of voltage.
        w = yij[st]
        out["rms_edge_yweighted"] = float(np.sqrt(((w * d) ** 2).mean()))
        out["mean_edge_yweighted"] = float(np.abs(w * d).mean())
    return out


def residual_split(Y, V, S_set, bus_type, matvec, n_bus):
    r"""Split the mismatch into its local and coupling parts.

    dS_i = S_i^set - V_i conj(sum_j Y_ij V_j)
         = [S_i^set - V_i conj(Y_ii V_i)] - V_i conj(sum_{j!=i} Y_ij V_j)
           \-------- local --------/   \------- coupling -------/

    A smooth error cancels in the coupling term but not in the local one, which
    is proportional to |Y_ii||V_i|^2.  Reporting them apart says which of the
    two the model-versus-baseline gap lives in, and therefore whether spatial
    smoothness could close it at all.  It is also the obvious candidate
    explanation for GridSFM, whose admittance-weighted edge roughness is the
    largest of any model by a factor of fifteen while its residual is the
    smallest: large coupling errors that cancel, with a well-controlled local
    term, would look exactly like that.
    """
    v = V[..., 0].reshape(-1).to(torch.float64)
    th = V[..., 1].reshape(-1).to(torch.float64)
    Vc = (v * torch.exp(1j * th)).to(torch.complex128)
    full = Vc * matvec(Y, Vc).conj()
    ydiag = _ybus_diag_complex(Y)
    local = Vc * (ydiag * Vc).conj()
    coupling = full - local
    bt = bus_type.reshape(-1)
    p_mask = bt != 1
    q_mask = (bt != 1) & (bt != 2)
    out = {}
    for tag, mask in (("p", p_mask), ("q", q_mask)):
        tot = (S_set.real if tag == "p" else S_set.imag) - (
            full.real if tag == "p" else full.imag)
        loc = -(local.real if tag == "p" else local.imag)
        cpl = -(coupling.real if tag == "p" else coupling.imag)
        out[f"{tag}_total_rms"] = float((tot[mask] ** 2).mean() ** 0.5)
        out[f"{tag}_local_rms"] = float((loc[mask] ** 2).mean() ** 0.5)
        out[f"{tag}_coupling_rms"] = float((cpl[mask] ** 2).mean() ** 0.5)
    return out


def _ybus_diag_complex(Y):
    if Y.is_sparse:
        c = Y.coalesce()
        idx = c.indices()
        m = idx[0] == idx[1]
        d = torch.zeros(Y.shape[0], dtype=c.values().dtype, device=Y.device)
        d.index_add_(0, idx[0][m], c.values()[m])
        return d
    return torch.diagonal(Y)


def manifold_split(eps, basis):
    """Energy of the error inside and outside the scenario subspace.

    The per-bus constant predictor's error is, by construction, minus the
    scenario's own deviation from the training mean, so it lies entirely in the
    span of the reference voltage variation.  A learned prediction has no such
    guarantee.  This is the sharp version of "smooth": the question is not
    whether the error varies slowly but whether it is a voltage pattern the
    grid can actually produce.  `basis` is an orthonormal [n_bus, k] built from
    the training-split reference deviations.
    """
    proj = eps @ basis                      # [n_scen, k]
    inside = float((proj ** 2).sum())
    total = float((eps ** 2).sum())
    return {"frac_in_manifold": inside / max(total, 1e-30),
            "rms_in": (inside / max(eps.numel(), 1)) ** 0.5,
            "rms_out": (max(total - inside, 0.0) / max(eps.numel(), 1)) ** 0.5}


def pignn_model(mode, gate, residual_feature_norm="signed_log"):
    from GNSMsg_SelfAttention_armijo import GNSMsg_EdgeSelfAttn
    return GNSMsg_EdgeSelfAttn(
        d=4, d_hi=24, K=40, pinn=True, gamma=.9, v_limit=True,
        use_armijo=True, n_heads=8, num_attn_layers=8,
        armijo_mode="geometric_safe", solver_update_mode="direct",
        dtheta_max=.30, dvm_frac=.10, physics_loss_form="logcosh",
        physics_residual_norm="graph", residual_feature_norm=residual_feature_norm,
        edge_feature_norm="none", global_context_mode=mode,
        global_context_gate_mode=gate,
    )


def make_graphkit_or_lumina(kind, checkpoint, device, parquet, batch, model_config=None):
    """Build graphkit/LUMINA through their own drivers, as rescore_slope_r2 does.

    Their configurations live in the drivers' argparse defaults; rebuilding them
    here by hand would invent hyperparameters.  graphkit is constructed from the
    adapter directly because its driver wraps a third-party model.
    """
    import importlib, sys as _sys

    if kind == "graphkit":
        from gridfm_graphkit_adapter import build_graphkit_model, forward_graphkit_parquet
        model = build_graphkit_model(task_name="PowerFlow", hidden_size=48,
                                     num_layers=12, attention_head=8).to(device)
        fwd = lambda bd: forward_graphkit_parquet(          # noqa: E731
            model, bd, device, task_name="PowerFlow",
            feature_transform="signed_log", vn_feature_mode="log")
        to_dev = True
    else:
        mod = importlib.import_module("train_valid_test_lumina")
        argv = ["rescore", "--PARQUET", parquet, "--task", "pf", "--BATCH", str(batch),
                "--train_ratio", "0.3333", "--valid_ratio", "0.3333", "--seed_value", "42",
                "--init_mode", "scratch"]
        if model_config:
            argv += ["--model_config", model_config]
        saved, _sys.argv = _sys.argv, argv
        try:
            dargs = mod.parse_args()
        finally:
            _sys.argv = saved
        model = mod.make_model(dargs, device)
        fwd = lambda bd: mod.lumina_forward(model, bd, dargs, device)   # noqa: E731
        to_dev = False

    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}
    r = model.load_state_dict(state, strict=False)
    print(f"[{kind}] checkpoint missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}",
          flush=True)
    if r.missing_keys or r.unexpected_keys:
        raise RuntimeError(f"{kind} checkpoint/model mismatch; refusing a non-identical rescore")
    model.eval()
    return model, fwd, to_dev


def make_gridsfm_model(checkpoint, device):
    from gridsfm import GridTransformerBackbone
    m = GridTransformerBackbone().to(device)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    result = m.load_state_dict(state, strict=False)
    print(f"[GridSFM] checkpoint missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}", flush=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("GridSFM checkpoint/model mismatch; refusing a non-identical rescore")
    return m


def forward_gridsfm(model, batch, device):
    from train_valid_test_gridsfm import gridsfm_forward
    args = SimpleNamespace(
        # Exact defaults used by the ppNR-v2 dispatch.  In particular, the v2
        # command did not enable voltage-mismatch transformer reclassification.
        task="pf", pf_injection="signed", vmin=.5, vmax=1.5, rate_a=0.0,
        treat_voltage_mismatch_as_transformer=False,
    )
    return gridsfm_forward(model, batch, args, device)


def forward_pignn(model, batch, device):
    bt = batch["bus_type"].to(device)
    kw = [batch[k].to(device) for k in (
        "Branch_f_bus", "Branch_t_bus", "Branch_status", "Branch_tau",
        "Branch_shift_deg", "Branch_y_series_from", "Branch_y_series_to",
        "Branch_y_series_ft", "Branch_y_shunt_from", "Branch_y_shunt_to",
        "Is_trafo")]
    # Neural path remains float32/complex64; the independent residual score
    # below uses the original complex128 sparse Y-bus.
    tau, shift = kw[3].float(), kw[4].float()
    ys = [x.to(torch.complex64) for x in kw[5:10]]
    Y = batch.get("Ybus", None)
    Y = Y.to(device=device, dtype=torch.complex64) if Y is not None else None
    S = batch["S_start"].to(device=device, dtype=torch.complex64)
    V0 = batch["V_start"].to(device=device).float()
    out = model(
        bt, kw[0], kw[1], kw[2], tau, shift, ys[0], ys[1], ys[2], ys[3], ys[4],
        batch["Is_trafo"].to(device), Y, S, V0,
        n_nodes_per_graph=batch["sizes"].to(device),
        Y_shunt_bus=batch["Y_shunt_bus"].to(device=device, dtype=torch.complex64),
        vn_log=batch.get("vn_log", None).to(device).float() if batch.get("vn_log", None) is not None else None,
    )
    return out[0] if isinstance(out, tuple) else out


def diag(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    share = not args.varying_topology
    ds = ChanghunDataset(
        args.parquet, per_unit=True, target_S_base=1e8, share_grid=share,
        share_ybus=share, lazy_row_groups=True, row_group_cache_size=4,
        complex_dtype="complex128",
    )
    n = len(ds); ntr = int(.3333*n); nv = int(.3333*n)
    train, _, test = random_split(ds, [ntr, nv, n-ntr-nv], generator=torch.Generator().manual_seed(42))
    loader = DataLoader(test, batch_size=args.batch, shuffle=False, num_workers=0,
                        collate_fn=collate_blockdiag)
    # The basis and the calibration offset are fitted on `fit_split`, which is
    # `train` unless --fit_parquet names a different corpus.  Testing always
    # stays on `test` from --parquet.
    fit_split = train
    if args.fit_parquet:
        # --varying_topology describes the corpus under test.  The fit corpus
        # is a different file and may well have a fixed Y, in which case the
        # caching is valid there and worth keeping: fitting a basis and an
        # offset over a large fit split with the cache off costs hours.
        # --fit_share_grid asserts that; it is on the caller to be right about
        # the fit corpus, so it is opt-in rather than inferred.
        fit_share = share or args.fit_share_grid
        fit_ds = ChanghunDataset(
            args.fit_parquet, per_unit=True, target_S_base=1e8,
            share_grid=fit_share, share_ybus=fit_share, lazy_row_groups=True,
            row_group_cache_size=4, complex_dtype="complex128",
        )
        fn = len(fit_ds); fntr = int(.3333*fn); fnv = int(.3333*fn)
        fit_split, _, _ = random_split(
            fit_ds, [fntr, fnv, fn-fntr-fnv],
            generator=torch.Generator().manual_seed(42))
        print(f"[fit] basis/offset from {args.fit_parquet} train split "
              f"({len(fit_split)} rows); testing on {args.parquet} "
              f"test split ({len(test)} rows)", flush=True)
    mbasis = manifold_basis(fit_split, args.batch, k=max(args.project_k, 32))
    baseline = None
    if args.model == "baseline_perbus":
        baseline = fit_per_bus_baseline(fit_split, args.batch)
        model = None
    elif args.model in ("graphkit", "lumina"):
        model, _ext_fwd, _ext_to_dev = make_graphkit_or_lumina(
            args.model, args.checkpoint, device, args.parquet, args.batch,
            args.model_config)
    elif args.model == "gridsfm":
        model = make_gridsfm_model(args.checkpoint, device)
    else:
        gate = "sdpa_sigmoid" if args.model in ("g4", "g5") else "scalar_tanh"
        mode = {"g1":"meanmax_pre", "g2":"attn_pre", "g3":"attn_post", "g4":"attn_pre", "g5":"attn_post"}[args.model]
        from GNSMsg_SelfAttention_armijo import GNSMsg_EdgeSelfAttn
        model = pignn_model(mode, gate).to(device)
        state = torch.load(args.checkpoint, map_location=device)
        result = model.load_state_dict(state, strict=False)
        print(f"[PIGNN] checkpoint missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}", flush=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("PIGNN checkpoint/model mismatch; refusing a non-identical rescore")
    if model is not None:
        model.eval()

    def _fwd(batch):
        if args.model in ("graphkit", "lumina"):
            # graphkit's adapter wants a batch already on the device; LUMINA's
            # forward builds its graph from a CPU batch and moves it itself.
            b = {k: (v.to(device) if hasattr(v, "to") else v)
                 for k, v in batch.items()} if _ext_to_dev else batch
            return _ext_fwd(b)
        return (forward_gridsfm(model, batch, device) if args.model == "gridsfm"
                else forward_pignn(model, batch, device))

    offset = None
    if args.debias:
        if baseline is not None:
            raise SystemExit("--debias is meaningless for the constant baseline")
        offset = fit_per_bus_offset(model, _fwd, fit_split, args.batch, device)
    p_all=[]; q_all=[]; scen_all=[]; bt_all=[]; vn_all=[]; ydiag_all=[]; deg_all=[]; tr_all=[]
    pb_all=[]   # GENCO-style per-bus power-balance norm, unmasked bus vector
    ang_chunks=[]; v_chunks=[]; rough_chunks=[]; eps_chunks=[]; split_chunks=[]
    vm_sse=va_sse=0.0; count=0; scenario_offset=0
    with torch.no_grad():
        for bi,batch in enumerate(loader):
            sizes=batch["sizes"].numpy().astype(int); offs=np.cumsum(np.r_[0,sizes[:-1]])
            if baseline is not None:
                # Constant per-bus predictor: tile the fitted profile over the
                # scenarios packed into this batch.
                vm_bar, th_bar = baseline
                tiled = np.stack([np.tile(vm_bar, len(sizes)), np.tile(th_bar, len(sizes))], axis=-1)
                Vpred = torch.from_numpy(tiled).to(device).unsqueeze(0)
            else:
                # Route through _fwd so every model family -- including the
                # graphkit/LUMINA paths, which need their own batch handling
                # -- uses one dispatch point.
                Vpred = _fwd(batch).detach()
            Vpred = Vpred.to(torch.float64)
            if offset is not None:
                dvm, dth = offset
                nrep = len(sizes)
                Vpred = Vpred.clone()
                Vpred[0, :, 0] -= dvm.repeat(nrep)
                Vpred[0, :, 1] = torch.atan2(
                    torch.sin(Vpred[0, :, 1] - dth.repeat(nrep)),
                    torch.cos(Vpred[0, :, 1] - dth.repeat(nrep)))
            # Calibrate first, then project.  The offset is fitted on raw
            # predictions, so subtracting it from an already-projected state
            # would remove b where only U U^T b is present and inject
            # (I - U U^T) b back outside the span -- the component the
            # projection exists to remove.
            if args.project_k:
                k = args.project_k
                Vpred = project_to_manifold(
                    Vpred, (mbasis[0][:, :k], mbasis[1][:, :k], mbasis[2], mbasis[3]))
            Vref = batch["V_newton"].to(device=device, dtype=torch.float64)
            vm_sse += float((Vpred[...,0]-Vref[...,0]).square().sum()); va_sse += float(angle_diff(Vpred[...,1],Vref[...,1]).square().sum()); count += Vref[...,0].numel()
            Y=batch["Ybus"]
            if Y is None: raise RuntimeError("diagnostic requires Ybus in collated batch")
            Y=Y.to(device=device,dtype=torch.complex128)
            Vc=Vpred[0,:,0].to(device)*torch.exp(1j*Vpred[0,:,1].to(device))
            Sc=Vc*ybus_matvec(Y,Vc).conj()
            S=batch["S_start"].to(device=device,dtype=torch.complex128)[0]
            dp=(S.real-Sc.real).abs().cpu().numpy(); dq=(S.imag-Sc.imag).abs().cpu().numpy()
            bt=batch["bus_type"][0].numpy().astype(int); vn=batch["vn_kv"][0].numpy().astype(float)
            c=Y.coalesce(); ii=c.indices()[0].cpu().numpy(); jj=c.indices()[1].cpu().numpy(); vv=c.values().abs().cpu().numpy(); diagv=np.zeros(len(bt)); m=ii==jj; np.add.at(diagv,ii[m],vv[m])
            f=batch["Branch_f_bus"][0].numpy().astype(int); t=batch["Branch_t_bus"][0].numpy().astype(int); st=batch["Branch_status"][0].numpy()!=0; deg=np.zeros(len(bt),dtype=int); np.add.at(deg,f[st],1); np.add.at(deg,t[st],1)
            tr=np.zeros(len(bt),dtype=bool); is_tr=batch["Is_trafo"][0].numpy()!=0; np.logical_or.at(tr,f[is_tr],True); np.logical_or.at(tr,t[is_tr],True)
            scen=np.zeros(len(bt),dtype=int)
            for si,(o,nodes) in enumerate(zip(offs,sizes)): scen[o:o+nodes]=scenario_offset+si
            scenario_offset += len(sizes)
            p_mask=bt!=1; q_mask=(bt!=1)&(bt!=2)
            v_chunks.append((Vpred[0].float().cpu().numpy(), Vref[0].float().cpu().numpy()))
            yij=np.zeros(len(f))
            cc=Y.coalesce(); ii2=cc.indices()[0].cpu().numpy(); jj2=cc.indices()[1].cpu().numpy(); vv2=cc.values().abs().cpu().numpy()
            off={}
            for x,y,z in zip(ii2,jj2,vv2):
                if x!=y: off[(int(x),int(y))]=z
            for e in range(len(f)):
                yij[e]=off.get((int(f[e]),int(t[e])), off.get((int(t[e]),int(f[e])), 0.0))
            eps_chunks.append((Vpred[0,:,0]-Vref[0,:,0]).reshape(len(sizes),-1).cpu())
            split_chunks.append(residual_split(
                Y, Vpred, S, batch["bus_type"][0].to(device), ybus_matvec, len(bt)//len(sizes)))
            rough_chunks.append(voltage_roughness(
                Vpred[0,:,0].cpu().numpy(), Vref[0,:,0].cpu().numpy(),
                f, t, batch["Branch_status"][0].numpy(), yij))
            ang_chunks.append(branch_angle_diagnostics(
                Vpred[0,:,1].cpu().numpy(), Vref[0,:,1].cpu().numpy(),
                f, t, batch["Branch_status"][0].numpy(), scen))
            p_all.append((dp[p_mask],scen[p_mask],bt[p_mask],vn[p_mask],diagv[p_mask],deg[p_mask],tr[p_mask]))
            q_all.append((dq[q_mask],scen[q_mask],bt[q_mask],vn[q_mask],diagv[q_mask],deg[q_mask],tr[q_mask]))
            # GENCO reports the mean l2 norm of the per-bus residual vector
            # (dP, dQ) "across all buses".  Components the model recovers
            # analytically are structurally zero there -- Q at PV and REF, P at
            # REF for PF -- which is exactly this scorer's p_mask/q_mask.  Zero
            # them BEFORE the norm, and keep every bus in the vector so the
            # denominator is all buses, slack included (contributing 0).
            pb_all.append(np.sqrt((dp * p_mask) ** 2 + (dq * q_mask) ** 2))
            if bi and bi % 25 == 0: print(f"[progress] batch={bi}/{len(loader)}",flush=True)
    def regression_block(v_chunks, n_bus):
        import torch as _t
        from prediction_diagnostics import (voltage_regression,
                                            voltage_regression_per_bus)
        Vp = _t.from_numpy(np.concatenate([c[0] for c in v_chunks], 0)).unsqueeze(0)
        Vr = _t.from_numpy(np.concatenate([c[1] for c in v_chunks], 0)).unsqueeze(0)
        pooled = voltage_regression(Vp, Vr)
        per_bus = voltage_regression_per_bus(Vp, Vr, n_bus)
        return {
            "pooled": {
                "slope_vmag": pooled["vmag"]["slope"], "R2_vmag": pooled["vmag"]["R2"],
                "std_pred_vmag": pooled["vmag"]["std_pred"],
                "std_ref_vmag": pooled["vmag"]["std_ref"],
                "slope_sin": pooled["theta_sin"]["slope"], "R2_sin": pooled["theta_sin"]["R2"],
                "slope_cos": pooled["theta_cos"]["slope"], "R2_cos": pooled["theta_cos"]["R2"],
            },
            "per_bus": {
                "n_bus": int(n_bus),
                "slope_vmag": per_bus["vmag"]["slope"], "R2_vmag": per_bus["vmag"]["R2"],
                "std_pred_vmag": per_bus["vmag"]["std_pred"],
                "std_ref_vmag": per_bus["vmag"]["std_ref"],
                "slope_theta": per_bus["theta"]["slope"], "R2_theta": per_bus["theta"]["R2"],
                **per_bus["sse"],
            },
        }

    def agg_rough(chunks):
        n=sum(c["n_edge"] for c in chunks)
        o={}
        for k in chunks[0]:
            if k=="n_edge": continue
            if k.startswith("rms"):
                o[k]=math.sqrt(sum(c[k]**2*c["n_edge"] for c in chunks)/n)
            elif k=="max_edge":
                o[k]=max(c[k] for c in chunks)
            else:
                o[k]=sum(c[k]*c["n_edge"] for c in chunks)/n
        o["ratio"]=o["rms_edge"]/max(o["rms_bus"],1e-30)
        o["n_edge"]=n
        return o

    def agg_angle(chunks):
        # Per-batch stats are recombined by their entry counts; max takes the
        # max and the median/p95 are approximated by an n-weighted mean of the
        # per-batch values, which is adequate for batches of equal size.
        out = {}
        for key in chunks[0]:
            tot = sum(c[key]["n"] for c in chunks)
            out[key] = {
                "mae": sum(c[key]["mae"] * c[key]["n"] for c in chunks) / tot,
                "rmse": math.sqrt(sum(c[key]["rmse"] ** 2 * c[key]["n"] for c in chunks) / tot),
                "median": sum(c[key]["median"] * c[key]["n"] for c in chunks) / tot,
                "p95": sum(c[key]["p95"] * c[key]["n"] for c in chunks) / tot,
                "max": max(c[key]["max"] for c in chunks),
                "n": tot,
            }
        return out

    def pb_summary(r, bt_one):
        """GENCO power-balance loss: mean over ALL bus-scenario pairs.

        Reported with the bus-type census so the masking can be checked, and
        with the two marginal means so the result can be reconciled against the
        existing mean|dP| / mean|dQ| rows.  Note mean r is NOT
        sqrt(mean|dP|^2 + mean|dQ|^2) -- the norm is taken per bus, before any
        averaging.
        """
        srt = np.sort(r)
        n_ = srt.size
        qf = lambda f: float(srt[min(n_ - 1, int(f * (n_ - 1) + 0.5))])
        return {"n_bus_scenarios": int(n_),
                "mean": float(r.mean()), "median": qf(0.5),
                "p95": qf(0.95), "p99": qf(0.99), "max": float(srt[-1]),
                "rms": float((r ** 2).mean() ** 0.5),
                "n_slack": int((bt_one == 1).sum()),
                "n_pv": int((bt_one == 2).sum()),
                "n_pq": int(((bt_one != 1) & (bt_one != 2)).sum()),
                "n_bus": int(bt_one.size)}

    def merge(chunks): return [np.concatenate([x[i] for x in chunks]) for i in range(7)]
    p=merge(p_all); q=merge(q_all)
    bt_last=batch["bus_type"][0].numpy().astype(int)[:int(len(bt)//len(sizes))]
    out={"model":args.model,"debias":bool(args.debias),"project_k":int(args.project_k),"checkpoint":args.checkpoint,"n_total":n,"n_test_scenarios":scenario_offset,
         "voltage":{"vm_rmse":float(np.sqrt(vm_sse/count)),"theta_rmse_deg":float(np.sqrt(va_sse/count)*180/math.pi)},
         "reference_spreads":reference_spreads(loader),
         "angle":agg_angle(ang_chunks),
         "roughness":agg_rough(rough_chunks),
         "residual_split":{k: sum(c[k]**2 * 1 for c in split_chunks)**0.5/len(split_chunks)**0.5
                           for k in split_chunks[0]},
         "manifold":manifold_split(torch.cat(eps_chunks,0).cpu(), mbasis[0][:, :32].cpu()) | {"k_energy": mbasis[4], "k_energy_theta": mbasis[5], "project_k": int(args.project_k)},
         "regression":regression_block(v_chunks, int(len(bt)//len(sizes))),
         "P":{"pooled":summary(p[0]),"per_scenario":scenario_summary(p[0],p[1])},
         "Q":{"pooled":summary(q[0]),"per_scenario":scenario_summary(q[0],q[1])},
         "power_balance":pb_summary(np.concatenate(pb_all), bt_last)}
    for ch,data in (("P",p),("Q",q)):
        for key,idx in (("bus_type",2),("nominal_voltage_kv",3),("degree",5),("transformer_involved",6)):
            add_group(out[ch],data[0],data[idx],key)
        add_yii_bins(out[ch], data[0], data[4])
    with open(args.output,"w") as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2),flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",choices=("g1","g2","g3","g4","g5","gridsfm","graphkit","lumina","baseline_perbus"),required=True)
    ap.add_argument("--model_config",default=None,help="LUMINA model config, passed through to its driver")
    ap.add_argument("--checkpoint",default="none"); ap.add_argument("--parquet",required=True)
    ap.add_argument("--batch",type=int,default=23); ap.add_argument("--output",required=True)
    ap.add_argument("--project_k",type=int,default=0,
                    help="project the prediction onto the rank-k training manifold (0 = off)")
    ap.add_argument("--fit_parquet",default=None,
                    help="fit the manifold basis and the per-bus offset on THIS "
                         "corpus's training split instead of --parquet's. Use it "
                         "to ask whether a basis fitted on the current corpus "
                         "still works on an OOD one; without it the basis is "
                         "refitted on the OOD data, which answers a different "
                         "question (can refitting rescue projection).")
    ap.add_argument("--varying_topology",action="store_true",
                    help="corpus contains line outages, so Ybus differs row to "
                         "row; disables the share_grid/share_ybus caching, "
                         "which would otherwise reuse one admittance matrix for "
                         "every scenario")
    ap.add_argument("--fit_share_grid",action="store_true",
                    help="keep the share_grid/share_ybus caching for the "
                         "--fit_parquet corpus even under --varying_topology. "
                         "Valid only when THAT corpus has a fixed Ybus; the "
                         "corpus under test is unaffected either way.")
    ap.add_argument("--debias",action="store_true",
                    help="subtract a per-bus offset fitted on the training split")
    diag(ap.parse_args())
