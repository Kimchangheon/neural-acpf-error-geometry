#!/usr/bin/env python
"""Pooled slope and R^2 for a saved PF checkpoint, for any of the four models.

RMSE says how far a prediction is from the reference; it does not say whether
the model tracks the reference at all. A model that ignores its input and emits
the corpus-mean voltage profile still scores an RMSE equal to the spread of the
data (~2-3e-2 pu here), which looks respectable next to a model that is
genuinely trying. Slope and R^2 separate the two: slope ~ 0 with R^2 ~ 0 is
collapse, slope ~ 1 with R^2 ~ 1 is a model following the solver.

``rescore_pf.py`` already did this for the GridFM mirrors and graphkit. This
adds gridsfm and lumina by reusing their drivers' own ``make_model`` and
``*_forward``, so the graph construction is byte-identical to training rather
than a reimplementation.

    python rescore_slope_r2.py --model gridsfm --PARQUET g.parquet \
        --ckpt run_best.pt --grid case118
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

import torch

from Dataset_optimized_complex_columns import ChanghunDataset
from collate_blockdiag_optimized_complex_columns import collate_blockdiag, ybus_matvec
from train_valid_test_gridfm import split_dataset, cap_subset
from prediction_diagnostics import (voltage_mae, voltage_regression, angle_wrap,
                                    voltage_regression_per_bus,
                                    branch_angle_diagnostics,
                                    fit_per_bus_calibration,
                                    apply_per_bus_calibration,
                                    residual_stats, ac_residuals)

DRIVER = {
    "graphkit": "train_valid_test_gridfm",
    "gridsfm": "train_valid_test_gridsfm",
    "lumina": "train_valid_test_lumina",
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=tuple(DRIVER))
    p.add_argument("--PARQUET", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--BATCH", type=int, default=16)
    p.add_argument("--train_ratio", type=float, default=0.3333)
    p.add_argument("--valid_ratio", type=float, default=0.3333)
    p.add_argument("--seed_value", type=int, default=42)
    p.add_argument("--max_test_samples", type=int, default=0)
    p.add_argument("--model_config", default="")
    p.add_argument("--json_out", default="")
    return p.parse_args()


def driver_args(mod, a):
    """Build the driver's own argparse namespace, so no default is invented."""
    argv = [
        "rescore",
        "--PARQUET", a.PARQUET,
        "--task", "pf",
        "--BATCH", str(a.BATCH),
        "--train_ratio", str(a.train_ratio),
        "--valid_ratio", str(a.valid_ratio),
        "--seed_value", str(a.seed_value),
    ]
    if a.model == "graphkit":
        argv += ["--gridfm_impl", "graphkit", "--hidden_size", "48",
                 "--num_layers", "12", "--n_heads", "8", "--zero_init_head",
                 "--vn_feature_mode", "log", "--feature_transform", "signed_log"]
    else:
        argv += ["--init_mode", "scratch"]
    if a.model == "lumina" and a.model_config:
        argv += ["--model_config", a.model_config]

    saved, sys.argv = sys.argv, argv
    try:
        return mod.parse_args()
    finally:
        sys.argv = saved


def main():
    a = parse()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = importlib.import_module(DRIVER[a.model])
    dargs = driver_args(mod, a)

    ds = ChanghunDataset(a.PARQUET, per_unit=True, target_S_base=1e8,
                         share_grid=True, share_ybus=True, lazy_row_groups=True,
                         row_group_cache_size=4, complex_dtype="complex128")
    train_ds, _, test_ds = split_dataset(ds, a.train_ratio, a.valid_ratio, a.seed_value)
    if a.max_test_samples:
        test_ds = cap_subset(test_ds, a.max_test_samples)
        # The calibration offset is an average, so it converges quickly; capping
        # the fitting split the same way keeps the extra pass cheap.
        train_ds = cap_subset(train_ds, a.max_test_samples)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=a.BATCH, shuffle=False,
                                         collate_fn=collate_blockdiag, num_workers=0)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=a.BATCH, shuffle=False,
                                               collate_fn=collate_blockdiag, num_workers=0)

    if a.model == "graphkit":
        from gridfm_graphkit_adapter import build_graphkit_model, forward_graphkit_parquet
        model = build_graphkit_model(task_name="PowerFlow", hidden_size=48,
                                     num_layers=12, attention_head=8).to(dev)
        fwd = lambda bd: forward_graphkit_parquet(  # noqa: E731
            model, bd, dev, task_name="PowerFlow",
            feature_transform="signed_log", vn_feature_mode="log")
    else:
        model = mod.make_model(dargs, dev)
        raw = mod.gridsfm_forward if a.model == "gridsfm" else mod.lumina_forward
        fwd = lambda bd: raw(model, bd, dargs, dev)  # noqa: E731

    state = torch.load(a.ckpt, map_location=dev)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # A DDP checkpoint may carry the wrapper prefix; strip it so either loads.
    state = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # Pool the whole test set before summarising: a regression slope cannot be
    # averaged over batches.
    # graphkit's adapter expects a batch already on the device; the gridsfm and
    # lumina forwards build their graphs from a CPU batch and move them
    # themselves, exactly as their training loops do. Handing those two a GPU
    # batch mixes devices inside the graph builder.
    to_dev = a.model == "graphkit"

    def collect(dl, want_topo):
        """Run a loader and pool the predicted/reference states."""
        P, R, BT = [], [], []
        FB, TB, ST = [], [], []
        YB, SS, BTb, VPb = [], [], [], []
        n_bus = 0
        with torch.no_grad():
            for batch in dl:
                sz = batch["sizes"]
                n_bus = int(sz[0]) if int(sz.min()) == int(sz.max()) else 0
                bd = ({k: (v.to(dev) if torch.is_tensor(v) else v)
                       for k, v in batch.items()} if to_dev else batch)
                V = fwd(bd)
                P.append(V.squeeze(0).float().cpu())
                R.append(batch["V_newton"].squeeze(0).float().cpu())
                BT.append(batch["bus_type"].squeeze(0).cpu())
                # Branch indices are already global within the block-diagonal
                # batch, so they only need the running node offset added.
                off = sum(x.shape[0] for x in P[:-1])
                FB.append(batch["Branch_f_bus"].squeeze(0).cpu().long() + off)
                TB.append(batch["Branch_t_bus"].squeeze(0).cpu().long() + off)
                ST.append(batch["Branch_status"].squeeze(0).cpu())
                if want_topo:
                    # Residuals are recomputed per batch because the packed
                    # Y-bus is block-diagonal only within a batch; pooling the
                    # states first and multiplying once would mix scenarios.
                    YB.append(batch["Ybus"])
                    SS.append(batch["S_start"])
                    BTb.append(batch["bus_type"].squeeze(0))
                    VPb.append(V.squeeze(0).detach().cpu())
        Vp = torch.cat(P, 0).unsqueeze(0)
        Vr = torch.cat(R, 0).unsqueeze(0)
        if not want_topo:
            return Vp, Vr, None, None, n_bus, None
        return (Vp, Vr, torch.cat(BT, 0),
                (torch.cat(FB, 0), torch.cat(TB, 0), torch.cat(ST, 0)), n_bus,
                (YB, SS, BTb, VPb))

    Vp, Vr, bt, topo, n_bus, resid = collect(loader, True)
    Vp_tr, Vr_tr, _, _, _, _ = collect(train_loader, False)


    mae = voltage_mae(Vp, Vr)
    reg = voltage_regression(Vp, Vr)
    dmag = (Vp[..., 0] - Vr[..., 0]).to(torch.float64)
    dang = angle_wrap((Vp[..., 1] - Vr[..., 1]).to(torch.float64))

    # PQ-restricted is the like-for-like view: models that pin |V| at PV and
    # slack are exact there by construction, which flatters the pooled figure.
    # 1 = slack, 2 = PV, everything else PQ (parquet convention, not MATPOWER).
    pq = (bt != 1) & (bt != 2)
    extra = {"frac_pq": float(int(pq.sum())) / float(bt.numel())}
    if int(pq.sum()) > 1:
        reg_pq = voltage_regression(Vp[:, pq], Vr[:, pq])
        extra.update(slope_vmag_pq=reg_pq["vmag"]["slope"],
                     R2_vmag_pq=reg_pq["vmag"]["R2"])

    out = {
        "grid": a.grid, "model": a.model, "n_bus_samples": int(Vp.shape[1]),
        "rmse_vmag_pu": float((dmag ** 2).mean() ** 0.5),
        "rmse_theta_deg": float((dang ** 2).mean() ** 0.5) * 180.0 / 3.141592653589793,
        "mae_vmag_pu": mae["mae_vmag_pu"],
        "slope_vmag": reg["vmag"]["slope"], "R2_vmag": reg["vmag"]["R2"],
        "std_pred_vmag": reg["vmag"]["std_pred"], "std_ref_vmag": reg["vmag"]["std_ref"],
        "slope_sin": reg["theta_sin"]["slope"], "R2_sin": reg["theta_sin"]["R2"],
        "slope_cos": reg["theta_cos"]["slope"], "R2_cos": reg["theta_cos"]["R2"],
        **extra,
    }

    # Per-bus companions.  n_bus comes from the collated sizes: every campaign
    # here is single-grid, so all graphs share it; if a batch ever mixed sizes
    # the helpers return nan rather than a wrong number.
    pb = voltage_regression_per_bus(Vp, Vr, n_bus)
    out.update({
        "n_bus": int(n_bus),
        "slope_vmag_per_bus": pb["vmag"]["slope"], "R2_vmag_per_bus": pb["vmag"]["R2"],
        "std_pred_vmag_per_bus": pb["vmag"]["std_pred"],
        "std_ref_vmag_per_bus": pb["vmag"]["std_ref"],
        "slope_theta_per_bus": pb["theta"]["slope"], "R2_theta_per_bus": pb["theta"]["R2"],
        **pb["sse"],
    })

    ang = branch_angle_diagnostics(Vp[..., 1], Vr[..., 1], *topo, n_bus)
    out.update({f"dtheta_{k}_{m}": ang[k][m]
                for k in ("per_bus", "per_bus_degauged", "gauge", "per_edge")
                for m in ("mae", "rmse", "median", "p95", "max")})


    # AC residuals, raw and after calibration, on the same masks and in
    # complex128 -- so the P/Q columns of every table can carry both.
    def score_residuals(offset):
        YB, SS, BTb, VPb = resid
        dps, dqs = [], []
        for Yb, Sb, btb, Vb in zip(YB, SS, BTb, VPb):
            Vb = Vb.unsqueeze(0).to(torch.float64)
            if offset is not None:
                Vb = apply_per_bus_calibration(Vb, offset, n_bus)
            Yd = Yb.to(torch.complex128)
            Sd = Sb.squeeze(0).to(torch.complex128)
            dp, dq = ac_residuals(Yd, Vb, Sd, btb, ybus_matvec)
            dps.append(dp.cpu()); dqs.append(dq.cpu())
        return residual_stats(torch.cat(dps), torch.cat(dqs))

    if resid is not None:
        out.update(score_residuals(None))

    # Post-hoc per-bus mean alignment: fitted on the training split, applied
    # unchanged to the unseen test scenarios.  It removes the bias term of the
    # error decomposition, which on these grids dominates the RMSE, and cannot
    # flatter the model because no test data enters the fit.
    off = fit_per_bus_calibration(Vp_tr, Vr_tr, n_bus)
    if off is not None:
        Vc = apply_per_bus_calibration(Vp, off, n_bus)
        cm = voltage_mae(Vc, Vr)
        cb = voltage_regression_per_bus(Vc, Vr, n_bus)
        ca = branch_angle_diagnostics(Vc[..., 1], Vr[..., 1], *topo, n_bus)
        out.update({
            "cal_offset_rms_vmag": float(off[0].square().mean() ** 0.5),
            "cal_rmse_vmag_pu": cm["rmse_vmag_pu"],
            "cal_mae_vmag_pu": cm["mae_vmag_pu"],
            "cal_rmse_theta_deg": cm["rmse_theta_deg"],
            "cal_slope_vmag_per_bus": cb["vmag"]["slope"],
            "cal_R2_vmag_per_bus": cb["vmag"]["R2"],
            **{f"cal_{k}": v for k, v in cb["sse"].items()},
            **{f"cal_dtheta_{k}_{m}": ca[k][m]
               for k in ("per_bus", "per_bus_degauged", "gauge", "per_edge")
               for m in ("mae", "rmse", "median", "p95", "max")},
        })
        out.update({f"cal_{k}": v for k, v in score_residuals(off).items()})
    print("RESCORE " + json.dumps(out))
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
