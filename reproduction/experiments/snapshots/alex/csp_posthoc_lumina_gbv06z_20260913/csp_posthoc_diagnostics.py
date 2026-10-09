#!/usr/bin/env python3
"""GBnetwork post-hoc diagnostics using the paper's frozen CSP implementation.

This script is intentionally a scorer: it never modifies weights, labels, or
the rank.  It imports the same split, calibration, basis, projector, known-state
restoration, and complex128 PB function used for the restore_known Table-1 rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from collate_blockdiag_optimized_complex_columns import collate_blockdiag
from controlled_error_geometry import calibrated_forward, make_model, pb_per_scenario, split_dataset
from diagnose_residual_distributions import angle_diff, fit_per_bus_offset, manifold_basis
from manifold_projection import project_state
from output_intervention_metrics import _restore_known_setpoints


def _energy(x: torch.Tensor) -> float:
    return float(x.square().sum().item())


def _block_residual(x: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor, *, circular: bool):
    d = angle_diff(x, mean) if circular else x - mean
    return d - (d @ basis) @ basis.T


def _rmse(sum_sq: float, n: int, degrees: bool = False) -> float:
    value = math.sqrt(sum_sq / n)
    return math.degrees(value) if degrees else value


def _new_identity_record():
    return dict(ec=0.0, ecsp_pure=0.0, ecsp_operational=0.0,
                ref_off=0.0, err_off=0.0, max_abs_pure=0.0,
                max_abs_operational=0.0, n=0)


def _update_identity(rec, ec, ecsp_pure, ecsp_operational, ref_off, err_off):
    # All quantities are per coordinate; equality is tested after summation and
    # also scenario-by-scenario, retaining the largest absolute discrepancy.
    lhs_pure = ecsp_pure.square().sum(1) - ec.square().sum(1)
    lhs_oper = ecsp_operational.square().sum(1) - ec.square().sum(1)
    rhs = ref_off.square().sum(1) - err_off.square().sum(1)
    rec["max_abs_pure"] = max(rec["max_abs_pure"], float((lhs_pure-rhs).abs().max()))
    rec["max_abs_operational"] = max(rec["max_abs_operational"], float((lhs_oper-rhs).abs().max()))
    rec["ec"] += _energy(ec); rec["ecsp_pure"] += _energy(ecsp_pure)
    rec["ecsp_operational"] += _energy(ecsp_operational)
    rec["ref_off"] += _energy(ref_off); rec["err_off"] += _energy(err_off)
    rec["n"] += ec.numel()


def _finish_identity(rec):
    rhs = rec["ref_off"] - rec["err_off"]
    pure = rec["ecsp_pure"] - rec["ec"]
    operational = rec["ecsp_operational"] - rec["ec"]
    denom = max(abs(rhs), abs(pure), 1.0)
    return dict(
        calibrated_error_energy=rec["ec"],
        csp_pure_error_energy=rec["ecsp_pure"],
        csp_operational_error_energy=rec["ecsp_operational"],
        reference_off_subspace_energy=rec["ref_off"],
        calibrated_error_off_subspace_energy=rec["err_off"],
        rhs_reference_off_minus_error_off=rhs,
        lhs_pure_projection=pure,
        lhs_operational_restore_known=operational,
        pure_discrepancy=pure-rhs,
        operational_discrepancy=operational-rhs,
        pure_relative_discrepancy=abs(pure-rhs)/denom,
        operational_relative_discrepancy=abs(operational-rhs)/denom,
        max_abs_scenario_discrepancy_pure=rec["max_abs_pure"],
        max_abs_scenario_discrepancy_operational=rec["max_abs_operational"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--g3")
    ap.add_argument("--gridsfm")
    ap.add_argument("--graphkit")
    ap.add_argument("--lumina")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lumina-args", default="")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()
    if args.rank != 16:
        raise ValueError("This paper diagnostic is fixed to the requested k=16 convention.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, _validation, test = split_dataset(args.parquet)
    # Exactly the paper basis routine: train NR states, circular angle centring,
    # rank clamp in fit_basis, and no held-out states.
    Uv, Ut, vbar, tbar, ev_train, et_train = manifold_basis(train, args.batch, k=args.rank)
    Uv, Ut = Uv.to(device=device, dtype=torch.float64), Ut.to(device=device, dtype=torch.float64)
    vbar, tbar = vbar.to(device=device, dtype=torch.float64), tbar.to(device=device, dtype=torch.float64)
    basis = (Uv, Ut, vbar, tbar)

    models = {name: (kind, checkpoint) for name, kind, checkpoint in (
        ("PIGNN-GC", "g3", args.g3),
        ("GridSFM", "gridsfm", args.gridsfm),
        ("GridFM-GraphKit", "graphkit", args.graphkit),
        ("LUMINA", "lumina", args.lumina),
    ) if checkpoint}
    if not models:
        raise ValueError("Provide at least one model checkpoint.")
    loaded = {}
    for name, (kind, checkpoint) in models.items():
        model, forward = make_model(kind, checkpoint, args.parquet, args.batch, device,
                                     lumina_args=args.lumina_args)
        loaded[name] = (model, forward, fit_per_bus_offset(model, forward, train, args.batch, device))

    common = dict(ref_total_v=0.0, ref_total_t=0.0, ref_off_v=0.0, ref_off_t=0.0,
                  oracle_v_sse=0.0, oracle_t_sse=0.0, n=0, oracle_pb=[], oracle_max_pb=0.0,
                  oracle_restore_pb=[], oracle_restore_max_pb=0.0,
                  restoration_delta_v_sse=0.0, restoration_delta_t_sse=0.0)
    per_model = {name: dict(off_v_num=0.0, off_v_den=0.0, off_t_num=0.0, off_t_den=0.0,
                            identity_v=_new_identity_record(), identity_t=_new_identity_record())
                 for name in models}
    loader = DataLoader(test, batch_size=args.batch, shuffle=False, num_workers=0,
                        collate_fn=collate_blockdiag)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            sizes = batch["sizes"].numpy().astype(int)
            if len(set(map(int, sizes))) != 1:
                raise ValueError("GBnetwork diagnostic requires a fixed topology per batch.")
            nbus, ns = int(sizes[0]), len(sizes)
            ref = batch["V_newton"].to(device=device, dtype=torch.float64)[0].reshape(ns, nbus, 2)
            rv, rt = ref[..., 0], ref[..., 1]
            ref_off_v = _block_residual(rv, vbar, Uv, circular=False)
            ref_off_t = _block_residual(rt, tbar, Ut, circular=True)
            common["ref_total_v"] += _energy(rv-vbar); common["ref_total_t"] += _energy(angle_diff(rt, tbar))
            common["ref_off_v"] += _energy(ref_off_v); common["ref_off_t"] += _energy(ref_off_t)
            oracle_v, oracle_t = project_state(rv, rt, basis)
            common["oracle_v_sse"] += _energy(oracle_v-rv)
            common["oracle_t_sse"] += _energy(angle_diff(oracle_t, rt))
            common["n"] += rv.numel()
            oracle_pb, oracle_max = pb_per_scenario(oracle_v, oracle_t, batch, device, return_global_max=True)
            common["oracle_pb"].append(oracle_pb); common["oracle_max_pb"] = max(common["oracle_max_pb"], oracle_max)
            # This is reported separately: it is the exact Table-1 operational
            # policy applied to an oracle projection, while the requested floor
            # above remains the stated affine reference projection.
            oracle_rv, oracle_rt = _restore_known_setpoints(oracle_v, oracle_t, batch, nbus)
            common["restoration_delta_v_sse"] += _energy(oracle_rv-oracle_v)
            common["restoration_delta_t_sse"] += _energy(angle_diff(oracle_rt, oracle_t))
            oracle_restore_pb, oracle_restore_max = pb_per_scenario(oracle_rv, oracle_rt, batch, device, return_global_max=True)
            common["oracle_restore_pb"].append(oracle_restore_pb); common["oracle_restore_max_pb"] = max(common["oracle_restore_max_pb"], oracle_restore_max)

            for name, (_model, forward, offset) in loaded.items():
                cal = calibrated_forward(forward, offset, batch)[0].reshape(ns, nbus, 2)
                cv, ct = cal[..., 0], cal[..., 1]
                evc, etc = cv-rv, angle_diff(ct, rt)
                off_v = evc - (evc @ Uv) @ Uv.T
                off_t = etc - (etc @ Ut) @ Ut.T
                record = per_model[name]
                record["off_v_num"] += _energy(off_v); record["off_v_den"] += _energy(evc)
                record["off_t_num"] += _energy(off_t); record["off_t_den"] += _energy(etc)
                pure_v, pure_t = project_state(cv, ct, basis)
                operational_v, operational_t = _restore_known_setpoints(pure_v, pure_t, batch, nbus)
                _update_identity(record["identity_v"], evc, pure_v-rv, operational_v-rv, ref_off_v, off_v)
                _update_identity(record["identity_t"], etc, angle_diff(pure_t,rt), angle_diff(operational_t,rt), ref_off_t, off_t)
            if bi and bi % 50 == 0:
                print(f"[diagnostic] {bi}/{len(loader)} test batches", flush=True)

    common_result = {
        "training_explained_variance": {"magnitude": ev_train, "angle": et_train},
        "test_reference_retained_variance": {
            "magnitude": 1-common["ref_off_v"]/common["ref_total_v"],
            "angle": 1-common["ref_off_t"]/common["ref_total_t"]},
        "oracle_pure_reference_projection": {
            "vmag_rmse": _rmse(common["oracle_v_sse"], common["n"]),
            "angle_rmse_deg": _rmse(common["oracle_t_sse"], common["n"], degrees=True),
            "mean_pb": float(torch.tensor([x for a in common["oracle_pb"] for x in a]).mean()),
            "max_pb": common["oracle_max_pb"]},
        "oracle_restore_known_reference_projection": {
            "mean_pb": float(torch.tensor([x for a in common["oracle_restore_pb"] for x in a]).mean()),
            "max_pb": common["oracle_restore_max_pb"],
            "restoration_change_vmag_rmse": _rmse(common["restoration_delta_v_sse"], common["n"]),
            "restoration_change_angle_rmse_deg": _rmse(common["restoration_delta_t_sse"], common["n"], degrees=True)},
    }
    models_result = {}
    for name, values in per_model.items():
        models_result[name] = {
            "off_error_fraction_calibrated": {
                "magnitude": values["off_v_num"]/values["off_v_den"],
                "angle": values["off_t_num"]/values["off_t_den"]},
            "removed_error_energy_calibrated": {"magnitude": values["off_v_num"], "angle": values["off_t_num"]},
            "identity_check": {"magnitude": _finish_identity(values["identity_v"]),
                               "angle": _finish_identity(values["identity_t"])},
        }
    out = dict(protocol={
        "split": "controlled_error_geometry.split_dataset: random_split seed=42; sorted membership order",
        "basis": "diagnose_residual_distributions.manifold_basis -> manifold_projection.fit_basis; train NR only; circular angle mean/wrap; k=16",
        "calibration": "diagnose_residual_distributions.fit_per_bus_offset + controlled_error_geometry.calibrated_forward; train only; circular angle offset",
        "csp": "manifold_projection.project_state then output_intervention_metrics._restore_known_setpoints, matching restore_known paper rows",
        "pb": "controlled_error_geometry.pb_per_scenario; final complex128 voltage; GENCO structural-zero PB",
        "oracle_floor": "pure project_state(reference), as requested; operational restore_known oracle PB additionally reported",
        "no_clipping": True,
        "known_state_restoration_after_csp": "PV/slack magnitude and slack angle restored from V_start",
        "rank_requested": args.rank, "rank_actual_magnitude": int(Uv.shape[1]), "rank_actual_angle": int(Ut.shape[1]),
        "test_scenarios": len(test)},
        reference=common_result, models=models_result)
    Path(args.out_json).write_text(json.dumps(out, indent=2, allow_nan=False))
    with Path(args.out_csv).open("w", newline="") as f:
        w=csv.writer(f); w.writerow(["section","model","block","metric","value"])
        for metric, values in common_result["training_explained_variance"].items(): w.writerow(["reference","","" if metric is None else metric,"training_explained_variance",values])
        for metric, values in common_result["test_reference_retained_variance"].items(): w.writerow(["reference","",metric,"test_reference_retained_variance",values])
        for metric, value in common_result["oracle_pure_reference_projection"].items(): w.writerow(["oracle_pure","","joint",metric,value])
        for name, values in models_result.items():
            for block, value in values["off_error_fraction_calibrated"].items(): w.writerow(["model",name,block,"off_error_fraction_calibrated",value])
            for block, value in values["removed_error_energy_calibrated"].items(): w.writerow(["model",name,block,"removed_error_energy_calibrated",value])
            for block, entry in values["identity_check"].items():
                for metric, value in entry.items(): w.writerow(["identity",name,block,metric,value])
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
