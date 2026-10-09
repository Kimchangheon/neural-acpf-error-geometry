#!/usr/bin/env python3
"""Check --perturb_sgen and --pv_q_from_vstart, including that OFF is a no-op.

The no-op check is the important half: these flags default to False precisely
so the existing 31-file corpus stays reproducible, and that guarantee is worth
nothing unless it is tested.
"""
import numpy as np
import pandapower.networks as pn

import case_generator_all_test_cases_pandapower_consider_ppc_branch_row as cg

SIMBENCH = ("/home/vault/iwi5/iwi5295h/PIGNN-Attn-LS/ScenarioSynthesis_PPC/"
            "CGMES/SimBenchSnapshots/ExportSim_260603_1024/data/CIM_GridAssist_1.zip")

BASE = dict(
    case_kwargs={}, ybus_mode="ppcY", jitter_load=0.10, jitter_load_q=0.15,
    jitter_gen=0.05, pv_vset_range=(0.95, 1.05), rand_u_start=True,
    angle_jitter_deg=5.0, mag_jitter_pq=0.02, scale_gen_with_load=True,
    start_mode="dc_compile", line_outage_prob=0.0, return_pp_solution=True,
)


def run(case_fn, seed, ls, **kw):
    return cg.case_generation_pandapower(
        case_fn=case_fn, seed=seed, load_scale_range=ls, **{**BASE, **kw})


def simbench_src():
    return {"cgmes_files": SIMBENCH, "case_name": "SimBench",
            "converter_kwargs": {"cgmes_version": "2.4.15"}}


def test_flags_off_are_a_noop():
    """Defaults must reproduce the pre-change pipeline exactly."""
    for grid in ("case118", "case300"):
        a = run(getattr(pn, grid), 11, (0.8, 1.2))
        b = run(getattr(pn, grid), 11, (0.8, 1.2),
                perturb_sgen=False, pv_q_from_vstart=False)
        assert np.array_equal(np.asarray(a[2]), np.asarray(b[2])), grid
        assert np.array_equal(np.asarray(a[3]), np.asarray(b[3])), grid
    print("OK  flags off -> byte-identical s_multi and u_start")


def test_pv_q_matches_vstart_when_on():
    """With the flag on, PV-bus Q must equal Im[u_start conj(Y u_start)]."""
    for grid in ("case118", "case300", "case1354pegase"):
        out = run(getattr(pn, grid), 5, (1.0, 1.0), pv_q_from_vstart=True)
        bus_typ, s_multi, u_start, Y = (np.asarray(out[1]), np.asarray(out[2]),
                                        np.asarray(out[3]), out[4])
        pv = np.flatnonzero(bus_typ == 2)
        q_want = (u_start * np.conj(Y @ u_start)).imag[pv]
        rel = np.abs(s_multi.imag[pv] - q_want).max() / max(np.abs(q_want).max(), 1e-30)
        assert rel < 1e-12, (grid, rel)

        # P at PV buses must NOT have been touched.
        ref = run(getattr(pn, grid), 5, (1.0, 1.0))
        assert np.allclose(s_multi.real[pv], np.asarray(ref[2]).real[pv], rtol=0, atol=1e-6), grid
        print(f"OK  {grid}: PV-Q == Q(u_start) (rel {rel:.2e}), P untouched")


def test_pv_q_off_disagrees_with_vstart():
    """Guard against the flag being a no-op by accident: OFF must differ."""
    out = run(pn.case300, 5, (1.0, 1.0))
    bus_typ, s_multi, u_start, Y = (np.asarray(out[1]), np.asarray(out[2]),
                                    np.asarray(out[3]), out[4])
    pv = np.flatnonzero(bus_typ == 2)
    q_vstart = (u_start * np.conj(Y @ u_start)).imag[pv]
    corr = np.corrcoef(s_multi.imag[pv], q_vstart)[0, 1]
    assert corr < 0.99, corr
    print(f"OK  case300 with flag off: corr(S_start.Q, Q(u_start)) = {corr:.3f} (< 0.99)")


def test_sgen_perturbation_moves_simbench():
    """SimBench: injection must actually respond to the load scale now."""
    src = simbench_src()

    def total_p(ls, **kw):
        return np.asarray(run(src, 3, ls, **kw)[2]).real.sum() / 1e6

    lo_off, hi_off = total_p((0.60, 0.60)), total_p((1.40, 1.40))
    lo_on = total_p((0.60, 0.60), perturb_sgen=True)
    hi_on = total_p((1.40, 1.40), perturb_sgen=True)

    swing_off = abs(hi_off - lo_off) / abs(lo_off)
    swing_on = abs(hi_on - lo_on) / abs(lo_on)
    print(f"    OFF: {lo_off:9.1f} -> {hi_off:9.1f} MW  ({swing_off:.1%} swing)")
    print(f"    ON : {lo_on:9.1f} -> {hi_on:9.1f} MW  ({swing_on:.1%} swing)")
    assert swing_off < 0.05, swing_off
    assert swing_on > 0.30, swing_on
    print("OK  SimBench responds to the load scale only with perturb_sgen")


def test_sgen_flag_is_noop_without_sgen(self=None):
    """case118 has no sgen, so the flag must change nothing there."""
    a = run(pn.case118, 9, (0.7, 1.3))
    b = run(pn.case118, 9, (0.7, 1.3), perturb_sgen=True)
    assert np.array_equal(np.asarray(a[2]), np.asarray(b[2]))
    print("OK  perturb_sgen is inert on a grid with no sgen (case118)")


if __name__ == "__main__":
    test_flags_off_are_a_noop()
    test_pv_q_matches_vstart_when_on()
    test_pv_q_off_disagrees_with_vstart()
    test_sgen_flag_is_noop_without_sgen()
    test_sgen_perturbation_moves_simbench()
    print("\nall checks passed")
