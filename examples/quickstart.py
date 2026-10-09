#!/usr/bin/env python3
"""CSP end to end, on synthetic data, in under a second.

    python examples/quickstart.py

What this is and is not
-----------------------
This demonstrates the *transform*. It is not evidence for the paper's claim.

The paper's empirical finding is that a real surrogate's error concentrates in
the span of the training solutions. Here that structure is put in by hand. The
reference states are built to lie in a low-dimensional manifold, and the
synthetic "surrogate" is given three error components:

  1. a fixed per-bus bias                  — what C removes,
  2. a component pointing off the manifold — what P removes,
  3. a scenario-dependent error *inside* the manifold — which CSP cannot
     remove, and should not: it is a real tracking error, not a direction the
     grid never visits.

The third component is why the corrected error is small but not zero, here and
in the paper. CSP is a way to stop paying for error the reference never
exhibits; it is not a way to make a bad surrogate good.

For the real thing, see `docs/REPRODUCING_THE_PAPER.md`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csp

N_BUS, N_TRAIN, N_TEST, TRUE_RANK, FIT_RANK = 30, 400, 200, 8, 8
SEED = 0


def build_grid(n_bus: int, g: torch.Generator):
    """A small meshed network: ring plus a few chords, series impedances only."""
    y = torch.zeros(n_bus, n_bus, dtype=torch.complex128)
    edges = [(i, (i + 1) % n_bus) for i in range(n_bus)]
    edges += [(int(a), int(b)) for a, b in
              torch.randint(0, n_bus, (n_bus // 3, 2), generator=g)
              if int(a) != int(b)]
    for a, b in edges:
        r = 0.01 + 0.03 * float(torch.rand((), generator=g))
        x = 0.05 + 0.15 * float(torch.rand((), generator=g))
        adm = 1.0 / complex(r, x)
        y[a, a] += adm
        y[b, b] += adm
        y[a, b] -= adm
        y[b, a] -= adm
    bus_type = torch.full((n_bus,), 3, dtype=torch.long)
    bus_type[0] = 1                      # slack
    bus_type[1:4] = 2                    # PV
    return y, bus_type


def sample_solutions(n_scen: int, manifold, g: torch.Generator):
    """States on the manifold, and the injections that make them exact."""
    mean_mag, mean_ang, b_mag, b_ang = manifold
    z = torch.randn(n_scen, b_mag.shape[1], generator=g, dtype=torch.float64)
    mag = mean_mag + 0.02 * (z @ b_mag.T)
    ang = mean_ang + 0.08 * (z @ b_ang.T)
    return torch.stack([mag, csp.wrap_angle(ang)], dim=-1)


def specified_injection(states: torch.Tensor, ybus: torch.Tensor) -> torch.Tensor:
    """S = V ⊙ conj(Y V). By construction the reference has zero mismatch."""
    v = (states[..., 0] * torch.exp(1j * states[..., 1])).to(torch.complex128)
    return v * (v @ ybus.T).conj()


def fake_surrogate(reference, bias, off_dirs, in_dirs, g: torch.Generator):
    """Reference plus the three error components described in the docstring."""
    bias_mag, bias_ang = bias
    n = reference.shape[0]
    off = 0.004 * (torch.randn(n, off_dirs.shape[1], generator=g,
                               dtype=torch.float64) @ off_dirs.T)
    inside = 0.0015 * (torch.randn(n, in_dirs.shape[1], generator=g,
                                   dtype=torch.float64) @ in_dirs.T)
    mag = reference[..., 0] + bias_mag + off + inside
    ang = csp.wrap_angle(reference[..., 1] + bias_ang + 0.35 * (off + inside))
    return torch.stack([mag, ang], dim=-1)


def main() -> None:
    g = torch.Generator().manual_seed(SEED)
    ybus, bus_type = build_grid(N_BUS, g)

    # A true low-dimensional solution manifold, and directions off it.
    q_mag = torch.linalg.qr(torch.randn(N_BUS, N_BUS, generator=g,
                                        dtype=torch.float64))[0]
    q_ang = torch.linalg.qr(torch.randn(N_BUS, N_BUS, generator=g,
                                        dtype=torch.float64))[0]
    manifold = (torch.ones(N_BUS, dtype=torch.float64),
                torch.zeros(N_BUS, dtype=torch.float64),
                q_mag[:, :TRUE_RANK], q_ang[:, :TRUE_RANK])
    off_dirs = q_mag[:, TRUE_RANK:]
    in_dirs = q_mag[:, :TRUE_RANK]

    train_ref = sample_solutions(N_TRAIN, manifold, g)
    test_ref = sample_solutions(N_TEST, manifold, g)

    bias = (0.006 * torch.randn(N_BUS, generator=g, dtype=torch.float64),
            0.010 * torch.randn(N_BUS, generator=g, dtype=torch.float64))
    train_pred = fake_surrogate(train_ref, bias, off_dirs, in_dirs, g)
    test_pred = fake_surrogate(test_ref, bias, off_dirs, in_dirs, g)

    # ---- fit on the training split only ---------------------------------
    transform = csp.CSP.fit(train_pred, train_ref, rank=FIT_RANK)
    print(transform)
    print(f"  the rank-{FIT_RANK} basis captures "
          f"{100 * transform.basis.energy_mag:.2f}% of reference |V| variance "
          f"and {100 * transform.basis.energy_ang:.2f}% of angle variance\n")

    # ---- apply to the held-out split ------------------------------------
    s_spec = specified_injection(test_ref, ybus)
    rows = []
    for name, state in transform.stages(test_pred).items():
        m = csp.evaluate(state, test_ref, ybus=ybus, s_specified=s_spec,
                         bus_type=bus_type)
        rows.append((name, m))

    head = f"{'':>8}  {'|V| RMSE':>10}  {'ang RMSE':>10}  {'mean PB':>10}  {'max PB':>10}  {'R2_w':>7}  {'a_w':>6}"
    print(head)
    print("-" * len(head))
    for name, m in rows:
        print(f"{name:>8}  {m['vmag_rmse']:10.3e}  {m['angle_rmse_deg']:10.4f}  "
              f"{m['mean_pb']:10.3e}  {m['max_pb']:10.3e}  "
              f"{m['within_r2']:7.4f}  {m['within_slope']:6.3f}")

    raw, csp_row = rows[0][1], rows[-1][1]
    print(f"\n  mean PB  {raw['mean_pb']:.3e} -> {csp_row['mean_pb']:.3e} "
          f"({raw['mean_pb'] / csp_row['mean_pb']:.1f}x lower)")
    print(f"  |V| RMSE {raw['vmag_rmse']:.3e} -> {csp_row['vmag_rmse']:.3e}")
    print(f"  R2_w     {raw['within_r2']:.4f} -> {csp_row['within_r2']:.4f}, "
          f"while a_w stays {raw['within_slope']:.3f} -> {csp_row['within_slope']:.3f}")
    print("\n  C removes the bias; P removes the off-manifold component; the\n"
          "  in-manifold tracking error survives both, which is why the last row\n"
          "  is small rather than zero. a_w barely moves throughout, because\n"
          "  neither step adds response amplitude -- they remove error the\n"
          "  reference never exhibits.")


if __name__ == "__main__":
    main()
