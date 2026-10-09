"""A cheap magnitude prior for AC power flow, analogous to the DC angle prior.

The corpus already carries a DC start angle that lands 0.568 deg from the Newton
solution, and anchoring LUMINA's angle head on it took the angle error from
11.577 deg to 0.136 deg.  The magnitude has no such prior: v_start is flat 1.0
at PQ buses, so feeding it in changes nothing (measured: 0.0080890 against a
0.0080865 control).

Measured candidates on case14 (per-bus sigma 0.023541, flat 1.0 gives 0.028820
on PQ buses):

    v = 1 - B''^-1 Q          RMSE 0.024130   R^2 ~ -0.05   (useless)
    one Newton step, flat th  RMSE 0.005577   R^2   0.944
    one Newton step, DC th    RMSE 0.001561   R^2   0.996

The last is better than every surrogate in the study, but it is not a fair
prior: its Jacobian depends on the scenario's own angles, so it costs a fresh
factorisation per scenario -- a large fraction of a full Newton solve.  The
middle one is the honest choice.  Its Jacobian is evaluated once at the flat
point (v = 1, theta = 0), where it depends only on Ybus and the bus types, so a
single factorisation is reused for every scenario in the grid: the same class of
cost as the DC solve that produced the angle prior.

One correction to that picture, found by checking the module against the
validated number rather than trusting it: the start point is not all-ones.  PV
and slack magnitudes carry their setpoints, and those are randomised per
scenario (measured range 0.9508 to 1.0600), so the exact Newton step is not a
constant-Jacobian method.  What is kept constant here is the Jacobian only,
evaluated at v = 1, theta = 0; the mismatch is evaluated at the scenario's own
start point.  That is the chord (modified Newton) step, one factorisation for
the grid plus one sparse mat-vec and one triangular solve per scenario.
"""
from __future__ import annotations

import torch


class FlatPointJacobianPrior:
    """One Newton step from the flat point, with the Jacobian factorised once.

    J is built at v = 1, theta = 0, so it is a property of the grid rather than
    of the scenario.  Each scenario then costs one triangular solve against the
    specified injections.
    """

    def __init__(self, Y: torch.Tensor, bus_type: torch.Tensor,
                 mismatch_angle: str = "start"):
        """mismatch_angle: 'flat' evaluates the mismatch at theta = 0, 'start'
        at the corpus's DC start angle.  The Jacobian is frozen at the flat
        point either way, so both cost one LU per grid.  Measured PQ |V| R^2 of
        the resulting prior: case14 0.95 either way, GBnetwork -4.56 at flat
        against +0.60 at the start angle.  On a 2224-bus grid the angles span
        far more than a flat linearisation tolerates, so 'start' is the default.
        """
        # Ybus arrives block-diagonal over the batch, so it must be sliced to a
        # single grid BEFORE it is densified.  On case14 with batch 32 the whole
        # thing is 448x448 and densifying it is harmless; on GBnetwork it is
        # 71168x71168, which is 81 TB in complex128.
        nb_all = int(bus_type.reshape(-1).shape[0])
        n = min(nb_all, int(Y.shape[0]))
        if Y.is_sparse:
            Yc = Y.coalesce()
            idx = Yc.indices()
            keep = (idx[0] < n) & (idx[1] < n)
            dense = torch.zeros((n, n), dtype=torch.complex128)
            dense[idx[0][keep], idx[1][keep]] = Yc.values()[keep].to(torch.complex128)
            Y = dense
        else:
            Y = Y[:n, :n].to(torch.complex128)
        bt = bus_type.reshape(-1)[:n]
        G, B = Y.real, Y.imag
        is_slack = bt == 1
        is_pv = bt == 2
        self.n = n
        self.ip = torch.where(~is_slack)[0]          # P equations, theta unknowns
        self.iq = torch.where(~(is_slack | is_pv))[0]  # Q equations, |V| unknowns

        v = torch.ones(n, dtype=torch.float64)
        th = torch.zeros(n, dtype=torch.float64)
        Vc = v.to(torch.complex128)
        Sc = Vc * torch.conj(Y @ Vc)
        # thd = 0 at the flat point, so the rotated conductance/susceptance
        # matrices collapse to G and -B.
        Gs, Bs = G, -B
        vv = torch.outer(v, v)
        H = vv * Bs; N = vv * Gs; M = -vv * Gs; L = vv * Bs
        dg = torch.diag
        H = H - dg(dg(H)) + dg(-Sc.imag - B.diagonal() * v ** 2)
        N = N - dg(dg(N)) + dg(Sc.real + G.diagonal() * v ** 2)
        M = M - dg(dg(M)) + dg(Sc.real - G.diagonal() * v ** 2)
        L = L - dg(dg(L)) + dg(Sc.imag - B.diagonal() * v ** 2)
        ip, iq = self.ip, self.iq
        J = torch.cat([torch.cat([H[ip][:, ip], N[ip][:, iq]], 1),
                       torch.cat([M[iq][:, ip], L[iq][:, iq]], 1)], 0)
        self.LU = torch.linalg.lu_factor(J)
        self.Y = Y
        self.np_ = len(ip)
        self.mismatch_angle = str(mismatch_angle)

    def __call__(self, P_spec: torch.Tensor, Q_spec: torch.Tensor,
                 v_start: torch.Tensor, theta_start: torch.Tensor = None) -> torch.Tensor:
        """Return the prior magnitude for one scenario, |V| at every bus.

        Known magnitudes (PV and slack) are passed through from v_start; only
        the PQ entries are moved.
        """
        ip, iq = self.ip, self.iq
        # Mismatch at the scenario's own start point (theta = 0, |V| = v_start,
        # which carries the PV/slack setpoints), not at all-ones.
        v0 = v_start.double()
        if self.mismatch_angle == "start" and theta_start is not None:
            Vc = (v0 * torch.exp(1j * theta_start.double())).to(torch.complex128)
        else:
            Vc = v0.to(torch.complex128)
        Sc = Vc * torch.conj(self.Y @ Vc)
        dP = P_spec.double()[ip] - Sc.real[ip]
        dQ = Q_spec.double()[iq] - Sc.imag[iq]
        rhs = torch.cat([dP, dQ]).unsqueeze(1)
        d = torch.linalg.lu_solve(*self.LU, rhs).squeeze(1)
        v = v0.clone()
        v[iq] = 1.0 + d[self.np_:]
        return v

    def batch(self, P_spec, Q_spec, v_start, theta_start=None):
        """All scenarios in one triangular solve.

        lu_solve takes many right-hand sides at once, so a batch costs one solve
        rather than one per scenario.  On GBnetwork the per-scenario loop was
        12.2 ms each, which is 2.5 min per epoch of training data alone.
        """
        ip, iq = self.ip, self.iq
        v0 = v_start.double()
        if self.mismatch_angle == "start" and theta_start is not None:
            Vc = (v0 * torch.exp(1j * theta_start.double())).to(torch.complex128)
        else:
            Vc = v0.to(torch.complex128)
        Sc = Vc * torch.conj(Vc @ self.Y.transpose(0, 1))
        dP = P_spec.double()[:, ip] - Sc.real[:, ip]
        dQ = Q_spec.double()[:, iq] - Sc.imag[:, iq]
        rhs = torch.cat([dP, dQ], dim=1).transpose(0, 1)      # [n_eq, n_scen]
        d = torch.linalg.lu_solve(*self.LU, rhs)              # [n_eq, n_scen]
        v = v0.clone()
        v[:, iq] = 1.0 + d[self.np_:, :].transpose(0, 1)
        return v
