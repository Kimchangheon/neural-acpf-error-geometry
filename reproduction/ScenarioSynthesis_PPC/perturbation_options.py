"""
perturbation_options.py
=======================

Selectable, option-gated perturbation primitives for the NR / PIGNN-Attn-LS
data-generation pipeline.

The whole point of this module is that *every* perturbation is opt-in and
configured by an explicit option, so you can build a spectrum of datasets
ranging from "load-only, fixed Y-bus" (cheap, share_grid-compatible) up to
"load + topology + admittance" (expensive, Y-bus varies per row).

Two families of perturbation, distinguished by whether they change Y_bus:

  FAMILY 1 — Y_bus FIXED  (share_grid=True is SAFE and fast)
      * global load scale          (load_perturbation)
      * per-bus P / Q load jitter
      * per-generator P jitter
      * PV voltage-setpoint jitter
      * start-point noise (angle / magnitude)

  FAMILY 2 — Y_bus VARIES  (share_grid MUST be False — correctness, not speed)
      * topology contingency       (sample_contingency)
      * admittance R/X jitter       (apply_admittance_jitter)

`PerturbationConfig.ybus_varies` tells the rest of the pipeline which family a
given dataset belongs to, so the training-time loader can pick share_grid
automatically and never silently reuse row-0's Y-bus for a row whose topology
or admittance is different.

Contingency sampling implements three research-backed schemes (see the journal
notes / perturbation_design.tex):

  * "bernoulli" — legacy: each line out independently w.p. p  (no k control)
  * "ratio"     — PF-Delta style: draw the contingency ORDER k from a fixed
                  categorical, e.g. P(N-0,N-1,N-2) = (0.55, 0.27, 0.18)
  * "poisson"   — probabilistically realistic: k ~ Poisson(L * q), capped,
                  so the expected number of simultaneous outages scales with
                  grid size (small grids ~N-0/N-1, large grids up to N-k)

All schemes enforce connectivity (slack can reach every bus) and never drop
the reference/slack generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# Connectivity helper
# ----------------------------------------------------------------------------
def _is_connected(net) -> bool:
    """True if no bus is left unsupplied (every bus reachable from a source)."""
    try:
        from pandapower.topology import unsupplied_buses
        return len(unsupplied_buses(net)) == 0
    except Exception:
        # If the topology routine itself fails, be conservative.
        return False


def _active_line_index(net) -> np.ndarray:
    if not hasattr(net, "line") or len(net.line) == 0:
        return np.array([], dtype=int)
    return np.asarray(net.line.index[net.line["in_service"].to_numpy(bool)], dtype=int)


def _droppable_gen_index(net) -> np.ndarray:
    """
    Generators that may be taken offline.  The slack / reference machine is
    never droppable (its outage would leave the system without an angle
    reference).  pandapower's slack is typically an ext_grid; a `gen` row
    flagged slack=True is also excluded.
    """
    if not hasattr(net, "gen") or len(net.gen) == 0:
        return np.array([], dtype=int)
    g = net.gen
    mask = g["in_service"].to_numpy(bool)
    if "slack" in g.columns:
        mask &= ~g["slack"].to_numpy(bool)
    return np.asarray(g.index[mask], dtype=int)


# ----------------------------------------------------------------------------
# Contingency order sampler  (how many components to drop this sample)
# ----------------------------------------------------------------------------
def _draw_order(
    mode: str,
    n_branches: int,
    rng,
    *,
    bernoulli_p: float,
    ratio_weights: Sequence[float],
    poisson_q: float,
    k_cap: int,
) -> int:
    """Return the contingency order k (number of components to drop)."""
    if mode == "none":
        return 0

    if mode == "bernoulli":
        # Legacy behaviour kept for reproducibility: number of lines out is
        # Binomial(L, p).  No explicit k control; we still cap it.
        if n_branches == 0:
            return 0
        k = int(np.sum(rng.random(n_branches) < bernoulli_p))
        return min(k, k_cap)

    if mode == "ratio":
        # PF-Delta style: P(N-0), P(N-1), P(N-2), ...
        w = np.asarray(ratio_weights, dtype=float)
        w = w / w.sum()
        return int(rng.choice(len(w), p=w))

    if mode == "poisson":
        # Size-scaled realism: expected outages = L * q.
        lam = max(n_branches, 0) * float(poisson_q)
        k = int(rng.poisson(lam))
        return min(k, k_cap)

    raise ValueError(f"Unknown contingency mode {mode!r}")


@dataclass
class ContingencyResult:
    """Description of what was dropped (for logging / dataset columns)."""
    order: int = 0                      # realised k (components actually out)
    dropped_lines: List[int] = field(default_factory=list)
    dropped_gens: List[int] = field(default_factory=list)
    changed_ybus: bool = False          # True iff any line was dropped


def sample_contingency(
    net,
    rng,
    *,
    mode: str = "none",                 # "none" | "bernoulli" | "ratio" | "poisson"
    elements: Sequence[str] = ("line",),  # subset of {"line", "gen"}
    # --- bernoulli mode ---
    bernoulli_p: float = 0.0,
    # --- ratio mode (PF-Delta) ---
    ratio_weights: Sequence[float] = (0.55, 0.27, 0.18),  # P(N-0,N-1,N-2)
    # --- poisson mode (size-scaled) ---
    poisson_q: float = 5e-4,            # per-branch steady-state unavailability
    # --- shared ---
    k_cap: int = 2,                     # hard cap on simultaneous outages
    max_tries: int = 25,                # connectivity retries before giving up
) -> ContingencyResult:
    """
    Draw and APPLY a topology contingency in place on `net`.

    Selectable by `mode`:
        "none"      -> never drops anything (N-0 only).
        "bernoulli" -> legacy per-line Bernoulli(p), capped at k_cap.
        "ratio"     -> draw order k from a fixed categorical `ratio_weights`
                       (e.g. PF-Delta's ~55/27/18 for N-0/N-1/N-2).
        "poisson"   -> draw k ~ Poisson(L * poisson_q), capped — k grows with
                       grid size, so small grids stay near N-0/N-1 and large
                       grids occasionally reach higher orders.

    `elements` chooses what may be dropped: lines only, or lines and
    generators (generator outages create a real supply deficit the slack must
    absorb — useful hard cases, never drops the slack itself).

    Connectivity is enforced: if a draw islands the network, it is reverted and
    re-drawn up to `max_tries`; if no feasible contingency of the drawn order is
    found, the sample falls back to N-0 (nothing dropped).

    NOTE: dropping a *line* changes Y_bus (sets `changed_ybus=True`); dropping a
    *generator* does NOT change Y_bus (it changes the injection / bus type).
    Datasets using line outages therefore require share_grid=False.
    """
    res = ContingencyResult()
    if mode == "none":
        return res

    line_pool = _active_line_index(net) if "line" in elements else np.array([], dtype=int)
    gen_pool = _droppable_gen_index(net) if "gen" in elements else np.array([], dtype=int)
    pool = [("line", i) for i in line_pool] + [("gen", i) for i in gen_pool]
    if len(pool) == 0:
        return res

    k = _draw_order(
        mode, n_branches=len(line_pool), rng=rng,
        bernoulli_p=bernoulli_p, ratio_weights=ratio_weights,
        poisson_q=poisson_q, k_cap=k_cap,
    )
    k = min(k, len(pool))
    if k <= 0:
        return res

    for _ in range(max_tries):
        sel = rng.choice(len(pool), size=k, replace=False)
        picked = [pool[s] for s in sel]
        lines = [i for kind, i in picked if kind == "line"]
        gens = [i for kind, i in picked if kind == "gen"]

        # apply
        if lines:
            net.line.loc[lines, "in_service"] = False
        if gens:
            net.gen.loc[gens, "in_service"] = False

        if _is_connected(net):
            res.order = k
            res.dropped_lines = list(map(int, lines))
            res.dropped_gens = list(map(int, gens))
            res.changed_ybus = len(lines) > 0
            return res

        # revert and retry
        if lines:
            net.line.loc[lines, "in_service"] = True
        if gens:
            net.gen.loc[gens, "in_service"] = True

    # No feasible contingency of this order found -> N-0 fallback.
    return res


# ----------------------------------------------------------------------------
# Admittance perturbation  (R / X jitter -> Y_bus varies per sample)
# ----------------------------------------------------------------------------
def apply_admittance_jitter(
    net,
    rng,
    *,
    sigma: float = 0.0,                 # 0.0 disables; e.g. 0.1 = +/-10%
    jitter_lines: bool = True,
    jitter_trafos: bool = True,
) -> bool:
    """
    Multiply branch resistance and reactance by independent uniform factors
    drawn from U(1-sigma, 1+sigma), per branch, per sample.  Recomputing Y_bus
    from the perturbed branch data is handled downstream by the normal PPC
    compilation (this only edits the pandapower element tables).

    Models temperature/measurement uncertainty in line parameters; also acts as
    data augmentation so the surrogate learns sensitivity to Y_bus, not just to
    the injections.  Returns True iff anything was perturbed (Y_bus varies).

    Selectable: set sigma=0.0 to disable entirely, or toggle lines/trafos.
    """
    if sigma <= 0.0:
        return False

    lo, hi = max(0.0, 1.0 - sigma), 1.0 + sigma
    changed = False

    if jitter_lines and hasattr(net, "line") and len(net.line):
        n = len(net.line)
        net.line["r_ohm_per_km"] = net.line["r_ohm_per_km"].to_numpy(float) * rng.uniform(lo, hi, n)
        net.line["x_ohm_per_km"] = net.line["x_ohm_per_km"].to_numpy(float) * rng.uniform(lo, hi, n)
        changed = True

    if jitter_trafos and hasattr(net, "trafo") and len(net.trafo):
        n = len(net.trafo)
        # vk_percent ~ |z|, vkr_percent ~ r.  Scale both; keep vkr <= vk.
        vk = net.trafo["vk_percent"].to_numpy(float) * rng.uniform(lo, hi, n)
        vkr = net.trafo["vkr_percent"].to_numpy(float) * rng.uniform(lo, hi, n)
        net.trafo["vk_percent"] = vk
        net.trafo["vkr_percent"] = np.minimum(vkr, 0.999 * vk)
        changed = True

    return changed


# ----------------------------------------------------------------------------
# Top-level config: records which knobs change Y_bus
# ----------------------------------------------------------------------------
@dataclass
class PerturbationConfig:
    """
    One object describing an entire perturbation regime.  Pass the relevant
    fields into case_generation_pandapower(); read `ybus_varies` to decide
    share_grid at training time.
    """
    # ---- Family 1: Y_bus FIXED ----
    load_scale_range: Optional[Tuple[float, float]] = None
    scale_gen_with_load: bool = True
    jitter_load: float = 0.0
    jitter_load_q: float = 0.0
    jitter_gen: float = 0.0
    pv_vset_range: Optional[Tuple[float, float]] = None
    rand_u_start: bool = False
    angle_jitter_deg: float = 0.0
    mag_jitter_pq: float = 0.0

    # ---- Family 2: Y_bus VARIES ----
    contingency_mode: str = "none"      # none|bernoulli|ratio|poisson
    contingency_elements: Tuple[str, ...] = ("line",)
    contingency_ratio_weights: Tuple[float, ...] = (0.55, 0.27, 0.18)
    contingency_poisson_q: float = 5e-4
    contingency_k_cap: int = 2
    admittance_sigma: float = 0.0

    @property
    def ybus_varies(self) -> bool:
        """
        True if this regime can change Y_bus across rows, in which case the
        training loader MUST use share_grid=False (else every row silently
        gets row-0's Y_bus).  Generator-only contingencies do NOT change Y_bus.
        """
        topo_changes_ybus = (
            self.contingency_mode != "none" and "line" in self.contingency_elements
        )
        return bool(topo_changes_ybus or self.admittance_sigma > 0.0)


# ---------------------------------------------------------------------------
# Zone-tilt multipliers
# ---------------------------------------------------------------------------
# Measured on EIA-930 subregion demand for ISNE (the eight ISO New England load
# zones), all 8,783 usable hours of 2024.  Per hour the zonal shares are formed,
# logged, centred per zone (removing "this zone is simply larger") and then per
# hour (removing the global load level, which the corpus already varies).  What
# is left is the spatial tilt on the same log scale as `sigma`.
#
# Zone identities do NOT transfer -- our zones are grown on a GB network and are
# not ISO-NE load zones -- so only dimensionless structure is carried over:
#
#   EIGENVALUES: how concentrated the tilt is in a few spatial modes.  An
#     independent-zone model implies all eigenvalues equal to 1; the measured
#     spectrum runs 3.94 .. 0.16, with the leading mode alone carrying 49% and
#     an effective rank of 4.2 out of 8.  Independence is badly wrong.
#   SD_RATIOS: how unequal the zones are in how much they move.  The measured
#     ratios span 2.88 .. 0.35 (cv 0.78), so equal variances are wrong too.
#
# The measured spectrum also carries an exact zero eigenvalue.  That one is
# structural rather than geographic -- shares sum to one, so the centred
# log-shares are constrained to sum to zero -- and it is reproduced exactly, by
# construction, rather than transferred.
EMPIRICAL_TILT_EIGENVALUES = (3.9420, 1.7377, 1.0793, 0.4597, 0.3514, 0.2741, 0.1557)
EMPIRICAL_TILT_SD_RATIOS = (2.8801, 1.0460, 0.8485, 0.7589, 0.7492, 0.7168, 0.6475, 0.3530)

# The same statistics measured on a second real system, so the transfer is not
# resting on one grid: RTE eCO2mix consolidated regional consumption, France's
# 12 administrative regions, all 17,566 half-hours of 2024, identical arithmetic.
# France agrees with ISO New England on the property that matters -- a normalised
# effective rank of 0.585 against 0.525, both far from the 1.0 independence would
# give -- but is less concentrated (top mode 32.9% against 49.3%) and much more
# even between zones (sd dispersion 0.24 against 0.78; ISO-NE's figure is driven
# by Vermont alone at 2.88x the mean).  Use it to check how much a conclusion
# depends on the ISO-NE spectrum specifically.
FRANCE_TILT_EIGENVALUES = (3.9470, 2.6270, 1.4010, 1.1730, 0.7486, 0.5952,
                           0.5143, 0.4074, 0.2442, 0.1974, 0.1455)
FRANCE_TILT_SD_RATIOS = (1.5570, 1.2640, 1.1850, 1.1060, 0.9950, 0.9400,
                         0.9160, 0.8910, 0.8620, 0.8380, 0.7750, 0.6700)

# Great Britain, measured the same way: Elexon P114 C0291 (CDCA-I029) Aggregated
# GSP Group Take, all 14 GSP groups, every settlement period of 2025 (17,518
# half-hours, settled RF/R3 runs).  This is the system the corpus is actually
# built on, so it is the default; the other two profiles exist to show how much
# a conclusion depends on which system the structure came from.
#
# GB is the most concentrated of the three (normalised effective rank 0.413,
# against 0.525 for ISO-NE and 0.585 for France) and the most uneven between
# zones (sd dispersion 0.90).  Note GSP Group Take is metered NET demand, so
# embedded solar and wind are subtracted from it; GB's high distributed
# generation share is the likely reason both its spread and its unevenness
# exceed the other two.
GB_TILT_EIGENVALUES = (5.7510, 3.4980, 1.6080, 0.8541, 0.5752, 0.4744, 0.2974,
                       0.2740, 0.2272, 0.1636, 0.1101, 0.0911, 0.0757)
GB_TILT_SD_RATIOS = (3.6220, 2.2600, 1.3650, 0.9820, 0.7830, 0.7190, 0.5840,
                     0.5630, 0.5550, 0.5460, 0.5440, 0.5150, 0.5100, 0.4530)

TILT_PROFILES = {
    "empirical_gb": (GB_TILT_EIGENVALUES, GB_TILT_SD_RATIOS),
    "empirical_isone": (EMPIRICAL_TILT_EIGENVALUES, EMPIRICAL_TILT_SD_RATIOS),
    "empirical_fr": (FRANCE_TILT_EIGENVALUES, FRANCE_TILT_SD_RATIOS),
}

# "empirical" is an alias for the default profile.  Corpora generated before GB
# data was available used the ISO-NE spectrum under this same name, and their
# filenames carry the bare "_zcempirical" tag; new runs resolve the alias before
# tagging, so anything written from now on names the profile it actually used.
DEFAULT_TILT_PROFILE = "empirical_gb"
TILT_PROFILES["empirical"] = TILT_PROFILES[DEFAULT_TILT_PROFILE]


def _resample_profile(values, n):
    """Stretch or squeeze a measured profile to n entries, preserving its shape."""
    v = np.asarray(values, dtype=float)
    if n == len(v):
        return v.copy()
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(v)), v)


def _clr_rms(cov, n):
    """Centred-log RMS implied by a latent covariance.

        R(Sigma)^2 = tr(H Sigma H) / Z,     H = I - 11^T / Z

    This is the amplitude the tilt actually realises, and it is what should be
    held equal when comparing covariance STRUCTURES: an IID and a GB covariance
    with the same nominal sigma do not realise the same amplitude, because GB's
    per-zone spreads are uneven and variance depends on their squares.
    """
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.sqrt(max(np.trace(H @ cov @ H), 0.0) / n))


def sample_zone_multipliers(n_zones, sigma, rng, correlation="none",
                            structure_seed=0, normalize="arithmetic",
                            amplitude_mode="sigma"):
    """Draw one multiplier per zone, mean-corrected so total load is unchanged.

    amplitude_mode="sigma"    `sigma` is the marginal latent sd, the original
                              meaning.  Realised centred-log RMS then depends on
                              the covariance shape: sigma*sqrt(1-1/Z) for IID,
                              but 1.32x that for the GB profile at Z=14.
    amplitude_mode="clr_rms"  `sigma` IS the target centred-log RMS.  The
                              covariance is scaled by one global scalar to hit
                              it, which leaves eigenvalue ratios, correlations
                              and spread heterogeneity untouched and changes
                              only amplitude.  Use this to compare structures.

    correlation="none"      independent lognormal, the original behaviour.
    correlation="empirical"/"empirical_fr"
                            lognormal carrying the measured spatial structure:
                            the correlation eigenvalue spectrum and the per-zone
                            spread inequality above, placed on a random
                            orthonormal basis of the sum-to-zero subspace.

    Randomising the basis is the honest choice: the measured spectrum says how
    much structure there is, while which zone pairs move together is a fact
    about New England geography that says nothing about a GB network.

    The basis is drawn from `structure_seed`, NOT from the per-scenario `rng`,
    and so is the same for every scenario in a corpus.  This is not a detail:
    a basis redrawn per scenario averages out to an isotropic ensemble, which
    reinstates exactly the independence this mode exists to remove.  It is also
    what the data means -- correlation between zones is a standing property of
    the grid, like the zone partition itself, not something resampled hourly.
    Callers using this mode must therefore hold the zone partition fixed across
    the corpus too, or zone index i will not refer to the same region twice.
    """
    if n_zones < 1:
        return np.ones(0)
    # sigma == 0 is NOT an early return: the draw still happens and is scaled to
    # zero.  A control corpus must consume the same latent vector at the same
    # point in the stream as a tilted one, or everything drawn afterwards --
    # load jitter, generator jitter, setpoint perturbation -- shifts and the
    # control stops being the untilted twin of the same scenario.

    # Both branches draw exactly n_zones standard normals, in the same place in
    # the stream, so an independent and a correlated corpus built from the same
    # seed see identical global scale and jitter and differ only in how the
    # latent vector is transformed.
    # The covariance SHAPE is built first, then scaled to the requested
    # amplitude, so that both branches consume the same white latent xi and
    # differ only in the transform applied to it.
    n = n_zones
    if correlation == "none":
        cov0 = np.eye(n)
    elif correlation.removesuffix("_diag") in TILT_PROFILES:
        # "<profile>_diag" keeps the profile's marginal spreads and drops every
        # cross-zone correlation: Sigma_diag = diag(Sigma).  It separates the two
        # things the empirical profile carries -- unequal zone spreads, and the
        # correlation between zones -- which are otherwise confounded in any
        # comparison against IID.
        diag_only = correlation.endswith("_diag")
        eigenvalues, sd_ratios = TILT_PROFILES[correlation.removesuffix("_diag")]
        # srng is consumed identically either way, so the shuffled spreads are
        # the same zone by zone and only the off-diagonals differ.
        srng = np.random.default_rng(structure_seed)
        # Basis of the subspace orthogonal to the constant vector.  The constant
        # direction is the grid-wide load level, which the global load scale
        # already covers; leaving it out is what makes this a tilt, and it is
        # also the exact zero eigenvalue seen in the measured spectrum.
        A = srng.normal(size=(n, n))
        A[:, 0] = 1.0
        Q, _ = np.linalg.qr(A)
        Qperp = Q[:, 1:]                              # [n, n-1]

        lam = _resample_profile(eigenvalues, max(n - 1, 1))
        lam = lam * (n - 1) / lam.sum() if lam.sum() > 0 else lam
        C = (Qperp * lam) @ Qperp.T

        d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
        C = C / np.outer(d, d)                        # back to unit diagonal

        sd = _resample_profile(sd_ratios, n)
        sd = sd / sd.mean()
        srng.shuffle(sd)                              # no zone is privileged
        cov0 = np.diag(sd ** 2) if diag_only else C * np.outer(sd, sd)
    else:
        raise ValueError(f"unknown zone correlation mode {correlation!r}")

    if amplitude_mode == "clr_rms":
        r0 = _clr_rms(cov0, n)
        scale = (sigma / r0) if r0 > 0 else 0.0
    elif amplitude_mode == "sigma":
        scale = sigma
    else:
        raise ValueError(f"unknown amplitude_mode {amplitude_mode!r}")
    cov = (scale ** 2) * cov0

    # eigh rather than Cholesky: the correlated covariance is singular by
    # construction (the constant direction was removed), so Cholesky fails.
    w, V = np.linalg.eigh(cov)
    z = V @ (np.sqrt(np.clip(w, 0.0, None)) * rng.normal(size=n))

    m = np.exp(z)
    if normalize == "none":
        # Raw exp(u).  The caller normalises, which is what load-weighted
        # normalisation needs: it depends on the per-zone load at tilt time,
        # which this function cannot see.
        return m
    if normalize == "arithmetic":
        # Legacy.  Sets the mean of the multipliers to 1, which preserves total
        # load only if every zone carries the same load -- they do not, so this
        # leaks a level change into what is meant to be a pure tilt.
        return m / m.mean()
    raise ValueError(f"unknown normalize mode {normalize!r}")
