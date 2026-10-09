# CSP — calibrated subspace projection for AC power-flow surrogates

Code for *"Tracking, calibration and error geometry of neural AC power-flow
surrogates"* (ICASSP 2027). Manuscript:
[`paper/submitted_manuscript.tex`](paper/submitted_manuscript.tex).

## The idea in one paragraph

A neural power-flow surrogate can have small voltage error and still return a
state that violates the network equations. The error does not fill the space —
it concentrates in a low-dimensional subspace spanned by the solutions the grid
actually produces. That makes most of it removable **after** training, from a
frozen checkpoint, in two steps:

1. **C** — subtract the per-bus bias measured on the training split.
2. **P_k** — project onto the rank-`k` training-solution subspace.

On the 2224-bus GBnetwork this cuts mean power-balance violation by a factor of
2 to 20 across four different surrogates, while voltage RMSE improves for every
one of them, at a cost at or below measurement noise.

```python
from csp import CSP, evaluate

transform = CSP.fit(train_predictions, train_references, rank=16)   # train split only
corrected = transform(test_predictions)

evaluate(corrected, test_references, ybus=Y, s_specified=S, bus_type=bt)
# {'vmag_rmse': ..., 'angle_rmse_deg': ..., 'within_r2': ..., 'mean_pb': ...}
```

## Install and try it

```bash
pip install -e .
python examples/quickstart.py     # synthetic grid, no downloads, ~1 s
pytest                            # 26 tests, including equivalence with the paper code
```

`examples/quickstart.py` builds a small grid, a surrogate with the three error
components the paper decomposes, and shows what each step of CSP removes.

## What CSP needs from you

Only tensors — it never touches your model.

| | shape | what it is |
|---|---|---|
| `train_predictions` | `[n_scenarios, n_bus, 2]` | your model's output on training scenarios |
| `train_references` | `[n_scenarios, n_bus, 2]` | solver solutions for those same scenarios |
| `test_predictions` | `[n_scenarios, n_bus, 2]` | whatever you want corrected |

The last axis is `(voltage magnitude in per-unit, voltage angle in radians)`.
For the power-balance metric you also supply `ybus`, the specified injections,
and `bus_type` (1 = slack, 2 = PV, else PQ).

Everything is fitted on the training split. Choose `k` on validation, never on
test; `Basis.energy_mag` reports how much reference variance a given `k`
captures, which is a reasonable way to shortlist candidates.

## Repository layout

```
csp/            the method — small, dependency-light, this is what you use
examples/       runnable quickstart, no data required
tests/          properties, plus numerical equivalence with the paper's code
docs/           how to use it, and how to reproduce the paper
paper/          submitted manuscript and its figures
reproduction/   the research archive: training drivers, job scripts, results
third_party/    released sources of GridSFM, LUMINA and GridFM-GraphKit
```

`csp/` and `reproduction/` are deliberately separate. The first is the method,
rewritten to be read and reused. The second is what actually ran on the cluster:
four training drivers, 57 batch scripts, and the result artifacts. It is kept
verbatim so the published numbers stay traceable, and it is **not** importable
as a library.

The two are tied together by `tests/test_equivalence_with_paper_code.py`, which
checks the rewritten basis, projection, calibration and power-balance code
against the archived implementations numerically. If that test passes, `csp/`
computes what the paper computed.

## Where to go next

| If you want to | Read |
|---|---|
| use CSP on your own surrogate | [`docs/USING_CSP.md`](docs/USING_CSP.md) |
| trace a number in the paper to the code that made it | [`docs/REPRODUCING_THE_PAPER.md`](docs/REPRODUCING_THE_PAPER.md) |
| get the corpora or the checkpoints | [`docs/DATA_AND_CHECKPOINTS.md`](docs/DATA_AND_CHECKPOINTS.md) |
| know what is in the research archive | [`reproduction/README.md`](reproduction/README.md) |

## Honest notes

- **CSP does not make a bad surrogate good.** It removes two error components —
  a fixed per-bus bias and directions the reference states never exhibit. A
  scenario-tracking error *inside* the subspace survives it, and should: that is
  real error, not a wrong direction. This is why the projection raises
  subspace-weighted R² substantially while leaving the response amplitude `a_w`
  almost unchanged.
- **The choice of `k` matters and is a validation decision.** The paper uses
  k = 16 on GBnetwork, selected before looking at test.
- **Seeds in the paper are model seeds.** The data split is held fixed, so the
  reported ± is sensitivity to initialisation and optimisation randomness.
  Two of the four three-seed sets mix in an older frozen baseline rather than a
  matched replicate; see `docs/DATA_AND_CHECKPOINTS.md`.

## Citation

```bibtex
@inproceedings{kim2027csp,
  title     = {Tracking, Calibration and Error Geometry of Neural AC Power-Flow Surrogates},
  author    = {Kim, Changhun and others},
  booktitle = {ICASSP},
  year      = {2027}
}
```

`third_party/` contains upstream projects under their own licences.
