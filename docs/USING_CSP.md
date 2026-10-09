# Using CSP on your own surrogate

CSP is a transform on a model's output. It does not change the model, does not
need gradients, and does not need retraining. If you can produce voltage states
as tensors, you can apply it.

## The minimum

```python
import torch
from csp import CSP, evaluate

# [n_scenarios, n_bus, 2] — last axis is (|V| in pu, angle in radians)
transform = CSP.fit(train_predictions, train_references, rank=16)
corrected = transform(test_predictions)
```

`train_predictions` and `train_references` must be the **same scenarios in the
same order**: the calibration is a paired difference. CSP raises if they are
not the same shape, but it cannot detect a permutation, so check that yourself.

## Choosing `k`

`k` is the one hyper-parameter. Pick it on a validation split.

```python
from csp import fit_basis

for k in (2, 4, 8, 16, 32):
    b = fit_basis(train_references, k)
    print(k, f"{100 * b.energy_mag:.2f}%", f"{100 * b.energy_ang:.2f}%")
```

`energy_mag` is the fraction of reference magnitude variance the `k` directions
capture. It tells you where the reference states stop varying, which is a good
place to start, but it is not a selection criterion on its own — a larger `k`
captures more reference variance *and* admits more error directions. Score
candidate `k` values on validation and take the best.

The paper uses k = 16 on a 2224-bus system, where the magnitude channel is
already past 98 % well before that.

## Producing the inputs

Nothing here is specific to our pipeline. A sketch:

```python
model.eval()
preds, refs = [], []
with torch.no_grad():
    for batch in train_loader:
        preds.append(model(batch))            # [b, n_bus, 2]
        refs.append(batch["solution"])        # [b, n_bus, 2]
train_predictions = torch.cat(preds)
train_references = torch.cat(refs)
```

If the training split does not fit in memory, fit the two steps separately:

```python
from csp import CSP, fit_basis, fit_offset_streaming

offset = fit_offset_streaming((model(b), b["solution"]) for b in train_loader)
basis = fit_basis(reference_subset, rank=16)   # the basis needs far fewer scenarios
transform = CSP(offset=offset, basis=basis)
```

The basis is an SVD of the reference states, so a few thousand scenarios are
usually plenty even when the calibration uses all of them.

## Reading the result

```python
for name, state in transform.stages(test_predictions).items():
    print(name, evaluate(state, test_references,
                         ybus=Y, s_specified=S, bus_type=bt))
```

`stages` returns `raw`, `C`, `P16`, `CSP16` — the four rows of the paper's
tables, from one model forward pass.

Two metrics deserve care:

**`within_r2` and `within_slope`.** Voltage magnitudes sit in a narrow band
around 1 pu, so a pooled R² is dominated by the between-bus spread and a model
that simply predicts each bus's mean will score well. These centre each bus on
its own mean first, so they measure scenario tracking. Read them together: R²
rising while the slope stays put means a wrong direction was removed; both
rising means response was recovered. CSP does the first.

**`mean_pb`.** The per-bus power-balance residual, scored only where the
quantity is specified — ΔP everywhere except the slack, ΔQ at PQ buses only.
Elsewhere the solver is free to choose and a residual is not an error. Computed
in complex128; in float32 a good surrogate's mismatch is at the noise floor of
the matrix-vector product itself.

## Projection scope

Three scopes exist. The default is what the paper uses.

```python
CSP.fit(..., scope="full")                              # project everything
CSP.fit(..., scope="restore_known", bus_type=bt)        # project, then write setpoints back
CSP.fit(..., scope="unknown_only", bus_type=bt)         # fit and project only free coordinates
```

In the paper's setting `full` and `restore_known` are numerically identical —
the projection does not move the prescribed coordinates, so restoring them has
nothing to do. `unknown_only` is a genuinely different operator and gives
different numbers; it is not what the published rows use. If your basis is
fitted differently from ours, `restore_known` is the safe default, because a
state whose PV setpoint has moved is not a solution to the problem that was
posed, however small its residual.

## Saving

```python
transform.save("csp_k16.pt")
transform = CSP.load("csp_k16.pt")
```

The file holds the offset, the basis and the scope. It is tied to one grid and
one model: the offset is that model's bias, and the basis is that grid's
solution subspace. CSP raises if you apply it to states with a different bus
count, but it cannot detect a different model on the same grid — keep them
paired yourself.

## When CSP will not help

- **If the error is mostly in-subspace.** CSP removes bias and off-subspace
  components. A model that tracks the wrong scenario within the subspace is not
  improved, and should not be.
- **If the topology changes.** The basis is fitted on one grid's solutions.
  Under an N-1 outage the transferred basis degrades; the paper reports both
  transferring it and refitting on condition-matched data, and refitting is
  better. See `tab:shift` in the manuscript.
- **If you have very few training scenarios.** `rank` is clipped to the number
  of scenarios, and a basis fitted on a handful of states describes those states
  rather than the grid.
