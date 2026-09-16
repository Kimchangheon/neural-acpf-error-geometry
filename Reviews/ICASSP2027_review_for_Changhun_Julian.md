# Internal review of the ICASSP 2027 manuscript

Changhun,

I reviewed the updated manuscript text. The paper has a clear contribution: it investigates how prediction-error direction affects AC power-balance residuals and uses that finding to motivate a simple post-training correction. The calibration/projection ablation, fixed-norm interventions, Newton-step comparison, and distribution-shift analysis provide a strong structure.

The revised Table 2 is useful because it now compares controls with similar retained reference variation. My main requests concern the interpretation and reproducibility of the results. I would prioritize the checks below over adding further models or substantially expanding the experiments.

This review is based on the updated manuscript text and the figures supplied earlier. I have not verified the experimental implementation or raw results. The technical questions below should therefore be treated as checks, not as established implementation errors.

## 1. Make the central geometry experiments reproducible

**Location: “Off-subspace error and directional sensitivity,” particularly the 1,563× Jacobian-gain result and fixed-norm intervention.**

The directional analysis is central to the paper’s contribution, so readers need enough information to understand exactly what was measured.

For the Jacobian experiment, please specify:

- Whether directions perturb magnitudes, angles, or both, including the angle units and any coordinate scaling.
- How the directions are sampled, projected, and normalized.
- How prescribed voltages and reference-angle directions are handled.
- Whether 4,096 paired directions means the total across 128 operating points or the count at each point.
- How near-zero in-subspace gains are handled when computing ratios.

Please report absolute in-/off-subspace gains and a few ratio quantiles alongside the median ratio. This would make the 1,563× result much easier to assess. Also keep the distinction between the median of paired ratios and the ratio of medians explicit; the current text uses both summaries for different analyses.

For the fixed-norm experiment, please confirm that the magnitude-error norm remains fixed **after restoring prescribed voltages**, using the same state that enters the PB evaluator. Otherwise the scoring transformation could change the norm and weaken the claim that direction alone is being varied. State the number of scenarios and whether Relative Mean PB is a ratio of aggregate means or an average of scenario-wise normalized values.

Finally, define the “linearized-to-nonlinear residual-change ratio.” If it compares norms only, values near one establish similar magnitudes, but not vector agreement. Either report a relative vector error such as `||Delta r - J e|| / ||Delta r||`, with the quantities defined, or narrow the interpretation accordingly.

## 2. Define the revised basis controls precisely

**Location: “Solution subspace specificity” and Table 2.**

Matching retained reference variation is a useful improvement. Please define that quantity mathematically. For a smoothing operator, retained output variance, retained squared norm, and one minus normalized reconstruction error are not generally interchangeable.

Please also specify the Laplacian weighting/normalization, the spatial operator controlled by beta, and whether all controls use identical centering, angle alignment, and restoration of prescribed variables. State whether “Pred. PB” is evaluated on calibrated predictions. The very large residuals after transforming reference solutions make these details particularly important.

One immediate wording correction:

> “On the test set, all three therefore preserve nearly identical reference variation”

should read:

> “On the test set, all three also preserve nearly identical reference variation.”

Matching on the training set does not guarantee matching on the test set; the table establishes the latter empirically.

## 3. Separate calibration gains from projection gains across grids and shifts

**Location: Figure 3 and “Cross-grid breadth and distribution shift.”**

Figure 3 compares raw predictions with CSP. It therefore demonstrates the benefit of the combined calibration/projection pipeline, but does not isolate the additional benefit of projection.

Please add calibration-only Mean PB and actual retained ranks for each system. This matters particularly for the small grids: if the retained basis is complete, `P = I`, and CSP becomes calibration alone. Such a result would still be valid, but its interpretation changes.

Please also clarify whether the surrogate is shared across grids or trained separately. The per-grid fitting of calibration and basis is already disclosed and should remain prominent.

A defensible sentence for the current comparison is:

> “Relative to raw GridSFM predictions, the combined calibration and projection pipeline reduces Mean PB across all 31 systems.”

For Table 4, please include calibration-only results under the same Control, N−1, N−2, and GBcorr conditions, using the same frozen model outputs. The current table shows how CSP behaves under shift and refitting, but not whether projection remains beneficial relative to calibration under those shifts. Please state scenario counts, outage sampling/islanding exclusions, and whether neural weights remain frozen during refitting.

The outage interpretation can also be more precise:

> “Under N−1/N−2 outages, tracking metrics and mean PB remain close to control, while the maximum observed PB increases substantially. This highlights the importance of examining extreme residuals alongside average performance.”

This reports the observations without inferring the location or cause of individual violations from maxima alone.

## 4. Clarify the evaluated model variants and implementation details

**Location: Abstract and experimental setup.**

The setup explains that all models are trained from scratch and that several backbones are adapted to AC PF. The abstract should make this equally clear, so the results are not mistaken for an evaluation of released pretrained foundation-model checkpoints.

Suggested wording:

> “We study this accuracy–consistency gap using AC-PF implementations of PIGNN-GC, GridSFM, gridfm-graphkit, and LUMINA trained from scratch on 2224-bus GBnetwork scenarios, with additional GridSFM evaluation across 31 systems.”

Please provide the exact configurations/versions, objective terms and weights, initialization, training budget, and model-selection rules. A compact setup table and an inspectable configuration snapshot would be sufficient. “Model-specific stable initialization” and “enhanced AC-PF configuration” currently leave important choices unspecified.

Please clarify whether k=16 was selected independently for each model/seed or selected once and shared. If shared, identify the validation objective and model/seed used. Confirm whether both magnitude and gauge-aligned angle matrices were centered before SVD.

For the common residual evaluator, please confirm consistent bus masks and known-variable restoration across scoring and diagnostics. State the reactive-limit/PV-to-PQ policy, which variables enter voltage RMSE, and the residual obtained when the same evaluator scores converged references. Clarify that fixing generator dispatch excludes the unknown slack injection.

## 5. Align Figure 1 with the deployed CSP transformation

**Location: Figure 1 and the CSP equation.**

Please check the current figure against the actual algorithm. The earlier supplied figure depicts projection of the true prediction error, which requires the unknown reference. Deployed CSP instead projects the calibrated prediction around the training mean.

Writing `P` for the projector and `mu` for the training mean, its resulting error is:

    e_CSP = P e_C - (I-P)(x* - mu).

The second term represents valid reference variation discarded by projection. The text already handles this correctly in the error tradeoff identity. Please make the figure consistent with that explanation, or explicitly label its error-decomposition panel as an oracle diagnostic. Avoid implying that projection guarantees a small residual.

## 6. Keep the performance interpretation precise

**Locations: Tracking metrics, timing discussion, and Newton refinement.**

The improvement in squared correlation is useful, but the response slope remains below one for all four models. A more precise description is:

> “Projection improves within-bus squared correlation with little change in the response slope, which remains below one for all four models.”

Please state the zero-variance convention for the per-bus-mean baseline: its correlation is mathematically undefined, although the table assigns zero. That convention currently appears only in commented source. Also identify the scenarios used for per-bus centering.

The timing results concern amortized batched GPU execution. Replace “negligible computational burden” with:

> “CSP incurs little additional cost in the evaluated batched GPU setting and improves the results of one Newton refinement step.”

The Newton experiment supports an improvement after one update; it does not establish faster complete convergence. The LUMINA overshoot is appropriately disclosed. Please make the three paired seed results available so the large aggregate improvement can be interpreted alongside that outlier.

## Small text and submission fixes

| Location | Proposed correction |
|---|---|
| Abstract | Spell out “mean power-balance residual” instead of introducing “Mean PB” without a definition. |
| State definition | Explicitly define `V` as the vector of voltage magnitudes and `theta` as phase angles before defining the complex voltage. The new notation is fine once this distinction is stated. |
| LUMINA Newton result | Write `$(1.653\pm0.036)\times10^{-3}$` so the multiplier applies to both the mean and standard deviation. |
| Table 4 caption | State that Mean, Median, p95, p99, and Max summarize PB in p.u., and that magnitude RMSE is also in p.u. |
| Experimental setup | Remove the extra period before “Training and inference use …”. |
| Code link | Check the intended repository URL and access. The earlier review received a 404 through its available route; this could reflect visibility or access restrictions. |
| Bibliography/build | The earlier package contained a duplicate `owerko2020opf` key. The updated bibliography was not supplied, so please confirm this is fixed and check the final compiled layout. |

The reported headline percentage reductions agree with the main table, and the affine projection-error identity is algebraically correct. I would focus the remaining work on making the geometry analysis auditable, separating calibration from projection effects, and keeping the model and inference claims within the evaluated setting. These changes should strengthen the paper without requiring a new research direction.
