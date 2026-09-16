# Adversarial technical review

# Overall assessment

Recommendation on the manuscript as supplied: major revision; I would not recommend acceptance in its current form. The idea is useful, and the reported within-model residual reductions are substantial, but the central mechanistic interpretation and experimental protocol need stronger support. This is a judgment on the evidence in the PDF, not a finding that the numerical results are false.

The paper's strongest elements are the explicit separation of state error and residual, training-only fitting of output statistics, an oracle fixed-norm diagnostic, the C/P/CSP factorial, the static mean control, and the candid statement that CSP does not project onto the AC-feasible set. Preserve these. The central revision should make the empirical claim narrower and the mathematical account more exact.

Five issues should be resolved before polishing the prose: (1) specify and verify the residual and known boundary values; (2) establish that all compared models solve the declared PF task; (3) correct the projector definition and include reference truncation; (4) test PCA–Jacobian alignment and the magnitude/angle confound; (5) compare corrected and uncorrected outputs on the same shifted scenarios.

Evidence labels matter. "Confirmed" or "visible" refers to the supplied text, equations, tables, or figures. "Unresolved" identifies information missing from the manuscript whose implementation could already be correct. Suggested experiments are proposals, not results. The mathematical examples below are constructed checks of the reasoning, not simulations of GBnetwork.

# How to use this review

The first five pages of the annotated PDF retain the original manuscript. Highlights and colored section marks contain native PDF comments; numbered chips at the foot of each page link to the full comments in the appendix. Every comment includes both a concern and a concrete remedy. The appended review is ordinary selectable PDF text, so it remains readable and printable when a viewer does not expose annotation pop-ups.

Severity: Critical means the issue can change the interpretation or validity of central results; Major means a substantial scientific or reproducibility weakness; Moderate means a meaningful limitation or precision issue; Minor means writing or presentation. Severity is not a statement that an unspecified implementation is definitely wrong. The numbered comments are ordered by manuscript location, not by priority.

Review scope: all five supplied pages, equations, tables, figures, and references were inspected. No dataset or checkpoint was supplied, and experiments were not reproduced. The code URL could not be retrieved through the review browser; no implementation audit was possible. Primary-source checks on 15 September 2026 targeted conventions, prior work, model identities, and selected metadata, not all 22 references.


# Anchored comments

## C01 | Major — Narrow the title and the central conclusion

Manuscript page 1. Evidence: Claim exceeds evidence.

Concern: The title suggests a general verdict on physics-informed learning. The manuscript studies particular trained models and a particular data generator; it does not establish that the method class intrinsically cannot be consistent. Section 2.3 correctly states that CSP itself has no feasibility guarantee. The defensible contribution is an empirical error-geometry diagnostic plus a distribution-dependent output correction.

Suggested improvement: Consider: "Voltage-Error Geometry and Calibrated Subspace Correction for Neural AC Power Flow". If retaining the provocative opening, qualify the abstract and conclusion with "in the evaluated models and operating distributions". Define consistency as a measured residual level, with a stated tolerance, and keep the no-guarantee statement prominent.

## C02 | Major — Define what counts as a small voltage error

Manuscript page 1. Evidence: Visible in reported results.

Concern: Table 1 reports raw magnitude RMSEs of 0.02850–0.04355 p.u.; all exceed the static mean baseline of 0.02341. LUMINA also has 17.068 degrees raw angle RMSE. These numbers do not support treating all four raw predictions as uniformly accurate. The strongest accuracy–residual mismatch should be identified at a specific output stage, with magnitude and angle reported together.

Suggested improvement: State the calibrated errors when motivating the gap, and explain the practical meaning of their units. Separate the high-tracking PIGNN-GC/GridSFM cases from the poorly tracking cases. Replace "achieve accurate voltage predictions" with "can exhibit modest voltage-magnitude error while retaining substantial power-balance mismatch" unless a task-specific accuracy threshold is justified.

## C03 | Moderate — Grid size is not established as the cause of amplification

Manuscript page 1. Evidence: Unsupported generalization.

Concern: Neither a large case study nor a plot of residual versus bus count establishes that increasing grid size causes an accuracy–consistency gap. Admittance scaling, near-zero branch impedances, operating point, angle errors, normalization, and the number of residual components can all change with grid size. Mean PB already averages over buses, while the maximum is sensitive to the number of observations.

Suggested improvement: Either remove this qualifier or compare matched error norms and directions across grids, reporting Jacobian gains, electrical parameter ranges, and residual normalization. Use causal wording only if the experiment controls the confounding factors; otherwise state that the phenomenon is observed on a 2,224-bus network.

## C04 | Moderate — Figure 1 depicts a stronger mechanism than demonstrated

Manuscript page 1. Evidence: Figure and text inconsistency.

Concern: The red off-subspace arrow is labeled as driving physics violation, while the diagram treats the reference as lying in the retained plane. A truncated training PCA basis generally misses part of a held-out reference. The miniature shared supervised-plus-physics loss also suggests that all named models use the same training objective, which is not established. Much of the figure text is unreadable at publication size.

Suggested improvement: Label the plane as an affine training PCA approximation and show a nonzero reference truncation component. Replace the causal arrow label with "candidate high-residual error directions". Give each evaluated training regime accurately or omit the shared loss. Enlarge the useful geometry panel and remove miniature network/loss graphics. See C20 and Appendix A for the corrected identity.

## C05 | Minor — Shorten the motivation to make room for methods

Manuscript page 1. Evidence: Writing.

Concern: The introduction spends substantial space on redispatch, market prices, and broad energy-system motivation before reaching the diagnostic question. This crowds out essential details about the models, physics loss, data generation, and evaluation protocol. The paper is an error-analysis study; the current application survey obscures that focus.

Suggested improvement: Use one short paragraph on repeated AC-PF evaluations, then explain the accuracy–residual gap, the PCA-based diagnostic, and the limited correction claim. Move broad policy and market context to one sentence. Spend the recovered space on the residual definition and the model/protocol specification.

## C06 | Minor — Repair the dangling citation and the PF/OPF transition

Manuscript page 1. Evidence: Confirmed editorial defect.

Concern: The isolated "[10]." is a sentence fragment. The preceding sentence pairs OPF and state estimation with two different roles for the network equations, but the citations are placed ambiguously. Elsewhere PF and OPF surrogates are discussed together without explaining the change in prediction task.

Suggested improvement: Suggested wording: "The same network equations also appear as constraints in AC optimal power flow [10] and as nonlinear measurement mappings in state estimation [9]." Then explicitly distinguish fixed-setpoint PF from dispatch optimization before introducing the foundation-model comparisons.

## C07 | Major — Error geometry describes a failure; it does not yet explain training failure

Manuscript page 2. Evidence: Missing evidence.

Concern: The Jacobian explains how a remaining state error produces a residual, but not why physics-aware optimization leaves that error. Finite unrolling, capacity, loss weights, residual normalization, optimization failure, and train–test shift can all generate the observed pattern. Without training and validation residuals, the paper cannot distinguish failure to minimize the objective from failure to generalize.

Suggested improvement: Report the actual loss and its normalization for each model, training/validation curves for the same evaluator, iteration budgets, and any inference correction settings. Compare supervised-only and physics-aware variants at matched state RMSE. Rephrase the central question as characterizing the geometry of residual-bearing prediction errors unless the optimization mechanism is investigated.

## C08 | Major — Make the distinction from compact/PCA learning experimentally meaningful

Manuscript page 2. Evidence: Novelty boundary.

Concern: Citing prior PCA compression is appropriate, but using the same subspace after prediction is a modest algorithmic change. The strongest potential novelty lies in the controlled diagnosis and the relation to residual sensitivity, not in PCA projection itself. The present evidence does not exclude generic low-rank denoising as the explanation.

Suggested improvement: Compare CSP with a model predicting PCA coefficients, simple shrinkage toward the training mean, and a matched-rank random or graph-spectral basis. Use the same labels and validation budget. State the novelty as post-training error analysis and correction, and identify exactly what the earlier compact-learning approach did and did not test.

Sources: [S04](https://arxiv.org/abs/2301.08840)

## C09 | Major — Make the promised implementation independently retrievable

Manuscript page 2. Evidence: Reproducibility limitation.

Concern: The supplied materials contain the PDF only. The cited repository could not be retrieved through the review browser; this does not establish that it is absent or private. Several essential protocol details are deferred to that repository, so the reported results could not be independently reproduced or checked against code during this review.

Suggested improvement: Provide an accessible, versioned archive with the evaluator, experiment configurations, split identifiers, basis/calibration fitting code, model checkpoints, and figure/table source data. Include a commit or release identifier in the paper. Keep the residual masks, model identities, rank policy, and data-generation essentials in the manuscript rather than relying entirely on the link.

## C10 | Moderate — An asymmetric Y-bus is possible, not implied by directed edges

Manuscript page 2. Evidence: Mathematical precision.

Concern: A directed graph representation does not itself make Y-bus asymmetric. For a conventional branch with complex tap t, Yft = −y/conj(t) and Ytf = −y/t. A phase-shifting tap can make these unequal; a real off-nominal tap alone does not make the off-diagonal terms unequal. Confusing representation directionality with electrical asymmetry can lead to implementation mistakes.

Suggested improvement: Write "a generally complex, potentially asymmetric bus-admittance matrix, including phase-shifting transformers". Specify the branch convention, line charging, shunts, tap side, phase-shift sign, branch status, and per-unit conversion in the evaluator. Confirm reconstructed injections against the reference solver before analyzing model errors.

Sources: [S02](https://matpower.org/docs/ref/matpower7.1/lib/makeYbus.html)

## C11 | Critical — Define the PF residual with explicit bus-type masks

Manuscript page 2. Evidence: Unresolved definition; results depend on it.

Concern: The displayed stacked ΔP/ΔQ residual over all buses is not a complete specification of conventional fixed-setpoint PF. Generator reactive power at PV buses and active/reactive slack generation are normally solved outputs. "Standard slack/PV/PQ conventions" does not tell the reader whether those channels are omitted, set to zero, compared with reference outputs, or recomputed from the prediction. These choices yield materially different PB values.

Suggested improvement: Define mP = 1 on PV/PQ buses and mQ = 1 on PQ buses, with the stated slack/distributed-slack convention; define PB_i = sqrt((mP_i ΔP_i)^2 + (mQ_i ΔQ_i)^2), the included buses, and the averaging denominator. If a different task is intended, specify every independent variable and target. Show that held-out reference solutions have near-solver-tolerance residual under exactly this evaluator.

Sources: [S01](https://matpower.app/manual/matpower/ACPowerFlow.html)

## C12 | Critical — Projection may alter known PV/slack voltage setpoints

Manuscript page 2. Evidence: Unresolved constraint handling.

Concern: The model predicts all 2N voltage coordinates, and CSP projects magnitudes and angles. The experiment also varies PV voltage setpoints. Projection can therefore alter quantities that define the input PF problem. If PB masks out PV reactive balance and slack balance, an output can appear consistent while violating the prescribed voltage magnitudes or slack angle. This is separate from optional operational inequalities.

Suggested improvement: Define the free state and reconstruct known coordinates exactly from u, or apply a constraint-preserving affine projection. Evaluate PV/slack magnitude deviations and slack-angle deviation separately. Document PV-to-PQ switching and generator Q-limit handling. If clamping follows CSP, include it in the actual method and rerun both the decomposition and final residuals; the unclamped orthogonal-projection identities then need qualification.

Sources: [S01](https://matpower.app/manual/matpower/ACPowerFlow.html)

## C13 | Major — Specify the physical units and the geometry of the norm

Manuscript page 2. Evidence: Missing units and scaling.

Concern: PB values in per unit cannot be interpreted or compared across grids without the power base. The concatenated state mixes per-unit magnitudes and angles; Euclidean geometry depends on whether angles use radians, degrees, or normalized units. Separate PCA blocks remove one mixing problem, but the Jacobian, Taylor norm, loss weighting, and reported aggregate metrics still require a consistent convention.

Suggested improvement: State S_base, voltage bases, net-generation sign, angle units in computation, normalization/denormalization, and bus weighting. Specify a state scaling D and use ||D e|| where a joint norm is needed. Convert residuals to MW/MVAr or MVA examples with the actual base and report threshold exceedance rates. Do not infer operational significance from a dimensionless PB alone.

## C14 | Major — Gauge alignment does not by itself make angle PCA well-defined

Manuscript page 2. Evidence: Missing mathematical convention.

Concern: Gauge alignment removes the arbitrary common phase, but ordinary centering, subtraction, and SVD still require a consistent local unwrapping of periodic angles. Values near −π and π can have a spurious large linear difference. Circular bias estimation followed by ordinary subtraction also needs an explicit wrap/unwrapping convention. Disconnected islands can require more than one reference.

Suggested improvement: Define the reference bus per connected component, angle alignment, wrap interval, and the local chart used for means and PCA. Confirm that no branch-cut crossings invalidate the linear chart on the evaluated data. Use wrapped angle errors for reporting and test invariance to a common phase shift. State how islands and bus-type changes are treated.

## C15 | Moderate — Train-set bias estimation is leakage-free but can be optimistic

Manuscript page 2. Evidence: Statistical limitation.

Concern: Fitting calibration on training predictions does not leak test labels, and the manuscript is right to freeze it. However, residuals on examples used to fit the neural model may not estimate the bias of unseen predictions. In Euclidean coordinates, the stated correction also simplifies exactly to training-reference mean plus a centered prediction, which clarifies what it can and cannot fix.

Suggested improvement: State b_hat = mean_train(prediction) − mean_train(reference) and show x_CSP = mean_train(reference) + P(prediction − mean_train(prediction)). Compare training-fit calibration with a held-out calibration subset or out-of-fold predictions, using equal data budgets and keeping final test data untouched. Report calibration stability versus sample count; preserve the circular-angle caveat.

## C16 | Major — Equation (1) does not define the block-diagonal projector described

Manuscript page 2. Evidence: Confirmed mathematical inconsistency.

Concern: SVD of a single stacked matrix X in R^(2N×Ntr), followed by Pk = Wk Wk^T, generally yields a joint rank-k projector with magnitude–angle cross-blocks. The prose and Section 3.2 instead specify separate magnitude and angle PCA bases. Their block-diagonal projector has rank kV + kθ, normally 2k when both are rank k. These are different algorithms.

Suggested improvement: Define XV = WV ΣV UV^T and Xθ = Wθ Σθ Uθ^T separately, then P = diag(WV,kV WV,kV^T, Wθ,kθ Wθ,kθ^T). State whether k = 16 means 16 per block or 16 total and give the actual dimensions after removing fixed coordinates. Report the numerical rank rule and feature scaling. Alternatively use a joint SVD consistently and revise the experiments to match.

## C17 | Major — Show that the retained subspace actually represents held-out solutions

Manuscript page 2. Evidence: Missing evidence.

Concern: No singular-value spectrum, explained-variance curve, reconstruction floor, or sample-size study is reported. Thus "low-dimensional" is asserted rather than quantified. A generator dominated by a global load multiplier can create a low-rank covariance even when the full operating space is much richer. Training variance explained alone does not show coverage of unseen states.

Suggested improvement: For V and θ separately, report cumulative variance, effective rank, and held-out truncation error across k. Project reference solutions themselves and recompute their PB under each original input. Repeat for control, outages, and GBcorr. Include sample-count sensitivity and richer independent/spatial input perturbations to separate properties of the scenario generator from properties of AC-PF.

## C18 | Major — Quantify the off-subspace error claimed in the abstract

Manuscript page 2. Evidence: Missing central measurement.

Concern: The paper claims substantial off-subspace prediction error, but gives no distribution of ||(I−P)e||²/||e||², no comparison with off-subspace reference variation, and no bus-wise localization. When k is tiny relative to N, even isotropic error places most energy outside the retained subspace. A large fraction alone therefore does not identify a special physical failure.

Suggested improvement: Report per-scenario in/out error energies for V and θ, their confidence intervals, and the same quantities for random matched-rank subspaces and projected references. For isotropic errors in d free dimensions the expected discarded energy fraction is (d−k)/d; use this as a sanity check, not an assumed error model. Relate residuals to direction after controlling for total error magnitude.

## C19 | Critical — A solution-covariance basis is not a residual-sensitivity basis

Manuscript page 2. Evidence: Central inference is not established.

Concern: Equation (3) establishes local sensitivity to error orientation. It does not establish that discarded PCA directions have larger Jacobian gain. For a regular reduced PF system, dx*/du = −Jx^−1 Ju, so local solution covariance depends on input covariance as well as Jx. Along varying solutions Jx dx* + Ju du = 0; it does not imply Jx dx* = 0 when the input is held fixed. The distinction is essential to the proposed mechanism.

Suggested improvement: Present high gain outside the retained PCA basis as a testable empirical hypothesis. Measure ||J e_parallel||, ||J e_perp||, their inner product, and gain per unit error, with uncertainty across operating points. Compare PCA with right-singular-vector and random bases under consistent scaling. Appendix B gives the covariance relation and a counterexample showing why no general implication follows.

## C20 | Critical — CSP discards true solution variation as well as prediction error

Manuscript page 2. Evidence: Missing term in the interpretation of Eq. (4).

Concern: Equation (4) is algebraically valid, but its error-reduction interpretation omits reference truncation. Let eC = x_hat−b_hat−x*, μ = mean_train(x*), and q = (I−P)(x*−μ). The exact CSP error is eCSP = P eC − q, not merely P eC. Consequently ||eCSP||² = ||P eC||² + ||q||². Squared state error improves only when removed calibrated-error energy exceeds lost reference variation. This distinction is especially relevant under shift.

Suggested improvement: Add the exact identity and the condition ||q||² < ||(I−P)eC||², per scenario or in expectation as appropriate. Measure both terms on held-out data. Distinguish projecting a prediction around the training mean from projecting an oracle error around the true solution. Include the PB of projected references to expose the intrinsic approximation floor. See Appendix A.

## C21 | Major — State-error contraction cannot guarantee residual contraction

Manuscript page 2. Evidence: No monotonicity guarantee.

Concern: Even if q is negligible, removing a Euclidean-orthogonal state component need not reduce PB: J maps orthogonal components to vectors that can reinforce or cancel. In a local squared residual norm, the cross-term 2⟨JPe, J(I−P)e⟩ can have either sign. Mean PB is a different, bus-grouped norm, so a spectral L2 argument alone would still not prove the reported metric improves.

Suggested improvement: Retain the empirical wording and explicitly exclude monotonic or feasibility guarantees. Report the fraction of individual scenarios whose residual worsens, not only the pooled mean. As an optional robust extension, select between C and CSP using the same input-based residual evaluator or use CSP as a warm start for an exact solver; label and benchmark this as an additional method, including its cost.

## C22 | Major — The correction requires reference solutions and grid-specific coordinates

Manuscript page 2. Evidence: Method scope and cost.

Concern: Even when the base PIGNN is trained without reference states, CSP requires paired solved states for the basis and bias. Its mean and basis are indexed by buses of a particular grid. A fixed matrix cannot directly act on a different bus count, and a bus relabeling requires the statistics to be relabeled consistently. These are material limitations for foundation-model deployment.

Suggested improvement: Describe CSP as supervised post-training adaptation using reference solutions. Report label counts and solver/precomputation cost, and specify target-grid calibration access. For a permutation Π, transform μ, b, and P consistently as Πμ, Πb, and ΠPΠ^T. Explain handling of new/deleted buses, disconnected islands, and changed bus types; do not imply a universal basis across unrelated grids.

## C23 | Moderate — Keep the no-feasibility-guarantee statement and define practical success

Manuscript page 3. Evidence: Scope clarification.

Concern: This limitation is scientifically appropriate and should be retained. However, the introduction and conclusion repeatedly use physics-consistency language without defining how much residual reduction is sufficient. Nonzero PB, even after a large relative improvement, is not evidence of satisfying a solver tolerance; and satisfying PF equations does not imply meeting operational limits.

Suggested improvement: State a numerical equality tolerance and report the fraction of complete scenarios meeting it. If operational usability is claimed, separately evaluate voltage and thermal limits and generator reactive limits with their actual task conventions. Otherwise describe the result as residual reduction and reserve feasibility/consistency claims for the explicitly tested conditions.

## C24 | Major — Realistic network topology does not establish realistic operating statistics

Manuscript page 3. Evidence: Missing data-generation specification.

Concern: The paper specifies the number of converged cases and varied quantities but omits their distributions, ranges, dependencies, and acceptance policy. Sampling system-wide loading and independent bus perturbations can produce a very particular low-rank voltage covariance. Conditioning on Newton–Raphson convergence also excludes some difficult cases and does not prove operational feasibility.

Suggested improvement: Give the network source/version, sample-generation equations, parameter ranges, covariance structure, generator participation/slack balancing, Q-limit policy, solver options/tolerance, attempted count, rejected count, and rejection reasons. Clarify whether the samples are synthetic perturbations of a realistic topology or measured operating conditions. Characterize loading, voltages, and stress in accepted versus rejected cases.

## C25 | Major — Specify exact splits and the unit of independence

Manuscript page 3. Evidence: Missing split and leakage controls.

Concern: An approximate split ratio is insufficient to reproduce the analysis. It is unclear whether the split is random by scenario, grouped by base operating profile, or separated by topology. Near-duplicate scenarios or outages derived from a shared base case can cross splits. The paper correctly says PCA/calibration use training data, but preprocessing, checkpoint selection, and foundation pretraining overlap are not documented.

Suggested improvement: Release exact counts, random seeds, split IDs/hashes, and the grouping rule. Fit normalizers and subspaces only on allowed training data, select checkpoints and ranks using validation data, and reserve the test set for final reporting. Declare whether any target topologies or benchmark cases occur in foundation-model pretraining; distinguish topology overlap from operating-point leakage.

## C26 | Critical — Explain how AC-OPF foundation models become fixed-setpoint PF surrogates

Manuscript page 3. Evidence: Task compatibility is unresolved.

Concern: The cited GridSFM model predicts AC-OPF states and dispatch, while this study defines a PF map with nodal generation/setpoints prescribed in u. An OPF prediction evaluated against unrelated fixed generation can show a large mismatch because the tasks differ. The manuscript does not specify whether it uses released pretrained weights, retrained architectures, fine-tuning, or an adapted input/output interface. This is a comparability concern, not proof that the experiments are incorrect.

Suggested improvement: For each model, list architecture, checkpoint/version, pretraining task, trainable parameters, PF adaptation, available inputs, predicted variables, and postprocessing. Confirm that residual evaluation uses setpoints consistent with the declared PF task. Name adapted architectures explicitly. Include a native-task sanity check for imported checkpoints and a matched PF training protocol before drawing conclusions about foundation-model physics.

Sources: [S03](https://www.microsoft.com/en-us/research/publication/gridsfm-a-foundation-model-for-ac-optimal-power-flow/), [S05](https://arxiv.org/html/2603.04300v1)

## C27 | Major — PIGNN-GC is a new experimental variant that needs a definition

Manuscript page 3. Evidence: Unspecified implementation.

Concern: The global-context extension is not defined, and it can materially change error correlation across buses. The cited PIGNN-Attn-LS includes iterative updates and a line-search correction. It is unclear whether those mechanisms remain active, how many steps are used, and what "Raw" means in this table. The study could inadvertently attribute changes due to global context or disabled correction to a general property of PIGNNs.

Suggested improvement: Specify the global aggregation/broadcast or attention operation, parameter count, layers, hidden size, iteration count, stopping rule, and correction settings. Define the output stage labeled Raw. Add an Attn-LS versus GC ablation if conclusions concern architecture, and show residual versus iteration budget. Verify node permutation behavior and treatment of disconnected components.

Sources: [S07](https://arxiv.org/html/2509.22458v2)

## C28 | Major — The claim about physics-aware training needs the actual objectives

Manuscript page 3. Evidence: Missing training specification.

Concern: The experimental setup gives no loss weights, supervised versus residual-only training regimes, optimizer, training budget, checkpoint rule, or convergence diagnostics. A foundation framework can contain multiple backbones and objectives. Very large residuals might reflect weak physics weighting, omitted constraints, unstable normalization, insufficient fitting, or adaptation failure; the text does not allow these alternatives to be assessed.

Suggested improvement: Add a compact configuration table with objective, masks, normalization, weights, optimizer, steps/epochs, precision, early stopping, seeds, and checkpoint selection. Show physics residuals on train/validation/test using the same final evaluator. Distinguish a frozen output correction experiment from a fair comparative model benchmark. Include equivalent optimization and hyperparameter search budgets where comparisons are made.

## C29 | Major — Make rank selection auditable and include simple limiting cases

Manuscript page 3. Evidence: Ambiguous selection protocol.

Concern: It is unclear which model/grid/seed selects k = 16 and how the no-RMSE-degradation constraint is implemented. The candidate set omits k = 0 (the mean predictor) and an identity/no-projection endpoint. Reusing one rank across four models and 31 grids may be a useful design choice, but it must be stated rather than implying every case independently selected the same optimum.

Suggested improvement: Define the exact validation objective, the magnitude/angle RMSE constraint and tolerance, tie-breaking, per-block rank, and whether selection is global or per model/grid. Publish validation rank curves and held-out test results at the frozen choice, including the mean and no-projection controls. Fit PCA without validation/test data; validation-label use for rank selection is legitimate when declared.

## C30 | Major — Define the pooled correlation and slope precisely

Manuscript page 3. Evidence: Metric definition incomplete.

Concern: Equation (5) does not define the probability measure over i and s, whether correlations are computed per bus then averaged or after flattening, whether buses are variance-normalized, or which split supplies the means. These choices produce different metrics. Pooling centered entries gives larger-variance buses more influence, so a high score can hide failure on less variable or electrically critical buses.

Suggested improvement: Write the finite-sample sums and state the weights, included buses, zero-variance policy, and centering set. Evaluation-only centering using the test cohort is acceptable if it never feeds predictions or rank selection. Report per-bus score distributions as well as the pooled score and include an angle-tracking metric. Distinguish the training mean baseline from the means used solely to compute test metrics.

## C31 | Moderate — Squared correlation is not predictive R-squared

Manuscript page 3. Evidence: Metric interpretation.

Concern: The symbol Rw² denotes squared Pearson correlation, not 1−SSE/SST. A perfectly reversed predictor has corr² = 1, and an arbitrarily shrunk positive predictor can also have corr² = 1. The accompanying slope a_w partly addresses this, but the text should explain that it is a regression gain, not a variance fraction. The constant-predictor score is assigned by convention because its correlation is mathematically undefined.

Suggested improvement: Call the metric squared within-bus correlation, optionally report signed correlation, and add centered normalized MSE or predictive R². Define a_w as the least-squares slope of predicted variation on true variation. Keep the zero-variance convention explicit and interpret both metrics jointly. Appendix D derives their relationship to centered prediction error.

## C32 | Major — Fixed magnitude-error norm does not fix complex-voltage error

Manuscript page 3. Evidence: Restricted intervention.

Concern: Holding the magnitude-error norm and angle prediction fixed isolates a change in magnitude-error orientation conditional on those angles. It does not hold ||V_complex_hat−V_complex*|| fixed: at each bus that error contains 2 V* (V*+eV)(1−cos eθ), which varies with the redistributed magnitude error. The residual also includes Jθ eθ and its interference with JV eV. Thus the intervention is not a clean joint-state or intrinsic-Jacobian-gain experiment.

Suggested improvement: Qualify the claim accordingly. Repeat with true/reference angles, an angle-only intervention, and a joint intervention in a declared scaled state norm. Report the fixed-angle residual floor and the cross-term between magnitude and angle residual contributions. Appendix C gives the exact complex-error identity and a small checked counterexample.

## C33 | Major — Add matched subspace and shrinkage controls

Manuscript page 3. Evidence: Control does not isolate PCA-specific physics.

Concern: The control exchanges concentration between a k-dimensional space and an approximately N−k-dimensional complement, using a single data-derived decomposition. It shows different responses along these selected paths, but cannot identify whether PCA is better than generic denoising, smoothness filtering, or a fortunate alignment with angle errors. Rescaling also amplifies the retained component rather than simply deleting the other component.

Suggested improvement: Compare random orthogonal bases at matched rank, graph-Laplacian/electrical smoothness bases, and validation-tuned shrinkage. Report component energies and gain after normalization. Sample several randomized rotations on the equal-norm sphere and report paired scenario-level PB differences. Label the panels "attenuation followed by norm restoration" so the intervention is not confused with deployable CSP.

## C34 | Moderate — Equation (6) has zero-denominator and state-validity cases

Manuscript page 3. Evidence: Undefined edge cases.

Concern: At α = 0 the first expression is undefined if the in-subspace error is zero; the control is undefined if the off-subspace error is zero. A perfect prediction is another degenerate case. Norm restoration may also produce nonpositive magnitudes or change prescribed PV/slack magnitudes. Clipping would change the norm and invalidate the claimed construction.

Suggested improvement: Specify per-scenario normalization and deterministic handling of zero/near-zero component norms, with counts of affected scenarios. Restrict perturbations to free coordinates, verify positive magnitudes, and report any domain violations. Do not add an epsilon or clip silently while still claiming exact norm preservation. Make α ∈ [0,1] explicit and report the actual retained fraction after normalization.

## C35 | Major — Test the Taylor approximation in the regimes actually evaluated

Manuscript page 3. Evidence: Mechanistic approximation unvalidated.

Concern: An exact residual experiment can show directional dependence without validating the first-order explanation. Several models have multi-degree angle errors even after calibration, and extremely large PB values suggest potentially high curvature or large electrical gains. The remainder in Eq. (3) has no stated bound, and the fixed-norm intervention may move the prediction further from a local linear regime.

Suggested improvement: Compare exact residual vectors with J e, using the correctly masked Jacobian, and report the relative Taylor remainder versus error size and loading. Test t e for t approaching zero to separate local sensitivity from nonlinear effects. Compute Jacobian-vector products in consistent units and include the magnitude–angle cross-term. Present nonlinear effects as an alternative explanation if the linear approximation is poor.

## C36 | Major — Figure 2 supports a heterogeneous, not universal, mechanism

Manuscript page 3. Evidence: Visible limitation of the main result.

Concern: At α = 0, the reported relative PB values are 0.586 and 0.476 for PIGNN-GC and GridSFM, but 0.954 and 0.997 for GraphKit and LUMINA. The latter are only 4.6% and 0.3% reductions. The upward control curve is evidence of anisotropy along a different path; it does not establish that existing off-subspace magnitude error is the main source of residual in those models. There are no uncertainty intervals to assess the tiny effect.

Suggested improvement: State that the strongest magnitude-direction evidence concerns PIGNN-GC and GridSFM. Give per-seed/scenario uncertainty, absolute PB at α = 1, and magnitude-versus-angle attribution. Revise the abstract/conclusion if the mechanism does not generalize. Avoid interpreting a visually flat curve as support equivalent to a 40–50% reduction.

## C37 | Major — Separate magnitude projection, angle projection, and norm reduction

Manuscript page 3. Evidence: Ablation incomplete for the proposed mechanism.

Concern: The C/P16/CSP16 factorial is useful and should be retained. However, Fig. 2 intervenes only on magnitude-error orientation with oracle references, while Table 1 projects both predicted magnitudes and angles and changes their norms. Consequently the large CSP gain, especially for LUMINA, cannot be attributed to the fixed-norm magnitude mechanism alone.

Suggested improvement: Extend the output ablation to C+PV-only projection, C+angle-only projection, and C+both, with identical boundary handling. Add soft shrinkage and projected-reference controls. Report PB changes relative to C as well as Raw, because Raw-to-CSP includes calibration. Explicitly state that the reference-dependent directional intervention is diagnostic and not an inference algorithm.

## C38 | Major — Measure the cost and compare with direct numerical correction

Manuscript page 3. Evidence: Practical effectiveness untested.

Concern: The proposed method is cheap in principle, but there are no timings, memory measurements, reference-solve costs, or comparisons with one or a few Newton/line-search steps. Lower PB alone does not establish a useful surrogate if it remains above tolerance and requires expensive cleanup. A mean or linearized predictor followed by a solver may be competitive.

Suggested improvement: Report raw/C/CSP inference latency and throughput at batch size one and realistic batches, plus low-rank fitting/storage cost. Apply P as W(W^T z), avoiding a dense 2N×2N matrix. Compare raw→solver, CSP→solver, mean→solver, and flat-start solver at the same tolerance, hardware, and failure policy. Include total time, iterations, and convergence rate; distinguish offline amortization from online speed.

## C39 | Major — Define the target-grid label budget and report more than PB

Manuscript page 3. Evidence: Cross-grid protocol and outcomes incomplete.

Concern: Figure 3 gives raw and CSP mean PB only. It cannot establish preserved accuracy or scenario tracking on every grid, and it does not distinguish zero-shot transfer from fitting a target-grid basis and bias using solved target data. For unrelated bus dimensions, a new basis is generally required. The details of model training/fine-tuning and rank choice per grid are absent.

Suggested improvement: Provide per-grid network ID, train/calibration/validation/test counts, checkpoint, adaptation regime, effective rank, baseMVA, and raw/C/CSP magnitude, angle, tracking, and PB metrics. State any pretraining overlap. Use "evaluation across grids with target-grid calibration" where appropriate and report paired uncertainty or per-grid effect distributions.

## C40 | Major — Rank 16 cannot mean the same reduction on 4-bus and 9,241-bus grids

Manuscript page 3. Evidence: Dimensional inconsistency to resolve.

Concern: Separate magnitude/angle state spaces on a 4-bus grid have dimension at most four, less after removing fixed coordinates. A literal rank-16 basis is impossible there. With k clipped to the full free dimension, projection becomes identity, and CSP can reduce to calibration. Alternatively, numerical-rank truncation can still remove directions, but then the effective rank is data-dependent. Small-grid gains therefore cannot automatically be evidence for low-rank projection.

Suggested improvement: Specify k_eff = min(k, admissible dimension, chosen numerical rank) separately for each block and list it per grid. Plot C alongside CSP. Identify grids where projection is identity or where only gauge/fixed-coordinate removal remains, and attribute their gains correctly. Avoid a blanket CSP16 label without explaining the clipping policy.

## C41 | Major — Treat the mean baseline as a demanding scientific control

Manuscript page 4. Evidence: Visible in Table 1.

Concern: The static mean has lower magnitude RMSE and lower mean PB than every raw neural model in Table 1. That does not make the mean a useful surrogate, since it has zero scenario response, but it shows how strongly these aggregate metrics reward a central operating profile. The corrected LUMINA magnitude RMSE of 0.02240 remains close to the mean baseline 0.02341.

Suggested improvement: Discuss this directly when motivating the analysis. Add a validation-tuned interpolation between the calibrated prediction and training mean, a linear/ridge predictor of principal coefficients, and a local linearized PF baseline. Compare tracking and tail residuals as well as mean errors. Verify that the mean baseline respects each scenario’s known boundary values under the same convention as all model outputs.

## C42 | Major — High correlation after CSP does not restore response amplitude

Manuscript page 4. Evidence: Visible under-response after correction.

Concern: PIGNN-GC improves squared correlation from 0.671 to 0.829 but its slope stays near 0.54. GridSFM reaches 0.936 with slope 0.724, and GraphKit/LUMINA remain at approximately 0.147/0.045. Projection can remove prediction variance uncorrelated with the target and thereby increase correlation without restoring the physically relevant response to changing inputs. The paper acknowledges tracking but underplays this remaining amplitude loss.

Suggested improvement: Interpret correlation and slope together and report centered normalized MSE/predictive R². Add validation-fit scalar or mode-wise gain calibration as a simple baseline, with regularization to avoid amplifying unstable modes. Do not call increased correlation alone recovery of scenario-dependent dynamics. Appendix D quantifies the centered-error implication of the reported mean metrics, with a seed-aggregation caveat.

## C43 | Moderate — CSP worsens angle RMSE relative to calibration for two leading models

Manuscript page 4. Evidence: Visible accuracy tradeoff.

Concern: For PIGNN-GC, angle RMSE rises from 0.777 degrees under C to 0.835 under CSP; for GridSFM it rises from 1.781 to 1.807. The statement about improvement in both voltage accuracy and balance is true for magnitude and relative to Raw, but not for every voltage coordinate or every output comparison. Rank selection also says "voltage RMSE" without specifying which quantity.

Suggested improvement: Write "improves voltage-magnitude RMSE and reduces mean PB relative to Raw; angle accuracy can trade off against residual reduction relative to C". State whether the validation constraint applies only to magnitude. Consider separate kV and kθ or a multi-metric validation rule, reporting the full tradeoff rather than bolding only selected gains.

## C44 | Major — Audit the orders-of-magnitude residual outliers

Manuscript page 4. Evidence: Extreme values require evaluator audit.

Concern: GraphKit and LUMINA have raw maximum PB of about 4,651 and 24,179 p.u.; GraphKit’s maximum increases after calibration despite its mean improving. These may be real effects of small impedances or localized errors, but their scale also warrants checking radians/degrees, power bases, phase-shifting taps, inactive branches, shunt duplication, masking, and postprocessed outputs. The paper provides no diagnostic that separates these possibilities.

Suggested improvement: Recompute reference and prediction injections with an independent solver/evaluator. Identify the buses/scenarios responsible for extremes and report voltage errors, incident admittances, bus type, and physical-unit mismatch there. Include robust quantiles and per-scenario maxima. Treat implementation error as a possibility to test, not an accusation; keep extremes in the results with explanations rather than silently clipping them.

## C45 | Major — Make the three-seed uncertainty and aggregation interpretable

Manuscript page 4. Evidence: Uncertainty and aggregation incomplete.

Concern: The meaning of a seed is unspecified: initialization, minibatch order, data split, calibration subset, or all of these. Mean ± standard deviation over three runs is not a confidence interval, and pooled bus–scenario observations are strongly dependent. Maxima depend on sample count and do not identify a stable upper-tail behavior. Figure 2, Figure 3, and Table 2 give no comparable uncertainty.

Suggested improvement: List individual run results and state what changes across seeds. Use paired comparisons on the same scenarios; bootstrap or cluster at the scenario/topology/profile level rather than treating buses as independent. For multiple training runs, separate training variability from test-sampling variability. Report PB quantiles of per-scenario maxima and sample counts; interpret tiny Fig. 2 effects cautiously.

## C46 | Critical — CSP-only shift results cannot identify whether CSP helps under shift

Manuscript page 4. Evidence: Distribution-shift attribution is unresolved.

Concern: Table 2 reports only corrected outputs under each condition. Without Raw and C on those same shifted scenarios, the increased outage maxima or GBcorr errors could arise from the surrogate, from projection truncation, from calibration shift, or from a combination. Comparing CSP across input distributions does not estimate the effect of CSP within a shifted distribution.

Suggested improvement: For each condition show Raw, C, P, and CSP on identical test scenarios, plus projection of reference solutions. Report paired ΔPB and ΔRMSE, fraction worsened, and in/out error and truncation energies. Include V/angle-only projections if space permits. This is necessary to claim that transferred solution geometry helps or harms under topology or spatial shift.

## C47 | Major — Specify outage sampling and topology-dependent solver conventions

Manuscript page 4. Evidence: Unspecified contingency protocol.

Concern: The paper omits the number and selection of N−1/N−2 contingencies, whether outages are uniform or targeted, whether islands are excluded, and whether operating injections are paired with control cases. Connectivity changes can alter angle references, active bus sets, solvability, and generator limit enforcement. A nominally unchanged basis is not well-defined if the coordinate system changes.

Suggested improvement: Publish outage lists or their generation rule, scenario counts, islanding and rejection rates, contingency-specific Y-bus, active bus mapping, slack/reference handling, and Q-limit policy. Pair operating conditions with control wherever possible. Split by outage topology when claiming unseen-topology generalization, and report separate connected, islanded, converged, and rejected outcomes.

## C48 | Major — Pooled p99 can hide a bad bus in every scenario

Manuscript page 4. Evidence: Tail interpretation exceeds pooled statistics.

Concern: A stable pooled bus-wise p99 does not show that only a few scenarios fail. One bad bus in every 2,224-bus scenario affects about 0.045% of pooled entries, so widespread scenario failure can be invisible even to p99. The reported maximum increases are about 9.26× for N−1 and 7.67× for N−2 relative to control, but the affected scenario fraction is unknown.

Suggested improvement: Report q_s = max_i PB_is, its median/p95/p99, and Pr(q_s > tolerance), along with counts of affected scenarios, buses, and outage topologies. Inspect the worst cases and stratify by electrical distance from the outage. Rephrase as "the pooled uppermost tail increases" unless scenario-level localization is established.

## C49 | Major — Define the GBcorr spatial perturbation mathematically

Manuscript page 4. Evidence: Distribution shift not reproducibly defined.

Concern: GB-derived spatial covariance and connected zones are not specified mathematically. Preserving total active load still changes transfer patterns, losses, reactive demands, slack generation, and voltage controls. Hence "spatial shift" is not a single controlled factor unless the other quantities and rejection policy are documented. The reader also cannot assess whether the shifted train/test sets share base profiles.

Suggested improvement: Give zone construction, covariance provenance/estimation, perturbation amplitudes, the total-load-preserving transform, positivity constraints, P–Q coupling, generator response, and acceptance criteria. State whether active and reactive totals are each fixed. Publish shifted train/validation/test counts and paired control profiles, and report how far each condition lies outside training support.

## C50 | Major — Separate recalibration from subspace refitting under GBcorr

Manuscript page 4. Evidence: Confounded adaptation experiment.

Concern: The GBcorr refit changes both calibration and the PCA representation (and apparently its mean), while the prose attributes the improvement to basis adaptation. That attribution is not identified by a joint refit. The refit also uses shifted reference solutions, an adaptation resource unavailable in a strict zero-shot setting. Its required label count and fitting cost are not given.

Suggested improvement: Run a factorial with frozen/refitted bias, mean, and basis, or clearly group mean+basis as the affine representation and isolate it from bias. Keep model weights fixed and use only disjoint shifted training data. Show PB and tracking versus adaptation sample count and the cost of obtaining those labels. Describe the joint result as recalibration plus subspace adaptation, not basis-only recovery.

## C51 | Moderate — Do not prescribe neural fine-tuning without testing simpler corrections

Manuscript page 4. Evidence: Speculative remedy.

Concern: The residual drop after refitting does not establish that remaining tracking loss requires changing the neural weights. Persistent slopes below one could be addressed partly by input-conditioned calibration, mode-wise gain correction, different ranks, or better affine representation; the paper has not compared these. The wording "may require" is tentative but still privileges an untested explanation.

Suggested improvement: Write "remaining tracking loss motivates comparing parameter fine-tuning with richer output calibration and subspace adaptation". Test a simple validation-fit gain correction before an expensive retraining claim. Any fine-tuning comparison should use the same shifted-label budget as the output adaptation.

## C52 | Major — Narrow the conclusion to the demonstrated empirical result

Manuscript page 4. Evidence: Conclusion exceeds demonstrated scope.

Concern: The conclusion moves from selected intervention paths to a general source of inconsistency across physics-informed and foundation approaches, and then toward estimation and optimization. Only PF residuals are evaluated; OPF objective quality, inequalities, state-estimation measurement noise, and operational decisions are not tested. Angle tradeoffs and poor tracking in two models are also omitted.

Suggested improvement: Conclude that train-derived affine projection lowers observed residuals on these benchmarks and that magnitude-error orientation matters most clearly for two evaluated models. Include distribution dependence, reference-data requirements, lack of feasibility guarantees, and unresolved tail behavior. Present OPF/state-estimation extension as future work with separate task-specific validation. A suggested conclusion appears in Appendix F.

## C53 | Minor — Repair the table cross-reference, captions, and units

Manuscript page 4. Evidence: Confirmed editorial defects.

Concern: Section 3.2 says "Table 3" although the results appear in Table 1. Table 2 contains "seperate" and "quntiles"; its model, checkpoint/seed, counts, and units are absent from the caption. The T/R footnote mentions the basis but the text also changes calibration. Table headers use inconsistent RMSE notation and omit p.u. units for magnitude and PB.

Suggested improvement: Change Table 3 to Table 1; use "separate" and "quantiles". Define T and R as complete adaptation protocols. State PIGNN-GC, the fixed checkpoint, cohort size, aggregation, and p.u. bases. Use consistent |V| RMSE (p.u.), angle RMSE (degrees), and PB (p.u.) headings; explain boldface and add the missing uncertainty/sample-count information.

## C54 | Minor — Make cross-grid results identifiable and readable

Manuscript page 4. Evidence: Presentation.

Concern: Figure 3 labels grids only by bus count, including duplicate counts, so readers cannot identify the actual cases. The x-axis spacing is categorical but can be read as a quantitative size axis. Labels are tiny at publication scale; the plot shows only a mean and supplies no per-grid tracking or uncertainty.

Suggested improvement: Use abbreviated case IDs and bus count, distinguish categorical ordering from logarithmic size scaling, and provide exact values in a machine-readable supplement. Include calibration-only results and paired error bars or intervals. A compact per-grid improvement plot with a separate detailed table would communicate the main result more clearly.

## C55 | Moderate — Replace the pending GraphKit citation and identify the actual model

Manuscript page 5. Evidence: Verified citation update.

Concern: Reference [18] says the official citation is pending. The official GraphKit documentation currently supplies a GENCO citation, already represented by [19]. More importantly, GraphKit is a training/evaluation framework, not an unambiguous model specification. The tested model must be identified without silently equating every GraphKit configuration with GENCO.

Suggested improvement: Update [18] with the actual software version/commit, URL, and access date; cite GENCO as instructed by the official documentation where applicable. Name the architecture, configuration, task head, and checkpoint used in Table 1. If an earlier/custom GraphKit model was used, state that explicitly. Keep software provenance distinct from an algorithm citation.

Sources: [S06](https://gridfm.github.io/gridfm-graphkit/#citation)

## C56 | Moderate — Strengthen directly relevant comparisons and verify reference metadata

Manuscript page 5. Evidence: Literature and reproducibility improvement.

Concern: The paper already cites PCA-based compact learning, which is good, but lacks a direct baseline and a standardized PF distribution-shift benchmark. PFΔ provides a particularly relevant independent test with load, generation, topology, and difficult operating cases. The references also contain inconsistent acronym capitalization and sparse software identifiers. A future print year is not inherently erroneous: reference [10] has a publisher record for volume 262 (2027).

Suggested improvement: Prioritize compact-learning/output-denoising and solver-correction comparisons over adding generic citations. Consider an independent PFΔ experiment and pin a current implementation revision. Standardize Newton–Raphson, AC-OPF, CANOS, FSNet, and software names; add persistent identifiers where available. Verify exact versions and published metadata without treating 2026/2027 entries as fabricated merely because of their dates.

Sources: [S04](https://arxiv.org/abs/2301.08840), [S08](https://openreview.net/pdf?id=Gi1HtsTAkv), [S09](https://www.sciencedirect.com/science/article/pii/S037877962600920X)

# Appendix A — Correct projection mathematics

## A1. Define the algorithm actually implemented

For separate magnitude and angle bases, form centered training matrices X_V and X_theta independently. Let W_V and W_theta contain their retained left singular vectors. Then P = diag(W_V W_V^T, W_theta W_theta^T), with rank(P) = k_V + k_theta. If both retained ranks are 16, the concatenated projector has rank 32. A joint SVD of the concatenated state generally gives a different, non-block-diagonal projector. Specify dimensions after eliminating prescribed coordinates and any gauge degree of freedom.

Ordinary Euclidean SVD and the identities below assume a fixed coordinate chart, a fixed orthogonal projector, and consistent scaling. For angles this requires valid local unwrapping. If clipping, nonlinear angle wrapping, scenario-dependent clamping, or weighted non-Euclidean projection is part of the deployed map, analyze that actual map instead of asserting these identities unconditionally.

## A2. The exact error contains a truncation term

Let μ be the training-reference mean, e_C = x_hat − b_hat − x*, and q = (I−P)(x*−μ). Substituting x_hat − b_hat = x* + e_C into Eq. (4) gives:

`x_CSP − x* = P e_C − (I−P)(x*−μ) = P e_C − q.`

Because P is orthogonal, P e_C is orthogonal to q. Therefore:

`||e_CSP||² = ||P e_C||² + ||q||².`

`||e_C||² − ||e_CSP||² = ||(I−P)e_C||² − ||q||².`

This is an exact finite-sample statement, not a Taylor approximation. CSP improves squared error precisely when removed calibrated-error energy exceeds discarded true-solution variation. Averaging this identity gives the corresponding expected-error condition. It explains why a useful control-distribution basis can become harmful under shift without invoking a change in the neural network.

A minimal check: let μ = (0,0), P = diag(1,0), and x* = x_hat = (0,1), with zero bias. A perfect prediction has zero initial error, yet its projection is (0,0) and has error norm one. This is a mathematical counterexample to unconditional accuracy improvement, not a claim that the paper promised such a theorem.

For the same reason, projecting an oracle error around x* in Fig. 2 and projecting a prediction around μ in Eq. (4) are not interchangeable. Report q for held-out references and the exact residual of μ + P(x*−μ) under that reference's own input u. That experiment reveals the projection's attainable approximation floor before model error enters.

## A3. Calibration has a simple affine interpretation

In Euclidean coordinates b_hat = mean_train(x_hat) − μ, hence:

`x_CSP = μ + P [x_hat − mean_train(x_hat)].`

Thus CSP matches training means and filters centered prediction variation. It does not restore under-responsive coefficients along retained directions. Calibration and projection are algebraically related: CSP = projection_only(x_hat) − P b_hat. Projection-only already removes bias components perpendicular to the retained subspace; the additional correction is the retained component of bias. This interpretation can replace the vague statement that the two mechanisms are wholly separate. Circular-angle calibration requires the chart qualification above.

## A4. Preserve known boundary values

One defensible implementation predicts only free PF coordinates and reconstructs prescribed voltage magnitudes/angles from u. If a full-state affine PCA representation is retained, impose its boundary equations while choosing the coefficients. For B = retained basis, target coefficient a0 = B^T(x_C−μ), and boundary requirement A(u)x = c(u), solve min_a ||a−a0||² subject to A(u)B a = c(u)−A(u)μ. When feasible, a = a0 + [A(u)B]^dagger [c(u)−A(u)μ−A(u)B a0]. If the boundary target lies outside the column space, the constraint cannot be satisfied exactly in that affine model; use free-coordinate prediction or augment the representation. This is a proposed extension, not the method evaluated in the manuscript.

# Appendix B — What the Jacobian does and does not establish

## B1. Direction matters locally, but PCA alignment is an additional hypothesis

Use a reduced, consistently scaled PF residual F(z,u), with prescribed coordinates eliminated and a fixed bus-type regime. At a regular solution F(z*,u) = 0, let J = dF/dz and H = dF/du. Differentiating along the solution map gives:

`J (dz*/du) + H = 0;  dz*/du = −J^−1 H.`

For small input perturbations with covariance C_u:

`C_z ≈ J^−1 H C_u H^T J^−T.`

PCA diagonalizes this covariance, not J^T J. Input excitation therefore matters. Only under additional structure, for example H C_u H^T proportional to identity in the residual coordinates, do the largest solution-variance directions correspond to the smallest right-singular gains of J. This is a conditional intuition, not a general property of AC-PF. Active-set changes and voltage-collapse points can also invalidate a single regular local model.

A constructed linear example makes the distinction explicit. Let F(z,u) = Jz−u, J = diag(1,10), and C_u = diag(1,10000). The solution covariance is diag(1,100): rank-one PCA retains the second coordinate even though its residual gain is 10, versus 1 in the discarded coordinate. Thus large solution variance can align with high, rather than low, residual sensitivity.

## B2. Residual components need not be orthogonal

For a fixed input and negligible truncation, write a = JPe and b = J(I−P)e. In squared L2 residual:

`||Je||² = ||a||² + ||b||² + 2 a^T b.`

Orthogonality before applying J does not remove the cross-term after applying J. For J = [[1,1],[0,epsilon]], e = (1,−1), and P = diag(1,0), the original residual norm is epsilon but the projected residual norm is one. J is nonsingular when epsilon > 0. This example demonstrates why a Euclidean projection alone gives no residual-descent theorem. It does not assert that this matrix is a GBnetwork Jacobian.

With nonzero q, the relevant post-CSP local residual is J(Pe_C−q), not JPe_C. A useful bound is ||r_CSP|| ≤ ||JP|| ||e_C|| + ||Jq|| + a local remainder. A small geometric truncation q can still have a large electrical effect if Jq is large.

## B3. Match the derivation to the reported metric

The paper's mean PB is an average of two-component bus residual norms, effectively a grouped L1-of-L2 quantity after masks. It is not the squared L2 norm of the full residual vector. Jacobian spectral calculations can diagnose sensitivity, but a proof about one norm does not establish equal improvement in another. Evaluate the exact reported PB, per-scenario maximum PB, and full residual L2 side by side.

Separate two different sensitivity phenomena: a large largest singular value amplifies a state error into residual, whereas a small smallest singular value amplifies an injection/residual perturbation into state error through J^−1. A large condition number alone does not identify which phenomenon is occurring. Report appropriate scaled gains, not simply a generic claim of ill-conditioning.

## B4. Minimum mechanistic measurements

For each held-out scenario, compute total and retained/discarded state-error energies, reference truncation energy, ||JPe||/||Pe|| and ||J(I−P)e||/||(I−P)e|| where defined, the residual cross-term, and the exact Taylor remainder. Analyze magnitude and angle separately and jointly. Compare rank-matched random bases, a graph/electrical basis, and PCA; include independent or spatially richer excitation. Report paired distributions with scenario-level uncertainty. These measurements would convert the appealing geometric narrative into a falsifiable mechanism.

## B5. An exact AC identity can strengthen the analysis

For fixed Y, let v* be the reference complex voltage, d the complex-voltage perturbation, and S(v) = diag(v) conj(Yv). Direct expansion gives an exact identity:

`S(v*+d) − S(v*) = diag(d) conj(Yv*)`

`+ diag(v*) conj(Yd) + diag(d) conj(Yd).`

The first two terms are linear over real coordinates; the final term is quadratic. With the paper's residual sign, the injection mismatch changes by the negative of this expression. This identity includes the phase-shifting/shunt model through Y and does not require Y to be symmetric. It assumes a fixed network and prescribed injections; voltage-dependent loads require their own additional terms.

The quadratic term satisfies ||diag(d) conj(Yd)||_2 ≤ ||Y||_2 ||d||_2². This is a loose bound, but it exposes why admittance scaling matters and gives a direct way to quantify the nonlinear remainder. A bus-wise bound uses |d_i| sum_j |Y_ij| |d_j|. These formulas could support a more concrete analysis than the unspecified O(||e||²) in polar coordinates. They do not imply that discarded PCA directions necessarily have greater gain; that remains an empirical question.

# Appendix C — Interpret the fixed-norm intervention correctly

## C1. Which norm is actually constant?

Eq. (6) holds ||e_V|| constant. Since e_theta is unchanged, a fixed block-scaled Euclidean polar-state norm also stays constant. However, that does not imply a fixed complex-voltage norm. At bus i, with reference magnitude V_i* and magnitude error δ_i:

`|(V_i*+δ_i) exp(j theta_hat_i) − V_i* exp(j theta_i*)|²`

`= δ_i² + 2 V_i* (V_i*+δ_i) [1−cos(theta_hat_i−theta_i*)].`

Redistributing δ across buses changes the second term even when sum_i δ_i² is fixed. For V* = (1,1), fixed angle error (0.3,0) radians, and magnitude errors (0.02,0) versus (0,0.02), the magnitude-error norms agree, but squared complex-voltage errors differ by 0.04[1−cos(0.3)]. This follows exactly from the phasor definition.

## C2. Fixed angles still interact with magnitude error

Locally r ≈ J_V e_V + J_theta e_theta. Varying e_V changes its interference with the unchanged angle contribution. It is therefore fair to say that the experiment measures the effect of magnitude-error orientation conditional on the calibrated angles. It is too strong to say it isolates an intrinsic gain property of the magnitude subspace or explains the full CSP effect in both channels.

Use four diagnostic variants: magnitude intervention with calibrated angles; magnitude intervention with reference angles; angle intervention with reference magnitudes; joint intervention in a declared scaled polar norm. Each variant is oracle-based and belongs in diagnosis, not deployment. Report the residual with reference magnitude and calibrated angles to reveal the angle-conditioned floor.

## C3. Attenuation is followed by amplification

The parameter alpha weights a component before renormalization. Its final off-subspace energy fraction is alpha² ||e_perp||² / (||e_parallel||² + alpha² ||e_perp||²). Thus the plot label "retained fraction alpha" is not literally the fraction of original error energy or amplitude after normalization. At alpha = 0 a nonzero retained component is expanded to the full original norm. State this explicitly and guard zero denominators, nonpositive magnitudes, and prescribed coordinates.

The reported relative PB values at alpha = 0 correspond to reductions of 41.4%, 52.4%, 4.6%, and 0.3% for the four listed models. Only the first two support a large effect for this particular intervention. Show uncertainty before interpreting the smaller changes as reliable, and avoid pooling these heterogeneous effects into a universal causal claim.

# Appendix D — Tracking, aggregate metrics, and numerical reading of Table 1

## D1. A precise pooled definition

Let T_is and H_is be true and predicted magnitudes after subtracting each bus's mean across the evaluation scenarios. With explicitly chosen weights w_is summing to one, define A = sum w T², B = sum w H², and C = sum w T H. Then rho = C/sqrt(AB), a = C/A, and the reported Rw² = rho². Declare behavior when A or B is zero. Different per-bus standardization or averaging rules yield different quantities.

For A > 0 and rho ≠ 0:

`centered NMSE = sum w(H−T)² / A = 1 − 2a + a²/rho².`

`predictive centered R² = 1 − centered NMSE.`

This identity makes clear why squared correlation alone can look excellent while amplitudes remain wrong. Using the printed mean a and Rw² values as illustrative plug-ins gives CSP centered NMSE approximately 0.271 (PIGNN-GC), 0.112 (GridSFM), 0.743 (GraphKit), and 0.915 (LUMINA). These are not recovered experimental metrics: the nonlinear transformation of cross-seed means is not the mean of per-seed transformed values, and the pooling convention is unspecified. Compute and report the actual quantity from predictions.

## D2. Report the correct comparison

From Table 1 means, raw-to-CSP mean PB drops by approximately 82.6% for PIGNN-GC and 80.9% for GridSFM. Relative to calibration alone, the additional drops are approximately 67.0% and 37.8%. These are ratios of printed means, not averages of per-scenario or per-seed relative improvements. State that distinction when quoting percentages.

PIGNN-GC's calibrated-to-CSP angle RMSE rises from 0.777 to 0.835 degrees; GridSFM's rises from 1.781 to 1.807. CSP improves their magnitude RMSE and PB while slightly worsening this angle metric. The static mean has lower raw magnitude RMSE and raw mean PB than all four neural models, reinforcing the need for dynamic tracking and gain metrics.

## D3. Pooled tails do not measure scenario reliability

With 2,224 buses, one failing bus per scenario represents only 1/2224 ≈ 0.045% of pooled bus–scenario entries. A pooled p99 can therefore remain almost unchanged even if every scenario has a serious local violation. The appropriate companion is the distribution of q_s = max_i PB_is and Pr(q_s > epsilon). Also give contingency-level failure rates, since many scenarios can share the same vulnerable topology.

Do not convert the paper's PB into an operational threshold without its actual power base. If S_base were 100 MVA, 0.055 p.u. would correspond to 5.5 MVA of mismatch magnitude; this is only a conditional unit-conversion example, not the identified base of GBnetwork or a universal acceptability threshold.

# Appendix E — Prioritized revision and experiment plan

## E1. Acceptance-critical checks

1. Specify and independently validate the masked residual, net injection convention, bus types, power base, transformer/shunt handling, gauge, and known boundary values. Reconstructed reference residuals must match the declared solver tolerance. If this check fails, rerun every dependent result before interpreting geometry.

2. Publish a model/protocol matrix: actual architecture/checkpoint, PF adaptation, native task, training objective, optimizer/budget, physics weights, data splits, inference correction, and calibration label access. Establish that the large foundation-model residuals are not a task/interface mismatch. If they are, repair the comparison and revise the claim.

3. Correct Eq. (1), add the exact truncation identity, and report rank spectra, held-out reference reconstruction error, projected-reference PB, and per-scenario gain/loss from CSP. Evaluate fixed coordinates consistently. This directly tests whether the correction removes error more than valid signal.

4. Add magnitude-only, angle-only, and joint interventions/projections; include random matched-rank and shrinkage controls. Check the Jacobian prediction and residual cross-terms. If PCA-specific alignment is weak, retain a denoising result and narrow the mechanism claim.

5. On each shifted cohort, compare Raw/C/P/CSP and projected references on identical examples. Separate refitted bias from affine-basis refitting; disclose shifted label count. Add per-scenario maximum residual and tolerance-exceedance rates. This identifies whether topology/spatial changes harm the surrogate or the correction.

## E2. High-value robustness checks

6. Report independent seeds and paired uncertainty for the diagnostic curves and shift results, plus the number of scenarios/topologies. Avoid bus-level pseudo-replication. Inspect the specific extreme-residual examples and validate them with a second evaluator.

7. Give per-grid adaptation budgets, effective ranks, accuracy/tracking, and calibration-only baselines across all 31 systems. Small-grid full-rank cases must be identified. Test at least one independent benchmark/generator, with PFΔ a relevant candidate [S08].

8. Compare to a validation-fit mean-shrinkage model, a principal-coefficient linear model, and short numerical correction at equal accuracy/tolerance. Report latency, throughput, memory, label/precomputation cost, and downstream solver success. Add uncertainty and exact figure/table data to the artifact release.

## E3. Optional extensions, not required to substantiate the existing paper

Soft mode shrinkage can trade denoising against truncation loss. Scenario-conditioned or local bases can model multiple operating regimes. A boundary-constrained coefficient projection can preserve specified controls. A residual-based accept/reject wrapper can avoid increasing its chosen residual metric relative to an uncorrected candidate, but does not guarantee feasibility or state accuracy. CSP as a solver warm start may offer more practical value than approximate residual reduction alone. Each extension needs its own validation and should not be described as already demonstrated.

## E4. Four-page narrative priorities

Compress the broad application motivation. Keep a precise PF problem statement and evaluator definition, the corrected projection identity, a compact model/protocol table, the essential diagnostic with uncertainty, the main output ablation, and one interpretable shift result. Move detailed hyperparameters, 31-grid tables, rank curves, and expanded tail diagnostics to an accessible supplement. Essential scientific definitions must remain understandable without opening the repository.

# Appendix F — Suggested replacement writing

These drafts use only numbers and observations already printed in the manuscript. Adopt them only after resolving the evaluator and task-compatibility questions; they do not certify the correctness of those results.

## Suggested title

Voltage-Error Geometry and Calibrated Subspace Correction for Neural AC Power Flow

## Suggested abstract

Neural AC power-flow surrogates can exhibit modest voltage-magnitude error while retaining substantial power-balance mismatch. We study this discrepancy in four evaluated model configurations on synthetic operating scenarios for the 2,224-bus GBnetwork. Using separate principal-component bases fitted to training voltage magnitudes and gauge-aligned angles, we analyze prediction errors relative to dominant training-solution variation. At fixed magnitude-error norm and calibrated angles, suppressing the off-subspace component reduces mean residual by 41.4% for PIGNN-GC and 52.4% for GridSFM, with much smaller effects for the other two configurations. We then apply train-fitted bias correction and affine subspace projection to model outputs. The combined correction reduces mean power-balance mismatch from 0.3159 to 0.0550 p.u. for PIGNN-GC and from 0.2687 to 0.0514 p.u. for GridSFM. Evaluation across 31 grids and under topology and spatial shifts illustrates both the benefit and distribution dependence of the correction. The method requires reference training solutions and does not guarantee AC feasibility. These results motivate evaluating voltage-error direction, response amplitude, and residual tails alongside voltage RMSE.

## Suggested method clarification

CSP projects the calibrated prediction onto an affine approximation of training-solution variation. If P is the orthogonal projector, μ the training-reference mean, and e_C the calibrated prediction error, its error is P e_C − (I−P)(x*−μ). The second term is the reference truncation error. Consequently projection reduces squared state error only when removed calibrated-error energy exceeds discarded reference variation. Its effect on AC residual is evaluated empirically because Euclidean projection need not reduce a nonlinear power-balance norm.

## Suggested Figure 2 interpretation

At fixed magnitude-error norm and fixed calibrated angles, the selected error-direction paths substantially change mean PB for PIGNN-GC and GridSFM. The off-subspace attenuation experiment has much smaller effects for GraphKit and LUMINA. These results establish sensitivity to magnitude-error orientation under the evaluated conditions, but do not by themselves identify the role of angle error or explain the full two-channel CSP correction.

## Suggested conclusion

Training-derived affine subspace correction reduces the observed mean power-balance residuals and voltage-magnitude errors in the evaluated neural AC-PF configurations. The fixed-norm diagnostic shows that magnitude-error orientation matters especially for PIGNN-GC and GridSFM. Projection also removes valid solution variation, and its benefit depends on the operating distribution; angle accuracy, response amplitude, and extreme residuals can remain problematic. The method uses reference training solutions and provides no feasibility guarantee. Future work should test its mechanism with Jacobian and channel-wise diagnostics and evaluate its value as a warm start for numerical correction under distribution shift.

# Source-check and validation notes

The Semantic Scholar skill was used for two batched academic searches: an exact-phrase compact-learning search returned one paper, and a broader PF/topology benchmark query returned eight records. Nine unique records were returned; only the compact-learning record was retained as directly useful from those searches. Unrequested citation-count fields appeared as zero in the helper's formatted output and were not treated as actual zero citations. Broader results included irrelevant records, so the main model, benchmark, and mathematical convention checks used the primary sources listed below.

Direct primary-source checks verified the PF bus conventions, transformer tap formulas, AC-OPF scope of the cited GridSFM/LUMINA sources, the GraphKit citation update, the PIGNN-Attn-LS predecessor's correction mechanism, the compact-learning comparison, and PFΔ. The publisher records support references [10] and [14]; their dates were not treated as evidence of fabrication. No claim is made that all bibliography entries, checkpoint implementations, or experimental numbers were independently verified.

The supplied figures and tables were visually inspected. A separate deterministic script checks the projection/truncation identity, the nonsingular residual-amplification counterexample, the covariance/PCA counterexample, the complex-voltage-norm counterexample, the exact AC injection expansion, and arithmetic derived from printed table values. These checks validate the review's mathematical reasoning and calculations only. They are not reproductions of the paper's power-flow experiments.

The review bundle includes an editable comment dataset and generation script. The annotated PDF contains native comments and a static full-text appendix. Automated checks confirm all 56 comments have located anchors, each navigation link has a valid destination, and all expected review sections are present; rendered pages are inspected for legibility before delivery.

# Primary sources checked

[S01] [MATPOWER manual: AC Power Flow](https://matpower.app/manual/matpower/ACPowerFlow.html). Bus-type conventions, free variables, solved generator quantities, and Q-limit switching.

[S02] [MATPOWER makeYbus source](https://matpower.org/docs/ref/matpower7.1/lib/makeYbus.html). Complex tap conventions, directed branch admittance entries, and shunt construction.

[S03] [Microsoft Research: GridSFM](https://www.microsoft.com/en-us/research/publication/gridsfm-a-foundation-model-for-ac-optimal-power-flow/). The released model is an AC-OPF surrogate with dispatch and voltage outputs.

[S04] [Park et al.: Compact Optimization Learning for AC Optimal Power Flow](https://arxiv.org/abs/2301.08840). Prior PCA output compression, coefficient learning, and subsequent solver-based correction. Journal DOI: 10.1109/TPWRS.2023.3313438.

[S05] [Li et al.: LUMINA, cited March 2026 version](https://arxiv.org/html/2603.04300v1). An AC-OPF framework comparing multiple architectures and training objectives; a model label alone is insufficient.

[S06] [Official gridfm-graphkit documentation: Citation](https://gridfm.github.io/gridfm-graphkit/#citation). The documentation now specifies the GENCO citation. GENCO is described at https://arxiv.org/abs/2608.09921.

[S07] [Kim et al.: PIGNN-Attn-LS, version 2](https://arxiv.org/html/2509.22458v2). The predecessor includes iterative inference and a backtracking correction, so inherited versus modified settings matter.

[S08] [Rivera et al.: PFΔ benchmark](https://openreview.net/pdf?id=Gi1HtsTAkv). Independent PF benchmark with operating-condition and topology variation, including difficult cases. Code: https://github.com/MOSSLab-MIT/pfdelta.

[S09] [Pacaud et al.: augmented Lagrangian GPU SCOPF, publisher record](https://www.sciencedirect.com/science/article/pii/S037877962600920X). Confirms the volume 262, article 113627, January 2027 bibliographic entry in manuscript reference [10].

[S10] [Wen et al.: physics-informed graph attention PF surrogate](https://www.mdpi.com/1996-1073/19/13/2972). Publisher record confirms manuscript reference [14], published 24 June 2026; not evidence that the reviewed models were implemented correctly.
