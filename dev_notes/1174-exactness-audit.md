The old tgmrf IMH and marginal-theta NUTS sampled the inner Laplace marginal.
Increasing chain length retained its deterministic integration error. The
shared importance estimator now integrates the full latent vector against
the existing normalized family likelihood and Gaussian priors. IMH retains
the estimate and selected particle on rejection. NUTS uses deterministic
gradients and a subsequent PSIS correction with explicit Tier 2 metadata.

The exactness audit also found MCEM pooling and conditional-mean subspace
correction claims that required qualification. Full corrections are tracked
in gcol33/tulpa#1175 and #1176. The guarded native collapsed spatial target is
tracked in #1177; repro-1174-collapsed.R demonstrates the current R-entry
guard, without claiming a returned exact-labelled fit.

The final test run exposed the existing literal CCD k-sample budget.
No-checkout bisection from 8c541675^ through 95cfbb3a, using the failing
settings test's source predicate, localized it to 8c541675. That commit
introduced the public CCD fitter with 40 particles to reproduce its
consumer's design. The value remains 40 and is now read from settings.
The progress assertion observes the pilot and per-theta dispatcher directly,
preserving its requirement that each suppress outer-grid progress.

The Gaussian unbiasedness, retained-state, calibrated-budget and correction
gate tests are in test-tgmrf-is.R. True joint C++-block comparisons are in
test-tgmrf-is-joint.R. The reference uses the same Gaussian beta prior and
dispersion convention as the nested model.
