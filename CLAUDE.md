# TULPA — Template Unified Latent Process Architecture

General-purpose Bayesian hierarchical modelling engine (v0.1.0). Engine
extracted from numdenom, which has since been renamed tulpaRatio.

## Architecture

The hub of a `tulpa*` package ecosystem. The engine owns inference, latent
structure, and the C++ interface; model packages plug an observation
likelihood in via `LikelihoodSpec` and inherit the rest.

- **tulpa** (engine, 0.1.0) — samplers, autodiff, spatial, temporal, priors, formula infrastructure. Imports tulpaMesh for SPDE mesh construction.
- **tulpaRatio** (1.4.3) — ratio, rate, and proportion models (renamed from numdenom).
- **tulpaObs** (0.0.235) — occupancy, N-mixture, and detection models.
- **tulpaGlmm** — RETIRED. Generalized linear mixed models are fitted by the
  engine directly: `tulpa(y ~ x + (1 | g), family =, mode =)` covers the
  families, inference modes and random-effect structure it carried, and the
  machinery that was only ever in tulpaGlmm has been absorbed (warm-starting a
  sampler from a Laplace/EB fit, the marginal-Laplace covariance correction,
  hyperparameter standard errors, empirical-Bayes dispersion estimation, and
  `VarCorr()`). Do not add a GLMM feature there.
- **tulpaMesh** (>= 0.1.3) — constrained Delaunay meshes for SPDE fields (an engine dependency, not a consumer).

Consumers depend on the engine via `Imports: tulpa (>= 0.0.13)` +
`LinkingTo: tulpa` and plug likelihoods in through `LikelihoodSpec`
(see `inst/include/tulpa/likelihood.h`).

## Design Philosophy

**2026 Bayesian engine: nested approximation + debias.**

Decompose the posterior so well-behaved blocks get cheap deterministic
approximations (Laplace, EP, VI, Pathfinder) and exact MCMC corrects only
the residual directions where the approximation is biased.

Positioning:
- INLA: nested approximation, **no** debias — biased on non-Gaussian residuals.
- Stan: exact MCMC, **no** nested approximation — pays full price on every block.
- tulpa: the synthesis.

Canonical compositions:
- IMH-Laplace = Laplace body + MH bias correction
- Pathfinder = L-BFGS mode + Gaussian fit + ELBO scoring
- nested_laplace + CCD = inner Laplace + outer hyperparameter integration

New backends are framed by which layer they add or replace (approximation,
debias, or outer integration), not as standalone alternatives.

## Key Design Principles

1. **Tier system encodes correctness, not hot path.** Tier 1 (exact MCMC), Tier 2 (Laplace), Tier 3 (VI). Hot path is consumer-dependent — for Gaussian-latent hierarchical workloads the typical path is Tier 2 with optional Tier 1 debiasing. Auto mode never silently chooses Tier 3.
2. **Gradient progression N→A→A_r→H** (Tier 1 / NUTS only): Never skip stages. All modes remain available.
3. **Runtime gradient verification**: Before NUTS sampling, verify active gradient against numerical.
4. **Model packages own their likelihood**: tulpa assembles linear predictors; model packages compute log-likelihood.
5. **No copy-paste logic**: Shared sub-computations in helpers, not duplicated across specialized functions. Conventions that keep this single-sourced: log-prior helpers are named `log_prior_*` (e.g. `log_prior_car_proper`, `log_prior_sigma2_pc`); the column-major Rcpp matrix builder `build_matrix_colmajor` is one template over the element type; the spatially-/temporally-varying-coefficient `print`/`summary` methods delegate to `.print_varying_coef` / `.summary_varying_coef` in `R/varying_coef.R`; multi-block prior detection is the single `.is_multi_block_prior` predicate;
   and every `cpp_nested_laplace_*` outer-grid entry hands its shared response
   and control arguments to its driver as one `tulpa::NlEntryInputs`, collected
   by the argument-free `TULPA_NL_ENTRY_INPUTS` macro and consumed by one runner
   per driver (`src/nl_entry_inputs.h`), so a knob added to a driver is added to
   the bundle and the runner rather than to eleven entry tails.
   `test-nl-entry-forwarding.R` asserts every shared argument is observable in
   the fit at all eleven entries; and `check_arg_length` (`laplace_spec_fit.h`)
   is the one place an entry refuses an R argument whose length does not match
   the one it will be indexed by.
6. **Statistical args vs `control` knobs**: front-door fitters (`tulpa()`, `tulpa_nested_laplace()`, `tulpa_nested_laplace_joint()`) carry only statistical arguments in their signature; all perf / numerical / tuning knobs live in a single `control = list()` (e.g. `control$re_cov`, `n_threads`, `max_iter`, `tol`, `adaptive_grid`, `prune`, `integration`, ...). Pre-release: no deprecation shims -- moved knobs hard-error.

## C++ Interface

- Exported headers: `inst/include/tulpa/`
- Model packages use `LinkingTo: tulpa` in DESCRIPTION
- Key types: `ModelData`, `ParamLayout`, `LikelihoodSpec`, `LikelihoodFn<T>`, `ResidualFn`
- **ABI version**: `TULPA_ABI_VERSION` in `model_data.h`. Bump when any exported struct layout
  changes. Model packages auto-check on first NUTS call — mismatch gives a clear error instead
  of segfault. Registered via `R_RegisterCCallable("tulpa", "tulpa_get_abi_version", ...)`.
- **Layout rule**: `ModelData` requires `n_processes > 0` and a `LikelihoodSpec`
  (ratio models live in tulpaRatio via that interface). New fields go in the
  stable sections — never insert before existing fields.
- **Two AD slots on `LikelihoodSpec`, not three** (gcol33/tulpa#493, ABI 42).
  `ll_double` and `ll_arena` only. `resolve_gradient_fn` dispatches on
  `gradient_fn`, then `ll_arena`, then the numerical fallback, and
  `AUTODIFF_FWD` resolves to the arena path, so a forward-dual slot was filled
  by every spec and read by none — six `fwd::Dual` instantiations of likelihood
  kernels that never ran, and an unexercised copy of a density is where a kernel
  falls silently out of step with its siblings. Do not reintroduce one; route
  the mode instead if forward mode is ever wanted.

## Prior anchors, and where a bad one is caught

Every PC prior in the package routes through `pc_prior.h`, whose calibration
`lambda = -log(alpha) / U` exists for `U > 0` and `alpha` in `(0, 1)` only.
Outside that the density is `-Inf` or NaN at every value of the scale, and it
reaches a gradient as a number rather than as a message. `pc_anchors_valid` is
the ONE predicate; the layers differ only in what they can do about it
(gcol33/tulpa#499):

- **R front doors** (`.check_pc_anchors`, `R/validate_helpers.R`) name the
  argument the user set.
- **The sampler entry** (`read_pc_anchors`, `sampler_model_data.h`) names the
  spec, and is what reads an anchor pair off a spec list at all — a pair the
  spec does not carry keeps the `ModelData` default.
- **The templated density** falls back to a flat prior on sigma. It runs inside
  gradient loops and OpenMP regions, where a throw is `std::terminate` rather
  than an R error (gcol33/tulpa#459), so an error is not available to it. The
  callers keep their change-of-variables terms, so an unvalidated path gets
  flat-on-sigma carried correctly to its own coordinate, never a NaN.

`nl_check_positive` / `nl_grid_axes_positive` (`nested_laplace_grid.h`) are the
same convention for a scale or precision that reaches a logarithm — `sigma_re`,
`tau_grid`, `sigma2_grid`, a lengthscale axis — checked at the entry point, the
counterpart of `nl_grid_axis_unit_interval` for the BYM2 mixing weight.

The HSGP, HSGP-ST and TVC scale priors hardcoded `P(sigma > 1) = 0.01` inline;
they now read `ModelData` fields defaulted to exactly that, settable through
`spatial_gp(approx = "hsgp", sigma_prior_U =, sigma_prior_alpha =)` and
`temporal_tvc(sigma_prior_U =, sigma_prior_alpha =)` (gcol33/tulpa#506).

**The two temporal-GP parameterizations are pinned to each other.** They differ
by exactly the forward transform's log-determinant, which is what fixes the
conditional variance both branches use: the non-centered branch reaches it
through the transform's scale `a_t` and the centered branch through
`cond_var_t`, and flooring different quantities leaves them orders of magnitude
apart wherever the floor binds (a long lengthscale on a fine time grid). Both
read one floored `ar1_one_minus_rho2`. `cpp_test_temporal_gp_density`
(`src/test_temporal_gp_fixture.cpp`) drives the shipped density at either
parameterization — the fixture builds the `ModelData` and the layout, not a
second copy of the density — and `test-temporal-gp-parameterization.R` asserts
the identity, including at a configuration where the floor binds at every step.

## Versioning

`0.1.0` is the first CRAN release. Routine work bumps the patch number, which
keeps counting past 9: `0.1.9` -> `0.1.10` -> `0.1.11`. A CRAN resubmission
bumps the patch; a release adding user-visible surface bumps the minor.

## Building

```r
devtools::load_all()
devtools::check(args = "--no-manual")
```

Test profiles (single source of truth `tests/testthat/helper-tiers.R`; tier table
in `tests/testthat/README.md`): `devtools::test()` runs tiers 1 + 2 (structural +
single-fit recovery); `Sys.setenv(TULPA_FAST = "1")` collapses to tier 1 only
(every fit / sampler skips, whole suite in seconds, for plumbing iteration);
`Sys.setenv(TULPA_SLOW_TESTS = "true")` adds tier 3 (samplers, multi-seed
coverage). CRAN runs tier 1 only.

**Release caveats** (the routine `check()` above does not catch these):

- `--no-manual` skips the PDF reference manual, so Rd LaTeX errors stay
  hidden locally. Non-ASCII typographic Unicode in roxygen (arrows, super/
  subscripts, math operators, Greek -- see the ASCII-only rule) only fails on
  `R CMD Rd2pdf` / win-builder / CRAN. Before any release, build the manual
  (drop `--no-manual`) or run `devtools::check_win_devel()`.
- Recovery / coverage tests are `skip_on_cran()`-gated, so a default
  `check()` run exercises plumbing only and a calibration regression passes
  silently. Validate with `NOT_CRAN=true` (recovery) and
  `TULPA_SLOW_TESTS=true` (the 20-seed aggregate coverage gate in
  `test-nested-laplace-recovery.R`) set in the environment.
- Heavy multi-recovery files can SIGKILL (exit 137) under a background test
  harness; run decisive files individually rather than the full suite at once.

## A varying coefficient's level: centre a proper field, pin an intrinsic one

A varying-coefficient term contributes `eta_i += x_i w(s_i)`, so `w -> w + c`
together with `beta -> beta - c` leaves eta EXACTLY unchanged whatever the
covariate. That alias has to go before the field reaches the likelihood, and
which instrument removes it depends on whether the field's own prior is proper
(gcol33/tulpaRatio#25):

- **Intrinsic** (ICAR, RW1, RW2, the interactions built from them): the constant
  direction carries no prior at all, so one has to be supplied. That is
  `sum_to_zero.h`'s augment-and-centre, and `soft_sum_to_zero.h`'s
  `s2z_precision(n)` is the soft form it replaced.
- **Proper** (an NNGP / GP field, a TVC AR1, a TYPE_I interaction): the constant
  direction already carries a prior, precision `1' Sigma^-1 1`. Nothing needs
  supplying; the direction needs REMOVING from the likelihood, which is what
  centring the field on its way into eta does. `s2z_centre_blocks` is the whole
  instrument, and a penalty on the sum is not a weaker version of it -- it
  stiffens a direction the sampler still has to traverse, at curvature
  `lambda n^2` against the field's own `1 / sigma^2`. There is also no constant
  that would make one right: matched to the field's own prior on the sum,
  `lambda = 1 / (1' Sigma 1)`, it is a term the field prior already carries;
  anything else is a second, unstated prior on the level.

Both SVC parameterizations go through `svc_center_eta` (`hmc_svc_autodiff.h`),
so the centring and the eta it feeds are one function and no path can build eta
from an uncentred field. The stored draws are centred to match
(`hmc_nuts_chain_iter_store.h`), under either parameterization.

**The non-centered path is where this is load-bearing, and the centered path is
where it was NOT the defect.** On `w = L z` a penalty on the sum becomes
`-0.5 lambda (v'z)^2` with `v = L'1`, whose stiffness rides the Vecchia cascade
-- large for early-ordered `z`, small for late ones -- so it is anisotropic
against the unit-variance prior on `z` and no diagonal mass matrix reaches it;
that is what collapsed the field amplitude and produced #144's divergence storm.
On the CENTERED path the same penalty was measured and does NOT misbehave:
paired arms on the poisson SVC fixture, 4 chains x 400 iterations, penalty
against centring, `beta_x` 1.661 / 1.654 against a truth of 1.650 at n = 80 and
1.632 / 1.598 against 1.647 at n = 120, Rhat 1.04 / 1.02 and 1.03 / 1.04, no
divergences either way. It is unified because one construction should identify
one alias, not because the other was breaking.

## Boundary: What Belongs in tulpa vs Model Packages

**tulpa owns** (generic, model-agnostic):
- Inference engines: Laplace, EM+Laplace, VI, ESS, NUTS, MI correction, Gibbs correction
- Autodiff: arena, forward, tape
- Latent structure: spatial, temporal, RE, SVC, TVC, ST, latent factors

### Temporal GP (irregularly-spaced times)

`temporal_gp(time_var, cov =, nu =, period =, parameterization =)` is a
continuous-time GP over the DISTINCT time instants, so it is the field for
irregular spacing where RW1/RW2/AR1 assume a grid. It is **sampler-path only**
(`mode = "hmc"` and the other sampler modes): the hyperparameters
`log_sigma2_temporal_gp` / `logit_phi_temporal_gp` are sampled jointly with the
field, and there is no nested-Laplace kernel laying a grid over a dense T x T
Gaussian. It cannot yet share a fit with a spatial or `latent()` block.

The whole density is `compute_temporal_prior()` in `tulpa_priors_temporal.h`,
templated, including the non-centered `z -> f` forward transform (it overwrites
`phi_temporal` in place for the observation loop, so nothing in eta assembly or
the gradient kernels has to know which parameterization ran). Kernels live in
`src/temporal_gp_kernel.h`, templated over the scalar type because
`(sigma2, phi)` are sampled -- a plain-double kernel cannot serve this path,
which is why the untemplated copies deleted in #284 were never wired.

Dispatch is `cov_is_markov()`: exponential (equivalently Matern `nu = 0.5`) is
an Ornstein-Uhlenbeck process, so its joint density factorizes into a
first-order Markov chain and evaluates in O(T) with no matrix; Matern 3/2 and
5/2, Gaussian and periodic have no finite-dimensional state-space form and take
a dense T x T Cholesky. Matern is closed-form at `nu` in {0.5, 1.5, 2.5} only
and R rejects the rest at construction (gcol33/tulpa#288 was those choices being
accepted and then silently run as exponential).
- ZI/OI parameter-layout hooks only (`ZIType` enum, `has_zi` / `has_oi`); the distribution-specific ZI likelihood math lives in model packages
- Censoring/truncation KERNELS only: `interval_gaussian` / `truncated_gaussian` are generic per-observation likelihood arms that model packages compose. General censored / survival responses (right-censored gaussian/lognormal, Weibull/exponential AFT with a censoring indicator) are an observation process and belong to tulpaObs via `LikelihoodSpec`; the engine does not grow a censoring-indicator front door (decided 2026-07-07, closes the recurring todo item)
- Generic S3 methods operating on posterior draws: coef, confint, vcov, logLik, summary
- Generic diagnostics: moran_i, durbin_watson, tulpa_variogram, compare_models, model_average
- Generic plotting: trace, density, pairs plots of posterior draws
- Rubin's rules pooling
- Parameter back-transformation (logit → probability)

**Model packages own** (e.g., tulpaObs, tulpaRatio):
- Likelihood functions (LikelihoodSpec), including the distribution-specific zero-inflation / hurdle / one-inflation and ratio (num/denom) likelihood math
- E-step weight computation (model-specific latent variable posterior)
- Data structures and encoding (how to map model data → binomial pseudo-data for Laplace)
- Model-specific diagnostics (waicOccu, ppcOccu, pitResiduals)
- Model-specific fitted/residuals/simulate
- Data formatting and simulation functions
- Print methods referencing model-specific parameter names

### EM+Laplace Engine

`tulpa_em_laplace()` is the generic EM driver: per-submodel `family` +
`offset` on the `m_step_encode` return blocks (gcol33/tulpa#3) and the
optional `m_step_extra(fits, weights, ...) -> fits` callback for non-η
parameters fired between M-step and E-step (gcol33/tulpa#4). `correction =
"mi"/"gibbs"` run post-EM multiple-imputation / warm-started Gibbs refits
pooled via `rubins_pool()` (`.mi_correction` / `.gibbs_correction` in
`R/em_correction.R`). An optional `beta_prior = list(mean, sd)` threads a
Gaussian fixed-effect prior into every `tulpa_laplace()` block and into the
MI/Gibbs refits (gcol33/tulpa#27); blocks may override it with their own
`beta_prior` field. See `?tulpa_em_laplace`.

### Random-effect covariance integration (free Sigma for random slopes)

For random-slope terms the engine treats the RE covariance(s) `Sigma`
themselves as the inferred quantity instead of a point estimate -- the
nested-approx + debias philosophy applied to a free `Sigma`. Both fitters
operate on a **list of covariance blocks**: one block per RE term, each either
**full** (correlated, `(1 + x | g)`) or **diagonal** (uncorrelated,
`(1 + x || g)`), with a scalar `(1 | g)` term as the degenerate `c = 1` block.
A single-term model is the length-1 case of the same path -- no special-casing.

- **`tulpa_re_cov_nested()`** (`R/nested_laplace_re_cov.R`) -- nested-Laplace
  integration over the joint `Sigma`. A full block parameterizes `Sigma = L L'`
  in **log-Cholesky** coords (log-diagonal + strict-lower of `L`, `c(c+1)/2`
  params, **general `c`**, always PD); a diagonal block uses `c` log-SD coords.
  Per-block params stack into one integration vector. Nodes are centred at the
  joint marginal-likelihood mode and rotated by the Cholesky of its posterior
  covariance, and each derived quantity (`sigma_i`, `rho_ij` for full blocks,
  `Sigma_ij`) is computed per cell then weighted-quantiled -- the "Marginalize
  Derived Quantities" rule (Bias-2). Inner solve is `tulpa_laplace()` (which is
  already multi-RE, correlated-or-diagonal) at the supplied covariances; outer
  is the `nested_laplace` + CCD recipe. Node layout defaults to CCD (`ccd_grid()`
  + corrected `ccd_weights()`, polynomial in total `k`); tensor product opt-in
  via `integration = "grid"`. Default `log_prior_theta` is the weakly-informative
  PC + LKJ hyperprior built **per block** by `re_cov_pc_lkj_prior()` (LKJ only on
  full blocks; `correlated = FALSE` gives the diagonal log-SD prior) and summed
  over blocks, with the exact change-of-variables Jacobian.
- **`tulpa_re_cov_gibbs()`** (`R/re_cov_gibbs.R`) -- the exact debias (Bias-1):
  Metropolis-within-Gibbs (MH on per-(term,group) `b`/`beta` with Laplace-shaped
  proposals, cross-term eta bookkeeping). `Sigma_m | b_m` is an **exact conjugate
  draw**: full block -> inverse-Wishart on the matrix; diagonal block ->
  per-coordinate scalar inverse-Wishart (== inverse-gamma). Removes the Laplace
  under-dispersion that biases `Sigma` low for binary/low-count small groups.
- **`tulpa_re_aghq()`** (`R/re_aghq.R`) -- a deterministic alternative debias:
  replaces each per-group Laplace integral with `n_quad`-point adaptive
  Gauss-Hermite quadrature (`n_quad = 1` is the joint Laplace; higher reduces
  small-cluster variance attenuation). The integration core is
  **structure-agnostic**: it integrates an abstract per-group conditional
  log-likelihood `ell_g(b)` against `N(b; 0, Sigma)` given a `b`-space oracle,
  so the grouping / quadrature / log-Cholesky `Sigma` / LKJ / marginal-Hessian
  machinery is shared across every structure. Three input forms select it:
  - `make_site(theta)` -- the common **single-arm, per-row-separable** case
    (`ell_g(b) = sum_i log f_i(eta_i + Z_i b)` on one linear predictor): the
    caller supplies the per-observation marginal and its first two eta
    derivatives, and the engine builds the oracle from them and the RE design.
    Handles random slopes / correlated blocks sharing **one** grouping factor.
  - `make_group(theta)` -- the **general / multi-arm** case: the caller supplies
    the per-group oracle directly (`grad_hess(g, b)` -> value/score/data-only
    observed info; `node_ll(g, B)` -> log-lik at the quadrature nodes). Arms,
    designs and observation granularity live entirely in the callback, so
    non-separable units and random effects on several coupled arms at once
    (e.g. a community N-mixture: species priors on BOTH the abundance and the
    detection coefficients, coupled through the latent count) integrate with no
    engine change. `re_terms` then carries only the covariance structure
    (`n_coefs` / `correlated` / `n_groups`); the per-observation `idx` / `Z` are
    optional.
  - `oracle` -- a **prebuilt native (compiled) oracle**, an external pointer to a
    `REGroupOracle` constructed in a consumer package's src/ via
    `LinkingTo: tulpa` against `<tulpa/aghq_oracle.h>`: the engine drives it
    directly with **no per-group / per-node round trip into R**. `re_terms` /
    `theta0` / `Sigma0` must still describe the layout the oracle exposes; the
    integration core is identical to the R-closure path. This is the production
    path for consumer-package community fitters (e.g. tulpaObs's
    `nmix_laplace_re()` passes a native `NMixCommunityOracle`).
  Supply exactly one of `make_site` / `make_group` / `oracle`. Fixed params + log-Cholesky
  `Sigma` coords are optimized jointly on `sum_g log M_g`; SEs from the
  exact-marginal Hessian. Optional `lkj_eta > 1` penalizes a weakly-identified
  correlation off the boundary without shrinking the marginal SDs. Distinct
  from `agq_fit()` (`R/agq.R`), which is intercept-only RE -- but its built-in
  `binomial`/`poisson`/`gaussian` densities are now the **shared compiled GLMM
  oracle** (`cpp_glmm_oracle_make`, `src/glmm_oracle.h`): a single C++ source of
  truth that `agq_fit()` (`Z = 1` intercept), the single-arm `make_site` path
  here, `tulpa_re_cov_nested(n_quad > 1)` and the Gibbs sweep all consume,
  replacing the per-fitter R density closures (`.agq_loglik_elt()` /
  `.agq_score_info()`, removed). The gaussian residual variance is `phi =
  sigma_eps^2`; binomial / poisson ignore `phi`. Recovery / equivalence:
  `test-re-aghq.R` (single-arm), `test-re-aghq-multiarm.R` (make_group == make_site
  at d=1/d=2, two-arm N-mixture oracle FD-checks + end-to-end).

Both summarize through the shared `.re_cov_derived_summary` over the per-block
covariance layout (weighted quantiles == sample quantiles at equal weight) and
expose the generic `tulpa_fit` accessors: each returns `draws` (fixed-effect
posterior -- the nested path mixture-samples `N(beta_k, Vb_k)` over the weighted
nodes, the Gibbs path uses its `beta_draws`) plus `means` / `param_names` /
`process_info`, while the `Sigma` posterior stays in `posterior`. With one block
the parameter names are bare (`sigma_1`, `rho_12`, ...); with several they are
prefixed by the block label (`g.sigma_1`, `h.sigma_1`, ...). `Sigma_mean` is a
matrix for one block, a named list for several. Both also report the **per-group**
posterior through `ranef()` (#264): the Gibbs path records the `b` its sweep
samples (`fit$re`, row-aligned with the `beta` draws, so `posterior_predict()`
picks it up too) and summarizes it empirically; the nested path retains each
node's Gaussian per-group posterior (`fit$re_nodes` / `fit$re_var_nodes`, from
the inner solve's `return_re_cov` blocks) and reports the exact moments and
CDF-inverted quantiles of the weighted mixture via
`.nl_gauss_mixture_summary()` -- the continuous counterpart of
`.nl_wtd_quantile()`, carrying both the within-node curvature and the `Sigma`
uncertainty. The AGHQ inner marginal (`n_quad > 1`) integrates each group out
instead, so that fit carries `ranef_unavailable` (a reason string `ranef()`
errors on) rather than an empty table. Tests: `test-re-cov-nested.R`,
`test-re-cov-gibbs.R`, `test-re-cov-recovery.R`, `test-re-cov-prior.R` (Jacobian
vs finite differences, diagonal + joint priors), `test-ccd-grid.R`,
`test-tulpa-re-cov-frontdoor.R` (single, diagonal, multi-term routing),
`test-ranef-re-cov.R` (per-group reporting, mixture summary vs Monte Carlo,
block ordering, cross-backend agreement).
**Status:** fully wired through the `tulpa()` front door. When any RE term
carries slopes (no scalar `sigma_re` to condition on), `mode = "laplace"`
redirects to `re_cov_nested` (default) or `re_cov_gibbs`
(`control$re_cov = "gibbs"`) and treats **every** RE term as a covariance block
-- correlated, uncorrelated `(... || g)`, multiple terms, and any accompanying
`(1 | g)` (a 1x1 block); nothing is silently conditioned at `sigma_re = 1`.
Plain random-intercept-only models keep the scalar-`sigma_re` design path.

### Generic S3 Methods and Diagnostics

Implemented in `R/methods_generic.R` (`coef`, `confint`, `vcov`, `logLik`,
`summary`, `plot`, `tidy`, `glance`, `ranef`), `R/diagnostics_generic.R`
(`compare_models`, `model_average`, `spatial_range`, `temporal_corr`), and
`R/diagnostics_sim.R` (`moran_i`, `durbin_watson`, `tulpa_variogram`,
`pit_residuals`, `test_uniformity`, `test_dispersion`, `test_outliers`,
`test_zero_inflation`, `check_model`).
Model packages inherit via `class = c("model_fit", "tulpa_fit")`.

### Convergence diagnostics (Rhat / ESS) live HERE, not in model packages

`R/convergence.R` owns `mcmc_diagnostics(fit, pars, measures, probs)` ->
data.frame(parameter, <selected measures>) and `select_main_params()`. The
default measures are `rhat, ess_bulk, ess_tail`; the full surface adds
`rhat_bulk`, `rhat_fold`, `ess_mean`, `ess_sd`, `mcse_mean`, `mcse_sd`, and
per-probability `ess_quantile` / `mcse_quantile`. `rhat` is the improved
Vehtari et al. (2021) value: `max(rank-normalized split-Rhat, folded
split-Rhat)`. All estimators are implemented **natively** and reproduce
`posterior::rhat` / `ess_*` / `mcse_*` to ~1e-12 -- do NOT add a `posterior`
/ `coda` dependency for these (they are generic engine output). Each statistic
is one entry in the `.tulpa_diag_measures` registry, so adding a column is a
one-liner. It reads `fit$draws` plus a chain structure (`fit$chain_id`,
`fit$n_chains`, or a 3D `[iter, chain, param]` array) -- the same layouts
`tulpa_draws_array()` (the `as_draws_array()`-style accessor) emits -- so it
works for any `tulpa_fit` subclass; downstream packages (tulpaObs, tulpaRatio)
call `tulpa::mcmc_diagnostics()` rather than re-deriving Rhat/ESS. The plotting
/ summary layer (`plot_rhat`, `plot_ess`, `diagnostic_summary`,
`check_diagnostics`, `n_divergent`) is built on it.

**Draws-provenance gate.** Chain diagnostics are only computed for fits whose
draws are an MCMC chain. Each backend declares its posterior representation via
the registry `emits` property (`"chain"` / `"iid"` / `"point"`), orthogonal to
`tier` (Tier-1 SMC emits `"iid"`; Tier-2 nested Laplace emits `"iid"`; Tier-3
VI emits `"iid"`). `tulpa_dispatch()` stamps it onto `fit$draws_kind`, and
`.tulpa_is_chain()` reads tag-then-registry, treating unknown as chain so
untagged fits still work. On a non-chain fit `mcmc_diagnostics()` withholds
Rhat/ESS (vacuous there: ESS = n_draws by construction) and dispatches to the
approximation-reliability table (`laplace_diagnostics()`, the PSIS/quad-ESS
view) — a point fit gets `NULL` with a message; `mcmc_draws()` is the
chain-only view (`NULL` on any non-chain fit); `check_diagnostics()` returns
`NA` ("not applicable"), and the plot/summary layer withholds the panels
rather than printing a vacuous convergence pass.
`posterior_sample(fit)` is the provenance-agnostic accessor for summaries;
`mcmc_draws(fit)` is the chain-only view (`NULL` otherwise) the diagnostics
gate on.

### Approximation accuracy: Pareto-k-hat (the iid-fit counterpart of Rhat/ESS)

`R/psis.R` owns the native PSIS core `tulpa_psis(log_ratios)` -> `pareto_k`,
`is_ess`, smoothed `log_weights` (Zhang-Stephens GPD fit + Vehtari et al. 2024
weakly-informative prior; reproduces `loo::psis()`, no `loo` dependency -- loo
is Suggests, test oracle only). This is the accuracy gate for non-chain fits,
the counterpart to what Rhat/ESS are for chains: where the gate WITHHOLDS the
chain diagnostic, `pareto_k` is the number to report instead.

`tulpa_re_cov_nested()` computes the **outer** k-hat (`fit$pareto_k`,
`fit$pareto_k_is_ess`, scope `"outer (hyperparameter) Gaussian proposal"`):
the inner-Laplace hyperparameter posterior is importance-sampled against the
Gaussian proposal the integrator places its CCD/grid with (`theta_hat`,
`L_scale`), via `.nested_outer_pareto_k()`. k-hat < 0.7 => the nested
integration is reliable; >= 0.7 => the (skewed / heavy-tailed) hyperparameter
posterior is misfit by the Gaussian grid and the fit should escalate to the
Gibbs debias. Controlled by `diagnose_k` (default TRUE) / `k_samples`
(default 500, each one extra inner Laplace solve); computed after the draw
synthesis with the RNG restored, so draws are bit-for-bit unchanged.

**A high k-hat on a small-group binary RE-covariance fit was read as a correct
signal -- the posterior being genuinely skewed -- and it was mostly the
PROPOSAL's scale (gcol33/tulpa#630).** On `test-psis.R`'s own tiny-binary
fixture, 25 groups x 3 binary observations, the proposal as the mode-find places
it scores 14.6 / 15.1 / 24.3 / 39.4 / 49.1 over five seeds; re-estimated from
its own PSIS-smoothed importance-weighted moments it scores 0.29 / 0.61 / 0.63 /
0.78 / 39.4. Four of five were a mis-scaled proposal, one is genuinely beyond a
Gaussian's reach, and the report now separates them. The skew-normal rescue is
never reached on that fixture: it fires only above the good band, which moment
matching has already cleared. See "One candidate dispatch" below.

`tulpa_nested_laplace()` (the grid-integrated path) computes the same outer
k-hat via `.nl_attach_pareto_k()` + `.nested_grid_pareto_k()`: it re-evaluates
`log_marginal` at a Gaussian-sampled grid by re-dispatching through the SAME
driver (NO new C++ -- the existing kernels already evaluate the inner marginal
at any grid handed to them), unconstrains the positive-scale axis with a `log`
transform, and PSIS-smooths. That axis carries NO change-of-variables Jacobian:
the default grid is geometric (uniform in `u = log theta`) and the integrator
weights it with plain `softmax(log_marginal)` applying no volume element, so
`log_marginal` already IS the u-space target (`R/psis.R:416-427`). Adding one
biases the scale posterior -- the #179 CAR_proper recovery is what establishes
that. Scoped to a single-block,
single positive-scale-axis grid (`.NL_POS_GRID`: `sigma`, `tau`, `sigma2`,
`phi_gp`, ...); multi-block, multi-axis, or bounded-parameter grids (a
correlation `rho_grid`) DECLINE to `pareto_k = NA` rather than apply a guessed
support transform. Same `control$diagnose_k` (default TRUE) / `k_samples` (500)
knobs, RNG-restored.

`fit_spde()` also computes the outer k-hat over `(range, sigma)` (both positive,
log transform), via `.spde_pareto_k()` + the shared `.nested_is_pareto_k()`
core: it reuses the mode-find's own Gaussian (`theta_hat`, `chol(post_cov)` on
the log scale) as the importance proposal and the SAME `log_marginal` + PC-prior
the CCD weights use (no extra Jacobian -- the SPDE integrator works on the log
scale). Both `method = "ccd"` (Hessian proposal) and `"grid"` (grid-moment
proposal) are covered; same `diagnose_k` / `k_samples` knobs. `.nested_is_pareto_k()`
is the shared batched IS-PSIS primitive: proposal `N(theta_hat, L L')` in the
integrator's own space, caller-supplied batched target. Neither the grid path
nor the SPDE path adds a log-Jacobian on a positive-scale axis; only an axis
whose grid is uniform in the NATURAL parameter picks one up, which is the
BYM2 mixing weight's `logit01` transform on the joint path (#221).

`diagnostic_summary()` surfaces `pareto_k` for any non-chain fit, falling back
to the grid's quadrature effective sample size (`sum(w)^2 / sum(w^2)`) when
k-hat was declined or not computed. The joint backend
(`tulpa_nested_laplace_joint`, both the single-block and multi-block paths) now
computes the same outer k-hat over its heterogeneous hyperparameter space
(`R/nested_laplace_joint_pareto_k.R`, gcol33/tulpa#42): a block-type-aware
per-axis transform registry unconstrains each axis -- positive scales (sigma,
tau, phi_*, ...) by `log`, the BYM2 mixing weight (`rho`) by logit, the copy
coefficient (`alpha`) by identity -- and the summed log-Jacobians enter the
importance target. Re-evaluation reuses the SAME kernel the integrator used
(the single-block path threads the generic `kernel_fn` + `hp_fn`; the
multi-block path round-trips a sampled `theta_grid` through the shared
`.joint_multi_cpp_grid` / `_phi_per_arm` / `_add_hp` helpers, serial so no tile
partition is reconstructed), and `.joint_pareto_k` defers to the shared
`.nested_is_pareto_k` core. CAR_proper's `rho_car` (and any other axis whose
support is the adjacency eigenvalue interval) is the unguessable axis: a fit
carrying one DECLINES to quad-ESS (`pareto_k = NA`) rather than apply a guessed
transform -- never a wrong k-hat. Gated by `control$diagnose_k` (default TRUE) /
`k_samples` (500), RNG-restored. The parallel-NUTS
multi-chain producer (`run_hmc_parallel_chains_cpp`, exposed via
`cpp_tulpa_fit_generic_chains`) emits the `(draws, chain_id, n_chains)` layout
`.tulpa_chain_list()` reads, verified end-to-end against `posterior` in
`tests/testthat/test-convergence.R` and on a native multi-chain fit in
`tests/testthat/test-generic-sampler.R` ("mcmc_diagnostics consumes a native
multi-chain fit").

### What an outer k-hat measures, and which lever moves it (gcol33/tulpa#629)

**Skewness is not what an outer k-hat measures; a heavy TAIL is.** The
importance ratio against a Gaussian proposal stays bounded on a skew-normal
target -- Gaussian tail on one side, lighter on the other -- so a genuinely
skewed hyperparameter posterior reads a GOOD k-hat: median grid-moment k-hat
0.224 at true skewness 0.851 and -0.073 at true skewness 0.967, over 11
synthetic single-axis targets x 15 grids x 20 seeds at the shipped
`k_samples = 500`. `.joint_pareto_score_skew` is therefore scored on 0-5.3% of
rows and adopted on 0-1.7%, and where it IS scored -- above the good band, i.e.
on a heavy tail -- it reads WORSE than the Gaussian it was asked to rescue
(median 0.80 to 1.19 against 0.46 to 0.57). Its own header says a skew-normal
cannot absorb a heavy tail; the measurement adds that the case it CAN absorb
does not produce a bad k-hat to begin with.

So there is no proposal-side rung to add to `k_refine`: over 165 cells the
largest gain any un-adopted candidate offers over the shipped choice on a miss
cell is **0.089**, crossing no band boundary, and the four cells offering any
gain are ones where the mixture cannot be BUILT because the grid is too coarse.
Nor should the escalation DECLINE on a skewed posterior -- every target in the
sweep reaches the good band on some grid, including one at true skewness 6.74
and excess kurtosis 141.

**Densifying and widening are not the same lever, and only one works on a heavy
tail.** On `heavy(df=2)` at half-width 3, going 5 -> 41 nodes moves the
grid-moment k-hat monotonically the WRONG way (1.304 -> 2.092); widening to
half-width 12 takes it to 0.077. The shipped `"grid"` rung extends the boundary
where integrand mass piles at an edge AND densifies the interior, so it carries
the half that works -- a rung that only densified would move such a fit
backwards.

**Read a candidate comparison at CELL level, never at row level.** Taking
`min(k_mix, k_skew)` per row and calling the difference a gain is a selection
over two noisy estimates: it manufactures 68 "materially improvable" rows at a
median gain of 0.307 that all disappear under the 20-seed median.

The four backends do NOT agree on this. `.nested_grid_pareto_k`,
`.spde_pareto_k` and `.nested_outer_pareto_k` each call `.nested_is_pareto_k`
ONCE and report the raw grid-moment Gaussian -- no moment matching, no mixture,
no rescue, no `proposal_source`. Over the same 165 cells they read `unreliable`
on 53 where the joint dispatch reads it on 8 (median k-hat 1.159 -> 0.736 ->
0.259 across the three layers), so 45 cells (27%) get a different band from
which backend ran them: gcol33/tulpa#630. That the reported k-hat also moves
with `control$k_samples`, a knob documented as affecting precision only, is
gcol33/tulpa#631; a closed-form Pareto control is flat across the same budgets,
so the estimator wiring is not what moves. Every number above is read at
`k_samples = 500`. Tests: `test-outer-proposal-lever.R`.

### One candidate dispatch behind all four outer-k backends (gcol33/tulpa#630)

An outer k-hat scores a PROPOSAL, so what it reports depends on which proposal
families were offered. `R/outer_pareto_candidates.R` is the one candidate layer:
`.k_cand_spec()` is the backend-agnostic contract (the target closure `lt`, the
Gaussian `(u_hat, Su)`, the integration nodes `(u_grid, w)` where the backend
has them, the `proposal_source`, the axis `names` -- all already restricted to
the subspace being scored, so no scorer indexes an axis set), `.k_dispatch()` is
the choice, `.k_dispatch_report()` adds the reported fields and writes the
`.kdiag_capture()` aperture from the SELECTED proposal. Four candidates,
cheapest first, each gated on the previous still missing the good band:
grid-moment Gaussian, moment-matched Gaussian, grid mixture, skew-normal rescue.
Each rung keeps the better of itself and what it was handed, so a rung can
never make a fit read worse than the rung before it. That is NOT the same as
reporting the minimum over all four, and the exception is deliberate: the
mixture is compared to the GRID-MOMENT Gaussian, not the moment-matched one,
because a refined Gaussian that got under the mixture only by widening past the
grid is not a faithful within-grid reading and must not mask a grid-width
deficiency. So a fit can report a mixture at 1.161 with a moment-matched
Gaussian of 0.889 in hand -- measured, 1 of the 3300 sweep cells
(gcol33/tulpa#634).

**A spec with no node set withholds the mixture rather than inventing one.** The
mixture's bump width is a grid RESOLUTION; a CCD design's spacing is not one. So
`fit_spde(method = "ccd")` and `tulpa_re_cov_nested()` offer the Gaussian,
moment matching and the rescue, and their radius cap stays `Inf` -- which is
what those paths always did. `.joint_cand_spec()` is the joint path's adapter;
everything joint-specific (the per-axis support registry, the pinned axes, the
mode-Hessian splice) is resolved before the spec is built.

Until 0.2.4 only the joint path scored more than one candidate. Over the 165
synthetic cells of gcol33/tulpa#629, the raw first pass reads `unreliable` on
53 where the full dispatch reads it on 8 (moment matching rescues 18 of them,
the two rescues 27 more), median k-hat 1.159 -> 0.736 -> 0.259, and 45 cells
(27%) change band. Realized on fits: the RE-covariance change above; 6 of 15
`tulpa_nested_laplace()` ICAR fits adopt a different candidate with 2 crossing a
band; `fit_spde()` unchanged on its fixtures. The joint path is BIT-IDENTICAL,
asserted by re-running #629's 3300-configuration sweep against its committed
baseline -- every column, including the adopted source.

**Two reported fields answer different questions.**
`pareto_k_proposal_source` names the family behind the number;
`pareto_k_first_pass` is the k-hat of the proposal exactly as the backend placed
it. A large gap says the NODES are badly scaled around the posterior even though
the verdict is fine; no gap says the placement was already right. That is the
"is the integration poor, or is the posterior not Gaussian" distinction
gcol33/tulpa#629 item (3) asked for. The source field also reports which
integrator actually ran: two of three `fit_spde(method = "ccd")` fits report
`grid_moment`, because `fit_spde_nested_ccd()` falls back to the grid path in
three documented cases.

`.k_dispatch()` declines below `.PSIS_MIN_EVAL` before any candidate runs -- the
mixture samples its components before scoring, so without that guard a sub-floor
budget would pay target evaluations to discover it cannot fit a GPD. Tests:
`test-outer-proposal-lever.R` (the contract, the no-grid spec, the floor),
`test-psis.R` (the RE-covariance arbiter). Write-up:
`dev_notes/issue630/RESULTS630.md`.

### Inner-Laplace reliability: gamma_3 (gcol33/tulpa#272)

The counterpart layer to Pareto-k-hat above: k-hat scores the OUTER
hyperparameter-grid integration around a FIXED inner Laplace; `gamma_3`
scores whether that inner Gaussian approximation to the latent-field
conditional posterior `pi(x_i | theta, y)` is itself a good fit. A fit can
carry a healthy outer k-hat and a poorly-approximated inner layer (or vice
versa) -- reading k-hat alone as a whole-fit verdict is exactly the #272
motivating bug (an occu_cover batch read 42/78 species "broken" on outer
k-hat alone when the point estimates, governed by the healthy inner layer,
were fine).

`gamma_3` is the leading-order Edgeworth skewness estimate of
`pi(x_i | theta, y)` relative to the Gaussian inner Laplace (Rue, Martino &
Chopin 2009 Sec 3.2.3's cubic correction term), generalized from their
augmented `x_j == eta_j` representation to tulpa's general
`eta = compute_eta(x)` representation (`src/inner_laplace_skew.h`; the
generalization is proved exact by construction, verified against the paper's
own eq. 21 in the special case, and cross-checked against a direct
numerically-integrated exact posterior skewness in
`tests/testthat/test-inner-skew.R`, matching to within the expected
leading-order undershoot as skewness grows). The paper's `gamma_1` (location
shift) ships alongside it since gcol33/tulpa#354 -- see below. A kurtosis term
does NOT: Sec 3.2.3 itself routes symmetric heavy-tailed cases to a different
numerical procedure (the spline-corrected Gaussian, eq. 17) rather than a
closed-form quartic, so one is not fabricated under the paper's name.

Per-observation third-log-lik-derivatives come from
`curvature3_obs_for_family()` (`src/laplace_family_curvature.h`, exact
analytic ladder for built-in families) or a central finite difference on the
Newton working weight for a consumer-package `LikelihoodSpec`
(`build_spec_curvature3_oracle`, `src/laplace_spec_curvature3.h`). The joint
multi-arm loops generalize the same sum across arms
(`build_joint_curvature3_fns`, `src/laplace_newton_joint.h`).

**A unit reading several linear predictors at once is scored by the widened
contraction, not declined (gcol33/tulpa#301, 0.0.139).** `src/curvature3_contract.h`
carries the derivation: with the unit's coordinates partitioned into K blocks
and `u^(a)` the probe direction restricted to block a,
`sum_{a,b,c} T^{abc} u^a u^b u^c = sum_a d/ds [u' L''(e + s u^(a)) u]_{s=0}`,
so the tensor is never materialised -- each block's term is one central
difference of the Hessian the likelihood already returns, `2K` extra
evaluations per unit, no storage, any block sizes. Two instances:
a multi-process `LikelihoodSpec` (a ZI mixture; blocks = processes, Hessian =
the row-major `n x n` `eta_weights_fn` block) and a `CellCouplingSpec` cell
(`src/cell_curvature3.h`; blocks = the cell's arms, Hessian = the `CellDerivs`
diagonal + cross blocks + rank-1 self-cross, so this is ONE finite-difference
layer on a quantity the spec computes analytically). The step is scaled PER
BLOCK off that block's own eta magnitude, matching the eta-space step the
scalar fallback takes; a single global step is measurably coarser once the
arms' eta scales diverge. The contraction is symmetrised over index
permutations, which for this block decomposition is algebraically the plain
sum (the three relabelings coincide) and buys robustness at a block whose own
quotient could not be formed rather than variance reduction.

`"coupled_likelihood"` is therefore retired: the only remaining decline is
`curvature3_unavailable` (a spec shipping no way to reach a third derivative)
and `coupled_arm` (a coupled fit for which no cell tensor could be built at
all). **Every decline path returns NaN, never a silently-wrong `0`**
("perfectly Gaussian") -- `compute_inner_skew_gamma3[_joint]` early-returns
all-NaN when the oracle is entirely absent, per-index only assigns a value when
at least one finite contribution accumulated, and one unreadable cell takes the
whole contraction to NaN rather than silently understating the sum.

`gamma_3` is a LOWER BOUND on the skewness, not a two-sided estimate: on the
coupled occupancy fixture it recovers 0.299 of an exact 0.530, which bands
"good" where the exact value bands "ok" (pinned in `test-inner-skew.R`).

Wired (opt-in `compute_skew` + `skew_idx`, defaulted off in every C++
kernel) through: the single-arm family/spec kernels (`laplace_newton.h`,
`laplace_spec.cpp`) and every `run_multi_block_nested_laplace[_joint[_sparse_impl]]`
single-arm driver -- covers icar/bym2/car_proper/temporal/nngp/hsgp/the ST
variants/SPDE. R-side: `tulpa_nested_laplace()`'s `.nl_inner_skew_at_theta()`
and `tulpa_nested_laplace_joint()`'s `.nlj_inner_skew_at_theta()`
(`R/laplace_diagnostics.R`, `R/nested_laplace_joint.R`) re-dispatch the SAME
kernel/`kernel_fn` at a length-1 grid pinned to the fitted MAP cell with
`compute_skew = TRUE` -- one extra deterministic Newton solve, no importance
sampling (unlike outer k-hat's `k_samples` batch). Default probe scope is
every arm's fixed-effects coefficients (`control$skew_idx` extends it; the
full latent field is not scored by default -- one linear solve per index).
Gated by `control$diagnose_skew` (default TRUE).

`.tulpa_gamma3_band()` (`good`/`ok`/`unreliable` at 0.5/1.0, a general
skewness-magnitude convention -- Bulmer 1979 -- not a Rue-Martino-Chopin
cutoff) and `.tulpa_combined_reliability()` (`R/laplace_diagnostics.R`)
combine the outer and inner bands into one verdict, surfaced by
`diagnostics()` / `print.laplace_diagnostics()`.

### Inner-Laplace importance k-hat: the likelihood-agnostic floor (gcol33/tulpa#303)

`gamma_3` needs a per-observation third derivative, so a coupled
multi-process likelihood declines permanently. `inner_pareto_k` is the SECOND
score on the same inner layer and needs no derivative at all: the inner
Gaussian at a fixed theta IS an importance proposal for the exact conditional
posterior, and the joint density is the target, so PSIS on that ratio scores
the approximation directly. Available wherever a mode was found.

Scope is the PROBED SUBSPACE, never the field. Importance sampling degrades
with dimension on its own, so a k-hat over `n_x` coordinates would report
`n_x` rather than the approximation. It runs on the same indices `gamma_3`
probes, along the same Gaussian-conditional-mean curve
`x(t) = mode + (t / sigma_i^2) v_i`, with the remaining coordinates
integrated out by the Gaussian conditional -- ONE dimension per probed index,
which is the regime PSIS is reliable in and makes the number comparable to
the `gamma_3` for that index. On that curve the proposal is exactly
`N(mode_i, sigma_i^2)`, so in the standardized offset `z` the log ratio is
`log p_joint(x(z sigma_i)) + z^2 / 2` up to a common constant.

`src/inner_laplace_probe.h` owns the shared `v_i = Sigma e_i` solve against
the live factor (no refactorization), used by BOTH inner diagnostics;
`src/inner_laplace_is.h` evaluates the joint density along the curve through
the Newton loop's own penalized-objective closure, which is why it carries no
likelihood knowledge and serves the single-arm, joint and joint-sparse loops
unchanged. The Pareto fit is the existing shared `.nested_is_pareto_k()`
(`R/psis.R`), which now accepts an injected draw matrix `Z` -- one
importance-sampling k-hat in the package, not two. Wiring rides the existing
`compute_skew` request, so no kernel signature changed.

Draws are engine-owned and deterministic (`INNER_IS_DRAWS = 256`, splitmix64
+ Box-Muller), not taken from R's stream: requesting the diagnostic leaves a
fit bit-for-bit unchanged and the k-hat does not flap with the seed. The
budget is a fixed constant rather than the outer `k_samples` because an inner
draw is one O(N) density evaluation with no factorization, whereas an outer
draw is a full inner Laplace solve.

A Pareto shape index is SCALE-FREE, which matters here because the inner
proposal is often near-exact: a gaussian-family coefficient (inner Laplace
exact, `gamma_3` exactly 0) measures k-hat 0.19 / 0.26 at importance
efficiency 1.000. The shape is therefore banded only on indices whose
realized efficiency falls below `.NL_DIAG$inner_k_material_ess` (0.995); the
raw value is reported regardless and `weights_uniform` records that no index
needed correcting. `.tulpa_combined_reliability()` folds the inner layer's
two scores into one band (the worse where both computed, the one that did
where only one did), so a fully coupled fit gets an inner verdict instead of
"not assessed". Tests: `test-inner-pareto-k.R` (injected-draw equivalence and
RNG invariance, the materiality gate, the gamma_3 cross-check ladder, the
coupled fixture against its exact quadrature, front-door wiring).

**Multi-block joint wiring (gcol33/tulpa#273):** the joint MULTI-block path
(`nested_laplace_joint_multi.R`, used when a joint fit carries a per-group
RE / trend field / arm-specific field block -- e.g. some `occu_cover()`
configurations) now threads `compute_skew`/`skew_idx` through its own
`call_kernel` (`.joint_multi_call_factory`) via `.nlj_multi_inner_skew_at_theta()`,
the multi-block counterpart of the single-block `.nlj_inner_skew_at_theta()`;
both attach points are wired identically (fitted-MAP-cell probe, every arm's
fixed-effects coefficients by default, NaN for a non-"separable" coupled
arm). For a fully-coupled fit like `occu_cover` the answer is still
provably all-NaN (every arm is coupled), now as an explicit NaN field rather
than an absent one.

**SPDE/GP single fits are one-cell runs of the shared machinery
(gcol33/tulpa#277, #282, 0.0.121/0.0.123).** `cpp_laplace_fit_gp`,
`cpp_laplace_fit_spde` and `cpp_laplace_fit_spde_precomputed` used to carry
their own Newton loops (`laplace_mode_gp()`, `spde_run_single_fit()`) -- an
independent implementation the joint-multi driver never touched, so #273's
`gamma_3` pass had to be wired into them separately. Each is now a thin
wrapper: `make_single_arm()` (`nested_laplace_joint_core.h`) + the same block
factory the nested entry uses (`make_nngp_block` / `make_spde_block` /
`make_spde_block_precomputed`) + the joint driver at a one-row grid, projected
back onto the single-fit contract by `nl_grid_cell_to_result_list()`
(`nested_laplace_grid.h`, reading the per-cell `log_det_Q` / `score_max` /
`converged` the grid driver now reports). The equivalence is EXACT and asserted
at `tolerance = 0` in `test-laplace-spatial-gp-spde-equiv.R`; anything the
joint-multi driver gains (`gamma_3` included) is inherited, not wired. With the
last of them migrated, `laplace_newton_solve_sparse` and `spde_run_single_fit`
are gone -- there is one Newton loop behind every SPDE/GP entry.

Two things the migration settled, both load-bearing:

- **NNGP is sparse-only.** `make_nngp_block` scatters its prior into the sparse
  builder alone, so `blocks_require_sparse()` (`latent_block.h`) forces the
  sparse Newton for it at any `n_x`: a non-`INDEXED_SINGLE` contribution OR a
  prior with only `add_prior_sparse` routes there. That also closes the silent
  case where the dense path calls an absent `add_prior` and contributes
  nothing. NNGP is the only block whose dispatch this changes (MCAR / HSGP-MO /
  latent factor are already non-`INDEXED_SINGLE`); the ST NNGP entry had been
  passing `force_sparse = true` by hand for it.

  The observation that motivated the pin -- the dense route measurably apart
  from the sparse one at `nn = 5`, and a 300-iteration non-convergence
  returning `log_marginal = NaN` at `nn = 8` against a 23-iteration
  convergence -- was misread as the two scatters disagreeing. They do not
  (gcol33/tulpa#278): both reproduce `Lambda = (I - A)' D^-1 (I - A)` to ~1e-16
  RELATIVE, and the 2.9e-03 `log|H|` gap was an absolute difference on entries
  of magnitude 1e13. The dense twin is deleted (0.0.124) as unreachable dead
  code, and `test-nngp-prior-scatter.R` holds the survivor to the definition.
  What actually diverged at `nn = 8` was the input to both scatters
  (gcol33/tulpa#283, 0.0.125). `batch_nngp_scatter` hands the neighbour
  factorizations to `cuda_batched_cholesky` once the batch reaches 50
  locations; cuSOLVER returns a COLUMN-major factor and every consumer in
  `linalg_fast.h` indexes ROW-major, so the accepted factor was the input
  covariances carrying a Cholesky diagonal. Conditional variances came back
  wrong by up to 0.28 and 47 of 150 nodes landed on the 1e-10 variance floor
  (true minimum: 2.5e-02), which is where the 1e13 precision entries and the
  infinite `cond(Lambda)` came from. **Every NNGP fit with 51+ locations on a
  CUDA machine was affected**; the CPU path below 50 was always correct. The
  layout is fixed, cuSOLVER's per-matrix `info` codes are read, and the batch
  is now accepted only after one element is checked against the CPU factor.
  `test-nngp-prior-scatter.R` straddles the dispatch threshold -- a fixture
  under 50 locations tests only the path that was already right.
- **Post-loop centring must compensate the intercept.** `center_effects_fn` runs
  ONCE after the Newton loop. The single-arm loop centres and then re-evaluates
  `log_marginal` / `H` at the shifted point; the joint loop computes
  `log_marginal` first and centres with a `CenterFold` into the arm intercept,
  so `eta` is preserved. The bespoke SPDE fit took the first route without the
  fold and reported a converged mode whose fixed-effect score was ~0.47.

`cpp_laplace_fit_spde_precomputed` (the rational/fractional-nu path) takes its
`Q`/`Aeff` already assembled from `.spde_rational_assemble` (`R/brasil.R`), so
`make_spde_block_precomputed` seeds the template `SpdeQBuilder` from that CSC
instead of `qb->init()`, makes `prep()` the `0.5 log|Q|` normalizer only (Q does
not move with the cell), and leaves `block.center` empty -- the latent is the
auxiliary weights `x` (field `u = Pr x`), which must NOT be sum-to-zero centred.
Everything else (`obs_indices`, the pattern, both prior scatters, `log_prior`)
is shared with the FEM entries through `spde_assemble_block`, since all of it
reads whatever CSC the builder holds. Migrating it changed one number: the
shared driver's `log_prior_per_arm_re` carries the RE prior normalizer
`0.5 G (log tau_re - log 2pi)` that the bespoke loop dropped, so an RE-carrying
fit's `log_marginal` shifts by that constant and is now comparable across
`sigma_re` (pinned in `test-laplace-spatial-gp-spde-equiv.R`). Still bespoke:
`implicit_diff.cpp`'s `cpp_spde_laplace_gradient`, which calls
`run_spde_laplace` directly for the `SpdeQBuilder`'s per-entry
`c0_contrib`/`g1_contrib` decomposition and a raw `SparseCholeskySolver&` for
Takahashi selected inversion -- internals the driver's `Rcpp::List`-only
interface does not expose.

`unwrap_skew_idx()` (`laplace_spec_fit.h`) remains the shared
1-based-R-to-0-based-probe conversion. The coupled (non-separable) cubic-term
derivation (#273 item 2) remains open.

**gamma_3 is consumed, not only graded (gcol33/tulpa#302, 0.0.140).**
`control$skew_correct` (ON by default since gcol33/tulpa#364) makes `summary()` /
`confint()` on a nested-Laplace fit report **Cornish-Fisher** marginal quantiles
at each coefficient's own
`gamma_3` (`.nl_skew_marginal()` in `R/laplace_diagnostics.R` over
`src/cornish_fisher.h`), gated to the `good` / `ok` bands and falling back to
the Gaussian quantiles elsewhere -- including where the requested level would
leave the expansion's monotone range. The gate is the COMBINED inner band
(gcol33/tulpa#346): `.nl_skew_correction_attach()` resolves the per-coefficient
band through `.subspace_bands()`, the worse of `gamma_3`'s band and the inner
importance k-hat's, which is what #304's selector already reads. Reading the two
scores differently in the two places had one fit judged reliable enough to
correct and unreliable enough to resample -- on the rare-event fixture `gamma_3`
alone admits every replicate while the combined band is `unreliable` on 38% of
them. `.nl_skew_correction_attach()` records the per-coefficient `gamma_3`, its
band, the k-hat, the combined band, the eligibility, the `reason` from a closed
vocabulary (`.SKEW_CORRECT_REASONS`) and the whole-fit verdict it was decided
under; `.nl_skew_gamma3_eligible()` is what the quantile path consumes, so a
declined coefficient reaches `.nl_skew_marginal()` as NA rather than being noted
and corrected anyway. A `skew_applied` attribute on the summary records what was
used at that level.

**There is no band on the CENTRE (gcol33/tulpa#376).**
`.NL_DIAG$centre_unreliable` shipped at 1.20 from gcol33/tulpa#362 and is now
`Inf`. `m_i = (1/2) sum_j c_j rho_ij` and `gamma_3(i) = sum_j c_j rho_ij^3` are
the same weighted sum at the first and third powers, so a large centre carrying a
small `gamma_3` is uniformly WEAK correlation -- the well-behaved
incidental-parameter regime -- not a strong direction being extrapolated, and the
band was anti-correlated with the pathology it was imagined for. Over seven
fixtures with an exact reference the cutoff ladder is monotone and zero only past
the largest admitted centre measured; the 376 coefficient-seeds it declined
recover 99.4% of the achievable gain once admitted (`t = -14.31`). The machinery
stays -- one predicate, the reason, the precedence, and the cutoff as an argument
on `.nl_skew_correction_attach()` -- so a finite value restores it everywhere at
once. `gamma1_not_computable` is what guards an unformed location term.
The correction is post-processing on the reported quantiles, so draws, modes
and weights are bit-for-bit unchanged either way. It applies on the JOINT
paths too since gcol33/tulpa#305 gave them the per-cell fixed-effect retention
`.nested_fixed_moments()` reads (see below).

NOT a skew normal, on purpose. RMC 2009 Sec 3.2.3 fit one under three
constraints -- mean `gamma^(1)`, variance 1, third log-density derivative at the
mode `gamma^(3)`. A skew normal saturates at `|skewness| ~ 0.995` with the shape
parameter diverging as that bound is approached, inside the band the correction
is gated to. Cornish-Fisher is the quantile-side inverse of the same Edgeworth
series, linear in `gamma_3`, and returns quantiles directly. The paper's own
spline-corrected Gaussian (eq. 17) is the FULL Laplace of Sec 3.2.2, reached
only for symmetric heavy-tailed cases a cubic term cannot describe --
`dev_notes/rmc2009/FACTS.md` holds the quoted passages.

**The correction is applied about the centre eq. (22) implies, not about the
Laplace mode (gcol33/tulpa#354).** `w(z; g) = z + (g/6)(z^2 - 1)` is the
quantile function of a MEAN-ZERO variate; eq. (22)'s density is not mean zero.
Expanding `exp(-z^2/2 + gamma_1 z + (gamma_3/6) z^3)` gives mean
`gamma_1 + gamma_3 / 2`, variance 1 and skewness `gamma_3` to the order kept, so
the reported quantile is
`mu_i + sigma_i {gamma_1 + gamma_3/2 + w(z_p; gamma_3)}`. Placing the mean-zero
variate at `mu_i` asserts `gamma_1 = -gamma_3/2` rather than an absent location
term, and that is the whole of what gcol33/tulpa#346 measured. RMC constrain
their skew normal's mean to `gamma^(1)` alone, setting aside the cubic term's
own contribution to the mean; the centre here is the one their expansion
implies, measured against exact quadrature rather than adopted (total
standardized distance to the exact marginal mean over 12 coefficients: 3.6171 at
`mu_i`, 1.0219 with `gamma_3/2` alone, 2.5799 with `gamma_1` alone, 0.1410 with
both).

**The measurement, 400 prior-predictive replicates of the rare-event
binomial-logit fixture, read off ONE solve per seed.** Endpoint error
`442.52 -> 100.05`, a 77.4% reduction with both endpoints better on 396 of 400.
Paired CRPS against the exact posterior, which reads the whole CDF: `-0.01643`
at `t = -1.89` against the exact posterior's own `-0.01662`, i.e. essentially
all of the achievable gain, with SBC uniformity `0.0833 -> 0.0329` against an
exact reference of `0.0290` and the PIT re-entering the simultaneous band at
`p = 0.089`. Two control arms hold the decomposition: `shift only` (a Gaussian
relocated by the 95%-level offset, no reshaping) scores `-0.0145`, which the
full correction beats -- that is the cubic term earning its place -- and
`no centre` (the same reshaping about `mu_i`) reproduces the #346 loss exactly,
`+0.00775` at `t = +3.54` with KS `0.1138`.

`gamma_1` is REQUIRED, not optional: a coordinate whose location term could not
be formed (a coupled or multi-process unit, a field past the eta-variance solve
budget) declines the whole correction and reports the Gaussian quantiles.
`gamma_3` is a LOWER bound on the true skewness besides, so the reshaping still
moves only part of the way.

**The correction is ON by default (gcol33/tulpa#364, 0.0.186).**
`control$skew_correct = FALSE` restores the uncorrected report per fit, exactly.
Three things were measured before the flip, all on current main after #376 and
#386. (1) The flip survives the shipped gate: scored against the read a
default-OFF fit gives -- the #336 grid mixture -- `t = -1.895` on the rare-event
intercept and `-3.765` / `-3.201` on the small-group Bernoulli design. (2)
Coverage across twelve model classes, read off ONE solve per seed by the shipped
`recov_sweep()`: pooled over 960 trials, `0.9510 -> 0.9542` at a standard error
of `0.0070`, every class inside the 3-se acceptance, gaussian identical to the
bit. Two small-sample classes move in opposite directions and are the whole of
the movement; summed distance from nominal over nine cells, `0.295 -> 0.175`.
(3) The decline paths are exact no-ops (`0.000e+00` on a coupled fit, an
inner-k-declined coefficient, a shape-band-declined one and a non-nested fit),
which is what #386 bought.

**The rare-event class covering LOWER is the exact answer, not a regression, and
fixed-truth coverage is what cannot say so.** A credible interval attains its
nominal rate averaged over the prior, not at one parameter value. Fixture A's
posterior is exact by quadrature, so it runs at fixed truths with the exact
posterior as an arm: pooled over five truths x 400 seeds, exact
`0.9470 / 0.8650 / 0.5630`, corrected `0.9290 / 0.8650 / 0.5630`, Gaussian
`0.9625 / 0.8210 / 0.4165`. The corrected interval reproduces what the exact
posterior does at two of three levels; at `beta = -2`, level 0.50, the Gaussian
contains the truth on 0 of 400 replicates and both the exact and the corrected on
367. Do not read a fixed-truth coverage drop as a defect without an exact arm.

Tests: section 4
of `test-inner-skew-correction.R` (the whole-marginal gate, on #335's
`recov_sbc()` / `sbc_report()` / `sbc_crps_compare()`, slow tier), section 10 of
`test-inner-skew.R` (the `gamma_1` arbiters), plus a paired
corrected-vs-Gaussian coverage gate in `test-nested-laplace-recovery.R`.

### The inner-Laplace location term gamma_1 (gcol33/tulpa#354)

`gamma_3` is the cubic coefficient of eq. (12)'s NUMERATOR along the Gaussian
conditional-mean curve. `gamma_1` is the FIRST-ORDER coefficient of its
DENOMINATOR, `-(1/2) log|H_{-i,-i}(x_i)|`, along the same curve, and it is what
RMC's own Epil GLMM says is the larger of the two ("the simplified Laplace
approximation does correct the Gaussian approximation in the mean, and the
correction for skewness is minor", `dev_notes/rmc2009/FACTS.md` Sec 5.2).

The `inner_laplace_skew.h` SCOPE note used to call it blocked, because the
likelihood-curvature perturbation is diagonal only in the paper's augmented
representation. That OVERSTATED it. With `w_j = -l_j''(eta_j)`,
`d log|M| = tr(M^{-1} dM)` gives

    gamma_1(i) = (1/2) sum_j l3_j * var(eta_j | x_i) * u_{i,j} / sigma_i,
    var(eta_j | x_i) = s_j - u_{i,j}^2 / sigma_i^2,   s_j := [A Sigma A']_jj,

because `[A_{-i} H_{-i,-i}^{-1} A_{-i}']_jj` IS the conditional variance of
`eta_j` given `x_i`. Substituting the second line and recognising the cubic sum
collapses it to

    gamma_1(i) = (1/2) [ (1/sigma_i) sum_j l3_j * s_j * u_{i,j} - gamma_3(i) ],

so the ONLY new quantity is `s_j`, the marginal variance of the linear
predictor, and it does not depend on the probed index -- one pass per fit.
`compute_eta` is a closure, not a matrix, so `A` is never assembled:
`s_j = sum_k (A e_k)_j (A Sigma e_k)_j` reads it off exact affine eta
differences plus the full solves the live factor already serves
(`inner_eta_var_scan`, `src/inner_laplace_skew.h`), identically on the dense and
CHOLMOD paths. `INNER_ETA_VAR_MAX_SOLVES` (5000) bounds that pass; past it the
term declines with `eta_var_budget`.

Three arbiters, none of them the shipped formula (`test-inner-skew.R` section
10): a central difference of the denominator log-determinant on an
independently written model (3e-11 relative over 17 index/model combinations);
the reduction to eq. (21) line 1 at `A = I` over an AR1 GMRF prior, term for
term to 1e-15 (a DIAGONAL prior makes every `a_ij` zero and both sides
trivially zero, so it has to be a coupled prior); and the exact 2-D quadrature
centre check above. The `j in I\i` restriction the paper imposes falls out
rather than being imposed: at `A = I`, `var(eta_i | x_i) = 0`.

Conditional-mode vs conditional-mean is settled, not assumed: eq. (13) replaces
the mode by the Gaussian conditional mean and eq. (19) reads pi_GG at its own
mean. Both curves pass through the joint mode AND share their first derivative
there (`dx*_{-i}/dx_i = -H_{-i,-i}^{-1} H_{-i,i}` is the Gaussian conditional
slope), so the dropped quadratic-form penalty is `O((x_i^(s))^4)` -- past the
order kept.

SCOPE: separable likelihoods (the `scalar` oracle), which is every built-in
family and every single-process `LikelihoodSpec`. A unit reading several linear
predictors at once DECLINES with `multi_eta_unit`. The widened form is written
out in the header --
`(1/2)[(1/sigma_i) sum_units sum_{a,b,c} T^{abc} S_u^{ab} u^c - gamma_3(i)]`,
with `S_u` the unit's K x K marginal eta covariance block -- and what is missing
is an oracle contracting `T` against a MATRIX in two slots and a vector in the
third; `Curvature3Oracle::unit` and `CellCubic3Fn` both contract the same vector
three times, so it is not reachable from them and is left unattempted rather
than approximated. The contraction is CHEAPER than the cubic one once such an
oracle exists (one central difference of the unit's Hessian per unit, against
#301's `2K`).

**Read the measured return before building it** (header SCOPE note,
`dev_notes/issue354`). Computed outside the engine from the coupled fixture's
own R log posterior, the widened term turns a loss into a gain on both
coefficients at one configuration (paired CRPS `t +2.31 -> -2.08` and
`+0.96 -> -2.30`, about 65% of what the exact posterior achieves) and does not
reach significance at another (`t -1.07`, `-1.37`). The location term is not
what dominates there: `gamma_1` is near zero on the occupancy coordinate,
`gamma_3 / 2` does essentially all of the centre, and `gamma_1` ALONE leaves the
centre marginally WORSE than the uncorrected mode at both configurations. What
limits the coupled case is `gamma_3` itself, a lower bound pinned at 0.4-0.7 of
the exact skewness on that coordinate.

### The fixed-effect marginal is a mixture, and is quantiled as one (gcol33/tulpa#336)

What the outer grid defines for coefficient j is a Gaussian MIXTURE over the
cells, `p(beta_j | y) = sum_k w_k N(mu_kj, V_kjj)`. Its mean and variance are
linear functionals and survive being collapsed to one Gaussian; a quantile is
nonlinear and does not. So `.nested_fixed_moments()` returns the components
(`mu` / `var` / `w` / `mass`) alongside the two moments, and
`.nl_fixed_interval()` (`R/nested_laplace_moments.R`) inverts the mixture CDF
through `.nl_gauss_mixture_summary()` for the bounds. `estimate`, `std.error`,
`vcov()` and the debias selectors read the moments and are unchanged, so an
interval difference is the marginal read and nothing else.

The mixture summarizer is NOT new for this: `ranef()` on the nested path
already reported the per-group posterior by inverting the same CDF, while the
fixed effects on the same fit collapsed first. Do not write a second
fixed-effect mixture summarizer -- the two reads meet at
`.nl_gauss_mixture_summary()`.

The #302 skew correction is deliberately NOT composed with this. Its `gamma_3`
is computed at the fitted MAP cell only, so a fit retains `gamma_3(j)` and the
composed marginal `sum_k w_k F^CF_kj` would need `gamma_3(k, j)` -- it is not
identified by retained state, and the three substitutes (scalar shift of the
mixture quantile; the MAP cell's value applied to every component; applied to
the dominant component alone) are each an unbacked assertion. The boundary:
**#336 corrects across-cell non-Gaussianity, #302 within-cell at the MAP.** A
CORRECTED coefficient keeps the #302 read.

**A DECLINED one keeps the mixture read (gcol33/tulpa#386).** There is no #302
read to preserve where the bands refuse, so falling back past the mixture to
`mu +/- z sigma` gave up the across-cell shape for nothing -- and on a fully
coupled fit, where `gamma_1` is unreachable and every coefficient declines, it
moved every bound while correcting none. `.nl_fixed_interval()` computes the
base read first and overwrites only the rows `.nl_skew_marginal()` applied to.
That per-row composition is what makes the correction safe as a DEFAULT: a fit
it cannot help reports what it reported before, bit for bit. `interval_source`
(`"mixture_cdf"` / `"gaussian_moment"` / `"skew_map_cell"` /
`"skew_map_cell/mixture_cdf"`) and `interval_declined` travel on `confint()` /
`summary()`, so a fit says which read produced its bounds and why; the per-row
`skew_applied` says which rows took which.

**A grid that dropped a positive-weight cell is read conditional on the cells
that remain (gcol33/tulpa#342).** `.nested_fixed_moments()` renormalizes over
the cells that retained a block, so the mean and covariance are those of a grid
that never held the dropped cell instead of the whole-grid mean shrunk toward
the origin by the dropped mass. The mixture components carry that same
weighting, so the mixture read serves such a grid too. What it reports is the
posterior CONDITIONAL ON THE RETAINED CELLS: the dropped mass is gone, and no
reweighting brings it back. `mass` on the moments -- and `retained_mass` on
`confint()` / `summary()` -- is the ORIGINAL retained share, 1 on a complete
grid and below 1 on a repaired one, so a reader always tells the two apart from
the fit alone.

Arbiters outside the quantile path: the defining CDF assembled by hand from the
fit's retained cells; `tulpa_posterior_draws()`, which samples a cell by weight
then that cell's Gaussian and so realizes the same mixture; and the reduction
cases. Reduction is exact for a GAUSSIAN-EQUIVALENT mixture (one cell, or
several with identical component means and variances at any weights) and is NOT
required of a merely symmetric one -- `0.5 N(-2, 1) + 0.5 N(2, 1)` is symmetric,
is not Gaussian, and its 95% interval is near `+/- 3.64` against the
moment-matched `+/- 1.96 sqrt(5)`. Asserting equality there would test the read
back into the approximation it replaced. Tests:
`test-nested-fixed-mixture-interval.R`, section 7 of
`test-nested-laplace-joint-fixed-moments.R`, and paired mixture-vs-collapsed
coverage gates in `test-nested-laplace-recovery.R` (`recov_sweep()` scores all
three reads off one solve per seed).

### The hyperparameter axis read: outside the grid, and inside a cell (gcol33/tulpa#357)

A reported per-axis hyperparameter interval is read off the outer grid, and the
grid gives only cell MASSES. Two orthogonal questions have to be answered before
a quantile exists, and `.NL_SUPPORT` (`R/nested_laplace_moments.R`) carries one
field for each:

- `outside` is a FACT about the node set the producer left behind, derived from
  its geometry: a tensor grid and a locally refined one `extend` past the outer
  coordinate by the mirrored half-cell, a posterior sample `clamp`s at its
  extreme order statistic, a CCD is a `moment_rule` that never reaches the
  quantile read at all.
- `within` is a CHOICE the caller makes about how each cell's mass is spread
  INSIDE its own box. `chord` places the cumulative mid-mass at each cell
  COORDINATE and interpolates between coordinates; `box_uniform` places the
  cumulative full mass at each cell EDGE and interpolates between edges. Same
  masses, same boxes, knots moved half a cell.

They are orthogonal, which is why `within` is a second FIELD and not a fifth
kind -- a density grid read either way is still a density grid, and a fifth kind
would have asserted that a fit asking for box-uniform produced a different node
set.

**The default is `box_uniform` (0.0.188).** The engine default lives in exactly
one place, `.NL_DIAG$within_cell` (`R/settings.R`), and `.NL_WITHIN_CELL[1]` and
`.NL_SUPPORT$density$within` have to agree with it (`test-settings.R` pins all
three together). `control$within_cell = "chord"` restores the previous report
per fit, exactly; point estimates, moments, draws and weights are untouched
either way, because this changes only where inside a cell the mass sits.

**What decided it was fixed-truth coverage at the placement the engine ships,
and the placement is what moved.** gcol33/tulpa#337 named fixed-truth coverage
as its verdict instrument in advance and box-uniform failed it, but until
gcol33/tulpa#361 the default axes were laid without reference to the posterior,
so every measurement of this choice -- including that one -- was taken on a grid
pinned coarser than any a user now gets. Re-measured on current main, summed
|coverage - nominal| over nominal 0.95 / 0.80 / 0.50, chord against box-uniform:
0.2900 / 0.1233 on #337's own instrument, 0.2004 / 0.0361 over the truth-swept
fits of the same fixture whose axis contained the truth, and 0.2467 / 0.1572
over nine (config, axis) rows spanning seven families. The one arrangement
box-uniform still loses on is the five-level pinned grid #337 recorded its
failure on, where that fixture's truth sits at fraction 0.9870 of its cell --
the worst position in the box sweep, with the coarser four-level grid tying and
the finer seven- and nine-level grids won. What failed there is a box POSITION,
not a resolution. Evidence: `dev_notes/issue357/RESULTS357C.md`.

**The position sensitivity belongs to any within-cell reconstruction.** An
endpoint resolved to within one cell has a realized coverage that depends on
where in that cell the unknown truth fell. That is measured for BOTH reads, and
at the shipped placement box-uniform's 95% swing is 0.110 against the chord
read's 0.067, where it was 0.415 on the coarse pinned grid; at nominal 0.50 the
two are 0.238 and 0.231. A fit reports its own exposure through
`outer_grid_cell_width` / `outer_grid_axis_sd` / `outer_grid_h_over_sd`
(`.tulpa_grid_resolution()`), so the regime is readable from the fit.

**A resolution-conditional default was scored, not assumed, and is dominated.**
The two reads converge as `h / sd` falls, so a rule keyed on it is expressible;
reading box-uniform only below a threshold scores 0.2517 / 0.2578 / 0.2133 /
0.2322 / 0.1733 at thresholds 1 / 1.25 / 1.5 / 2 / 3 against 0.1572 for
box-uniform everywhere. The threshold that scores best is the one that fires
almost always, which is the fixed rule.

**The regime it is weakest in is pinned, not hidden.** On a coarse grid PINNED
by the caller with four crossed blocks and the placement pass off -- the regime
every measurement before gcol33/tulpa#361 was taken in -- box-uniform wins one
resolution and loses the next (per-axis summed deviation 0.5083 against 0.6500
at four levels, 0.6875 against 0.5917 at five). Four axes give four independent
box positions, so the position sensitivity is at its largest there, and the
chord read's advantage is over-coverage: it sits at exactly 1.0000 on 11 of 24
cells at nominal 0.80 and 0.95. `test-nested-laplace-recovery.R` asserts the
SHAPE of that -- box narrower on every axis, box the one whose realized coverage
spreads -- rather than a winner.

**Two scope limits, both structural.** A locally CCD-refined grid's replacement
clouds sit INSIDE one base cell, so a Voronoi partition of its node set is not
the design's own boxes: `mixed` declines the default and reports `chord` with
`theta_within_cell_declined = "support_mixed"`. That is a read a performance
knob changes, which is why the decline is RECORDED per axis rather than silent
-- an unrecorded one is gcol33/tulpa#317's defect again. And
`.re_cov_derived_summary()` is pinned to `chord`: its values are DERIVED
quantities at the nodes (`sigma_i`, `rho_ij`, `Sigma_ij`), not the design's own
cell coordinates on the axis being reported, so half the gap between two of them
is not a cell width -- the same objection that makes `sample` decline. The
measurement was taken on the outer hyperparameter axes and is not extended past
them.

### The draw budget moves the outer k-hat, not just its interval (gcol33/tulpa#631)

`control$k_samples` was documented as the outer k-hat's precision knob. It is
not, under the automatic PSIS tail rule. `.psis_tail_len(S) = min(S/5,
3 sqrt(S))` fits a tail FRACTION of `3 / sqrt(S)` once `S` passes 225 -- 13.6%
at 500 draws, 1.3% at 50000 -- so a larger budget describes a DEEPER quantile of
the weight distribution rather than the same one more precisely. On a synthetic
heavy-tailed outer target the reported shape runs 0.57 / 1.41 / 3.53 / 7.94 over
500 to 50000 draws, crossing every band the engine reads.

**The estimator is not what moves.** `tulpa_psis()` reproduces `loo::psis()` to
1e-13 at every one of those budgets, and a closed-form Pareto control (Gaussian
target under a narrower Gaussian proposal, exact index `1/(1 - sq^2/sp^2)`) is
FLAT across the same range. Held at a fixed tail fraction the k-hat stops moving
and the seed spread narrows -- 0.569 / 0.702 / 0.708 / 0.692 at 13.6%, against a
range of more than 7 under the automatic rule. The Hill estimator on the same
order statistics stays between 0.02 and 0.36 throughout, and a regression of the
log-ratio on the squared whitened radius is near zero over the realized range:
the weights are nearly bounded where they were actually sampled, and the tail
index of such a ratio is a function of DEPTH, which the automatic rule lets the
budget choose silently.

What this changed: the `k_quality` escalation's precision rung (gcol33/tulpa#627)
doubles `k_samples` on a bootstrap-CI miss, and under the automatic rule that
rung MOVED THE ESTIMAND -- the one thing #627's design says the variance-mover
must not do. It now pins the GPD tail size to the fraction the fit's own first
pass used, so the extra draws sharpen the same number. The fraction is at most
1/5 by construction, so it never trips the 20% cap warning, and an explicit
`control$k_tail_points` is left alone.

**The outer paths now hold the fraction, and the fraction is inherited rather
than chosen.** `.k_outer_tail_points()` (`R/psis.R`), resolved once in
`.k_dispatch()` -- gcol33/tulpa#630's single candidate loop, so all four backends
inherit it -- takes `floor(n * .psis_tail_len(ref) / ref)` at
`ref = .nl_diag("k_samples")` and floors it under the published rule. A DEFAULT
fit is bit-for-bit unchanged on every backend: the helper returns `NULL` at the
reference budget, so the explicit-request path and its 20% cap are never
entered. `tulpa_psis()`'s own default stays the published rule, so the
`loo::psis` equivalence oracle is untouched.

The `max()` is load-bearing. Below the reference budget the published rule is in
its `S / 5` regime and is the MORE generous of the two -- 40 tail points at 200
draws against the fraction's 27 -- so taking the fraction as a REPLACEMENT buys
a stable estimand by making every cheap diagnostic noisier, measured as a
per-arm k-hat crossing its band on `test-joint-pareto-k-proposal.R`'s 200-draw
fixture. Floored, no budget is fitted on fewer tail points than before and the
fraction is confined to `[13.5%, 20%]` over a 500x range instead of collapsing
to 1.3% at 50000 draws.

Which fraction is statistically better is still a bias-variance question this
does not answer, and it does not need to: the property bought is that one fit's
band does not depend on another fit's cost knob. **A skewed target was never
what put a fit in the bad band.** `test-outer-skew-rescue.R` asserted that a
skewness-0.9 target reads unreliable and is repaired by the skew-normal rescue;
measured across budgets it reads 0.017 at the shipped budget and 1.627 at 10000
draws under the published rule, so the fixture reached the rescue only by
scoring at eight times the default. That is the #629 finding from the other
side. With the read stabilized the rescue is not adopted on any fixture in the
repo (gcol33/tulpa#634). Write-ups `dev_notes/issue631/RESULTS631.md` and
`RESULTS631B.md`; tests `test-outer-k-budget.R`.

**There is now ONE budget to hold a fraction against (gcol33/tulpa#632).** The
blocker on that fix was that the engine had two defaults: the joint path
hardcoded `500L` in six places while its three siblings read
`.nl_diag("k_samples")`, which was `200L`. Since the budget sets the tail
FRACTION (20.0% at 200 where the `S/5` cap binds, 13.6% at 500) and the k-hat is
read against the FIXED `k_usable` band, the same hyperparameter posterior was
characterised at two different quantiles depending on which backend scored it.
The registry now holds one value, 500 -- the joint path's, raised there at
gcol33/tulpa#127 when outer scoring stopped being adaptive-batched and the
single budget started carrying the whole estimate, and the budget every shipped
outer-k number was read at. `k_samples_ok` / `k_samples_good` (the `k_quality`
entry budgets) and `k_bootstrap` moved with it.

The drift is instructive about the lint that missed it: `test-settings.R` looked
for `k_samples` next to the literal `200`, and gcol33/tulpa#127 had renamed the
joint variable to `diagnose_draws` and raised it to 500 in the same commit, so
the site was invisible to a rule keyed to one name and one number. The rule is
keyed to the CONCEPT now, and a companion test evaluates every entry point's
default against the registry, since a source lint cannot see a second
registry-shaped default that reads a different key.

**Any measurement scored against an outer grid has to STATE its within-cell
read (gcol33/tulpa#599).** The two constructions place the same mass in the same
cells and differ by half a cell in where inside one they place it, which on a
coarse grid is the scale the LOCATION is resolved at -- so the resolution floor
narrows on the widths and widens on the median, and which part of the read a
candidate rule is shown to reach moves with it. Three measurement files
(`test-nested-laplace-joint-box-mass.R`, `-barycentre.R`, `-descriptor-plane.R`)
pinned every other input and let this one fall through to `.NL_DIAG$within_cell`,
so 0.0.188's flip inverted what they measure while the rules themselves were
untouched. They now share one fixture, `ogd_fixture_sim()` / `ogd_fixture_fit()`
in `helper-outer-grid-dump.R`, whose `within_cell` argument defaults to the
shipped read and is held EQUAL to `.NL_DIAG$within_cell` by
`test-outer-grid-dump.R` -- a later flip fails there, naming the files to
re-measure, instead of silently re-targeting them.

### An axis's NODES and its RESOLUTION are different requests (gcol33/tulpa#633)

`copy_alpha` carries prior structure -- the atom at 0, which is what gives the
"no copy" base model posterior mass, plus a log-spaced slab over [0.1, 3] -- so
`alpha_grid`, which STATES the nodes, also restates that structure. It was the
only way to change the axis, which made it the one outer axis a copy fit could
not integrate more finely: consumers close their own `alpha.grid` off under
`copy()` precisely to stop two sources of truth, leaving no route at all.

`alpha_n` (`field_coef$n` on the single-block path) re-reads the engine's axis
at a higher RESOLUTION: same bounds, same atom, more nodes between them. Both
together is an error, not a silent ranking. `.nl_copy_alpha_axis()` is the one
resolver behind both joint paths, and `.nl_grid_axis(key, n =)` generalizes it
to any declared axis, refusing one declared as explicit `nodes` -- there is no
resolution to vary between nodes someone wrote down.

**The saturation was in the PLACEMENT, not the prune**, which is what #633 left
open. Measured engine-side rather than through the consumer -- an engine defect
needs an engine fixture, or only a consumer can see it: on an ICAR chain with a
gaussian copy arm, raising the donor `sigma_grid` 13 -> 21 -> 29 leaves the alpha
axis at its declared 6 nodes at every setting and grid ESS at 1.7 / 3.1 / 4.3
while the cell count more than doubles, and `prune = TRUE` reproduces both to
the digit. With the resolution raised alongside, 2.3 / 6.8 / 12.5.

Not claimed: that the axis auto-densifies from its own posterior sharpness. No
outer axis does -- they are re-PLACED by the mode-Hessian recenter, not
re-resolved -- and whether the engine should instead solve for a declared ESS
floor is #633's second open question, untouched. Write-up
`dev_notes/issue633/RESULTS633B.md`; tests `test-copy-alpha-resolution.R`.

**`$` partial-matches on a list.** Reading the new field as `fc$n` resolved to
`fc$name` on every spec that names its coefficient, feeding a character into an
integer check and erroring every existing `field_coef = list(name = , grid = )`
fixture. Spec-field reads are `[[`.

### The coordinate dimension is data, and there is one distance (gcol33/tulpa#389)

`tulpa_linalg::coords_dist(coords, i, j)` (`src/linalg_fast.h`) is the ONLY
neighbour-to-neighbour distance on the NNGP paths, and it sums over every column
the coordinate matrix carries. The three loops that need it -- the Laplace
kernel (`laplace_core.cpp`), the batched builder (`gpu_nngp_laplace.h`) and the
PG-Gibbs sweep (`pg_shared.h`) -- each used to form it by hand over columns 0
and 1, reading column 1 UNCONDITIONALLY. On an `n x 1` coordinate matrix that
offset is `1 * nrow + i`, `n` doubles past the end of the allocation, so the
neighbour covariance was built from whatever the R heap held behind the matrix.
Nothing crashed: the read lands inside the heap and returns a finite double.
**The fit stopped being a function of its data** -- same seeds, one process,
`n_threads = 1`, 10 of 20 fits differing at 49 locations with `log_marginal`
moving 4.09 nats.

**Which SIZES broke is not a property of the size.** The first reading proposed
a size-dependent buffer edge on the evidence that 49 broke while 45 was clean.
Re-running the same script in a later session broke 30 and 60 and left 120
clean, where the first had 30 and 60 clean and 120 broken. What lands behind the
coordinate matrix is the process's allocation and free history, so the pattern
moves between sessions and nothing is keyed on `n`. It was never the batched
Cholesky either -- 49 locations dispatches to the CPU path (`batch_size >= 50`
is the threshold) and broke there.

**A constant extra column is the arbiter.** It cancels in
`(coords(o1,k) - coords(o2,k))^2`, so it is the same geometry with the read made
in-bounds: the 1-column fit now reproduces `cbind(co, 0)` and `cbind(co, 1e6)`
bit-for-bit. That is what says the out-of-bounds column was being READ, rather
than the numbers having moved for some other reason.

**Two kinds of consumer, and only one can be made general.** A site that copies
coordinates into a FLAT buffer at stride 2 -- `GPData::coords` and its siblings,
behind every sampler spec, plus the HSGP 2-D basis -- is 2-D by a layout the
samplers and the ABI share. Those REFUSE the input through
`tulpa_linalg::require_coords_2col()` rather than misread it (a
`std::invalid_argument`, so `linalg_fast.h` keeps its Rcpp-free include list and
Rcpp converts it at the `.Call` boundary). The nested-Laplace NNGP/GP kernels
carry the matrix straight to `coords_dist()` and are correct at any `ncol >= 1`;
their `nrow(coords) == n_spatial` checks are the whole requirement. Do NOT add
an arity guard there -- it would reject a 1-D domain the engine now handles.

**Prediction reads the metric the fit was built on.** `cpp_gp_field_predict()`
goes through the same helper and requires only that the prediction and fitted
coordinates agree on their column count. A predictor pinned to two columns
against a dimension-general fit is a silent metric mismatch, not an error.

**The R side imposed the arity rather than checking it.** Four sampler specs and
two prediction paths passed coordinates through `matrix(as.numeric(x), n, 2)`:
an `n x 1` matrix is RECYCLED so column 2 equals column 1 (every location on the
diagonal, every distance scaled by `sqrt(2)`), an `n x 3` matrix is truncated.
`.coords_2col()` / `.coords_plain()` (`R/validate_helpers.R`) are the two
replacements -- error, or strip attributes and keep the shape.

Any measurement taken on a 1-D-coordinate NNGP fixture predates this and should
be re-measured before it is cited; `nngp_120` in
`dev_notes/issue361/RESULTS361EXT.md` is the one in this repo. Tests:
`test-nngp-coords-arity.R`.

### The mode-SD clamp is a substitution, and the two ends answer oppositely (gcol33/tulpa#387)

`.nl_recenter_axis()` lays a re-placed outer axis at `mode +/- span * sd` over
`n_pts` nodes in the axis's own unconstraining coordinate, and the mode SD is
clamped into `[min_sd_u, max_sd_u]`. Whenever a bound binds, the axis is laid
from a number the engine SUBSTITUTED for a curvature the stencil could not read,
and until 0.0.189 that fit was indistinguishable from one laid from a measured
spread. `.nl_recenter_sd_clamp()` (`R/nested_laplace_auto_grid.R`) is the one
place either bound is applied -- behind all four rescues, the spatiotemporal
driver's hand-inlined copy of the same two bounds included -- so what the pass
DOES about a clamp is a policy in `R/settings.R` rather than a constant buried
in a node generator, and the state it returns travels on the fit as
`outer_grid_recenter_sd_clamp` / `_sd_raw` / `_sd_used`.

**The ceiling declines, the floor substitutes, and it is one paired table that
says so.** 200 fixed-truth seeds on each of six configurations x two placement
policies, arms differing only in this setting, summed |coverage - nominal| over
nominal 0.95 / 0.80 / 0.50: `sd_clamp_policy = "decline"` scores 0.1393 against
`"clamp"`'s 0.1464 and never loses a trial -- of the 35 it changes it improves 7
and worsens none (sign test p = 0.0078) at width ratio 1.0000 -- while declining
on the FLOOR costs 22 trials against 9. The asymmetry is the substantive result:
a clamped floor WIDENS a too-narrow axis, the direction that cannot rail, so
substituting there is right; a clamped ceiling lays an axis over a flatness the
stencil could not resolve, so declining is. A third ceiling policy,
`"relative"` (cap the re-placed span by the incoming axis's own span rather than
by an absolute bound), scores 0.1536 and loses 7 to 1; it is kept as a
selectable arm, not shipped.

**Both CONSTANTS are kept, and the ladder is why.** `min_sd_u = 0.15` is a
minimum of its ladder in both directions (0.05 loses 374 trials and wins none;
0.30 loses 30 and wins none) and is the bound that actually binds -- 3 of 7 rows
and every fit of those rows, where the ceiling reaches 2 of 268 axis reads.
`max_sd_u = 3` is the best rung on calibration across a factor-of-15 ladder over
which the summed deviation moves only 0.2843 to 0.3071.

**A fixture built to reach the ceiling cannot arbitrate it, and that is a
property of the read rather than of the cap.** Shrinking an `iid` design until
27.5% of raw mode SDs pass 3 gives a NON-MONOTONE coverage response, and
`rail387.R` names the driver: no axis rails at any rung, and the reported 95%
bound lies OUTSIDE the node range on ~89% of those fits, identically at 0.8 /
1.5 / 2 / 3. So on a genuinely diffuse axis the reported interval is an
`"extend"` extrapolation in nearly every fit and what moves across the ladder is
where that extrapolation lands. Scoring the cap itself needs a fixture whose
posterior the 5-node axis can CONTAIN, which makes it a question about the
ceiling together with `span`, `n_pts` and the outer-cell read
(gcol33/tulpa#390), not about the ceiling alone. The earlier reading that the
ceiling produces 95% widths in the hundreds came from the one shipped row that
reaches it, `nngp_120`, whose fits are not reproducible (gcol33/tulpa#389) --
that row cannot be cited.

**A bound-decline is PER AXIS wherever a rescue re-places several axes at once.**
The spatiotemporal driver moves `(tau_spatial, tau_temporal, rho)` together, and
taking the whole pass down because one axis hit a bound discards the placement
of the axes the mode-find DID resolve -- on the `ar1` fixture, both precision
axes thrown away because `rho` alone was unresolvable. Each axis keeps its own
incoming nodes instead and `outer_grid_recenter_sd_declined` names which did so
and on which bound, so a partially re-placed grid does not read as a fully
re-placed one; the registry rescue has the same shape, declining outright only
when NOTHING moved. A decline for any other reason is a failure of the mode-find
and still takes the pass down. Tests: `test-recenter-sd-clamp.R`,
`test-fit-st-nested-auto-grid.R`.

### Placement is a cheap question paid for with an expensive grid (gcol33/tulpa#636)

Placement reads two things -- the argmax cell and an FD curvature stencil at it
-- and reads both off `log_marginal`. It reads NEITHER off the integration the
detecting pass paid for, so when a placement fires every inner Newton solve of
the grid it detected on is discarded: 120 cells at 25.2 s/cell on the reported
`occu_cover` fit, before the placed grid's own 120.

`control$recenter_pilot` (joint front door, default `FALSE`) detects on a
THINNED grid over the same spans -- both endpoints kept plus at most
`.NL_RECENTER$pilot_n = 3` nodes between them -- and solves the full grid once,
at the placed axes. **The rescues need no change for this**, which is the whole
reason it is a caller-side wiring rather than a rewrite: each already detects on
the fit it is handed and writes onto the prior it is handed, so
`(pilot fit, full prior)` places the full grid. `R/nested_laplace_pilot.R`.

With `F` full cells, `P` pilot cells, `S` stencil cells and a firing rate `p`,
the un-piloted cost is `F + p (S + F)` against the pilot's `P + p S + F`, so it
pays exactly when `P < p F`. `p` is a property of the WORKLOAD -- every species
of #636's 78-species run fires -- which is why this is a knob and not a policy.
Measured over 48 paired seeds on a `(sigma, alpha, phi)` donor + copy fixture:
0.78x the cells solved on the configuration placing on 10 of 12 seeds, 1.29x on
the one that never places.

**A coarse detector is not the same detector, and the asymmetry is built in.**
The collapse trigger is `ess_grid < 2`, which falls with the cell count, so a
pilot fires somewhat more readily -- the two arms disagreed on 9 of 48 pairs,
every one the pilot placing where the full grid did not, 7 of the 9 reaching a
HIGHER maximum inner log-marginal (+0.35 to +2.14 nats) and 2 losing 0.46 and
0.72. It cannot go the other way: a pilot whose placement declines is followed
by the full fit, which then gets its own detection through the same rescue pair,
so the pilot is a PRE-SCREEN that can only ADD a placement. A fit declining both
is bit-identical to the un-piloted one, asserted at `tolerance = 0`.

Scope is the joint path. The registry stores a family's axes PRE-PAIRED
(`.nl_fill_family_axes()` crosses them into one row per tuple), so thinning them
per axis there would re-pair the grid rather than thin it. `.NL_PATH_CROSSES`
declares that property beside the `.NL_PATH_AXES` table that declares which
fields each path reads; `.nl_pilot_block()` is its only consumer, and a block on
a paired path is left whole and named in `outer_grid_pilot$axes_kept`. A copy
coefficient is thinned by RESOLUTION (`alpha_n` / `field_coef$n`, #633) and
never by subsampling -- its atom-plus-slab is prior structure, so a subsample of
a STATED axis states a different axis.

**A recentred axis's reported `h / sd` is NOT the layout's 1.25, and the two
were read as one number.** The layout's ratio is in PLACEMENT SDs: the axis is
`mode +/- span * sd_used` over `n_pts` nodes, so its spacing is `1.25 * sd_used`.
`.nl_axis_h_over_sd()` divides that by the grid-WEIGHTED posterior SD the placed
grid realizes, so what it reports is `1.25 * sd_used / sd_realized` -- equal to
1.25 only where the placement SD is the one the weights realize. Wherever
`.nl_recenter_sd_clamp()` SUBSTITUTED a bound it is not, and the floor
`min_sd_u = 0.15` exists precisely to widen an axis sharper than it. Measured:
the floor bound on 18 of 18 placed fits, `sd_raw` median 0.0543 against the
substituted 0.15, realized 0.0489, reported 3.83 -- and `1.25 * 0.15 / 0.0489 =
3.83`. A reported 6.03 is `sd_raw = 0.031`, inside the fixture's own range
(2.37 to 6.79). The placement is still working there: the same axis un-recentred
reads a median 55.2. An axis NO rescue moves (a copy `alpha`, a caller-pinned
dispersion axis) carries no such relation at all -- median 1605 and 1.95 on the
same fixture -- so a large reading there is a sharp posterior on a fixed axis,
not a mis-sized one.

`.nl_grid_ess(w)` is now the one read of a grid's quadrature ESS, behind
`.joint_pareto_grid_regime()` and `diagnostic_summary()` alike. Write-up
`dev_notes/issue636/RESULTS636.md`; tests `test-recenter-pilot.R`.

### A cell's cost is the predictive-variance loop, and the screen is now reachable and priced (gcol33/tulpa#638, #639, #640)

On the single-block path the dominant per-cell cost is not the inner Newton
solve. It is the per-row loop filling `fitted_eta_var`: one back-solve per
observation per cell. Switch it off and the per-cell cost is FLAT in N (0.225
to 0.254 s over N = 900 to 10800 on a 30x30 ICAR lattice, 11 cells, serial);
leave it on and it is affine, `0.217 + 4.685e-4 N`. So the loop is the whole of
the size dependence, and `control$fitted_var = FALSE` is 21x at N = 10800.
Newton iterations are 8% of a cell, which is why the continuation and
inexact-Newton levers that look like the obvious ones are competing for the
small half.

**The loop's cost is set by DISTINCT loading vectors, not by N.** A row's
loading vector is `(X row, RE group, per-block index and weight)` and none of
it moves with the outer cell -- only the per-block scalar `d_fac_b(k)` does,
and it scales every entry that block contributes at every row alike. So two
rows agreeing on all of it carry the SAME vector at every cell, and the driver
solves one and hands the value to the rest. That is falsifiable against the
affine line above, and it holds: an intercept-only design with 900 distinct
rows of 5400 costs 0.244 of what its N predicts and **1.047 of what its
distinct count predicts**, 4.07x. Where every row is distinct the dedup buys
nothing, by construction -- it is not a faster solve, it is the observation
that the solve was being repeated. The key is compared on exact IEEE bit
patterns and every merge is confirmed element-by-element, so a hash collision
costs a comparison and cannot fuse two rows.

**The cheap screen was hardcoded off on every single-block entry** -- both
runners in `nl_entry_inputs.h` passed `prune_tol = 0.0` -- so the one path
where a cell is expensive enough for screening to pay was the one path that
could not screen. `control$prune` / `$prune_tol` / `$screen_iters` /
`$fitted_var` now thread through `TULPA_NL_ENTRY_INPUTS` to all eleven entries,
under the same `.joint_prune_safety_gate` the joint door applies, so a
single-block fit can no more return a silently pruned answer than a joint one.

**The screen's depth is a cost-against-ranking trade, and 5 was the wrong side
of it.** The screen only has to RANK cells and each is warm-started from its
already-screened neighbour, so every step above what the ranking needs is paid
on the whole grid including the cells it keeps. Measured over four fixtures at
depths 1/2/3/5: Spearman 1.000 against the full solve, argmax kept, at most
1.2e-3 of the true mass dropped, fixed effects within 4.8e-5, gate never fired
-- the same answer at every depth. What moves is cost, and on the fixture that
isolates the screen's own price (single-block with the variance pass off) depth
5 is a NET LOSS at 0.75x while depth 1 is 1.33x. `.NL_SCREEN$iters` is 2, not 1:
on a small 8-cell ICAR fixture depth 1 scored Spearman 0.976 where depth 2
scored 1.000, matching the issue's own 0.9989-against-1.0000. Pruning stays
opt-in, so the default reaches only a caller who asked to screen.

Read row 3 of that table with care: with the variance pass ON the screen looks
good at any depth for a reason that is not about the screen, since `cheap_eval`
runs with `want_var = false` and so skips the O(N) loop the full solve pays.

`.NL_SCREEN` (`R/settings.R`) is the one registry for both numbers, and the
depth lives at 24 C++ sites besides -- `CHEAP_SCREEN_ITERS` plus the Rcpp entry
literals, which cannot read an R registry. `test-settings.R` pins all three
together, so a partial move fails loudly rather than leaving the engine with
two defaults the way `k_samples` did. Tests: `test-nl-fitted-var-dedup.R`,
`test-nl-single-block-prune.R`, `test-screen-depth.R`, and the cheap-screen
block of `test-nl-entry-forwarding.R` at all eleven entries.

### Three posterior arbiters, and coverage is only one (gcol33/tulpa#335)

Binary coverage at one or two nominal levels reads one or two points of the
marginal CDF. It cannot say whether an approximation is biased, over- or
under-dispersed, or asymmetric, and #336 above is where it ran out: over 200
paired seeds the mixture read and the collapsed-Gaussian read moved 0 or 1
trials. `R/sbc.R` holds the two instruments that read more than two points,
ALONGSIDE the fixed-truth sweeps and not replacing them.

**The scorer lives in `R/sbc.R` behind the exported `sbc()`, and there is one
copy (gcol33/tulpa#380).** It was written and arbitrated as sections 1 to 6 of
`tests/testthat/helper-sbc.R` and moved there unchanged; that file now holds
only the engine FIXTURES (sections 7 to 9), which read the package functions.
Do not reintroduce a private copy -- the duplicate scorer is the specific thing
the promotion prevents. `sbc(experiment = )` selects `recov_sbc` (a
`simulator` / `fitter` pair) or `recov_posterior_sbc` (the six-callback
`model`); the drivers, `sbc_report()`, `sbc_crps_compare()` and the band stay
INTERNAL, and the only other exports are the five predictive shapes
(`sbc_mixture` and its siblings), which are the argument type a fitter returns
rather than alternative verbs. Methods: `print`, `summary(baseline = )` (adds
the paired proper-score ranking), `plot` (the ECDF difference against the band),
`diagnostics`, and `diagnostics(fit, sbc = )` for the fit and its calibration
read together. Two guards ride the door: the prior-predictive path refuses a
scored quantity whose truth does not move across simulations -- what the nested
door's improper fixed-effect prior looks like from outside -- unless it is named
in `flat_prior` (checked in both directions), and the posterior path refuses a
`pool()` returning no more than one of its inputs and verifies disjoint group
LABELS when `model$group_ids` is supplied. The unobservable half of that premise
(where the effects at those labels came from) is recorded as unverified, never
claimed. Tests: `test-sbc-frontdoor.R` for the door, `test-sbc-crps.R` and
`test-posterior-sbc.R` for the scorer and the fixtures, both of which now run
their acceptance reads THROUGH `sbc()`.

- **`recov_sbc(simulator, fitter, n_seed)`** -- simulation-based calibration.
  Draw the truth from the prior (`theta_s ~ p(theta)`, `y_s ~ p(y | theta_s)`),
  fit, take `u_s = F_s(theta_s)`; under correct inference `u_s ~ Uniform(0, 1)`
  and the whole ECDF is the measurement. It reports the PIT, the FOLDED PIT
  (`2 |u - 1/2|`, also uniform, and where a symmetric dispersion error shows
  after cancelling in the raw ECDF), and the JOINT LOG-LIKELIHOOD RANK, which
  reads the whole data set at once and catches an approximation whose
  per-coefficient marginals look right. Arms are paired off one solve per seed
  the way `recov_sweep()` pairs its mixture / collapsed / skew reads.
- **`sbc_crps()`** -- the strictly proper score, closed-form for a Gaussian
  mixture through the pairwise kernel terms (Grimit, Gneiting, Berrocal &
  Johnson 2006), so the nested tier's own mixture is scored with no Monte Carlo.
  `sbc_crps_compare()` pairs it seed by seed.

**CRPS is a proper POSTERIOR score only in a prior-predictive experiment.** With
`theta = theta_0` fixed across seeds the CRPS-optimal forecast is a point mass
at `theta_0`, so a sharper wrong posterior wins. This is enforced, not just
documented: `recov_sbc(truth = )` records the experiment, and
`sbc_crps_compare()` errors on a fixed-truth result rather than ranking it. The
fixed-truth sweeps keep coverage and width.

**The bands are simultaneous, calibrated exactly.** A pointwise binomial band is
NOT a simultaneous band -- at n = 100, holding each order statistic at 95% holds
all of them together at 0.4471. `sbc_crossing_prob()` is the exact crossing
probability for `g_i <= U_(i) <= h_i`: the constraints reduce to
`#{i : h_i <= p} <= K(p) <= #{i : g_i < p}` at the union of the boundary points,
`K` is a binomial Markov chain in `p`, and the forward pass masks the state to
the band's own width. `sbc_ecdf_band()` bisects one boundary family against it
-- equal local levels (Beta quantiles, the SBC band) or constant width (the KS
band, kept as the conservative cross-check). Arbiters: closed forms
(`P(all U <= t) = t^n`), brute-force simulation, measured simultaneous coverage,
and the published Kolmogorov critical value, which the constant-width member
reproduces to four figures.

**Every discrete PIT is randomized within its atom** (`u = F(theta^-) + V
P(theta)`, i.e. `(r + V) / (n_ref + 1)` for a rank), so one uniform reference and
one band serve rank, grid-axis and continuous quantities alike. Reading
`rank / n_ref` against a continuous uniform is the classic silent SBC bug and is
kept as a negative control.

MEASURED on a gaussian random-intercept fixture whose exact posterior is
available in closed form (the engine's read tracks it to 1.3e-05 in the PIT),
2000 prior-predictive replicates: SBC separates the #336 mixture read from the
collapsed Gaussian carrying the same two moments (simultaneous p = 0.44 against
1.9e-04 raw, 2.5e-04 folded on the intercept) where 200-seed coverage could not,
and the paired CRPS does not (t = 1.01) because the score is dominated by the
two moments the reads share. The slope separates under neither, which is
gcol33/tulpa#325's finding again. All three broken control arms are caught, and
the #332 residual-scale crossing shows on the joint log-likelihood rank while
its intercept marginal stays inside the band. Tests: `test-sbc-crps.R`; write-up
`dev_notes/issue335/RESULTS.md`.

### Posterior SBC: calibration conditional on an observed data set (gcol33/tulpa#339)

`recov_sbc(truth = "prior_draw")` above reports calibration AVERAGED over the
prior. A user fitting their own data asks something narrower -- is the inference
reliable in the posterior geometry THIS data set produces -- and the prior
average can both miss a defect confined to a small region and flag one the
observed data rules out. `recov_posterior_sbc()` (`R/sbc.R` section 6, reached
through `sbc(experiment = "posterior")`,
after Sailynoja, Schmitt, Buerkner & Vehtari, *Stat Comput* 36:78 2026,
doi:10.1007/s11222-026-10825-9, Algorithm 2) answers the narrow one:
`theta' ~ pi(theta | y_obs)`, `y ~ pi(y | theta')`, and the PIT is taken under
the AUGMENTED posterior `pi(theta | y, y_obs)`. That is ordinary SBC with
`pi(theta | y_obs)` in the role of the prior, so #335's whole instrument set --
the exact simultaneous band, the folded read, the randomized discrete PIT, the
CRPS closed forms -- carries over unchanged, and `truth = "posterior_draw"` is a
proper-score experiment for the same reason `"prior_draw"` is.

**Two premises make it an SBC experiment, and each has a negative control.**
Neither is a detail; both are ways the construction silently degrades into
something else.

- **The augmented posterior conditions on BOTH data sets.** Fitting the
  replicate ALONE is ordinary SBC under a hand-made prior, and it is the easiest
  collapse to fall into. Control: the sigma read leaves the band at p = 2.7e-10,
  worst of the four -- `y_obs` was most informative about the hyperparameter --
  though the failure is not confined to it.
- **The replicate is conditionally independent of `y_obs` given theta.** The
  nested tier integrates the random effects out, so theta carries no per-group
  value and a replicate on the SAME groups couples the two data sets through the
  unmodelled group effects. Every replicate is drawn on FRESH groups. Control:
  re-observing the observed regions with their effects drawn from
  `p(u | y_obs, theta)` takes the intercept, the hyperparameter and the joint
  log-likelihood outside the band (p = 0, 0, 2.7e-14) while the slope survives
  at p = 0.27 -- it reads within-region contrasts, which a shared per-region
  effect cancels out of.

**The gaussian fixture is the CONSTRUCTION arbiter, not an engine verdict**
(`sbc_psbc_gaussian()`, section 8). Its augmented posterior is available in
closed form, so the whole scheme runs with the exact posterior at both stages
and its PIT must be uniform; a departure there is a defect in the pooling, the
seeding or the replicate, never in the engine. It cannot say anything ABOUT the
engine, because a gaussian log-likelihood is quadratic in eta -- the inner
Laplace IS the conditional posterior and the two reads agree to 1e-04 in the
PIT. `sbc_psbc_re()` (section 9) is the family-general fixture where the inner
Gaussian is an approximation and the verdict is real.

**The DRIVER splits the truth-draw and replicate RNG streams**
(gcol33/tulpa#350): `draw_theta` gets `s`, `simulate` gets `.sbc_rep_seed(s)`,
so the obvious `set.seed(seed)` at the top of each callback is the correct
fixture and no fixture carries an offset. Handing both the same seed makes the
replicate's noise a function of the truth, which is not `p(y | theta')` and
shows up as a non-uniform PIT with nothing wrong in the inference under test --
a harness whose default failure mode is a false alarm against the engine. The
offset the driver applies is the `660000L` every fixture used to apply itself,
so the seeds a fixture actually sees did not move and the single-arm RE
configurations of the measurement below reproduce unchanged; the coupled
occupancy and `occu_cover` configurations carried a different offset and were
re-measured.

**A rank arm needs a marginal likelihood, and off the gaussian that means
quadrature.** `sbc_loglik_re()` is ADAPTIVE Gauss-Hermite -- recentred and
rescaled at each region's own integrand mode, mode located by one vectorized
scan plus central-difference Newton across all regions at once. A fixed rule
scaled by sigma places its nodes by the PRIOR while the integrand is the
likelihood times that prior, and the likelihood bump is narrower than the node
spacing: measured against the closed form it stalls at 3.1e-03 by 64 nodes once
beta is a couple of units from its estimate. The adaptive rule is EXACT for the
gaussian family at any node count from 2 (the integrand is Gaussian there), and
converges geometrically elsewhere. Nodes come from the Golub-Welsch
eigendecomposition of the probabilists' Hermite recurrence -- twelve lines of
base R, no dependency. Same reason `tulpa_re_aghq()` adapts.

The MEASUREMENT is `dev_notes/issue339/`: 15 configurations at N = 1000
including `occu_cover`, a pre-registered family-wise verdict rule, and a power
curve (80% power at roughly a 10% over-dispersion or a 0.14-SD location bias;
measured false-positive rate 0.0117 against a nominal 0.05, so a rejection is
the strong statement and a pass the weak one). Its headline is that the cheap
band and the expensive check disagree in BOTH directions:

- `pois_40` -- outer k-hat 0.196, `gamma_3` and inner k-hat both `good`, so the
  band's cleanest verdict, `reliable (both layers good)` -- FAILS calibration at
  p = 2.3e-13 on the intercept.
- `binom_30` -- outer k-hat 1.413, well past the 0.7 escalation threshold --
  PASSES at p = 0.17.

So the shipped band is a screen, not a verdict, and neither direction of it is
safe to read as one. Tests: `test-posterior-sbc.R`.

### Per-cell fixed-effect retention on the joint tier (gcol33/tulpa#305)

`.nested_fixed_moments()` (`R/methods_generic.R`) is the ONE grid marginalizer
behind `summary()` / `confint()` / `vcov()` on every nested tier, and it reads
ONE representation: `$grid_modes[[k]]` and `$grid_hessians[[k]]`, cell k's
fixed-effect mode and marginal precision. `tulpa_nested_laplace()` fills that
pair in `.nl_attach_grid_hessians()`; until 0.0.142 the joint driver filled
neither, so EVERY `tulpa_nested_laplace_joint()` fit -- single-block and
multi-block alike -- reported point estimates with `NA` for both bounds and
every standard error, and #302's skew correction was recorded on such a fit
but had no quantiles to correct.

Both joint paths now fill the same pair through one shared helper,
`.joint_attach_grid_fixed()` (`R/nested_laplace_joint_helpers.R`), reached from
the one exit point `.joint_finalize_grid_fixed()`. Do NOT add a joint-specific
marginalizer; the whole point is that the two tiers meet at
`.nested_fixed_moments()`.

Three things make this work and are load-bearing:

- **The fixed block is a contiguous latent prefix.** `.joint_layout()` and
  `.joint_multi_layout()` both start `beta_start` at 0 and lay the arms out
  consecutively, so every arm's coefficients together are latent indices
  `1:n_fixed` -- the same span `.joint_fixed_layout()` names. The extraction is
  arm-aware because it takes the whole block at once, not because it loops arms.
- **The block comes from the joint tier's own extraction**,
  `extract_inner_vcov_block_cell()` (`src/joint_inner_vcov.cpp`), with the field
  sum-to-zero columns from `.joint_constraint_cols()`. So the reported
  covariance is the CONSTRAINED one `tulpa_posterior_draws()` samples from; an
  unconstrained block would disagree with the fit's own draws. The single-block
  path's `.nl_attach_grid_hessians()` is a separate, unconstrained extraction
  whose numbers are pinned -- the two are deliberately not merged.
- **The block is extracted inside each cell's own solve (gcol33/tulpa#307).**
  Both joint Newton loops (`laplace_newton_joint.h`,
  `laplace_newton_joint_sparse.h`) take a `JointFixedBlockRequest` -- the leading
  block size plus the constraint groups, both fixed by the latent layout before
  the first solve -- and return the block on the `LaplaceResult` `re_cov_flat`
  contract, which the grid driver emits as `cov_block_per_grid`. Nothing keeps
  the precision: the dense loop builds one cell's CSC, extracts, and releases it;
  the sparse loop reads the builder's own CSC, so there it costs no copy at all.
  `store_Q` is once again the caller's own knob, passed straight through.
  The kernel closure requests the block only when `store_extras` is set, which
  keeps the outer Pareto-k batch from paying for hundreds of cells it would
  discard, and the cheap screen never requests it.
- **Refinement carries the blocks, so nothing declines for having moved the
  grid.** Adaptive refinement and the var-of-means consistency pass carry them
  through `.joint_glue_extras_to_res` alongside the modes; local-CCD refinement
  carries them through `.joint_local_ccd_refine(cov_blocks =)` the same way, so
  the `grid_fixed_declined = "local_ccd_refined"` decline #305 recorded is gone.

`control$keep_grid_hessians` (default `TRUE`) switches the retention off, and
`$grid_fixed_declined` always says why a fit has none.

**The retained pair is parallel to the weights by construction, not by luck
(gcol33/tulpa#345).** `.joint_attach_grid_fixed()` preallocates `grid_modes` /
`grid_hessians` at `n_grid` and writes each cell's slot in turn, so a cell with
no usable block must SKIP its slot -- `l[[k]] <- NULL` REMOVES the element. An
interior blank cell hid that: the next write lands at its own index and
re-extends the list, and shifting NULL padding left is a content no-op. A
TRAILING blank cell had nothing after it, so the pair came back one shorter than
`weights` and `.nested_fixed_moments()`'s length check returned `NULL`, NA-ing
the entire coefficient table with `grid_fixed_declined` reporting `NA`.
Refinement is what makes the trailing case reachable, since it appends cells at
the END of the grid and an appended cell whose inner solve returns
`log_marginal = NaN` gets softmax weight 0 and no block. The cell is carried as
an empty slot rather than dropped from `weights`: it holds zero weight, so
`mass` stays 1 and nothing is reported conditional on a reduced grid, which is
what the #342 renormalization and the zero-weight `keep` filter were already
written for. `"no_weighted_cell_block"` is the reason when the retention holds
no cell the weights put mass on at all.

One extraction algebra serves every caller: `src/inv_block_extract.h` holds
`InvBlockConstraint` (the conditioning-by-kriging correction `W = H^{-1}A'`,
`M = A W`, `chol(M)`) and `extract_inv_diag_blocks()`, both templated on a solve
oracle. `laplace_newton.h`'s `inv_block_layout` path drives it against the LIVE
Newton factor with no constraint; `extract_inner_vcov_block_cell()` drives it
against a freshly factorized cell with one. The joint loops go through the
latter, which is why the block they produce is byte-identical to what
`cpp_joint_inner_vcov_blocks()` returns for the same cell -- same bytes in, same
routine. The loops do NOT reuse their own live factor: after
`joint_pd_step_solve` that factor may be of a ridge-escalated matrix (the s2z
path) or absent entirely (the PSD path densifies instead), which is the same
reason `compute_skew` declines there. Factorizing the cell's own CSC -- the
bytes `store_Q` would have handed out -- is what makes the block available on
every path and identical to the pre-#307 report.

Measured (ICAR chain, `n_fixed = 8`, 40-cell grid): 868 bytes/cell retained,
flat as `n_x` goes 408 -> 6008 (`O(n_fixed^2)`, not `O(n_x)`). The removed
transient is the whole grid's precision, 41.7 / 104.2 / 218.7 KB per cell at
those sizes and linear in the cell count.

Arbiters, none of them engine code: the inverse numerical Hessian of the #300
coupled fixture's independently-written R log posterior (1.4e-09 relative); a
one-arm joint fit against the single-block fit of the same model (5e-08 on
coefficients and SEs, poisson / binomial / gaussian); the independent R
law-of-total-covariance `.joint_mixture_moments()` (8e-17); and the fit's own
posterior draws (0.1% at 200000 draws). Coverage runs through the SAME
`recov_sweep()` harness with `fit_fn = recov_fit_joint`. Tests:
`test-nested-laplace-joint-fixed-moments.R` and the joint blocks in
`test-nested-laplace-recovery.R`.

**The engine registers its own coupled likelihood (gcol33/tulpa#300, 0.0.137).**
`CellCouplingSpec` is virtual-dispatched per cell, but every genuinely
non-separable instance used to live downstream in tulpaObs, so this repo could
not reach its own coupled paths. `test_occupancy_mixture`
(`src/test_cell_coupling_occupancy_mixture.h`, registered by
`cpp_register_test_occupancy_mixture_coupling()`) is a two-arm occupancy mixture
carrying one occupancy row and J detection rows per cell: a cell with a
detection factorises, a cell with none does not, so its cross-arm and
cross-visit second derivatives are nonzero. It fills both dense cross blocks
rather than taking the rank-1 self-cross shortcut, so a third-derivative tensor
has an explicit Hessian to difference. `cpp_cell_coupling_evaluate()`
(`src/cell_coupling_probe.cpp`) drives ANY registered spec at one cell and hands
back the value, gradient, negative-Hessian diagonal and every dense cross block,
allocating buffers by the spec's own `dense_cross_pairs()` exactly as the kernel
does. The exact two-dimensional quadrature of the fixture's conditional
posterior lives in `test-inner-skew.R` section 9 (validated against the scalar
`.exact_intercept_skew()` reference, against the compiled spec cell by cell, and
for grid convergence); its marginal skewnesses (0.53 / -0.13) are the ground
truth #301 has to reproduce. Scaffolding: `tests/testthat/helper-coupled-fixture.R`.

**SPDE smoothness is general in `nu`, and `nu > 0` (gcol33/tulpa#279, #280,
#281, 0.0.122).** `SpdeQBuilder` holds the operator chain `M_0 = C`, `M_1 = G`,
`M_j = G (C^-1 G)^(j-1)` built by `init(..., alpha)`, and `rebuild(kappa, tau)`
is the binomial expansion of `Q = tau^2 K (C^-1 K)^(alpha-1)` over it -- one
loop for every integer alpha, replacing an `if (alpha == 1) ... else` whose
else-branch assembled `alpha = 2` for any higher order. The sparsity pattern is
the union of the chain's levels, so it widens with the order; `alpha = 2` is
reproduced term for term, so `nu = 1` is byte-identical. The rational assembly
shifts the `alpha = 2` stencil, so rational callers `init(..., 2)` regardless of
the fractional `nu` they approximate. The Matern axis conversion is
`spde_range_sigma_to_kappa_tau()` (`spde_qbuilder.h`), carrying `nu` in BOTH
coordinates (`tau = 1 / (sqrt(4 pi nu) kappa^nu sigma)`) and matching R's
`.spde_kappa_tau()`; `.spde_precision_Q()` (`R/marginal_se_spatial.R`) mirrors
the same expansion for the analytic marginal-SE path. Two paths stay
alpha = 2-only and now say so rather than silently downgrading: joint-hyper
NUTS (its non-centered transform differentiates that assembly) and
`cpp_spde_laplace_gradient` (its analytic `dQ/dtheta` is written term by term
for it). `nu <= 0` is refused by `.validate_spde_nu()` -- the parameterisation
is degenerate there, so the `alpha = 1` operator is unreachable from the Matern
front door and its `eps_ridge` workaround is gone.

### PD enforcement is one policy with two backends (gcol33/tulpa#344)

A coupled log posterior's negative Hessian need not be PD away from the mode --
the occupancy mixture's dark-cell term `log(psi (1-p)^J + 1 - psi)` is not
concave in `(eta_occ, eta_det)` -- so both joint Newton loops condition it
through `src/joint_pd_step.h`: `pd_lm_escalate()` (smallest diagonal load making
the factorization succeed; Nocedal & Wright, *Numerical Optimization* 2e,
Alg. 3.3) and `pd_eigen_clamp_solve()`. `joint_pd_step_solve` is the CHOLMOD
backend, `joint_pd_step_solve_dense` the dense one. With `H` already PD the
first attempt succeeds and the step IS the plain Newton step, byte-identically.
Do not write a third conditioner; the point is that the two loops meet at the
same two functions.

Until 0.0.159 only the SPARSE loop had a policy, and `pd_mode` was never passed
to the dense one -- so `control$hessian = "psd"` was inert on the dense path,
which is what every small coupled fit runs on (`n_x` below `SPARSE_THRESHOLD =
200`). A negative pivot made `sqrt` NaN, the whole step NaN, and the finite
guard added nothing: the loop reported its START VECTOR as the mode on a third
of prior-predictive draws of the engine's own coupled fixture, at any iteration
budget. The arbiter was already in the repo -- `force_sparse = TRUE` on the same
fixture converged in 7 to 9 iterations to an independently optimized mode while
the dense path returned `(0, 0)`.

**A stalled solve declines with `not_converged`, and `$modes` survives it.**
Convergence is read FIRST by every probe attach (`.inner_skew_attach_probe()`)
and by the joint fixed-block retention, because it is upstream of every other
absence those steps can trip over. `$modes` is kept -- the warm-start chain
(`init_fit$modes[1, ]` as `x_init`), adaptive refinement and local CCD all read
it -- but the inference read off it is withheld: `.fit_fixed_table()`'s
per-cell-mode average reads only cells that reached a mode, so a stalled fit
reports `NA` from `coef()` / `confint()` / `summary()` with
`interval_declined = "not_converged"` rather than its start vector as an
estimate. A non-PD Hessian at the returned point also withholds the stored
precision and the fixed block, whose inverse is not a covariance there.

### The Type-IV interaction metric, and what a diagonal one cannot do (gcol33/tulpa#585)

`mass_matrix = "gmrf"` replaces the Welford-adapted variances over a Knorr-Held
Type-IV `st_delta` block with `diag(Q^-1)` of that block's own posterior
precision,

    Q = tau (Q_s (x) Q_t) + diag(h_lik) + the two sum-to-zero margins,

evaluated at the position each warmup mass window ends on. The metric stays
DIAGONAL: `MassMatrixType::GMRF` resolves to `DIAG` plus a flag inside
`select_and_init_mass_matrix` before anything else reads the metric, so no
kinetic-energy, momentum, drift or U-turn path branches on it. `AUTO` never
selects it, per the measurement below.

`h_lik` is the per-observation eta-space curvature, reached through
`LikelihoodSpec::eta_weights_fn` -- the IRLS callback `laplace_mode_spec_dense`
drives. eta at each observation comes from `generic_eta_at()`
(`log_post_generic_impl.h`), the same assembly the observation loop uses, so a
latent component added to eta cannot be forgotten by one of the two.
`st_type_iv_precision.h` owns the matrix form of the prior, which
`tulpa_priors_st.h` owns as a quadratic form; the two are pinned to each other
by test, since only the quadratic form is evaluated by the log-posterior.

**Nothing in tulpa sets `ModelData::has_spatiotemporal`.** `spatiotemporal()`
errors at the R door, and the whole Type-IV sampler path is a consumer-package
configuration, which is why the deleted `SparseGMRFBlock` sat half-wired through
four releases with no test able to see it. `src/test_st_iv_fixture.cpp` is the
fixture that closes that: it fills a Type-IV `ModelData` directly, as a consumer
would, and exposes the layout, the override, the log-posterior and a NUTS fit.
Do NOT add to this area without a fixture case -- there is no other way in.

**The measurement says the override does not beat the adapted diagonal, and
says why.** 96 paired NUTS fits (6 configurations x 8 seeds x 2 metrics, the two
arms sharing the data and the chain seed) separate the two arms on nothing:
pooled geometric-mean ratio of leapfrog steps per effective sample 1.03
(worst parameter), 0.91 (interaction block), 1.09 (beta / log_tau), every
per-configuration sign test p > 0.47 against per-pair ratios spanning 0.18 to
9.2. `dev_notes/issue585/`.

The reason is not the sampler and needs no sampler to see. A diagonal metric can
only rescale coordinates, so what matters is `cond(Q)` after the best diagonal
rescaling. Measured on four configurations: `cond(Q)` = 1.3e5 to 2.1e5;
after the marginal rescaling `diag(Q^-1)`, 1.1e5 to 2.0e5; after Jacobi
`1/diag(Q)`, unchanged to four figures. Delete the S + T soft sum-to-zero margin
directions and the same `Q` conditions at **11.5 to 51**. The entire stiffness
is `s2z_precision(T) (I_S (x) J_T) + s2z_precision(S) (J_S (x) I_T)`, whose
eigendirections are `1_S (x) a` and `b (x) 1_T` -- not coordinate-aligned, so no
diagonal metric of any kind reaches them, and both arms sit at max treedepth.

That also says what would work, and it is measured rather than asserted: a mass
`M = D + s2z_precision(T) R'R + s2z_precision(S) C'C`, rank S + T over a
diagonal, takes the same fits to `cond` **6.8 to 18.6** -- four orders of
magnitude -- and its low-rank part is known in closed form from S and T with no
factorization and no position. That is `mass_matrix = "gmrf_margin"`, below.

The Jacobi read was built, measured (it changes the conditioning by 0.0%) and
deleted rather than shipped as a second metric name.

### The metric that reaches the margins: diagonal plus rank S + T (gcol33/tulpa#597)

`mass_matrix = "gmrf_margin"` is the #585 diagonal PLUS the block's two soft
sum-to-zero margins as an explicit low-rank term,

    M = D + lambda_row R'R + lambda_col C'C,

`R` the row-sum operator (S x ST), `C` the column-sum operator (T x ST), both
precisions fixed by S and T alone -- no position, no likelihood pass, no
factorization. It resolves to `DIAG` plus the term the same way `"gmrf"`
resolves to `DIAG` plus a flag, and `AUTO` selects neither.

**The storage is generic over GROUP SUMS, not over the Type-IV margins**
(`src/hmc_mass_lowrank.h`). Column g of `U` is the indicator of a group of
block coordinates, which is the shape EVERY soft sum-to-zero penalty in the
engine contributes: `s2z_precision(n) (sum_i phi_i)^2` on an intrinsic ICAR /
RW1 / RW2 field is ONE group over the whole block, the interaction is S + T
groups. `make_margin_mass_term()` is the two-margin builder on top of it. So
extending this past the interaction is a caller supplying groups, not new
storage -- which is the question gcol33/tulpa#597 asked to settle before the
storage was written. Whether it HELPS those blocks is not measured.

Three things the implementation gets from that shape. The inverse is Woodbury
on a k x k inner matrix `Lambda^-1 + U' D^-1 U` (k = the group count), so
`inv_mass_times_p` and `kinetic_energy` are O(n + nnz(U) + k^2) per leapfrog
step against a dense metric's O(n^2), with one k x k factorization per metric
install. That inner matrix is PD for any positive weights even though
`U' D^-1 U` is singular here (the grand total sits in every group), because
`Lambda^-1` is added to it. And the momentum draw is
`p = D^(1/2) z1 + U Lambda^(1/2) z2` with `z1`, `z2` independent, whose
covariance is `D + U Lambda U' = M` exactly -- a sum of two independent
Gaussians, so no square root of the sum is formed. #597 records that
construction as a trap to avoid; it is not one, and `test-lowrank-mass.R`
scores the realized covariance against a dense `M` to say so.

The term is an OVERLAY on the diagonal, not a fifth `MassMatrixType`: the three
per-step methods take the diagonal answer everywhere and let each term rewrite
its own block, and `set_diagonal()` / `init()` DROP the overlay, because a term
carries its own copy of the diagonal it was built against. So a fit whose term
is refused falls back to exactly the #585 metric. `apply_drift`
(`src/hmc_mass_drift.h`, split out of `hmc_nuts_optimized.cpp` so a test can
drive it) is the second place the metric meets a momentum; the two are pinned
to each other by test rather than trusted to stay in step.

**Measured on the same paired design #585 used** -- 6 configurations x 8 seeds,
arms sharing the data and the chain seed, adding an arm being a configuration
(`SWEEP585_METRICS`, `SWEEP585_ADAPT_DELTA`). At `adapt_delta = 0.95`,
leapfrog steps per effective sample, `gmrf_margin` / `diag`: pooled geometric
mean **0.0401**, 46 of 48 pairs, sign test p = 8.4e-12; raw sampling leapfrog
steps 0.0529 on **48 of 48**, p = 7.1e-15. `ess_min` RISES at the same time
(26.8 -> 83.5 on `pois_5x5_T4`), so the cheaper steps are not emptier ones.
Max-treedepth saturation disappears: the diagonal arms hit the depth-10 cap on
26% to 98% of iterations on every centered configuration and the margin arm
hits it on none. The same shard set reproduces #585's null for the `"gmrf"`
arm (pooled 1.08, p = 0.67), so the two live in one run.

**Read that arm at 0.95, not at 0.8.** At `adapt_delta = 0.8` the NON-CENTERED
configuration runs 250 divergences per 1000 iterations at an adapted step size
of 2.06, with at least one chain whose every parameter had zero draw variance;
at 0.95 it takes 1.25 against the plain diagonal's 32.6. The dual averaging
was landing on a step size the geometry does not support once `tau` moves --
non-centered puts `1 / tau` on both margin precisions while the metric is
installed once per warmup window, where the centered parameterization's are
tau-free.

**Divergences on the SMALL fixtures are the open part.** At 0.95 the two
largest configurations (96 and 100 interaction coordinates) are clean in every
arm, and the three small centered 3x3 ones are not: `pois_3x3_T4_rw2` runs 43.5
divergences per fit against the diagonal's 0.1, and gets WORSE at the tighter
target rather than better. Whether that is new pathology or newly VISIBLE
pathology is NOT settled by this sweep -- the diagonal arms on those fixtures
sit at max treedepth with `ess_min` around 10 to 16 of 1000 draws, and a chain
that does not explore a funnel does not diverge in it either. Carried as
gcol33/tulpa#598; do not read the 20x cost win as clearance to default this
metric on.
Write-up: `dev_notes/issue597/RESULTS597.md`. Tests: `test-lowrank-mass.R`
(storage and algebra against a dense `M`), `test-st-iv-margin-mass.R` (the
Type-IV wiring, the parameterization's lambda scaling, and the conditioning).

### The RW2 kernel is wider than a sum, and one predicate says so (gcol33/tulpa#600)

`st_sum_to_zero_penalty` pins the Knorr-Held interaction along its row sums and
its column sums, `S + T - 1` directions. That is EXACTLY
`null(Q_s (x) Q_t)` when the temporal marginal is RW1. Under a NON-CYCLIC RW2
one it is not:

    null(Q_s (x) Q_t) = null(Q_s) (x) R^T + R^S (x) null(Q_t),

and `null(Q_t)` gains the linear ramp, so the kernel is `T + 2S - 2` and the
`S - 1` left over -- site-specific linear time trends summing to zero across
sites -- carried NO prior curvature at all. The engine already held both
numbers on the two sides of the same prior and they disagreed: the normalizer
at `tulpa_priors_st.h:220` reads `rank_space * rank_time`, `8 * 2 = 16` against
`ST = 36` on the 3x3 / T = 4 fixture, so it was written for a 20-dimensional
kernel while the penalty beside it pinned 12.

**The precision is forced, not chosen.** `s2z_precision(n)` holds the field
MEAN -- the coefficient of the constant direction, `1' phi / 1' 1` -- at
`sd = kappa`. `s2z_precision_weighted(u'u)` holds the coefficient of an
arbitrary direction `u' phi / u' u` at the same sd, and at `u = 1` it IS
`s2z_precision(n)`. The trend family is that at `u = v`, the centred ramp, so
it is the u = 1 constant generalized rather than a second convention beside it.

`st_needs_trend_pin()` (`src/st_null_space.h`) is the ONE predicate, read by
the density (`tulpa_priors_st.h`), the sparse matrix form
(`st_type_iv_precision.h`) and the mass override (`hmc_mass_st_gmrf.cpp`)
alike. TYPE_II takes the same term by the same derivation (its kernel is
`R^S (x) null(Q_t)`); TYPE_III's is already spanned by the column sums and
TYPE_I is proper. **Cyclic RW2 gets nothing** -- a ramp is not periodic, so
`rw2_rank` reports `T - 1` there and the kernel is the constants alone.

**`hmc_mass_lowrank.h` is now generic over WEIGHTED group sums**, not indicator
groups: a trend group covers the same coordinates a row group does and differs
only in its weights, so weights are what make the family expressible at all
rather than an optimization. `group_w` empty still means unit weights, filled
by `factorize()`, so the hot loops never branch on it.

**What it cost, measured on the engine's own log posterior.**
`st_iv_num_hessian()` -- the arbiter #585 and #597 used -- gives
`cond(M^-1 Q)` under `gmrf_margin` swept over `log_tau_st`. RW1 SATURATES at
17.4, because every direction of the block eventually stiffens with `tau`. RW2
grew LINEARLY in `tau` without bound -- 13.6 / 18.6 / 75.2 / 529 / 3900 over
`log_tau` 0 to 6, a factor of 7.4 per factor of 7.4 in `tau` -- and now
saturates at 25.4. The shipped assertion could not see it: it probed ONE
position, `log_tau_st = 0.4`, where RW2 measured 18.58 against its own
`cond_lr < 100`. `test-st-iv-margin-mass.R` now asserts the RATIO across two
positions.

**This closes gcol33/tulpa#598's one unexplained row.** On the same harness,
fixture, seeds and settings, `pois_3x3_T4_rw2`'s divergences fall from 43.5 to
8.4 per fit and, per EFFECTIVE sample -- the normalization that removes
"diverges because it moves" -- from 5.48 to 0.627 against the adapted dense
metric's 0.691. The 20x that "exploration alone does not account for" is gone.
What remains is the `tau` funnel #598 section 1 established as real geometry
(`d(log_tau) = +1.92`, positive on 6 of 6 fits), present in every arm. An RW1
configuration is unchanged BIT FOR BIT across the fix, all three arms, which is
what says the predicate is an exact no-op where it answers false.

**`AUTO` still does not select `gmrf_margin`.** One configuration was
re-measured, and the small fixtures still reach `ess_min` in the teens out of
1000 draws in every arm. Write-up: `dev_notes/issue600/RESULTS600.md`.

### Checkpoint / resume across every fitter (gcol33/tulpa#50)

Every fitter with an outer loop of independent, expensive units supports
checkpoint/resume: a killed or rebooted fit reloads the completed units and
runs only the rest. The mechanism is one content-addressed binary append log
(`src/checkpoint_io.h`, `CheckpointLog<Payload>`): magic + fingerprint header,
then per-unit `key + payload + FNV-1a checksum` records. A torn final record
(killed mid-write) is detected on load (short read / checksum mismatch),
**truncated** to the last valid record, and re-run; a header fingerprint
mismatch (different data / settings / grid) **errors** rather than resuming onto
a stale result. `CheckpointLog<Payload>` is generic over the per-unit payload
via two ADL customization points (`ckpt_serialize` / `ckpt_deserialize`), so the
file format, load/append/torn-tail logic, and fingerprinting live once.

Two specializations:
- **`GridCheckpoint` = `CheckpointLog<LaplaceResult>`** (`nested_laplace_checkpoint.h`):
  the unit is an outer grid cell, keyed by its hyperparameter coordinate (the
  `theta_grid` row plus any per-arm phi), so adaptive-refinement cells append
  under their own keys and resume is order-independent. Wired through the joint
  driver, the shared `run_multi_block_nested_laplace` (every single-block
  kernel: icar / bym2 / car_proper / temporal / the ST variants / nngp / hsgp),
  `cpp_nested_laplace_multi`, and the sparse driver (`fit_spde`). `make_nl_grid_checkpoint()`
  is the single-arm kernel factory (a per-kernel structural seed + the shared
  obs inputs + the grid axes). The structural seed comes from
  `NlFieldIdentity` (same header): one named method per structural group --
  `areal()` (optional BYM2 mixing scale), `nngp()`, `hsgp()`, `temporal()`
  (optional panel group count) -- chained by each `cpp_nested_laplace_*` entry
  in the order the groups contribute. Fold order IS the fingerprint, so adding
  a group means adding a method rather than copying byte-fold loops, and
  `test-nl-field-identity.R` holds every field model's seed against the folds
  written by hand before the extraction. The RE-covariance CCD path
  (`tulpa_re_cov_nested`) uses a small R-level analogue (atomic-RDS node cache
  keyed by node index, the CCD grid being deterministic given the fingerprint).
- **`ChainCheckpoint` = `CheckpointLog<HMCResultCpp>`** (`hmc_chain_checkpoint.h`):
  the unit is a whole MCMC chain. A chain is deterministic in
  `(seed, chain_id, data, settings)`, so a resumed chain is **bit-for-bit**
  identical to the uninterrupted one -- no mid-trajectory RNG / dual-averaging /
  metric state to restore. Wired into `run_hmc_parallel_chains_cpp` (completed
  chains load serially, the parallel loop skips them and `save()`s the rest under
  a mutex) and exposed on `cpp_tulpa_fit_generic_chains(checkpoint_path=)`.

R surface: `control$checkpoint = list(path =, resume =)` on the nested-Laplace
fitters (`.nl_checkpoint_args()` parses it; the front door fresh-deletes on
`resume = FALSE` so within-fit calls append, and the k-hat diagnostic
re-evaluations run with it stripped so they do not pollute the file); a
`checkpoint = ` arg on `fit_spde()` / `tulpa_re_cov_nested()`; a
`checkpoint_path = ` arg on the NUTS producer. Tests:
`test-nested-laplace-joint-checkpoint.R` (joint) and
`test-checkpoint-universal.R` (single-block, RE-cov, per-chain NUTS:
equivalence, resume-loads-nothing, torn-tail re-solve, fingerprint mismatch).

### Matrix CHOLMOD Fix

tulpa's `R_init_tulpa` calls `M_cholmod_start` which requires Matrix's
CHOLMOD stubs. Fixed by adding `@importFrom Matrix sparseMatrix` to
`tulpa-package.R` so Matrix DLL loads before tulpa's init.

## Extensibility: Custom Latent Blocks

This shipped. The real API is `tgmrf()` (R closures for `Q`/`mu`/`log_prior`)
and `tgmrf_cpp()` (a user `.cpp` compiled via `sourceCpp` against
`inst/include/tulpa/`, keyed by SHA256 + ABI), consumed as a block through
`latent()` and fit at any tier via `tulpa_tgmrf(mode = "imh"/"nuts"/"vi"/"nuts_joint")`
or on the nested-Laplace front door via `latent()`. See `?tgmrf_cpp`,
`vignettes/tgmrf.Rmd`, and `inst/examples/`. The names below
(`tulpa_custom_latent()`, `tulpa_fit(..., tier=)`) are the original sketch and
never shipped under those spellings.

For latent structures not provided by the engine (custom GMRFs, novel
spatial priors, exotic temporal kernels), users supply a templated C++
snippet that tulpa compiles on-the-fly. Same machinery as `LikelihoodSpec`
+ `LinkingTo: tulpa`, but with an ad-hoc entry point — no full model
package required.

User writes templated C++ that compiles against tulpa's AD types:

```cpp
template <typename T>
Eigen::SparseMatrix<T> my_Q(const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta);

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> my_mu(const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta);

template <typename T>
T my_log_prior(const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta);
```

User binds it in R:

```r
custom <- tulpa_custom_latent(
  cpp_file   = "my_block.cpp",
  theta_init = c(1, 1),
  graph      = my_graph
)

fit <- tulpa_fit(y ~ x + latent(custom), data = d, tier = "laplace")
```

tulpa compiles via `Rcpp::sourceCpp` against `inst/include/tulpa/`,
registers the block in the latent-structure registry, and inference
layers pick it up automatically. Because the user code compiles with
tulpa's templated AD types (`A`, `A_r`), the block works under **any
tier including NUTS** — no R callback, no broken gradient chain.

Comparison:
- INLA `rgeneric`: R callback, no AD, Laplace tier only.
- INLA `cgeneric`: C function, no AD, faster but no exact-MCMC support.
- Stan: full DSL + parser + codegen for the entire model.
- TMB: templated C++ snippet, autodiff via CppAD — closest analog.

Cost: extends existing `LinkingTo: tulpa` machinery with a `sourceCpp`-
driven entry point. No DSL, no parser, no codegen.

## Origin

Engine extracted from numdenom (82K lines, faster than Stan on all 18
benchmarks); numdenom was then renamed tulpaRatio as the engine became the
hub of the `tulpa*` ecosystem.
Name: Twin Peaks reference + acronym (Template Unified Latent Process Architecture).
