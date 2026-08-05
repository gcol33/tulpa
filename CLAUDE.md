# TULPA — Template Unified Latent Process Architecture

General-purpose Bayesian hierarchical modelling engine (v0.0.99). Engine
extracted from numdenom, which has since been renamed tulpaRatio.

## Architecture

The hub of a `tulpa*` package ecosystem. The engine owns inference, latent
structure, and the C++ interface; model packages plug an observation
likelihood in via `LikelihoodSpec` and inherit the rest.

- **tulpa** (engine, 0.0.99) — samplers, autodiff, spatial, temporal, priors, formula infrastructure. Imports tulpaMesh for SPDE mesh construction.
- **tulpaRatio** (1.3.0) — ratio, rate, and proportion models (renamed from numdenom).
- **tulpaObs** (0.0.22) — occupancy, N-mixture, and detection models.
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
5. **No copy-paste logic**: Shared sub-computations in helpers, not duplicated across specialized functions. Conventions that keep this single-sourced: log-prior helpers are named `log_prior_*` (e.g. `log_prior_car_proper`, `log_prior_sigma2_pc`); the column-major Rcpp matrix builder `build_matrix_colmajor` is one template over the element type; the spatially-/temporally-varying-coefficient `print`/`summary` methods delegate to `.print_varying_coef` / `.summary_varying_coef` in `R/varying_coef.R`; and multi-block prior detection is the single `.is_multi_block_prior` predicate.
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

## Versioning

The patch number keeps counting past 9: `0.0.99` -> `0.0.100` -> `0.0.101`.
`0.1.0` is reserved for the first stable CRAN release, so it is never a routine
bump.

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

## Boundary: What Belongs in tulpa vs Model Packages

**tulpa owns** (generic, model-agnostic):
- Inference engines: Laplace, EM+Laplace, VI, ESS, NUTS, MI correction, Gibbs correction
- Autodiff: arena, forward, tape
- Latent structure: spatial, temporal, RE, SVC, TVC, ST, latent factors
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
(default 200, each one extra inner Laplace solve); computed after the draw
synthesis with the RNG restored, so draws are bit-for-bit unchanged. NOTE:
small-group binary RE-covariance posteriors are genuinely skewed, so a high
k-hat there is a correct signal, not a defect.

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
support transform. Same `control$diagnose_k` (default TRUE) / `k_samples` (200)
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
`k_samples` (200), RNG-restored. The parallel-NUTS
multi-chain producer (`run_hmc_parallel_chains_cpp`, exposed via
`cpp_tulpa_fit_generic_chains`) emits the `(draws, chain_id, n_chains)` layout
`.tulpa_chain_list()` reads, verified end-to-end against `posterior` in
`tests/testthat/test-convergence.R` and on a native multi-chain fit in
`tests/testthat/test-generic-sampler.R` ("mcmc_diagnostics consumes a native
multi-chain fit").

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
leading-order undershoot as skewness grows). Only the skewness term ships;
the paper's `gamma_1` (location shift) and a kurtosis term are NOT
attempted -- see the scope note atop `inner_laplace_skew.h` for why (the
denominator log-determinant response the location-shift term needs is
diagonal only in the paper's augmented representation, and the paper itself
routes heavy-tailed cases to a different numerical procedure rather than a
closed-form quartic).

Per-observation third-log-lik-derivatives come from
`curvature3_obs_for_family()` (`src/laplace_family_curvature.h`, exact
analytic ladder for built-in families) or a central finite difference on the
Newton working weight for a consumer-package `LikelihoodSpec`
(`build_spec_curvature3_fn`, `src/laplace_spec_curvature3.h`). Both decline
(empty oracle) for a `LikelihoodSpec` with `n_processes != 1` -- a coupled
multi-process likelihood (ZI, tulpaObs's `occu_cover`) has no single per-obs
term this formula scores. The joint multi-arm loops generalize the same sum
across arms (`build_joint_curvature3_fns`,
`src/laplace_newton_joint.h`) -- sound because tulpa's own production
coupling registry only ever registers `"separable"`; a genuinely coupled arm
(`skip_arm[k]`) is excluded from the sum, not silently scored against its
unused per-obs likelihood. **Every decline path returns NaN, never a
silently-wrong `0`** ("perfectly Gaussian") -- `compute_inner_skew_gamma3[_joint]`
early-returns all-NaN when the oracle is entirely absent, and per-index
only assigns a value when at least one finite contribution accumulated
(the pre-existing bug this fixed: an absent oracle summed to `acc = 0`,
`0 / sigma_i^3 == 0`, read as "no skew" instead of "not computable").

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
(gcol33/tulpa#277, 0.0.121).** `cpp_laplace_fit_gp` and `cpp_laplace_fit_spde`
used to carry their own Newton loops (`laplace_mode_gp()`,
`spde_run_single_fit()`) -- an independent implementation the joint-multi
driver never touched, so #273's `gamma_3` pass had to be wired into them
separately. Each is now a thin wrapper: `make_single_arm()`
(`nested_laplace_joint_core.h`) + the same block factory the nested entry uses
(`make_nngp_block` / `make_spde_block`) + the joint driver at a one-row grid,
projected back onto the single-fit contract by `nl_grid_cell_to_result_list()`
(`nested_laplace_grid.h`, reading the per-cell `log_det_Q` / `score_max` /
`converged` the grid driver now reports). The equivalence is EXACT and asserted
at `tolerance = 0` in `test-laplace-spatial-gp-spde-equiv.R`; anything the
joint-multi driver gains (`gamma_3` included) is inherited, not wired.

Two things the migration settled, both load-bearing:

- **NNGP is sparse-only.** `make_nngp_block` scatters its prior into the sparse
  builder alone, and the dense route disagrees with it -- measurably at
  `nn = 5`, and at `nn = 8` a 300-iteration non-convergence returning
  `log_marginal = NaN` against a 23-iteration convergence. `blocks_require_sparse()`
  (`latent_block.h`) reads that off the blocks: a non-`INDEXED_SINGLE`
  contribution OR a prior with only `add_prior_sparse` forces the sparse path.
  That also closes the silent case where the dense path calls an absent
  `add_prior` and contributes nothing. NNGP is the only block whose dispatch
  this changes (MCAR / HSGP-MO / latent factor are already non-`INDEXED_SINGLE`);
  the ST NNGP entry had been passing `force_sparse = true` by hand for it.
- **Post-loop centring must compensate the intercept.** `center_effects_fn` runs
  ONCE after the Newton loop. The single-arm loop centres and then re-evaluates
  `log_marginal` / `H` at the shifted point; the joint loop computes
  `log_marginal` first and centres with a `CenterFold` into the arm intercept,
  so `eta` is preserved. The bespoke SPDE fit took the first route without the
  fold and reported a converged mode whose fixed-effect score was ~0.47.

`cpp_laplace_fit_spde_precomputed` (the rational/fractional-nu path) still has
its own Newton loop via `spde_run_single_fit()` -> `laplace_newton_solve_sparse()`,
which is why that function survives #277's checklist. Its `Q`/`Aeff` arrive
already assembled from `.spde_rational_assemble` (`R/brasil.R`), so following the
others needs `make_spde_block` to accept a precomputed CSC `Q` with `prep()` a
no-op and centring off. Also still bespoke: `implicit_diff.cpp`'s
`cpp_spde_laplace_gradient`, which calls `run_spde_laplace` directly for the
`SpdeQBuilder`'s per-entry `c0_contrib`/`g1_contrib` decomposition and a raw
`SparseCholeskySolver&` for Takahashi selected inversion -- internals the
driver's `Rcpp::List`-only interface does not expose.

`unwrap_skew_idx()` (`laplace_spec_fit.h`) remains the shared
1-based-R-to-0-based-probe conversion. The coupled (non-separable) cubic-term
derivation (#273 item 2) remains open.

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
  obs inputs + the grid axes). The RE-covariance CCD path
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
