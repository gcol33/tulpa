# TULPA — Template Unified Latent Process Architecture

General-purpose Bayesian hierarchical modelling engine (v0.0.2). Engine
extracted from numdenom, which has since been renamed tulpaRatio.

## Architecture

The hub of a `tulpa*` package ecosystem. The engine owns inference, latent
structure, and the C++ interface; model packages plug an observation
likelihood in via `LikelihoodSpec` and inherit the rest.

- **tulpa** (engine, 0.0.2) — samplers, autodiff, spatial, temporal, priors, formula infrastructure. Imports tulpaMesh for SPDE mesh construction.
- **tulpaRatio** (1.3.0) — ratio, rate, and proportion models (renamed from numdenom).
- **tulpaObs** (0.0.1) — occupancy, N-mixture, and detection models.
- **tulpaGlmm** (0.0.0.9000) — generalized linear mixed models.
- **tulpaMesh** (0.1.1) — constrained Delaunay meshes for SPDE fields (an engine dependency, not a consumer).

Consumers depend on the engine via `Imports: tulpa (>= 0.0.2)` +
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
5. **No copy-paste logic**: Shared sub-computations in helpers, not duplicated across specialized functions.
6. **Statistical args vs `control` knobs**: front-door fitters (`tulpa()`, `tulpa_nested_laplace()`, `tulpa_nested_laplace_joint()`) carry only statistical arguments in their signature; all perf / numerical / tuning knobs live in a single `control = list()` (e.g. `control$re_cov`, `n_threads`, `max_iter`, `tol`, `adaptive_grid`, `prune`, ...). Pre-release: no deprecation shims -- moved knobs hard-error.

## C++ Interface

- Exported headers: `inst/include/tulpa/`
- Model packages use `LinkingTo: tulpa` in DESCRIPTION
- Key types: `ModelData`, `ParamLayout`, `LikelihoodSpec`, `LikelihoodFn<T>`, `ResidualFn`
- **ABI version**: `TULPA_ABI_VERSION` in `model_data.h`. Bump when any exported struct layout
  changes. Model packages auto-check on first NUTS call — mismatch gives a clear error instead
  of segfault. Registered via `R_RegisterCCallable("tulpa", "tulpa_get_abi_version", ...)`.
- **Legacy fields**: `ModelData` contains deprecated ratio-specific fields (y_num, y_denom,
  X_num/denom_flat, model_type). Used only when `n_processes == 0`. Will move to tulpaRatio.
  New fields go in the stable sections — never insert before existing fields.

## Building

```r
devtools::load_all()
devtools::check(args = "--no-manual")
```

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

## File Organization

```
R/          — Spatial, temporal, priors, backends, formula, validation
src/        — Laplace, VI, ESS, HMC/NUTS, autodiff, spatial priors, temporal priors
inst/include/tulpa/ — Exported C++ headers for model packages
tests/testthat/     — Unit and integration tests
```

## Boundary: What Belongs in tulpa vs Model Packages

**tulpa owns** (generic, model-agnostic):
- Inference engines: Laplace, EM+Laplace, VI, ESS, NUTS, MI correction, Gibbs correction
- Autodiff: arena, forward, tape
- Latent structure: spatial, temporal, RE, SVC, TVC, ST, latent factors
- Generic S3 methods operating on posterior draws: coef, confint, vcov, logLik, summary
- Generic diagnostics: moran_i, durbin_watson, tulpa_variogram, compare_models, modelAverage
- Generic plotting: trace, density, pairs plots of posterior draws
- Rubin's rules pooling
- Parameter back-transformation (logit → probability)

**Model packages own** (e.g., tulpaObs, tulpaRatio):
- Likelihood functions (LikelihoodSpec)
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
matrix for one block, a named list for several. Tests: `test-re-cov-nested.R`,
`test-re-cov-gibbs.R`, `test-re-cov-recovery.R`, `test-re-cov-prior.R` (Jacobian
vs finite differences, diagonal + joint priors), `test-ccd-grid.R`,
`test-tulpa-re-cov-frontdoor.R` (single, diagonal, multi-term routing).
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
(`compare_models`, `modelAverage`, `spatialRange`, `temporalCorr`), and
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
untagged fits still work. On a non-chain fit `mcmc_diagnostics()` returns `NULL`
with a message (Rhat is vacuous and ESS = n_draws by construction there),
`check_diagnostics()` returns `NA` ("not applicable"), and the plot/summary
layer withholds the panels rather than printing a vacuous convergence pass.
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
k-hat there is a correct signal, not a defect. `diagnostic_summary()` surfaces
`pareto_k` for these fits, falling back to the grid's quadrature effective
sample size (`sum(w)^2 / sum(w^2)`) for the C++ nested paths
(`tulpa_nested_laplace`, joint, spde) that store weights but have no sampled
k-hat yet -- a sampled k-hat there needs a C++ inner-marginal entry point and
is a follow-on. The parallel-NUTS
multi-chain producer (`run_hmc_parallel_chains_cpp`, exposed via
`cpp_tulpa_fit_generic_chains`) emits the `(draws, chain_id, n_chains)` layout
`.tulpa_chain_list()` reads, verified end-to-end against `posterior` in
`tests/testthat/test-convergence.R` and on a native multi-chain fit in
`tests/testthat/test-generic-sampler.R` ("mcmc_diagnostics consumes a native
multi-chain fit").

### Matrix CHOLMOD Fix

tulpa's `R_init_tulpa` calls `M_cholmod_start` which requires Matrix's
CHOLMOD stubs. Fixed by adding `@importFrom Matrix sparseMatrix` to
`tulpa-package.R` so Matrix DLL loads before tulpa's init.

## Extensibility: Custom Latent Blocks (design sketch)

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
