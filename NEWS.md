# tulpa NEWS

## 0.0.2

* feat: `tulpa_re_aghq()` -- a callback-driven adaptive Gauss-Hermite refinement
  of a grouped random-effect covariance. Generalizes `agq_fit()` (intercept-only
  RE, built-in `binomial`/`poisson`/`gaussian`) to **random slopes and
  correlated multi-coefficient blocks** sharing one grouping factor, with the
  per-observation marginal likelihood supplied by the caller through a
  `make_site` callback. This lets a downstream package refine a custom marginal
  (e.g. a latent-state-integrated occupancy / detection likelihood) through the
  same quadrature. Reuses the existing log-Cholesky covariance parametrization
  (`.re_cov_*`), `gauss_hermite_prob()`, and an optional LKJ correlation penalty;
  the fixed parameters and `chol(Sigma)` are optimized jointly on the
  exact-marginal log-likelihood, with SEs from the marginal Hessian. Recovery
  tests in `tests/testthat/test-re-aghq.R`.

## Unreleased

* refactor(laplace)!: removed the 8 dead family-enum single-point Laplace
  C-callables (`tulpa_laplace_mode_{dense,spatial,dense_multi_re,bym2,gp,
  multiscale_gp,multiscale_temporal,rsr}`) and their `LaplaceMode*Fn` typedefs.
  No package consumes them -- every model package routes single-point Laplace
  through the `LikelihoodSpec` path (`tulpa_laplace_spec_*`). **ABI break**
  (`TULPA_ABI_VERSION` 25 -> 26); downstream packages must rebuild. The shared
  `LaplaceShimResult` POD is retained (reused by the spec shims).

* refactor(nested-laplace)!: collapsed the 15 spatio-temporal nested-Laplace
  entries (`*_st_<spatial>_<temporal>`) to 5 per-spatial-family entries
  (`*_st_{icar,car_proper,bym2,hsgp,nngp}`) that select the temporal kernel
  (rw1 / rw2 / ar1) at runtime via a `temporal_type` argument, dispatched
  through a single `make_temporal_ops` registry. Adding a temporal kernel is
  now O(1) -- one registry branch, no new cross-product function at any layer
  (Rcpp entry, extern-C shim, or exported ABI). **ABI break** (`TULPA_ABI_VERSION`
  24 -> 25): the 15 `tulpa_nested_laplace_st_*` registered callables + their
  `NestedLaplaceSt*Fn` typedefs became 5; downstream packages must rebuild.
  Dense/sparse per-kernel equivalence preserved (`test-nested-laplace-st-sparse-equivalence.R`).

* feat(nmix): `tulpa_nmix_laplace()` gains `mixture = c("P", "NB")` -- a
  negative-binomial abundance mixing distribution
  (`N_i ~ NegBin(mean = lambda_i, size = r)`, `neg_binomial_2` convention) in
  addition to the Royle (2004) Poisson kernel. The per-site marginal, its
  scores (including the analytic dispersion score `d log L / d log r`), and the
  full joint observed-information Hessian are closed form; the dispersion
  `log_r` is profiled by block coordinate ascent outside the inner beta-Newton
  and reported with its standard error in `vcov`. Matches
  `unmarked::pcount(mixture = "NB")` on coefficients, log-likelihood, and
  standard errors to machine precision, with the usual analytic-derivative
  speed advantage. Poisson remains the default and is unchanged.

* feat(nmix): the spatial nested-Laplace N-mixture fits
  (`tulpa_nmix_laplace_icar()`, `tulpa_nmix_laplace_car_proper()`,
  `tulpa_nmix_laplace_bym2()`) gain `mixture = "NB"`. The NB size `r` is
  integrated as an additional outer grid dimension alongside the spatial
  hyperparameters (`tau` / `rho` / `sigma`); the posterior `r_mean` / `r_sd`
  are reported from the grid weights. The inner `(beta, z)` / `(beta, v, w)`
  Newton is unchanged in dimension -- only the likelihood pieces and the
  NB-aware `Var[N|y]` rank-1 correction depend on `r`. Poisson remains the
  default with identical behaviour and grid shape.

* refactor(api): `tulpa_nested_laplace()` and `tulpa_nested_laplace_joint()`
  collapse their perf/numerical knobs into a single `control = list()` argument,
  matching `tulpa()`. The top-level signatures now carry only statistical
  arguments (`y`/`n_trials`/`X`/`prior`/`spec`/`family`/`phi`/`likelihood`/...;
  `responses`/`prior`/`copy`/`phi_grid`/`prior_sigma`/`prior_alpha`). Tuning
  knobs move into `control`: single-arm `max_iter`, `tol`, `n_threads`, `x_init`,
  `keep_grid_hessians`; joint additionally `n_threads_outer`, `tile_warm`,
  `prune`, `prune_tol`, `store_Q`, `adaptive_grid`,
  `adaptive_grid_edge_thresh`, `adaptive_grid_max_passes`,
  `var_of_means_consistency`, `force_sparse`, `verbose`. Pre-release breaking
  change -- pass these inside `control = list(...)` (no deprecation shim). The
  dead single-arm `verbose` knob was dropped. Internal callers (`em_laplace`,
  the tgmrf pilots) and the shipped examples were migrated.

* feat(laplace): `tulpa_laplace_beta()` gains a `beta_prior` argument, forwarded
  to the inner `tulpa_laplace()` fits (both the outer `phi` search and the final
  refit) so the beta arm can carry a Gaussian fixed-effect penalty. Pure R
  passthrough -- `tulpa_laplace()` already applied `beta_prior` for
  `family = "beta"`. Enables penalised beta-regression arms downstream
  (tulpaObs `cover_priors()` positive arm). Rejected with `spatial`, matching
  `tulpa_laplace()`.

* feat(laplace): correlated random slopes `(1 + x | g)` on the Laplace engine
  (gcol33/tulpa#28). `tulpa_laplace()` RE terms accept a per-term covariance via
  `L` (lower-triangular Cholesky, `Sigma = L L'`) or `cov`; the off-diagonal now
  enters both the joint Hessian (mode finding) and the marginal fixed-effect SE.
  Previously a multi-coefficient RE block could only carry a per-coefficient
  marginal-sigma vector (a diagonal covariance), so `(1 + x | g)` was
  inexpressible under Laplace and downstream packages routed it to NUTS. The C++
  multi-RE kernel already consumed a packed Cholesky; this wires the R API to it
  through a single `.re_cov_spec()` helper and rebuilds the marginal Schur
  complement with a block-diagonal precision. It also fixes a pre-existing bug
  in the marginal-SE linear predictor -- the eta reconstruction treated every RE
  term as intercept-only, so the returned `H_beta` silently ignored random
  slopes (this affected `(x || g)` as well). Validated against an independent
  full-precision Schur in `tests/testthat/test-laplace-corr-re.R`. Estimating the
  covariance itself (the EM M-step for a full `Sigma`) is the follow-up; the
  engine now fits correlated slopes at a supplied covariance.

* feat(laplace): `tulpa_laplace(return_re_cov = TRUE)` returns per-group
  posterior covariance blocks `cov_blocks` -- one `n_coefs x n_coefs` matrix per
  (RE term, group) in term-major then group order, each a diagonal block of the
  *full* inverse Hessian (fixed effects and other groups marginalized out), i.e.
  `Cov(u_g | y, Sigma)`, not the inverse of a diagonal block. Built by reusing
  the Cholesky factor from the log-determinant (one back-solve per block column,
  no refactorization). This is the primitive a full-covariance EM M-step
  consumes to update `Sigma_k <- mean_g [u_g u_g' + Cov(u_g)]`; tulpaObs#11 uses
  it to fit `(1 + x | g)` deterministically. Non-spatial multi-RE path only
  (rejected with `spatial`).

* feat(nuts): expose tulpa's across-chain OpenMP runner through the model-facing
  C ABI (gcol33/tulpa#30). New registered callable `tulpa_run_nuts_chains`
  (header accessor `tulpa::get_nuts_chains_fn()`) runs `n_chains` chains in one
  call and fills a caller-allocated array of `NUTSResult`, so downstream packages
  stop re-implementing chain orchestration (offset-seed loops / PSOCK clusters)
  in R and get the engine's thread-parallel path for free. `init` and the
  optional `inv_metric_diag` are chain-major `[n_chains * n_params]`, so a fresh
  fit broadcasts one init while a resume passes each chain's `final_position` +
  `inv_metric_out` (with `n_warmup = 0`) — composing with #29 to continue a whole
  multi-chain fit. The OpenMP loop now lives in one pure-C++ core
  (`run_hmc_parallel_chains_cpp`) shared by the C ABI and the existing
  Rcpp-returning `run_hmc_parallel_chains`. New generic R entry point
  `cpp_tulpa_fit_generic_chains()` returns draws stacked chain-major with a
  `chain_id` vector — the layout `mcmc_diagnostics()` (#26) consumes directly —
  plus per-chain `epsilon` / `inv_metric` / `final_position`. Validated in
  `tests/testthat/test-generic-sampler.R`, including a cross-chain Rhat/ESS check
  through `mcmc_diagnostics()`. **ABI bump 23 -> 24** (new callable only; no
  struct layout change).

* feat(nuts): the NUTS C-ABI now returns the state needed to resume or
  warm-start a chain (gcol33/tulpa#29). `NUTSResult` gains `inv_metric_out`
  (the adapted inverse-mass diagonal at end of warmup) and `final_position`
  (the last raw sampler state); `epsilon` was already returned. Feeding them
  back as `init` + `inv_metric_diag` with `n_warmup = 0` continues the chain
  from the previous fit's geometry instead of rediscovering it. The inputs
  already existed on `NUTSFn`; only the result fields were missing. The
  generic R entry point `cpp_tulpa_fit_generic()` gains optional `init` /
  `inv_metric_init` arguments and returns `inv_metric` / `final_position`,
  exercised in `tests/testthat/test-generic-sampler.R`. **ABI bump 22 -> 23**
  (two trailing pointers appended to `NUTSResult`; `NUTSFn` unchanged).

* feat(diagnostics): extend the native MCMC convergence surface
  (`R/convergence.R`) toward posterior parity (gcol33/tulpa#26).
  `mcmc_diagnostics()` gains `measures` and `probs` arguments selecting from
  improved `rhat` (now the maximum of rank-normalized split-Rhat and folded
  split-Rhat, matching `posterior::rhat`), `rhat_bulk`, `rhat_fold`,
  `ess_bulk`, `ess_tail`, `ess_mean`, `ess_sd`, `mcse_mean`, `mcse_sd`, and
  per-probability `ess_quantile` / `mcse_quantile`. The default columns
  (`rhat`, `ess_bulk`, `ess_tail`) are unchanged. New measures are registered
  in one table (`.tulpa_diag_measures`), so adding a statistic is a one-line
  change. Two estimator bugs are fixed so the native code reproduces
  `posterior` to machine precision (~1e-12): the rank-normalization divisor is
  now the Blom `S + 1/4` (was `S - 1/4`) and the Geyer `tau_hat` tail term no
  longer double-counts. New exported helpers: `tulpa_draws_array()` (an
  `as_draws_array()`-style `[iter, chain, param]` accessor), `n_divergent()`,
  and `check_diagnostics()`. The plotting / summary layer (`plot_rhat`,
  `plot_ess`, `diagnostic_summary`, `plot_diagnostics`, `plot_acf`,
  `plot_pairs`) now resolves the previously undefined `get_draws_array()`,
  `grep_params()`, and `n_divergent()` helpers and runs end-to-end on a
  multi-chain fit. Validated in `tests/testthat/test-convergence.R`.

* feat(re): random-effect blocks support an optional per-term intercept.
  `ModelData::re_has_intercept` (default: all terms carry the implicit group
  intercept) lets a term be slope-only (lme4 `(0 + x | g)`): every
  `re_n_coefs[t]` coefficient is a slope read from the slope design matrix and
  there is no `z = 1` column. The change is threaded through the design lookup
  (`slope_at()` / `obs_re_contrib()` in `laplace_spec.cpp`) and the autodiff
  RE contribution (`log_post_generic_impl.h`), so the value and its gradient
  stay consistent. `re_term_has_intercept()` (in `model_data.h`) centralises
  the per-term test. **ABI bump 21 -> 22** (new `ModelData` field). Enables
  gcol33/tulpaObs#10 slope-only bar syntax.

* fix(spatial): `spatial_car()` / `spatial_bym2()` /
  `spatial_car_proper()` with `level = "group"` now accept datasets that
  cover only a subset of adjacency cells. Closes gcol33/tulpa#25. The
  Besag / ICAR / BYM2 / proper-CAR field is well-defined on every node
  of the graph regardless of whether each node has an observation;
  unobserved cells simply contribute no likelihood term (matching
  INLA's `f(cell, model = "besag", graph = g)`). `validate_spatial()`
  and `prior_from_spec()` now resolve `group_var` to 1-based adjacency
  row indices via a new `.resolve_spatial_idx()` helper:
  - integer / numeric `group_var` -> 1-based row indices,
    validated against `[1, n_spatial_units]`.
  - character / factor `group_var` with `rownames(adjacency)` set ->
    matched by name (preserves cell identity for sparse subsets).
  - character / factor `group_var` without rownames -> legacy
    `as.integer(as.factor(.))`, retained for back-compat; errors with
    an actionable message when level count differs from adjacency
    size.

* refactor(joint-laplace): unify single-block and multi-block joint
  dispatch (Phase J-E). `tulpa_nested_laplace_joint()`'s single-block
  path (`prior = list(type = "bym2"/"icar"/"car_proper", ...)`) now
  packs the prior into a length-1 `blocks_spec` and dispatches through
  the same `cpp_nested_laplace_joint_multi` entry that drives the
  list-of-blocks API. The three legacy `cpp_nested_laplace_joint_bym2`
  / `_icar` / `_car_proper` R-facing wrappers (524 lines in
  `src/nested_laplace_joint.cpp`) are deleted; the inner Newton driver
  (`run_multi_block_nested_laplace_joint`) was already shared, so the
  refactor is a routing change with bit-identical log_marginal on every
  joint test (147/147 pass). User-facing R API and result shape are
  unchanged. **C ABI:** the `tulpa_nested_laplace_joint_bym2` shim
  (`R_RegisterCCallable` entry in `tulpa_shims.cpp`) is removed;
  external embedders should call `tulpa_nested_laplace_joint()` from
  R or build a shim on top of `cpp_nested_laplace_joint_multi`.
  `TULPA_ABI_VERSION` bumped **19 → 20**.

* fix: joint nested-Laplace reparam `(sigma, alpha)` → `(sigma_occ, sigma_pos)`.
  Closes gcol33/tulpa#18. The BYM2 / ICAR / CAR_proper backends of
  `tulpa_nested_laplace_joint()` previously parameterized the joint
  outer grid as `(sigma, rho/rho_car, alpha)`, where `sigma` was the
  shared field amplitude and `alpha` scaled the copy arm's contribution.
  At small `n_pos` and low cover-arm sample fraction (e.g. d7 Cell B,
  `n_s = 25`, `n_pos ≈ 46`), the cover-arm likelihood identified only the
  *product* `alpha * sigma`; sigma was pulled toward its prior and alpha
  inflated to compensate (~ −15% sigma bias, ~ +27% alpha bias on 30
  seeds). The reparam scales each arm's contribution to a unit-precision
  latent by its own sigma — `eta_arm = X beta + sigma_arm * z_s`, with
  `sigma_arm = sigma_occ` on donor arms and `sigma_arm = sigma_pos` on
  the copy arm. Each axis is now anchored by its own arm's likelihood,
  so the posterior ridge along constant `alpha * sigma` disappears.
  `alpha = sigma_pos / sigma_occ` is recovered post-hoc and attached to
  `theta_grid` / `theta_mean` / `theta_sd`. **API:** `copy$alpha_grid` is
  superseded by `copy$sigma_pos_grid`; `alpha_grid` still works with a
  deprecation warning that translates it to
  `alpha_grid * median(prior$sigma_grid)`. ICAR / CAR_proper joint
  kernels now take `sigma_grid` (donor sigma in sigma-space) instead of
  `tau_grid`; tulpaObs callers translate `tau_grid` to
  `sigma_grid = 1/sqrt(tau_grid)` internally. **C ABI:**
  `tulpa_nested_laplace_joint_bym2_impl` switches its grid args from
  `sigma_spatial_grid` / `alpha_grid` to `sigma_occ_grid` /
  `sigma_pos_grid`. `TULPA_ABI_VERSION` bumped **18 → 19**.

* feat: EM+Laplace MI and Gibbs corrections. `tulpa_em_laplace()` gains two
  post-EM correction modes (`correction = "mi"` / `"gibbs"`) that replace
  the previous "not yet implemented" stub. MI draws `n_imputations` hard
  `z`'s from the converged posterior weights `P(z|y, theta_hat)`, refits
  each block on the hard draws, and pools per-submodel coefficients via
  `rubins_pool()`. Gibbs runs a warm-started `z|theta -> theta|z` Markov
  chain of length `n_gibbs` starting from the EM fits — every step
  refreshes weights via the user's `e_step`, draws hard z, refits — and
  pools the chain via Rubin's rules. The fixed-effect `(beta, se)`
  extraction is now a shared helper (`.attach_beta_se`) consumed by both
  the new corrections and the existing `tulpa_em_mc()` MCEM driver, so
  there is one source of truth for "Laplace fit -> Rubin pool input".
  Bernoulli is the default per-observation hard-z draw; multi-class
  latent structures supply their own via the new `draw_z` callback.
  Return shape gains `correction`, `pooled`, and `draws` fields when a
  correction is requested. No ABI bump (R-side only); closes
  `TODO.md` P3.8.

* feat: ABI v14 — SPDE nested-Laplace upgraded to the v10-style universal
  shim (store_modes, store_Q, paired range/sigma grids, formula-side iid-RE
  block). Replaces the v0 `cpp_nested_laplace_spde` entry and folds the
  SPDE C-callable into the shared `NestedLaplaceShimResult` block used by
  ICAR / BYM2 / NNGP / HSGP. The dedicated `SpdeNestedLaplaceShimResult`
  struct is removed. Latent layout:
  `[beta (p)] [re (n_re_groups)] [w_mesh (n_mesh)]`.
  `TULPA_ABI_VERSION` bumped **13 → 14**; downstream packages must rebuild.

## 2026-05-13 — ABI v13: Phase D — delete legacy ratio path

Closes the tulpaRatio migration tracker (gcol33/tulpa#15). After v12
gated the legacy ratio body of `compute_log_post_impl<T>` behind a
generic-layout check, Phase D removes the body itself and every
consumer of `LegacyRatioData` / `LegacyRatioLayout`.

* `TULPA_ABI_VERSION` bumped **12 → 13**. Downstream packages must
  rebuild against the v13 headers.
* **Removed exported types.** `ModelData::LegacyRatioData legacy`
  (`inst/include/tulpa/model_data.h`) and
  `ParamLayout::LegacyRatioLayout legacy`
  (`inst/include/tulpa/param_layout.h`) are gone. `n_processes > 0`
  with a non-null `data.likelihood_spec` is now the only supported
  configuration.
* **Removed Rcpp entry points** (D-1): `cpp_hmc_fit`, `cpp_hmc_fit_gp`,
  `cpp_hmc_fit_gp_v2`, `cpp_ess_fit`, `cpp_ess_get_n_params`,
  `cpp_vi_fit`, `cpp_vi_get_n_params`, `cpp_sghmc_fit`, `cpp_sgld_fit`,
  `cpp_compute_log_post_test`, `cpp_compute_log_prior_test`,
  `cpp_compute_log_lik_only_test`, `cpp_log_post_split_n_params`.
  Internal samplers (`run_ess_sampler`, `run_sghmc_sampler`, `fit_vi`,
  `run_mclmc_sampler`) and their C-callable shims
  (`tulpa_run_ess_sampler`, `tulpa_sghmc_fit`, `tulpa_fit_vi`,
  `tulpa_mclmc_fit`) remain — downstream packages reach them via the
  generic ModelData/ParamLayout API. Dev tools
  `tools/icar_collapsed_check.R` and `tools/bym2_gradient_check.R`
  also removed.
* **Removed dispatcher branches** (D-2). `resolve_gradient_fn`
  (`src/hmc_gradient_dispatch.h`) now only resolves the generic
  `spec->gradient_fn` / `compute_gradient_generic_arena` /
  `compute_gradient_generic_numerical` paths. Mode overrides
  (`AUTODIFF_TAPE`, `AUTODIFF_ARENA`, `AUTODIFF_FWD`) and the H-mode
  specialized fallthroughs are gone. Callers reaching the dispatcher
  with `n_processes == 0` get `Rcpp::stop` with a pointer to this
  entry. `hmc_gradient_dispatch_predicates.h` deleted.
* **Removed log-posterior orchestrators' legacy body** (D-2).
  `compute_log_post`, `compute_log_prior`, `compute_log_lik_only`
  (`src/hmc_sampler.cpp`) now forward to
  `compute_log_post_generic_spec_double`; the
  `accumulate_log_prior_and_state` / `accumulate_obs_log_lik` body
  and its 5 `hmc_sampler_log_prior_*.h` fragments are gone, along
  with `hmc_log_posterior_split.h`. `compute_log_post_impl<T>`
  (`src/log_post_impl.h`) reduces to the same forward for
  `T = double` and a defensive `T(0)` no-op for autodiff `T` (arena
  AD now routes through `compute_log_post_generic<Var>`).
* **Removed gradient kernels** (D-3, ~30 files, ~17 KLOC). All
  hand-coded H-mode kernels (composite + 4 phases, vectorized +
  5 fragments, analytical, autodiff, feature, gp, hsgp, msgp, svc,
  tvc, st, temporal_gp, ms_temporal, latent), the collapsed-spatial
  machinery (`hmc_icar_collapsed_*` ×9, `hmc_gp_collapsed_*` ×5),
  the legacy ratio likelihood (`hmc_likelihood.h`,
  `hmc_observation_likelihood.h`), and the legacy fallback gradients
  (`compute_gradient_numerical` / `_autodiff` / `_arena` / `_forward`
  / `_numerical_impl`) are deleted. The 6 `log_post_impl_*_block.h`
  fragments and the 2 Rcpp ModelData populators
  (`model_data_rcpp.h`, `hmc_modeldata_builders.h`) follow.
  `verify_gradient_runtime` now always uses
  `compute_gradient_generic_numerical` as the reference.
* **Simplified samplers** (D-4). `compute_param_layout`
  (`src/hmc_param_layout.cpp`) requires `n_processes > 0`; model
  packages place model-specific scalars (overdispersion etc.) in the
  LikelihoodSpec extra-parameter block at `layout.extra_offset`.
  ESS's `build_gaussian_priors` and `get_non_gaussian_params`
  (`src/ess_sampler.h`) walk `process_beta_start` and
  `extra_offset` only.
  `hmc_nuts_mass_init.cpp` drops the family-specific block-spec
  heuristics (NB+ICAR / Bin+ICAR forced DENSE, NB phi-pair 2×2
  block) — re-introducing them would need a LikelihoodSpec hint.
* **ST_IV mass-matrix override disabled.** The precision-informed
  diagonal mass setup at warmup end
  (`src/hmc_nuts_chain_iter_nuts.h`) reconstructed `eta` from
  `data.legacy.X_num_flat` and branched on the legacy `ModelType`.
  ST_IV chains now fall back to the adapted DIAG mass matrix until
  the override is re-expressed through `spec->eta_weights_fn`. One
  no-op per chain at warmup end; practical impact on sampling
  efficiency is small.
* **Removed skipped tests.** `tests/testthat/test-log-post-split.R`
  and `tests/testthat/test-hmc-modeldata-builders.R` are deleted (every
  test was a Phase-D skip). The legacy-ratio gradient-check test in
  `test-spatial-car-proper.R` is removed; the two R-side
  `spatial_car_proper()` construction tests stay.
* **Cumulative numbers.** Phase D-1..D-5 deletes ~57 files and
  ~18 000 lines of legacy ratio infrastructure across `src/`,
  `inst/include/tulpa/`, `tests/`, and `tools/`. Net code reduction
  before the v13 maintenance window starts.

Downstream rebuild notes:
* `tulpaRatio` already routes through the generic LikelihoodSpec
  path via `tulpa_bridge.cpp` + per-family payloads in `lik_specs/`
  (B1+B2 of the migration); rebuild against v13 headers, no logic
  changes needed.
* `tulpaObs` never used the legacy ratio path; rebuild against v13.
* `tulpaGlmm` Day-22+ already targets the generic path; rebuild
  against v13.

## 2026-05-12 — ABI v12: generic-layout safety in compute_log_post_impl + ESS port

* `TULPA_ABI_VERSION` bumped **11 → 12**. Downstream packages must
  rebuild against the v12 headers.
* **Critical fix.** `compute_log_post_impl<T>` (`src/log_post_impl.h`)
  now early-returns to `compute_log_post_generic_spec_double` when
  the caller built `ModelData` with `n_processes > 0` and a non-null
  `likelihood_spec`. Previously the function reached lines 83-84 and
  unconditionally read `params[layout.legacy.beta_num_start]`, which
  is `params[-1]` for generic-layout callers — segfault. This was the
  blocker for tulpaGlmm Day-22 `inference = "ess"` (see deferred
  `fix.md` entry from 2026-05-06). The early-return makes the function
  safe for both layouts; the legacy ratio body remains in place for
  `n_processes == 0` callers (i.e. nobody outside this file at the
  moment, but tulpaRatio's `hmc_sampler.cpp` keeps its own copy).
* **ESS generic-layout port.** `tulpa_ess::build_gaussian_priors`
  (`src/ess_sampler.h`) now walks every process's β block
  (`layout.process_beta_start[k]` for `k` in `0..n_processes`) when
  `data.n_processes > 0`, instead of only `layout.legacy.beta_num_start
  / beta_denom_start`. Previously generic-layout ESS produced an
  empty β prior block and β was never sampled.
* **ESS RWMH coverage of model-specific extras.**
  `tulpa_ess::get_non_gaussian_params` now appends every parameter
  in `[layout.extra_offset, layout.extra_offset + n_extra_params)`
  to the RWMH list. LikelihoodSpec authors pack their model-specific
  scalars (e.g. log_phi for negative-binomial, log_sigma for Gaussian)
  into that block; ESS now walks them. Legacy ratio `log_phi_num /
  log_phi_denom` indices remain in the list for `n_processes == 0`.
* `LegacyRatioData` / `LegacyRatioLayout` (`inst/include/tulpa/model_data.h`,
  `param_layout.h`) are still exported but stay deprecated — the
  in-engine consumers are the H-mode gradient kernels, the legacy
  AD fallback (`hmc_gradient_fallback.cpp`), the composite gradient,
  and `tulpa_hmc::compute_log_post` inside `hmc_sampler.cpp`. None
  of those are reached when the dispatcher (`hmc_gradient_dispatch.h`)
  sees `n_processes > 0`. Full removal is a follow-up cut after the
  collapsed-spatial double-evaluator (MCLMC / SGHMC consumer) is
  reworked.
* Downstream rebuild notes: tulpaRatio uses the generic-LikelihoodSpec
  path via `tulpa_bridge.cpp` + per-family payloads in
  `lik_specs/`; it never touched `ModelData::legacy` and rebuilds
  cleanly against v12 headers. tulpaGlmm Day-22 ESS shim can now
  call `tulpa::get_ess_fn()(...)` end-to-end on a generic-layout
  `ModelData` without segfaulting.

## 2026-05-12 — ABI v11: caller-supplied inv-mass diagonal for NUTS

* `TULPA_ABI_VERSION` bumped **10 → 11**. Downstream packages must
  rebuild against the v11 headers.
* Registered C-callable `tulpa_run_nuts_generic` (`nuts_api.h`
  `NUTSFn`) gains a new positional parameter `const double*
  inv_metric_diag` immediately before `NUTSResult* result_out`. Pass
  `nullptr` to keep the v10 behaviour (default structural warm-start
  of the mass matrix). Pass a length-`n_params` vector to seed the
  diagonal inverse-mass — useful for warm-starting NUTS from an
  analytical-approximation method (Laplace, VI, etc.).
* `run_hmc_chain_cpp` / `run_hmc_chain` (`hmc_sampler_funcs.h`) take a
  matching trailing `inv_metric_init` `std::vector<double>` (default
  empty). Within `run_hmc_chain_cpp` (`hmc_nuts_chain_setup.h`), a
  non-empty caller diagonal overrides the structural diagonal set by
  `warm_start_mass_matrix`. Values are clamped to `[1e-3, 1e3]`
  before being installed via `mass.set_diagonal`, then
  `find_reasonable_epsilon` re-runs against the seeded metric.
* Mass-matrix *adaptation* is unchanged: the standard dual-averaging
  + expanding-windows path still refines the diagonal during warmup.
  The caller's vector is the *initial* metric, not a frozen one.
* Internal callers of `run_hmc_chain_cpp` (`hmc_nuts_parallel.cpp` ×3,
  `tulpa_generic_sampler.cpp` ×1) continue to pass no `inv_metric_init`
  via the default empty vector; the local forward declaration in
  `tulpa_generic_sampler.cpp` was updated to match the canonical
  declaration's parameter list.
* Downstream rebuild notes: `tulpaGlmm` exercises this end-to-end via
  Day-32's `hmc_warm_start = "laplace"` argument. `tulpaObs` and
  `tulpaRatio` need to be reinstalled against v11 — both already
  updated to pass `nullptr` for the new parameter (no logic change).

## 2026-05-11 — NNGP Laplace: full off-diagonal precision scatter

* `laplace_mode_gp` (and the spatial-only / ST-combo NNGP entries in
  `nested_laplace.cpp`) now assemble the **full NNGP precision matrix**
  `Λ = (I - A)' D⁻¹ (I - A)` in every Newton iteration, replacing the
  diagonal-on-w approximation that only kept `1/v_i` on the focal
  diagonal of each row.
* What was missed before: the gradient contribution to neighbours
  (`+a_{i,k}·q_i/v_i`), the off-diagonal Hessian entries
  (focal, neighbour_k) and (neighbour_k, neighbour_kp), and the
  pairwise precision between members of every conditioning set. The
  Newton mode for `w` was therefore shrunk toward zero and pointwise
  field recovery on smooth latent fields collapsed (cor ≈ 0).
* New helpers in `gpu_nngp_laplace.h`:
    - `batch_nngp_scatter(..., alpha_out = nullptr)` — backward-compatible
      extra optional output capturing the per-row conditional regression
      weights (already computed internally; just exposed).
    - `apply_nngp_full_prior_dense` — scatters the full precision
      contribution into a dense `(grad, H)` pair via the alpha + cv
      bundle.
    - `apply_nngp_full_prior_sparse` — same, into a `SparseHessianBuilder`.
    - `make_nngp_prior_sparsity_pattern` — emits the `(row, col)` pairs
      required to back the sparse path.
* Wired into the four scatter call sites: `laplace_mode_gp` dense Newton,
  `laplace_mode_gp` sparse Newton (with pattern expansion), the
  spatial-only `cpp_nested_laplace_nngp` scatter lambda, and
  `make_nngp_spatial_ops::add_prior_at_k` (the ST-combo NNGP block).
  Log-prior calls (`log_prior` lambdas) are unchanged — they only need
  `cm` and `cv`, and the existing `batch_nngp_scatter` signature still
  supports that without `alpha_out`.
* Effect (downstream measurement from tulpaGlmm Day-31 smoke):
  `cor(w_mean, f_true)` jumps from near zero to **≈ 0.81 Pearson** on
  a 120-location Poisson + smooth-GP simulation. β recovery unchanged.
* Additive: no `TULPA_ABI_VERSION` bump (still v10). Public shim
  signatures are unchanged; only the inner Laplace scatter is upgraded.
  Downstream packages must rebuild against this commit to pick up the
  new behaviour (no source changes required).

## 2026-05-11 — nested-Laplace ST family: 5 more indexed × indexed combos

* Adds five additional joint spatial × temporal nested-Laplace shims,
  built on the same `run_two_indexed_nested_laplace` driver and joint
  inner Newton introduced earlier today:
    - `tulpa_nested_laplace_st_icar_rw1`
    - `tulpa_nested_laplace_st_icar_rw2`
    - `tulpa_nested_laplace_st_car_proper_rw1`
    - `tulpa_nested_laplace_st_car_proper_rw2`
    - `tulpa_nested_laplace_st_car_proper_ar1`
  Each routes through a per-combo Rcpp entry plus a C-callable
  `_impl` wrapper; matching typedefs + getters live in
  `nested_laplace_api.h`.
* New internal factory pattern `IndexedPriorOps` in `nested_laplace.cpp`
  with per-kind builders `make_icar_ops`, `make_car_proper_ops`,
  `make_rw1_ops`, `make_rw2_ops`, `make_ar1_ops`. The shared
  `run_two_indexed_nested_laplace` driver now consumes
  `std::function`-typed callbacks, so adding the next indexed × indexed
  combination is a few lines of Rcpp glue rather than a re-derivation.
* Refactored `cpp_nested_laplace_st_icar_ar1` to use the new factories
  (identical behavior; just dropped the inline lambdas).
* Additive: no `TULPA_ABI_VERSION` bump (still v9). The new shims are
  resolved via `R_GetCCallable` at first use; downstream packages
  rebuilt against ABI v9 pick them up automatically.

## 2026-05-11 — nested-Laplace joint spatial × temporal (ICAR × AR1)

* New shim `tulpa_nested_laplace_st_icar_ar1` for joint nested-Laplace
  inference with an ICAR spatial field AND an AR1 temporal field in the
  same fit. The joint inner Newton solves over the full latent vector
  `[beta] [re] [w_spatial (n_s)] [w_temporal (n_t)]` at each grid point;
  the cross-block `H[w_s, w_t]` is non-zero, so the two fields cannot be
  Laplace-marginalized separately. The hyperparameter grid is supplied
  caller-side as paired vectors of length `n_grid` (Cartesian product of
  `τ_spatial × τ_temporal × ρ_temporal` built on the R side).
* New C-callable typedef `NestedLaplaceStIcarAr1Fn` + getter
  `get_nested_laplace_st_icar_ar1_fn()` in `nested_laplace_api.h`.
* New internal building blocks in `nested_laplace.cpp`: the templated
  `run_two_indexed_nested_laplace` driver and helpers
  `nl_compute_eta_two_indexed` / `nl_scatter_obs_two_indexed`. These are
  the shared substrate for the remaining 11 (spatial_kind × temporal_kind)
  combinations.
* `TULPA_ABI_VERSION` bumped 8 → 9. Downstream packages must be rebuilt
  against this header set.

## 2026-05-11 — nested-Laplace HSGP returns modes + store_Q

* `tulpa_nested_laplace_hsgp` now sets `store_modes = 1` (was 0) and
  gained a `store_Q` flag matching the rest of the nested-Laplace family.
  The basis-coefficient latent `[beta] [re] [beta_M (n_basis)]` is
  returned per `(σ², ℓ)` grid point, and with `store_Q = 1` the joint Q
  at the mode is retained in the standard `NestedLaplaceShimResult::Q_*_flat`
  slots.
* `cpp_nested_laplace_hsgp` gained a trailing `bool store_Q = false`
  argument and now passes `store_modes = true` to the grid driver. The
  C-callable `tulpa_nested_laplace_hsgp_impl` signature picks up the
  matching `int store_Q` parameter; the public typedef
  `NestedLaplaceHsgpFn` in `nested_laplace_api.h` is updated to match.
* This unblocks HSGP mixture-of-MVN sampling in tulpaGlmm — the
  observation-level spatial effect `f_i = Σ_j Φ_ij · √S(λ_j; σ²_k, ℓ_k) · β_M_j`
  can be reconstructed caller-side from modes + posterior draws over the
  basis coefficients plus the per-draw grid index.
* `TULPA_ABI_VERSION` bumped 7 → 8. Downstream packages (tulpaGlmm,
  tulpaObs) must be rebuilt against the updated headers.

## 2026-05-11 — nested-Laplace BYM2 returns modes + store_Q

* `tulpa_nested_laplace_bym2` now sets `store_modes = 1` (was 0) and
  gained a `store_Q` flag matching the rest of the nested-Laplace family.
  The reparameterised latent `[beta] [re] [phi (n_spatial)] [theta
  (n_spatial)]` is returned per grid point, and with `store_Q = 1` the
  joint Q at the mode is retained in the standard
  `NestedLaplaceShimResult::Q_*_flat` slots.
* `cpp_nested_laplace_bym2` gained a trailing `bool store_Q = false`
  argument and now passes `store_modes = true` to the grid driver. The
  C-callable `tulpa_nested_laplace_bym2_impl` signature picks up the
  matching `int store_Q` parameter; the public typedef
  `NestedLaplaceBym2Fn` in `nested_laplace_api.h` is updated to match.
* This unblocks BYM2 mixture-of-MVN sampling in tulpaGlmm — the total
  spatial effect `w_s = σ·(√ρ · scale · φ_s + √(1−ρ) · θ_s)` can be
  reconstructed caller-side from modes + posterior draws over the
  (σ, ρ) grid.
* `TULPA_ABI_VERSION` bumped 6 → 7. Downstream packages (tulpaGlmm,
  tulpaObs) must be rebuilt against the updated headers.

## 2026-05-11 — nested-Laplace store_Q on RW1/RW2/AR1/CAR_proper

* `tulpa_nested_laplace_rw1`, `tulpa_nested_laplace_rw2`,
  `tulpa_nested_laplace_ar1`, and `tulpa_nested_laplace_car_proper` now
  accept a `store_Q` flag (matching the ICAR shim added in v5). When set
  the shim retains the joint negative-Hessian Q at each grid point's mode
  in `NestedLaplaceShimResult::Q_*_flat`, so downstream packages can draw
  mixture-of-MVN posteriors `sum_k w_k · N(mode_k, Q_k^{-1})` without
  re-doing the Newton assembly R-side.
* The underlying `cpp_nested_laplace_<rw1|rw2|ar1|car_proper>` entries
  gained a trailing `bool store_Q = false` argument. Default is `false`,
  so existing callers that don't ask for Q keep the previous behaviour
  and footprint.
* `TULPA_ABI_VERSION` bumped 5 → 6. Downstream packages (tulpaGlmm,
  tulpaObs) must be rebuilt against the updated headers.

## 2026-05-06 — Takahashi partial inverse as a registered C-callable

* New free function `tulpa::takahashi_partial_inverse_dense(n, Lp, Li, Lx,
  Z_out)` in `sparse_cholesky.{h,cpp}` runs the Takahashi recursion on a
  caller-supplied lower-triangular `L` (CSC) and writes a dense column-major
  `n*n` `Z` with `Q^{-1}` on `pattern(L + L^T)` and zeros elsewhere. A
  matching `takahashi_partial_inverse_csc` returns just the `Zx` values on
  pattern(L). The existing `SparseCholeskySolver::selected_inversion_diagonal`
  now routes through the new helper so there is one source of truth for the
  recursion (no copy-paste).
* Registered C-callable `tulpa_takahashi_partial_inverse_dense` exposes the
  pure-function variant to downstream packages. Resolved via
  `tulpa::get_takahashi_partial_inverse_dense_fn()` in
  `inst/include/tulpa/sparse_solver_api.h`; the getter `Rf_error`s if the
  symbol is missing (i.e. caller built against newer headers than the loaded
  tulpa).
* No struct layout changes; `TULPA_ABI_VERSION` stays at 4. Downstream
  packages that want the new shim need only rebuild against the updated
  `sparse_solver_api.h`.

## 2026-05-05 — multi-term + slope REs on the spec-Laplace path

* `tulpa_laplace_spec_dense` (and its public C ABI shim) now accepts the
  full multi-term, multi-coefficient RE structure populated by `populate_re`
  in downstream model packages: K = `data.n_re_terms` random-effect terms,
  each with `q_t = re_n_coefs[t]` coefficients per group (intercept-only
  when `q_t == 1`, intercept + slopes when `q_t > 1`), uncorrelated
  (`(x||g)`) or correlated (`(x|g)`) prior covariance. Per-process
  sharing is uniform across terms via `data.sharing.re`.
* The legacy single-term path (`layout.re_start` / `layout.re_end` /
  `log_sigma_re_idx` with `data.n_re_terms == 0`) is preserved and stays
  bit-identical numerically.
* The `result_out->mode` writeout from `tulpa_laplace_spec_dense_impl`
  now concatenates every RE term's block in term order
  (`re_start_multi[t]..re_end_multi[t]`), matching the new
  `SpecLatentLayout` ordering.
* New internal test fixture `cpp_laplace_spec_test_multi_re` exercises
  the multi-term path end-to-end against hand-derived linear-Gaussian
  reference solutions; new tests in `tests/testthat/test-laplace-spec.R`
  cover (a) two crossed intercept-only terms, (b) one correlated random
  slope, (c) one uncorrelated random slope.
* `TULPA_ABI_VERSION` bumped from 3 → 4. Downstream packages that ship
  with the spec-Laplace dispatcher (e.g. tulpaGlmm) must be rebuilt
  against the new headers.
