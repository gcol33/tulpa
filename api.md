# tulpa API reference

Engine for Bayesian hierarchical modelling. This indexes the **public R surface**
(Part 1) and the **C++ model-package interface** (Part 2). It is a map, not a
manual -- see `?<fn>` for full argument lists and the headers in
`inst/include/tulpa/` for the C++ contracts.

Design frame (see `CLAUDE.md`): the posterior is decomposed so well-behaved
blocks get a cheap deterministic approximation (Laplace / nested-Laplace / VI /
Pathfinder) and exact MCMC corrects only the residual directions. Functions are
grouped below by which layer they occupy.

---

# Part 1 -- R public API

## Front door

| Function | Purpose |
|---|---|
| `tulpa()` | Formula front door. Parses `y ~ x + latent(...) + (1 \| g)`, selects a tier/backend (auto or explicit), fits. |
| `tulpa_parse_formula()`, `find_latent_terms()`, `no_latent_terms()` | Formula parsing / latent-term extraction. |
| `findbars()`, `nobars()`, `parse_bar_term()` | `lme4`-style random-effect bar parsing (`(1 + x \| g)`). |
| `spatial()` | Inline areal varying-coefficient field, written in the model formula like a bar: `spatial(graph, ~ 1 + time \|\| cell)`. The bar LHS expands to one independent CAR / Besag field per design column (intercept field + per-region slopes); `\|\|` only (a single `\|` / MCAR, nesting, and `by=` are reserved). Distinct from the bare `spatial(col)` areal-naming term and the `spatial =` constructors below. |
| `latent()`, `latent_factor()`, `latent_factors()` | Wrap a latent prior block / latent-factor model for use in a formula. |
| `inference_mode_info()` | Inspect the backend registry (tiers, inputs, selection rules). |

## Families, data, simulation

| Function | Purpose |
|---|---|
| `tulpa_family()`, `tulpa_gaussian()` | Family/link objects. |
| `tulpa_build_model_data()` | Assemble the internal `ModelData` from a formula + data. |
| `tulpa_simulate()` | Simulate from a specified hierarchical model. |

## Tier 2 -- nested approximation (Laplace family)

The typical hot path for Gaussian-latent workloads.

| Function | Purpose |
|---|---|
| `tulpa_laplace()` | Conditional Laplace at a **supplied** prior / `Sigma`. Returns mode, Laplace log-marginal, optional Hessian. The inner solve everything else composes on. |
| `tulpa_laplace_beta()` | Fixed-effect-only Laplace. |
| `tulpa_nested_laplace()` | Outer grid over the hyperparameters of one latent prior block (spatial/temporal/GMRF), inner Laplace per cell, integrated marginals. `likelihood=` accepts a model-supplied `NestedLikelihood`. |
| `tulpa_nested_laplace_joint()` | Multi-arm joint nested-Laplace (shared field with `copy=` scaling, `(sigma, alpha)` + `phi` grids). |
| `tulpa_re_cov_nested()` | **Nested-Laplace integration over a free random-effect covariance `Sigma`** (e.g. `(1 + x \| g)`). Log-Cholesky parameterization (general rank), grid centred+rotated at the marginal-likelihood mode, marginalized weighted quantiles of `sigma_i` / `rho_ij` / `Sigma_ij`. Corrects the plug-in-MAP summary bias. |
| `tulpa_em_laplace()`, `tulpa_em_mc()` | Generic EM drivers (deterministic / Monte-Carlo E-step) with `correction = "mi"/"gibbs"` post-EM refits. |
| `imh_laplace()` | IMH-Laplace: Laplace body + independence-MH bias correction. |
| `agq_fit()` | Adaptive Gauss-Hermite quadrature fit. |
| `fit_spde()` | SPDE continuous-spatial fit. |

(N-mixture / occupancy Laplace fitters such as `nmix_laplace()` live in the
consumer package tulpaObs, not the engine.)

## Tier 1 -- exact MCMC and debias

| Function | Purpose |
|---|---|
| `tulpa_gibbs()` | Gibbs sampler. |
| `tulpa_re_cov_gibbs()` | **Exact RE-covariance posterior**: Metropolis-within-Gibbs (MH on groups/`beta`, conjugate inverse-Wishart draw for `Sigma`). The debias counterpart to `tulpa_re_cov_nested()` for binary/low-count small groups. |
| `tulpa_nuts_beta()`, `tulpa_nuts_spde()` | NUTS samplers. |
| `tulpa_tgmrf(mode = "nuts" / "nuts_joint" / "imh")` | tgmrf samplers (NUTS / joint NUTS / IMH). The method is an argument, not a parallel verb. |
| `mala()` | MALA sampler. |

## Tier 3 / other approximations

| Function | Purpose |
|---|---|
| `tulpa_tgmrf(mode = "vi")` | Variational inference for tgmrf blocks. |
| `pathfinder()` | L-BFGS mode + Gaussian fit + ELBO scoring. |
| `bridge_sampling()` | Marginal-likelihood estimation via bridge sampling. |

## Latent structure constructors

Spatial: `spatial_bym2()`, `spatial_car()`, `spatial_car_proper()`, `spatial_gp()`,
`spatial_hsgp()`, `spatial_multiscale()`, `spatial_rsr()`, `spatial_spde()`,
`spatial_spde_custom()`, `spatial_svc()`. Inline areal varying-coefficient
fields use `spatial()` (front door, above), the areal analogue of `spatial_svc()`.
Temporal: `temporal_ar1()`, `temporal_gp()`, `temporal_multiscale()`,
`temporal_rw1()`, `temporal_rw2()`, `temporal_tvc()`.
Space-time / varying coefficients: `spatiotemporal()`, `spatiotemporal_gp()`,
`spatiotemporal_effects()`, `svc()`, `tvc()`.
GMRF: `tgmrf()`, `tgmrf_cpp()`.

## Priors

`tulpa_priors()`, `priors_default()`, `prior_from_spec()`, and the prior builders
`prior_normal()`, `prior_beta()`, `prior_gamma()`, `prior_exponential()`,
`prior_half_cauchy()`, `prior_half_normal()`, `prior_pc()` (PC priors).

## Generic S3 methods (on `tulpa_fit`)

Model packages inherit by setting `class = c("model_fit", "tulpa_fit")`.
S3 methods: `coef`, `confint`, `vcov`, `logLik`, `summary`, `plot`, `print`,
`tidy`, `glance`, `ranef`, `predict`, `fitted`, `residuals`, `simulate`, `nobs`.
Information criteria / cross-validation: `tulpa_criteria()` (WAIC / DIC / CPO /
LPML / PSIS-LOO), `tulpa_kfold()`, `tulpa_reloo()`, `tulpa_psis()`, `bayes_R2()`,
`tulpa_powerscale_sensitivity()`, `bridge_sampling()`.
Posterior-predictive: `posterior_predict()`, `pp_check()`, `prior_predict()`.

## Diagnostics

Convergence (native, no `posterior`/`coda` dep): `mcmc_diagnostics()`,
`select_main_params()`, `check_diagnostics()`, `diagnostic_summary()`,
`n_divergent()`, `geweke_test()`.
Plots: `plot_rhat()`, `plot_ess()`, `plot_acf()`, `plot_energy()`,
`plot_divergences()`, `plot_pairs()`, `plot_diagnostics()`, `plot_map()`,
`plot_map_panel()`.
Model comparison: `compare_models()`, `model_average()` (stacking / pseudo-BMA).
Residual / GOF: `moran_i()`, `durbin_watson()`, `tulpa_variogram()`,
`pit_residuals()`, `test_uniformity()`, `test_dispersion()`, `test_outliers()`,
`test_zero_inflation()`, `check_model()`.
Derived effects: `spatial_range()`, `temporal_corr()`, `post_hoc_lm()`.

## Profiling

`tulpa_profile(expr)` times the inner sparse-Laplace solve one phase at a time
and returns a data frame (`phase`, `seconds`, `calls`, `ms_per_call`, `share`,
ordered by time; the fit is attached as `attr(, "value")`). It resets the
process-global phase accumulator, forces `expr`, then reads it back, so the
times cover the whole fit including the parallel outer-grid worker threads.
Phases: `pattern_build`, `prep`, `eta`, `scatter` (per-obs likelihood + Hessian
assembly), `analyze`, `factorize` (numeric Cholesky), `solve`, `line_search`,
`log_det`, `log_lik_prior`. Use it to settle whether a joint `occu_cover()` /
`cover()` fit is bound by the assembly scatter or the Cholesky factorize. The
raw accumulator is reachable as `cpp_profile_reset()` / `cpp_profile_read()`
(microseconds + call counts per phase); the C++ instrumentation site is
`TULPA_PROFILE_PHASE(idx)` from `laplace_profile.h`.

## Pooling and utilities

`rubins_pool()` (multiple-imputation pooling), `tulpa_draws_array()`
(`as_draws_array()`-style accessor), `ccd_grid()` / `ccd_to_theta()` (central
composite design nodes + whitening map for high-dimensional outer integration),
`validate_mode()`, `tulpa_cache_dir()`, `sn_cdf()` / `sn_quantile()` /
`sn_match()` (skew-normal helpers).

---

# Part 2 -- C++ model-package API (`LinkingTo: tulpa`)

A model package (tulpaObs, tulpaRatio, tulpaGlmm) supplies its **likelihood**;
tulpa assembles the linear predictors and runs every inference tier. Exported
headers live in `inst/include/tulpa/`; add `LinkingTo: tulpa` to the model
package `DESCRIPTION`.

## ABI contract

`TULPA_ABI_VERSION` (defined in `model_data.h`, which is the value to read --
restating it here is what let this page drift 10 versions behind) guards the
binary interface. Bump it when any exported struct layout changes; new fields go in the
stable sections, never before existing fields. tulpa registers
`R_RegisterCCallable("tulpa", "tulpa_get_abi_version", ...)`, and model packages
auto-check on first NUTS call -- a mismatch gives a clear error instead of a
segfault. Inter-package C++ calls go through `R_RegisterCCallable` /
`R_GetCCallable` (see the `*_api.h` headers).

## `LikelihoodSpec` (`likelihood.h`)

The single likelihood boundary consumed by every tier. Key members:

| Member | Role |
|---|---|
| `abi_version` | Set automatically to `TULPA_ABI_VERSION`. |
| `n_processes`, `name` | Number of linear predictors; label. |
| `ll_double`, `ll_arena`, `ll_fwd` | `LikelihoodFn<T>` for each AD mode (double / reverse-arena / forward-dual). |
| `eta_weights_fn` | `EtaWeightsFn`: per-obs eta-space `grad_eta[k]` + `neg_hess_eta[k*np+l]`. **Must return the expected (Fisher) information, not the AD-observed Hessian**, so the Newton Hessian stays PD on non-canonical links. Required for the spec-driven Laplace / nested-Laplace path; ignored by NUTS/VI/ESS. |
| `residual_fn`, `extra_grad_fn` | H-mode per-obs residual / extra-parameter gradients (optional). |
| `n_extra_params`, `extend_layout`, `extra_prior`, `extra_prior_arena` | Model-specific extra parameters: count, `ParamLayout` extension, and prior contribution (double + arena-AD variants). |
| `gradient_fn` | Optional fully hand-coded full-vector gradient (bypasses the templated AD path). |

`LikelihoodFn<T>` and `EtaWeightsFn` are the function-pointer typedefs at the top
of `likelihood.h`. The spec is **append-only** -- add members at the end.

## `NestedLikelihood` (`nested_likelihood.h`)

Bundle that lets a model package drive `tulpa_nested_laplace(likelihood = )` from
R. The package builds it in its own C++ and returns
`Rcpp::XPtr<NestedLikelihood>(p, /*finalize=*/true)`:

| Field | Role |
|---|---|
| `spec` | A **single-process** (`n_processes == 1`) `LikelihoodSpec`; `eta_weights_fn` + `ll_double` must be non-null, and `neg_hess_eta[0]` must be Fisher info. |
| `response_data` | Opaque per-obs response (e.g. `{y, det_prob}` for occupancy), passed to the spec callbacks as `model_response_data`. |
| `keepalive` | `std::shared_ptr<void>` owning the spec object + response storage so both outlive the fit. |

tulpa's own families reach the nested grid through `builtin_family_spec`
(`src/laplace_builtin_family_spec.h`); a model package supplies its own spec
instead -- e.g. tulpaObs's marginalized single-season occupancy likelihood
(`src/occ_nested_likelihood.cpp`).

## Core data types

| Type (header) | Role |
|---|---|
| `ModelData` (`model_data.h`) | Inputs: response, designs, latent-structure data. Requires `n_processes > 0` and a `LikelihoodSpec` (ratio models live in tulpaRatio via that interface) -- never insert before existing fields. |
| `ParamLayout` (`param_layout.h`) | Parameter-vector positions; `extend_layout` appends model-specific slots. |
| `ResidualFn`, `LikelihoodFn<T>`, `EtaWeightsFn` (`likelihood.h`) | Likelihood callback typedefs. |

## Per-tier entry headers

`laplace_api.h`, `laplace_spec_api.h` (spec-driven conditional Laplace),
`nested_laplace_api.h`, `joint_nested_laplace_api.h`, `nuts_api.h`, `vi_api.h`,
`ess_api.h`, `smc_api.h`, `sghmc_api.h`, `mclmc_api.h`, `pg_api.h`,
`sparse_solver_api.h`, `spde_api.h`, `tgmrf.h`. Latent-structure data structs:
`gp_data.h`, `hsgp_data.h`, `st_data.h`, `svc_data.h`, `tvc_data.h`,
`temporal_data.h`, `spde_model_data.h`. Autodiff: `autodiff_arena.h`,
`autodiff_fwd.h`. Shared: `types.h`, `portable_math.h`, `priors_capped.h`.
