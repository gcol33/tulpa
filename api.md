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
| `tulpa_parse_formula()` | Formula parsing / latent-term extraction. |
| `findbars()`, `nobars()` | `lme4`-style random-effect bar parsing (`(1 + x \| g)`). |
| `spatial()` | Inline areal varying-coefficient field, written in the model formula like a bar: `spatial(graph, ~ 1 + time \|\| cell)`. The bar LHS expands to one independent CAR / Besag field per design column (intercept field + per-region slopes); `\|\|` only (a single `\|` / MCAR, nesting, and `by=` are reserved). Distinct from the bare `spatial(col)` areal-naming term and the `spatial =` constructors below. |
| `latent()`, `latent_factor()` | Wrap a latent prior block / latent-factor model for use in a formula. |
| `inference_mode_info()` | Inspect the backend registry (tiers, inputs, selection rules). |
| `validate_mode()` | Assert that a fit used the mode it was asked for. |
| `tulpa_check_control()` | Validate a `control = list()` against its canonical key set. |

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
| `tulpa_eb()` | **Empirical Bayes over a free random-effect covariance `Sigma`**: the mode of the same objective `tulpa_re_cov_nested()` integrates, with the fixed effects reported conditional on it. Cheaper, and the intervals are conditional on `Sigma_hat` rather than marginal over `Sigma`. |
| `tulpa_em_laplace()`, `tulpa_em_mc()` | Generic EM drivers (deterministic / Monte-Carlo E-step) with `correction = "mi"/"gibbs"` post-EM refits. |
| `imh_laplace()` | IMH-Laplace: Laplace body + independence-MH bias correction. |
| `agq_fit()` | Adaptive Gauss-Hermite quadrature fit. |
| `tulpa_re_aghq()` | **Adaptive Gauss-Hermite refinement of a grouped RE covariance**: replaces each per-group Laplace integral with `n_quad`-point AGHQ. Three input forms select the oracle -- `make_site` (single-arm, per-row separable), `make_group` (general / multi-arm), or a prebuilt compiled `oracle` from a consumer package. |
| `tulpa_ep()` | Expectation Propagation for a fixed-effect GLM: one Gaussian site per observation, matching marginal moments (exact for a Gaussian likelihood). Reachable via `tulpa(mode = "ep")`. |
| `tulpa_multinomial()`, `tulpa_ordinal()` | Nominal K-class logistic and ordered cumulative-logit regression via Laplace. |
| `fit_spde()` | SPDE continuous-spatial fit. |
| `fit_st_nested()` | Additive spatiotemporal GLM by nested Laplace. |
| `tulpa_hyper_grid()` | Outer hyperparameter-grid integration driven by a user-supplied inner fit, for a block the shipped kernels do not cover. |

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
| `tulpa_integrator()`, `with_tulpa_integrator()` | Select the symplectic integrator HMC / NUTS leapfrogs with, globally or for one expression. |
| `tulpa_cache_dir()`, `tulpa_cache_clear()` | Report and empty the `tgmrf_cpp()` compiled-block cache. |

## Tier 3 / other approximations

| Function | Purpose |
|---|---|
| `tulpa_tgmrf(mode = "vi")` | Variational inference for tgmrf blocks. |
| `pathfinder()` | L-BFGS mode + Gaussian fit + ELBO scoring. |
| `bridge_sampling()` | Marginal-likelihood estimation via bridge sampling. |

## Latent structure constructors

Spatial: `spatial_bym2()`, `spatial_car()`, `spatial_car_proper()`, `spatial_gp()`,
`spatial_multiscale()`, `spatial_rsr()`, `spatial_spde()`,
`spatial_spde_custom()`, `spatial_svc()`. There is no `spatial_hsgp()`: the
Hilbert-space basis approximation is an argument on the GP constructor,
`spatial_gp(approx = "hsgp", m =, c =)`. Inline areal varying-coefficient
fields use `spatial()` (front door, above), the areal analogue of `spatial_svc()`.
Temporal: `temporal_ar1()`, `temporal_ar2()`, `temporal_ar()` (general order
`p`), `temporal_gp()`, `temporal_multiscale()`, `temporal_rw1()`,
`temporal_rw2()`, `temporal_rtr()`, `temporal_tvc()`.
Space-time / varying coefficients: `spatiotemporal()`, `spatiotemporal_gp()`,
`svc()`, `tvc()`.
GMRF: `tgmrf()`, `tgmrf_cpp()`.
Graph and neighbour construction: `adjacency()`, `check_adjacency()`,
`node_index()` (cell identifiers to graph node indices),
`compute_nngp_neighbors()`.
Inline-bar plumbing, for a package building its own varying-coefficient door:
`tulpa_is_spatial_bar()`, `tulpa_bar_field_specs()` (expand a bar into per-column
field specs), `tulpa_bar_field_replicate()` (replicate an areal graph across the
levels of a factor).

## Priors

`tulpa_priors()`, `priors_default()`, `prior_from_spec()`, and the prior builders
`prior_normal()`, `prior_beta()`, `prior_gamma()`, `prior_exponential()`,
`prior_half_cauchy()`, `prior_half_normal()`, `prior_pc()` (PC priors).
`re_cov_pc_lkj_prior()` builds the per-block PC + LKJ hyperprior the free-Sigma
backends default to, with the exact change-of-variables Jacobian.

## Generic S3 methods (on `tulpa_fit`)

Model packages inherit by setting `class = c("model_fit", "tulpa_fit")`.
S3 methods: `coef`, `confint`, `vcov`, `logLik`, `summary`, `plot`, `print`,
`tidy`, `glance`, `ranef`, `predict`, `fitted`, `residuals`, `simulate`, `nobs`.
lme4 / posterior interop: `fixef()` (synonym for `coef()`) and `as_draws()` /
`as_draws_array()` / `as_draws_matrix()` / `as_draws_df()` / `as_draws_rvars()`.
These are tulpa's own generics, and `.onLoad()` also registers the `tulpa_fit`
methods on `lme4::fixef`, `nlme::fixef`, `lme4::ranef`, `nlme::ranef` and the
`posterior::as_draws*` family when those packages are present -- so a `tulpa_fit`
drops into an lme4- or posterior-shaped workflow without any of them becoming a
dependency. A Gaussian-approximation fit (`mode = "laplace"`, `mode = "eb"`)
carries no draws; `as_draws(fit, n_draws = )` opts in to sampling the
approximation instead.
Information criteria / cross-validation: `tulpa_criteria()` (WAIC / DIC / CPO /
LPML / PSIS-LOO), the single-criterion doors `dic()` and `cpo()`,
`tulpa_kfold()`, `tulpa_reloo()`, `tulpa_psis()`, `tulpa_loglik()` (streaming
pointwise log-likelihood), `bayes_R2()`, `tulpa_powerscale_sensitivity()`,
`bridge_sampling()`.
Posterior-predictive: `posterior_predict()`, `pp_check()`, `prior_predict()`.

## Reading a fit

Draws, in whatever representation the backend produced.
`posterior_sample()` is the provenance-agnostic accessor for summaries;
`mcmc_draws()` is the chain-only view (`NULL` on any non-chain fit) the
convergence diagnostics gate on; `tulpa_draws_array()` is the
`as_draws_array()`-style `[iter, chain, param]` accessor.
`tulpa_posterior_draws()` samples the nested tier's own outer-grid mixture --
each draw picks a cell by its weight, then that cell's inner Gaussian, with the
cell index on `attr(., "cells")`. It is the faithful primitive for marginalizing
a nonlinear derived quantity, which collapsing the grid to one Gaussian biases.

Extracted structure: `VarCorr()` (RE variances and correlations), `ranef()`,
`temporal()`, `spatiotemporal_effects()`, `smooth_effects()` (fitted covariate
smooths), `latent_factors()`.

## Diagnostics

`diagnostics(fit)` is the front door. It reads the draws PROVENANCE and returns
the diagnostic that applies: Rhat / ESS / MCSE for a chain fit, the
approximation-reliability table for an i.i.d. one, `NULL` for a point fit.
`diagnostics(fit, sbc = )` reads the fit and a calibration result together.

Convergence (native, no `posterior`/`coda` dep): `mcmc_diagnostics()`,
`select_main_params()`, `check_diagnostics()`, `diagnostic_summary()`,
`n_divergent()`, `geweke_test()`.
Approximation reliability: `laplace_diagnostics()` -- the outer Pareto-k-hat,
the inner skewness estimate `gamma_3`, the inner importance k-hat, and the
combined whole-fit band.
Plots: `plot_rhat()`, `plot_ess()`, `plot_acf()`, `plot_energy()`,
`plot_divergences()`, `plot_pairs()`, `plot_diagnostics()`, `plot_map()`,
`plot_map_panel()`.
Model comparison: `compare_models()`, `model_average()` (stacking / pseudo-BMA).
Residual / GOF: `moran_i()`, `durbin_watson()`, `tulpa_variogram()`,
`pit_residuals()`, `tulpa_pit()` (the PIT from a predictive CDF),
`test_uniformity()`, `test_dispersion()`, `test_outliers()`,
`test_zero_inflation()`, `check_model()`.
Derived effects: `spatial_range()`, `temporal_corr()`, `post_hoc_lm()`.

## Calibration (simulation-based calibration)

The diagnostics above score ONE fit: whether the machinery that produced that
posterior behaved. `sbc()` scores something a single fit cannot answer --
whether an inference algorithm's posteriors are CALIBRATED across the
generative model -- by reading the whole marginal CDF instead of the one or two
nominal levels a coverage sweep reads.

| Function | Purpose |
|---|---|
| `sbc(experiment, ...)` | The verb. `"prior_predictive"` is ordinary SBC (Talts et al. 2018), averaged over the prior. `"posterior"` is calibration conditional on an observed data set (Sailynoja et al. 2026, Alg. 2): `theta' ~ pi(theta \| y_obs)`, a replicate at `theta'`, PIT under the augmented `pi(theta \| y, y_obs)`. |
| `sbc_mixture()`, `sbc_normal()`, `sbc_discrete()`, `sbc_rank()`, `sbc_draws()` | The predictive SHAPES a fitter callback reports per (arm, quantity). Everything downstream dispatches on the `kind` tag, so a new backend shape is one entry in three switches rather than a parallel scorer. |
| `summary(res, baseline = )` | Adds the PAIRED CRPS ranking against one arm, seed by seed. Refused on an experiment where the CRPS is not a proper posterior score. |
| `plot(res, folded = )` | The PIT ECDF difference from uniform against the simultaneous band. |
| `diagnostics(res)` | The per-(arm, quantity) report as a data frame. |

Three reads of the same PIT sample: the raw ECDF against an EXACT simultaneous
band (a pointwise binomial band is not one -- at n = 100 it holds all order
statistics together only 44.71% of the time); the FOLDED PIT `2 |u - 1/2|`,
where a symmetric dispersion error shows after cancelling in the raw ECDF; and
the CRPS, closed form for a Gaussian mixture so the nested tier's own posterior
is scored with no Monte Carlo. Every discrete PIT is randomized within its atom.

Two guards ride the door, each recorded on `res$premises`. The prior-predictive
path refuses a scored quantity whose truth does not move across simulations
(what the nested door's improper fixed-effect prior looks like from outside)
unless it is named in `flat_prior`. The posterior path refuses a `pool()`
returning no more than one of its inputs, and verifies disjoint group LABELS
when `model$group_ids` is supplied.

See `vignette("sbc")` and `?sbc`.

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

`rubins_pool()` (multiple-imputation pooling), `ccd_grid()` / `ccd_weights()` /
`ccd_to_theta()` (central composite design nodes, the corrected R-INLA
integration weights, and the whitening map, for high-dimensional outer
integration), `sn_cdf()` / `sn_quantile()` / `sn_match()` (skew-normal helpers).

Outer-grid axis provenance: `hyper_axis_spec()` describes one axis, and
`auto_grid()` / `is_auto_grid()` mark and read whether a grid setting is an
engine DEFAULT or a caller PIN. Field presence is not provenance -- a consumer
package stamping an engine default onto a fit made it read as a user pin, which
left the auto-recentring pass inert (gcol33/tulpa#293), so the mark is what any
placement pass has to consult.

---

# Part 2 -- C++ model-package API (`LinkingTo: tulpa`)

A model package (tulpaObs, tulpaRatio) supplies its **likelihood**;
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
| `ll_double`, `ll_arena` | `LikelihoodFn<T>` for each AD mode (double / reverse-arena). `AUTODIFF_FWD` resolves to the arena path, so there is no separate forward-dual slot. |
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
