# TULPA — Template Unified Latent Process Architecture

General-purpose Bayesian hierarchical modelling engine. Extracted from numdenom.

## Architecture

Three-package ecosystem:
- **tulpa**: Engine — samplers, autodiff, spatial, temporal, priors, formula infrastructure
- **numdenom**: Ratio models (Depends: tulpa)
- **tulpaObs**: Occupancy models (Depends: tulpa)

Model packages plug likelihoods into tulpa via `LikelihoodSpec` (see `inst/include/tulpa/likelihood.h`).

## Key Design Principles

1. **Tier system is non-negotiable**: Tier 1 (exact MCMC), Tier 2 (Laplace), Tier 3 (VI). Auto mode never silently chooses Tier 3.
2. **Gradient progression N→A→A_r→H**: Never skip stages. All modes remain available.
3. **Runtime gradient verification**: Before sampling, verify active gradient against numerical.
4. **Model packages own their likelihood**: tulpa assembles linear predictors; model packages compute log-likelihood.
5. **No copy-paste logic**: Shared sub-computations in helpers, not duplicated across specialized functions.

## C++ Interface

- Exported headers: `inst/include/tulpa/`
- Model packages use `LinkingTo: tulpa` in DESCRIPTION
- Key types: `ModelData`, `ParamLayout`, `LikelihoodSpec`, `LikelihoodFn<T>`, `ResidualFn`
- **ABI version**: `TULPA_ABI_VERSION` in `model_data.h`. Bump when any exported struct layout
  changes. Model packages auto-check on first NUTS call — mismatch gives a clear error instead
  of segfault. Registered via `R_RegisterCCallable("tulpa", "tulpa_get_abi_version", ...)`.
- **Legacy fields**: `ModelData` contains deprecated ratio-specific fields (y_num, y_denom,
  X_num/denom_flat, model_type). Used only when `n_processes == 0`. Will move to numdenom.
  New fields go in the stable sections — never insert before existing fields.

## Building

```r
devtools::load_all()
devtools::check(args = "--no-manual")
```

## File Organization

```
R/          — Spatial, temporal, priors, backends, formula, validation
src/        — HMC/NUTS, autodiff, spatial priors, temporal priors, Laplace, VI, ESS
inst/include/tulpa/ — Exported C++ headers for model packages
tests/testthat/     — Unit and integration tests
```

## Boundary: What Belongs in tulpa vs Model Packages

**tulpa owns** (generic, model-agnostic):
- Inference engines: NUTS, Laplace, VI, ESS, EM+Laplace, MI correction, Gibbs correction
- Autodiff: arena, forward, tape
- Latent structure: spatial, temporal, RE, SVC, TVC, ST, latent factors
- Generic S3 methods operating on posterior draws: coef, confint, vcov, logLik, summary
- Generic diagnostics: moran_i, durbin_watson, tulpa_variogram, compare_models, modelAverage
- Generic plotting: trace, density, pairs plots of posterior draws
- Rubin's rules pooling
- Parameter back-transformation (logit → probability)

**Model packages own** (e.g., tulpaObs, numdenom):
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
parameters fired between M-step and E-step (gcol33/tulpa#4). MI / Gibbs
corrections are stubbed (`correction = "mi"/"gibbs"` raises a clear error).
See `?tulpa_em_laplace`.

### Generic S3 Methods and Diagnostics

Implemented in `R/methods_generic.R` (`coef`, `confint`, `vcov`, `logLik`,
`summary`, `plot`, `tidy`, `glance`, `ranef`), `R/diagnostics_generic.R`
(`compare_models`, `modelAverage`, `spatialRange`, `temporalCorr`), and
`R/diagnostics_sim.R` (`moran_i`, `durbin_watson`, `tulpa_variogram`,
`pit_residuals`, `test_uniformity`, `test_dispersion`, `test_outliers`,
`test_zero_inflation`, `check_model`).
Model packages inherit via `class = c("model_fit", "tulpa_fit")`.

### Matrix CHOLMOD Fix

tulpa's `R_init_tulpa` calls `M_cholmod_start` which requires Matrix's
CHOLMOD stubs. Fixed by adding `@importFrom Matrix sparseMatrix` to
`tulpa-package.R` so Matrix DLL loads before tulpa's init.

## Origin

Engine extracted from numdenom (82K lines, faster than Stan on all 18 benchmarks).
Name: Twin Peaks reference + acronym (Template Unified Latent Process Architecture).
