# TULPA — Template Unified Latent Process Architecture

General-purpose Bayesian hierarchical modelling engine. Extracted from numdenom.

## Architecture

Three-package ecosystem:
- **tulpa**: Engine — samplers, autodiff, spatial, temporal, priors, formula infrastructure
- **numdenom**: Ratio models (Depends: tulpa)
- **tulpaOcc**: Occupancy models (Depends: tulpa)

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
- Generic diagnostics: moranI, durbinWatson, variogram, compare_models, modelAverage
- Generic plotting: trace, density, pairs plots of posterior draws
- Rubin's rules pooling
- Parameter back-transformation (logit → probability)

**Model packages own** (e.g., tulpaOcc, numdenom):
- Likelihood functions (LikelihoodSpec)
- E-step weight computation (model-specific latent variable posterior)
- Data structures and encoding (how to map model data → binomial pseudo-data for Laplace)
- Model-specific diagnostics (waicOccu, ppcOccu, pitResiduals)
- Model-specific fitted/residuals/simulate
- Data formatting and simulation functions
- Print methods referencing model-specific parameter names

### TODO: EM+Laplace Engine for tulpa

tulpa needs a generic EM+Laplace engine that model packages can plug into.
Currently the Laplace M-step exists (`tulpa_laplace`, `cpp_laplace_fit_*`),
but the EM loop, MI correction, and Gibbs correction are missing.

**Proposed API:**

```r
# Generic EM engine: model package provides callbacks
tulpa_em_laplace(
  e_step,           # function(psi_hat, p_hat, ...) → list(weights, ...)
  m_step_encode,    # function(weights, ...) → list of (y, n_trials, X, ...) per submodel
  spatial = NULL,
  re_list = list(),
  max_iter = 50L,
  tol = 1e-4,
  damping = 0.3,
  correction = c("auto", "mi", "gibbs", "none"),
  n_imputations = 20L,    # MI: draws per round
  n_gibbs = 10L,           # Gibbs: total iterations
  verbose = TRUE
)
```

**E-step callback** (model-specific): Takes current parameter estimates,
returns posterior weights for latent variables. For occupancy:
`P(z_i=1 | y_i, psi_i, p_i)`.

**M-step encode callback** (model-specific): Takes weights, returns
submodel data (y, n_trials, X) ready for `tulpa_laplace()`. For occupancy:
builds binomial pseudo-data for psi and detection submodels.

**MI correction** (generic in tulpa): Draws hard z from weights,
refits submodels unweighted, pools via Rubin's rules. K=20 default.

**Gibbs correction** (generic in tulpa): Markov chain z|params → fit|z,
warm-started. ~10 iterations with burn-in.

### TODO: Generic S3 Methods

Move from tulpaOcc to tulpa (operating on `tulpa_fit` base class):
- `coef.tulpa_fit`, `confint.tulpa_fit`, `vcov.tulpa_fit`
- `logLik.tulpa_fit`, `summary.tulpa_fit`
- `plot.tulpa_fit` (trace/density/pairs)
- `tidy.tulpa_fit`, `glance.tulpa_fit`, `ranef.tulpa_fit`

Model packages inherit these via `class = c("tulpaOcc_fit", "tulpa_fit")`.

### TODO: Generic Diagnostics

Move from tulpaOcc to tulpa:
- `moranI(residuals, coords, k)` — spatial autocorrelation
- `durbinWatson(residuals)` — temporal autocorrelation
- `variogram(residuals, coords, n_bins)` — empirical semivariogram
- `compare_models(..., criterion)` — WAIC/AIC/BIC comparison with weights
- `modelAverage(..., criterion)` — Burnham & Anderson model averaging

### Matrix CHOLMOD Fix

tulpa's `R_init_tulpa` calls `M_cholmod_start` which requires Matrix's
CHOLMOD stubs. Fixed by adding `@importFrom Matrix sparseMatrix` to
`tulpa-package.R` so Matrix DLL loads before tulpa's init.

## Origin

Engine extracted from numdenom (82K lines, faster than Stan on all 18 benchmarks).
Name: Twin Peaks reference + acronym (Template Unified Latent Process Architecture).
