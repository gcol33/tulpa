# tulpa

*Bayesian models without the full sampling bill*

[![Lifecycle: experimental](https://lifecycle.r-lib.org/articles/figures/lifecycle-experimental.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![R >= 4.1](https://img.shields.io/badge/R-%3E%3D%204.1-blue.svg)](https://cran.r-project.org/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/17)

**A Bayesian hierarchical modelling engine that splits the posterior: deterministic approximation (Laplace, EP, VI, Pathfinder) on the well-behaved blocks, exact MCMC on the residual directions where the approximation is biased.**

INLA nests deterministic approximations for speed and applies no exact correction. Stan runs exact MCMC and pays the full sampling price on every block, including the many a Laplace approximation handles exactly. `tulpa` nests the approximation over the Gaussian-latent structure, then debiases only the directions where that approximation is wrong, so it keeps nested-approximation speed while recovering exact-MCMC calibration where it counts. Spatial fields, temporal effects, spatially varying coefficients, and free random-effect covariances are built in. Model packages supply an observation likelihood through a templated C++ interface and inherit the rest.

```r
library(tulpa)

# A Poisson GLMM: counts with a site-level random intercept
set.seed(1)
site <- rep(seq_len(40), each = 5)
b    <- rnorm(40, 0, 0.7)
x    <- rnorm(200)
y    <- rpois(200, exp(0.2 + 0.5 * x + b[site]))
d    <- data.frame(y, x, site = factor(site))

# mode = "auto" picks a reliable Tier 1 / Tier 2 method and records why
fit <- tulpa(y ~ x + (1 | site), data = d, family = "poisson")
summary(fit)
fit$selection_reason
```

The fit is a `tulpa_fit` object with generic accessors (`coef`, `confint`, `vcov`, `summary`, `tidy`, `glance`, `ranef`) and a full diagnostics surface (`mcmc_diagnostics`, `check_model`, `pp_check`).

## Tiers encode correctness, not the hot path

Inference is organised so the tier you pick states the guarantee you get, not which code path is fastest. Tier 1 is exact MCMC; Tier 2 is Laplace with optional Tier-1 debiasing; Tier 3 is variational. `mode = "auto"` chooses between Tier 1 and Tier 2 and records its reasoning on `fit$selection_reason`. Tier 3 is never selected silently.

|        | Nested approximation                      | Exact-MCMC correction                   |
|--------|-------------------------------------------|-----------------------------------------|
| INLA   | yes                                       | no — biased on non-Gaussian residuals   |
| Stan   | no — pays the full price on every block   | yes                                     |
| tulpa  | yes                                       | yes                                     |

| Tier   | Methods                                                          |
|--------|------------------------------------------------------------------|
| Tier 1 | HMC/NUTS, MALA, IMH-Laplace, Gibbs (exact)                       |
| Tier 2 | Laplace, EM+Laplace, nested Laplace + CCD (deterministic)        |
| Tier 3 | Variational inference, Pathfinder (opt-in)                       |

Correction layers — importance sampling and multiple-imputation / Gibbs debiasing — sit on top of the Tier-2 fit and pool through Rubin's rules. Canonical compositions, each named for the layer it adds:

- **IMH-Laplace** — Laplace body, Metropolis–Hastings bias correction.
- **Pathfinder** — L-BFGS mode, Gaussian fit, ELBO scoring.
- **Nested Laplace + CCD** — inner Laplace, outer hyperparameter integration.

This makes the engine useful for spatial and spatio-temporal regression (disease mapping, abundance, occupancy), hierarchical models with correlated random slopes and free covariances, and workflows where calibrated uncertainty matters as much as point estimates.

## Latent structure

| Layer     | Choices                                          |
|-----------|--------------------------------------------------|
| Spatial   | HSGP, NNGP, ICAR, BYM2, CAR (proper), SPDE, RSR  |
| Temporal  | RW1, RW2, AR1, GP, multiscale                    |
| Other     | Random effects, SVC, TVC, latent factors         |

Spatial fields are specified through helpers (`spatial_bym2()`, `spatial_gp()`, `spatial_hsgp()`, `spatial_spde()`, …) and temporal structure through `temporal_rw1()`, `temporal_ar1()`, `temporal_gp()`, and friends.

```r
# Areal BYM2 field: a spatial(unit) term plus the structure spec
fit <- tulpa(
  cases ~ covariate + spatial(region),
  data    = areal_df,
  family  = "poisson",
  spatial = list(type = "bym2", adjacency = W)
)

# Continuous Gaussian-process field, addressed by coordinates
fit <- tulpa(
  y ~ x,
  data    = point_df,
  family  = "gaussian",
  spatial = spatial_gp(~ lon + lat)
)
```

## Random-effect covariances as inferred quantities

For random-slope terms the engine treats the covariance `Sigma` itself as the inferred quantity rather than a point estimate — the nested-approximation + debias philosophy applied to a free `Sigma`. Three routes are available, and all summarise derived quantities (`sigma_i`, `rho_ij`) by marginalising the joint posterior:

- `tulpa_re_cov_nested()` — nested-Laplace integration over `Sigma` in log-Cholesky coordinates.
- `tulpa_re_cov_gibbs()` — exact debias via Metropolis-within-Gibbs with conjugate inverse-Wishart draws.
- `tulpa_re_aghq()` — deterministic debias via adaptive Gauss–Hermite quadrature.

## Forcing a backend

```r
# Let auto pick, and read its reasoning
fit <- tulpa(y ~ x + (1 | site), data = d, family = "poisson")
fit$selection_reason

# Exact MCMC
fit <- tulpa(y ~ x + (1 | site), data = d, family = "poisson", mode = "mala")

# Cheap deterministic fit conditioning on a fixed hyperparameter
fit <- tulpa(y ~ x + (1 | site), data = d, family = "poisson", mode = "laplace")
```

## Diagnostics

Rhat (improved rank-normalised / folded split-Rhat, Vehtari et al. 2021), bulk/tail ESS, and MCSE are implemented natively and reproduce `posterior` to ~1e-12, so downstream packages call `tulpa::mcmc_diagnostics()` rather than re-deriving them.

```r
summary(fit)
coef(fit)
ranef(fit)
mcmc_diagnostics(fit)          # Rhat, bulk/tail ESS
check_model(fit)               # posterior-predictive + residual diagnostics
```

Posterior-predictive checks (`pp_check`, `check_model`), residual tests (`pit_residuals`, `test_dispersion`, `test_zero_inflation`), and spatial/temporal diagnostics (`moran_i`, `durbin_watson`, `tulpa_variogram`) round out the surface.

## Model packages

`tulpa` ships the engine. Observation likelihoods live in companion packages that link against it via `LinkingTo: tulpa`. A model package supplies a templated `LikelihoodSpec` and a small amount of data encoding; the spatial, temporal, prior, and inference machinery comes from `tulpa`.

- [tulpaObs](https://github.com/gcol33/tulpaObs) — single-season, dynamic, community, integrated, and JSDM occupancy; abundance; distance sampling; removal; false-positive occupancy; and hurdle-Beta cover.
- `tulpaRatio` (in progress) — ratio models (numerator / denominator likelihoods on shared spatial fields).

## Custom latent blocks

User-supplied templated C++ snippets compile against `tulpa`'s autodiff types (`A`, `A_r`) and register as new latent structures. Because the snippet uses the same AD types as the built-in blocks, custom blocks work under every tier including NUTS, with full gradient flow through the user code. For a Gaussian GMRF block, `tgmrf()` offers a pure-R route: supply `Q(theta)` and `mu(theta)` as closures and the engine derives the closed-form score and Hessian numerically, with no DSL, parser, or codegen (`vignette("tgmrf", package = "tulpa")`).

| Interface        | Mechanism                              | Exact-MCMC support |
|------------------|----------------------------------------|--------------------|
| INLA `rgeneric`  | R callback, no autodiff                | Laplace tier only  |
| INLA `cgeneric`  | C function, no autodiff                | no                 |
| Stan             | full DSL with parser + codegen         | yes                |
| TMB              | templated C++ snippet (CppAD)          | yes (closest analog) |
| **tulpa**        | templated C++ snippet, shared AD types | yes — all tiers    |

## Installation

```r
# Development version from GitHub
install.packages("pak")
pak::pak("gcol33/tulpa")

# Pin a release
pak::pak("gcol33/tulpa@v0.0.2")
```

`pak` resolves the dependency tree, including `tulpaMesh` (declared in `Remotes:`). `tulpa` compiles its C++ backend on first install, so a C++17 toolchain is required: Rtools on Windows, Xcode CLI tools on macOS, `r-base-dev` on Linux.

## Documentation

- `vignette("tgmrf", package = "tulpa")` — user-defined GMRF latent blocks.
- Function reference via `?tulpa`, `?tulpa_nested_laplace`, `?tulpa_re_cov_nested`, `?mcmc_diagnostics`.

## Status

Experimental — expect breaking changes. The C++ ABI version (`TULPA_ABI_VERSION` in `inst/include/tulpa/model_data.h`) is checked at runtime, so a mismatch between `tulpa` and a model package fails with a readable error instead of a segfault.

## Support

> "Software is like sex: it's better when it's free." — Linus Torvalds

I'm a PhD student who builds R packages in my free time because I believe good tools should be free and open. I started these projects for my own work and figured others might find them useful too.

If this package saved you some time, buying me a coffee is a nice way to say thanks. It helps with my coffee addiction.

[![Buy Me A Coffee](https://img.shields.io/badge/-Buy%20me%20a%20coffee-FFDD00?logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/gcol33)

## License

MIT (see the LICENSE file).

## Citation

```bibtex
@Manual{tulpa,
  author = {Colling, Gilles},
  title  = {tulpa: Template Unified Latent Process Architecture for Bayesian Hierarchical Models},
  year   = {2026},
  note   = {R package version 0.0.2},
  url    = {https://github.com/gcol33/tulpa}
}
```
