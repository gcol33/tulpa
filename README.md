# tulpa

*hierarchical Bayesian models, approximated then corrected*

[![Lifecycle: experimental](https://lifecycle.r-lib.org/articles/figures/lifecycle-experimental.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![R >= 4.1](https://img.shields.io/badge/R-%3E%3D%204.1-blue.svg)](https://cran.r-project.org/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/17)

**A Bayesian hierarchical modelling engine that splits the posterior: a deterministic approximation (Laplace, EP, VI, Pathfinder) handles the well-behaved Gaussian-latent blocks, and exact MCMC corrects only the directions where that approximation is biased.** Autodiff, the samplers, and the spatial / temporal / covariance machinery are templated C++.

Two established tools sit at opposite ends. INLA fits these models with fast deterministic approximations and applies no exact correction. Stan runs exact MCMC and pays the full sampling price on every block, including the many a deterministic approximation handles exactly. `tulpa` nests the approximation over the Gaussian-latent structure, then debiases only the directions where it is wrong. The deterministic body carries most of the posterior; the MCMC correction runs over the residual directions alone, a low-dimensional remainder — so the cost stays near a deterministic approximation instead of scaling like full MCMC over the whole model, while the calibration matches exact MCMC where it counts.

```r
library(tulpa)

# Poisson counts with a correlated site-level intercept and slope
set.seed(1)
site <- rep(seq_len(40), each = 8)
x    <- rnorm(320)
b0   <- rnorm(40, 0, 0.7)                          # site intercepts (truth SD 0.7)
b1   <- rnorm(40, 0, 0.4)                          # site slopes     (truth SD 0.4)
y    <- rpois(320, exp(0.2 + (0.5 + b1[site]) * x + b0[site]))
d    <- data.frame(y, x, site = factor(site))

fit <- tulpa(y ~ x + (1 + x | site), data = d, family = "poisson")

fit$selection_reason
#> "random-slope term(s) present; RE covariance(s) integrated via re_cov_gibbs (1 block(s))"

summary(fit)
#>             estimate  std.error    2.5%   97.5%
#> (Intercept)    0.214      0.103  -0.003   0.401
#> x              0.488      0.087   0.322   0.658

fit$posterior          # the random-effect covariance, marginalized to sigma / rho
#>   parameter    mean     sd   median   ci_lo   ci_hi
#>     sigma_1   0.688  0.102    0.678   0.516   0.918   # intercept SD (truth 0.7)
#>     sigma_2   0.469  0.074    0.461   0.345   0.636   # slope SD     (truth 0.4)
#>      rho_12  -0.145  0.206   -0.156  -0.523   0.254   # correlation  (truth 0)
```

`mode = "auto"` picks a backend and records the reason on the fit — the dial neither INLA nor Stan hands you on a single object. The random-effect covariance is not a plug-in point estimate: `sigma_1`, `sigma_2`, and `rho_12` are read off the joint posterior with full credible intervals. The result is a `tulpa_fit` with the usual accessors (`coef`, `confint`, `vcov`, `summary`, `tidy`, `glance`, `ranef`) and a full diagnostics surface (`mcmc_diagnostics`, `check_model`, `pp_check`).

## One front door, every backend

`tulpa()` is the only verb you need. The same call fits a plain GLMM, a spatial field, a temporal effect, or a free random-effect covariance; `mode` and `control` pick how hard the correctness check runs, without changing the front door.

```r
# Default: auto routes and records why on the fit
fit <- tulpa(y ~ x + (1 + x | site), data = d, family = "poisson")
fit$selection_reason
#> "random-slope term(s) present; RE covariance(s) integrated via re_cov_gibbs (1 block(s))"

# Same model, cheaper deterministic covariance integration (nested Laplace + CCD)
fit <- tulpa(y ~ x + (1 + x | site), data = d, family = "poisson",
             control = list(re_cov = "nested"))
fit$selection_reason
#> "random-slope term(s) present; RE covariance(s) integrated via re_cov_nested (1 block(s))"

# Drop the correlation: an uncorrelated (diagonal) covariance block
fit <- tulpa(y ~ x + (1 + x || site), data = d, family = "poisson")
```

## Tiers encode correctness

Each tier names a correctness guarantee; which code path runs fastest depends on the model and the data. `mode = "auto"` chooses between Tier 1 and Tier 2 and records why. Tier 3 is opt-in.

| Tier   | Guarantee        | Methods                                                   |
|--------|------------------|----------------------------------------------------------|
| Tier 1 | Exact MCMC       | HMC/NUTS, MALA, IMH-Laplace, Gibbs                        |
| Tier 2 | Laplace          | Laplace, EM+Laplace, nested Laplace + CCD                 |
| Tier 3 | Variational      | Variational inference, Pathfinder (opt-in)               |

Correction layers — importance sampling and multiple-imputation / Gibbs debiasing — sit on top of a Tier-2 fit and pool through Rubin's rules. Each composition is named for the layer it adds:

- **IMH-Laplace** — Laplace body, Metropolis–Hastings bias correction.
- **Pathfinder** — L-BFGS mode, Gaussian fit, ELBO scoring.
- **Nested Laplace + CCD** — inner Laplace, outer hyperparameter integration.

## Latent structure

| Layer     | Choices                                          |
|-----------|--------------------------------------------------|
| Spatial   | HSGP, NNGP, ICAR, BYM2, CAR (proper), SPDE, RSR  |
| Temporal  | RW1, RW2, AR1, GP, multiscale                    |
| Other     | Random effects, SVC, TVC, latent factors         |

Spatial fields are specified through helpers (`spatial_bym2()`, `spatial_gp()`, `spatial_spde()`, …; the Hilbert-space GP is `spatial_gp(approx = "hsgp")`) and temporal structure through `temporal_rw1()`, `temporal_ar1()`, `temporal_gp()`, and friends.

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

This makes the engine useful for spatial and spatio-temporal regression (disease mapping, abundance, occupancy), hierarchical models with correlated random slopes and free covariances, and workflows where calibrated uncertainty matters as much as point estimates.

## Random-effect covariances as inferred quantities

For random-slope terms the engine infers the full posterior of the covariance `Sigma` itself — the approximation + debias philosophy applied to a free `Sigma`. The `sigma_i` / `rho_ij` table in the flagship above is not a point estimate: each derived quantity is summarised by marginalising the joint posterior rather than plugging in the mode of its components. Three routes share that summary, selectable from the front door or callable directly:

- `tulpa_re_cov_gibbs()` — exact debias via Metropolis-within-Gibbs with conjugate inverse-Wishart draws (the auto default for slopes).
- `tulpa_re_cov_nested()` — nested-Laplace integration over `Sigma` in log-Cholesky coordinates (`control = list(re_cov = "nested")`).
- `tulpa_re_aghq()` — deterministic debias via adaptive Gauss–Hermite quadrature.

Correlated `(1 + x | g)`, uncorrelated `(1 + x || g)`, multiple terms, and any accompanying `(1 | g)` are each handled as covariance blocks; nothing is silently conditioned at a fixed variance.

## Diagnostics

Rhat (improved rank-normalised / folded split-Rhat, Vehtari et al. 2021), bulk/tail ESS, and MCSE are implemented natively and reproduce `posterior` to ~1e-12, so downstream packages call `tulpa::mcmc_diagnostics()` directly. For deterministic (non-chain) fits the engine reports Pareto-k̂ — the accuracy counterpart to Rhat — instead of withholding a vacuous convergence pass.

```r
summary(fit)
coef(fit)
ranef(fit)
mcmc_diagnostics(fit)          # Rhat, bulk/tail ESS
check_model(fit)               # posterior-predictive + residual diagnostics
```

Posterior-predictive checks (`pp_check`, `check_model`), residual tests (`pit_residuals`, `test_dispersion`, `test_zero_inflation`), and spatial/temporal diagnostics (`moran_i`, `durbin_watson`, `tulpa_variogram`) round out the surface.

## Model packages

`tulpa` ships the engine. Observation likelihoods live in companion packages that link against it via `LinkingTo: tulpa`: a model package supplies a templated `LikelihoodSpec` and a little data encoding, and inherits the spatial, temporal, prior, and inference machinery.

- [tulpaObs](https://github.com/gcol33/tulpaObs) — single-season, dynamic, community, integrated, and JSDM occupancy; abundance; distance sampling; removal; false-positive occupancy; and hurdle-Beta cover.
- `tulpaRatio` — ratio, rate, and proportion models (numerator / denominator likelihoods on shared latent fields). Release pending.

## Custom latent blocks

User-supplied templated C++ snippets compile against `tulpa`'s autodiff types (`A`, `A_r`) and register as new latent structures. Because the snippet uses the same AD types as the built-in blocks, custom blocks work under every tier including NUTS, with full gradient flow through the user code. For a Gaussian GMRF block, `tgmrf()` offers a pure-R route: supply `Q(theta)` and `mu(theta)` as closures and the engine derives the score and Hessian numerically, with no DSL, parser, or codegen (`vignette("tgmrf", package = "tulpa")`).

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
pak::pak("gcol33/tulpa@v0.0.92")
```

`pak` resolves the dependency tree, including `tulpaMesh` (on CRAN, used for SPDE mesh construction). `tulpa` compiles its C++ backend on first install, so a C++17 toolchain is required: Rtools on Windows, Xcode CLI tools on macOS, `r-base-dev` on Linux.

## Documentation

Getting started:

- [Quickstart](https://gcol33.github.io/tulpa/articles/quickstart.html)
- [Data formatting](https://gcol33.github.io/tulpa/articles/data-formatting.html)

Inference:

- [Inference modes](https://gcol33.github.io/tulpa/articles/inference-modes.html)
- [Priors](https://gcol33.github.io/tulpa/articles/priors.html)
- [Random slopes and free covariances](https://gcol33.github.io/tulpa/articles/random-slopes.html)
- [EM + Laplace](https://gcol33.github.io/tulpa/articles/em-laplace.html)
- [Model comparison](https://gcol33.github.io/tulpa/articles/model-comparison.html)
- [Reliability and Pareto-k](https://gcol33.github.io/tulpa/articles/reliability-pareto-k.html)

Latent structure:

- [Spatial models](https://gcol33.github.io/tulpa/articles/spatial-models.html)
- [Temporal models](https://gcol33.github.io/tulpa/articles/temporal-models.html)

Extending the engine:

- [Custom GMRF latent blocks](https://gcol33.github.io/tulpa/articles/tgmrf.html)
- [Checkpoint and resume](https://gcol33.github.io/tulpa/articles/checkpoint.html)

The [function reference](https://gcol33.github.io/tulpa/reference/) lists every fitter, accessor, and diagnostic; `?tulpa` is the front door.

## Status

Experimental — expect breaking changes. The C++ ABI version (`TULPA_ABI_VERSION` in `inst/include/tulpa/model_data.h`) is checked at runtime, so a mismatch between `tulpa` and a model package is caught at load and reported with a readable error.

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
  note   = {R package version 0.0.92},
  url    = {https://github.com/gcol33/tulpa}
}
```
