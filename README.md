
# tulpa <img src="man/figures/logo.svg" align="right" height="139" alt="tulpa logo" />

*templated unified latent process approximations*

<!-- badges: start -->

[![Lifecycle:
experimental](https://lifecycle.r-lib.org/articles/figures/lifecycle-experimental.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![R \>=
4.1](https://img.shields.io/badge/R-%3E%3D%204.1-blue.svg)](https://cran.r-project.org/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=cplusplus&logoColor=white)](https://en.cppreference.com/cpp/17)
[![Codecov test
coverage](https://codecov.io/gh/gcol33/tulpa/graph/badge.svg)](https://app.codecov.io/gh/gcol33/tulpa)
<!-- badges: end -->

tulpa fits Bayesian hierarchical models in R, including models for
counts, proportions, repeated measurements, and observations distributed
across space or time.

A formula describes how measured predictors and unobserved effects
contribute to the response. The engine combines these effects with an
observation distribution and fits the model using nested Laplace
approximation or Markov chain Monte Carlo (MCMC). Its numerical routines
are written in templated C++. The result includes posterior estimates,
uncertainty intervals, and diagnostics that help you assess the fit.

tulpa supports:

- Random intercepts and slopes, including uncertainty in their variances
  and correlations.
- Smooth covariate effects, spatial fields, temporal processes, and
  combinations of these structures.
- Deterministic approximations, MCMC sampling, and correction methods
  selected through the same fitting function.
- Model-specific observation likelihoods and custom latent processes
  supplied through R or C++ interfaces.

The [calibration
article](https://gillescolling.com/tulpa/articles/sbc.html) shows how to
assess posterior calibration through repeated simulations. [Reliability
diagnostics](https://gillescolling.com/tulpa/articles/reliability-pareto-k.html)
explain what can be checked on an individual fit. The package is
experimental and its interface may change.

## Installation

Install from r-universe:

``` r
install.packages(
  "tulpa",
  repos = c("https://gcol33.r-universe.dev",
            "https://cloud.r-project.org")
)
```

Source installations require a C++17 compiler and the build tools for R.

| Operating system | Build tools |
|:---|:---|
| Windows | [Rtools](https://cran.r-project.org/bin/windows/Rtools/) matching your R version |
| macOS | Xcode Command Line Tools: `xcode-select --install` |
| Debian / Ubuntu | `r-base-dev` and a C++17 compiler |

Check the toolchain before installing the development version:

``` r
# install.packages(c("pkgbuild", "pak"))
pkgbuild::has_build_tools(debug = TRUE)
pak::pak("gcol33/tulpa")
```

The build links against R’s BLAS and LAPACK libraries for matrix
operations. Package dependencies, including `tulpaMesh` for spatial mesh
construction, are installed with tulpa. R’s [installation
guide](https://cran.r-project.org/doc/manuals/r-release/R-admin.html)
describes the platform toolchains.

### Optional multithreading

The nested-Laplace fitters expose OpenMP thread controls through their
`control` argument. The
[single-process](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.html)
and
[joint-model](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.html)
references describe the available controls. The build uses the OpenMP
flags provided by R’s toolchain.

On macOS, OpenMP requires a runtime compatible with the compiler used to
build R packages. Follow the R project’s [macOS OpenMP
setup](https://mac.r-project.org/openmp/) before compiling tulpa with
OpenMP enabled.

## Usage

1.  Fit a model with `tulpa()`.
2.  Inspect the estimates and inference method with `summary()`.
3.  Extract the group effects with `ranef()`.

The built-in `InsectSprays` data contain insect counts and the spray
treatment used for each observation. This Poisson model gives each
treatment a random intercept, estimating its effect relative to the
overall log count.

``` r
library(tulpa)

fit <- tulpa(
  count ~ (1 | spray),
  data = InsectSprays,
  family = "poisson"
)
summary(fit)
```

``` r
effects <- ranef(fit)
```

| Treatment  | Estimate | Lower | Upper |
|:-----------|---------:|------:|------:|
| spray\[A\] |     0.74 | -0.02 |  1.48 |
| spray\[B\] |     0.79 |  0.00 |  1.53 |
| spray\[C\] |    -1.14 | -1.96 | -0.34 |
| spray\[D\] |    -0.33 | -1.11 |  0.43 |
| spray\[E\] |    -0.67 | -1.48 |  0.10 |
| spray\[F\] |     0.87 |  0.10 |  1.61 |

Supply a data frame with the response, predictors, and grouping
variables named in the formula. Choose an observation family appropriate
to the response. The [data-formatting
guide](https://gillescolling.com/tulpa/articles/data-formatting.html)
covers response encodings, grouping variables, and spatial inputs.

## Random effects, smoothers, and fields

Formula terms describe the model structure: `(1 | group)` adds random
intercepts, `(1 + x | group)` adds correlated random intercepts and
slopes, and `s(x)` adds a smooth covariate effect. Spatial and temporal
specifications describe dependence between locations or times.

[Random slopes and
covariances](https://gillescolling.com/tulpa/articles/random-slopes.html)
walks through inference for group-level variances and correlations. The
[spatial](https://gillescolling.com/tulpa/articles/spatial-models.html)
and
[temporal](https://gillescolling.com/tulpa/articles/temporal-models.html)
articles show how to supply the corresponding structures.

## Inference and diagnostics

Automatic mode selects an inference method and records its reason on the
fit. An explicit mode selects MCMC, Laplace approximation, or
variational inference; variational methods require an explicit choice.
The [inference
guide](https://gillescolling.com/tulpa/articles/inference-modes.html)
describes the methods and the models each supports.

Diagnostics depend on how the posterior was computed. MCMC fits report
chain convergence and effective sample sizes. Approximation fits report
the available integration and inner-approximation checks. The
[reliability
guide](https://gillescolling.com/tulpa/articles/reliability-pareto-k.html)
explains how to interpret these results and their limits.

## Custom models and processes

Companion packages supply observation likelihoods and reuse tulpa’s
inference and latent structures.
[tulpaObs](https://github.com/gcol33/tulpaObs) provides models for
occupancy, abundance, and detection; tulpaRatio provides models for
ratios, rates, and proportions.

For a custom Gaussian latent process, supply its precision matrix and
mean through R functions or a compiled C++ definition. The
[custom-process
tutorial](https://gillescolling.com/tulpa/articles/tgmrf.html) shows
both interfaces. The [EM + Laplace
article](https://gillescolling.com/tulpa/articles/em-laplace.html)
describes the shared fitting framework for models with unobserved
observation states.

## Documentation

Start with the
[quickstart](https://gillescolling.com/tulpa/articles/quickstart.html)
for a complete path from a formula to a fitted model.

- [Priors](https://gillescolling.com/tulpa/articles/priors.html):
  specify prior distributions for coefficients and latent structures.
- [Model
  comparison](https://gillescolling.com/tulpa/articles/model-comparison.html):
  compare fitted models and combine predictions.
- [Calibration](https://gillescolling.com/tulpa/articles/sbc.html):
  assess posterior calibration through repeated simulations.
- [Checkpoint and
  resume](https://gillescolling.com/tulpa/articles/checkpoint.html):
  save and continue sampling runs.
- [Function reference](https://gillescolling.com/tulpa/reference/):
  arguments, return values, and examples for the fitting and diagnostic
  functions.

## Getting help

Use [GitHub issues](https://github.com/gcol33/tulpa/issues) for
questions and bug reports. Include a small reproducible example and the
output of `sessionInfo()` when reporting a problem.

## Support

> “Software is like sex: it’s better when it’s free.” — Linus Torvalds

I’m a PhD student who builds R packages in my free time because I
believe good tools should be free and open. I started these projects for
my own work and figured others might find them useful too.

If this package saved you some time, buying me a coffee is a nice way to
say thanks. It helps with my coffee addiction.

[![Buy Me A
Coffee](https://img.shields.io/badge/-Buy%20me%20a%20coffee-FFDD00?logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/gcol33)

## License

MIT. See [LICENSE](LICENSE).

## Citation

To cite the tulpa package in publications, use:

``` text
Colling G (2026). _tulpa: Templated Unified Library for Posterior
Approximation in Bayesian Hierarchical Models_. R package version
0.7.5, <https://github.com/gcol33/tulpa>.
```

BibTeX:

``` bibtex
@Manual{tulpa,
  title = {{tulpa}: Templated Unified Library for Posterior Approximation in Bayesian Hierarchical Models},
  author = {Gilles Colling},
  year = {2026},
  note = {R package version 0.7.5},
  url = {https://github.com/gcol33/tulpa},
}
```

Download the citation as
[BibTeX](https://gillescolling.com/tulpa/tulpa.bib) or
[RIS](https://gillescolling.com/tulpa/tulpa.ris).
