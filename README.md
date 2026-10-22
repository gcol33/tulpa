
# tulpa <img src="man/figures/logo.svg" align="right" height="139" alt="tulpa logo" />

*Bayesian hierarchical models for grouped, spatial and temporal data*

<!-- badges: start -->

[![CRAN
status](https://www.r-pkg.org/badges/version/tulpa)](https://CRAN.R-project.org/package=tulpa)
[![R-CMD-check](https://github.com/gcol33/tulpa/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/gcol33/tulpa/actions/workflows/R-CMD-check.yaml)
[![Codecov test
coverage](https://codecov.io/gh/gcol33/tulpa/graph/badge.svg)](https://app.codecov.io/gh/gcol33/tulpa)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
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

- **Grouped effects.** Random intercepts and slopes, including
  uncertainty in their variances and correlations.
- **Smooth, spatial and temporal structure.** Smooth covariate effects,
  spatial fields, temporal processes, and combinations of these
  structures.
- **Several inference methods.** Deterministic approximations, MCMC
  sampling, and correction methods are selected through the same fitting
  function.
- **Custom models.** Model-specific observation likelihoods and custom
  latent processes are supplied through R or C++ interfaces.

[Calibration](https://gillescolling.com/tulpa/articles/sbc.html) shows
how to assess posterior calibration through repeated simulations.
[Reliability
diagnostics](https://gillescolling.com/tulpa/articles/reliability-pareto-k.html)
explains what can be checked on an individual fit. The package is
experimental and its interface may change.

## Installation

Install the released version from CRAN:

``` r
install.packages("tulpa")
```

CRAN provides binaries for Windows and macOS, which need no build tools.

The development version compiles C++17 code. Install the build tools for
your operating system first:

| Operating system | Build tools |
|----|----|
| Windows | [Rtools](https://cran.r-project.org/bin/windows/Rtools/) matching your R version |
| macOS | Run `xcode-select --install` in Terminal |
| Debian / Ubuntu | Install `r-base-dev` |

Check that R can find the compiler:

``` r
pkgbuild::has_build_tools(debug = TRUE)
```

Then install tulpa from GitHub:

``` r
# install.packages("pak")
pak::pak("gcol33/tulpa")
```

The build links against the BLAS and LAPACK libraries that R itself uses
for matrix operations, so no other system library is needed. Package
dependencies, including
[tulpaMesh](https://CRAN.R-project.org/package=tulpaMesh) for spatial
mesh construction, are installed with tulpa.

### Optional: multithreading

OpenMP lets the nested-Laplace fitters use several CPU threads through
`n_threads` in their `control` list. The
[single-process](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.html)
and
[joint-model](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.html)
reference pages describe the available controls.

Rtools on Windows and the default compiler on Linux include OpenMP.
Without it, the package installs and runs on one thread.

#### macOS setup

Apple’s compiler needs an additional OpenMP runtime. Install it in
Terminal:

``` sh
brew install libomp
```

Before installing tulpa from source, add the settings for your Mac to
`~/.R/Makevars`, R’s personal compiler-settings file. Create the
directory and file if needed.

**Apple Silicon:**

``` makefile
CPPFLAGS += -I/opt/homebrew/opt/libomp/include -Xclang -fopenmp
LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
```

**Intel:**

``` makefile
CPPFLAGS += -I/usr/local/opt/libomp/include -Xclang -fopenmp
LDFLAGS += -L/usr/local/opt/libomp/lib -lomp
```

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
to the response. [Data
formatting](https://gillescolling.com/tulpa/articles/data-formatting.html)
covers response encodings, grouping variables, and spatial inputs.

## Random effects, smoothers, and fields

Formula terms describe the model structure. `(1 | group)` adds random
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
[Inference
modes](https://gillescolling.com/tulpa/articles/inference-modes.html)
describes the methods and the models each supports.

Diagnostics depend on how the posterior was computed. MCMC fits report
chain convergence and effective sample sizes. Approximation fits report
the available integration and inner-approximation checks. [Reliability
diagnostics](https://gillescolling.com/tulpa/articles/reliability-pareto-k.html)
explains how to interpret these results and their limits.

## Custom models and processes

Companion packages supply observation likelihoods and reuse tulpa’s
inference and latent structures.
[tulpaObs](https://github.com/gcol33/tulpaObs) provides models for
occupancy, abundance, and detection;
[tulpaRatio](https://github.com/gcol33/tulpaRatio) provides models for
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
[quickstart](https://gillescolling.com/tulpa/articles/quickstart.html):
a complete path from a formula to a fitted model.

- [Data
  formatting](https://gillescolling.com/tulpa/articles/data-formatting.html):
  response encodings, grouping variables and spatial inputs.
- [Random slopes and
  covariances](https://gillescolling.com/tulpa/articles/random-slopes.html):
  group-level variances and correlations.
- [Spatial
  models](https://gillescolling.com/tulpa/articles/spatial-models.html)
  and [temporal
  models](https://gillescolling.com/tulpa/articles/temporal-models.html):
  supplying dependence between locations or times.
- [Inference
  modes](https://gillescolling.com/tulpa/articles/inference-modes.html):
  the inference methods and the models each supports.
- [Priors](https://gillescolling.com/tulpa/articles/priors.html): prior
  distributions for coefficients and latent structures.
- [Reliability
  diagnostics](https://gillescolling.com/tulpa/articles/reliability-pareto-k.html):
  what can be checked on an individual fit.
- [Calibration](https://gillescolling.com/tulpa/articles/sbc.html):
  posterior calibration through repeated simulations.
- [Model
  comparison](https://gillescolling.com/tulpa/articles/model-comparison.html):
  comparing fitted models and combining predictions.
- [Checkpoint and
  resume](https://gillescolling.com/tulpa/articles/checkpoint.html):
  saving and continuing sampling runs.
- [Custom latent
  processes](https://gillescolling.com/tulpa/articles/tgmrf.html) and
  [EM +
  Laplace](https://gillescolling.com/tulpa/articles/em-laplace.html):
  extending the engine.
- [Full reference](https://gillescolling.com/tulpa/reference/): every
  function, grouped by job.

## Getting help

Questions and bug reports go to the [GitHub
issues](https://github.com/gcol33/tulpa/issues). A bug report is easiest
to act on with a small reproducible example, for instance made with
[reprex](https://reprex.tidyverse.org/), and the output of
`sessionInfo()`.

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

tulpa is licensed under MIT. Three bundled components carry their own
notices, listed in `inst/COPYRIGHTS`:

- Sobol direction numbers by Frances Y. Kuo and Stephen Joe, under a
  3-clause BSD licence.
- Double-precision exp and log routines ported from Arm Optimized
  Routines, copyright Arm Limited, under MIT.
- The WALNUTS transition, ported from walnutpie, under MIT.

## Citation

To cite the tulpa package in publications, use:

``` text
Colling G (2026). _tulpa: Templated Unified Library for Posterior
Approximation in Bayesian Hierarchical Models_.
doi:10.32614/CRAN.package.tulpa
<https://doi.org/10.32614/CRAN.package.tulpa>. R package version 0.7.5,
<https://CRAN.R-project.org/package=tulpa>.
```

BibTeX:

``` bibtex
@Manual{colling2026tulpa,
  title = {{tulpa}: Templated Unified Library for Posterior Approximation in Bayesian Hierarchical Models},
  author = {Gilles Colling},
  year = {2026},
  note = {R package version 0.7.5},
  doi = {10.32614/CRAN.package.tulpa},
  url = {https://CRAN.R-project.org/package=tulpa},
}
```

Download the citation as
[BibTeX](https://gillescolling.com/tulpa/tulpa.bib) or
[RIS](https://gillescolling.com/tulpa/tulpa.ris).
