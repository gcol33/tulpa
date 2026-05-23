# tulpa

[![Lifecycle: experimental](https://lifecycle.r-lib.org/articles/figures/lifecycle-experimental.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Template Unified Latent Process Architecture for Bayesian Hierarchical Models**

`tulpa` is a Bayesian hierarchical modelling engine. It decomposes the
posterior so cheap deterministic approximations (Laplace, EP, VI,
Pathfinder) handle the well-behaved blocks and exact MCMC corrects only
the residual directions where the approximation is biased. Spatial
fields, temporal effects, spatially varying coefficients, and random
effects are built in. Model packages supply an observation likelihood
through a templated C++ interface and inherit the rest.

## Positioning

|        | Nested approximation                       | Exact-MCMC correction                         |
|--------|--------------------------------------------|-----------------------------------------------|
| INLA   | yes                                        | no, so biased on non-Gaussian residuals       |
| Stan   | no, so pays the full price on every block  | yes                                           |
| tulpa  | yes                                        | yes                                           |

Canonical compositions:

- **IMH-Laplace**: Laplace body, MH bias correction.
- **Pathfinder**: L-BFGS mode, Gaussian fit, ELBO scoring.
- **Nested Laplace + CCD**: inner Laplace, outer hyperparameter integration.

New backends are framed by which layer they add or replace
(approximation, debias, or outer integration), not as standalone
alternatives.

## Latent structure and inference

| Layer            | Choices                                       |
|------------------|-----------------------------------------------|
| Spatial          | HSGP, NNGP, ICAR, BYM2, SPDE                  |
| Temporal         | RW1, RW2, AR1, GP                             |
| Other            | Random effects, SVC, TVC, latent factors      |
| Inference tiers  | HMC/NUTS, EM+Laplace, variational             |
| Correction       | Importance sampling, MI/Gibbs debiasing       |

Tier 1 is exact MCMC; Tier 2 is Laplace with optional Tier-1 debiasing;
Tier 3 is variational. Auto mode picks between Tier 1 and Tier 2 and
prints its choice. Tier 3 is opt-in only.

## Installation

```r
install.packages("pak")
pak::pak("gcol33/tulpa@v0.0.1")
```

pak resolves the dependency tree, including `tulpaMesh` (declared in
`Remotes:`). For the development version, drop the tag:

```r
pak::pak("gcol33/tulpa")
```

Requires a C++17 toolchain: Rtools on Windows, Xcode CLI tools on macOS,
`r-base-dev` on Linux. `tulpa` compiles its C++ backend on first install.

## Model packages

`tulpa` ships the engine. Observation likelihoods live in companion
packages that link against it:

- [tulpaObs](https://github.com/gcol33/tulpaObs) covers single-season,
  dynamic, community, integrated, and JSDM occupancy; abundance;
  distance sampling; removal; false-positive occupancy; and hurdle-Beta
  cover.
- `tulpaRatio` (in progress) will cover ratio models (numerator /
  denominator likelihoods on shared spatial fields).

A model package supplies a templated `LikelihoodSpec` and a small amount
of data encoding. Spatial, temporal, prior, and inference machinery
comes from `tulpa`.

## Custom latent blocks

User-supplied templated C++ snippets compile against `tulpa`'s autodiff
types (`A`, `A_r`) and register as new latent structures. Because the
snippet uses the same AD types as the built-in blocks, custom blocks
work under every tier including NUTS, with full gradient flow through
the user code.

Comparison:

- INLA `rgeneric`: R callback, no AD, Laplace tier only.
- INLA `cgeneric`: C function, no AD, faster but no exact-MCMC support.
- Stan: full DSL with parser and codegen for the entire model.
- TMB: templated C++ snippet with CppAD; closest analog.

## Status

Experimental. The C++ ABI version (`TULPA_ABI_VERSION` in
`inst/include/tulpa/model_data.h`) is checked at runtime, so a
mismatch between `tulpa` and a model package fails with a readable
error instead of a segfault. Expect breaking changes.
