# tulpa

[![Lifecycle: experimental](https://lifecycle.r-lib.org/articles/figures/lifecycle-experimental.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Template Unified Latent Process Architecture for Bayesian Hierarchical Models**

`tulpa` is an engine for Bayesian hierarchical models with spatial fields,
temporal effects, spatially varying coefficients, and random effects.
Model packages plug their observation likelihood into the engine through a
templated C++ interface and inherit spatial structure, temporal structure,
priors, autodiff, and inference for free.

## Design

Decompose the posterior so well-behaved blocks get cheap deterministic
approximations (Laplace, EP, VI, Pathfinder) and exact MCMC corrects only
the residual directions where the approximation is biased.

| Layer            | Choices                                       |
|------------------|-----------------------------------------------|
| Spatial          | HSGP, NNGP, ICAR, BYM2, SPDE                  |
| Temporal         | RW1, RW2, AR1, GP                             |
| Other            | Random effects, SVC, TVC, latent factors      |
| Inference tiers  | HMC/NUTS, EM+Laplace, variational             |
| Correction       | Importance sampling, MI/Gibbs debiasing       |

Tier selection encodes correctness: Tier 1 is exact MCMC, Tier 2 is
Laplace with optional Tier-1 debiasing, Tier 3 is variational. Auto mode
never silently chooses Tier 3.

## Installation

```r
install.packages("pak")
pak::pak("gcol33/tulpaMesh")
pak::pak("gcol33/tulpa@v0.0.1")
```

For the development version, drop the tag:

```r
pak::pak("gcol33/tulpa")
```

Requires a C++17 toolchain (Rtools on Windows, Xcode CLI tools on macOS,
`r-base-dev` on Linux). `tulpa` compiles its C++ backend on first install.

## Model packages

`tulpa` ships the engine. Observation likelihoods live in companion
packages that depend on `tulpa`:

- [tulpaObs](https://github.com/gcol33/tulpaObs) — single-season, dynamic,
  community, integrated, and JSDM occupancy; abundance; distance sampling;
  removal; false-positive occupancy; cover (hurdle Beta).

A model package supplies a templated `LikelihoodSpec` and a small amount
of data encoding. Spatial, temporal, prior, and inference machinery is
inherited from `tulpa`.

## Custom latent blocks

User-supplied templated C++ snippets compile against `tulpa`'s autodiff
types and register as new latent structures. Custom blocks work under
every tier, including NUTS, because they compose with the engine's AD —
no R callback, no broken gradient chain.

## Status

Experimental. The C++ ABI (`TULPA_ABI_VERSION` in `inst/include/tulpa/model_data.h`)
is checked at runtime; mismatches between `tulpa` and a model package
fail loudly instead of segfaulting. Expect breaking changes.
