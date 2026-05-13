# Nested Laplace Approximation for tulpa

## The problem

Occupancy models at European scale (10×10km grid, ~600K cells, many species) are infeasible with MCMC. spOccupancy with NNGP(8) takes 30 hours for 100×100km (~6K cells). Scaling to 10×10km requires a fundamentally faster inference method.

INLA (Integrated Nested Laplace Approximation) solves this by replacing MCMC with deterministic approximations. But INLA is not on CRAN, ships as binary-only, and is a 500K-line C dependency we don't control. tulpa already has Laplace approximation + NNGP + CUDA — we're 80% of the way to a CRAN-compatible INLA alternative.

## What "nested" means

Standard Laplace (what tulpa has now): approximate the joint posterior p(latent, hyperparams | data) with a single Gaussian at the mode.

Nested Laplace (what INLA does):

1. **Outer layer**: For hyperparameters θ (range, variance, RE precision — typically 2-5 parameters), evaluate on a numerical grid
2. **Inner layer**: At each grid point θ_k, run Laplace on the latent field to get p(latent | data, θ_k) — this is fast because latent Gaussian fields are near-Gaussian
3. **Combine**: Numerically integrate over the grid to get marginal posteriors for each hyperparameter and each latent field element

Why this is better: the inner Laplace is highly accurate (Gaussian approximation to a near-Gaussian target), and the outer integration is exact over a low-dimensional grid. No MCMC sampling noise, no convergence diagnostics, deterministic results.

## What tulpa already has

- Laplace approximation engine (C++) — this IS the inner step
- NNGP with configurable neighbors — O(N) spatial scaling
- HSGP alternative for smooth fields
- Polya-Gamma Gibbs sampler — useful for comparison/validation
- HMC/NUTS — gold standard for validation
- CUDA/GPU backend — parallelizable inner evaluations
- Multi-threading support
- Sparse matrix operations via Eigen

## What needs to be added

### 1. Hyperparameter grid construction

INLA uses a central composite design (CCD) centered at the joint mode of the hyperparameters. For k hyperparameters this gives ~2k + 2^k + 1 grid points. For k=3 that's 15 evaluations, for k=5 it's 43.

Alternative: start simple with a regular grid around the mode, refine later.

```
Inputs: log-posterior evaluated at mode, Hessian at mode
Output: set of grid points {θ_1, ..., θ_G} with integration weights
```

### 2. Inner Laplace at each grid point

tulpa's existing Laplace already does this. The interface would be:

```
For each θ_k in grid:
  Fix hyperparameters to θ_k
  Run Laplace on latent field → get mode x*(θ_k), Hessian H(θ_k)
  Store: log p(data | x*, θ_k) + log p(x* | θ_k) + log |H|^{-1/2}
```

The key optimization: warm-start each inner Laplace from the previous grid point's solution. The mode shifts smoothly as θ changes, so convergence is fast after the first evaluation.

### 3. Numerical integration

Combine grid evaluations:

```
p(θ | data) ∝ p(data | x*(θ), θ) × p(x*(θ) | θ) × p(θ) × |H(θ)|^{-1/2}
```

Integrate over the grid using the CCD weights (or simple trapezoidal rule) to get:
- Marginal posteriors for each hyperparameter
- Marginal posteriors for each latent field element (via weighted average of conditional posteriors)

### 4. Marginal correction (optional, INLA's "simplified Laplace")

For each latent field element x_i, INLA applies a skewness correction to the Gaussian approximation. This matters for non-Gaussian likelihoods (binomial, Poisson). Can be added later — the base nested Laplace without correction is already a big improvement over single Laplace.

## Implementation plan

**Phase 1: Proof of concept (R-level loop)**
- Use tulpa's existing `laplace_fit()` as the inner engine
- Write the outer grid + integration in R
- Benchmark against INLA on a simple spatial occupancy model
- Validate against HMC results

**Phase 2: C++ integration**
- Move the outer loop into C++ for speed
- Warm-starting between grid points
- Parallel inner evaluations (each grid point is independent → trivially parallel)
- GPU acceleration for inner Laplace

**Phase 3: Smart grid**
- Central composite design
- Adaptive grid refinement (add points where the integrand is large)
- Skewness correction for marginals

## Expected performance

For occupancy models with NNGP(8-15) on the inner Laplace:

| Grid size (cells) | MCMC (spOcc) | Single Laplace (tulpa now) | Nested Laplace (proposed) |
|---|---|---|---|
| 6,000 (100×100km) | 30 hours | ~minutes | ~minutes (more accurate) |
| 60,000 (30×30km) | infeasible | ~tens of minutes | ~tens of minutes |
| 600,000 (10×10km) | infeasible | ~hours | ~hours (proper uncertainty) |

The speed difference between single and nested Laplace is small (G inner evaluations instead of 1, with G=15-43 for typical models). The accuracy difference is significant — proper hyperparameter uncertainty instead of point estimates.

## Why this matters for the package ecosystem

- tulpa becomes a CRAN-compatible INLA alternative
- tulpaObs with the `occu()` API replaces both spOccupancy (MCMC, slow) and INLAocc (INLA dependency, non-CRAN)
- Full stack under our control: inference engine, spatial approximations, occupancy API
- GPU acceleration gives a further edge over both INLA and spOccupancy

## Questions to discuss

1. How many hyperparameters do typical occupancy models have? (This determines grid size)
2. Does tulpa's current Laplace accept fixed hyperparameters, or does it always optimize them jointly? (Need the former for the inner step)
3. Should the nested Laplace be a new `method = "nested_laplace"` option alongside `"laplace"` and `"hmc"`, or a separate engine?
4. Priority: implement for spatial GP models first (where the accuracy gain is largest), then extend to all model types?
