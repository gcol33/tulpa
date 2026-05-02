# tulpa Features Plan — Nested Laplace & Solver Infrastructure

**Date:** 2026-03-25 (drafted), updated 2026-04-29.
**Status:** Phase 0 + Phase 1 + most of Phase 2 shipped. Remaining work is
selected inversion polish, k≥3 implicit-diff for hyperparameter NUTS
(`Feature 7`), and large-N stochastic log-det (`Feature 8`). This document
remains the engineering reference; current punch-list lives in `TODO.md`.

## Status as of 2026-04-29

| Feature | Status | Where it lives |
|---|---|---|
| F1 CHOLMOD sparse solver | ✅ shipped | `src/sparse_cholesky.{h,cpp}` |
| F2 Q-builder abstraction | ✅ shipped (SPDE) | `src/spde_qbuilder.h` |
| F3 Nested Laplace (a/b/c) | ✅ shipped | generic driver `src/nested_laplace_grid.h`; eight prior backends in `src/nested_laplace.cpp` (icar, bym2, car_proper, rw1, rw2, ar1, nngp, hsgp) + SPDE in `src/spde_laplace.cpp`; CCD grid in `R/ccd_grid.R` |
| F4 Selected inversion | ✅ shipped | `SparseCholeskySolver::selected_inversion_diagonal` (Takahashi via simplicial LL') |
| F5 Amortized Laplace warm-start | ✅ shipped | `tulpa::run_nested_laplace_grid` warm-starts each grid point from the previous mode; `SparseCholeskySolver` keeps the symbolic factor across calls |
| F6 GPU-batched NNGP | ✅ CPU + cuSOLVER path; live CUDA test pending (TODO.md #7) | `src/gpu_nngp_laplace.h`, `src/gpu_cuda.h` |
| F7 Implicit differentiation | ✅ shipped (k=1, k=2) | `src/implicit_diff.{h,cpp}`; full k≥3 NUTS-over-hyperparams still scaffolded |
| F8 Stochastic log-det | ✅ shipped | `src/stochastic_logdet.cpp` (Lanczos) + cross-DLL shim `tulpa_stochastic_log_det` |
| F9 Consolidate Laplace code | ✅ shipped | `laplace_newton.h::laplace_newton_solve` template; eight model-specific entries in `laplace_core.cpp` all dispatch through it |

Cross-DLL ABI surface added on top of all of the above (see `fix.md` for the
shim plan): `laplace_api.h`, `pg_api.h`, `vi_api.h`, `ess_api.h`,
`sparse_solver_api.h`, `priors_capped.h` are in working tree and compile;
`nested_laplace_api.h` and `spde_api.h` are next.

---

## Motivation

tulpa's Tier 2 (Structured) inference is currently a single-point Laplace: find the joint
mode of (latent, hyperparameters), approximate with a Gaussian at the mode. This gives fast
point estimates but underestimates hyperparameter uncertainty — the range/variance parameters
are pinned to their modal values with no integration over alternatives.

INLA solves this with nested Laplace approximation: grid over hyperparameters (outer), Laplace
on the latent field at each grid point (inner), numerical integration to get proper marginals.
INLA is not on CRAN, ships as binary-only, and depends on PARDISO (proprietary sparse solver).

tulpa can replicate the statistical method using CHOLMOD (the sparse solver R already ships
via the Matrix package) + GPU-batched NNGP + amortized warm-starting. The result: a
CRAN-compatible INLA alternative with GPU acceleration, under our full control.

### Current Laplace bottleneck

`laplace_core.cpp` uses **dense Cholesky** (hand-rolled O(N³), `laplace_cholesky_solve` at
line 240). The log-determinant computation (`laplace_cholesky_log_det`) is also dense and
duplicated 7+ times across the file. This limits Tier 2 to ~5K latent field elements before
it becomes slower than MCMC.

---

## Feature 1: CHOLMOD Sparse Solver Integration

**Priority:** P0 — everything else builds on this
**Complexity:** Medium
**Files:** new `src/sparse_cholesky.h`, `src/sparse_cholesky.cpp`, modify DESCRIPTION

### What

Replace dense Cholesky in `laplace_core.cpp` with CHOLMOD supernodal sparse Cholesky from
SuiteSparse, accessed via R's `Matrix` package C API.

### Why

- CHOLMOD is supernodal (converts sparse problem to many small dense BLAS ops → 5-20x faster
  than column-by-column Eigen `SimplicialLDLT` on irregular sparsity patterns)
- Already compiled and linked in every R installation via `Matrix`
- Supports symbolic-once / numeric-many factorization (critical for nested Laplace: same
  sparsity pattern across all hyperparameter grid points, only values change)
- Log-determinant falls out for free: `log det = 2 * sum(log(diag(L)))`
- MIT-licensed since SuiteSparse 7.0

### How

1. Add `Matrix` to `LinkingTo` in DESCRIPTION (already in `Imports`)
2. Create `src/sparse_cholesky.h` — thin C++ wrapper around CHOLMOD:
   ```cpp
   namespace tulpa {
   class SparseCholeskySolver {
   public:
       // Phase 1: symbolic analysis (done once per sparsity pattern)
       void analyze(const cholmod_sparse* A);

       // Phase 2: numeric factorization (repeated per grid point)
       bool factorize(const cholmod_sparse* A);

       // Solve Ax = b using the factorization
       void solve(const double* b, double* x, int n);

       // Log-determinant from Cholesky diagonal
       double log_determinant() const;

       // Selected elements of A^{-1} (Takahashi equations) — Feature 4
       void selected_inversion(std::vector<double>& diag_inv);

   private:
       cholmod_common common_;
       cholmod_factor* factor_ = nullptr;
       bool analyzed_ = false;
   };
   }
   ```
3. Convert `laplace_core.cpp` precision matrices from `std::vector<std::vector<double>>` to
   `cholmod_sparse` (CSC format — same format R's `Matrix` uses internally)
4. Replace all 7 copies of `laplace_cholesky_solve` + `laplace_cholesky_log_det` with calls
   to `SparseCholeskySolver`
5. Dense fallback for tiny problems (n < 200): keep current code path, branch at runtime

### Validation

- Numerical: CHOLMOD results must match dense Cholesky to < 1e-10 relative error
- Performance: benchmark at N = 1K, 5K, 20K, 100K on ICAR grid and NNGP precision
- All existing tests pass (identical log-marginals within tolerance)

### CRAN compatibility

`Matrix` is a recommended package (ships with R). `LinkingTo: Matrix` is established practice
(used by lme4, TMB, glmmTMB). No new system dependencies.

---

## Feature 2: Q-Builder Abstraction

**Priority:** P0 — required for nested Laplace
**Complexity:** Low-Medium
**Files:** new `inst/include/tulpa/precision_builder.h`, modify spatial builders

### What

A function interface that builds the sparse precision matrix Q(θ) given hyperparameter values θ.
Each spatial model type registers its own Q-builder. The nested Laplace engine calls this at
each grid point without knowing what spatial model generated Q.

### Why

The nested Laplace inner step needs to rebuild Q at different hyperparameter values. Currently
Q is built once in R and passed to C++. We need a C++ function that can rebuild Q quickly
given new θ, while preserving the sparsity pattern (so CHOLMOD's symbolic analysis is reused).

### Interface

```cpp
namespace tulpa {

// Q-builder: given hyperparameters, fill numeric values of sparse precision matrix
// The sparsity pattern (row_ptr, col_idx) is fixed at construction time
// Only the values array changes across calls
struct PrecisionBuilder {
    // Spatial type (determines which builder function to use)
    SpatialType type;

    // Fixed structure (set once, never changes)
    int n;                              // dimension
    std::vector<int> col_ptr;           // CSC column pointers
    std::vector<int> row_idx;           // CSC row indices

    // Rebuild values given hyperparameters
    // theta layout depends on type:
    //   ICAR: [log_tau]
    //   NNGP: [log_range, log_variance]
    //   SPDE: [log_range, log_variance]
    //   BYM2: [log_tau, logit_rho]
    //   HSGP: [log_range, log_variance]
    void build(const double* theta, double* values) const;

    // Number of hyperparameters for this spatial type
    int n_theta() const;

    // Hyperparameter names (for output labeling)
    std::vector<std::string> theta_names() const;
};

}
```

### Implementations

| Type | Q(θ) | Notes |
|------|-------|-------|
| ICAR | Q = τ · (D - W) | Values scale linearly with τ. Cheapest rebuild. |
| BYM2 | Q = τ · [ρ · Q_icar + (1-ρ) · I] | Two-parameter blend. |
| NNGP | Q = L'L where L is lower-triangular from neighbor conditioning | Rebuild L from covariance function C(d; ρ, σ²) applied to neighbor distances. |
| SPDE | Q = κ⁴C + 2κ²G + G C⁻¹ G | κ = √8ν / ρ. Rebuild from FEM matrices C, G (fixed). |
| HSGP | Diagonal: diag(S(λ_j; ρ, σ²)⁻¹) | Spectral density evaluated at eigenvalues. Trivial rebuild. |

### Key property

All types preserve the sparsity pattern across θ values. Only the numeric values change.
This means CHOLMOD `analyze()` runs once, `factorize()` runs per grid point.

---

## Feature 3: Nested Laplace Approximation

**Priority:** P1 — the main feature
**Complexity:** High
**Depends on:** Features 1, 2
**Files:** new `src/nested_laplace.h`, `src/nested_laplace.cpp`, new `R/nested_laplace.R`

### What

Three-layer nested Laplace approximation following Rue, Martino & Chopin (2009):

1. **Outer**: Evaluate the Laplace-approximated marginal posterior p̃(θ | y) on a grid/CCD
2. **Inner**: At each grid point θ_k, run Newton-Raphson on the latent field x to find
   mode x*(θ_k) and Hessian H(θ_k) — using CHOLMOD (Feature 1) and Q-builder (Feature 2)
3. **Integrate**: Numerically integrate over the grid to get marginal posteriors

### Phase 3a: R-level proof of concept

Implement the outer loop in R, calling tulpa's existing `laplace_mode()` as the inner engine.

```r
nested_laplace <- function(formula, data, family, spatial, ...,
                           grid_type = c("ccd", "regular"),
                           n_grid = NULL) {
  # 1. Find mode of hyperparameters (existing optimize_theta)
  theta_hat <- optimize_hyperparameters(...)
  H_theta <- hessian_at_mode(theta_hat)

  # 2. Build grid around mode
  grid <- build_hyperparameter_grid(theta_hat, H_theta, type = grid_type)

  # 3. Inner Laplace at each grid point
  inner_results <- lapply(grid$points, function(theta_k) {
    Q_k <- build_precision(theta_k, spatial)  # Q-builder
    laplace_mode(y, n, X, Q_k, family, x_init = prev_mode)  # warm-start
  })

  # 4. Numerical integration
  marginals <- integrate_nested(grid, inner_results)

  # Return: marginal posteriors for hyperparameters + latent field
  structure(marginals, class = "tulpa_nested_laplace")
}
```

**Deliverable:** Validated against HMC on a spatial binomial model (500 locations, ICAR).
Must agree on hyperparameter marginals within reasonable tolerance.

### Phase 3b: C++ inner loop

Move the inner loop (steps 2-4) into C++. Key optimizations:

1. **Warm-starting**: Initialize Newton-Raphson at each grid point from the previous point's
   mode. Modes shift smoothly with θ → typically 2-3 Newton steps instead of 10-15.

2. **CHOLMOD symbolic reuse**: `analyze()` once, `factorize()` at each grid point.

3. **Parallel inner evaluations**: Grid points are independent → OpenMP parallel loop over
   grid points. Each thread gets its own CHOLMOD workspace.

4. **GPU path**: For NNGP, the n × (m×m) neighbor covariance computations at each grid
   point can be batched on GPU (Feature 6).

### Phase 3c: Smart grid

1. **Central Composite Design (CCD)**: For k hyperparameters, gives 2k + 2^k + 1 points.
   Standard in INLA. Better coverage than regular grid for k > 3.

2. **Adaptive refinement**: After initial CCD evaluation, add points where the integrand
   is large (high posterior density regions that the CCD missed).

3. **Skewness correction** (simplified Laplace): For each latent field element x_i, compute
   a third-derivative correction to the Gaussian marginal. Matters for binomial/Poisson
   likelihoods where the posterior is skewed. INLA's "simplified Laplace" strategy.

### Integration with tier system

- New backend: `"nested_laplace"` in Tier 2 (Structured)
- Same epistemic guarantee as single Laplace: correct if latent Gaussian assumptions hold
- Better uncertainty quantification (proper hyperparameter marginalization)
- `INFERENCE_TIERS$structured$backends` becomes `c("laplace", "nested_laplace")`
- Auto mode prefers `nested_laplace` over `laplace` when hyperparameter count > 1

### Hyperparameter count by model type

| Model | Typical θ | k | CCD points |
|-------|-----------|---|------------|
| ICAR + intercept | τ_spatial | 1 | 3 |
| BYM2 | τ, ρ | 2 | 9 |
| NNGP (Matérn) | range, variance | 2 | 9 |
| NNGP + temporal RW1 | range, σ²_spatial, τ_temporal | 3 | 15 |
| NNGP + RE | range, σ²_spatial, σ²_RE | 3 | 15 |
| Multi-species occupancy | range, σ²_spatial, σ²_det, σ²_occ | 4 | 25 |
| Full spatiotemporal | range, σ²_sp, τ_temp, ρ_ar1, σ²_RE | 5 | 43 |

For k ≤ 5, CCD is efficient. For k > 5 (multi-species with many variance components),
need either adaptive strategies or NUTS over hyperparameters (Feature 7).

---

## Feature 4: Selected Inversion (Takahashi Equations)

**Priority:** P1 — needed for marginal variances
**Complexity:** Medium
**Depends on:** Feature 1
**Files:** extend `src/sparse_cholesky.h/.cpp`

### What

Compute selected elements of Q⁻¹ (specifically the diagonal: marginal variances of each
latent field element) without computing the full inverse. Uses the elimination tree from
CHOLMOD's Cholesky factorization.

### Why

Users want `posterior_var(x_i)` for each spatial location i. Naive approach: invert Q
(O(N³) dense, impossible at N = 600K). Selected inversion computes only the diagonal
elements of Q⁻¹ in O(nnz(L)) — linear in the number of nonzeros in the Cholesky factor.

### Algorithm (Takahashi, Fagan & Chin 1973)

Given Q = LDL' (from CHOLMOD), compute Z = Q⁻¹ element-by-element, but only for positions
(i,j) where L_{ij} ≠ 0. Process columns of L in reverse elimination order:

```
For j = n down to 1:
    Z[j,j] = 1/D[j] - sum_{k>j, L[k,j]≠0} L[k,j] * Z[k,j]
    For i > j where L[i,j] ≠ 0:
        Z[i,j] = -sum_{k≥i, L[k,j]≠0} L[k,j] * Z[k,i]
```

The diagonal of Z gives the marginal variances. Off-diagonal elements give marginal
covariances for neighbors (useful for credible intervals on spatial contrasts).

### Implementation

~100-150 lines of C++ given CHOLMOD's factor struct (which provides L in CSC format and
the permutation). Add as `SparseCholeskySolver::selected_inversion()`.

### Validation

- Compare diagonal of selected inversion against `solve(Q)` diagonal for small (N < 500) problems
- Must match to machine precision

---

## Feature 5: Amortized Laplace (Warm-Start Inner Solve)

**Priority:** P1 — biggest speed win for nested Laplace
**Complexity:** Low
**Depends on:** Feature 3
**Files:** modify `src/nested_laplace.cpp`
**Reference:** Margossian, Vehtari et al. (2023-2024)

### What

When iterating over hyperparameter grid points, initialize each inner Newton-Raphson from
the previous grid point's mode. Since the mode x*(θ) varies smoothly with θ, successive
solves converge in 2-3 iterations instead of 10-15.

### Why

5-10x speedup on the inner loop, which is the dominant cost of nested Laplace.

### How

1. Order grid points to minimize distance between consecutive θ values (Hamiltonian path
   through the grid, or simply sort by first principal component of the CCD)
2. Pass `x_init = x_mode_previous` to `laplace_mode()` at each grid point
3. Reduce `max_iter` from 50 to 10 for grid points after the first (with convergence check)
4. If convergence fails in 10 iterations, fall back to full solve from scratch

### Validation

- Log-marginal at each grid point must match full solve to < 1e-8
- Timing comparison: amortized vs. cold-start on NNGP model with 5K locations, k=3 hyperparams

---

## Feature 6: GPU-Batched NNGP Neighbor Computations

**Priority:** P2 — scaling to 600K+
**Complexity:** Medium-High
**Depends on:** Feature 1 (CHOLMOD), existing CUDA backend stub
**Files:** new `src/gpu_nngp.cu`, extend `src/gpu_backend.h`

### What

For NNGP with m neighbors and N locations, the Q-builder needs N independent m×m dense
Cholesky factorizations (one per location's neighbor covariance matrix). This is
embarrassingly parallel → batch on GPU via cuBLAS.

### Why

At N = 600K, m = 15: that's 600K × 15 × 15 dense covariance matrices to build and factor.
On CPU (sequential): ~minutes. On GPU (batched cuBLAS `cublasDgetrfBatched`): ~seconds.

This is the single highest-impact GPU optimization for spatial scaling.

### How

1. Compute all N distance matrices on GPU (neighbor distances already stored in `GPData`)
2. Apply covariance kernel elementwise (Matérn/Exponential) → N dense m×m matrices
3. Batch Cholesky via `cusolverDnDpotrfBatched` or `cublasDgetrfBatched`
4. Extract: L_i for each location → assemble into NNGP sparse precision Q
5. Transfer Q values back to CPU for CHOLMOD factorization of the full system

### Memory

- N × m × m × 8 bytes = 600K × 15 × 15 × 8 = ~1 GB. Fits in 16 GB VRAM with room.
- For m = 30: ~4 GB. Still fits.

### Fallback

CPU path remains default. GPU path activated by `solver = "gpu"` or detected automatically
when CUDA is available and N > threshold.

---

## Feature 7: Implicit Differentiation for Hyperparameter Gradients

**Priority:** P2 — enables NUTS over hyperparameters (future)
**Complexity:** High
**Depends on:** Features 1, 2, 3
**Files:** new `src/laplace_gradient.h`

### What

Compute ∂/∂θ of the Laplace-marginalized log-posterior log p̃(y | θ). This gradient enables
NUTS sampling over hyperparameters instead of grid integration — necessary when k > 5.

### Why

CCD with k = 8 hyperparameters needs 273 grid points. With k = 12 (multi-species occupancy
with many variance components), it's 4121 points — infeasible. NUTS over θ with the
Laplace-marginalized likelihood as target scales gracefully in k.

This is exactly the tmbstan pattern (TMB's Laplace + Stan's NUTS), but native in tulpa.

### Math

The Laplace-marginalized log-posterior is:

```
log p̃(y | θ) = log p(y | x*(θ), θ) + log p(x*(θ) | θ) - 0.5 log|H(θ)| + const
```

where x*(θ) = argmax_x log p(x | y, θ) is the inner Laplace mode.

The gradient via the implicit function theorem:

```
d/dθ log p̃ = ∂/∂θ [log p(y|x,θ) + log p(x|θ)]|_{x=x*}
             + [∂/∂x log p(y|x,θ) + ∂/∂x log p(x|θ)]|_{x=x*} · dx*/dθ
             - 0.5 d/dθ log|H(θ)|
```

The second term vanishes at the mode (gradient of inner objective = 0 at mode). So:

```
d/dθ log p̃ = ∂/∂θ [log p(y|x*,θ) + log p(x*|θ)] - 0.5 tr(H⁻¹ dH/dθ)
```

The trace term requires selected elements of H⁻¹ (Feature 4) and dH/dθ (from the Q-builder).

### Implementation

1. Q-builder (Feature 2) extended with `dQ/dtheta`: how Q values change with each
   hyperparameter. For ICAR: dQ/d(log τ) = Q (trivial). For NNGP: need derivative of
   covariance kernel w.r.t. range and variance.

2. Likelihood derivatives ∂/∂θ are straightforward when θ only enters through Q
   (spatial hyperparameters). More complex if θ includes likelihood parameters (phi for negbin).

3. The trace term `tr(H⁻¹ dH/dθ)` uses selected inversion (Feature 4) for the diagonal
   elements of H⁻¹, combined with the diagonal of dH/dθ.

### Integration with tier system

When this is available, nested Laplace can optionally use NUTS for the outer layer instead
of a grid. This becomes a new backend: `"laplace_nuts"` in Tier 2, automatically selected
when k > 5.

---

## Feature 8: Stochastic Log-Determinant (Very Large Systems)

**Priority:** P3 — only needed for N > 100K without NNGP
**Complexity:** Medium
**Depends on:** Feature 1
**Files:** new `src/stochastic_logdet.h`

### What

Stochastic Lanczos Quadrature (SLQ) estimator for log det(Q) when Q is too large for
exact Cholesky factorization. Only requires matrix-vector products Q·v.

### Why

For SPDE meshes at N > 100K, even CHOLMOD's sparse Cholesky becomes expensive. SLQ gives
an unbiased estimate using ~20-50 matrix-vector products — each O(nnz(Q)).

For NNGP this is unnecessary (log-det is O(Nm) from the triangular factor directly), so
this feature is only relevant for SPDE/ICAR on very large irregular meshes.

### Algorithm

1. Draw k probe vectors z_1, ..., z_k ~ N(0, I) (k = 20-50)
2. For each z_i, run Lanczos iteration (30-50 steps) to get tridiagonal T_i
3. log det ≈ (N/k) Σ_i z_i' log(Q) z_i ≈ (N/k) Σ_i [Lanczos estimate of z_i' f(Q) z_i]
4. The Lanczos estimate uses eigendecomposition of T_i (cheap: T_i is 30×30)

### Implementation

~200 lines of C++. The core is a Lanczos tridiagonalization loop + eigendecomposition of
small tridiagonal matrices (use Eigen's `SelfAdjointEigenSolver` for the small system).

### When to use

Automatic threshold: if N > 100K and spatial type is SPDE or ICAR (not NNGP/HSGP),
switch from exact CHOLMOD log-det to SLQ. User-overridable.

---

## Feature 9: Consolidate Laplace Code (Prerequisite Refactor)

**Priority:** P0 — do before or alongside Feature 1
**Complexity:** Medium
**Files:** `src/laplace_core.cpp`

### What

The current `laplace_core.cpp` has massive code duplication:
- `laplace_cholesky_log_det` pattern appears 7+ times (lines 366, 853, 1293, 1873, 2288, 2901)
- `laplace_cholesky_solve` is a hand-rolled dense Cholesky duplicated across model-specific
  Laplace functions
- Multiple near-identical Laplace mode-finding functions for different model types

### Why

Before adding CHOLMOD and nested Laplace on top, the existing code must be consolidated.
Otherwise we'd be adding new features to duplicated code — exactly the anti-pattern
CLAUDE.md's "Never Copy-Paste Logic" principle prohibits.

### How

1. Extract a single `laplace_inner_solve(Q, W, grad, x_init, opts) → LaplaceResult` that:
   - Builds H = Q + diag(W) from the precision Q and likelihood Hessian W
   - Runs Newton-Raphson using a solver backend (dense now, CHOLMOD after Feature 1)
   - Computes log-det and log-marginal
   - Returns mode, Hessian, log-marginal

2. Each model-specific Laplace function (binomial, negbin, Poisson, spatial, etc.) becomes:
   - Build Q and compute W at each Newton step (model-specific)
   - Call `laplace_inner_solve` (shared)

3. The solver backend becomes swappable: `enum SolverType { DENSE, CHOLMOD, GPU }`

### Validation

All existing Laplace tests must produce identical results (bitwise on log-marginals).

---

## Execution Order

```
Phase 0 — Foundation (do first)
├── Feature 9: Consolidate Laplace code (eliminate duplication)
├── Feature 1: CHOLMOD integration (sparse solver)
└── Feature 2: Q-builder abstraction

Phase 1 — Nested Laplace (main feature)
├── Feature 3a: R-level proof of concept
├── Feature 3b: C++ inner loop + warm-starting (includes Feature 5)
├── Feature 4: Selected inversion (marginal variances)
└── Feature 3c: Smart grid (CCD, adaptive, skewness correction)

Phase 2 — Scaling
├── Feature 6: GPU-batched NNGP
└── Feature 7: Implicit differentiation (NUTS over hyperparameters)

Phase 3 — Edge cases
└── Feature 8: Stochastic log-determinant (N > 100K non-NNGP)
```

### Dependencies

```
Feature 9 (consolidate) ──→ Feature 1 (CHOLMOD) ──→ Feature 3 (nested Laplace)
                                    │                       │
                                    ├──→ Feature 4 (sel inv) ──→ Feature 7 (implicit diff)
                                    │
Feature 2 (Q-builder) ─────────────→ Feature 3
                                    │
                                    └──→ Feature 6 (GPU NNGP)

Feature 8 (stochastic logdet) is independent, only needed for SPDE/ICAR at N > 100K
```

---

## Performance Targets

| Scenario | Current (dense Laplace) | After Phase 1 | After Phase 2 |
|----------|------------------------|---------------|---------------|
| ICAR 1K cells, binomial | ~1s | ~0.2s | ~0.2s |
| ICAR 6K cells, binomial | ~30s | ~2s | ~2s |
| NNGP(15) 6K locations | ~minutes | ~30s (nested, 9 grid pts) | ~10s (GPU) |
| NNGP(15) 60K locations | infeasible | ~5min (nested) | ~1min (GPU) |
| NNGP(15) 600K locations | infeasible | ~1hr (nested) | ~10min (GPU) |
| SPDE mesh 100K | infeasible | ~20min (nested + CHOLMOD) | ~5min (GPU) |

Nested Laplace adds ~G× overhead vs single Laplace (G = grid points, typically 9-43),
but amortized warm-starting reduces this to ~3-5× in practice. The proper hyperparameter
uncertainty is worth the cost.

---

## Competitive Position After Implementation

| | INLA | tulpa (after this plan) | spOccupancy | TMB |
|---|---|---|---|---|
| CRAN | No | Yes | Yes | Yes |
| Solver | PARDISO (proprietary) | CHOLMOD (MIT, ships with R) | dense | Eigen SimplicialLDLT |
| GPU | No | Yes (NNGP batching) | No | No |
| Nested Laplace | Yes (full) | Yes (phased) | No (MCMC only) | No (single Laplace) |
| Selected inversion | Yes | Yes (Feature 4) | No | No |
| Skewness correction | Yes | Phase 3c | No | No |
| Occupancy models | Via inlaOcc | Via tulpaOcc | Native | Manual |
| Spatial types | SPDE, ICAR, AR | NNGP, ICAR, BYM2, HSGP, SPDE, GP | NNGP | Any (manual) |
| NUTS for hyperparams | No | Feature 7 | No | Via tmbstan |
| Control | Binary blob | Full source | Full source | Full source |

Key advantage: tulpa is the only engine offering nested Laplace + GPU + CRAN + occupancy
models in a single stack. INLA has nested Laplace but no GPU and no CRAN. spOccupancy has
CRAN but only MCMC. TMB has CRAN but only single Laplace.

---

## Open Questions

1. **CHOLMOD thread safety**: CHOLMOD uses a `cholmod_common` workspace that is not
   thread-safe. For parallel inner evaluations (Feature 3b), each OpenMP thread needs its
   own `cholmod_common`. Verify this works with R's version of SuiteSparse.

2. **Matrix package API stability**: The C API for CHOLMOD via `Matrix` is technically
   internal. Check if Martin Maechler provides stable C-callable routines, or if we should
   vendor a SuiteSparse copy.

3. **Hyperparameter transforms**: INLA works on internal (log/logit) scale for
   hyperparameters. tulpa's Q-builder should also work on transformed scale to ensure the
   grid is well-conditioned. Verify the existing `optimize_theta` uses the same transforms.

4. **Memory for parallel grid evaluation**: Each inner Laplace needs its own workspace
   (CHOLMOD factor, mode vector, Hessian). For k=5 (43 grid points) with N=600K: 43 × N × 8
   bytes ≈ 200 MB for modes alone. Manageable, but should be allocated up front.
