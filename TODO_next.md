# tulpa — Next Features TODO

**Date:** 2026-03-25
**Status pass:** 2026-04-29 — items #1–#6 all landed and tested.
**Current state:** 247+ tests passing, CHOLMOD + nested Laplace + SPDE working

> **2026-04-29 audit.** Every item in this file is already implemented and
> covered by tests. Verification run on 2026-04-29:
>
> | # | Item | Status | Tests passed |
> |---|------|--------|---|
> | 1 | `spatial_spde()` R API | ✅ done | 23 (test-spatial-spde-api) + 29 (test-spde) |
> | 2 | Selected inversion (Takahashi) | ✅ done | 34 (test-selected-inversion) |
> | 3 | Rational SPDE (fractional ν) | ✅ done | 17 (test-rational-spde) |
> | 4 | BYM2 nested Laplace | ✅ done | 36 (test-nested-laplace-bym2) |
> | 5 | SPDE in tulpaOcc (`occu_spde`) | ✅ done | 10 (test-spde-occ in tulpaOcc) |
> | 6 | GPU-batched NNGP (CPU fallback path) | ✅ done | 3 (test-gpu-nngp) |
>
> See `R/spatial.R:2388` (`spatial_spde`), `R/rational_spde.R`,
> `src/sparse_cholesky.cpp:91` (`selected_inversion_diagonal`),
> `src/nested_laplace.cpp:130` (`cpp_nested_laplace_bym2`),
> `src/gpu_nngp_laplace.h` + `src/gpu_cuda.h` (cuSOLVER batched Cholesky
> with CPU fallback), tulpaOcc `R/spatial.R:197` (`occu_spde`).
>
> Live CUDA-kernel testing for #6 still requires a GPU build of the package
> (`launch_build.ps1` from RESOLVE) — the CPU-fallback path is what tests
> currently exercise. That's the only remaining loose end across the six.

---

## 1. R-level `spatial_spde()` API ✅ DONE

**Priority:** P0 — makes SPDE usable
**Effort:** Low
**Depends on:** tulpaMesh (scaffolded), SPDE Laplace (done)

### What
Wire tulpaMesh as hard dependency of tulpa. Create `spatial_spde()` R function that:
1. Takes `coords` (formula or matrix) + optional `boundary` + `max_edge` + `cutoff`
2. Calls `tulpaMesh::tulpa_mesh()` to build mesh
3. Calls `tulpaMesh::fem_matrices()` to get sparse C, G, A
4. Stores mesh + FEM matrices in a `tulpa_spatial` object
5. Fit functions extract CSC slots and call `cpp_laplace_fit_spde` / `cpp_nested_laplace_spde`

Also accept external matrices: `spatial_spde_custom(C, G, A)` for fmesher/rSPDE users.

### User-facing API
```r
# Native (zero external deps)
fit <- tulpa_fit(y ~ x1 + x2, data = df, family = "binomial",
                 spatial = spatial_spde(coords = ~ lon + lat, max_edge = c(0.5, 2)))

# With fmesher
mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.5, 2))
fem <- fmesher::fm_fem(mesh)
A <- fmesher::fm_basis(mesh, loc = coords)
fit <- tulpa_fit(y ~ x1, data = df, family = "binomial",
                 spatial = spatial_spde_custom(C = fem$c0, G = fem$g1, A = A))
```

### Files to create/modify
- tulpa `DESCRIPTION`: add `tulpaMesh` to Imports
- tulpa `R/spatial.R`: add `spatial_spde()`, `spatial_spde_custom()`
- tulpa `R/fit.R` or equivalent: route SPDE spatial spec to `cpp_laplace_fit_spde`

---

## 2. Selected Inversion (Takahashi Equations) ✅ DONE

**Priority:** P1 — needed for posterior summaries
**Effort:** Medium (~150 lines C++)
**Depends on:** CHOLMOD (done)
**Landed at:** `src/sparse_cholesky.cpp:91` (`SparseCholeskySolver::selected_inversion_diagonal`), exported via `cpp_selected_inversion_diagonal`. 34 tests pass.

### What
Compute diagonal of Q⁻¹ (marginal variances per latent field element) without full matrix inversion. Uses the elimination tree from CHOLMOD's Cholesky factorization.

### Algorithm
Given Q = LDL' from CHOLMOD, compute Z = Q⁻¹ element-by-element for positions where L is nonzero:
```
For j = n down to 1:
    Z[j,j] = 1/D[j] - Σ_{k>j, L[k,j]≠0} L[k,j] * Z[k,j]
    For i > j where L[i,j] ≠ 0:
        Z[i,j] = -Σ_{k≥i, L[k,j]≠0} L[k,j] * Z[k,i]
```
Complexity: O(nnz(L)) — linear in Cholesky factor sparsity.

### Files to create/modify
- tulpa `src/sparse_cholesky.h`: add `selected_inversion(std::vector<double>& diag_inv)` to `SparseCholeskySolver`
- tulpa `src/sparse_cholesky.cpp`: implement using CHOLMOD factor struct (L in CSC + permutation)

### Validation
- Compare diagonal of selected inversion against `solve(Q)` diagonal for small problems (N < 500)
- Must match to machine precision

---

## 3. Rational SPDE (Fractional Matérn Smoothness) ✅ DONE

**Priority:** P1 — the differentiator vs INLA
**Effort:** Medium
**Depends on:** SPDE Laplace (done), SpdeQBuilder (done)
**Landed at:** `R/rational_spde.R` (`rational_spde_coefficients`), C++ side wired through `src/spde_qbuilder.h` and `src/spde_laplace.cpp`. 17 tests pass.

### What
Extend SPDE from integer smoothness (ν = 1) to fractional smoothness (any ν > 0) using rational polynomial approximation of (κ² - Δ)^(α/2). Following Bolin, Simas & Xiong (2023, JRSS-B).

### How
Classical SPDE: Q = τ²(κ⁴C + 2κ²G + GC⁻¹G) — only works for α = 2 (ν = 1 in 2D).

Rational SPDE: Q = Pₘ'·C·Pₘ where Pₘ is a degree-m rational polynomial in L = κ²C + G.

Implementation:
1. Compute rational polynomial coefficients via best rational approximation (Padé or Chebyshev) for target α
2. Build Pₘ(L) as a product of sparse matrices: Pₘ = Πᵢ(L - rᵢI)/(L - pᵢI) where rᵢ, pᵢ are roots/poles
3. Q = Pₘ'·C·Pₘ — same sparsity pattern as classical SPDE
4. Add ν as estimable hyperparameter in nested Laplace grid (3D: range × σ² × ν)

### Key property
Same sparsity pattern → CHOLMOD symbolic reuse. Same SpdeQBuilder pattern — just different coefficient computation in `rebuild()`.

### Reference
- Bolin, Simas & Xiong (2023). Rational SPDE approach for Gaussian random fields with general smoothness. JRSS-B.
- rSPDE R package (reference implementation)

### Files to create/modify
- tulpa `src/spde_qbuilder.h`: add `RationalSpdeQBuilder` that stores rational polynomial coefficients
- tulpa `src/spde_laplace.cpp`: add `cpp_laplace_fit_rspde`, `cpp_nested_laplace_rspde`
- tulpa `R/spatial.R`: add `nu` parameter to `spatial_spde()`

---

## 4. BYM2 Nested Laplace ✅ DONE

**Priority:** P1 — quick win
**Effort:** Low
**Depends on:** Nested Laplace ICAR (done)
**Landed at:** `src/nested_laplace.cpp:130` (`cpp_nested_laplace_bym2`), exported via `R/RcppExports.R`. 36 tests pass.

### What
2D grid over (tau, rho) for BYM2 spatial models. CCD with k=2 → 9 grid points.

### How
Nearly identical to `cpp_nested_laplace_icar` but:
- Grid over (log_tau, logit_rho) instead of just log_tau
- At each point: BYM2 precision Q = τ[ρ·Q_icar + (1-ρ)·I]
- Warm-start chain, shared CHOLMOD (same pattern)

### Files to create/modify
- tulpa `src/nested_laplace.cpp`: add `cpp_nested_laplace_bym2`
- tulpa `R/nested_laplace.R`: add `nested_laplace_bym2()`
- Tests: compare against single-point Laplace at grid points

---

## 5. Wire SPDE into tulpaOcc ✅ DONE

**Priority:** P1 — end-to-end occupancy + continuous spatial
**Effort:** Medium
**Depends on:** spatial_spde() API (#1)
**Landed at:** tulpaOcc `R/spatial.R:197` (`occu_spde`) wraps `tulpa::spatial_spde`. 10 tests pass.

### What
Enable `occ(..., spatial = spatial_spde(...))` in tulpaOcc. The occupancy likelihood plugs into tulpa's SPDE Laplace via the cross-package callback interface.

### How
tulpaOcc's `occ_fit()` currently handles ICAR/BYM2/NNGP spatial via the occupancy likelihood callback. SPDE needs:
1. Recognize `type = "spde"` in the spatial specification
2. Pass mesh FEM matrices (C, G, A) to tulpa's SPDE Laplace
3. The occupancy likelihood (detection × occupancy marginal) maps through A to mesh nodes
4. Nested Laplace over SPDE hyperparameters (range, σ²)

### User-facing API
```r
library(tulpaOcc)
fit <- occ(
  ~ elevation + forest,    # occupancy covariates
  ~ effort,                # detection covariates
  data = survey_data,
  spatial = spatial_spde(coords = ~ x + y, max_edge = c(0.5, 2))
)
summary(fit)
# Posterior for range, sigma, occupancy probability at each mesh node
```

### Files to create/modify
- tulpaOcc `R/fit.R`: route SPDE spatial spec
- tulpaOcc `src/occ_fit.cpp`: SPDE variant of occupancy Laplace (or delegate to tulpa's SPDE Laplace with occupancy likelihood callback)
- tulpaOcc `R/spatial.R`: re-export `spatial_spde` from tulpa

---

## 6. GPU-Batched NNGP ✅ DONE (CPU fallback path; live CUDA test pending)

**Priority:** P2 — scaling to 600K locations
**Effort:** High
**Depends on:** CUDA build infrastructure (RESOLVE project)
**Landed at:** `src/gpu_cuda.h` dlopen-loads cuSOLVER and exposes `cusolverDnDpotrfBatched`; `src/gpu_nngp_laplace.h` (`batch_nngp_scatter`) builds m×m C matrices and dispatches to GPU when available, with transparent CPU fallback; `src/gpu_backend.h::gpu_batched_cholesky_solve` completes the forward + backward solve. CPU-fallback path tested (3 tests in test-gpu-nngp). Live CUDA-kernel testing requires building tulpa with CUDA on; see `~/.claude/skills/cuda-build/SKILL.md`.

### What
For NNGP with m neighbors and N locations, the bottleneck is N independent m×m dense Cholesky factorizations. Batch these on GPU via `cusolverDnDpotrfBatched`.

### Performance target
At N=600K, m=15: CPU takes seconds per gradient → GPU takes milliseconds.

### How
1. Allocate GPU buffers for N covariance matrices (each m×m)
2. Compute covariance values on GPU (kernel: distance → Matérn/exponential)
3. Batch Cholesky: `cusolverDnDpotrfBatched` factorizes all N matrices simultaneously
4. Batch solve: `cusolverDnDpotrsBatched` for conditional means
5. Transfer results back to CPU for the Newton step

### Integration
- New spatial solver backend: `GPSolver::GPU` (already in types.h enum)
- Wire into `laplace_mode_gp` via solver dispatch
- Fallback to CPU for machines without CUDA

### Files to create/modify
- tulpa `src/gpu_nngp.cu`: CUDA kernel for batched NNGP
- tulpa `src/gpu_backend.cpp`: CPU↔GPU orchestration
- tulpa `src/Makevars`: conditional CUDA compilation
- tulpa `configure`: detect CUDA toolkit

### Dependencies
- CUDA Toolkit ≥ 11.0
- cuSOLVER library
- Reference: RESOLVE project's `launch_build.ps1` for CUDA build on Windows

---

## Execution Order

```
#1 spatial_spde() API ──→ #5 Wire into tulpaOcc
       │
       ├──→ #3 Rational SPDE
       │
       └──→ #2 Selected inversion (independent)

#4 BYM2 nested (independent, quick win)

#6 GPU NNGP (independent, long-term)
```

Recommended order: #1 → #4 → #2 → #3 → #5 → #6
