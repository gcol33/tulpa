# Bug: SPDE inner Newton diverges on ABI v14 shim signature

## Summary

The ABI v14 SPDE shim (`tulpa_nested_laplace_spde` in
`src/tulpa_shims_nested_laplace.h`, signature with `re_idx`, `n_re_groups`,
`sigma_re`, `store_Q`, and `NestedLaplaceShimResult`) does not converge.
Across every grid point:

- CHOLMOD reports "matrix not positive definite" repeatedly.
- Inner Newton iterations hit `max_iter` (50 by default).
- `log_marginal` values are identical across all grid points (so grid
  weights collapse to uniform `1/n_grid`).
- The fixed-effect estimates do not recover their true values
  (intercept off by 0.3 + drift, slope ~0).
- The mesh-node latent has very large magnitudes (range hundreds when
  the true marginal SD is O(1)).

Reproduces for `family = "poisson"` and `family = "gaussian"`.

## Reproduction

```r
library(tulpaMesh)
library(Matrix)

set.seed(42); n <- 100L
coords <- cbind(runif(n), runif(n))
mesh <- tulpaMesh::tulpa_mesh(coords = coords, max_edge = 0.05, extend = 0.2)
fem  <- tulpaMesh::fem_matrices(mesh, obs_coords = coords, lumped = TRUE)
A    <- as(fem$A, "CsparseMatrix")
G1   <- as(fem$G, "CsparseMatrix")
n_mesh <- as.integer(fem$n_mesh)
true_field <- 1.5 * sin(2 * pi * coords[,1]) + 1.0 * cos(2 * pi * coords[,2])
eta  <- 0.3 + true_field
y    <- rpois(n, exp(eta))
X    <- matrix(1.0, nrow = n, ncol = 1)

res <- tulpa:::cpp_nested_laplace_spde(
  y = as.numeric(y), n_trials = as.integer(rep(1L, n)),
  X = X,
  re_idx = as.numeric(rep(0, n)), n_re_groups = 0L, sigma_re = 1.0,
  A_x = A@x, A_i = A@i, A_p = A@p,
  n_obs = n, n_mesh = n_mesh,
  C0_diag = fem$va,
  G1_x = G1@x, G1_i = G1@i, G1_p = G1@p,
  range_grid = 0.3, sigma_grid = 1.0,
  nu = 1.0, family = "poisson", phi = 1.0,
  max_iter = 50L, tol = 1e-6, n_threads = 1L
)
# res$n_iter == 50 (max_iter), res$log_marginal finite but unconverged.
```

## Likely cause

Two related observations:

1. **Stale unit tests.** `tests/testthat/test-spde.R` still calls
   `cpp_nested_laplace_spde(y, n_trials, X, A_x, ...)` without `re_idx`,
   `n_re_groups`, `sigma_re`. Both tests "nested Laplace SPDE runs with
   2D hyperparameter grid" and "nested Laplace SPDE warm-start reduces
   iterations" fail with `argument "re_idx" is missing`. Suggests the
   signature was bumped without test-coverage being re-baselined.

2. **Inner Newton step / matrix assembly.** Because every grid point
   produces identical `log_marginal` and CHOLMOD always fails on the
   joint precision Q, the most likely site is the Q assembly inside
   `src/spde_laplace.cpp::cpp_nested_laplace_spde`. Worth checking:
   - Whether the `beta` and `re` blocks contribute their priors to Q
     (rather than only `w_mesh` contributing through `tau * (kappa^4 C0
     + 2 kappa^2 G1 + G1 C0^-1 G1)`).
   - Whether the cross-block H[beta, w_mesh] term uses the *current*
     iterate (vs. zero or stale `A` only).
   - Whether the step-halving / damping logic was carried over from the
     pre-v14 ICAR Newton.

## Impact

- **Upstream:** SPDE shim is non-functional today on the v10-style ABI
  v14 surface.
- **Downstream (tulpaGlmm Day-41):** the glue
  (`tulpaGlmm/R/nested_laplace.R::nl_build_spde` +
  `tulpaGlmm/src/glmm_nested_laplace.cpp::cpp_glmm_nl_spde_fit`) is
  wired through, structurally correct (right shapes, right dispatch
  labels, right draws reconstruction `A %*% w_mesh`), but the smoke
  test
  (`tulpaGlmm/dev_notes/day-spde-smoke.R`) cannot check field recovery
  until this is fixed. The smoke test was reduced to structural-only
  checks with a TODO to re-enable the correlation gate after the
  Newton converges.

## Suggested fix priority

Re-baseline `tests/testthat/test-spde.R` against the new signature,
**then** trace why the new Newton diverges. Reusing the ICAR-pattern
Newton from `src/nested_laplace.cpp` would likely "just work" — the
v10-style ABI alignment is precisely the spec under which ICAR / RW /
BYM2 / HSGP / NNGP all converge today.

## Discovered

2026-05-13, during tulpaGlmm Day-41 SPDE wiring. Also see
`tulpaMesh/fix-mesh-zero-triangles.md` for a related blocker in the
mesh refinement path that surfaced during this work.
