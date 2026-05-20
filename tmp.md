# tulpa joint sparse Laplace path — resumption notes

State as of 2026-05-20. Read alongside `hessian_plan.md` and
`potential_issues.md` for the original design intent.

## Tactical context

- Target workload: 20M outer-grid cells, n_sites = 10^6+, joint multi-arm
  Laplace (MOTIVATE cover hurdle shape: binomial + gaussian/beta-on-positives,
  shared spatial via INLA `copy=` semantics).
- Blocker at scale was `DenseMat H` in `NewtonScratchJoint` (n_x x n_x
  allocation; 8 TB at n_x = 10^6). Sparse path bypasses it entirely.
- This session built the foundation 1.0–1.5b. 1.4b is partially done.

## What's complete and validated

**1.0** `LatentBlock` interface extended with `BlockContribKind` enum
(INDEXED_SINGLE / INDEXED_MULTI / DENSE_BASIS), `PriorFillKind` enum,
`add_prior_pattern`, `add_prior_sparse`, `obs_indices`, `basis_eval` fields.
See `src/latent_block.h`.

**1.1** `src/joint_hessian_pattern.h::build_joint_hessian_pattern` enumerates
the joint Hessian sparsity pattern once at fit-time. Dispatches on
contrib_kind for per-arm × per-block fill; handles cross-block coupling.

**1.2** `scatter_arm_obs_joint_multi_sparse` in
`src/nested_laplace_joint_multi.h`. Sparse twin of the dense obs scatter.
Lower-triangle single-write semantics; handles all three contrib_kinds.

**1.3a** All 7 joint indexed-single block factories in
`src/nested_laplace_joint_multi.cpp` populate the four new fields:
ICAR (copy + non-copy), BYM2-φ + BYM2-θ, CAR_proper (copy + non-copy),
RW1, RW2, AR1, IID. Sparse helpers added in
`src/laplace_spatial_priors.{h,cpp}` (`add_icar_prior_sparse`,
`add_car_proper_prior_sparse`, `add_car_pattern`) and
`src/laplace_temporal_priors.{h,cpp}` (`add_rw1/rw2/ar1_precision_sparse`
+ `add_rw1/rw2/ar1_pattern`).

**1.3b** SPDE block as INDEXED_MULTI. New factory header
`src/spde_block_factory.h::make_spde_block`. R parser entry added in
`R/nested_laplace_joint.R::.joint_block_spec_for_cpp` (type "spde"). C++
dispatch added in `build_joint_blocks_from_spec`. Uses shared
`SpdeQBuilder` and per-arm `ARows` (`build_A_rows`); `prep(k_grid)`
rebuilds Q values per outer cell.

**1.3c** Plain HSGP as DENSE_BASIS. New factory
`src/hsgp_block_factory.h::make_hsgp_block`. R parser entry + C++ dispatch
added. Uses cached `sqrt_S` (rebuilt in `prep`) and per-arm `phi_flat`.

  - **Out of scope (flagged for follow-up):** latent factors (don't fit
    DENSE_BASIS; they're INDEXED_MULTI with sum-to-zero / first-zero
    cross-obs constraints — needs new constraint-aware extension);
    HSGP-SVC and HSGP-MSGP (per-term scaling complexity).

**1.3d** tgmrf as USER_CSC. New factory
`src/tgmrf_block_factory.h::make_tgmrf_block`. R parser + C++ dispatch
added. Per-grid CSC triples + log|Q_k| + log p(theta_k) copied into shared
vectors at factory time.

**1.4a** Sparse joint Newton solver +
`run_multi_block_nested_laplace_joint_sparse_impl` + dispatch in
`run_multi_block_nested_laplace_joint`:

- `src/laplace_newton_joint_sparse.h::laplace_newton_solve_joint_sparse`
  — CHOLMOD-only factor/solve, no dense fallback, no `DenseMat H` in
  `NewtonScratchJointSparse`.
- `compute_eta_joint_sparse_dispatch` in
  `src/nested_laplace_joint_multi.h` — eta accumulator dispatching on
  contrib_kind.
- `run_multi_block_nested_laplace_joint_sparse_impl` — full sparse outer
  driver. Builds pattern once, reuses across all outer-grid cells.
  Serial outer-grid (parallel sparse = follow-up; pattern enumeration is
  the only barrier).
- Dispatch: routes to sparse when `force_sparse || n_x >=
  SPARSE_THRESHOLD || any block.contrib_kind != INDEXED_SINGLE`.
- `add_per_arm_beta_re_priors_sparse` in
  `src/nested_laplace_joint_core.h`.

**1.5b** `force_sparse` plumbed through R: `tulpa_nested_laplace_joint`
→ `.joint_dispatch_multi` / single-block `call_kernel` →
`.joint_call_kernel_via_multi` → `cpp_nested_laplace_joint_multi` →
`run_multi_block_nested_laplace_joint`. New test
`tests/testthat/test-nested-laplace-joint-sparse-equivalence.R` asserts
sparse and dense paths agree to 1e-6 (log_marginal) and 1e-5 (modes) on
BYM2 + copy, ICAR no-copy, CAR_proper. **12 assertions pass.** BYM2/ICAR
multi-block regression suites still pass.

## What's partially done (1.4b) — pick up here

**Goal:** wire all 15 `cpp_nested_laplace_st_*` entries
(icar/bym2/car_proper/hsgp/nngp × rw1/rw2/ar1) to a true sparse path,
the same way 1.4a wires the joint multi-arm driver.

**Done this session:**
- `SpatialBlockOps` (in `src/nested_laplace.cpp` around line 815) extended
  with `add_prior_pattern` + `add_prior_sparse` fields.
- `IndexedPriorOps` (around line 1068) extended with same fields.
- `make_rw1_ops`, `make_rw2_ops`, `make_ar1_ops` populate the sparse fields
  using the helpers from `laplace_temporal_priors.{h,cpp}`.
- `make_icar_spatial_ops` populates the sparse fields using
  `add_car_pattern` + `add_icar_prior_sparse`.

**Still to do:**

1. Populate sparse fields in `make_bym2_spatial_ops` and
   `make_car_proper_spatial_ops` (same shape as ICAR — use
   `add_car_pattern` for both; `add_icar_prior_sparse` for BYM2-φ and
   `add_car_proper_prior_sparse` for CAR_proper). BYM2 is two sub-blocks
   (φ at `start..start+n_units`, θ at `start+n_units..start+2*n_units`)
   so it needs two prior fills.

2. **HSGP / NNGP spatial ops:** these need new sparse helpers because
   their prior/design patterns are not simple adjacency. Decision: either
   (a) implement properly (HSGP via DENSE_BASIS-like full block fill;
   NNGP via `(I-A)' D^{-1} (I-A)` precision pattern) or (b) stub with
   explicit `Rcpp::stop("force_sparse not yet supported for hsgp/nngp
   spatial — see 1.4c follow-up")`. Recommend (b) for first ship; (a)
   belongs in 1.4c.

3. Write the sparse path of `run_spatial_x_indexed_temporal_nested_laplace`
   (in `src/nested_laplace.cpp`, around line 955). Sibling to the
   existing dense runner. Needs:
   - Build the joint H pattern via the existing
     `build_joint_hessian_pattern` (works for ST too if you frame the
     spatial + temporal blocks as `LatentBlock` entries) OR write a
     dedicated `build_st_hessian_pattern` (probably cleaner — ST has a
     known fixed shape: β/β + β/RE + RE/diag + spatial-prior +
     temporal-prior + per-obs (β/spatial, β/temporal, spatial/temporal,
     spatial/spatial, ...)).
   - Sparse twin of `nl_scatter_obs_spatial_x_indexed_temporal` (the obs
     scatter that walks `spatial_ops.obs_p/obs_local_idx/obs_weight`
     plus the temporal `t_idx`).
   - Call `laplace_newton_solve_sparse` instead of `laplace_newton_solve`.
   - Pass through `spatial_ops.add_prior_sparse` and
     `temporal_ops.add_prior_sparse` for the prior side.

4. Add `bool force_sparse = false` parameter to all 15
   `cpp_nested_laplace_st_*` Rcpp entries; route to sparse path when
   `force_sparse || n_x >= SPARSE_THRESHOLD`.

5. Plumb `force_sparse` through the R wrappers
   (`tulpa_nested_laplace_st_*` if they exist; look in
   `R/nested_laplace.R` — grep for `cpp_nested_laplace_st_`).

6. Extend `tests/testthat/test-nested-laplace-joint-sparse-equivalence.R`
   (or new `test-nested-laplace-st-sparse-equivalence.R`) with at least
   ICAR×AR1 dense-vs-sparse asserts. If full coverage: 9 combos for
   ICAR/BYM2/CAR_proper × RW1/RW2/AR1; HSGP/NNGP skipped if stubbed.

**Compile-check cadence:** every 2-3 file edits, run
```
/c/Program\ Files/R/R-4.6.0/bin/Rscript.exe -e \
  'setwd("C:/GillesC/documents/dev/tulpa"); print(tryCatch(pkgbuild::compile_dll(quiet = TRUE), error=function(e) paste("FAIL:", conditionMessage(e))))'
```

**Validation test:**
```
/c/Program\ Files/R/R-4.6.0/bin/Rscript.exe -e \
  'Sys.setenv(NOT_CRAN="true"); setwd("C:/GillesC/documents/dev/tulpa"); \
   devtools::load_all(quiet=TRUE); \
   testthat::test_file("tests/testthat/test-nested-laplace-joint-sparse-equivalence.R", reporter="summary")'
```

## What's pending after 1.4b

**1.4c** Wire single-arm `cpp_nested_laplace_nngp` /
`cpp_nested_laplace_spde` / `cpp_nested_laplace_hsgp` to sparse
LatentBlock-based factories (the standalone drivers in `spde_laplace.cpp`
and `nested_laplace.cpp` predate the LatentBlock interface; should be
refactored to use the new factories from 1.3b/c).

**1.4d** Wire `laplace_spec_dense` (model-package facing `LikelihoodSpec`
dense entry) to sparse dispatch.

**1.5a** Per-block pattern-correctness tests — direct C++ unit tests of
`build_joint_hessian_pattern` and each block's `add_prior_pattern`.
Currently we rely on numerical equivalence (1.5b) which would catch
pattern bugs in covered combos but not in untested tail combos.

**1.5c** Shape coverage + scale smoke tests — exercise the sparse path
at n_s ≥ 10000 to confirm it doesn't crash or pathologically slow down.
Less precise than 1.5b but tests scale-up behavior.

## Quick orientation tips

- Sparse joint path is at `src/nested_laplace_joint_multi.h::
  run_multi_block_nested_laplace_joint_sparse_impl` — read this first to
  see the established pattern (eta dispatch → scatter dispatch → newton
  → log_marginal); ST sparse path should mirror it.
- The dispatch fork is at the top of
  `run_multi_block_nested_laplace_joint`. Same fork pattern needed in
  `run_spatial_x_indexed_temporal_nested_laplace`.
- Pattern enumeration helpers in `laplace_spatial_priors.cpp`
  (`add_car_pattern`) and `laplace_temporal_priors.cpp`
  (`add_rw1/rw2/ar1_pattern`) are the building blocks. Cross-block fill
  (β/spatial, spatial/temporal) is what's currently in
  `build_joint_hessian_pattern` for joint multi — replicate the
  per-obs enumeration shape for ST.

## File-by-file delta from this session

- `src/latent_block.h` — interface extension
- `src/joint_hessian_pattern.h` — NEW
- `src/laplace_newton_joint_sparse.h` — NEW
- `src/nested_laplace_joint_multi.h` — sparse impl + dispatch
- `src/nested_laplace_joint_multi.cpp` — block factories +
  hsgp/spde/tgmrf dispatch + force_sparse param
- `src/nested_laplace_joint_core.h` — sparse β/RE helper
- `src/nested_laplace.cpp` — partial: ops bundle sparse fields +
  RW1/RW2/AR1 + ICAR
- `src/laplace_spatial_priors.{h,cpp}` — sparse twins +
  add_car_pattern
- `src/laplace_temporal_priors.{h,cpp}` — sparse twins +
  rw1/rw2/ar1 patterns
- `src/spde_block_factory.h` — NEW
- `src/hsgp_block_factory.h` — NEW
- `src/tgmrf_block_factory.h` — NEW
- `R/nested_laplace_joint.R` — force_sparse plumbing + spde/hsgp/tgmrf
  parser entries
- `tests/testthat/test-nested-laplace-joint-sparse-equivalence.R` — NEW
- `hessian_plan.md`, `potential_issues.md` — design docs (from earlier)
