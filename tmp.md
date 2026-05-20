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

## 1.4b complete — landed this session

**All Still-to-do items resolved:**

1. **BYM2 / CAR_proper spatial ops sparse fields** populated in
   `make_bym2_spatial_ops` (phi via `add_icar_prior_sparse` at tau=1
   plus theta IID diagonal; pattern via `add_car_pattern` on phi) and
   `make_car_proper_spatial_ops` (`add_car_proper_prior_sparse` +
   `add_car_pattern`).

2. **HSGP / NNGP spatial ops** kept on the dense path. Their sparse
   fields are left empty; the ST dispatch wrapper refuses
   `force_sparse = TRUE` with a clear `Rcpp::stop("... see 1.4c
   follow-up")` message. Auto-routing on `n_x >= SPARSE_THRESHOLD`
   skips them too (sparse_supported guard).

3. **Sparse ST runner** added in `src/nested_laplace.cpp` as
   `run_spatial_x_indexed_temporal_nested_laplace_sparse_impl` plus a
   dedicated `build_st_hessian_pattern` (chosen over reusing the
   joint-multi pattern builder — ST has a known fixed shape so the
   dedicated enumerator is cleaner) and `nl_scatter_obs_spatial_x_
   indexed_temporal_sparse`. Dispatch wrapper
   `run_spatial_x_indexed_temporal_nested_laplace_dispatch` routes
   force_sparse + sparse-supported through the sparse impl, falls back
   to the existing templated dense runner otherwise.

4. **`bool force_sparse = false`** added to all 15
   `cpp_nested_laplace_st_*` Rcpp entries. compileAttributes
   regenerated; R bindings expose `force_sparse = FALSE` naturally.

5. **Shim layer** (`src/tulpa_shims.cpp` lines 271-545) updated: the
   forward declarations now carry `bool force_sparse = false` defaults
   so existing `cpp_nested_laplace_st_*` calls in
   `tulpa_shims_nested_laplace.h` still link without changes.

6. **`tests/testthat/test-nested-laplace-st-sparse-equivalence.R`**
   added. 6 active dense-vs-sparse equivalence tests + 1 HSGP refusal
   smoke = 18 + 1 assertions, all green. Smoke fit on ICAR × AR1
   agrees to 1.2e-11 between paths.

### 1.4d partially landed — SparseCholeskySolver::factorize_with_ridge_retry

`SparseCholeskySolver::factorize_with_ridge_retry` added in
`src/sparse_cholesky.{h,cpp}`. On supernodal `NOT_POSDEF` it falls back to
simplicial LDL' with `dbound = ridge_init` and `CHOLMOD_NATURAL` ordering
(matching the dense column-natural elimination order), then as a last
resort applies a uniform diagonal ridge that grows geometrically. Wired
into both `laplace_newton_solve_sparse` (sparse_hessian.h) and
`laplace_newton_solve_joint_sparse` (laplace_newton_joint_sparse.h).

This makes the sparse path **robust** on doubly rank-deficient combos
(ICAR/BYM2 × RW1/RW2): Newton converges, log_marginal is finite, modes
are finite. It does NOT make sparse match dense bit-for-bit on those
inputs, because:

- Dense's clamp (`if (sum <= 0) sum = 1e-6`) is *asymmetric* — it only
  bumps non-positive Schur complements, not small-positive ones.
- CHOLMOD's `dbound` is *symmetric* — it clamps all `|D_jj| < dbound`,
  bumping both rank-deficient AND small-positive pivots.

For pathological rank-deficient inputs these clamps activate on
different pivot subsets and produce O(1)-O(10) differences in log_det.
The 4 doubly rank-deficient tests now use `.expect_finite_and_bounded`
(max_abs_div = 25 on log_marginal) rather than strict equivalence.

The fully principled fix (uniform upstream diagonal ridge in BOTH dense
and sparse paths so that no clamp ever fires) is still open. It would
remove the dense pivot-clamp hack entirely. Deferred because:

- Healthy fits are unaffected by ridge (1e-10 is well below typical
  pivots, so log_det shift is negligible).
- Rank-deficient log_marginals would shift slightly (replacing one
  numerically-arbitrary baseline with another, more consistent one).
- Existing rank-deficient absolute-value baselines (if any) would shift
  by ~9-23 units depending on ridge magnitude. Worth a closer audit
  before flipping.

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

## 1.5c complete — scale smoke validated linear scaling

Joint multi-arm sparse path (ICAR, chain graph, 2 arms = binomial +
gaussian with copy):

| n_s   | n_x  | wall_s | mem_MB | n_iter |
| ----- | ---- | ------ | ------ | ------ |
| 1e3   | 1004 |   0.06 |    181 |      5 |
| 1e4   | 1e4+ |   0.58 |    252 |      5 |
| 1e5   | 1e5+ |   6.48 |    176 |      5 |
| 1e6   | 1e6+ | 111.17 |    485 |      6 |

Log-log slope = **1.09** (linear=1.0, quadratic=2.0). No hidden
O(n_x^2) work in pattern enumeration, scatter, factor, or eta accumulation.

Extras:
- **BYM2 at n_s=1e5** (n_x ~ 2e5): wall 13.8s = 2.1x ICAR — perfectly
  linear in n_x (banded Cholesky O(n) post-symbolic).
- **4-cell outer grid at n_s=1e4**: per-cell 0.345s (lower than
  single-cell 0.58s). Symbolic-factor reuse works; warm-start from
  previous cell mode also drops Newton iters from 5 to 4 on cells 2-4.

Dev script: `dev_notes/scale_smoke_sparse_joint.R` and
`dev_notes/scale_smoke_bym2_and_grid.R`. Both gitignored (the smoke
runs in seconds at smallish sizes; rerun on demand).

For the original 20M outer-grid / n_s=1e6 target: per-cell ~2 minutes
serial. Outer grid parallelizes trivially (independent Newtons per
cell, solver pool already in `run_nested_laplace_grid`). Typical
hyper-grids are 10-100 cells, not 20M; large hyper-grids would call
for a cheaper screening pass (already in place via cheap-pass + tile
pilots).

## 1.4c complete — SPDE/HSGP/NNGP single-arm drivers unified

All three standalone drivers now route through the LatentBlock
infrastructure instead of bespoke pattern/scatter code. Net diff
**-424 lines, +412 lines** (mostly the new `make_nngp_block`).

* `cpp_nested_laplace_spde`: -170 lines after replacing inline
  pattern + scatter + log_prior with `make_spde_block` +
  `run_multi_block_nested_laplace_joint_sparse_impl`.
* `cpp_nested_laplace_hsgp`: migrated from dense Newton to sparse
  path via `make_hsgp_block` (DENSE_BASIS); pattern builder
  unconditionally fills the full beta x M, RE x M, M x M sub-blocks.
* `cpp_nested_laplace_nngp`: new `src/nngp_block_factory.h::make_nngp_block`
  (INDEXED_SINGLE + NN_K). Reuses the existing
  `apply_nngp_full_prior_sparse` and `make_nngp_prior_sparsity_pattern`
  helpers in `gpu_nngp_laplace.h`. Migrated from dense to sparse path.

All three single-arm entries now share the same pattern enumerator
(`build_joint_hessian_pattern`), scatter (`scatter_arm_obs_joint_multi_sparse`),
prior-scatter dispatch (`block.add_prior_sparse`), and CHOLMOD
factorization (`factorize_with_ridge_retry` from 1.4d). Future
additions to the joint path automatically reach single-arm.

Tests: test-spde (39 pass), test-nested-laplace-gp (16 pass), joint
sparse equivalence (12 pass), ST sparse equivalence (32 pass).

## 1.5a complete — direct pattern-correctness tests

`cpp_test_joint_pattern` Rcpp entry exposes the joint H sparsity
pattern as a CSC triple, bypassing all Newton/scatter machinery. Seven
test cases in `tests/testthat/test-joint-hessian-pattern.R` exercise:
ICAR (chain adjacency + observed-site beta cross), BYM2 (ICAR on phi,
diagonal on theta, same-site phi/theta coupling), RW1 (tridiagonal),
RW2 (pentadiagonal), AR1 (tridiagonal), IID (diagonal-only), and the
ICAR × RW1 cross-block coupling case. 70 assertions pass.

## 1.4d complete — robust factorization wired through every Laplace entry

`dispatch_factor_solve` / `dispatch_factor_log_det` in
`src/laplace_cholesky_dispatch.h` now call
`SparseCholeskySolver::factorize_with_ridge_retry` instead of bare
`factorize`. This is the dispatch wrapper used by the
`LikelihoodSpec` dense Laplace entry (`laplace_mode_spec_dense_impl`),
so model packages (`numdenom`, `tulpaObs`) plugging through that ABI
now inherit the same rank-deficient robustness the joint multi and ST
sparse drivers got in the first 1.4d landing.

Net 1.4d coverage:
* `laplace_newton_solve_sparse` (single-arm sparse direct entries)
* `laplace_newton_solve_joint_sparse` (joint multi-arm sparse)
* ST sparse runner (via the above)
* `dispatch_factor_solve` / `dispatch_factor_log_det` (LikelihoodSpec
  dense entry — `laplace_mode_spec_dense_impl`)

Every Laplace driver in tulpa now goes through `factorize_with_ridge_retry`
on the sparse path.

## 1.5d complete — shape coverage smoke

One small fit per row of the `hessian_plan.md` workload table, exercising
every (contrib_kind, prior_kind, n_arms, block-mix) cell currently in the
engine. Counterpart to 1.5a (pattern correctness), 1.5b (dense/sparse
equivalence) and 1.5c (linear scaling): not a perf test, just a logged
green-canary per shape so the §1.5 checklist closes.

| shape                  | n_arms | p_arm | blocks            | n_x | wall_s | n_iter |
| ---------------------- | ------ | ----- | ----------------- | --- | ------ | ------ |
| MOTIVATE cover hurdle  |   2    |   3   | bym2 + copy       |  56 |  0.11  |   4    |
| Single-arm BYM2        |   1    |   3   | bym2              |  53 |  0.02  |   5    |
| N-mix + occupancy      |   2    |   2   | icar              |  44 |  0.11  |   7    |
| Multi-process ratio    |   4    |   2   | icar + ar1        |  44 |  0.11  |   5    |
| Spline-heavy 1-arm     |   1    |  60   | iid               | 460 |  0.17  |   5    |
| ST joint               |   1    |   2   | icar x ar1        |  30 |  0.01  |   5    |
| SPDE mesh              |   1    |   1   | spde (FEM)        | 211 |  0.27  |   4    |
| HSGP                   |   1    |   1   | hsgp (M=25)       |  26 |  0.04  |   4    |
| NNGP                   |   1    |   1   | nngp (nn=8)       |  81  |  0.14  |   6    |
| User tgmrf             |   1    |   1   | tgmrf (AR1 CSC)   |  31 |  0.06  |   7    |

Every row: `log_marginal` finite at every outer-grid cell, modes finite,
inner Newton converged within 7 iterations. Joint multi-arm paths
(MOTIVATE, N-mix shape, multi-process ratio) exercised the `force_sparse =
TRUE` Newton; ST joint also routed through the sparse Newton. SPDE,
HSGP, and NNGP single-arm entries now share the same pattern enumerator
after 1.4c, so this run is the first end-to-end matrix-coverage
confirmation that all three migrated drivers still run on shapes outside
their dedicated tests.

Latent factors deliberately skipped — DENSE_BASIS with a
constraint-aware extension (sum-to-zero / first-zero on factor loadings
across obs) is out of scope until item 2 of the open list lands.

Smoke script: `dev_notes/shape_coverage_smoke.R` (gitignored). Re-run on
demand via `source("dev_notes/shape_coverage_smoke.R")`.

## 1.4e complete — uniform upstream diagonal ridge

Single `LAPLACE_UNIFORM_RIDGE * I = 1e-10 * I` added to H at every
Laplace factorize callsite — dense dispatch, sparse joint Newton,
sparse single-arm Newton, posterior sampler. Replaces both the dense
`if (sum <= 0) sum = 1e-6` in-pivot clamp and the sparse
`factorize_with_ridge_retry` simplicial-LDL'-with-dbound + uniform-ridge
fallback. Both paths now factor the SAME matrix and agree to numerical
tolerance even on doubly rank-deficient priors.

**Audit results (dev_notes/ridge_audit.R).** On three small ST fits:

| shape                  | n_grid | max abs(d-s) before | max abs(d-s) after | dense shift | sparse shift |
| ---------------------- | ------ | ------------------- | ------------------ | ----------- | ------------ |
| ICAR x AR1 (PD)        |   2    | 1.5e-10             | 1.6e-10            | 0.0         | 0.0          |
| ICAR x RW1 (rank-def)  |   2    | 11.22               | 3.6e-6             | -1.82       | +4.64        |
| ICAR x RW2 (rank-def)  |   2    | 10.53               | 4.1e-6             | -1.48       | +4.64        |

- **PD fits**: unchanged to numerical tolerance (ridge=1e-10 is well below
  typical pivots O(1)-O(10)).
- **Rank-deficient fits**: dense and sparse converge on the SAME new
  baseline; both shift toward a common value. The shift magnitude
  (~1-5 units, smaller than the rough 9-23 estimate in this doc's
  prior section) reflects the new theoretical baseline:
  `log_det = sum(log(lambda_i + ridge))` where lambda_i = 0 for the
  k rank-deficient directions contribute `k * log(ridge)`. This is a
  vague Gaussian prior on those directions; documented one-time
  baseline shift, not a regression.

**Files changed:**
- `src/laplace_cholesky.h` — `LAPLACE_UNIFORM_RIDGE`, `add_uniform_ridge_dense`,
  removed `if (sum <= 0) sum = 1e-6` clamp from both
  `cholesky_factorize_impl` and `cholesky_factorize_impl_raw`.
- `src/sparse_hessian.h` — `SparseHessianBuilder::add_uniform_ridge`,
  wired into `laplace_newton_solve_sparse` (per-iter + final).
- `src/laplace_newton_joint_sparse.h` — same wiring into
  `laplace_newton_solve_joint_sparse`.
- `src/laplace_cholesky_dispatch.h` — `dispatch_factor_solve` and
  `dispatch_factor_log_det` apply `add_uniform_ridge_dense` then call
  plain `factorize` (no more `factorize_with_ridge_retry`).
- `src/laplace_core.cpp::cpp_laplace_sample` — adds the ridge before
  the posterior-sampler Cholesky.
- `src/sparse_cholesky.{h,cpp}` — `factorize_with_ridge_retry` removed
  (no callers; per CLAUDE.md "no back-compat shims").
- `tests/testthat/test-nested-laplace-st-sparse-equivalence.R` —
  `.expect_finite_and_bounded` tightened from `max_abs_div = 25` to
  `lm_tol = 1e-4`; header docstring updated.

**Tests**: 22 nested-laplace files (~1100+ assertions), full ST sparse
equivalence (28), joint sparse equivalence (12), sparse-cholesky (30),
joint-hessian-pattern (70), spde (40), em-laplace (56), imh-laplace (22),
tgmrf families (186 across 8 files), nmix-spatial (50) — all green. No
regressions on hard-coded absolute log_marginal expectations.

## What's still open

Stage 2 work — arrow Schur, iterative S, Vecchia for NNGP, SYRK/Woodbury
for DENSE_BASIS — remains future, gated on Stage-1 profiling at scale.

Stage 1.6 (DENSE_BASIS follow-ups) is now complete; see §1.6 below.

---

## Stage 1.6 — DENSE_BASIS follow-ups (landed)

All three sub-stages shipped this session. Implementation notes below;
original design sketch retained verbatim further down for reference.

### 1.6b complete — HSGP-SVC via per-arm phi scaling

`.joint_block_spec_for_cpp` (R) accepts an optional `svc_column` field on
type='hsgp' specs. When set, the R parser row-scales each arm's basis at
fit-time:
```
phi_scaled[k][i, m] = phi[k][i, m] * arm[k]$X[i, svc_column]
```
The scaled phi is passed to the existing C++ `make_hsgp_block` factory
unchanged — no new contrib_kind, no C++ surface change. The block then
contributes
```
eta_i += X[i, svc_column] * sum_m Phi[i, m] * sqrt_S_m * beta_m
       = X[i, svc_column] * f(s_i)
```
which is the SVC semantic. Validation: error on out-of-range or
non-positive svc_column, error when nrow(phi[[k]]) != nrow(arm[k]$X).

R signature change: `.joint_block_spec_for_cpp(p, n_arms, block_index)`
gains a fourth arg `arms = NULL`; the multi-block dispatcher passes the
normalised arm specs. Other block types ignore the arms arg.

### 1.6c complete — HSGP multi-scale composition

No new code. The 1.3c block factory + the joint multi-block dispatcher
already compose multiple HSGP blocks; declaring two `type='hsgp'` block
specs in `prior_list` with different eigenvalue spectra additively
decomposes the latent field. Outer grid is the Cartesian product of each
block's paired `(sigma2_grid, lengthscale_grid)` axes. Test in
`test-nested-laplace-joint-hsgp-svc.R` exercises a 4-cell × 4-cell = 16-
cell multi-scale fit.

Multi-output co-regionalization HSGP (option (b) in the design plan) is
deferred until a downstream workload asks for it — would need a K x K
LKJ-cholesky hyperparameter on the basis coefficients.

### 1.6a complete — latent factors via BILINEAR_FACTOR contrib_kind

**Block layout (F = 1).** A single latent factor block stores
```
x[block.start .. block.start + n_latent)        ->  u_1, ..., u_n_latent
x[block.start + n_latent .. block.start + size) ->  lambda_1, ..., lambda_n_arms
```
where `size = n_latent + n_arms`. Each obs (i, k_arm) contributes
```
eta_i += x[u_slot(obs_idx(i))] * x[lambda_slot(k_arm)]
```
which is bilinear in x — not linear. Both u and lambda are jointly
optimized by the inner Newton.

**New contrib_kind.** `BlockContribKind::BILINEAR_FACTOR` plus a new
field on LatentBlock:
```
std::function<std::pair<int,int>(int i, int k_arm)> obs_factor_lambda;
```
returning the global (u_slot, lambda_slot) pair for each obs.

**Scatter (Gauss-Newton).** Dispatch in
`scatter_arm_obs_joint_multi_sparse` (sparse joint path) emplaces two
active dofs per obs with bilinear weights:
```
active[u_slot]      = lambda * d_eff
active[lambda_slot] = u * d_eff
```
The existing active × active intra-block, β × active and RE × active
formulas then fill the right Gauss-Newton gradient and Hessian entries —
including the (u, lambda) mixed-curvature cross term. The g_i term (full-
Newton mixed second derivative) is dropped per Gauss-Newton convention,
matching INLA / Stan-Laplace / TMB practice for nonlinear-eta GLM blocks.

**Eta accumulator.** A dedicated BILINEAR_FACTOR case in
`compute_eta_joint_sparse_dispatch` adds `d_e * u * lambda` once per
obs. The active-dof linear formula used by INDEXED_MULTI / DENSE_BASIS
would double-count the product, so eta computation is the one place
BILINEAR can't reuse the existing dispatch surface.

**Pattern enumerator.** `detail::resolve_indexed_dofs` returns the (u,
lambda) pair as two block-local active dofs; the existing β × active,
RE × active, and active × active per-obs enumeration then fills every
required pattern entry (including the (u, lambda) cross within the
block). No new pattern-builder branch needed beyond the resolver.

**Identifiability.** A bilinear (u, lambda) product has overall-sign and
overall-scale degeneracies. Both are broken by tight Gaussian anchors
on (u_1, lambda_1):
```
u_1     ~ N(0, anchor_eps^2)
lambda_1 ~ N(1, anchor_eps^2)
```
with `anchor_eps = 1e-3` by default. The other slots carry user-facing
priors: `u_j ~ N(0, sigma_u^2)`, `lambda_k ~ N(0, sigma_lambda^2)`,
defaults sigma_u = sigma_lambda = 1.0. Anchoring via priors rather than
hard reparam keeps the per-slot scatter uniform across all (u, lambda)
slots — much simpler than reconstructing a pinned slot inside every
scatter call.

**Outer grid.** Zero outer-grid axes for first-ship latent factors.
sigma_u, sigma_lambda, anchor_eps are scalar block-spec fields, not
gridded hyperparameters. The block is hyperparameter-free from the
outer-integration perspective.

**Files added.**
- `src/latent_factor_block_factory.h` — `make_latent_factor_block`.
- `tests/testthat/test-nested-laplace-joint-latent-factor.R` — smoke +
  recovery (lambda_1 anchor, lambda ratio, u correlation) + validation
  (n_latent < 2 errors).

**Files changed.**
- `src/latent_block.h` — `BILINEAR_FACTOR` enum value + `obs_factor_lambda`.
- `src/joint_hessian_pattern.h` — `resolve_indexed_dofs` BILINEAR case.
- `src/nested_laplace_joint_multi.h` — scatter dispatch + eta dispatch.
- `src/nested_laplace_joint_multi.cpp` — R parser dispatch for
  `type = 'lf'`.
- `R/nested_laplace_joint.R` — `.joint_block_spec_for_cpp` entry,
  `arms` arg threading.
- `R/nested_laplace.R` — `.NL_REGISTRY$lf` (1 x 0 grid),
  `.nl_axis_quantiles` 0-col guard.
- `.joint_posterior_moments_multi` — guard 0-axis blocks.
- `.joint_multi_layout` — recognize `lf` block size = n_latent + n_arms.

**Test results.** 13 / 13 assertions pass: smoke (3), recovery (5,
including lambda_1 ≈ 1 ± 0.05, lambda ratio within ±0.30, u correlation
> 0.6 against simulated truth), validation (2). Full broader sweep
green: joint sparse equivalence, joint multi parity, joint BYM2 /
ICAR / beta / prune / adaptive / phi-grid / parallel, multi-block
recovery (110 assertions), nested-laplace-cpp, ST sparse equivalence,
joint-hessian-pattern (70), spde (40), nested-laplace-gp (16),
sparse-cholesky (30), HSGP-SVC + multi-scale (14).

**Out of scope for this ship.**
- F > 1 factors. Would need a rotation constraint (lower-triangular
  Lambda matrix) on top of the (u_1, lambda_1) anchors. The
  BILINEAR_FACTOR scatter / pattern / eta surface generalizes
  trivially — the new factory would emit F × 2 = 2F active dofs per
  obs and tile F (u_f, lambda_f) sub-blocks. Identifiability ceiling
  is the only design work left.
- Tau / sigma axes on the outer grid. The factor field amplitude is
  pinned at sigma_u = 1.0 (default) and absorbed by the lambda
  loadings. Gridding sigma_u would add one outer-grid axis per
  factor; doable but not needed for the F = 1 first ship.
- Full-Newton mixed-curvature cross term (drop g_i term in scatter).
  Standard Gauss-Newton practice — leave the g_i term to a future
  refinement only if a recovery study shows it matters.

---

## Stage 1.6 design plan (original sketch, for reference)

Three DENSE_BASIS block types were flagged out-of-scope in 1.3c. Each
needs different surgery on the LatentBlock interface; this section is
the design sketch to pick up in a fresh session.

### 1.6a — Latent factors (highest priority, hardest)

**Goal.** A length-`K` factor block with one loading vector per obs.
Shape: `eta_arm_k = X_k beta_k + sum_{f=1..F} lambda_f[arm_k] * eta_factor_f[obs_idx_f]`,
where `lambda_f[k_arm]` is the per-arm loading on factor `f` and
`eta_factor_f` is the latent factor field (length `n_latent`). Standard
identifiability constraints: `sum_i eta_factor_f[i] = 0` (sum-to-zero)
OR `eta_factor_f[1] = 0` (first-zero) for each factor.

**Why DENSE_BASIS doesn't fit.** DENSE_BASIS as currently defined
(`hsgp_block_factory.h`) assumes every obs touches every coefficient
in the block via `basis_eval(i, k_arm, out)`. Latent factors are
**INDEXED_MULTI** in disguise: each obs touches exactly one index in
`eta_factor_f` per factor, multiplied by `lambda_f[arm_k]`. That fits
INDEXED_MULTI's `(local_idx, weight)` interface — `weight = lambda_f[arm_k]`
— except the loadings `lambda_f[arm_k]` are themselves latent (must be
estimated), not fixed at fit-time.

**The constraint piece.** Sum-to-zero on `eta_factor_f` makes the prior
precision `Q_f` rank-deficient by 1 per factor. The 1.4e uniform ridge
handles this for the Newton solve, but the rank deficit breaks
identifiability with `lambda_f` (any uniform shift of `eta_factor_f`
can be absorbed by adjusting per-arm intercepts in `beta_k`). Solutions:

  1. **Hard constraint via reparameterization.** Internally store
     `eta_factor_f[2..n_latent]` and reconstruct `eta_factor_f[1] = -sum(rest)`
     (sum-to-zero) or `eta_factor_f[1] = 0` (first-zero). Newton solver
     never sees the constrained DOF. Requires a `reparam_in` /
     `reparam_out` pair on the block — new hooks.
  2. **Lagrange multiplier.** Append a Lagrange row to H enforcing
     `sum(eta_factor_f) = 0`. Cleaner but inflates n_x by F (one extra
     DOF per factor) and breaks the sparsity assumption.
  3. **Soft constraint (penalty).** Add `kappa * (sum eta_factor_f)^2`
     to log_prior with kappa large. Numerically OK but doesn't truly
     remove the flat direction; relies on uniform ridge to mop up.

  **Recommendation: option (1).** Matches how INLA handles the same
  constraint (`extraconstr` in `f()` formula), preserves sparsity,
  doesn't grow n_x. The new `LatentBlock` field would be
  `std::function<void(...)> reparam_to_constrained` /
  `std::function<void(...)> reparam_from_constrained`, called inside
  the Newton loop's scatter and eta-accumulator.

**Loadings as hyperparameters.** Each factor has F * n_arms loadings
`lambda_f[k_arm]`. Treat them as hyperparameters (outer-grid axes) —
prohibitive at F >= 2 because the grid grows as
`product(loading_grid_size^(F * n_arms))`. Alternatives:

  1. **Pin one loading per factor.** `lambda_f[arm_1] := 1` (canonical
     for identifiability), grid only over the remaining `lambda_f[arm_k]`
     for `k > 1`. Reduces to `(n_arms - 1) * F` outer-grid axes.
  2. **Treat loadings as latent.** Append `lambda` to the latent vector
     `x`, fit jointly with `eta_factor`. Newton then optimizes both.
     Identifiability constraint enforced via the reparam in (1) above.
     Cleaner — no outer-grid axis explosion — but requires gradient/
     Hessian contributions from `lambda` (per-arm scatter ops).

  **Recommendation: option (2).** Keeps the outer grid small and
  matches how SPDE-SVC and other multi-parameter latent priors handle
  per-arm scaling. Per-arm scatter now has TWO contributions per obs
  for each factor: gradient wrt `eta_factor_f[obs_idx_f(i)]`
  (= `dlogL/deta * lambda_f`) and gradient wrt `lambda_f[arm_k]`
  (= `dlogL/deta * eta_factor_f[obs_idx_f(i)]`).

**Files to touch**:
- `src/latent_block.h` — add `reparam_to_constrained` /
  `reparam_from_constrained` fields; new contrib_kind value (or reuse
  INDEXED_MULTI with a constraint flag).
- New `src/latent_factor_block_factory.h` — analogous to `make_spde_block`.
- `src/nested_laplace_joint_multi.cpp` — R parser entry for `type = "lf"`,
  C++ dispatch in `build_joint_blocks_from_spec`.
- `R/nested_laplace_joint.R` — `.joint_block_spec_for_cpp` entry, new
  spec validation.
- `tests/testthat/test-latent-factor.R` — recovery test (simulate F=2
  factors with known loadings + sum-to-zero, fit, check loading
  recovery + factor field recovery up to sign/scale identifiability).

**Effort estimate**: 2-3 days. The hard part is the reparam + scatter
gradient terms for `lambda`; the rest is plumbing.

### 1.6b — HSGP-SVC (spatially-varying coefficient)

**Goal.** Replace a fixed-effect column with an HSGP-expanded coefficient
field: `eta_i = X_i[1] * f_1(s_i) + X_i[2] * f_2(s_i) + ...`, where each
`f_j(s)` is its own HSGP block with shared eigenvalues but a per-arm
basis evaluation `Phi_i * X_i[j] * sqrt(S)`. That is the workload table's
"HSGP-SVC (per-term scaling)" row.

**Why current HSGP doesn't fit.** `make_hsgp_block` (1.3c) computes
`basis_eval(i, k_arm, out) = Phi[i] * sqrt_S` — a constant basis
evaluation per obs. For SVC each obs needs `Phi[i] * X_i[j] * sqrt_S`,
which depends on a covariate at obs `i` (column `j` of the arm's design).

**Design.** Two clean options:

  1. **One HSGP block per SVC term.** User declares
     `list(type = "hsgp", basis = Phi, ..., svc_column = j)` for each
     column. The block factory bakes `Phi * X[, j]` into a per-arm
     scaled basis at fit-time, then proceeds as a standard DENSE_BASIS
     HSGP. Simple, composable, no new fields on LatentBlock.
  2. **Single HSGP-SVC block with `svc_columns = c(j1, j2, ...)`.**
     One block contributes K independent factor fields (each on the
     same eigenvalue spectrum); per-obs basis_eval emits a
     `(K * m_total)`-wide vector. Saves repeated `sqrt_S` computation
     across the K terms but couples them; harder to give per-term
     `sigma2 / lengthscale` axes.

  **Recommendation: option (1).** Composable, matches how INLA
  spatial-SVC works (one `f()` per column), and the user gets
  per-term hyperparameter axes for free via the outer-grid Cartesian
  product. Tradeoff: K independent symbolic factors instead of one.
  At HSGP scales (M = 50-500) this is fine.

**Files to touch**:
- `R/nested_laplace_joint.R` — `.joint_block_spec_for_cpp` entry for
  `type = "hsgp"` accepts an optional `svc_column` field (1-based index
  into `arm$X`). The R-side builds the scaled `Phi_scaled[i, m] =
  Phi[i, m] * X[i, svc_column]` once and passes it to the existing C++
  factory unchanged.
- New test exercising HSGP-SVC + joint dispatch.

**Effort estimate**: half-day. Mostly R-side plumbing; existing
DENSE_BASIS infrastructure handles the rest.

### 1.6c — HSGP-MSGP (multi-scale / multi-output)

**Goal.** Two interpretations. Default to (a) until user clarifies:

  (a) **Multi-scale.** A latent field that's the sum of `S` independent
      HSGP components at different lengthscales: `f(s) = sum_{l=1..S} f_l(s)`,
      where each `f_l` has its own `(sigma2_l, lengthscale_l)` axis. Used
      for multi-scale spatial heterogeneity (long-range trend + local
      variation).
  (b) **Multi-output.** One basis, shared across `K` correlated arms,
      with a `K x K` cross-arm covariance matrix on the basis
      coefficients. Closer to a co-regionalization model.

**Design for (a) "multi-scale".** Same as HSGP-SVC option (1): one
HSGP block per scale, each contributes its own basis to the joint
latent. Outer-grid Cartesian product over `(sigma2, lengthscale)` per
scale; expect to use 2-3 scales in practice. No new infrastructure
beyond what HSGP-SVC needs.

**Design for (b) "multi-output".** Needs a new `K x K` matrix
hyperparameter — outer-grid over LKJ-cholesky factors or similar.
Heavier; defer until a user actually requests it.

**Effort estimate**: half-day for (a), 2-3 days for (b).

### Recommended ordering

1. **1.6b (HSGP-SVC)** first — half-day, pure R-side, validates the
   "compose by adding more blocks" design pattern against an actual
   workload.
2. **1.6c (HSGP-MSGP, multi-scale)** — half-day, falls out of 1.6b
   trivially. Defer the multi-output variant.
3. **1.6a (latent factors)** — 2-3 days, the hard one. Standalone PR
   because it adds the constraint-aware extension to LatentBlock.

Total ~3-4 days end-to-end for full Stage 1.6.

### Open design questions for new session

- Does the latent-factors workload need centred (sum-to-zero) or
  first-zero constraint? Different downstream interpretation; pick
  whichever matches the published applications you have in mind.
- For HSGP-SVC, do we need a `(sigma2, lengthscale)` axis pair PER
  scaled column, or do the SVC fields share hyperparameters? Sharing
  shrinks the grid but couples the inferences.
- Multi-output HSGP-MSGP: who needs it? If no one in tulpa's user base
  has it on a roadmap, defer indefinitely.

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
- `src/nested_laplace.cpp` — full: ops bundle sparse fields +
  RW1/RW2/AR1 + ICAR + BYM2 + CAR_proper + sparse pattern + scatter +
  runner + dispatch wrapper + force_sparse on 15 ST Rcpp entries
- `src/tulpa_shims.cpp` — 15 ST forward decls extended with
  `bool force_sparse = false`
- `src/sparse_cholesky.{h,cpp}` — `factorize_with_ridge_retry` with
  simplicial LDL'+dbound+natural-ordering fallback and uniform-ridge
  last resort (1.4d)
- `src/sparse_hessian.h` — `laplace_newton_solve_sparse` calls
  `factorize_with_ridge_retry` instead of bare `factorize` (1.4d)
- `src/laplace_newton_joint_sparse.h` — same (1.4d)
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
- `tests/testthat/test-nested-laplace-st-sparse-equivalence.R` — NEW
- `hessian_plan.md`, `potential_issues.md` — design docs (from earlier)
