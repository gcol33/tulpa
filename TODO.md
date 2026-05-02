# tulpa — Open Work

Punch list. Completed items removed; see git history for what shipped.

## P0.5 — Refactor follow-ups (in flight)

Three dedups and two splits landed on 2026-05-02:

- ICAR/CAR neighbor-loop kernel (src/icar_kernel.h) — replaces four
  near-identical copies of phi'Q phi and Q*phi across hmc_sampler.cpp,
  hmc_gradients.cpp, hmc_icar_collapsed.h, and the inline neighbor-loop
  in spatial_gmrf_prior_grad.
- Newton-loop helpers (src/laplace_newton_loop.h) — eval_penalized_log_lik,
  step_halving_update, max_abs_step, finalize_log_marginal. Used by both
  the dense (laplace_newton.h) and sparse (sparse_hessian.h) PIRLS solvers.
- Cholesky dispatch (src/laplace_cholesky_dispatch.h) — dispatch_factor_solve
  + dispatch_factor_log_det collapse the two near-identical sparse-then-
  dense fallback blocks in laplace_newton.h into one.
- tulpa_priors.h split into 12 per-family headers (re, icar, gp, msgp,
  hsgp, temporal, mstemporal, svc, tvc, latent, st, zioi). Umbrella
  preserves the `#include "tulpa_priors.h"` API.
- gibbs_spatial.h split into data + icar + bym2 + hsgp. GibbsResult
  hoisted from mid-ICAR-section into the shared data header.
- hmc_gradient_vectorized.h split into 5 fragments (workspace, passes,
  fused, main, shared). Umbrella header opens `namespace vectorized {`,
  includes the 5 fragments in dependency order, closes the namespace.
  Fragments are header-guarded but namespace-less so they expand inside
  the surrounding `namespace tulpa_hmc {` opened in hmc_gradients.cpp.
  pkgbuild::compile_dll(force = TRUE) clean.
- hmc_sampler.h split into 7 fragments (decls, nuts_infra, mass_blocks,
  adapt, chain_state, funcs, config). Umbrella opens
  `namespace tulpa_hmc {`, includes fragments in dependency order
  (mass-block hierarchy precedes DenseMassMatrix; DualAveraging precedes
  ChainState), closes the namespace. Fragments are header-guarded and
  namespace-less. pkgbuild::compile_dll(force = TRUE) clean.
- log_post_impl.h split into 8 fragments: 1 namespace-scope helper
  (`log_post_car_proper_det.h`) + 7 function-body fragments included
  inside `compute_log_post_impl<T>` (priors_basic, priors_disp_spatial,
  priors_gp, priors_temporal, priors_svc_tvc_latent, priors_st_zi,
  likelihood). Function-body fragments rely on lexical scope (params,
  data, layout, log_post, beta_*, phi_*, re_vals, ...) and are NOT
  standalone-compilable; each is included exactly once per umbrella so
  no header guards. pkgbuild::compile_dll(force = TRUE) clean; full
  testthat suite passes.
- hmc_nuts_sampler.cpp split into 9 namespace-scope fragments
  (dual_avg, leapfrog, find_epsilon, helpers, optimized, softabs,
  mass_init, chain, parallel). Umbrella `.cpp` retains all includes
  + `using namespace Rcpp` + opens `namespace tulpa_hmc`, then
  includes fragments in original definition order, then closes the
  namespace. Fragments are header-guarded and namespace-less; they are
  NOT standalone translation units (Makevars-style globbing already
  picks up only `.cpp` files in `src/`, so the fragment `.h` files are
  not compiled standalone). pkgbuild::compile_dll(force = TRUE) clean;
  full testthat suite passes.
- hmc_icar_collapsed.h split into 9 fragments (workspace, kernels,
  logdet, grad, unit_lik, mode, full, grad_H, log_post). Umbrella
  retains the original comment header + includes + the
  `using tulpa_hmc::ModelData/ModelType/SpatialType` declarations,
  then includes the 9 fragments in dependency order (workspace before
  matvecs/CG; kernels before logdet/grad; both before unit_lik/mode;
  full + grad_H build on the prior helpers; log_post is the top-level
  wrapper). Fragments are header-guarded but namespace-less and rely
  on the umbrella's using-decls — they are NOT standalone-compilable.
  pkgbuild::compile_dll(force = TRUE) clean; full testthat suite
  passes (825 PASS, 0 FAIL, 2 SKIP — same as pre-split).
- hmc_gradient_gp_impl.h split into umbrella + 4 fragments at function
  boundaries: `hmc_gradient_gp_handcoded.h` (gp NNGP),
  `hmc_gradient_gp_collapsed.h` (collapsed GP via inner Laplace),
  `hmc_gradient_icar_collapsed_grad.h` (ICAR/BYM2 collapsed — 540 lines,
  the largest fragment), and `hmc_gradient_temporal_gp.h` (the three
  GP+temporal / temporal-GP / multiscale-GP+temporal combos). Different
  from the analytical/composite splits: these fragments contain
  *complete* function definitions, not function-body slices, so each
  fragment is self-contained (relies only on the namespace tulpa_hmc
  and helper includes already opened in `hmc_gradients.cpp`). Each is
  `#include`d exactly once by the umbrella so no header guards.
  pkgbuild::compile_dll(force = TRUE) clean; testthat 825 PASS / 0
  FAIL / 2 SKIP.
- hmc_gradient_composite_impl.h split into umbrella + 4 function-body
  fragments along the existing `Phase 1`–`Phase 4` markers
  (phase1 extract params, phase2_priors, phase3_loop, phase4_post).
  Same pattern as the analytical / log_post_impl splits — fragments rely on
  lexical scope (`params`, `data`, `layout`, `grad`, `cp`, `beta_num`,
  `beta_denom`, `sigma_re`, `re`, `phi_num`, `phi_denom`, `obs_log_lik`,
  `fuse_lp`, `is_binomial`, `N`, ...) declared inside
  `compute_gradient_composite`. Fragments are NOT standalone-compilable;
  each is `#include`d exactly once by the umbrella so no header guards.
  Phase 4 ends with the fused log-posterior output block; the umbrella
  adds nothing after the last include but the closing `}`.
  pkgbuild::compile_dll(force = TRUE) clean; testthat 825 PASS / 0 FAIL
  / 2 SKIP.
- hmc_gradient_analytical_impl.h split into umbrella + 5 function-body
  fragments (priors_basic, priors_misc, lik_vec, lik_scalar, post).
  Same pattern as the prior log_post_impl.h split — fragments rely on
  lexical scope (`params`, `data`, `layout`, `grad`, `beta_num`,
  `beta_denom`, `sigma_re`, `tau_re`, `re_prior_grad_sigma`,
  `grad_re_slopes_lik`, `n_re_terms_slopes`, `slopes_nc`,
  `re_nc_flat`, `nc_L_flats`, `nc_sigmas_vec`, `phi_num`,
  `phi_denom`, `obs_log_lik`, `compute_lp`, ...) declared inside
  `compute_gradient_analytical`. The fragments are NOT
  standalone-compilable; each is `#include`d exactly once by the
  umbrella so no header guards. Umbrella keeps
  `can_use_analytical_gradient`, the helper forward decls,
  `compute_gradient_analytical`'s signature + parameter extraction,
  the five fragment includes (priors_basic → priors_misc → lik_vec
  → lik_scalar → post; the post fragment ends with the fused
  log-posterior output block), and the closing brace.
  pkgbuild::compile_dll(force = TRUE) clean; testthat 825 PASS / 0
  FAIL / 2 SKIP.
- hmc_rcpp_fit.cpp split into 2 translation units (multi-`.cpp`).
  `hmc_rcpp_fit.cpp` keeps the main `cpp_hmc_fit` export (~733 lines);
  the new `hmc_rcpp_fit_gp.cpp` owns `cpp_hmc_fit_gp` and
  `cpp_hmc_fit_gp_v2` (~682 lines). Each new TU re-includes
  `hmc_sampler.h`, `hmc_gradient_check.h`, `hmc_modeldata_builders.h`,
  `Rcpp.h`, `<atomic>`. `pkgbuild::compile_dll(force = TRUE)` clean;
  testthat 825 PASS / 0 FAIL / 2 SKIP.
- laplace_core.cpp split into 3 translation units (multi-`.cpp`).
  `laplace_core.cpp` now keeps the NNGP/Matérn helpers, the dense and
  dense-multi-RE mode finders, and their R exports. `laplace_core_spatial.cpp`
  owns `laplace_mode_spatial` / `_bym2` / `_rsr` plus the matching
  `cpp_laplace_fit_spatial` / `_bym2` / `_rsr` exports;
  `laplace_core_gp.cpp` owns `laplace_mode_gp` / `_multiscale_gp` /
  `_multiscale_temporal` plus their exports. The `laplace_result_to_list`
  helper was promoted from a per-TU `static` to an `inline` in
  `laplace_core.h` (under `namespace tulpa`) so every TU shares one
  definition. `pkgbuild::compile_dll(force = TRUE)` clean; testthat
  825 PASS / 0 FAIL / 2 SKIP.
- pg_binomial.cpp split into 6 translation units (multi-`.cpp` split,
  not umbrella-+-fragments — `Rcpp::compileAttributes()` only scans
  `.cpp` files for `// [[Rcpp::export]]`, so each export must live in
  its own `.cpp`). Pattern: `pg_binomial.cpp` keeps namespace tulpa
  helpers/updates + `pg_binomial_gibbs_impl` + the basic exports
  (`cpp_pg_binomial_gibbs`, `cpp_pg_binomial_gibbs_spatial`); five
  new siblings own one variant each — `pg_binomial_bym2.cpp`,
  `pg_binomial_gp.cpp` (carries the file-local
  `pg_nngp_conditional` helper), `pg_binomial_temporal.cpp`,
  `pg_binomial_multiscale_gp.cpp`, `pg_binomial_rsr.cpp`. Each new
  `.cpp` re-includes `pg_shared.h`, `pg_rng.h`, `linalg_fast.h`,
  `Rcpp.h`, and forward-declares the `tulpa::update_*` helpers it
  needs. After the split, `Rcpp::compileAttributes()` re-emitted the
  same export entries (only their order changed) and
  `pkgbuild::compile_dll(force = TRUE)` was clean; full testthat
  suite still 825 PASS / 0 FAIL / 2 SKIP.

**Still open from the 2026-05-02 punch list:**

- **Modularization milestones** (items #1, #2, #4, #5, #8 below) —
  multi-session work each.

**Audit + first response shipped 2026-05-02:**

- tulpaGlmm gap audit appended to
  `~/Documents/dev/tulpaGlmm/REIMPLEMENTATION_PLAN.md` as
  *Phase 0.5 — tulpa gap audit*. Headline finding: the original
  Phase-0 claim that "only NUTS is exported via R_RegisterCCallable"
  is stale — all nine ABI shim headers already exist plus extras.
- Audit's Gap 1 (Laplace shims are family-enum-coupled, not
  LikelihoodSpec-driven) has a first-cut fix landed:
  - New `EtaWeightsFn` callback on `LikelihoodSpec`
    (`inst/include/tulpa/likelihood.h`) — optional per-obs
    `(grad_eta, neg_hess_eta)` for IRLS.
  - `tulpa::laplace_mode_spec_dense_impl` in `src/laplace_spec.cpp` —
    routes per-obs log-lik + IRLS weights through the spec instead of
    the `family` C-string. Reuses `dispatch_factor_solve` /
    `finalize_log_marginal` so it stays in lockstep with the
    family-enum solver.
  - `inst/include/tulpa/laplace_spec_api.h` exposes the C-shim;
    `tulpa_laplace_spec_dense` registered in `src/tulpa_shims.cpp`.
  - `TULPA_ABI_VERSION` bumped 2 → 3 (LikelihoodSpec layout change).
  - `tests/testthat/test-laplace-spec.R` cross-checks the spec path
    against the family-enum reference for Gaussian + iid RE and
    Gaussian no-RE — modes match within 1e-5.
  - **First-cut scope:** `n_processes == 1`, at most one iid RE term.
    Multi-process / random-slope / spatial / temporal variants are
    follow-on work; the shim rejects them with a clear error today.

## P1 — Sibling-package work

### 1. Clean downstream generic prototype
**Where:** start from the smallest downstream package / example that can avoid
the old ratio engine entirely.
**What:** implement a true `LikelihoodSpec` integration: package-owned response
struct, likelihood callbacks, optional residual / extra-gradient callbacks, and
an R-side builder for generic `ModelData` with `n_processes > 0`.
**Do not:** continue incremental `tulpaRatio` bridge wrappers around its
vendored C++ entry points. That preserves the wrong boundary.
**Unblocks:** item 2 (strip `ModelData::legacy`).

### 2. Agent F — `tulpaGlmm` gap audit
**Where:** `~/Documents/dev/tulpaGlmm/REIMPLEMENTATION_PLAN.md`.
**What:** report-only. List what tulpa must expose for GLMM (Gaussian /
Poisson / binomial `LikelihoodSpec` factories, dispersion handling, link
functions). No implementation.

### 3. Wire SPDE into tulpaOcc
**Where:** sibling `tulpaOcc`.
**State:** `occu_spde` already lives at `tulpaOcc R/spatial.R:197` and wraps
`tulpa::spatial_spde`. The end-to-end occupancy + continuous-spatial wiring
through tulpa's nested SPDE Laplace still needs verification at higher N and
the in-package vignette.

## P2 — Tulpa core

### 4. Strip `ModelData::legacy` (`LegacyRatioData`)
**Where:** `src/hmc_sampler.cpp` (~413 references) and adjacent headers.
**Blocked on:** item 1 — once one downstream model runs through
`LikelihoodSpec`, the `n_processes == 0` branch can be removed.
**Action:** delete `LegacyRatioData`, drop the `n_processes == 0` branches
throughout, bump `TULPA_ABI_VERSION`.

### 5. Live CUDA-kernel testing for GPU-batched NNGP
**State:** CPU-fallback path tested (3 tests in `test-gpu-nngp`). The cuSOLVER
batched-Cholesky path compiles but has not been run on a real GPU build.
**Action:** build with CUDA on (`launch_build.ps1` from RESOLVE), run the
NNGP path against the CPU reference, and pin a numerical-equivalence test.

## P3 — Cosmetic / scaffolded

### 6. Namespace cleanup
**State:** functionally resolved — `tulpa_hmc::*` etc. all alias to unified
`tulpa::*` types. A mass-rename to drop the `tulpa_*` prefixes inside `src/`
would remove the second-name confusion. Pure cosmetic.

### 7. `likelihood.h` forward-decl audit
**State:** done in spirit (`inst/include/tulpa/likelihood.h:9-10` `#include`s
the actual autodiff headers). Audit to confirm no fwd-decl drift creeps back
in during future edits.

### 8. EM+Laplace — MI and Gibbs corrections
**Where:** `R/em_laplace.R:241`.
**State:** EM loop, per-submodel family/offset (#3), and `m_step_extra`
callback (#4) all ship. Only the correction passes remain stubbed —
`correction = "mi"/"gibbs"` raises a clear error today.
**Missing:** MI correction (draw hard z from weights, refit, Rubin's pool)
and Gibbs correction (warm-started z|θ → θ|z chain).

## P4 — Deferred (need decisions)

- **Inference-mode dispatch registry** (P2.2) — replace hardcoded dispatch
  with a registry / dispatch table.
- **`compute_log_lik_only` single-pass** — currently derives likelihood by
  subtraction (2× `compute_log_post`); a single O(N) observation-loop pass
  would halve the cost for SMC weight evaluation.
  (`src/hmc_sampler.cpp:2362`, `TODO(#6 follow-up)`)
- **ST_IV NC parameterization** (Kronecker spectral decomposition; math-heavy).
- **CUDA wiring** (`src/gpu_cuda.h`); Windows build risk.
- **OpenCL backend** (currently pure stubs in `gpu_backend.h:200,220`).
- **`tests/testthat/test-inference-modes.R` placeholder** — fill out once
  P2.2 dispatch refactor has a stable shape.
- **Formula parser duplicate-bar coalescing** — `(1|g) + (x|g)` produces two
  RE specs; lme4 merges them. Wasteful but not incorrect for diagonal cov.

## Recommended order

1. **#1 clean generic prototype**, then **#4 strip legacy**.
2. **#2 Agent F (tulpaGlmm gap audit)** — parallelizable with #1, report-only.
3. **#8 EM MI/Gibbs** when tulpaGlmm needs it.
4. **#5 CUDA live test** — schedulable anytime; needs a GPU build.

`plan.md` is the live narrative; this file is the punch list.
