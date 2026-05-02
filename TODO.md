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

**Still open from the 2026-05-02 punch list:**

- **Modularization milestones** (items #1, #2, #4, #5, #8 below) —
  multi-session work each.

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
