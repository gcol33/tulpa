# tulpa — Open Work

Punch list. Completed items removed; see git history for what shipped.

## P1 — Sibling-package work

### 1. Agent E — `tulpaRatio` refactor
**Where:** `~/Documents/dev/numdenom` (sibling repo).
**What:** rename `Package: numdenom` → `tulpaRatio` in DESCRIPTION; strip the
vendored `src/` engine; depend on tulpa via `LinkingTo: tulpa`; register the
ratio likelihood through `LikelihoodSpec`. Verify one benchmark reproduces.
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
**Blocked on:** item 1 (Agent E) — once `tulpaRatio` registers via
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

1. **#1 Agent E (tulpaRatio)** in sibling repo, then **#4 strip legacy**.
2. **#2 Agent F (tulpaGlmm gap audit)** — parallelizable with #1, report-only.
3. **#8 EM MI/Gibbs** when tulpaGlmm needs it.
4. **#5 CUDA live test** — schedulable anytime; needs a GPU build.

`plan.md` is the live narrative; this file is the punch list.
