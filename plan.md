# tulpa Implementation Plan

## Next-session pickup

**Branch:** `master`.
**Last engine commit:** `6c759b2 refactor: extract shared helpers in nested_laplace + shims, fix naming` (2026-04-30).

Working tree is clean. Rebuild to start:

```bash
"/c/Program Files/R/R-4.6.0/bin/R.exe" CMD INSTALL --no-test-load --no-multiarch . > /tmp/install.log 2>&1
"/c/Program Files/R/R-4.6.0/bin/Rscript.exe" dev_notes/run_nl_regression.R
```

**Current priority:** Agent E (`tulpaRatio` refactor, `~/Documents/dev/numdenom`)
or EM MI/Gibbs corrections (`R/em_laplace.R:241`).

### Reference docs

- `fix.md` — full ABI shim surface (all shipped as of 2026-04-30).
- `CLAUDE.md` — boundary rules, EM+Laplace engine state.
- `dev_notes/run_nl_regression.R` — nested-Laplace + CCD regression runner.
- `nested_laplace_proposal.md`, `features_plan.md` — math and engineering context.

---

## Ecosystem layout

| Package | Role | State |
|---|---|---|
| `tulpa` | Generic Bayesian engine | Active |
| `tulpaRatio` | Ratio/rate/proportion models | Still vendored numdenom; refactor pending |
| `tulpaOcc` | Occupancy models | Plug-in |
| `tulpaGlmm` | GLMM via lme4/glmmTMB-style API | Scaffolded |
| `tulpaMesh` | CDT meshes for SPDE | Mature |

Model packages register likelihoods via `LikelihoodSpec` (`inst/include/tulpa/likelihood.h`).

---

## Open work

### Model packages (parallel)

**Agent E — `tulpaRatio` refactor** (`~/Documents/dev/numdenom`)
- Rename `Package: numdenom` → `tulpaRatio` in DESCRIPTION.
- Strip vendored `src/` engine; depend on tulpa via `LinkingTo: tulpa`.
- Register ratio likelihood through `LikelihoodSpec`. Verify one benchmark.
- Unblocks: strip `ModelData::legacy` (item below).

**Agent F — `tulpaGlmm` gap audit** (`~/Documents/dev/tulpaGlmm/REIMPLEMENTATION_PLAN.md`)
- Report-only. List what tulpa must expose for GLMM (Gaussian / Poisson /
  binomial `LikelihoodSpec` factories, dispersion handling, link functions).

### Tulpa core

**Strip `ModelData::legacy` (`LegacyRatioData`)** — blocked on Agent E.
- Delete `LegacyRatioData`; drop `n_processes == 0` branches throughout
  `hmc_sampler.cpp` (~413 references). Bump `TULPA_ABI_VERSION`.

**Wire SPDE into tulpaOcc** — `occu_spde` ships in tulpaOcc; end-to-end
verification at higher N and the in-package vignette still pending.

**EM MI/Gibbs corrections** (`R/em_laplace.R:241`)
- EM loop, per-submodel family/offset, and `m_step_extra` all ship.
- Missing: MI correction (draw hard z, refit, Rubin's pool) and Gibbs
  correction (warm-started z|θ → θ|z chain).

**Live CUDA test for GPU-batched NNGP** — CPU path tested; cuSOLVER
batched-Cholesky path compiles but needs a GPU build (`launch_build.ps1`).

### Cosmetic / low-priority

- **Namespace cleanup**: mass-rename `tulpa_hmc::*` / `tulpa_zi::*` to
  `tulpa::*` inside `src/`. Aliases already resolve correctly; pure cosmetic.
- **`likelihood.h` forward-decl audit**: confirm no fwd-decl drift.
- **Inference-mode dispatch registry** (P2.2): replace hardcoded dispatch
  with a registry / dispatch table.
- **`compute_log_lik_only` single-pass**: currently 2× `compute_log_post`;
  a single O(N) pass halves SMC weight cost (`src/hmc_sampler.cpp:2362`).

## Deferred (need decisions)

- **ST_IV NC parameterization** (Kronecker spectral decomposition; math-heavy).
- **CUDA wiring** (`src/gpu_cuda.h`); Windows build risk.
- **OpenCL backend** (pure stubs in `gpu_backend.h:200,220`).
- **`tests/testthat/test-inference-modes.R` placeholder** — fill once P2.2 lands.
- **Formula parser duplicate-bar coalescing** — `(1|g) + (x|g)` produces two
  RE specs; lme4 merges them. Not incorrect for diagonal cov; low priority.
