# tulpa Implementation Plan

## Next-session pickup

**Branch:** `master`.
**Last engine commit:** `6c759b2 refactor: extract shared helpers in nested_laplace + shims, fix naming` (2026-04-30).

Working tree is clean. Rebuild to start:

```bash
"/c/Program Files/R/R-4.6.0/bin/R.exe" CMD INSTALL --no-test-load --no-multiarch . > /tmp/install.log 2>&1
"/c/Program Files/R/R-4.6.0/bin/Rscript.exe" dev_notes/run_nl_regression.R
```

**Current priority:** clean core design before more downstream refactors.
Do not continue incremental `tulpaRatio` bridge work. The right boundary is:
model packages provide `LikelihoodSpec` + model-specific response data, and
all inference runs through `tulpa::ModelData` with `n_processes > 0`. The
legacy ratio path (`ModelData::legacy`, `n_processes == 0`) should be removed
from core once one clean downstream prototype proves the generic path.

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
| `tulpaRatio` | Ratio/rate/proportion models | Defer; do not drive core design from its vendored engine |
| `tulpaObs` | Occupancy models | Plug-in |
| `tulpaGlmm` | GLMM via lme4/glmmTMB-style API | Scaffolded |
| `tulpaMesh` | CDT meshes for SPDE | Mature |

Model packages register likelihoods via `LikelihoodSpec` (`inst/include/tulpa/likelihood.h`).

---

## Open work

### Model packages (parallel)

**Clean downstream prototype**
- Pick the smallest downstream model package / example that can use the
  generic interface without carrying an old engine.
- Implement only the package-owned pieces: response struct, `LikelihoodSpec`
  callbacks, layout extension for extra likelihood parameters, and R-side data
  builder.
- No adapter layer that forwards old package-native C++ entry points into tulpa
  shims. That keeps two engines alive and hardens the wrong boundary.
- Unblocks: strip `ModelData::legacy` (item below).

**Agent F — `tulpaGlmm` gap audit** (`~/Documents/dev/tulpaGlmm/REIMPLEMENTATION_PLAN.md`)
- Report-only. List what tulpa must expose for GLMM (Gaussian / Poisson /
  binomial `LikelihoodSpec` factories, dispersion handling, link functions).

### Tulpa core

**Strip `ModelData::legacy` (`LegacyRatioData`)** — blocked on one clean
generic downstream prototype.
- Delete `LegacyRatioData`; drop `n_processes == 0` branches throughout
  `hmc_sampler.cpp` (~413 references). Bump `TULPA_ABI_VERSION`.

**Wire SPDE into tulpaObs** — `occu_spde` ships in tulpaObs; end-to-end
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

## Design Note

The clean architecture is not "make `tulpaRatio` call exported tulpa shims".
That preserves `tulpaRatio`'s old API surface and keeps the ratio-specific
engine shape in play. The clean architecture is:

1. `tulpa` owns samplers, shared latent structures, parameter layout, mass
   matrices, diagnostics, and backend dispatch.
2. Model packages own likelihood semantics only: response data, per-observation
   log likelihoods, residuals / extra gradients, priors for extra parameters,
   and R builders that populate generic `ModelData`.
3. Every downstream model enters tulpa through `n_processes > 0` and
   `LikelihoodSpec`; no downstream package depends on `ModelData::legacy`, and
   no core sampler branches on ratio-specific fields.
4. Once a prototype runs end-to-end this way, remove `ModelData::legacy` in one
   core cleanup rather than continuing compatibility shims.
