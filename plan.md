# tulpa Implementation Plan

Ecosystem layout (`~/Documents/dev/`):

| Package | Role | State |
|---|---|---|
| `tulpa` | Generic Bayesian engine | Current focus |
| `tulpaRatio` | Ratio/rate/proportion models | Still vendored numdenom; needs refactor to plug-in |
| `tulpaOcc` | Occupancy models | Plug-in |
| `tulpaGlmm` | GLMM via lme4/glmmTMB-style API | Scaffolded |
| `tulpaMesh` | CDT meshes for SPDE | Mature |

Model packages register likelihoods via `LikelihoodSpec` (`inst/include/tulpa/likelihood.h`).
tulpa stays model-agnostic.

This plan supersedes the standalone roadmaps that grew alongside it. See `TODO.md`
(architecture blockers), `TODO_next.md` (post-CHOLMOD features), `features_plan.md`
(nested Laplace + solver infra), and `nested_laplace_proposal.md` (math) for full
context — the items below pull every still-open thread into one ordered list.

## Round 1 — tulpa core (parallel, no file overlap) ✅ DONE

1. **Agent A** — `prior_predict` + `tulpa_simulate` generic via `LikelihoodSpec` dispatch. ✅ `06dd2dc`
2. **Agent B** — GPU backward solve completion in `gpu_batched_cholesky_solve`. ✅ `16f0e4a`
3. **Agent C** — Wire proper-CAR into sampler dispatch. ✅ `aaa5562`
4. **Agent D** — Wire CG/PCG NNGP into the NNGP path with config flag. ✅ `693a666`

## Round 1.5 — formula parser cleanup ✅ DONE

Done in this branch (still uncommitted):

- Response handled as a language object end-to-end (`response_expr`); no
  `deparse → parse(text=...)` round-trip. Backtick-quoted names and
  `cbind(succ, fail)` survive cleanly.
- Nested grouping `(1 | a/b)` carries a structured `group_vars = c("a","b")`
  on the parsed RE spec; `tulpa_build_model_data` reads the vector instead of
  `grepl(":", gvar)` + `strsplit`. Display label `"a:b"` retained.
- Vestigial `terms = deparse(lhs)` field dropped; `print` derives a display
  string from the stored `slope_terms` language objects on demand.
- `||` is expanded lme4-style: `(1 + x || g)` becomes `(1 | g) + (0 + x | g)`,
  one bar per coefficient, each correlated=TRUE. The sampler now sees a
  diagonal covariance as independent bars instead of a single bar with
  `correlated = FALSE` that nothing downstream actually honored.
- Slope formula is constructed with the original formula's environment, so
  user-scope helpers in slope expressions resolve correctly.
- `offset(...)` terms in the fixed RHS are extracted via `model.offset(mf)`
  and exposed as `built$offset` (numeric vector or `NULL`). Design matrix
  excludes them, as before.
- General grouping expressions (`(1 | factor(g))`) carried as
  `group_expr` (language) and evaluated against `data` at build time.
- Test coverage extended: `cbind(succ, fail)`, backtick responses, structured
  `group_vars`, missing-column errors, `||` expansion variants (with/without
  intercept, multi-slope), `offset()` extraction, formula-env slope helpers.

**Still open:** duplicate-bar coalescing — `(1|g) + (x|g)` produces two RE
specs on the same group. lme4 merges these into one bar. Low priority; the
sampler currently builds two RE blocks, which is slightly wasteful but not
incorrect for diagonal cov.

## Round 2 prerequisites — already resolved ✅

Audited 2026-04-28. The blockers `TODO.md` flagged in March are all gone:

- **P0.1 — Dual type system**: `inst/include/tulpa/{model_data,param_layout,types}.h`
  are the single source of truth. `src/hmc_sampler.h` lines 36-61 do
  `using tulpa::*;` inside `namespace tulpa_hmc`, so every `tulpa_hmc::ModelData`
  reference resolves to the unified `tulpa::ModelData`. Legacy ratio fields
  are nested under `ModelData::legacy` (`LegacyRatioData`); zero direct
  `data.y_num` access leaked outside the legacy gate.
- **P0.2 — `compute_log_post_generic`**: fully wired. `src/log_post_impl.h:1852-2200`
  routes RE / spatial (ICAR, BYM2, GP, multiscale-GP, HSGP) / temporal
  (RW1/RW2/AR1/GP/multiscale) / SVC / TVC / latent / ST / ZI / OI through
  `priors::*` helpers in `src/tulpa_priors.h` (1513 lines), with effect
  routing via `data.sharing.*`.
- **P1.2 — `beta_zi_start` vs `zi_beta_offset`**: canonical name
  `beta_zi_start` is used in both the exported header and every call site.
- **P2.4 — `MAX_PROCESSES` guard**: in place at `log_post_impl.h:1863`.

Implication: **Agent E (`tulpaRatio` refactor) is unblocked.** Model
packages already link the unified types.

## Round 2 — model packages (parallel)

5. **Agent E** — `tulpaRatio` refactor. `Package: numdenom` → `tulpaRatio` in
   DESCRIPTION, strip vendored `src/` engine files, depend on tulpa via
   `LinkingTo: tulpa`, register ratio likelihood through `LikelihoodSpec`.
   Verify one benchmark reproduces. Lives in `~/Documents/dev/numdenom`.
6. **Agent F** — Audit `tulpaGlmm/REIMPLEMENTATION_PLAN.md`. Report what
   tulpa must expose (Gaussian/Poisson/binomial `LikelihoodSpec` factories,
   dispersion handling, link functions). No implementation yet — gap list only.

## Round 3 — tulpa core (sequential, owns hmc_sampler.cpp)

Item 7 is blocked on Round 2 (Agent E) so `ModelData::legacy` can be removed
without breaking ratio models. The rest can start now.

7. Strip `ModelData::legacy` (`LegacyRatioData`) and remove
   `n_processes == 0` branches throughout `hmc_sampler.cpp` (~413 references).
   Bump `TULPA_ABI_VERSION`.
8. Laplace + NNGP backend in `dispatch_laplace_spatial` (`R/fit_laplace.R`,
   `src/laplace_core.cpp`).
9. Laplace + SPDE backend.
10. BYM2 H-mode analytical gradient (`src/hmc_sampler.cpp:5570`). Derive
    math first, then implement.
11. **`spatial_spde()` user-facing R API** (from `TODO_next.md`). Hard
    dependency on tulpaMesh. `spatial_spde(coords, boundary, max_edge,
    cutoff)` builds mesh + FEM matrices, stores them in a `tulpa_spatial`
    object. `spatial_spde_custom(C, G, A)` for fmesher/rSPDE users. Wire
    both into `cpp_laplace_fit_spde` / `cpp_nested_laplace_spde`.
12. **Wire SPDE into tulpaOcc** (from `TODO_next.md`).

## Round 4 — Nested Laplace (the strategic piece)

This is the differentiator: a CRAN-compatible INLA alternative with GPU
acceleration. Math is in `nested_laplace_proposal.md`; engineering plan in
`features_plan.md`.

13. Outer hyperparameter grid + numerical integration over θ.
14. Inner Laplace at each grid point, warm-started from the previous point.
15. Posterior marginal extraction (latent field + hyperparameters).
16. **Takahashi selected inversion** for partial inverse of sparse Cholesky
    factors — needed to get marginal variances out of the inner Laplace
    cheaply (from `TODO_next.md`).
17. **Rational SPDE** for fractional Matérn smoothness ν ∉ {0.5, 1.5, 2.5}
    (from `TODO_next.md`).
18. **BYM2 nested Laplace** (from `TODO_next.md`).
19. **GPU-batched NNGP** for the inner Laplace at scale (from `TODO_next.md`).

## P2 — Scaling friction (interleaved with Round 3 / 4)

From `TODO.md`. P2.4 is already done; the rest are still open:

- **P2.1** Precompute `eta = X * beta` before the observation loop.
- **P2.2** Replace hardcoded dispatch in priors and inference modes with a
  registry / dispatch table.
- **P2.3** Unify namespace fragmentation (`tulpa_hmc`, `tulpa_zi`) into
  `tulpa::`. Mostly cosmetic at this point — `tulpa_hmc::*` aliases already
  resolve to the unified `tulpa::*` types — but a mass-rename would remove
  the second-name confusion.
- **P2.5** Audit `inst/include/tulpa/likelihood.h` forward declarations
  against the real autodiff types.

## Deferred (need decisions)

- ST_IV NC parameterization (Kronecker spectral decomposition; math-heavy).
- CUDA wiring (`src/gpu_cuda.h`); Windows build risk, do last.
- OpenCL backend (currently pure stubs in `gpu_backend.h:200,220`).
- Generic S3 methods + diagnostics moved from tulpaOcc to tulpa
  (`coef`/`confint`/`vcov`/`logLik`/`summary`/`plot`/`tidy`/`glance`/`ranef`,
  plus `moranI`/`durbinWatson`/`variogram`/`compare_models`/`modelAverage`).
  See the TODO at the bottom of `CLAUDE.md`.
- EM+Laplace generic engine (callbacks for E-step + M-step encoding,
  MI / Gibbs corrections). Scaffolded in `R/em_laplace.R`; missing the
  full loop. See `CLAUDE.md` for the proposed API.
- `tests/testthat/test-inference-modes.R` placeholder.
- Formula parser duplicate-bar coalescing (low priority).

## Status

- Round 1 + Round 1.5: pushed to `origin/feature/lkj-chol-helpers`.
- Round 2 prerequisites: already resolved (audit 2026-04-28). See section above.
- Round 2 (Agent E + F): unblocked, parallelizable. Agent E is the long
  pole because it lives in a sibling repo and reproduces a benchmark.
- Round 3: items 8-12 can start in parallel with Round 2; only item 7
  (legacy strip) waits on Agent E.
- Round 4 (nested Laplace): the strategic differentiator. Can begin in
  parallel with later Round 3 items once SPDE backends are stable.
