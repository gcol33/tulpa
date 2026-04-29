# tulpa Implementation Plan

## Next-session pickup (read this first)

**Branch:** `feature/lkj-chol-helpers` (ahead of `origin/feature/lkj-chol-helpers` by 2 commits).
**Last engine commit:** `9cd5a89 feat(nested-laplace): NNGP and HSGP continuous-spatial backends` (2026-04-29).

### What just landed

- Round 4 nested-Laplace surface, all driven by `tulpa::run_nested_laplace_grid`:
  - Areal: `icar`, `bym2`, `car_proper` (2D grid over (τ, ρ) using eigenvalue bounds of D⁻¹W).
  - Continuous-spatial: `nngp` (2D over (σ², φ_gp)), `hsgp` (2D over (σ², ℓ)).
    HSGP is split into `src/hmc_hsgp_kernels.h` (Eigen-free spectral density + 1D
    eigenfunctions) and `src/hmc_hsgp.h` (Eigen-using matvec helpers); nested entries
    only include the kernels header to keep translation units lightweight.
  - Temporal: `rw1`, `rw2`, `ar1`.
  - SPDE: `cpp_nested_laplace_spde` (separate entry, rebuilds Q via `SpdeQBuilder`).
- `R/ccd_grid.R`: `ccd_grid(k, f_0)` and `ccd_to_theta(z, theta_hat, L, log_scale)` for
  k≥3 hyperparameter integration. Full factorial up to k=6, Resolution-V half-fraction
  for k≥7. 37 tests pass.
- `inst/include/tulpa/param_layout.h`: explicit half-open `[start, end)` convention
  documented after audit closing gcol33/tulpa#2 (the issue's premise that
  `re_end_multi` was inclusive was wrong; tulpa is uniformly exclusive everywhere).
- Cross-DLL ABI surface (see `fix.md`): `R_RegisterCCallable` shims so
  downstream packages (tulpaGlmm, tulpaOcc, tulpaMesh) can reach inference drivers
  without going through `Rcpp::List`. Already shipped: laplace_mode_dense /
  _spatial / _dense_multi_re / _bym2 / _gp / _multiscale_gp /
  _multiscale_temporal / _rsr; nested_laplace_icar / _bym2 / _car_proper /
  _rw1 / _rw2 / _ar1 / _nngp / _hsgp / _spde; pg_binomial / _negbin /
  _negbin_spatial; fit_vi; run_ess_sampler with `joint_sigma_re`;
  sparse_chol (create / analyze / factorize / solve / log_det / sel_inv_diag)
  + stochastic_log_det.

### Working-tree state at session start

```
M  R/nested_laplace.R          (NNGP/HSGP dispatch, in commit 9cd5a89)
M  src/nested_laplace.cpp      (NNGP/HSGP entries, in commit 9cd5a89)
M  src/hmc_hsgp.h              (kernels split-out, in commit 9cd5a89)
A  src/hmc_hsgp_kernels.h      (in commit 9cd5a89)

 M inst/include/tulpa/model_data.h   (uncommitted, ABI surface)
 M src/autodiff_utils.h               ( "    capped half-cauchy)
 M src/ess_sampler.h                  ( "    joint_sigma_re toggle)
 M src/tulpa_init.cpp                 ( "    register_shims hook)
?? src/tulpa_shims.cpp                (uncommitted, 835 LOC of shim impls)
?? inst/include/tulpa/laplace_api.h
?? inst/include/tulpa/pg_api.h
?? inst/include/tulpa/vi_api.h
?? inst/include/tulpa/ess_api.h
?? inst/include/tulpa/sparse_solver_api.h
?? inst/include/tulpa/priors_capped.h
?? fix.md                              (the change notes — read this)
```

Working tree is clean as of the nested-Laplace shim landing — pull and rebuild
to start.

### Open work, ordered (highest tulpaGlmm value first)

1. **EM+Laplace engine additions** for tulpaGlmm callbacks. File two follow-ups
   _against the future `em_laplace_api.h`_:
   - **gcol33/tulpa#3** — per-submodel family + offset on the `m_step_encode` callback's
     return shape; thread through existing `tulpa_laplace()` dispatch (already handles
     family per-call, just not per-submodel).
   - **gcol33/tulpa#4** — optional `m_step_extra(fits, weights, ...) -> fits` callback
     fired between M-step and next E-step, for non-η parameters (NB φ, Gamma shape,
     Beta φ, etc.). Pure plumbing.
2. **Stretch shim APIs:** `mclmc_api.h`, `smc_api.h`, `sghmc_api.h`. Same pattern as the
   already-landed Laplace shims.

### Pre-flight before starting

```bash
cd "C:/GillesC/Documents/dev/tulpa"
git status
# Inspect uncommitted shim block; commit if intent matches.

# Clean rebuild + full nested-Laplace + CCD regression (254 pass / 1 skip / 0 fail expected):
"/c/Program Files/R/R-4.6.0/bin/R.exe" CMD INSTALL --no-test-load --no-multiarch . > /tmp/install.log 2>&1
"/c/Program Files/R/R-4.6.0/bin/Rscript.exe" dev_notes/run_nl_regression.R
cat dev_notes/nl_regression.log
```

### Reference docs

- `fix.md` — full shim-surface plan with implementation order and salvaged-log notes.
- `CLAUDE.md` — the "Surface Bugs" rule, the EM+Laplace TODO at the bottom, and the
  generic-S3 / generic-diagnostics moves still to come.
- `dev_notes/run_nl_regression.R` — minimal regression runner; mirrors
  `dev_notes/run_gp_tests.R` + `nngp_smoke.R` + `hsgp_smoke.R`.
- `nested_laplace_proposal.md`, `features_plan.md`, `TODO.md` — historical context.

---

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
(punch list, `TODO_next.md` was rolled in and deleted 2026-04-29),
`features_plan.md` (nested Laplace + solver infra, annotated with shipped
status), `nested_laplace_proposal.md` (math), and `fix.md` (cross-DLL ABI shim
surface, the current active work) for full context — the items below pull
every still-open thread into one ordered list.

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
- `||` is preserved lme4-style as a single RE term carrying
  `correlated = FALSE`. `(1 + x || g)` parses to one spec with intercept +
  slope and `correlated = FALSE`; `(x || a/b)` to one spec per nesting
  level, each `correlated = FALSE`. Downstream packages branch on
  `correlated` to choose between independent-σ and Cholesky
  parameterizations (gcol33/tulpa#1, fixed 2026-04-29). Earlier drafts
  expanded `||` into independent `|` bars and lost the `correlated` flag —
  reverted.
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
8. **Laplace + NNGP backend** ✅ DONE (audit + dispatch wire-up).
   `cpp_laplace_fit_gp` already existed in `src/laplace_core.cpp`; what was
   missing was the route through the unified Laplace API.
   - `dispatch_laplace_spatial` now accepts `spatial$type == "gp"` and
     delegates to a new shared helper `laplace_gp_at()` (single source of
     truth — no duplication with `cpp_laplace_fit_gp` call site).
   - `gp_cov_type_for_laplace()` maps the spec's covariance to the
     Laplace kernel's integer convention (0 = exponential, 1 = Matern 1.5,
     2 = Matern 2.5). Gaussian / spherical / general-nu Matern raise a
     clear "use HMC instead" error rather than falling back silently.
   - The Hessian post-step is skipped for GP for the same reason as SPDE
     (eta = X*beta + w, not X*beta + Z*u).
   - GP + iid RE block in the same fit raises a clear error.
   - Unvalidated `spatial_gp()` spec (no `neighbor_info`) raises a clear
     error pointing at `validate_gp(spatial, data)`.
   - 5 new test cases in `tests/testthat/test-laplace-gp-dispatch.R`
     using a hand-built synthetic NNGP neighbor structure (no fmesher,
     no full validation chain).
9. **Laplace + SPDE backend** ✅ DONE — subsumed by item 11's wire-up
   (`57b606e`). Single-point SPDE Laplace works through `laplace_spde_at()`;
   `fit_spde()` and `dispatch_laplace_spatial` agree to 1e-10. The
   nested-grid SPDE Laplace path (already exposed via `cpp_nested_laplace_spde`)
   belongs to Round 4 item 13–18.
10. **BYM2 H-mode analytical gradient** ✅ DONE (audited 2026-04-29).
    `compute_laplace_gradient_bym2_H` (`src/hmc_icar_collapsed.h:1606`)
    implements direct 2S traces + IFT cross-Hessians for both `log_sigma`
    and `logit_rho`; the dispatch in `compute_gradient_icar_collapsed`
    (`hmc_sampler.cpp:5892+`) calls it with envelope-theorem residual
    scattering and a numerical fallback. Previous "BYM2: numerical fallback"
    comments were stale and have been corrected. Standard BYM2 + HMC
    gradient verified at `h_vs_n = 1.31e-08` via
    `tools/bym2_gradient_check.R`. Verification on the *collapsed* path is
    blocked by an unrelated segfault in the setup harness that hits both
    ICAR and BYM2 collapsed; reopened only if the collapsed + HMC path
    (deprecated in `R/spatial.R:391`) is revived.
11. **`spatial_spde()` user-facing R API** ✅ DONE (audit + dispatch wire-up
    on this branch). The constructor (`spatial_spde()` / `spatial_spde_custom()`)
    and the standalone `fit_spde()` were already in place; what was missing
    was the routing through the unified `tulpa_laplace()` API.
    - `dispatch_laplace_spatial` now accepts `spatial$type == "spde"` and
      delegates to a new shared helper `laplace_spde_at()` that both
      `fit_spde`'s single-point branch and the dispatcher call (single
      source of truth, per the engineering principles).
    - `tulpa_laplace()` skips the fixed-effect Hessian post-step for SPDE
      because eta = X*beta + A*w (mesh-projected), not X*beta + Z*u —
      proper SPDE Hessian / uncertainty propagation is a separate item.
    - SPDE + iid RE block in the same fit raises a clear error (use HMC).
    - Dispatch contract pinned by 4 new test cases in `tests/testthat/test-spde.R`
      using a tiny synthetic mesh via `spatial_spde_custom` (no fmesher dep
      for the dispatch tests).
12. **Wire SPDE into tulpaOcc** — `occu_spde` ships in tulpaOcc; end-to-end
    verification at higher N pending. (Tracked as TODO.md #5.)

## Round 4 — Nested Laplace (the strategic piece) ✅ DONE

This is the differentiator: a CRAN-compatible INLA alternative with GPU
acceleration. Math is in `nested_laplace_proposal.md`; engineering plan in
`features_plan.md` (now annotated with shipped status).

13. ✅ Outer hyperparameter grid + numerical integration over θ —
    `tulpa::run_nested_laplace_grid` in `src/nested_laplace_grid.h`.
14. ✅ Inner Laplace at each grid point, warm-started — every `cpp_nested_laplace_*`
    threads `prev_mode` through the driver, and `SparseCholeskySolver` reuses
    its symbolic factor across the grid.
15. ✅ Posterior marginal extraction — `R/nested_laplace.R` returns
    `theta_grid` + `weights` + `theta_mean` + `theta_sd` + per-grid-point modes.
16. ✅ Takahashi selected inversion — `SparseCholeskySolver::selected_inversion_diagonal`
    + cross-DLL shim `tulpa_sparse_chol_sel_inv_diag`.
17. ✅ Rational SPDE — `SpdeQBuilder::rebuild_rational(kappa, tau, poles, weights)`
    in `src/spde_qbuilder.h`.
18. ✅ BYM2 nested Laplace — `cpp_nested_laplace_bym2`.
19. ✅ GPU-batched NNGP at scale — `tulpa::batch_nngp_scatter` in
    `src/gpu_nngp_laplace.h`. CPU path tested; live CUDA test still pending
    (TODO.md #7).

**Plus shipped beyond the original list:**

- `cpp_nested_laplace_car_proper` (proper CAR, 2D over (τ, ρ) using the
  D⁻¹W eigenvalue interval).
- `cpp_nested_laplace_nngp` (continuous-spatial GP, 2D over (σ², φ_gp)).
- `cpp_nested_laplace_hsgp` (Hilbert-space GP, 2D over (σ², ℓ); kernels
  split out into `src/hmc_hsgp_kernels.h` for Eigen-free use).
- `cpp_nested_laplace_rw1` / `_rw2` / `_ar1`.
- `R/ccd_grid.R` — k≥3 hyperparameter integration via central composite design.

Regression: 254 pass / 1 skip / 0 fail across the full nested-Laplace + CCD
test surface (`dev_notes/run_nl_regression.R`).

## P2 — Scaling friction (interleaved with Round 3 / 4)

From `TODO.md`. P2.4 is already done; the rest are still open:

- **P2.1** Precompute `eta = X * beta` before the observation loop ✅ DONE.
  Hoisted into `eta_fixed[k]` per process. T=double dispatches to the
  OpenMP-parallel `tulpa_linalg::matvec`; autodiff types use a templated
  fallback. 201/201 targeted tests pass.
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

## Status (2026-04-29)

- Round 1 + Round 1.5: pushed to `origin/feature/lkj-chol-helpers`.
- Round 2 prerequisites: resolved (2026-04-28 audit).
- Round 2 (Agent E + F): unblocked, parallelizable. Agent E is the long
  pole because it lives in a sibling repo and reproduces a benchmark.
- Round 3: items 8-12 can start in parallel with Round 2; only item 7
  (legacy strip) waits on Agent E.
- Round 4 (nested Laplace): ✅ shipped. Generic infra + 8 backends + CCD.
- **Active focus:** cross-DLL ABI shim surface (`fix.md`). Laplace surface
  (dense / spatial / dense_multi_re / bym2 / gp / multiscale_gp /
  multiscale_temporal / rsr) and nested-Laplace surface (icar / bym2 /
  car_proper / rw1 / rw2 / ar1 / nngp / hsgp / spde) are now complete.
  Remaining work is the EM+Laplace follow-ups (gcol33/tulpa#3, #4) and the
  stretch MCLMC / SMC / SGHMC shim layer. Tracked as item 2b in `TODO.md`.
