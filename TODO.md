# tulpa — Open Work

Distilled from `plan.md` on 2026-04-29. The historical `TODO.md` (architecture
blockers) and `TODO_next.md` (post-CHOLMOD features) are fully resolved and have
been deleted; everything below is what's actually open.

## P0 — Active

### 1. ~~BYM2 H-mode analytical gradient~~ ✅ DONE (verified 2026-04-29)
`compute_laplace_gradient_bym2_H` at `src/hmc_icar_collapsed.h:1606` covers:
- direct sigma/rho traces against the dense 2S Hinv;
- indirect IFT corrections via cross-Hessians for both `log_sigma` and
  `logit_rho`;
- envelope-theorem residual scattering for beta/RE/dispersion;
- analytical priors (half-Cauchy on sigma, Uniform Jacobian on rho).
The dispatch in `compute_gradient_icar_collapsed` (`hmc_sampler.cpp:5892+`)
calls it with a numerical-Laplace fallback if `success=false`.

**Verification:**
- Standard BYM2 + HMC: `h_vs_n = 1.31e-08`.
- Collapsed BYM2 + HMC: `h_vs_n = 3.83e-08` (was blocked by a segfault in
  `compute_log_post_impl` for collapsed paths — fixed by guarding
  `params[spatial_start=-1]` accesses and redirecting AUTODIFF mode
  requests for collapsed to numerical, since autodiff cannot differentiate
  through the inner Newton mode-finding).
- Collapsed ICAR + HMC: `h_vs_n = 4.83e-09` (sanity check).
- `tools/bym2_gradient_check.R` is the verification harness.

The collapsed BYM2 + HMC path is still deprecated in the user-facing R API
(`R/spatial.R:391` — collapsed creates poor posterior geometry under HMC),
but the analytical gradient itself is now verified correct and the path is
ready if a Gibbs backend reactivates it.

### 2. ~~Round 4 — Nested Laplace strategic differentiator~~ ✅ DONE (2026-04-29)
**State:** generic outer-grid driver `tulpa::run_nested_laplace_grid` ships in
`src/nested_laplace_grid.h`; eight per-prior backends drive through it
(`src/nested_laplace.cpp`):

- Areal: `icar`, `bym2`, `car_proper` (2D over (τ, ρ) using D⁻¹W eigenvalue bounds).
- Continuous-spatial: `nngp` (2D over (σ², φ_gp); diagonal-on-w prior matching
  `laplace_mode_gp`), `hsgp` (2D over (σ², ℓ); fixed Φ basis, sqrt(S) reweighted
  per grid point — split into `src/hmc_hsgp_kernels.h` + `src/hmc_hsgp.h`).
- Temporal: `rw1`, `rw2`, `ar1`.
- SPDE: `cpp_nested_laplace_spde` (rebuilds Q via `SpdeQBuilder`, lives in
  `src/spde_laplace.cpp`).

Inner Newton warm-starts from the previous grid point (CHOLMOD symbolic factor
reused across the grid via `SparseCholeskySolver`). `R/nested_laplace.R`
provides the user-facing dispatcher accepting either a `prior=` list or
`spec=`/`data=` (tulpa_temporal / tulpa_spatial spec). For k≥3 hyperparameters,
`R/ccd_grid.R` provides `ccd_grid()` + `ccd_to_theta()` (full factorial up to
k=6, Resolution-V half-fraction for k≥7).

**Verification:** 254 pass / 1 skip / 0 fail across nested_laplace +
nested_laplace_cpp + nested_laplace_bym2 + nested_laplace_temporal +
nested_laplace_car_proper + nested_laplace_gp + ccd_grid test files. Regression
runner: `dev_notes/run_nl_regression.R`.

**Math reference:** `nested_laplace_proposal.md`. **Engineering:**
`features_plan.md`.

### 2b. Cross-DLL ABI surface (`*_api.h` shims) ✅ DONE (2026-04-30)
**State:** all shims committed. Full `R_RegisterCCallable` surface in
`src/tulpa_shims.cpp` + `src/tulpa_init.cpp`; public headers in
`inst/include/tulpa/`. Plan in `fix.md`.

**Shipped:**
- `laplace_api.h` — all 8 variants: `_dense`, `_spatial`, `_dense_multi_re`,
  `_bym2`, `_gp`, `_multiscale_gp`, `_multiscale_temporal`, `_rsr`.
  (`laplace_newton_solve` / `_sparse` dropped — templated on lambdas,
  cannot cross DLL boundary; callers use high-level shims instead.)
- `nested_laplace_api.h` — all 8 backends: icar / bym2 / car_proper /
  rw1 / rw2 / ar1 / nngp / hsgp.
- `spde_api.h` — SPDE Laplace + nested SPDE Laplace.
- `pg_api.h` — `pg_binomial_gibbs`, `pg_negbin_gibbs`, `pg_negbin_spatial_gibbs`.
- `vi_api.h` — `fit_vi` (mean-field / low-rank / full-rank by D).
- `ess_api.h` — `run_ess_sampler` with `joint_sigma_re` toggle.
- `sparse_solver_api.h` — opaque CHOLMOD handle + stochastic-Lanczos log-det.
- `sghmc_api.h` — `tulpa_sghmc_fit` + `tulpa_sgld_fit`.
- `mclmc_api.h` — MCLMC + MAMCLMC via ModelData entry (gcol33/tulpa#5).
- `smc_api.h` — SMC over ModelData with pluggable mutation (gcol33/tulpa#6);
  `compute_log_post` factored into `log_prior` + `log_lik_only` as prereq.
- `priors_capped.h` — capped-normal / capped-half-Cauchy; tulpaGlmm delegates here.

**Remaining open:**
- `compute_log_lik_only` currently derives by subtraction (2× `compute_log_post`)
  instead of a single O(N) observation-loop pass (`src/hmc_sampler.cpp:2362`,
  `TODO(#6 follow-up)`). Correct but ~2× slower than necessary for SMC weight
  evaluation.

**ABI policy:** `TULPA_ABI_VERSION` stays at 1 pre-release; downstream packages
rebuild against current source until first tagged release.

## P1 — Sibling-package work

### 3. Agent E — `tulpaRatio` refactor
**Where:** `~/Documents/dev/numdenom` (sibling repo).
**What:** rename `Package: numdenom` → `tulpaRatio` in DESCRIPTION; strip the
vendored `src/` engine; depend on tulpa via `LinkingTo: tulpa`; register the
ratio likelihood through `LikelihoodSpec`. Verify one benchmark reproduces.
**Unblocks:** stripping `ModelData::legacy` (item 7 below).

### 4. Agent F — `tulpaGlmm` gap audit
**Where:** `~/Documents/dev/tulpaGlmm/REIMPLEMENTATION_PLAN.md`.
**What:** report-only. List what tulpa must expose for GLMM (Gaussian /
Poisson / binomial `LikelihoodSpec` factories, dispersion handling, link
functions). No implementation.

### 5. Wire SPDE into tulpaOcc
**Where:** sibling `tulpaOcc`.
**State:** `occu_spde` already lives at `tulpaOcc R/spatial.R:197` and wraps
`tulpa::spatial_spde`. The end-to-end occupancy + continuous-spatial wiring
through tulpa's nested SPDE Laplace still needs verification at higher N and
the in-package vignette.

## P2 — Tulpa core (Round 3, can run in parallel with P1)

### 6. Strip `ModelData::legacy` (`LegacyRatioData`)
**Where:** `src/hmc_sampler.cpp` (~413 references) and adjacent headers.
**Blocked on:** item 3 (Agent E) — once `tulpaRatio` registers via
`LikelihoodSpec`, the `n_processes == 0` branch can be removed.
**Action:** delete `LegacyRatioData`, drop the `n_processes == 0` branches
throughout, bump `TULPA_ABI_VERSION`.

### 7. Live CUDA-kernel testing for GPU-batched NNGP
**State:** CPU-fallback path tested (3 tests in `test-gpu-nngp`). The cuSOLVER
batched-Cholesky path compiles but has not been run on a real GPU build.
**Action:** build with CUDA on (`launch_build.ps1` from RESOLVE), run the
NNGP path against the CPU reference, and pin a numerical-equivalence test.

## P3 — Cosmetic / scaffolded

### 8. P2.3 — Namespace cleanup
**State:** functionally resolved — `tulpa_hmc::*` etc. all alias to unified
`tulpa::*` types. A mass-rename to drop the `tulpa_*` namespace prefixes
inside `src/` would remove the second-name confusion. Pure cosmetic.

### 9. P2.5 — `likelihood.h` forward-decl audit
**State:** done in spirit (`inst/include/tulpa/likelihood.h:9-10` `#include`s
the actual autodiff headers, no forward decls). Audit step is just to
confirm no fwd-decl drift creeps back in during future edits.

### 10. EM+Laplace generic engine
**Where:** `R/em_laplace.R`.
**State:** EM loop ships and runs. Per-submodel `family` + `offset`
(gcol33/tulpa#3) and `m_step_extra` callback (gcol33/tulpa#4) are both
implemented. Only MI and Gibbs corrections remain stubbed —
`correction = "mi"/"gibbs"` raises a clear error today.
**Missing:** MI correction (draw hard z from weights, refit, Rubin's pool)
and Gibbs correction (warm-started z|θ → θ|z chain). See the TODO comment
at `R/em_laplace.R:241`.

### 11. ~~Generic S3 methods / diagnostics — move from tulpaOcc into tulpa~~
**State (2026-04-29 audit):** done.
- Methods live in `R/methods_generic.R`: `coef.tulpa_fit`,
  `confint.tulpa_fit`, `vcov.tulpa_fit`, `logLik.tulpa_fit`,
  `summary.tulpa_fit`, `plot.tulpa_fit`, `tidy.tulpa_fit`,
  `glance.tulpa_fit`, `ranef.tulpa_fit` (`tidy`/`glance`/`ranef` declared as
  generics in the same file).
- Diagnostics live in `R/diagnostics_generic.R`: `moranI`, `durbinWatson`,
  `variogram`, `compare_models`, `modelAverage`.
Model packages inherit via `class = c("model_fit", "tulpa_fit")`.

## P4 — Deferred (need decisions)

- **ST_IV NC parameterization** (Kronecker spectral decomposition; math-heavy).
- **CUDA wiring** (`src/gpu_cuda.h`); Windows build risk.
- **OpenCL backend** (currently pure stubs in `gpu_backend.h:200,220`).
- **`tests/testthat/test-inference-modes.R` placeholder** — fill out once the
  inference-mode dispatch refactor (P2.2) has a stable shape.
- **Formula parser duplicate-bar coalescing** — `(1|g) + (x|g)` produces two
  RE specs; lme4 merges them. Wasteful but not incorrect for diagonal cov.

## Recommended order

1. **#2b ABI shim surface (Laplace + nested + SPDE)** — highest tulpaGlmm
   value. Pattern is fixed; remaining work is mechanical. Commit the
   uncommitted block first, then add the missing `_bym2` / `_gp` / etc.
   entries, then `nested_laplace_api.h` + `spde_api.h`.
2. **#3 Agent E (tulpaRatio)** in the sibling repo, then **#6 strip legacy**.
3. **#4 Agent F (tulpaGlmm gap audit)** — schedulable anytime, report-only.
4. **#10 EM+Laplace** when tulpaGlmm needs it; ship #3/#4 follow-ups through
   `em_laplace_api.h` rather than as freestanding patches.
5. **#7 CUDA live test** — schedulable anytime; needs a GPU build.

(Items #1, #2 (Round 4 generic infra), and #11 were retired by the 2026-04-29
audits — all turned out to be implemented; only the deprecated-path BYM2
H-mode verification remains open under #1 and is low priority.)

`plan.md` remains the live narrative; this file is the punch list.
