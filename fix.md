# Fix: Export internal inference drivers via R_RegisterCCallable

**Date:** 2026-04-29
**Source:** tulpaGlmm Phase 0 audit (local note, not a GitHub issue)
**Blocks:** tulpaGlmm Phase 1 (Laplace, nested-Laplace, SPDE, VI, PG-Gibbs, ESS modes)

## The problem

`src/tulpa_init.cpp` registers only three C-callable symbols:

```cpp
R_RegisterCCallable("tulpa", "tulpa_run_nuts_generic",     ...);
R_RegisterCCallable("tulpa", "tulpa_compute_param_layout", ...);
R_RegisterCCallable("tulpa", "tulpa_get_abi_version",      ...);
```

Every other inference driver tulpa already implements internally is unreachable from
downstream packages, because the existing entry points return `Rcpp::List` /
`Rcpp::NumericMatrix` — not ABI-stable across separately-compiled DLLs.

## What already exists internally

| Driver | Source | Used by old tulpaGlmm? |
|---|---|---|
| Laplace (8 variants) | `src/laplace_core.cpp`, `src/laplace_newton.h`, `src/sparse_hessian.h` | yes (warm-start path) |
| Nested Laplace | `src/nested_laplace.cpp` + `src/implicit_diff.cpp` | yes (Tier 2) |
| SPDE Laplace | `src/spde_laplace.cpp` + `src/spde_qbuilder.h` | yes (spatial) |
| VI (mean-field / low-rank / full-rank) | `src/vi_sampler.cpp` (`fit_vi`) + `src/vi_*.h` | scaffolding present |
| PG-Gibbs (binomial / negbin / spatial) | `src/pg_binomial.cpp`, `src/pg_negbin.cpp`, `src/pg_spatial.cpp` | yes (1.9–3.7× HMC) |
| ESS sampler | `src/ess_sampler.cpp` | yes (LKJ + ASIS slopes) |
| MCLMC / SMC / SGHMC / gibbs_spatial | `src/mclmc.cpp`, `src/smc_sampler.cpp`, `src/sghmc_sampler.cpp`, `src/gibbs_spatial.cpp` | guarded |
| Sparse Cholesky (CHOLMOD), sparse Hessian, stochastic log-det | `src/sparse_cholesky.cpp`, `src/sparse_hessian.h`, `src/stochastic_logdet.cpp` | yes |
| 12 spatial / temporal / SVC / TVC / ZI kernels | `src/hmc_*.h` | yes |

Evidence in tulpaGlmm's salvage inventory:
`~/Documents/dev/tulpaGlmm/dev_notes/salvage_inventory.md` and 33 logs at
`~/Documents/dev/tulpaGlmm/dev_notes/salvaged_logs/`.

## The fix

For each driver, add a thin C-shim wrapper following the existing
`tulpa_run_nuts_generic_impl` pattern (raw pointers + POD result struct + manual
buffer management with `free_buffers()` cleanup). Register each shim in
`tulpa_register_callables()`. Add a matching public header in
`inst/include/tulpa/<driver>_api.h` that exposes a typed function pointer plus a
`get_<driver>_fn()` resolver via `R_GetCCallable`.

### New public headers (mirrors `inst/include/tulpa/nuts_api.h`)

| Header | Drivers exported |
|---|---|
| `inst/include/tulpa/laplace_api.h` | `laplace_mode`, `laplace_mode_dense_multi_re`, `laplace_mode_spatial`, `laplace_mode_bym2`, `laplace_mode_gp`, `laplace_mode_multiscale_gp`, `laplace_mode_multiscale_temporal`, `laplace_mode_rsr`, `laplace_newton_solve`, `laplace_newton_solve_sparse` |
| `inst/include/tulpa/nested_laplace_api.h` | nested-Laplace driver + implicit-diff hook |
| `inst/include/tulpa/spde_api.h` | Q-builder + `laplace_mode_spatial` for tulpaMesh |
| `inst/include/tulpa/vi_api.h` | `fit_vi` (auto-dispatches mean-field / low-rank / full-rank by D); expose enum override |
| `inst/include/tulpa/pg_api.h` | `pg_binomial_gibbs`, `pg_negbin_gibbs`, `pg_spatial_gibbs` |
| `inst/include/tulpa/ess_api.h` | `run_ess_sampler` with joint-sampling toggle |
| `inst/include/tulpa/mclmc_api.h` | MCLMC entry (stretch) |
| `inst/include/tulpa/smc_api.h` | SMC entry (stretch) |
| `inst/include/tulpa/sparse_solver_api.h` | CHOLMOD factor / solve / log-det, sparse Hessian, stochastic log-det |

### Pattern to copy

`src/tulpa_init.cpp:19-74` (NUTS shim) and `inst/include/tulpa/nuts_api.h:67-90`
(typedef + resolver). Same shape for each new driver — the only thing that varies is
the argument list and the result struct.

### ABI bump

Pre-release: `TULPA_ABI_VERSION` stays at 1 and is not bumped per change.
Will start tracking once the package has its first tagged release; until
then, rebuild downstream packages against the current tulpa source.

## Implementation order (highest-value first)

1. `laplace_api.h` — fast point estimates + warm-start. Highest tulpaGlmm value.
   - DONE: `laplace_mode_dense`, `laplace_mode_spatial` (2026-04-29 first pass).
   - DONE: `laplace_mode_dense_multi_re` (2026-04-29 follow-up). Round-trip
     against `cpp_laplace_fit_multi_re` matches mode bit-for-bit on a 200-obs
     binomial GLMM with two intercept-only RE blocks.
   - DONE: `_bym2`, `_gp`, `_multiscale_gp`, `_multiscale_temporal`, `_rsr`
     (2026-04-29). Header pattern documented in `laplace_api.h`; impls in
     `tulpa_shims.cpp` follow the same pack-into-Rcpp-then-delegate shape
     as the dense / spatial entries. Coords for `_gp` / `_multiscale_gp`
     are passed flat with an explicit `coord_dim` (currently must be 2 —
     the underlying NNGP kernel only consumes the first two columns).
   - DROPPED: `laplace_newton_solve`, `laplace_newton_solve_sparse` — both
     are templated on the lambdas (compute_eta, scatter, center, log_prior)
     and so cannot pass through `R_RegisterCCallable`. Downstream packages
     should call the high-level `laplace_mode_*` shims instead, or build a
     dedicated entry that hides the lambdas.
2. `pg_api.h` — proven 1.9–3.7× faster than HMC on binomial / negbin in salvaged logs. DONE.
3. `vi_api.h` — single shim, three variants behind it. DONE.
4. `sparse_solver_api.h` — required by SPDE work.
   - DONE: opaque CHOLMOD handle (create / analyze / factorize / solve /
     log_det / sel_inv_diag) + stochastic Lanczos log-det one-shot
     (2026-04-29). Cross-DLL round-trip verified: dense reference matches to
     1e-15 on n=5 tridiagonal SPD; SLQ matches dense logdet to 6 sig figs at
     n_probes=60, n_lanczos=5.
5. `spde_api.h` + `nested_laplace_api.h` — unlocks `spatial = tulpa_mesh(...)`. DONE
   (2026-04-29). `nested_laplace_api.h` exposes the eight backends
   (icar / bym2 / car_proper / rw1 / rw2 / ar1 / nngp / hsgp) behind a shared
   `NestedLaplaceShimResult` (universal `log_marginal[n_grid] + n_iter[n_grid]`
   block, optional `modes[n_grid * n_x]` row-major). `spde_api.h` exposes the
   SPDE driver behind `SpdeNestedLaplaceShimResult`, which adds `Q_nnz` and
   omits modes (the SPDE backend never stored them).
   Hyperparameter posterior weights / `theta_mean` / `theta_sd` are computed
   R-side from `log_marginal` + the input grid (see `R/nested_laplace.R`).
   Shim impls in `tulpa_shims.cpp` go through the existing Rcpp wrappers
   (`cpp_nested_laplace_*`, `cpp_nested_laplace_spde`) and unpack the
   returned `Rcpp::List` into the POD result struct — same shape as the
   already-landed Laplace shims that go through `tulpa::laplace_mode_*`.
6. `ess_api.h` — correlated slopes via LKJ-Cholesky / ASIS. DONE.
   - DONE: `joint_sigma_re` toggle now wired through `tulpa_ess::ESSConfig`
     and run_ess_sampler. When set, performs a joint Metropolis move
     (log_sigma_re' = log_sigma_re + delta, re' = re * exp(delta))
     after the per-iteration ESS + RWMH passes, with Jacobian
     n_re * delta. Diagonal RE only (legacy single-term + multi-term);
     correlated slopes still need ASIS.
7. `mclmc_api.h` / `smc_api.h` / `sghmc_api.h` — stretch. DONE (2026-04-30).
   - DONE: `sghmc_api.h` exposes both `tulpa_sghmc_fit` and `tulpa_sgld_fit`
     behind the NUTS-style `ModelData` + `ParamLayout` interface (2026-04-29).
   - DONE: `mclmc_api.h` — MCLMC + MAMCLMC wired to a ModelData entry point;
     `log_prob_grad` built from `compute_log_post_double` + autodiff, mirroring
     NUTS (gcol33/tulpa#5, committed c9d38a7).
   - DONE: `smc_api.h` — SMC over ModelData with pluggable mutation kernel;
     `compute_log_post` factored into `log_prior` + `log_lik_only` as prereq
     so SMC particle weights use the likelihood alone (gcol33/tulpa#6,
     committed 8da9b76 + 21c5c06 + 590817e/148645f).
   - Remaining: `compute_log_lik_only` currently derives by subtraction
     (2× `compute_log_post`). A single O(N) pass would halve the cost per
     SMC weight evaluation. See `src/hmc_sampler.cpp:2362`.

Estimated total: ~600 LOC of shims in `src/tulpa_init.cpp` + ~600 LOC across nine new
public headers + ABI version bumps.

## Notes from salvaged logs (relevant numerical lessons)

- Bad seed for NB HMC: `1821108667` (610 s slowdown without capped-SD prior).
  DONE: regression test lives in
  `~/Documents/dev/tulpaGlmm/tests/testthat/test-family-negbin.R:32-52`
  ("nb2 sampler does not stall on the bad-seed regression case"). Asserts
  the run completes in < 60 s with no divergences and `|log_phi| < 6`
  (i.e. capped prior held). Re-verified passing 2026-04-29 after
  `capped_normal_log_prior` was centralized into `tulpa::priors`.
- ESS+RWMH was disabled for Poisson with random effects because `log_sigma_re`
  and `z` require joint sampling — DONE: `joint_sigma_re` toggle in
  `ess_api.h` is now wired through `ESSConfig` and `run_ess_sampler` (see
  item 6 above).
- Order-2 analytical gradient is the default; order-3 was dramatically slower
  (`bench_grad_order.log`: 0.2 ESS/sec vs 12.9 at n=100 NB).
- Capped-SD prior on `log φ` (NB) and on RE σ (binomial) materially improved
  high-variance seeds. DONE: prior helper centralized in the public header
  `inst/include/tulpa/priors_capped.h` as
  `tulpa::priors::log_prior_capped_normal(x, mean, sd, cap, cap_weight)`.
  Templated for arena::Var / fwd::Dual / double.
  Quartic soft cap on `|x - mean| > cap` (the original recipe from
  `build_tulpa_capped_sd.log`). The previous in-tree duplicates
  (`tulpaGlmm::capped_normal_log_prior` in `glmm_extra_prior.h`) now
  delegate to the tulpa helper — `glmm_extra_prior.h` is a thin
  back-compat alias.
- Half-Cauchy variant for σ caps lives in `src/autodiff_utils.h::tulpa::math`
  as `log_prior_capped_half_cauchy(log_sigma, scale, sigma_max)`. Pass
  `sigma_max <= 0` to disable the hard upper bound (then identical to
  `log_prior_half_cauchy`).

---

## 2026-05-06 — `tulpa_run_ess_sampler` segfaults on the generic LikelihoodSpec layout — **RESOLVED 2026-05-13**

**Resolution.** The two root causes named in the original entry below
are both fixed in tulpa, verified by re-running the
`dev_notes/smoke_day22_minimal.R` reproducer plus a direct
`cpp_glmm_ess_fit` probe from tulpaGlmm.

- `src/log_post_impl.h` was simplified to a thin generic-layout
  dispatcher. The unconditional `&params[layout.legacy.beta_num_start]`
  reads at the old lines 83–84 are gone. `compute_log_post_impl<T>`
  now early-returns `T(0)` whenever
  `data.n_processes == 0 || data.likelihood_spec == nullptr`, and
  forwards to `compute_log_post_generic_spec_double` for `double`.
- `src/ess_sampler.h::build_gaussian_priors` now walks
  `layout.process_beta_start[k]..+process_beta_count[k]` for the β
  block (lines ~223–240). Generic-layout β enters the ESS sampling
  step correctly.

**Downstream.** tulpaGlmm `R/ess.R::fit_ess` calls
`cpp_glmm_ess_fit` directly; live regression suite in
`tests/testthat/test-inference-ess.R` (15 assertions, all passing).
The "blocked on upstream" stop in `R/ess.R` is removed. PLAN.md §5.5
marked shipped.

**Known follow-on (not blocking).** ESS can emit
`"max shrinks without accepting"` warnings when the initial
`log_sigma_*` is far from the posterior mode and the bracket
collapses. Symptom is per-iter warnings, not divergence or wrong
posterior. Candidate fix: a short ML-II pre-pass on `log_sigma_*`
before handing off.

---
### Original entry (for context)


**Surfaced by:** tulpaGlmm Day-22 (PG-Gibbs and VI shipped on Day-20/21;
ESS was the next entry on PLAN.md §5.5).

**Symptom.** Calling `tulpa::get_ess_fn()(...)` with a `ModelData` /
`ParamLayout` produced by tulpaGlmm (i.e. `n_processes = 1`,
`processes[0]` populated, `likelihood_spec` set, every `legacy.*`
field at sentinel `-1`) immediately segfaults during the first
`compute_log_post_double` call.

**Root cause.** `src/log_post_impl.h:83-84`:

```cpp
const T* beta_num   = &params[layout.legacy.beta_num_start];
const T* beta_denom = &params[layout.legacy.beta_denom_start];
```

These dereferences are unconditional. For the generic layout
`legacy.beta_num_start == -1`, so the access is `&params[-1]` →
out-of-bounds read. NUTS, VI, and the Laplace shims all reach the
same `compute_log_post_impl<T>` but tulpaGlmm has been working with
them because those callers happen to provide enough legacy padding
(or the compiler inlines around the OOB read). ESS is the first path
that segfaults reproducibly.

**Reproducer.** `~/Documents/dev/tulpaGlmm/dev_notes/smoke_day22_minimal.R`
— FE-only Gaussian, N=100, no RE, no ZI, no spatial. Crashes on the
first `cpp_glmm_ess_fit` call with exit code 139.

**Fix.** Either (a) gate the legacy `beta_num` / `beta_denom`
dereferences behind `if (layout.legacy.beta_num_start >= 0)` and
provide a generic-layout branch in `compute_log_post_impl<T>`, or
(b) write an ESS-specific `compute_log_post_double_generic` that
mirrors the NUTS / VI computation but does not touch `legacy.*`. The
NUTS path's `tulpa_run_nuts_generic_impl` already does the right
thing for the generic layout — that code can be used as the spec.

The internal ESS sampler also assembles its β Gaussian-prior block
from `legacy.beta_num_start..end` only
(`src/ess_sampler.h::build_gaussian_priors`). For the generic layout,
the β block is at `layout.process_beta_start[0]..start +
process_beta_count[0]`. A working generic ESS shim needs to walk that
range to add β to the Gaussian-prior list (otherwise β never gets
sampled, only RE / log_sigma_re / extra do).

**Downstream impact.** Blocks tulpaGlmm Day-22 (`inference = "ess"`).
The shim and R driver are written
(`tulpaGlmm/src/glmm_ess.cpp`, `tulpaGlmm/R/ess.R`); they will work
verbatim once `tulpa::compute_log_post_impl<T>` is generic-layout-
safe.

**Status.** Pending in tulpa. Day-22 in tulpaGlmm now surfaces a
clear "blocked on tulpa fix.md" error rather than calling the
crashing shim.
