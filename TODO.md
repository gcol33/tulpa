# tulpa — Scaling & Architecture TODO

Original audit: 2026-03-24. Status pass: 2026-04-28. Final pass: 2026-04-29.

> **2026-04-29 status pass.** Every item P0–P2 is now resolved; this file
> is kept as historical context. P2.1 (eta precompute) landed in
> `e484f1b`. P2.2 (priors & inference dispatch table) was refactored on
> 2026-04-29: per-distribution `format.tulpa_prior_<dist>()` methods
> replace the 7-branch if/else, `BACKEND_FAMILY_SUPPORT` registry replaces
> the hardcoded `gibbs_families` vector, and `ALL_BACKENDS` is derived
> from `INFERENCE_TIERS` instead of being typed out. P2.3 and P2.5 were
> already resolved as a side-effect of P0.1: `src/hmc_sampler.h:36-55`
> (and the equivalent header for `tulpa_zi`, `tulpa_temporal`, `tulpa_gp`,
> etc.) carries `using tulpa::*` aliases so every `tulpa_hmc::ModelData`
> reference resolves to the unified `tulpa::ModelData`, and
> `inst/include/tulpa/likelihood.h:9-10` uses `#include` of the actual
> autodiff headers instead of forward declarations. See `plan.md` for the
> live roadmap.
>
> **2026-04-28 status pass.** P0.1 / P0.2 / P1.1 / P1.2 / P2.4 all
> resolved. Section "P0 architecture blockers" retained below as
> historical context.

## P0 — Architecture Blockers ✅ RESOLVED

> Resolved 2026-04-28. The unified `tulpa::ModelData` / `ParamLayout` /
> enums live in `inst/include/tulpa/`. `src/hmc_sampler.h` lines 36-61
> alias them via `using tulpa::*;` inside `namespace tulpa_hmc`, so every
> `tulpa_hmc::ModelData` reference resolves to the unified type. Legacy
> ratio fields are nested under `ModelData::legacy` with no direct leaks.
> `compute_log_post_generic` (`log_post_impl.h:1852`) routes RE / spatial /
> temporal / SVC / TVC / latent / ST / ZI / OI through `priors::*` helpers
> in `src/tulpa_priors.h`.

### P0.1 Unify the dual type system ✅ DONE

**Problem:** Two incompatible `ModelData` structs exist — `tulpa::ModelData` (exported in `inst/include/tulpa/model_data.h`) and `tulpa_hmc::ModelData` (internal in `src/hmc_sampler.h`). Same for `ParamLayout`, `ZIType`, `SpatialType`, etc. Model packages would link against the exported types, but the engine operates on the internal types. They can never connect.

**Root cause:** We copied numdenom's C++ as-is (with its `tulpa_hmc` namespace), then separately wrote clean exported headers in `tulpa::` namespace. The two were never wired together.

**Fix — single source of truth:**

1. Make `inst/include/tulpa/model_data.h` the ONE definition of `ModelData`. Add all fields from `tulpa_hmc::ModelData` that are missing (the full RE structure, all spatial/temporal/ST/ZI/latent fields, prior hyperparameters). Delete the ratio-specific fields (`y_num`, `y_denom`, `X_num_flat`, `X_denom_flat`, `ModelType` enum) — these belong in numdenom, not tulpa.

2. Make `inst/include/tulpa/param_layout.h` the ONE definition of `ParamLayout`. Add all fields from `tulpa_hmc::ParamLayout`. Delete ratio-specific fields (`beta_num_start`, `beta_denom_start`, `log_phi_num_idx`, `log_phi_denom_idx`).

3. Make `inst/include/tulpa/types.h` the ONE definition of all enums (`SpatialType`, `TemporalType`, `ZIType`, `GradientMode`, `MassMatrixType`, `STType`). Delete the duplicate enums in `hmc_sampler.h`.

4. In `src/hmc_sampler.h`: remove the `ModelData` and `ParamLayout` struct definitions. Replace with `#include "tulpa/model_data.h"` and `#include "tulpa/param_layout.h"`. Add `using tulpa::ModelData; using tulpa::ParamLayout;` in the `tulpa_hmc` namespace so existing code compiles unchanged.

5. In every `src/*.h` and `src/*.cpp`: replace `tulpa_hmc::ModelData` → `tulpa::ModelData`, `tulpa_zi::ZIType` → `tulpa::ZIType`, etc. Or use `using` declarations in the `tulpa_hmc` namespace to alias them.

6. Verify: `devtools::load_all()` compiles. All 70 tests pass. The exported headers and internal code now operate on the same types.

**Files to touch:**
- `inst/include/tulpa/model_data.h` — expand to full field set
- `inst/include/tulpa/param_layout.h` — expand to full field set
- `inst/include/tulpa/types.h` — add all enums
- `src/hmc_sampler.h` — delete struct definitions, include exported headers
- `src/hmc_sampler.cpp` — update namespace references
- `src/log_post_impl.h` — update using declarations
- `src/tulpa_generic_sampler.cpp` — use `tulpa::` types directly
- All other `src/*.h`, `src/*.cpp` — namespace updates

**Risk:** High. Touches every C++ file. Do in a single focused session with compilation after each step.

### P0.2 Complete `compute_log_post_generic` — spatial/temporal/RE routing ✅ DONE

**Problem:** The generic log-posterior (`log_post_impl.h:1851-1962`) only handles fixed effects + ZI/OI. Spatial, temporal, RE, SVC, TVC, ST, and latent factor contributions are missing (line 1936 TODO). This means the generic interface can only fit intercept + covariate models — no mixed models, no spatial, no temporal.

**Root cause:** The legacy `compute_log_post_impl` has ~1800 lines of prior computation and linear predictor assembly, all hardcoded for two processes (eta_num, eta_denom). We need to extract these into reusable helpers that work with N processes.

**Fix — extract shared helpers, compose in generic function:**

1. **RE helper:** Extract the RE prior computation (lines ~50-400 of legacy `compute_log_post_impl`) into a standalone templated function:
   ```cpp
   template<typename T>
   T compute_re_prior_and_values(const std::vector<T>& params, const ModelData& data,
                                  const ParamLayout& layout, std::vector<T>& re_vals);
   ```
   This function computes RE priors (sigma priors, Cholesky correlation priors, non-centered transform) and populates `re_vals`. It does NOT touch eta — that's the caller's job.

2. **Spatial helper:** Extract spatial prior computation (ICAR quadform, BYM2 mixing, GP NNGP likelihood, HSGP basis eval + prior) into:
   ```cpp
   template<typename T>
   T compute_spatial_prior_and_effects(const std::vector<T>& params, const ModelData& data,
                                        const ParamLayout& layout, std::vector<T>& spatial_effects);
   ```

3. **Temporal helper:** Same pattern for RW1/RW2/AR1/GP temporal priors.

4. **SVC, TVC, ST, latent helpers:** Same pattern for each.

5. **Wire into `compute_log_post_generic`:** After each helper returns the prior contribution and effect values, route effects into the appropriate processes using `data.sharing`:
   ```cpp
   // After compute_re_prior_and_values fills re_vals:
   for (int k = 0; k < np; k++) {
       if (data.sharing.re[k]) {
           eta[k] = eta[k] + re_contrib_for_obs_i;
       }
   }
   ```

6. **Rewire legacy function:** Make `compute_log_post_impl` (the ratio-specific one) call the same helpers. This eliminates the copy-paste risk and ensures both paths stay in sync. The legacy function becomes: call helpers → route effects into eta_num/eta_denom → call ratio likelihood.

**Files to touch:**
- `src/log_post_impl.h` — extract helpers, rewrite both functions to use them
- New file: `src/tulpa_priors.h` — shared prior computation helpers
- New file: `src/tulpa_effects.h` — shared effect routing helpers

**Risk:** Very high. The legacy log_post is the most critical code in the engine. Extract helpers one at a time (RE first, then spatial, then temporal). After each extraction, verify all 70 tests still pass.

## P1 — Will Break on Real Data

### P1.1 Fix formula parser deparse-reparse round-trips ✅ DONE (2026-04-28)

> Done in commit `f1ca844`. AST-based parser end-to-end:
> response carried as language object (`parsed$response_expr`), nested
> groups expose structured `group_vars`, `||` expanded lme4-style,
> `offset()` extracted via `model.offset()`, slope formulas carry the
> formula's environment. Vestigial `terms = deparse(lhs)` removed.
> 12 new test cases pin the new behavior (60 → 98 expectations).

**Problem:** `R/formula.R` claims to be AST-based but 5 locations fall back to `deparse()` + regex. This will break on complex terms: `poly(x,2)`, backtick-quoted names, multi-line deparse output.

**Locations:**

1. `has_implicit_intercept()` (line 184-191): Deparses the LHS language object, then runs 3 regexes.
   **Fix:** Pattern-match on the AST directly:
   ```r
   has_implicit_intercept <- function(term) {
     # 0 + x → call("+", 0, x) — no intercept
     if (is.call(term) && length(term) == 3) {
       if (identical(term[[1]], as.name("+")) && is.numeric(term[[2]]) && term[[2]] == 0) return(FALSE)
       if (identical(term[[1]], as.name("-")) && is.numeric(term[[2]]) && term[[2]] == 1) return(FALSE)
     }
     # bare 0
     if (is.numeric(term) && term == 0) return(FALSE)
     TRUE
   }
   ```

2. `parse_bar_term()` line 135: `group_var <- deparse(rhs)` then `grepl("/", group_var)`.
   **Fix:** Check `is.call(rhs) && rhs[[1]] == as.name("/")`. Extract parts via `rhs[[2]]` and `rhs[[3]]`.

3. `parse_bar_term()` lines 146, 149: `strsplit(group_var, "/")`.
   **Fix:** Already have `rhs[[2]]` and `rhs[[3]]` from the AST after fix #2.

4. `tulpa_build_model_data()` lines 331-337: Six `gsub()` calls to strip intercept from slope text.
   **Fix:** In `parse_bar_term`, decompose the LHS into structured output: `list(has_intercept=TRUE, slope_terms=list(quote(x), quote(z)))`. Store the actual language objects, not deparsed text. Then in `tulpa_build_model_data`, build the slope formula from the stored language objects directly.

5. `tulpa_build_model_data()` line 340: `as.formula(paste("~", slope_text))`.
   **Fix:** Build from the language objects stored in fix #4: `slope_formula <- call("~", slope_call)`.

**Files to touch:**
- `R/formula.R` — rewrite `has_implicit_intercept`, `parse_bar_term`, slope handling in `tulpa_build_model_data`
- `tests/testthat/test-formula.R` — add tests for `poly(x,2)`, backtick-quoted names, `I(x^2)` terms

### P1.2 Fix `beta_zi_start` / `zi_beta_offset` field name mismatch ✅ DONE

**Problem:** `compute_log_post_generic` (`log_post_impl.h:1900`) uses `layout.beta_zi_start` and `layout.beta_oi_start` — field names from `tulpa_hmc::ParamLayout`. The exported `tulpa::ParamLayout` uses `zi_beta_offset` and `oi_beta_offset`.

**Fix:** After P0.1 unifies the types, this resolves automatically — there will be only one ParamLayout with one set of field names. If the unified layout uses `zi_beta_offset`, update the references in `compute_log_post_generic`. If it uses `beta_zi_start`, update the exported header. Pick one name and use it everywhere.

**Depends on:** P0.1

## P2 — Scaling Friction

### P2.1 Precompute eta = X * beta before observation loop ✅ DONE (2026-04-29)

> Hoisted into `eta_fixed[k]` per process before the observation loop in
> `compute_log_post_generic`. T=double dispatches to the OpenMP-parallel
> `tulpa_linalg::matvec`; autodiff types use a templated fallback with the
> same FLOP count but better X_flat cache locality than the original
> i-major path. Original problem framing exaggerated complexity ("cubic"
> — actually same total FLOPs); the real wins are vectorization on the
> double path and the cleaner separation of "fixed-effects assembly" from
> "effect routing" in the observation loop.

**Problem:** `compute_log_post_generic` (`log_post_impl.h:1928-1932`) computes `X[i,:] * beta` element-by-element inside the N-observation loop. For numerical gradients, this entire loop runs p+1 times, giving O(p * N * p_k) — cubic in dimensions.

**Fix:** Before the observation loop, compute all linear predictors at once via matrix-vector multiply:
```cpp
// Before observation loop:
std::vector<std::vector<double>> eta_all(np);
for (int k = 0; k < np; k++) {
    eta_all[k].resize(data.N, 0.0);
    // eta_all[k] = X_k * beta_k (single Eigen matvec or tulpa_linalg::matvec)
    tulpa_linalg::matvec(data.processes[k].X_flat.data(), &params[layout.process_beta_start[k]],
                          eta_all[k].data(), data.N, data.processes[k].p);
}
// In observation loop: eta[k] = eta_all[k][i] + spatial + temporal + RE + ...
```

**Files to touch:**
- `src/log_post_impl.h` — `compute_log_post_generic`

### P2.2 Replace hardcoded dispatch in priors and inference modes ✅ DONE (2026-04-29)

> Three sub-fixes landed in one pass:
> 1. `print_prior()` now dispatches via `format.tulpa_prior_<dist>()` S3
>    methods. Each `prior_*()` constructor adds a `tulpa_prior_<dist>`
>    subclass; adding a new distribution = one new `format.*` method.
>    Single source of truth: the constructor and its format method.
> 2. `gibbs_families` moved to `BACKEND_FAMILY_SUPPORT$gibbs` registry
>    near `INFERENCE_TIERS`. New helper `backend_supports_family(backend,
>    family)` handles the three family slots (`name`, `distribution`,
>    `numerator$distribution`). `auto_select_mode()` now calls the helper.
> 3. `all_backends` removed; `ALL_BACKENDS <- unlist(lapply(INFERENCE_TIERS,
>    "[[", "backends"), use.names = FALSE)` derives from the tier registry
>    at package load.
>
> Files: `R/priors.R`, `R/inference_modes.R`. test-priors (7) and
> test-inference-modes (2) still pass.


**Problem:**
- `R/priors.R` `print_prior()`: 7-branch if/else on distribution string. New distributions require new branches.
- `R/inference_modes.R` `auto_select_mode()`: Hardcoded `gibbs_families` vector with 13 family names. New families require updating the list.
- `R/inference_modes.R` `select_inference_mode()`: `all_backends` vector duplicates the backend lists in `INFERENCE_TIERS`.

**Fix:**
- `print_prior()`: Use S3 subclasses per distribution (`tulpa_prior_normal`, `tulpa_prior_pc`, etc.) with `format()` methods. Or: store a `format_fn` on the prior object at construction time.
- `auto_select_mode()`: Families should declare their backend compatibility (e.g., `family$supports_gibbs = TRUE`). The mode selector queries the family, not a hardcoded list.
- `all_backends`: Derive from `INFERENCE_TIERS` at package load time: `all_backends <- unlist(lapply(INFERENCE_TIERS, "[[", "backends"))`.

**Files to touch:**
- `R/priors.R` — refactor `print_prior()` to method dispatch or registry
- `R/inference_modes.R` — derive `all_backends` from tiers; move family compatibility to family objects

### P2.3 Unify namespace fragmentation ✅ DONE (resolved by P0.1)

> The first-line fix the TODO required is in place: every internal
> namespace (`tulpa_hmc`, `tulpa_zi`, `tulpa_temporal`, `tulpa_gp`, ...)
> opens with `using tulpa::TYPE;` declarations, so qualified names like
> `tulpa_hmc::ModelData` and `tulpa_temporal::TemporalType` resolve to
> the unified types in `inst/include/tulpa/`. Verified at
> `src/hmc_sampler.h:36-55`, `src/hmc_zi.h:16`, `src/hmc_temporal.h:24-26`,
> `src/hmc_gp.h:26-31`. The "long-term migration of internal code to
> drop the `tulpa_*` namespace prefixes everywhere" is a cosmetic style
> refactor explicitly outside this TODO's scope.


**Problem:** Three namespaces for the same concepts: `tulpa::` (exported), `tulpa_hmc::` (sampler), `tulpa_zi::` (zero-inflation), `tulpa_temporal::`, `tulpa_gp::`, etc. Model packages see `tulpa::ZIType` but internal code uses `tulpa_zi::ZIType`.

**Fix:** After P0.1, the authoritative types live in `tulpa::` (exported headers). Internal namespaces (`tulpa_hmc`, `tulpa_zi`, etc.) should `using tulpa::ZIType;` etc. at the top. Long-term: migrate internal code to use `tulpa::` directly.

**Depends on:** P0.1

### P2.4 Add `MAX_PROCESSES` runtime guard ✅ DONE

**Problem:** `compute_log_post_generic` uses `T eta[MAX_PROCESSES]` (stack array, MAX_PROCESSES=8) but never validates that `data.n_processes <= MAX_PROCESSES`.

**Fix:** Add a runtime check at the top of `compute_log_post_generic`:
```cpp
if (data.n_processes > MAX_PROCESSES) {
    Rcpp::stop("n_processes (%d) exceeds MAX_PROCESSES (%d)", data.n_processes, MAX_PROCESSES);
}
```

**Files to touch:**
- `src/log_post_impl.h`

### P2.5 Forward declarations in likelihood.h may not match real autodiff types ✅ DONE

> No longer applicable. `inst/include/tulpa/likelihood.h:9-10` includes
> the actual headers (`tulpa/autodiff_arena.h`, `tulpa/autodiff_fwd.h`)
> rather than forward-declaring the types, and the spec uses fully
> qualified `arena::Var` and `::fwd::Dual`. There is no fwd-decl drift
> to audit.


**Problem:** `inst/include/tulpa/likelihood.h` lines 12-13 forward-declare `namespace arena { struct Var; }` and `namespace fwd { struct Dual; }`. The actual types in the engine may live under different namespaces (e.g., `tulpa::arena::Var`).

**Fix:** After P0.1 namespace unification, verify the forward declarations match the actual type paths. Update if needed.

**Depends on:** P0.1, P2.3

## Execution Order

```
P0.1 (unify types)          ← FIRST, everything depends on this
  ├→ P1.2 (field name fix)  ← resolves automatically
  ├→ P2.3 (namespace unify) ← mostly done by P0.1
  └→ P2.5 (fwd decls)       ← verify after P0.1

P0.2 (complete generic log_post) ← SECOND, enables tulpaGlmm
  └→ P2.1 (eta precompute)  ← do during P0.2, not after

P1.1 (formula parser)       ← independent, do anytime

P2.2 (hardcoded dispatch)   ← independent, low urgency
P2.4 (MAX_PROCESSES guard)  ← 1 line, do anytime
```
