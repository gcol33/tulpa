# Clean Migration — API surface + likelihood-boundary unification

Living plan for the multi-session refactor that removes tulpa's API-design
leaks and unifies the likelihood boundary across inference tiers. **Read this
first** when picking the work back up. Update the status table as phases land.

Started 2026-05-24. Driven on `main` (no feature branches — see constraints).

---

## 1. Goal

Three things, in the author's words: "fix all these issues and fix the leaking
of bad design", "unify the boundary", "full solver unification".

Concretely:
1. **One user front door.** `tulpa()` should reach the nested-Laplace
   (hyperparameter-integration) layer, not just the conditional fitters.
   Today there are two disconnected R-API surfaces.
2. **Clean engine signatures.** Collapse the ~25 perf/numerical knobs on the
   nested engines into `control = list()`, matching `tulpa()`.
3. **No back-compat shims.** Pre-release: rename or hard-error, never deprecate
   (per global CLAUDE.md). Remove the dead alias + `tulpa_priors_legacy`.
4. **Keystone — unify the likelihood boundary across tiers.** The Laplace /
   nested-Laplace tier must consume a model-supplied `LikelihoodSpec` like the
   sampler tiers already do, instead of a hardcoded family switch. This moves
   occupancy's `det_prob` out of tulpa's generic kernel and into tulpaObs.
   **Decision: full solver unification** (see §4) — one spec-driven inner Newton
   that also handles GMRF latent blocks; retire the family-enum inner loop.

### Scope decisions already made (do not re-litigate)
- **Full refactor + update siblings** in the same effort (tulpaObs primarily;
  tulpaRatio is C++/LikelihoodSpec-only at the ABI level; tulpaGlmm/tulpaMesh
  minor).
- **Unify the likelihood boundary** (yes).
- **Full solver unification** (the maximal path), not the surgical
  likelihood-only fix.

---

## 2. Status

| Phase | What | Status |
|---|---|---|
| 0 | Baseline load_all + blast-radius map | ✅ done |
| 1 | Remove dead `nested_laplace()` alias; ASCII-only `ccd_grid` roxygen | ✅ done — commit `175fa62` |
| L | **Keystone:** full solver unification (spec-driven Laplace handles GMRF blocks; det_prob → tulpaObs) | 🔄 in progress — **L1 done** (`228ee05`); **L2 done** (`9ece117` ICAR + bym2): spec solver carries INDEXED_SINGLE blocks, verified vs the nested kernel across families + RE; **L3 next** (route the real nested entries through it) |
| 2 | Collapse nested engine knobs into `control = list()` | ⬜ pending |
| 3 | Remove `tulpa_priors_legacy` | ⬜ pending |
| 4 | Route single-arm `latent()` formulas through `tulpa()`; register nested backends | ⬜ pending |
| 5 | Split `nested_laplace_joint.R` (2625 lines) | ⬜ pending |
| 6 | Update siblings (tulpaObs et al.); final `devtools::check` | ⬜ pending |

### Recommended sequencing for a fresh session
The keystone (L) is foundational — Phase 4 (unification) routes through the
nested kernel and Phase 6 (siblings) depend on the final likelihood contract.
Phases 2 and 3 are R-only and independent of L — good low-risk warm-ups. Phase
5 is cosmetic; do it after 2 (which edits the same joint file). Suggested order:
**L → 2 → 3 → 4 → 5 → 6**, but 2/3 can be done first if you want momentum
before the big C++ change.

---

## 3. Architecture findings (why the keystone is what it is)

There are **two inner-Newton implementations** today:

- **Spec-driven, general** — `src/laplace_spec.cpp::laplace_mode_spec_dense_impl`.
  Full inner Newton (multi-process, multi-term RE, random slopes, correlated
  REs via tanh-Cholesky, sparse/dense Cholesky, log-marginal). Likelihood comes
  entirely from `LikelihoodSpec.eta_weights_fn` (per-obs eta-space grad +
  neg-Hessian) and `LikelihoodSpec.ll_double`. tulpaRatio/tulpaObs already use
  this for *conditional* Laplace. **It does NOT handle GMRF latent blocks** —
  only fixed effects + iid/slope RE terms (`log_prior_latent`,
  `build_term_precision` are RE-only).
- **Family-enum, hardcoded** — the nested-Laplace outer-grid kernels
  (`src/nested_laplace_multi.h`, `src/nested_laplace.cpp`,
  `src/nested_laplace_joint_multi.h`). Per grid cell they run their own inner
  Newton over latent = [beta, RE, GMRF-block fields], assembling the GMRF block
  precision `Q(theta)` into the Hessian. The per-observation likelihood is the
  fixed family switch in `src/laplace_family_link.h`
  (`grad_hess_for_family(..., det_prob)` / `log_lik_for_family`), with
  `det_prob` as the occupancy hook.

**The leak:** the nested kernel duplicates the inner Newton AND carries the
likelihood as a family enum + occupancy-specific `det_prob`. The sampler tiers
(NUTS/ESS/VI) already accept a model `LikelihoodSpec`; the Laplace tier's
*conditional* path does too — but the *nested* path bypasses it.

**`LikelihoodSpec` contract** (`inst/include/tulpa/likelihood.h`): append-only
struct. The relevant member for Laplace is
`EtaWeightsFn eta_weights_fn` — per-obs `grad_eta[k]` and `neg_hess_eta[k*np+l]`
in eta-space (the natural `X'WX` IRLS assembly). Already documented as "required
to use this spec through `tulpa_laplace_spec_*`".

**Numerical invariant to preserve:** the working weight (`neg_hess_eta`) is the
**expected (Fisher) information**, not the observed Hessian, for non-canonical
links (beta, scaled-Bernoulli) so the Newton Hessian stays positive-definite.
See `variance_fn`/`grad_hess_bernoulli` in `laplace_family_link.h`. Built-in
family specs must return Fisher info, not AD-observed Hessian.

---

## 4. Keystone (Phase L) — design + step plan

**Target end-state:** ONE inner Laplace solver (the generalized
`laplace_mode_spec_dense_impl`) over latent = [beta per process, RE terms, GMRF
blocks], spec-driven likelihood. The nested outer-grid driver builds each GMRF
block's `Q(theta)` per grid cell and calls the unified solver. Built-in families
ship as `LikelihoodSpec`s inside tulpa. Occupancy's scaled-Bernoulli + `det_prob`
live in a tulpaObs `LikelihoodSpec`. The family-enum inner loop is retired.

**Steps (verify + commit each):**
- **L1 — Built-in family specs. ✅ DONE (`228ee05`).** `builtin_family_spec()`
  in `src/laplace_builtin_family_spec.h` wraps the family-enum closed forms
  (`grad_hess_for_family` / `log_lik_for_family`) as `eta_weights_fn` +
  `ll_double` — one generic adapter, no per-family copy-paste, Fisher weight
  preserved. Test harness `cpp_laplace_spec_test_family` (in `laplace_spec.cpp`)
  + `tests/testthat/test-laplace-spec-builtin-family.R` cross-check all 8
  families (+ iid RE) against `cpp_laplace_fit` to 1e-5. This adapter is what
  L3/L4 will feed into the nested kernel.
- **L2 — GMRF blocks in the spec solver.** Generalize `SpecLatentLayout`,
  `compute_eta_spec`, `scatter_spec`, `log_prior_latent`, `apply_latent_step` to
  carry GMRF block segments. **Design decided (2026-05-24), correcting the doc's
  earlier "pass Q as CSC" sketch:**
  - **Reuse `tulpa::LatentBlock`, not a flat CSC.** Each block's `add_prior`
    (scatters `-Q.field` into grad, `+Q` into H), `log_prior`, `center`, `d_fac`
    (BYM2/IID reparam coefficient), and `idx` (obs->unit) callbacks already
    encode the per-block prior math via the existing factories (`add_icar_prior`,
    `add_rw1_precision`, ...). A raw CSC can express ICAR but not BYM2's `d_fac`
    or HSGP's basis — reusing `LatentBlock` is the single-source-of-truth move
    and what "mirror the nested kernel" actually means.
  - **`laplace_mode_spec_dense_impl` gains `const std::vector<LatentBlock>*
    blocks = nullptr, int k_grid = 0`.** nullptr => existing behaviour, so every
    current caller (gaussian/family/multi-RE harnesses, tulpaRatio/tulpaObs
    conditional Laplace) is untouched.
  - **Layout contract:** compacted latent is `[beta per proc | RE terms |
    blocks]`. For the single-arm case this is bit-identical to the nested
    kernel's `x` (`block.start == p + n_re_groups == block's compacted
    latent_offset`). The solver gathers `params -> x_latent` (one Rcpp buffer,
    allocated once outside the Newton loop) only for the three block callbacks
    that take `const Rcpp::NumericVector&` (add_prior/log_prior/center); eta and
    the likelihood cross-terms read `params[block_param_start + idx - 1]`
    directly. Guard: assert `block.start == computed block_latent_offset`.
  - **Scope of L2 = `INDEXED_SINGLE` blocks, dense, single-process** — exactly
    `run_multi_block_nested_laplace`'s capability (its `accumulate_latent_cross_
    terms` is `INDEXED_SINGLE`-only). Covers icar, bym2 (two blocks), car_proper
    (has `prep`), rw1, rw2, ar1, iid, nngp, tgmrf. **DENSE_BASIS (hsgp) and
    BILINEAR_FACTOR (lf) defer to the sparse joint path at L4** — they never went
    through the dense single-arm driver.
  - **Verify:** `cpp_laplace_spec_test_block` harness builds an ICAR block
    (mirroring `cpp_nested_laplace_icar`) + builtin-family spec; cross-check
    mode/log_marginal at a single tau against `cpp_nested_laplace_icar`
    (tau_grid length 1) to 1e-5. The family-enum vs spec likelihood equality is
    already proven (L1), so any mismatch isolates the block-scatter wiring.
- **L3 — Route single-block nested through the unified solver.** Numerical-
  equivalence test vs current (`tests/testthat/test-nested-laplace*.R` must pass
  to tolerance).
- **L4 — Route multi-block + joint through it** (`cpp_nested_laplace_multi`,
  `cpp_nested_laplace_joint_multi`). Watch the `(sigma, alpha)` reparam and the
  post-Newton phi-centering invariant (centering must shift each arm's
  intercept — see global memory `feedback_centering_breaks_joint_logmarginal`).
- **L5 — Retire family-enum + det_prob.** Remove the nested kernel's
  `grad_hess_for_family`/`det_prob` usage; delete `det_prob` from tulpa's R + C++
  surface. Add the occupancy scaled-Bernoulli `LikelihoodSpec` to tulpaObs
  (`eta_weights_fn` returning the `q*sigma*(1-sigma)^2/(1-q*sigma)` Fisher info).
- **L6 — Recovery tests** (per global "statistical code needs recovery tests"):
  parameter recovery for built-in families + occupancy through the unified path,
  fit-to-convergence, `|est-truth| < tol` across seeds, CI coverage.

**Guard:** the 24 `test-nested-laplace*.R` + joint test files are the numerical-
equivalence safety net. Keep them green at L3/L4.

---

## 5. R-side phases (2–5)

- **Phase 2 — knobs → control.** `tulpa_nested_laplace` and
  `tulpa_nested_laplace_joint`: keep statistical args top-level
  (`responses/prior/copy/phi_grid/prior_sigma/prior_alpha`, `y/n_trials/X/spec/
  prior/family/phi/sigma_re/re_idx/...`); move perf/numerical knobs into
  `control = list()`: `max_iter, tol, n_threads, n_threads_outer, tile_warm,
  prune, prune_tol, x_init, store_Q, adaptive_grid, adaptive_grid_edge_thresh,
  adaptive_grid_max_passes, var_of_means_consistency, force_sparse,
  keep_grid_hessians`. Update all tulpa tests + Rd. Breaks tulpaObs callers
  (Phase 6).
- **Phase 3 — remove `tulpa_priors_legacy`.** Thin wrapper over `tulpa_priors()`
  in `R/priors.R:339`. Remove fn + export + Rd; migrate callers to
  `tulpa_priors(beta=, sigma=prior_pc(...), phi=prior_pc(...))`.
- **Phase 4 — `tulpa()` reaches nested.** Register nested backends in
  `BACKEND_REGISTRY` (`R/inference_modes.R`) with `input = "nested"`, Tier 2.
  Add a `.tulpa_fitter_args` branch (`R/tulpa.R`) that builds a `prior` from
  parsed `latent()` blocks (the formula layer already parses
  `latent(<tulpa_latent_block>)`; `prior_from_spec()` bridges spatial/temporal
  specs). Auto-select nested when latent blocks present. **Joint stays a
  registered model-package engine entry** — the single-response formula API
  cannot express multiple arms; forcing it would be a new leak.
- **Phase 5 — split `nested_laplace_joint.R`** (2625 lines) into: public entry,
  backend table, hyperpriors, refinement/consistency passes, helpers. No
  behaviour change; load_all + tests green.

---

## 6. Siblings (Phase 6)

On disk under `~/Documents/dev/`: **tulpaObs** (heavy R-level user of the
nested engines), **tulpaRatio** (C++/LikelihoodSpec via `LinkingTo: tulpa` — the
R signature changes mostly don't touch it; the keystone's det_prob removal does
not affect it), tulpaGlmm, tulpaMesh.

tulpaObs call sites to migrate (new `control=` signatures + occupancy spec from
Phase L): `R/em_nested_laplace.R` (calls `tulpa_nested_laplace(..., family=
"bernoulli", det_prob=q_i)` at ~:308 — this becomes a tulpaObs LikelihoodSpec),
`R/family_cover_hurdle.R` (`tulpa_nested_laplace_joint` at ~:880/901),
`R/laplace.R`, `R/occu_fit.R`, `R/sla_cover_hurdle_joint.R`, plus
`tests/testthat/test-*.R`. tulpaObs already ships `LikelihoodSpec`s
(`src/occ_likelihood.h`, `src/occu_fit.cpp`) for the NUTS path — the occupancy
Laplace spec joins them.

Finish with `devtools::check(args="--no-manual")` on tulpa, `load_all` on
siblings.

---

## 7. Working setup

- **R:** `C:\Program Files\R\R-4.6.0\bin\x64\Rscript.exe` (Windows R, not WSL).
  Never `Rscript -e '...'` inline (segfaults on Windows) — write a `.R` file.
- **Scratch harnesses** (untracked, in `dev_notes/`): `_verify.R` (load_all,
  optional test filter arg), `_document.R` (roxygen regen). Recreate if missing:
  ```r
  # dev_notes/_verify.R
  suppressWarnings(suppressMessages(
    devtools::load_all("C:/Users/Gilles Colling/Documents/dev/tulpa", quiet=TRUE)))
  cat("LOAD_OK\n")
  ```
  Run: `"/c/Program Files/R/R-4.6.0/bin/x64/Rscript.exe" dev_notes/_verify.R`
- **After C++ edits:** `Rcpp::compileAttributes()` then `devtools::document()`;
  delete orphaned `.Rd` for internal C++ funcs.

## 8. Constraints (from global CLAUDE.md + memory)

- **Work on `main`** — no feature branches, no worktrees.
- **Concurrent tree edits:** an external process edits this repo mid-session.
  **Stage commits explicitly** (named files), never `git add -A`. Commit small
  and often.
- **Pre-release: no back-compat shims** — rename or hard-error; no deprecation /
  translation / fallback layers (tulpa/tulpaObs/tulpaRatio have no external
  users).
- **ASCII-only roxygen/Rd** — bare Unicode outside `\eqn{}` breaks the LaTeX PDF
  manual. `θ→theta`, `≥→>=`, `×→x`, etc. (`² ³` and em-dash are Latin-1 and OK).
- **No copy-paste across specialized functions** — shared sub-computations as
  `static inline` helpers (this is exactly the two-inner-Newton problem L solves).
- **Statistical code needs recovery tests** — shape/dim smoke tests are not
  validation; add parameter-recovery + CI-coverage tests for any new fit path.
- **Anchor cost on Laplace, not NUTS** — the hot path is composed approximation
  (Laplace + small residual MCMC), not NUTS-over-everything.

## 9. Naming note (not in scope, but record it)

`numdenom` was renamed to **tulpaRatio**. tulpa's own `CLAUDE.md` still says
"numdenom: Ratio models" and "Extracted from numdenom" — the live-sibling
reference is stale (the historical "extracted from" mention is fine as history).
