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
| L | **Keystone:** full solver unification (spec-driven Laplace handles GMRF blocks; det_prob → tulpaObs) | 🔄 in progress — **L1 done** (`228ee05`); **L2 done** (`9ece117` ICAR + bym2); **L3 done** (`4829dea` functor loop, `192591a` spec np==1 → shared loop, `4146d9f` spec_inner_solve_np1 + det_prob, `fd29078` nested driver → spec): every single-block / np==1 multi-block nested kernel now solves through one spec inner solve; duplicate obs+latent-cross scatter deleted; beta-prior convention reconciled. **L4 done** (`a1e4f18`/`32569bd` joint multi-arm spec-driven), **L4.3 done** (`82573d0` np>=2 spec loop collapsed), **L5 done** (`det_prob` + `bernoulli` family retired; single-arm driver takes a model `LikelihoodSpec` via `XPtr<NestedLikelihood>`; occupancy scaled Bernoulli moved to tulpaObs). Equivalence net green. **L6 done** (`test-nested-laplace-recovery.R`: beta + RE-SD recovery + CI coverage for all 6 built-in families through the unified path; occupancy recovery shipped in L5). **Keystone (L) complete — Phase 4 next** |
| 2 | Collapse nested engine knobs into `control = list()` | ✅ done — commit `274649e`: both `tulpa_nested_laplace()` / `tulpa_nested_laplace_joint()` take statistical args top-level + a single `control = list()` (matching `tulpa()`); 15 perf/numerical knobs moved in, read via `control$x %||% default`. Dead single-arm `verbose` dropped. Internal callers (`em_laplace`, tgmrf pilots), all tulpa tests, Rd, and shipped examples migrated. Behavior-preserving (force_sparse + tgmrf R-vs-C++ equivalence green; full nested/joint/tgmrf suite 0 fail). **Straggler fixed later (commit `42cb396`):** the migration missed the `log_marginal_at` closures in `tgmrf_vi`/`tgmrf_nuts` (still passed `max_iter/tol/n_threads` top-level → `tryCatch(error=NULL)` swallowed the "unused argument" error into -Inf → init guards fired). The "0 fail" check ran with the recovery/agreement assertions `skip_on_cran`, which is why it hid. Root cause was the inner-solve closure being copy-pasted across imh/vi/nuts; now unified into one `.tgmrf_make_log_marginal()` (single contract source) + eager non-swallowing structural check at the pilot mode. Verified under `NOT_CRAN=true`. |
| 3 | Remove `tulpa_priors_legacy` | ✅ done — commit `7f6b5c5`: dead back-compat shim over `tulpa_priors()` with **zero callers** (R/tests/examples/vignettes). Removed fn + `@export` + generated Rd; NAMESPACE regenerated. Prior tests green. |
| 4 | Route single-arm `latent()` formulas through `tulpa()`; register nested backends | ✅ done — commit `f642c38`: `nested_laplace` (single-arm) + `nested_laplace_joint` registered Tier 2 with a new `input = "nested"` contract; `auto_select_mode` / `select_backend_for_mode` route `latent(...)` blocks to `nested_laplace` (auto + structured + explicit), precedence over the size heuristics. `.tulpa_fitter_args` "nested" branch builds the prior from parsed latent blocks (each tgmrf is already a valid multi-block prior block), threads a single `(1\|g)` through the kernel's `re_idx`/`n_re_groups`/`sigma_re`. Joint is a registered model-package engine — never auto-selected, errors loudly if forced (single formula can't express multiple arms). Fail-loud guards (latent + non-nested backend, nested + no block, joint via formula, >1 RE term) run **before** the reachability check so the latent message wins. `test-tulpa-entry-nested.R`: routing/registry contract + guard errors + **formula route is numerically identical to a direct `tulpa_nested_laplace()` call** (theta_grid/log_marginal/moments) + front-door beta recovery across seeds. |
| 5 | Split `nested_laplace_joint.R` (2625 lines) | ✅ done — commit `e1d6ebc`: byte-identical code move (verified: LF-normalized reconstruction `cmp`-equals the HEAD blob, all 58 U+2014 em-dashes intact, zero double-encoding). The 2592-line monolith split into the public entry (`tulpa_nested_laplace_joint`, kept in `nested_laplace_joint.R`, 409 ln) + 5 concern files: `_hyperpriors.R` (sigma/alpha PC + half-normal), `_backends.R` (single-block dispatch table + `.joint_call_kernel_via_multi`), `_helpers.R` (cartesian/arm/copy/layout), `_refine.R` (adaptive grid + var-of-means consistency), `_multi.R` (list-of-blocks dispatch). No `Collate`, no load-order dep (only fn defs + one deferred-closure list), no roxygen/`@export` moved -> NAMESPACE + `man/` untouched, no re-document. load_all green; joint equivalence net green (icar/bym2/car-proper/adaptive-grid/phi-grid/multi/multi-block/sigma-pos-prior/alpha-ridge/prune/sparse-equiv/beta/latent-factor/hsgp-svc/hsgp-mo/parallel + entry-nested + the heavy joint multi-recovery net, all under `NOT_CRAN=true`). |
| 6 | Update siblings (tulpaObs et al.); final `devtools::check` | ✅ done — commit `edee90c` (auto). tulpaObs call sites were already migrated by its own commits `86c6036` (L5 occupancy `LikelihoodSpec` via `likelihood=`) + `be3bdb1` (joint `control=list()` + N-mixture family); this phase **verified** them against the reinstalled current-HEAD tulpa and fixed the fallout the **first** full `R CMD check` surfaced. tulpa reinstalled from source (ABI 24, headers byte-identical), tulpaObs recompiled+loaded clean against it. **tulpaObs suite: 39/39 files, 928 pass / 0 fail** (NOT_CRAN=true so all recovery/CI-coverage assertions ran; 3 benign warnings = N-mixture `K_max` truncation-boundary diagnostics, tulpaObs-owned, unrelated). **tulpa `R CMD check --no-manual`: Status OK (0/0/0)** after three fixes (all in-scope tulpa repo, none touching runtime R/C++): (1) WARNING — `tulpa_nested_laplace.Rd` `\link{LikelihoodSpec}` (broken; `LikelihoodSpec` is a C++ struct w/ no Rd page) -> `\code{}` in the L5 roxygen; (2) NOTE — `clean_migration.md` + `dev_document.R` -> `.Rbuildignore`; (3) ERROR — `test-nmix-laplace.R` `coef(um_fit)`/`logLik(um_fit)` fell through to `coef.default` ("$ operator not defined for this S4 class") in the shared `test_check` session (order-dependent search-path shift after another file attaches a pkg; passes in isolation). Fixed by namespace-qualifying unmarked's S4 generics (`unmarked::coef`/`unmarked::logLik`) — immune to the shadowing. **Migration complete (all 7 phases).** |

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
- **L3 — Route single-block nested through the unified solver. ✅ DONE.**
  `4829dea` lifts the data log-lik in `laplace_newton_solve` to a templated
  functor (one Newton loop for family + spec). `192591a` delegates the spec
  solver's np==1 path to it. `4146d9f` extracts `spec_inner_solve_np1`
  (laplace_spec_solve.h) as the single np==1 inner solve and threads `det_prob`
  through `BuiltinFamilyResponse`. `fd29078` routes
  `run_multi_block_nested_laplace` through `spec_inner_solve_np1` (one-process
  `ModelData`/`ParamLayout`/`builtin_family_spec` built once, pooled scratch per
  outer-grid thread, predictive variance / cheap-pass / fitted_eta preserved),
  deleting the driver's duplicate `scatter_obs_grad_hess_base` +
  `accumulate_latent_cross_terms`. Covers icar/bym2/car_proper/rw1/rw2/ar1/iid/
  nngp/tgmrf and the np==1 multi-block entry. Beta-prior log-density convention
  reconciled (spec path includes it; modes unchanged). Equivalence net: full
  suite 2421 pass / 0 fail, spec-block icar/bym2 now assert exact log-marginal.
- **L4 — Route the joint multi-arm driver through it. ✅ DONE.** `a1e4f18`
  functor-izes both joint Newton loops (`laplace_newton_solve_joint{,_sparse}_ll`
  take a `JointLogLik`; family forwarders build the functor — pure refactor).
  `32569bd` makes `cpp_nested_laplace_joint_multi` spec-driven: `JointArm` carries
  an optional per-arm `LikelihoodSpec` bundle (+ `det_prob`); `build_joint_arm_specs`
  resolves each arm to an `ArmSpecView` once per grid (built-in via
  `builtin_family_spec`, else model-supplied), `arm_grad_hess` bridges
  `eta_weights_fn -> GradHess` so all four scatters (dense / sparse / indexed-cached
  / dense-basis) flow through one boundary, and `JointSpecLogLik` sums per-arm
  `ll_double`. Family-enum forwarders + `JointFamilyLogLik` deleted. `sync_dispersion`
  refreshes the built-in `phi` from the live arm after prep (phi_grid axis). The
  `(sigma, alpha)` reparam and post-Newton phi-centering invariant are untouched
  (see global memory `feedback_centering_breaks_joint_logmarginal`). Equivalence
  net green: joint/spec suite 924 pass / 0 fail (phi-grid 14/0, multi-recovery
  106/0).
- **L4.3 — Collapse the np>=2 multi-process spec loop into the shared solve.
  ✅ DONE.** `82573d0` — the np>=2 branch of `laplace_mode_spec_dense_impl`
  was already spec-driven (no family enum) but hand-rolled its own Newton loop,
  duplicating the machinery L3 unified for np==1. Generalized `spec_inner_solve_np1`
  -> `spec_inner_solve` (eta buffer N*np; the shared `laplace_newton_solve_ll`
  treats eta as opaque, so np lives entirely in the spec helpers), generalized
  `scatter`/`gather_compacted_latent` to all np processes, and collapsed the np==1
  special-case + the ~155-line np>=2 loop into one call (deleted `apply_latent_step`).
  GMRF blocks stay gated to np==1, so np>=2 has no blocks / no centering. This is
  now the single inner Laplace loop for every spec path. Net -188 lines.
  Equivalence net green: `test-laplace-spec.R` 17/0 (carries the np==2 gaussian2p
  fixture that drove the deleted loop), spec block icar/bym2 27/0, nested 52/0,
  multi-block 42/0.
- **L5 — Retire det_prob; model-supplied nested likelihood. ✅ DONE.** The
  occupancy `det_prob` hook and the `bernoulli` family are gone from tulpa:
  `grad_hess_for_family`/`log_lik_for_family` lost their `det_prob` arg, the
  scaled-Bernoulli closed forms (`bernoulli_sigma`/`*_bernoulli`) are deleted
  (plain Bernoulli is `binomial` with `n_trials = 1`), and the `det_prob`
  plumbing is removed from `BuiltinFamilyResponse`, `FamilyLogLik`,
  `scatter_obs_grad_hess_base`, `laplace_newton_solve`, `eval_penalized_log_lik`,
  the dead `JointArm.det_prob`, the single-arm driver, `cpp_nested_laplace_multi`,
  the R `tulpa_nested_laplace` / `em_laplace` block convention, and the Rd. In
  its place the single-arm nested driver takes an optional model-supplied
  `LikelihoodSpec` (mirroring `JointArm.spec`): a new `tulpa::NestedLikelihood`
  bundle (`inst/include/tulpa/nested_likelihood.h` -- `{spec, response_data,
  shared_ptr keepalive}`) is built in a model package's C++ and passed from R as
  an `XPtr<NestedLikelihood>` via `tulpa_nested_laplace(likelihood = )` (a
  single-block prior auto-wraps into the spec-capable multi-block path); the
  inner solve routes its score / Fisher weight / log-lik through the spec. The
  marginalized single-season occupancy likelihood (scaled Bernoulli, Fisher info
  `q*sigma*(1-sigma)^2/(1-q*sigma)`) now lives in tulpaObs
  (`src/occ_nested_likelihood.cpp::occ_make_nested_likelihood`), wired into
  `.tobs_occu_state_marginal_fit`. tulpa ships a byte-identical reference as the
  test harness `cpp_nested_laplace_test_occupancy_likelihood`
  (`src/nested_laplace_test_occupancy.cpp`); `test-nested-laplace-occupancy.R`
  drives the full recovery/calibration net through `likelihood = `, with the
  q = 1 case asserting spec == built-in binomial. Equivalence net green: the
  spec + nested + joint sweep (`laplace-spec*`, all `nested-laplace*` incl.
  joint multi-recovery) 0 fail.
- **L6 — Recovery tests. ✅ DONE.** Per the global "statistical code needs
  recovery tests" rule, `test-nested-laplace-recovery.R` certifies that the
  unified path *recovers* parameters, not just that it reproduces the
  family-enum mode (the L1 equivalence net). A region-grouped IID block keeps
  the RE identified while the outer grid integrates its SD; the fixed-effect
  posterior is the Gaussian mixture over grid cells (`grid_modes` +
  `solve(grid_hessians)`, weighted) -- beta is marginalized over the
  hyperparameter grid, never read off a plug-in-MAP cell. In-suite: poisson +
  binomial, 12 seeds, beta bias + interval calibration + RE-SD recovery. Slow
  gate (`TULPA_SLOW_TESTS=true`): all 6 built-in families (gaussian, poisson,
  binomial, neg_binomial_2, gamma, beta), 20 seeds, fit-to-convergence,
  `|bias| < 0.12`, and >= 85% *aggregate* CI coverage (pooled over all
  family x coefficient cells, plus a loose per-cell floor) -- a hard
  per-coefficient 17/20 gate is mis-designed at N = 20 (a correctly calibrated
  ~90%-coverage Laplace fails it on some coefficient ~80% of the time across 12
  cells; pooling 240 trials makes the mean a stable standard). Occupancy
  recovery + calibration already shipped with L5 in
  `test-nested-laplace-occupancy.R`. **This completes the keystone (Phase L).**

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
Phase L): `R/em_nested_laplace.R` (the `family="bernoulli", det_prob=q_i` call
became `occ_make_nested_likelihood()` + `likelihood=` — **done in L5**),
`R/family_cover_hurdle.R` (`tulpa_nested_laplace_joint` at ~:880/901),
`R/laplace.R`, `R/occu_fit.R`, `R/sla_cover_hurdle_joint.R`, plus
`tests/testthat/test-*.R`. tulpaObs already ships `LikelihoodSpec`s
(`src/occ_likelihood.h`, `src/occu_fit.cpp`) for the NUTS path — the occupancy
Laplace spec joins them.

Finish with `devtools::check(args="--no-manual")` on tulpa, `load_all` on
siblings.

**Outcome (2026-05-26).** All tulpaObs call sites were already on the new API
(its own commits `86c6036` + `be3bdb1`); `R/laplace.R` and
`R/sla_cover_hurdle_joint.R` never call the engine directly (they route through
`tulpa_em_laplace()` / consume the joint fit's `store_Q` output), and `abun.R`
shares the occupancy EM path — so no further edits were needed there. The work
was **verification + check fallout**: reinstall current-HEAD tulpa, recompile
tulpaObs against it (clean), run the tulpaObs suite under `NOT_CRAN=true`
(39/39, 928 pass / 0 fail), and clear the first full `R CMD check`'s 1 ERROR /
1 WARNING / 1 NOTE (see the Phase-6 status row). Final tulpa check: **Status
OK**.

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

---

## 10. Follow-up (2026-05-26): collapse the unrot-flagged ABI families + retire the family-enum single-point solver

`unrot` flagged two CRITICAL clone clusters left after the keystone:
1. the spatio-temporal nested-Laplace cross-product
   (`tulpa_nested_laplace_st_<spatial>_<temporal>`, 15 callables);
2. the single-block nested temporal entries (`*_{rw1,rw2,ar1}`); and
3. the family-enum single-point Laplace family (`tulpa_laplace_mode_*` C-ABI +
   the `tulpa::laplace_mode_*` solver bodies behind `cpp_laplace_fit*`).

Key finding: the keystone (Phase L) unified the **inner solve** (`spec_inner_solve`)
for the *nested* path, but the *standalone single-point* path
(`cpp_laplace_fit*` -> `tulpa::laplace_mode_*` in `laplace_core*.cpp`) is still on
the old **family-enum inner Newton** -- the surviving second inner Newton that
goal #4 ("retire the family-enum inner loop") removed only on the nested side.

| Step | What | Status |
|---|---|---|
| ST | Collapse 15 `*_st_<spatial>_<temporal>` -> 5 `*_st_<spatial>` (runtime `temporal_type` via `make_temporal_ops`). ABI 24->25. | DONE -- commit `f0c39ef` |
| B1 | Remove the 8 dead `tulpa_laplace_mode_*` C-callables + `LaplaceMode*Fn` typedefs + `tulpa_shims_laplace.h` + `copy_mode_result`; keep `LaplaceShimResult` (spec shims reuse it). No sibling consumes them. ABI 25->26. | DONE -- commit `cdf348a` |
| A | Collapse the 3 single-block temporal entries `*_{rw1,rw2,ar1}` -> one `*_temporal` (runtime `temporal_type` via the same `make_temporal_ops`). ABI 26->27. tulpaGlmm temporal call sites updated. | DONE -- commit `48c3de7` |
| B2 | Retire the family-enum single-point solvers (`tulpa::laplace_mode_*`); route the spec-shaped fits onto `spec_inner_solve`. | DONE -- `b0967fe` (dead trio), `2c66415` (engine ext), `9e165d7` (route + delete bodies), `6ffbe47` (test shift) |

### B2 spec (next phase) -- "kill the second inner Newton, single-point side"

Scope decided from a usage + architecture sweep (R callers / test refs / spec
support). **No `TULPA_ABI_VERSION` bump:** `cpp_laplace_fit*` are Rcpp `.Call`
exports + internal C++, not registered cross-package callables. tulpaRatio has
its **own** `_tulpaRatio_cpp_laplace_fit*` (separate compiled copies) -- not
tulpa's -- so B2 is contained within tulpa.

- **B2-dead (clean, low-risk).** Delete `cpp_laplace_fit_{rsr,multiscale_gp,
  multiscale_temporal}` (0 R callers, 0 test refs in tulpa) + their
  `tulpa::laplace_mode_{rsr,multiscale_gp,multiscale_temporal}` bodies
  (`laplace_core_spatial.cpp`, `laplace_core_gp.cpp`) + any rsr/multiscale-only
  static helpers that orphan. Regenerate RcppExports. Also delete the
  B1-orphaned forward declarations of every `tulpa::laplace_mode_*` in
  `tulpa_shims.cpp:48-137` (left behind when B1 removed the shims that used
  them -- harmless but dead).
- **B2-live (the prize, delicate).** Route `cpp_laplace_fit`, `_multi_re`,
  `_spatial` (icar), `_bym2` through `spec_inner_solve` /
  `laplace_mode_spec_dense_impl` and delete the four
  `tulpa::laplace_mode_{dense,dense_multi_re,spatial,bym2}` bodies. Equivalence
  is already proven by the `cpp_laplace_spec_test_{family,multi_re,icar,bym2}`
  harnesses -- the marshalling they do (build `ModelData`/`ParamLayout`/
  `builtin_family_spec` + the ICAR/BYM2 `LatentBlock`) is exactly what the real
  exports need; promote it from the harness into the export. Must preserve every
  current input: `weights`, `offset`, `beta_prior` (mean/sd), `return_re_cov`
  (the EM M-step covariance blocks -- `spec_inner_solve` already supports it via
  `inv_block_layout` + `LaplaceResult.re_cov_flat`), and `x_init`.
  **Test shift:** the `cpp_laplace_spec_test_*` harnesses cross-check spec vs
  family-enum; once the family-enum bodies are gone there is nothing to compare
  against, so convert those equivalence assertions to parameter-recovery /
  CI-coverage tests (the L6 discipline) before deleting the bodies.
  Heaviest test surface: `cpp_laplace_fit_spatial` (13 test refs).
- **B2-NNGP (intentional boundary -- keep specialized).** `cpp_laplace_fit_gp`
  (6 R callers) stays on its specialized NNGP path. NNGP is **not** a dense
  `LatentBlock`/spec backend even in the nested driver -- the nested nngp/hsgp
  entries route through `run_multi_block_nested_laplace_joint_sparse_impl`
  (sparse), not the dense `spec_inner_solve`. Routing single-point NNGP through
  spec would require making NNGP a spec block, disproportionate and divergent
  from the nested path. Document the boundary; do not force it.

### B2 outcome (2026-05-26)

Done in four commits, no `TULPA_ABI_VERSION` bump (contained within tulpa).

- **B2-dead (`b0967fe`).** Deleted `cpp_laplace_fit_{rsr,multiscale_gp,
  multiscale_temporal}` + their `tulpa::laplace_mode_*` bodies (only shared
  helpers used -> nothing orphaned) + all 8 B1-orphaned `tulpa::laplace_mode_*`
  forward decls in `tulpa_shims.cpp`. RcppExports regenerated.
- **Engine ext (`2c66415`, additive/behaviour-preserving).** `BuiltinFamilyResponse`
  gained an optional per-obs `weights` (scales the eta score + Fisher weight,
  `ll_double` left unweighted to match the retired solver); `scatter_spec` /
  `log_prior_latent` / `spec_inner_solve` thread an optional `const BetaPrior*`
  (per-coef mean+precision; null = the scalar `sigma_beta` ridge); new
  `laplace_mode_spec_dense_solve` returns the full `LaplaceResult` (+ `return_re_cov`
  builds the per-(term,group) inv-block layout from the spec layout). The void
  `laplace_mode_spec_dense_impl` (the cross-package shim entry) is now a thin
  wrapper over it -- shim ABI unchanged.
- **Route + delete bodies (`9e165d7`).** `cpp_laplace_fit{,_multi_re,_spatial,
  _bym2}` marshal into `laplace_mode_spec_dense_solve` and the four
  `tulpa::laplace_mode_{dense,dense_multi_re,spatial,bym2}` bodies are deleted.
  New `laplace_spec_fit.h`: `build_spec_family_inputs` (the iid-RE built-in-family
  ModelData/ParamLayout shared by fit/spatial/bym2) + `pack_to_spec_re_params`
  (a term's marginal-SD or packed Sigma-Cholesky `pack` -> spec log-Cholesky
  params; exact same Q + log|Q|). multi_re carries the full multi-term marshalling
  (slopes / correlated via the pack conversion; the `(0+x|g)` no-intercept design
  detected by an all-ones first column) + weights + offset + per-coef beta prior
  + warm start + `return_re_cov`; spatial builds the ICAR block + x_init, bym2 the
  phi+theta blocks. **Mode preserved bit-for-bit; log-marginal now folds in the
  beta-prior log-density on the single-point/spatial paths too**, so the
  single-point entry agrees exactly with the nested kernel at one cell (the old
  `cpp_laplace_fit_spatial`-omits-beta gap is closed).
- **Test shift (`6ffbe47`).** Deleted the now-redundant `cpp_laplace_spec_test_
  {family,icar,bym2}` harnesses. `test-laplace-spec-builtin-family.R` now checks
  `cpp_laplace_fit`'s MAP against the `stats::glm` MLE (gaussian/poisson/binomial/
  gamma) + an 8-family convergence smoke + a fit-vs-multi_re marshalling-consistency
  check; the block tests drive `cpp_laplace_fit_{spatial,bym2}` directly against
  the nested kernel. Kept `cpp_laplace_spec_test_{gaussian,gaussian2p,multi_re}`
  (custom user spec / np==2 / closed-form Gaussian -- each reaches what no
  single-response export does).

**Only one inner Newton remains** (`spec_inner_solve`); the family enum survives
only as the per-obs closed forms behind `builtin_family_spec`. `cpp_laplace_fit_gp`
(NNGP) intentionally stays specialized. Verified green across the spec, family,
block, RE-cov, AGHQ, EM, front-door, and nested-kernel suites (NOT_CRAN=true).

### Sibling caveat (2026-05-26)

tulpaObs is clean and was verified against ABI 27 (recompile + targeted nested/
occupancy test). **tulpaGlmm is under a large concurrent external rewrite** of
`src/glmm_nested_laplace.cpp` (~757-line uncommitted diff touching the temporal
path); the ST + A temporal-shim call-site updates (`get_nested_laplace_temporal_fn`)
are present in that working tree but entangled in the external WIP, so tulpaGlmm's
rebuild/commit against ABI 27 is owned by that effort, not this one.

---

## 11. Follow-up after B2 (2026-05-26): fold the spatial x temporal driver onto the unified inner solve

`unrot` (re-run after B2) flags the nested-Laplace spatial-family clones as the
top remaining CRITICAL clusters: `cpp_nested_laplace_st_*` (5, J=86%),
`tulpa_nested_laplace_st_*` (5 shims), `tulpa_nested_laplace_*` (8 shims). The
boilerplate (`pack_laplace_shim_inputs`, `marshal_adj`, `copy_nested_laplace_
result`) is already extracted and the per-family C-ABI signatures are genuinely
heterogeneous (icar: tau; car: tau+rho; bym2: sigma+rho+scale; nngp: coords+nn;
hsgp: basis+eig) -- those shims cannot collapse into one entry. The real rot is
one layer down.

**Key finding -- B2's "only one inner Newton remains" was scoped, not total.**
The keystone (L) unified the single-arm `run_multi_block_nested_laplace` and the
joint multi-arm `cpp_nested_laplace_joint_multi` onto `spec_inner_solve`. But the
**spatio-temporal dispatch path** (`run_spatial_x_indexed_temporal_nested_laplace`
+ its `_sparse_impl`, in `nested_laplace.cpp`) is a **third inner solve** that L
left out of scope (L2 was explicitly "INDEXED_SINGLE blocks ... exactly
run_multi_block's capability"). It still:
- runs its own Newton via `laplace_newton_solve_sparse` with a hand-rolled
  `nl_scatter_obs_spatial_x_indexed_temporal{,_sparse,_cached}` scatter that calls
  `grad_hess_for_family` **directly** (the family enum, not the spec boundary);
- carries a parallel block representation, `struct SpatialBlockOps`, built by five
  `make_<x>_spatial_ops` factories -- duplicating the per-family block math that
  the pure-spatial entries express as `tulpa::LatentBlock`;
- carries an ST-specific sparsity cache (`st_scatter_index_cache`).

**Why it folds cleanly (verified by reading, 2026-05-26).** Spatial x temporal is
structurally just two INDEXED_SINGLE `LatentBlock`s sharing observations -- the
same shape BYM2 already runs through `run_multi_block_nested_laplace` (two blocks),
except the two blocks index by different factors (spatial_idx vs temporal_idx).
`scatter_spec` (`laplace_spec.cpp:746-796`) assembles the latent-block Hessian
generically: each block resolves its **own** `blk.idx(i, 0)` and the **block x
block cross term** (`:788-794`) uses each block's independently-resolved index --
there is no assumption the two blocks share an index (BYM2 just happens to). And
`run_multi_block_nested_laplace` already routes to sparse Cholesky at
`n_x >= SPARSE_THRESHOLD`, iterates blocks with per-block `idx` for the predictive
variance (`nested_laplace_multi.h:253`), and supports `store_Q` / cheap-pass.
`LatentBlock` carries the sparse path (`add_prior_pattern` / `add_prior_sparse`,
`latent_block.h:225-235`). The family-enum vs spec weight equality is already
proven (L1: `builtin_family_spec` wraps `grad_hess_for_family`), so any mismatch
isolates block wiring. For np==1, single-DOF spatial + single-DOF temporal, the ST
scatter's `gh.neg_hess * w_a * w_b` (w=1) equals `scatter_spec`'s `d_a*d_b*s_hess`
(d=1) exactly.

This is the genuine continuation of goal #4 ("retire the family-enum inner loop"):
after this phase there is **one** inner Laplace solve (`spec_inner_solve`) for every
nested path, single-arm / multi-arm / spatial / temporal / spatio-temporal.

### Family landscape (drives the sub-steps)

| Family | pure-spatial driver | ST driver (today) |
|---|---|---|
| icar / car_proper / bym2 (areal) | `run_multi_block_nested_laplace` (dense, spec-driven; **inline** LatentBlock build) | `run_spatial_x_indexed_temporal_*` (own scatter, `SpatialBlockOps`) |
| nngp / hsgp | `run_multi_block_nested_laplace_joint_sparse_impl` (`make_nngp_block` / `make_hsgp_block`) | `run_spatial_x_indexed_temporal_*` **dense only** (`make_{nngp,hsgp}_spatial_ops`; sparse fields empty) |

Note the latent inconsistency the collapse also fixes: the inline pure-spatial ICAR
`LatentBlock` sets only the **dense** prior callbacks, while `make_icar_spatial_ops`
also has the sparse ones -- so pure-spatial ICAR at large `n_spatial` is dense where
it need not be. One block factory (with sparse) used by both paths cures it.

### Plan (verify + commit each; the 24 `test-nested-laplace*.R` files are the net)

| Step | What | Status |
|---|---|---|
| C0 | One `make_<x>_latent_block()` per areal family (icar/car_proper/bym2) returning `LatentBlock`(s) with **all** callbacks set (dense + sparse `add_prior_pattern`/`add_prior_sparse` ported from `make_<x>_spatial_ops`, + car_proper's `prep` log\|Q(rho)\|). Rewrite pure-spatial `cpp_nested_laplace_{icar,car_proper,bym2}` to build via these. Behaviour-preserving (modes/log-marginal identical; ICAR now sparse-capable). Single source of truth for areal spatial blocks. | TODO |
| C1 | Rewrite `cpp_nested_laplace_st_{icar,car_proper,bym2}` as `run_multi_block_nested_laplace(blocks = [spatial block(s) from C0, temporal block from make_temporal_ops])`. Delete dense `run_spatial_x_indexed_temporal_nested_laplace`, `nl_scatter_obs_spatial_x_indexed_temporal{,_cached}`, `nl_compute_eta_base_x_indexed_temporal`, the `st_scatter_index_cache`, and `make_{icar,car_proper,bym2}_spatial_ops`. Kills the family-enum ST scatter for areal. Convert any equivalence-vs-old-path test to recovery (B2's test-shift discipline). | TODO |
| C2 | Rewrite `cpp_nested_laplace_st_{nngp,hsgp}` onto `run_multi_block_nested_laplace_joint_sparse_impl(blocks = [make_nngp_block/make_hsgp_block, temporal block])`. Delete `make_{nngp,hsgp}_spatial_ops`, the sparse `run_spatial_x_indexed_temporal_nested_laplace_sparse_impl`, the dispatch wrapper, and finally `struct SpatialBlockOps`. (Moves ST nngp/hsgp dense -> sparse, an improvement.) | TODO |
| C-clean | Remove orphaned helpers; `Rcpp::compileAttributes()` + `devtools::document()`; full nested/joint/spec suite under `NOT_CRAN=true`. No `TULPA_ABI_VERSION` bump unless a shim signature changes (the ST shims stay; their cpp bodies just rebuild blocks differently). | TODO |
