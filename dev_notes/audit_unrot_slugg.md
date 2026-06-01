# tulpa — unrot + slugg audit

- Date: 2026-06-01 · branch `main` · HEAD `405fa12` (working tree dirty: nested_laplace_joint_multi.{cpp,h}, Makevars.win, tulpa_pch.h)
- Tools: `unrot scan` (AST clone / dead-code / scaling) + `slugg` grep-battery placeholder audit.
- Scope for findings: `R/`, `src/`, `inst/include/`. `experimental/` and `tests/` triaged out of severity (see packaging note P1).
- Line numbers drift; re-locate by symbol name.

---

## Part A — unrot (structural rot)

### P1 — packaging: `experimental/` ships in the tarball (HIGH)
`.Rbuildignore` excludes `dev_notes`, `dev_scripts`, `docs`, etc. but **not** `experimental/`.
`R CMD build` therefore bundles the entire `experimental/` tree (calibration scripts,
`morse_atlas.R`, self-cube probes) into the CRAN tarball. ~40% of every unrot clone group is
`experimental/*` noise for the same reason. Fix: add `^experimental$` to `.Rbuildignore`.

### Real clones (R/ + src/, scratch excluded)

| # | Sim | Location | Symbols | Note |
|---|-----|----------|---------|------|
| C1 | IDENTICAL 293 tok | `R/spatial_svc.R:426` `svc.tulpa_fit` ≡ `R/temporal_tvc.R:307` `tvc.tulpa_fit` | byte-identical posterior-extraction method across the spatial/temporal varying-coefficient twins | strongest dedup target — one shared `.vc_posterior(fit, axis)` helper |
| C2 | IDENTICAL 86 tok | `R/spatial_rsr_spde.R:167` `spatial_rsr` ≡ `R/temporal_rtr_posteriors.R:1` `temporal_rtr` | restricted-spatial-regression / restricted-temporal-regression constructors identical | shared constructor parameterized by axis |
| C3 | IDENTICAL 171 tok | `src/pg_binomial.cpp:75` `chol_solve` ≡ `src/pg_negbin.cpp:73` `chol_solve_nb` | dense Cholesky solve duplicated across the two PG samplers | lift to one `pg_chol_solve` in a shared PG header |
| C4 | RENAMED 95% (10 fn) | `src/tulpa_shims_nested_laplace.h:156-517` `tulpa_nested_laplace_{icar,bym2,car_proper,nngp,hsgp,st_*}_impl` | per-spatial-type shim bodies are near-identical; adding a type = copy a 150-280 tok block | dispatch table keyed by spatial type; body once, type-specific ops plugged in |
| C5 | RENAMED 95% (8 fn) | `R/spatial_gp.R`,`spatial_svc.R`,`spatiotemporal.R`,`temporal_core.R`,`temporal_gp.R`,`temporal_tvc.R` latent-block constructors | `spatial_gp`/`spatial_svc`/`temporal_rw1`/… share ~460-tok constructor scaffolding | extract a `.latent_block_scaffold()`; each constructor supplies only its kernel/prior |
| C6 | IDENTICAL 42-55 tok (6 fn) | `R/priors.R:120-263` `prior_normal/half_normal/half_cauchy/gamma/exponential/beta` | prior-object constructors identical modulo name/params | one `.make_prior(dist, ...)` factory |
| C7 | RENAMED 87% (3 fn) | `src/nested_laplace.cpp:999-1106` `cpp_nested_laplace_st_{icar,car_proper,bym2}` | spatiotemporal kernel triplet copy-paste-specialized | template over the spatial Q-builder |
| C8 | RENAMED 86% (3 fn) | `src/tulpa_shims_stochastic.h:49-137` `tulpa_{sgld,sghmc,mclmc}_fit_impl` | stochastic-gradient sampler shims share body | dispatch table |
| C9 | RENAMED 84% (4 fn) | `src/autodiff_utils.h:424-528` `log_lik_{zi_binomial,oi_binomial,zi_negbin,zoib}` | zero/one-inflated likelihood kernels duplicate the mixture scaffold | shared inflation wrapper around the base lpmf |

Lower-value families (enum parsers `parse_*` in `src/hmc_*.h`, `forward_solve_t`/`backward_solve_t`
triangular solves, `print.tulpa_*` methods, `cpp_test_linalg_*`) are listed by unrot but are short
and stable; defer.

### Scaling issues (CRITICAL/HIGH from unrot)
- `tulpa_nested_laplace_*` (C4) and `cpp_nested_laplace_st_*` (C7) are the flagged
  copy-paste-specialization families: O(N-lines) to add a spatial type. Registry/dispatch fix
  collapses both to O(1)-per-type. Tracked: gcol33/tulpa#49 (verified pass).
- `format_tulpa_prior_*` (7 fn), `cuda_batched_*` (6 fn): medium; the prior-formatter family folds
  into the C6 factory.

### Dead code — TREAT AS LOW-CONFIDENCE
unrot's C++ call graph flagged ~100 HIGH "dead" functions, almost all in
`src/hmc_gp_*.h`, `src/hmc_latent_grad.h`, `src/hmc_multiscale_temporal_grad.h`,
`src/hmc_gp_nc.h`. These are **template-instantiated AD gradient kernels and
`R_RegisterCCallable` targets** that a static call graph cannot see (they are reached via the AD
tape, function pointers, and the shim registry). Per the project's standing note, exported-but-unwired
backends are intentional surface. **Do not delete on this signal.** The only plausibly-real entries
are the `inst/examples/tgmrf_periodic_ar1.cpp` example functions (example code, expected unreferenced).

### Naming drift (cosmetic)
`.format_duration`/`format_duration`, `car_proper_log_prior`/`log_prior_car_proper`,
`.re_logchol_to_L`/`.re_L_to_logchol`, `.re_cov_theta_to_L_list`/`.re_cov_L_list_to_theta`.
The `car_proper` pair is the one worth aligning (same concept, two orders).

---

## Part B — slugg (placeholders / fabricated quantities)

**House posture: HONEST.** The codebase consistently returns `-Inf` weight on a non-PD grid cell,
`NA_real_` on unavailable uncertainty, or fails loudly via `stop()`. No fabricated reported quantity
was found.

### Findings

| Sev | Location | What it is | Verdict |
|-----|----------|-----------|---------|
| **S2** | `R/nested_laplace_re_cov.R:704` (and `R/re_cov_gibbs.R:231`) | On a non-PD / unusable numerical Hessian, the **outer posterior covariance of theta falls back to `diag(0.5^2, k)`** (resp. `diag(p)`) **silently** | This scale places the CCD/grid integration nodes (resp. shapes the Gibbs MH proposal); it is **not** a reported quantity, and the outer Pareto-k diagnostic flags a misfit proposal. Low statistical impact, but the fallback emits no warning — a wrong scale degrades integration silently. **Recommend:** `warning()` on the fallback so a numerically-bad fit is visible. |
| OK | `R/mala.R:141` `grad_prop <- grad_curr # placeholder` | reused gradient on the rejected branch | benign: reached only when `log_p_prop` non-finite ⇒ `log_alpha=-Inf` ⇒ `accepted` always FALSE, so `grad_prop` never propagates. Dead-store for variable definedness. |
| OK | `src/log_post_generic_impl.h:125` `compute_spde_hyper_prior` "stub returns 0 when joint_hypers==false" | conditional return | by-design: the SPDE PC hyper-prior is owned by the outer nested/CCD integration in the non-joint path; it is added there, not dropped. |
| OK | `R/spatiotemporal.R:352` `s_idx <- rep(1L, N) # placeholder` | structural pseudo-index | C++ uses `Phi` instead; `s_idx` unused on that path. |
| OK | `R/fit_spde_nuts.R:172` range/sigma numeric placeholders | pass-through scaffolding | overwritten by the Rcpp kernel. |
| GATE | `R/tulpa.R` (multiple), `R/em_laplace.R:210`, `R/fit_laplace.R:{493,508,518}`, `R/nested_laplace.R:1425` | `stop("... not yet supported / routed through tulpa()")` | absent-not-faked; loud gates. The front-door wiring gaps are documented, not silently approximated. |

No test would catch the S2 fallback (recovery sims use well-conditioned Hessians), consistent with
slugg's note that simplified simulations cannot exercise the numerical-failure branch.

---

## Fixes applied (2026-06-01)
- P1: added `^experimental$` to `.Rbuildignore` — `experimental/` no longer ships in the tarball.
- S2: `nested_laplace_re_cov.R:704` and `re_cov_gibbs.R:231` now `warning()` on the
  Hessian-failure diagonal fallback instead of falling back silently. Parse-verified.
