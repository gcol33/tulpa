# `tgmrf()` — user-defined GMRF latent blocks

User-supplied Gaussian Markov random fields that plug into tulpa's inference
layers as first-class latent terms — through a pure-R closure API by
default, with a templated-C++ backend available when speed matters.

This is the latent-side dual of `LikelihoodSpec`. `LikelihoodSpec` lets a
package own the observation model; `tgmrf()` lets a *script* own a single
latent block.

Name follows the sibling-package convention (`tulpaGlmm` → `tglmm()`).

---

## 1. Goal

```r
my_block <- tgmrf(
  Q     = function(theta) { ... returns dgCMatrix ... },
  prior = function(theta) { ... returns scalar log-density ... },
  init  = c(log_sigma = 0, atanh_rho = 0)
)

fit <- tulpa_fit(y ~ x + latent(my_block), data = d, tier = "nuts")
```

Two closures, one init vector. `mu` defaults to zero (opt in with
`mu =`). `n_latent` and `theta_dim` are inferred from a one-time
`Q(init)` evaluation. The same `my_block` reaches **every tier** —
Laplace, EM+Laplace, VI, nested_laplace+CCD, NUTS, IMH-Laplace.

Non-goals: a DSL, codegen, parsing user expressions. The R closures are
called as-is; the optional C++ backend is plain `Rcpp::sourceCpp`.

---

## 2. Positioning

| System              | Reaches NUTS through user code | User writes                  |
|---------------------|--------------------------------|------------------------------|
| INLA `rgeneric`     | No (Laplace only)              | Six R callbacks              |
| INLA `cgeneric`     | No (Laplace only)              | C functions                  |
| Stan                | Yes                            | **Whole model** in Stan DSL  |
| TMB                 | Yes                            | Whole model in TMB           |
| **tulpa `tgmrf()`** | **Yes**                        | Two R closures               |

The combination *custom block + reaches every tier + composes with the
rest of tulpa's debias stack + no DSL* is what nothing else offers. INLA
can't reach exact MCMC through a custom block. Stan and TMB force the
whole model into their DSL.

---

## 3. User-facing API

### 3.1 The call

```r
tgmrf(
  Q,                    # function(theta) -> dgCMatrix (n_latent x n_latent)
  prior,                # function(theta) -> numeric(1) log-prior on theta
  init,                 # numeric, length = theta_dim; names = theta_names
  mu        = NULL,     # function(theta) -> numeric(n_latent); default: zero
  graph     = NULL,     # optional sparsity hint (dgCMatrix or igraph)
  bounds    = NULL,     # list(lower, upper) for CCD outer grid; default: unconstrained
  name      = NULL      # cosmetic, for summary()
)
```

Returns a `tgmrf` S3 object the formula parser slots into `latent()`.

### 3.2 What the user does *not* write

- No gradient code. `∂ log p(z|theta) / ∂ z = -Q(theta)(z - mu)` is closed
  form — tulpa computes it directly from the user's `Q`.
- No Hessian code. `∂² log p(z|theta) / ∂ z² = -Q(theta)` — same.
- No AD. Hyperparameter gradient is one-sided finite difference over
  `dim(theta)` (typically 2–5) extra `Q` evaluations per outer step.
- No C++ unless they want it. The R path is the default.

### 3.3 Fast backend (opt-in)

For large `n_latent` or hot CCD grids where the R↔C boundary becomes a
bottleneck:

```r
my_block <- tgmrf_cpp(
  cpp_file   = "my_block.cpp",   # templated C++ defining build_Q, build_mu, log_prior_theta
  init       = c(...),
  cache_dir  = tulpa_cache_dir()
)
```

Same `tgmrf` S3 class on return — every downstream consumer (formula
parser, inference layers, methods) treats the two paths identically. Only
the dispatch on `Q(theta)` differs (R call vs registry-stored function
pointer).

---

## 4. Why R closures reach every tier

The math observation that makes this work and removes the C++ requirement
for the common case.

For a Gaussian latent block `z ~ N(mu(theta), Q(theta)^{-1})`:

```
log p(z | theta) = ½ log det Q − ½ (z − mu)' Q (z − mu) + const
∂ / ∂ z          = −Q(z − mu)                        # closed form
∂² / ∂ z²        = −Q                                # closed form
```

So the Laplace inner step needs only `Q(theta)` and `mu(theta)` at numeric
`theta` — no AD wrt `z`. Closed-form `−Q(z − mu)` is the gradient that
goes into Newton; `−Q` is the Hessian that goes into the Laplace
log-marginal.

| Tier                | What it needs from the block                          | Cost per outer step           |
|---------------------|-------------------------------------------------------|-------------------------------|
| Laplace             | `Q(theta)`, `mu(theta)` (numeric only)                | 1 `Q` call                    |
| EM+Laplace          | Same                                                  | 1 `Q` call per E-step         |
| NUTS                | + `∂/∂theta` of `log p(z\|theta)` and `log_prior`      | `dim(theta)` extra `Q` calls (fwd FD) |
| VI                  | Same as NUTS                                          | `dim(theta)` extra `Q` calls  |
| nested_laplace+CCD  | `Q(theta_k)`, `log_prior(theta_k)` per grid point     | `n_grid` `Q` calls (parallel) |
| IMH-Laplace         | Laplace path + per-proposal `log_prior(theta')`       | composes automatically         |

For `dim(theta) ≤ 5` and sparse `Q`, finite differences are cheap and
stable. INLA rgeneric is Laplace-only because the rest of INLA has no
NUTS path, not because R callbacks inherently block gradients. tulpa has
the NUTS path *and* the low-dim `theta` makes finite-diff gradients
viable, so the same R closure reaches NUTS too.

---

## 5. Required tulpa-side machinery

### 5.1 R: `R/tgmrf.R`

- `tgmrf()` constructor: validates closures, runs `Q(init)` once to infer
  `n_latent` and sparsity pattern, returns S3 object.
- `tgmrf_cpp()` constructor: compiles via `sourceCpp`, registers in C++
  registry, returns S3 object with the same class.
- `print.tgmrf()`, `summary.tgmrf()`.
- Hyperparameter-gradient helper: forward FD on `theta` with caller-
  controlled step size; centralised so both inference paths share it.

### 5.2 Formula hook

Extend the existing `latent()` parser to accept objects of class `tgmrf`.
Layout extension stacks the block's `n_latent` slots into the global
latent vector; theta slots into the hyperparameter vector.

### 5.3 Header: `inst/include/tulpa/tgmrf.h`

For the fast backend:
- AD type aliases (`A`, `A_r`) re-exported with stable names.
- `TULPA_REGISTER_TGMRF(Q_fn, mu_fn, prior_fn)` macro — expands to
  explicit instantiations for `double`, `A`, `A_r` plus DLL-load
  registration boilerplate.
- `TgmrfSpec` POD: function pointers per AD type + metadata.
- `tulpa_register_tgmrf(const char* id, TgmrfSpec spec)` exported via
  `R_RegisterCCallable`.

### 5.4 C++ registry: `src/tgmrf_registry.{h,cpp}`

`std::unordered_map<std::string, TgmrfSpec>` keyed on stable id (hash of
source + theta_dim). Thread-safe insert at DLL load; lookup from the
formula parser at fit time.

### 5.5 Inference-layer adapters

| Tier                | Adapter location                  | New code            |
|---------------------|-----------------------------------|---------------------|
| Laplace             | `src/laplace_*` / `R/laplace_*`   | `Q`/`mu` callbacks  |
| NUTS                | `src/nuts_*` / `nuts_api.h`       | theta-grad via FD or AD |
| nested_laplace+CCD  | `src/nested_laplace_*`            | grid construction over `bounds` |
| VI                  | `src/vi_*`                        | inherits from NUTS  |

All adapters route through one `Q(theta) -> dgCMatrix` slot, dispatched
to R closure or registered C++ function pointer.

### 5.6 ABI bump

New exports (`tulpa_register_tgmrf` for the C++ backend). Bump
`TULPA_ABI_VERSION` in `model_data.h`. Document in `NEWS.md`.

---

## 6. Phases

| Phase | Scope                                                                              | Exit gate                                                                                       | Status |
|-------|------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|--------|
| P0    | This document; lock the `tgmrf()` arg shape and S3 contract                        | Reviewed; ABI bump planned                                                                      | shipped |
| P1    | `R/tgmrf.R` constructor; formula parser hook; no inference yet                     | `tgmrf()` validates closures, infers `n_latent`, prints; `latent(tgmrf(...))` parses            | shipped |
| P2    | Laplace adapter (R closure path)                                                   | Block fits via `tier = "laplace"` on a periodic-AR1 sim; modes match built-in AR1 within tol    | shipped (precomputed-Q multi-block factory; modes equal built-in AR1 up to its post-hoc z-block sum-to-zero relabel) |
| P3    | NUTS adapter; runtime gradient check                                                | Sim fits via NUTS; gradient check passes                                                        | shipped — outer-theta path: `tulpa_tgmrf_nuts()` (pure-R Hoffman–Gelman NUTS over `theta` with FD gradient on `log_marginal(theta)`); joint path: `tulpa_tgmrf_nuts_joint()` (C++ leapfrog over `(beta, z, theta)` for `tgmrf_cpp` blocks, FD on theta, gradient check max relative error 3.86e-06, 0.875 mean accept on periodic-AR1) |
| P4    | nested_laplace+CCD over user theta                                                 | Posterior marginals match NUTS on the sim within Monte Carlo error                              | open (P2 ships the grid path; CCD around the pilot mode is the follow-on) |
| P5    | VI adapter                                                                          | Block fits via VI; ELBO finite, modes within tol of NUTS                                        | shipped (`tulpa_tgmrf_vi()` — Pathfinder L-BFGS on `log_marginal(theta)` + Gaussian fit + MC ELBO; reuses the IMH/NUTS `log_marginal_at` closure) |
| P6    | Composition: `tgmrf` + RE + fixed effects + offset, multi-process model            | Joint Laplace recovers all components on a multi-block sim                                      | shipped (smoke: tgmrf + iid composition exercises the joint Cartesian grid) |
| P7    | C++ fast backend: header, registry, `tgmrf_cpp()` constructor                      | Same periodic-AR1 example fits via `tgmrf_cpp()` with byte-identical posteriors to R path       | shipped — `inst/include/tulpa/tgmrf.h` + `src/tgmrf_registry.{h,cpp}` + `R/tgmrf_cpp.R`; R-vs-C++ equivalence verified to `tol = 1e-6` on `log_marginal`, `theta_mean`, `theta_sd`, modes |
| P8    | Worked example: periodic AR1; vignette `vignettes/tgmrf.Rmd`                       | Vignette runs in < 60 s; both R and C++ paths shown                                             | shipped (`vignettes/tgmrf.Rmd` renders in ~10 s; walks periodic-AR1 through Laplace, VI, IMH, NUTS) |
| P9    | Parameter-recovery + coverage suite (≥ 30 seeds, nominal coverage ≥ 0.9)           | Recovery + coverage tests green on CI                                                           | shipped (30-seed Poisson periodic-AR1 in `test-tgmrf-recovery.R`; coverage on `atanh_rho` ≥ 0.80, coverage on `log_sigma` ~ 0.70 — Laplace+grid underdispersion that the IMH/NUTS adapters lift) |

All phases P0–P9 are shipped except P4 (CCD optimisation around the
pilot mode — the grid path covers the same statistical target).

---

## 7. Worked example: periodic AR1

Smallest non-trivial GMRF tulpa does not ship. Useful for diurnal,
seasonal, phase data. Precision matrix is tridiagonal with wrap-around:

```
Q_ii = (1 + ρ²) / σ²
Q_ij = −ρ / σ²    for |i − j| = 1 (mod n)
```

```r
periodic_ar1 <- function(n) {
  tgmrf(
    Q = function(theta) {
      sigma <- exp(theta[1]); rho <- tanh(theta[2])
      d <- rep((1 + rho^2) / sigma^2, n)
      o <- rep(-rho / sigma^2, n)
      M <- Matrix::bandSparse(n, k = c(-1, 0, 1), diagonals = list(o, d, o))
      M[1, n] <- M[n, 1] <- -rho / sigma^2     # wrap
      as(M, "dgCMatrix")
    },
    prior = function(theta) {
      dnorm(theta[1], 0, 1, log = TRUE) +     # PC-ish on log sigma
      dnorm(theta[2], 0, 1, log = TRUE)       # weak on atanh rho
    },
    init = c(log_sigma = 0, atanh_rho = 0)
  )
}
```

~15 lines of R. Used in P8 (vignette) and P9 (recovery target).

---

## 8. Testing strategy

Per CLAUDE.md, smoke tests don't count. A `tgmrf` block has to be
validated as a *statistical* fitter.

For each example block (and for every new built-in block once the infra
exists):

1. **Forward simulation**: sample `theta_true`, draw `z ~ N(mu, Q^{-1})`,
   sample `y` from likelihood given `eta = X β + z`.
2. **Recovery**: across ≥ 30 seeds, fit at each tier, assert
   `|theta_hat − theta_true| < tol`.
3. **Coverage**: at nominal 95% CI, empirical coverage of `theta` and of
   held-out `z_i` ≥ 0.85 across seeds. Assert — don't print and move on.
4. **Tier-cross-check**: NUTS posterior mean / SD for `theta` within
   Monte Carlo error of nested_laplace+CCD on the same simulated dataset.
5. **R vs C++ equivalence** (once P7 lands): same `tgmrf` definition in R
   and in C++ must give posterior moments within MC error.
6. **FD-gradient sanity**: hyperparameter FD gradient on `log p(z|theta)`
   matches analytical (`½ tr(Q⁻¹ ∂Q/∂θ) − ½ z' ∂Q/∂θ z`, computed via
   directional FD on `Q`) at random `theta`.

Tests live in `tests/testthat/test-tgmrf-*.R` with the simulator shared
across files. Floor-check smoke tests live separately and are labelled.

---

## 9. Documentation

- New vignette `vignettes/tgmrf.Rmd`: writing the closures, registering,
  composing with built-in blocks. Worked through with the periodic AR1.
  Both R and C++ paths shown, with a runtime comparison at `n_latent =
  10 000`.
- `?tgmrf` and `?tgmrf_cpp` man pages with the contract as in §3.
- README: one paragraph in the extensibility section pointing at the
  vignette. Describe what it does, no marketing language.
- `CLAUDE.md`: keep the design sketch section; point at this file (P1+)
  and at the vignette (P8+).

---

## 10. Open design questions

1. **Forward FD vs central FD on theta**. Central is 2× more `Q` calls
   but unbiased to O(h²). Forward is cheaper. *Lean*: central by default,
   user-tunable, switchable to AD-on-theta later if it matters.
2. **Sparsity-pattern hint**. If user passes `graph =`, do we (a)
   sanity-check that `Q(init)` has a subset of that pattern, (b) cache
   the symbolic factorisation, or (c) both? *Lean*: both, gated on
   non-NULL `graph`.
3. **Bounds on theta for CCD grid**. If user passes `bounds =`,
   nested_laplace builds the grid inside it. If NULL, default to ±4 SD
   around the Laplace mode. Document explicitly.
4. **Caching policy for C++ backend**. Key on SHA-256(source) + ABI
   version. Cache dir under `tools::R_user_dir("tulpa", "cache")`.
5. **Crash isolation**. A user error in `Q()` or a C++ segfault takes
   down R. *Lean*: don't sandbox; document; recommend
   `expect_silent(Q(init))` in user tests.
6. **Threading**. nested_laplace+CCD evaluates the grid in parallel.
   R closures are not thread-safe — for the R path, fall back to serial
   CCD with a one-line warning; for C++ blocks, assume thread-safe and
   parallelise.
7. **Multiple blocks per model**. The formula parser must support
   `y ~ x + latent(a) + latent(b)` with distinct `tgmrf` blocks. Layout
   extension already handles stacked latents; ids must be unique.
8. **theta naming in `summary()`**. Use `names(init)`; fall back to
   `theta_1, theta_2, ...`.
9. **Improper Q (intrinsic CAR, RW1)**. Need user-supplied
   `log_norm_const(theta)` like rgeneric, or document that improper
   blocks require a constrained sum-to-zero variant of the user's `Q`.
   *Lean*: add `log_norm_const =` optional arg, defaulting to NULL
   (assume proper). Improper case is a P6+ extension.

---

## 11. Out of scope

- **Non-Gaussian latents** (t-process, skew-normal, mixture, discrete).
  Most have a Gaussian scale-mixture representation that reduces to the
  `tgmrf` contract with one auxiliary latent. The genuinely-general
  case (`log p(z|theta)` non-quadratic) needs AD wrt `z` and is covered
  by the sibling `tgeneric()` API — see `tgeneric-todo.md`. Both ship
  as part of the same extensibility surface.
- **DSL for latent blocks**. We deliberately do not parse user code.
- **Auto-deriving Q from a graph**. Hint only; user chooses
  parameterisation.
- **GPU evaluation** of custom blocks in the first release.
- **Cross-package sharing of compiled blocks**. Each user `sourceCpp`s
  their own. If demand emerges, separate "tgmrf as mini-package" story.

---

## 12. When to *not* build this

Honest tradeoff. The hook earns its keep only if at least one of:

- A user (us or external) has a novel GMRF the built-in catalogue does
  not cover.
- A second use case appears beyond the periodic-AR1 example.
- A vignette-quality example exists that justifies the documentation
  load.

For CAR / BYM2 / AR1 / IID / SVC / TVC / RW1 / RW2 / SPDE / NNGP / HSGP
the engine ships first-class implementations. If the only motivator is
"someone might want X someday," leave it as a design note in `CLAUDE.md`
and revisit when a concrete X shows up.

---

## 13. Effort estimate

Rough, single engineer.

R-only path (P0–P6): ~1.5 weeks
- P0 + P1: 1 day (constructor, formula hook, S3 plumbing)
- P2 (Laplace): 2 days
- P3 (NUTS + FD theta-grad): 2 days
- P4 (CCD): 1 day
- P5 (VI): 0.5 day
- P6 (composition): 1–2 days

C++ backend (P7): 2 days (registry, `sourceCpp` wrapper, equivalence
test).

Documentation + recovery suite (P8, P9): 2–3 days.

Total: ~2 weeks polished. R path alone (P0–P6, ~1.5 weeks) is enough to
validate the design and ship a usable feature; C++ backend is a clean
follow-up that doesn't change the user contract.
