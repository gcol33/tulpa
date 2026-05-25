# `tgeneric()` — user-defined non-Gaussian latent blocks

Companion to [`tgmrf()`](generic-todo.md) for latents whose log-density
is not quadratic in `z`. User supplies a templated/closure log-density
`log_p_z(z, theta)`; tulpa handles gradients (analytically where it can,
AD where it can't) and routes through every inference tier.

Many practical "non-Gaussian latents" *can* be reformulated as a
Gaussian scale-mixture and are better served by `tgmrf()` + one
auxiliary latent — §12 gives the decision flowchart users should see in
the vignette. `tgeneric()` covers everything else and ships alongside
`tgmrf()` as part of the same extensibility surface.

Name follows the sibling-package convention (`tulpaGlmm` → `tglmm()`).

---

## 1. Goal

```r
my_block <- tgeneric(
  log_p_z = function(z, theta) { ... returns scalar ... },
  prior   = function(theta) { ... returns scalar ... },
  init_z  = numeric(n_latent),         # starting value for z
  init    = c(...),                    # starting value for theta
  graph   = NULL                       # Hessian-of-log_p_z sparsity hint
)

fit <- tulpa_fit(y ~ x + latent(my_block), data = d, tier = "nuts")
```

Three closures total (one more than `tgmrf` because `log_p_z` is no longer
recoverable from `Q`). `n_latent` inferred from `length(init_z)`;
`theta_dim` from `length(init)`. Same `latent()` formula slot, same
downstream consumers.

---

## 2. Positioning

| System              | Non-Gaussian latent through user code | User writes |
|---------------------|---------------------------------------|-------------|
| INLA `rgeneric`     | **No** — GMRF latent only             | n/a         |
| INLA `cgeneric`     | **No** — GMRF latent only             | n/a         |
| Stan                | Yes                                   | Whole model in Stan DSL |
| TMB                 | Yes                                   | Whole model in TMB |
| **tulpa `tgeneric`**| Yes                                   | One closure + prior + init |

INLA can't do this at all — the latent field assumption is hardwired.
Stan and TMB can, but force the whole model into their DSL.
`tgeneric()` puts the user-defined log-density into a single slot of a
formula and lets the rest of tulpa stay as-is.

---

## 3. User-facing API

### 3.1 The call

```r
tgeneric(
  log_p_z,             # function(z, theta) -> numeric(1)
  prior,               # function(theta) -> numeric(1) log-prior on theta
  init_z,              # numeric(n_latent); starting value for z mode-finding
  init,                # numeric(theta_dim); names = theta_names
  grad_z   = NULL,     # optional analytical d log_p_z / d z; AD/numDeriv if NULL
  hess_z   = NULL,     # optional analytical d² log_p_z / d z²; AD/numDeriv if NULL
  graph    = NULL,     # Hessian sparsity pattern (dgCMatrix); strongly recommended
  bounds   = NULL,     # list(lower, upper) for CCD outer grid
  name     = NULL
)
```

Returns a `tgeneric` S3 object the formula parser slots into `latent()`.

### 3.2 What is and isn't optional

| Argument        | Optional? | Cost if omitted                                    |
|-----------------|-----------|----------------------------------------------------|
| `log_p_z`       | No        | —                                                  |
| `prior`         | No        | —                                                  |
| `init_z`        | No        | Mode-finder needs a start                          |
| `init`          | No        | Same for theta                                     |
| `grad_z`        | Yes       | Fall back to numDeriv (R) / AD (C++)               |
| `hess_z`        | Yes       | Fall back to numDeriv (R) / AD (C++); **expensive** without `graph` |
| `graph`         | Yes but strongly recommended | Hessian assumed dense → O(n_latent²) per step |

`grad_z` and `hess_z` analytical overrides are the user's escape hatch
when numerical differentiation is too slow or unstable. If the user
provides both, the engine never differentiates `log_p_z` numerically.

### 3.3 Fast backend (opt-in)

```r
my_block <- tgeneric_cpp(
  cpp_file = "my_block.cpp",   # templated log_p_z, prior; grad/Hess via AD
  init_z   = numeric(n_latent),
  init     = c(...),
  graph    = NULL
)
```

Same `tgeneric` S3 class on return — formula parser, inference layers
and methods treat the two paths identically. C++ AD removes the per-call
R↔C boundary and makes the Hessian assembly tractable at scale.

---

## 4. Why this is harder than `tgmrf`

`tgmrf` is cheap because the latent log-density is quadratic in `z`,
so `∂/∂z = −Q(z − μ)` and `∂²/∂z² = −Q` come for free from the user's
`Q(theta)`. `tgeneric` has no such structure.

| Cost                    | `tgmrf`                  | `tgeneric` (R)                       | `tgeneric` (C++ AD)     |
|-------------------------|--------------------------|--------------------------------------|-------------------------|
| Gradient wrt `z`        | closed form (`−Q(z−μ)`)  | `numDeriv::grad`, O(n_latent) calls  | reverse-mode, O(1) pass |
| Hessian wrt `z`         | closed form (`−Q`)       | `numDeriv::hessian`, O(n_latent²) calls | reverse-mode-on-forward, O(nnz) |
| Per-Laplace-step cost   | 1 `Q` call               | O(n_latent²) `log_p_z` calls         | O(1) AD pass            |

For the R path this means **`tgeneric` is viable up to `n_latent ≈ 100`**
without `graph`/`hess_z`, and up to `~1000` with sparsity. C++ AD
removes the wall.

The Hessian sparsity hint (`graph =`) is the single most important
optimisation — without it, numDeriv computes `n_latent²` finite
differences. With it, only the nonzero pattern is probed.

---

## 5. Required tulpa-side machinery

### 5.1 R: `R/tgeneric.R`

- `tgeneric()` constructor: validates closures, runs `log_p_z(init_z,
  init)` once to confirm it returns a scalar, returns S3.
- Wrappers around `numDeriv::grad` / `numDeriv::hessian` that respect a
  user-supplied sparsity pattern (compute only nonzero entries).
- `tgeneric_cpp()` constructor: compiles via `sourceCpp`, registers
  function pointers, returns S3 with the same class.
- `print.tgeneric()`, `summary.tgeneric()`.

### 5.2 Formula hook

Extend `latent()` parser to accept `class(x) %in% c("tgmrf", "tgeneric")`.
Layout extension stacks `init_z` slots into the global latent vector.
Inference dispatch reads class.

### 5.3 Header: `inst/include/tulpa/tgeneric.h`

For the fast backend:
- AD type aliases.
- `TULPA_REGISTER_TGENERIC(log_p_z_fn, prior_fn)` macro — instantiates
  `double`, `A`, `A_r` and registers.
- `TgenericSpec` POD: function-pointer slots per AD type + sparsity
  pattern hook.

### 5.4 C++ registry: `src/tgeneric_registry.{h,cpp}`

Same structure as the `tgmrf` registry; separate map so the formula
parser can dispatch without a string check.

### 5.5 Inference adapters

| Tier               | What's needed                                       | Difficulty                            |
|--------------------|-----------------------------------------------------|---------------------------------------|
| Laplace            | `log_p_z`, `∂/∂z`, `∂²/∂z²` at numeric `theta`      | Newton on a non-quadratic objective; may need damping / line search |
| EM+Laplace         | Same                                                | Same                                  |
| NUTS               | + `∂/∂theta` of `log_p_z` and `log_prior`           | Same as `tgmrf` (FD or AD on theta)   |
| nested_laplace+CCD | `log_marginal(theta_k)` per grid point              | Inner Laplace per grid point — pricier than `tgmrf` |
| VI                 | Same as NUTS                                        | Same                                  |
| IMH-Laplace        | Laplace path + per-proposal `log_prior(theta')`     | Composes                              |

The Laplace inner step is where the bulk of the new code lives — it
needs convergence diagnostics that `tgmrf`'s pure-Newton Laplace doesn't,
because the user's `log_p_z` may not be log-concave in `z`.

### 5.6 ABI bump

New exports (`tulpa_register_tgeneric`). Bump `TULPA_ABI_VERSION`.

---

## 6. Phases

Strict ordering — `tgmrf` must ship and stabilise first, because
`tgeneric` reuses its inference adapters, registry pattern, and S3
plumbing.

| Phase | Scope                                                                              | Exit gate                                                          |
|-------|------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| Q0    | `tgmrf` shipped through its P6 (composition).                                      | `tgmrf` recovery suite green.                                      |
| Q1    | `R/tgeneric.R` constructor + numDeriv-based grad/Hess + sparsity-aware probing     | Validates `log_p_z(init_z, init)`; sparse Hessian on a 3-node toy is correct vs analytical |
| Q2    | Laplace adapter with damped Newton + line search                                   | Fits an IID Student-t latent (n_latent = 50) on a sim; modes match closed-form Student-t MAP |
| Q3    | NUTS adapter; FD on theta                                                          | Same Student-t sim fits via NUTS; posterior moments match Stan on the same model |
| Q4    | C++ fast backend: header, registry, `tgeneric_cpp()` constructor                   | Same sim fits via `tgeneric_cpp()` with byte-identical posteriors to R path |
| Q5    | nested_laplace+CCD over user theta                                                 | Posterior marginals match NUTS within MC error                     |
| Q6    | Worked example: **monotone latent coefficients** for dose-response splines        | Vignette runs in < 90 s; example fits real-ish data                |
| Q7    | Parameter-recovery + coverage suite (≥ 30 seeds, nominal coverage ≥ 0.9)           | Recovery + coverage green on CI                                    |

Phases Q1–Q3 are the critical path. Q4 is the scaling unlock. Q5–Q7 are
proof and documentation.

---

## 7. Worked example: monotone latent coefficients

The cleanest "doesn't reformulate" case. Coefficients `z_1 ≤ z_2 ≤ ... ≤
z_K` parameterise a monotone dose-response curve. The latent has a hard
ordering constraint, encoded via softplus increments:

```r
monotone_block <- function(K) {
  tgeneric(
    log_p_z = function(z, theta) {
      sigma <- exp(theta[1])
      delta <- diff(z)                        # non-negative by construction below
      sum(dexp(delta, rate = 1/sigma, log = TRUE)) - log(sigma)
    },
    prior = function(theta) dnorm(theta[1], 0, 1, log = TRUE),
    init_z = sort(rnorm(K)),
    init   = c(log_sigma = 0)
  )
}
```

Why this example:

- Genuinely non-Gaussian (exponential increments).
- No clean Gaussian scale-mixture reformulation.
- Practical: monotone splines, IRT, dose-response, ROC curves.
- `n_latent = K` is typically 5–50 — sits in the R-path sweet spot.

Used in Q6 (vignette) and Q7 (recovery).

---

## 8. Testing strategy

Same standard as `tgmrf` per CLAUDE.md. Smoke tests don't count.

1. **Forward simulation** of `(theta_true, z, y)`.
2. **Recovery** across ≥ 30 seeds: `|theta_hat − theta_true| < tol`.
3. **Coverage**: empirical 95% CI coverage of `theta` and held-out `z_i`
   ≥ 0.85.
4. **Tier-cross-check**: NUTS vs nested_laplace+CCD posterior moments
   within MC error.
5. **R vs C++ equivalence** (Q4+).
6. **Hessian-sparsity correctness**: user-supplied `graph` must agree
   with `numDeriv::hessian(log_p_z, ...)` nonzero pattern at random `z`.
7. **Stan cross-check**: for the worked example, posterior moments within
   MC error of a hand-written Stan implementation.

Tests live in `tests/testthat/test-tgeneric-*.R`. Labelled smoke tests
separately.

---

## 9. Documentation

- New vignette `vignettes/tgeneric.Rmd`: writing `log_p_z`, choosing
  whether to provide `grad_z`/`hess_z`, the sparsity hint and why it
  matters, when to use C++ backend. Worked through with the monotone
  example. Includes a **decision flowchart**: "Before writing tgeneric,
  can your latent be reformulated as Gaussian scale-mixture? If yes,
  use tgmrf + aux variable instead."
- `?tgeneric` and `?tgeneric_cpp` man pages.
- `vignettes/tgmrf.Rmd` cross-link explaining when to graduate from
  `tgmrf` to `tgeneric`.
- README: one paragraph in extensibility section. Be explicit that
  `tgeneric` is for residual cases.
- `CLAUDE.md`: point at this file (Q1+) and the vignette (Q6+).

---

## 10. Open design questions

1. **Non-log-concave `log_p_z`**. The Laplace mode is no longer unique;
   Newton can fail. *Lean*: damped Newton with multi-start, document the
   risk, warn if Hessian has negative eigenvalues at the mode.
2. **R-side AD instead of numDeriv?** `madness`, `Deriv`, `numDeriv`
   itself. *Lean*: numDeriv (most stable, no dep churn); revisit if
   stability bites.
3. **Sparsity-pattern enforcement**. If user passes `graph =`, do we
   trust them, or verify at registration? *Lean*: verify at
   registration via one random `log_p_z` Hessian probe; fail loudly on
   pattern mismatch.
4. **Constrained latents**. Monotonicity, sum-to-K, positivity. Should
   `tgeneric` expose a transform layer (user writes log_p_z on
   unconstrained space; engine handles Jacobian)? *Lean*: no — keep
   `tgeneric` minimal; transforms are a user concern. Document the
   monotone example as the canonical pattern.
5. **Crash isolation**. Same as `tgmrf` — accept R/C++ segfaults, document.
6. **Discrete latents**. Out of scope for `tgeneric`. Marginalise in
   `log_lik` or build a dedicated HMM/forward-backward path.
7. **Multi-block composition with `tgmrf`**. The formula must allow
   `y ~ x + latent(a_tgmrf) + latent(b_tgeneric)`. Layout extension
   already stacks; the inference layer dispatches per block class.
8. **Caching for C++ backend**. Same scheme as `tgmrf_cpp`.

---

## 11. Out of scope

- **Discrete latents** (HMM states, mixture components, ordinal RE).
  Separate path; needs forward-backward or label augmentation.
- **Constrained-latent transform layer**. User handles via reparameter-
  isation; we don't ship a built-in.
- **GPU evaluation** of custom `log_p_z`.
- **Cross-package compiled-block sharing**.

---

## 12. User-facing scope guidance (decision flowchart for the vignette)

Before writing a `tgeneric` block, ask:

```
                  ┌─ Yes ─→ tgmrf() + ω auxiliary (one extra latent)
Heavy-tailed (t)?─┤
                  └─ No ─→ next
                       │
                       ↓
                  ┌─ Yes ─→ tgmrf() + skewness auxiliary
Skew-normal?    ──┤
                  └─ No ─→ next
                       │
                       ↓
                  ┌─ Yes ─→ tgmrf() + Pólya-gamma augmentation (already in tulpa)
Logistic-style? ──┤
                  └─ No ─→ next
                       │
                       ↓
                  ┌─ Yes ─→ tgmrf() + component label aux
Mixture?        ──┤
                  └─ No ─→ tgeneric()
```

Almost every practical non-Gaussian latent in spatial / temporal stats
takes the **left branch**. The right branch (genuine `tgeneric` use
cases) covers:

- Hard-constrained latents (monotone, simplex, manifold-valued).
- Repulsive / attractive point-process latents (Strauss).
- Self-exciting latents not expressible as a marked Poisson process.
- Latents with non-quadratic, non-mixable penalty structure.

If a user lands on `tgeneric` without ruling out the left branch, the
vignette should send them back. This is documentation guidance, not a
build gate — `tgeneric` ships regardless.

---

## 13. Effort estimate

Rough, single engineer, **assumes `tgmrf` already shipped through P6**.

R-only path (Q1–Q3): ~2 weeks
- Q1 (constructor + sparse numDeriv): 2–3 days
- Q2 (damped-Newton Laplace, line search, diagnostics): 4–5 days
- Q3 (NUTS adapter, theta FD): 2–3 days

C++ backend (Q4): 3 days (registry, sourceCpp wrapper, AD-on-z
plumbing, equivalence test).

CCD + composition (Q5): 1–2 days.

Documentation + worked example + recovery suite (Q6, Q7): 3–4 days.

Total: **~3 weeks polished**, of which ~2 weeks are the Laplace adapter
because non-log-concave `log_p_z` needs more care than `tgmrf`'s pure
quadratic case.

Materially more work than `tgmrf` (~2 weeks), driven by the Laplace
adapter for non-log-concave `log_p_z`. Both ship as part of the same
extensibility surface; sequencing only reflects the plumbing dependency.
