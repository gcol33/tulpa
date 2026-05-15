# next.md — (a.iii) joint NUTS over (log_kappa, log_tau, z, beta, log_phi)

Plan for the next session. Self-contained: assumes the reader has the
post-`49a30ad` repo state and nothing more.

## Goal

Flip `cpp_tulpa_fit_spde_nuts` from "fixed Matern hypers, sample
(beta, w_mesh, log_phi)" to "joint NUTS over (log_kappa, log_tau, z,
beta, log_phi)" using the non-centered transform `w = L^{-T} z` from
(a.ii). The reserved `log_kappa_spde_idx` / `log_tau_spde_idx` slots
in `ParamLayout` become valid; the `spde_w_start..end` block changes
semantics from `w` to `z`; the prior on the latent block becomes
unit Gaussian on `z`; PC priors land on `(range, sigma)` per Fuglstad
et al. 2019.

## What is already in place

Foundations from earlier commits:

- `inst/include/tulpa/param_layout.h:115-122` — reserved
  `log_kappa_spde_idx`, `log_tau_spde_idx` (sentinel -1 today),
  `spde_w_start..end` (currently semantics = `w`, will flip to `z`).
- `inst/include/tulpa/spde_model_data.h` — `SpdeModelData` carries the
  FEM matrices on `ModelData::spde_data`. Already exported.
- `src/spde_nc_transform.{h,cpp}` ((a.ii)) — `SpdeNcTransform` class
  with `forward`, `backward`, and `spde_nc_transform_arena` hook.
  Verified vs finite differences in `tests/testthat/test-spde-nc-transform.R`.
- `inst/include/tulpa/autodiff_arena.h` ((a.i)) —
  `Arena::add_custom_backward` for variadic-IO AD. ABI 17.
- `src/hmc_param_layout.cpp:489-502` — current SPDE layout block
  (allocates `spde_w_start..end`, leaves hyper slots at -1).
- `src/tulpa_priors_spde.h::compute_spde_prior` — current centered
  prior `-0.5 w' Q w` on fixed Q.
- `src/log_post_generic_impl.h:285-300` —
  `add_generic_spatial_effect` accumulates `eta_i += sum_j A_ij * w_j`
  from `state.spde_w` (filled by `compute_spde_prior` from the params
  vector).

## Implementation plan

Six steps. Each ends in a green build + commit so a parallel writer
collision (see end-of-session notes for `9d7ded21`) costs at most one
step's work.

### Step 1 — Allocate the hyper slots in `ParamLayout`

File: `src/hmc_param_layout.cpp`.

When `data.spatial_type == SpatialType::SPDE` and `data.has_spde`,
set `log_kappa_spde_idx = idx; idx++; log_tau_spde_idx = idx; idx++;`
right after the existing `spde_w_start..end` block. Keep the sentinel
-1 path for the inactive case.

Also update the comment on `inst/include/tulpa/param_layout.h:115-122`
to describe the now-active slots.

Add a unit test in `tests/testthat/test-spatial-spde-api.R` (or a new
file) that builds an SPDE-typed `ModelData`, calls `compute_param_layout`,
and asserts `log_kappa_spde_idx >= 0`, `log_tau_spde_idx >= 0`, and
the relative ordering `spde_w_end <= log_kappa_spde_idx <
log_tau_spde_idx`.

### Step 2 — Switch the prior to non-centered

File: `src/tulpa_priors_spde.h`.

Rewrite `compute_spde_prior<T>(params, data, layout, spde_w_out)`:

- Reinterpret `params[spde_w_start .. spde_w_end)` as `z`, not `w`.
- Return `-0.5 sum_j z_j^2` (pure unit Gaussian — no `Q` involved).
- Stop populating `spde_w_out` here. The downstream eta path now owns
  the `z -> w` transform and will produce `w` itself.

Why no `log|Q|` term: the change of variable `w = L^{-T} z` cancels
exactly. In `z`-space the prior is N(0, I) and the Jacobian of the
forward direction is constant in `(z, theta)` *given that the eta
likelihood sees `w(z, theta)` directly* — the absorbing factor lives
inside the implicit function theorem result that (a.ii) already
encodes in the adjoint. No explicit log-det in the structured HMC
log-post.

Add a new helper for the hyper prior in the same file:

```cpp
template<typename T>
T compute_spde_hyper_prior(
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout)
{
    if (!layout.is_spde || !data.has_spde) return T(0.0);
    const auto& s = data.spde_data;
    T log_kappa = params[layout.log_kappa_spde_idx];
    T log_tau   = params[layout.log_tau_spde_idx];
    // PC prior on (range, sigma) from Fuglstad et al. 2019, expressed
    // in (log_kappa, log_tau) via the change of variable below.
    // ... implementation in step 4.
    return T(0.0);   // stub here; step 4 fills in.
}
```

Wire `compute_spde_hyper_prior` into the structured HMC path in
`src/log_post_generic_impl.h::initialize_generic_state`, right after
the existing `compute_spde_prior` call.

### Step 3 — Wire the NC transform into the eta path

File: `src/log_post_generic_impl.h`.

Currently `add_generic_spatial_effect` reads `state.spde_w` and sums
`A_ij * w_j` into `eta_i`. After step 2, `state.spde_w` is empty (the
prior no longer populates it). We need a new step that, before the
eta loop, computes `w = L^{-T}(theta) z` and stashes it on
`state.spde_w`.

Sketch (pseudocode):

```cpp
// In initialize_generic_state, after compute_spde_prior:
if (layout.is_spde && data.has_spde) {
    const int n_mesh = data.spde_data.n_mesh;
    state.spde_w.resize(n_mesh);

    // Pull z from params, compute w via the NC transform.
    std::vector<T> z(n_mesh);
    for (int j = 0; j < n_mesh; j++) z[j] = params[layout.spde_w_start + j];
    T log_kappa = params[layout.log_kappa_spde_idx];
    T log_tau   = params[layout.log_tau_spde_idx];

    // Branch on T to call the right transform path.
    apply_spde_nc_transform<T>(state.spde_w, z, log_kappa, log_tau, data);
}
```

`apply_spde_nc_transform<T>` is the integration headache. Three
specializations:

1. `T = double`: pure forward pass. Build a `SpdeNcTransform` (or look
   one up from a thread-local cache keyed on `&data.spde_data`), call
   `transform.forward(z, exp(log_kappa), exp(log_tau))`, copy result.

2. `T = arena::Var`: use `spde_nc_transform_arena(ar, z_vars, log_kappa,
   log_tau, transform)`. The `transform` reference must outlive the
   arena's backward sweep, so the cache lifetime matters.

3. `T = fwd::Dual` (forward-mode AD, used by gradient verification):
   either skip (forward-mode is not on the production HMC path, only
   used by `verify_gradient_runtime` which can fall back to numerical
   for SPDE), or implement a forward-mode adjoint of the NC transform
   (extra work; defer).

The cleanest place to hold the cache is a `mutable std::unique_ptr<SpdeNcTransform>`
on `SpdeModelData` itself, lazily initialised on first use. `ModelData`
is passed by const-reference into the log_post evaluator, so making
this `mutable` is correct (the cache is implementation detail, not
observable state). Initialise on first access:

```cpp
inline SpdeNcTransform& get_spde_nc_transform(const SpdeModelData& s) {
    if (!s.nc_transform) {
        s.nc_transform = std::make_unique<SpdeNcTransform>();
        s.nc_transform->init(s.n_mesh, s.C0_diag, s.G1_x, s.G1_i, s.G1_p);
    }
    return *s.nc_transform;
}
```

ABI implication: adding `mutable std::unique_ptr<SpdeNcTransform> nc_transform`
to `SpdeModelData` changes the layout. Bump `TULPA_ABI_VERSION` 17 -> 18.
Forward-declare `SpdeNcTransform` in `tulpa/spde_model_data.h` so the
header doesn't pull in Eigen for downstream packages.

### Step 4 — PC prior on (range, sigma)

Fuglstad et al. (2019) "Constructing priors that penalize the
complexity of Gaussian random fields", JASA. Standard SPDE PC prior:

- Anchor `(range_0, alpha_range)` such that `P(range < range_0) = alpha_range`
- Anchor `(sigma_0, alpha_sigma)` such that `P(sigma > sigma_0) = alpha_sigma`

Density:

```
pi(range)       = lambda_r / 2 * range^{-3/2} * exp(-lambda_r * range^{-1/2})
pi(sigma)       = lambda_s * exp(-lambda_s * sigma)
lambda_r        = -log(alpha_range) * sqrt(range_0)
lambda_s        = -log(alpha_sigma) / sigma_0
```

Change of variable to (log_kappa, log_tau):

```
range = sqrt(8 nu) / kappa             ->  range = sqrt(8 nu) * exp(-log_kappa)
sigma = 1 / (sqrt(4 pi) * kappa * tau) ->  sigma = exp(-log_kappa - log_tau) / sqrt(4 pi)
```

Jacobians:

```
d(range)/d(log_kappa) = -range
d(sigma)/d(log_kappa) = -sigma
d(sigma)/d(log_tau)   = -sigma
```

|J| of the map (log_kappa, log_tau) -> (range, sigma) =
|det [ [-range, 0], [-sigma, -sigma] ]| = range * sigma. So the joint
log-density in (log_kappa, log_tau) is:

```
log p(log_kappa, log_tau) = log pi(range) + log pi(sigma) + log(range * sigma)
                          = log pi(range) + log pi(sigma) + log_range + log_sigma
```

with `log_range = log(sqrt(8 nu)) - log_kappa` and
`log_sigma = -0.5 log(4 pi) - log_kappa - log_tau`.

Implement in `compute_spde_hyper_prior` (the stub from step 2). Take
the four PC anchors `(range_0, alpha_range, sigma_0, alpha_sigma)`
from new fields on `SpdeModelData`. Add them to `SpdeModelData` (ABI
already bumped in step 3, so this is free).

### Step 5 — Rcpp entry point

File: `src/tulpa_spde_sampler.cpp`.

Add new arguments to `cpp_tulpa_fit_spde_nuts`:

```cpp
double prior_range_0,    double prior_range_alpha,
double prior_sigma_0,    double prior_sigma_alpha,
double log_kappa_init,   double log_tau_init
```

Drop or default the old fixed-hyper `kappa, tau_spde` arguments — keep
the call signature compatible by giving them `R_NaReal` defaults and
treating any non-NA value as a fixed-hyper override. Two modes:

- Joint mode (default, prior anchors supplied): PC prior on hypers,
  layout reserves the two hyper slots, NC transform active.
- Fixed mode (kappa + tau_spde supplied): legacy path, hypers stay
  outside the parameter vector. Useful for the existing nested-Laplace
  outer-loop callers.

Initialise the parameter vector:

```cpp
init[layout.spde_w_start + j]      = 0.0;          // z = 0
init[layout.log_kappa_spde_idx]    = log_kappa_init;
init[layout.log_tau_spde_idx]      = log_tau_init;
init[layout.extra_offset]          = log_phi_init;
```

Default `log_kappa_init`, `log_tau_init` from the PC prior modes:
`range_mode = ((-log(alpha_range))^2 / 4)` ... actually just take
the user-supplied prior anchors and pick `range_init = range_0`,
`sigma_init = sigma_0`. Convert to (log_kappa, log_tau).

Output column names: rename `w[i]` columns to `z[i]`, but also
post-process to add `w[i]` columns by transforming each draw via
`SpdeNcTransform::forward` at that draw's (log_kappa, log_tau, z).
Add `log_kappa`, `log_tau`, `kappa`, `tau`, `range`, `sigma` columns
to the draws matrix.

### Step 6 — R wrapper, docs, end-to-end test

File: `R/fit_spde_nuts.R`.

- Add `prior_range`, `prior_sigma` arguments (each a `c(value, alpha)`
  pair, matching `spatial_spde()`'s convention).
- Default to the values from `spatial$prior_range`, `spatial$prior_sigma`
  if user does not supply.
- Drop the `range`, `sigma` arguments (or keep with NULL defaults that
  trigger joint sampling).
- Convert the C++ output to user-friendly summaries: posterior of
  range, sigma, kappa, tau, plus the existing beta/w/phi summaries.

File: `man/tulpa_nuts_spde.Rd` — `devtools::document()` regenerates.

File: `tests/testthat/test-spde-nuts-joint.R` (new). End-to-end recovery
test on simulated SPDE Gaussian data:

```r
test_that("joint NUTS recovers (range, sigma, beta) on simulated SPDE", {
  skip_if_not_installed("fmesher")
  # Simulate 300 obs from a Matern SPDE with known (range = 0.4, sigma = 1).
  # Fit with broad PC priors (P(range < 0.1) = 0.05, P(sigma > 3) = 0.05).
  # Assert that 90% credible intervals on (range, sigma, beta) cover the truth
  # in 9 out of 10 replicate seeds (or a single seed with 90% CIs of width
  # bounded; pick whichever is more robust).
})
```

Also verify with `verify_gradient_runtime` that the joint gradient
matches central differences (the existing warmup-time check should
just work).

## Open design questions

1. **Where to cache the `SpdeNcTransform` instance.**
   Step 3 proposes `mutable unique_ptr` on `SpdeModelData`. Alternative:
   thread-local cache keyed on the `SpdeModelData*` pointer. The
   `mutable` approach is cleaner but couples `spde_model_data.h` to
   the existence of `SpdeNcTransform` (forward decl is enough — no
   Eigen leakage). Pick the `mutable` route unless it conflicts with
   how downstream packages instantiate `SpdeModelData` on the heap.

2. **Forward-mode (`fwd::Dual`) support.**
   Skip in step 3, since the production HMC path uses `arena::Var`.
   Forward-mode is only used by `verify_gradient_runtime` against the
   numerical reference; we can either disable that check for SPDE
   (set the runtime check to numerical-only when `layout.is_spde`) or
   implement forward-mode NC transform later. Lean toward the former
   as a clean cut.

3. **Initial `(log_kappa, log_tau)` choice.**
   Default from `(prior_range_0, prior_sigma_0)`. If the user
   supplies fixed-mode legacy args, use them. Add a warning if the
   PC prior is so vague that the modes are far from the data scale.

4. **Sparse vs dense `M_theta` in the adjoint.**
   The (a.ii) implementation densifies `M_theta = L^{-1} dQ/dtheta L^{-T}`
   for the trace computation — O(n_mesh^3). For meshes with
   n_mesh > 2000 this becomes the bottleneck. Replace with a sparse
   partial-inverse path (Takahashi recursion already lives in
   `src/sparse_cholesky.{h,cpp}` for ICAR; reuse the same pattern).
   Defer to a follow-up.

5. **Rational-alpha (fractional `nu`) extension.**
   (a.ii) only supports `alpha = 2`. The Rcpp shim still accepts
   rational poles/weights for backward compat, but the joint NUTS
   path errors out unless `alpha == 2`. Deferred.

## Test plan

In order:

1. Layout test (step 1): `cpp_test_param_layout` style helper or
   direct test against `compute_param_layout`. Asserts the new slot
   indices line up.

2. Prior test (step 2): build a small joint state, evaluate
   `initialize_generic_state` for `T = double`, assert that the
   returned log_post equals `-0.5 sum z^2` plus expected fixed-effect
   prior plus zero hyper-prior contribution (stub).

3. Eta-path test (step 3): on a small mesh, set `z` to random values,
   evaluate eta via the structured HMC path, compare against direct
   `SpdeNcTransform::forward` + sparse `A * w` matvec. Should match
   to 1e-12.

4. Hyper prior test (step 4): evaluate `compute_spde_hyper_prior` at
   several (log_kappa, log_tau) points, compare against an R-side
   `INLA::inla.pc.dprec` or hand-coded reference.

5. Gradient verification (step 5): kick a chain with `verify_gradient = TRUE`
   on a small dataset, ensure the runtime check passes for the joint
   parameter vector.

6. End-to-end recovery (step 6): see test sketch above.

## Risks and pitfalls

- **`SpdeNcTransform` lifetime in arena callbacks.** The arena hook
  captures `transform` by reference. The cache must outlive the arena
  scope. Step 3's mutable-on-ModelData design is safe because
  `ModelData` outlives every gradient evaluation by construction.

- **Gradient sign convention in the implicit-function-theorem step.**
  (a.ii) is verified vs FD on the standalone path. The integration
  point in step 3 must preserve the contract: `dL/dz` flows from the
  arena into `params[spde_w_start + j]`'s adjoint slot via the standard
  arena propagation; the NC adjoint adds `dL/d(log_kappa)`,
  `dL/d(log_tau)` to the corresponding params adjoints. Bug here will
  show up immediately as a `verify_gradient_runtime` failure on warmup.

- **PC prior numerical stability.** `range^{-3/2}` blows up as
  `range -> 0`. Express the log-density directly in `(log_kappa, log_tau)`
  and never form `range^{-3/2}` explicitly — substitute
  `log range = log(sqrt(8 nu)) - log_kappa` from the start.

- **Permutation in the Cholesky.** (a.ii) uses
  `Eigen::NaturalOrdering` to avoid permutation. If you swap to
  `AMDOrdering` for performance later, the back-substitution and
  adjoint formulas need a `P z` and `P^T y` step inserted. Don't
  swap without re-deriving.

- **ABI bump on `SpdeModelData`.** Step 3 adds the cache pointer.
  Bump `TULPA_ABI_VERSION` 17 -> 18 in `inst/include/tulpa/model_data.h`.
  Downstream packages (tulpaObs in particular) will reject mismatched
  ABI on first call — recompile them after the bump.

- **Parallel-writer collisions.** Last session lost in-progress work
  to a phantom `git merge` that left the result in `.git/AUTO_MERGE`
  but did not apply it to the working tree (recovered from tree
  object `9d7ded21`). If the parallel writer is still active, commit
  at every milestone (after each step) so a collision at most loses
  one step's work.

## File-by-file change summary

| File | Change |
| --- | --- |
| `inst/include/tulpa/param_layout.h` | Update comment on hyper slots (now valid) |
| `inst/include/tulpa/spde_model_data.h` | Forward-decl `SpdeNcTransform`; add `mutable unique_ptr` cache; PC prior anchor fields |
| `inst/include/tulpa/model_data.h` | Bump `TULPA_ABI_VERSION` 17 -> 18 |
| `src/hmc_param_layout.cpp` | Allocate `log_kappa_spde_idx` / `log_tau_spde_idx` after `spde_w` block |
| `src/tulpa_priors_spde.h` | Rewrite `compute_spde_prior` to NC form; add `compute_spde_hyper_prior` (PC) |
| `src/log_post_generic_impl.h` | Wire NC transform into eta path; call `compute_spde_hyper_prior` |
| `src/spde_nc_transform.{h,cpp}` | (Optional) sparse partial-inverse trace path; (Optional) `fwd::Dual` specialisation |
| `src/tulpa_spde_sampler.cpp` | New PC prior args; init from prior modes; output transform `z -> w`, add hyper columns |
| `R/fit_spde_nuts.R` | New `prior_range`, `prior_sigma` args; user-friendly hyper summaries |
| `man/tulpa_nuts_spde.Rd` | Regenerate via `devtools::document()` |
| `tests/testthat/test-spatial-spde-api.R` | Layout slot assertions |
| `tests/testthat/test-spde-nuts-joint.R` | New end-to-end recovery test |

## When (a.iii) is done

- TODO.md "### 2. Joint NUTS over (log_kappa, log_tau_spde)" can be
  removed entirely (or moved to a "completed" section).
- The arc closes a feature that has been blocking the SPDE-NUTS
  story since the initial Phase 1 land in `17a492c`.
- Downstream wiring in `tulpaObs` (TODO.md item #3) becomes the
  next natural arc: `occu_spde` already wraps `spatial_spde`; it
  just needs to flip its NUTS call to the joint-hyper version.
