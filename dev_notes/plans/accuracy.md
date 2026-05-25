# tulpa joint-Laplace accuracy plan

Three accuracy wins, ranked. The bulk of the work is Win 1 (engine
reparameterization). Wins 2 and 3 are smaller and depend on it.

Reference points used throughout:
- `R/nested_laplace_joint.R` — joint driver (`.nl_nested_laplace_joint`,
  `.joint_attach_alpha_moments`, `.alpha_grid_moments`).
- `src/nested_laplace_joint_multi.cpp` + `src/nested_laplace_joint_multi.h`
  — per-cell Laplace block, log-marginal accumulator.
- `src/laplace_newton_joint.h` — joint Newton step.
- Recent context: commits `fa6d2f3` (marginalize derived α), `f5b9d2b` /
  `d50ce4a` (regularizing σ hyperpriors including σ=0).

---

## Diagnosis: where bias enters today

Current parameterization on the copy block is `(σ_occ, σ_pos)` on a 2D
grid; α is **derived** post hoc as `σ_pos / σ_occ`.

Two structural problems flow from this:

1. **Ratio variance dominates at small σ_occ.** When σ_occ is weakly
   identified (typical in real data: sparse positives, short series), the
   denominator of the derived α is noisy. The posterior of α inherits that
   noise even when the *copy strength* itself is well-identified — a
   field can be cleanly scaled but the absolute amplitudes are not.
2. **No clean way to encode α = 0.** The plausible prior on α=0 ("the
   positive arm does not copy the occupancy field") is awkward to express
   as a joint prior on `(σ_occ, σ_pos)`. Setting σ_pos=0 collapses the
   positive arm's residual field, which is not the same statement.
   `rgeneric`'s copy model gets this corner wrong in the opposite
   direction (no α=0 atom), which is why it underperforms tulpaObs at α=0
   today.

Win 1 fixes both at once.

---

## Win 1 — reparameterize the copy block as `(σ, α)`

**Goal.** Replace the internal `(σ_occ, σ_pos)` grid on the copy block
with a `(σ, α)` grid, where σ is the single BYM2 field amplitude on the
occupancy arm and α is the scalar copy coefficient that scales it into
the positive arm. α has a direct posterior; nothing is derived.

### Math (target shape)

Per copy block, one shared latent field `u ~ BYM2(σ, ρ)` lives on the
occupancy arm. The positive arm's contribution from this block is
`α · u`. The joint Laplace cell becomes:

```
ψ_occ = X_occ β_occ + u                          # u has amplitude σ
ψ_pos = X_pos β_pos + α · u                      # copies u, scaled by α
```

The hyperparameter grid integrates over `(σ, α, ρ)` instead of
`(σ_occ, σ_pos, ρ)`. The conditional Laplace step (Newton on the joint
latent given hypers) is unchanged in structure — only the precision
contribution from the copy block changes from "two independent fields
with amplitudes σ_occ and σ_pos" to "one field u with amplitude σ, and a
deterministic affine map `u → αu` into the positive arm".

This is the same `Q`-matrix structure INLA uses for `f(idx, copy=...)`.

### Prior on `(σ, α)`

- **σ**: keep current PC-style hyperprior on the single field amplitude
  (the regularizing prior already shipped in `f5b9d2b` / `d50ce4a`,
  including σ=0).
- **α**: PC-style prior **with a point mass / continuous bridge at
  α=0**. This is the structural fix `rgeneric` lacks. Two acceptable
  shapes:
  1. Continuous PC prior on |α| relative to a base scale `α₀`
     (`P(|α| > α₀) = p₀`), with a small atom on α=0 controlled by an
     "include-zero" weight (analogous to the σ=0 inclusion in
     `prior_sigma_*` after `d50ce4a`).
  2. PC prior on `α` against the base model "α=0" — a half-Normal /
     half-Laplace shrinkage toward zero with explicit positive mass at
     the boundary, exposed as `prior_alpha = list(scale = ., zero = .)`.

Default Option 2 (cleaner: zero is the base model). Same API surface as
`prior_sigma_*` so users see one consistent hyperprior shape.

### Grid construction

Today (in `.joint_block_axis_grid`, R/nested_laplace_joint.R:2118):

```r
gr <- expand.grid(sigma_occ = sigma_occ,
                  sigma_pos = sigma_pos_grid,
                  ...)
```

After Win 1:

```r
gr <- expand.grid(sigma = sigma_grid,
                  alpha = alpha_grid,
                  ...)
```

Constraints on `alpha_grid`:
- include 0 explicitly (mirrors `d50ce4a` for σ).
- log-spaced in |α| with a small symmetric portion in (-α₀, α₀) so
  near-zero curvature is captured for the marginal weighted-quantile
  summary.
- same size as today's `sigma_pos_grid` so total cell count is
  unchanged — no runtime regression.

### Block spec surface (user-facing API)

Today the copy block is declared via `copy$sigma_pos_grid`. After Win 1,
the supported surface becomes:

```r
copy = list(
  sigma_grid = ...,          # was already implicit via prior$sigma_grid
  alpha_grid = ...,          # NEW: direct copy-coefficient grid
  prior_alpha = list(...)    # NEW: PC-style prior on alpha
)
```

Backward-compat shim (one minor release): if a user passes
`copy$sigma_pos_grid`, translate via
`alpha_grid = sigma_pos_grid / anchor_sigma` with a deprecation warning,
mirroring the existing translation of the older `copy$alpha_grid` in
R/nested_laplace_joint.R:836–849. The shim is one helper, deletable.

### C++ touch points

`src/nested_laplace_joint_multi.{h,cpp}` and
`src/laplace_newton_joint.h`:

1. Copy-block Q assembly: instead of stacking two independent
   `Q_BYM2(σ_occ, ρ)` and `Q_BYM2(σ_pos, ρ)`, build one
   `Q_BYM2(σ, ρ)` for u, and add the deterministic linear map
   `pos_arm += α · u` to the design / linear-predictor accumulation.
2. Cross-arm precision contribution: the positive arm's contribution to
   the Hessian gains an `α²` factor on the shared u block, and the
   off-block coupling (between u and positive-arm fixed effects) gains
   an `α` factor. Both are local, well-localized changes.
3. Log-marginal accumulator: no change in structure — still integrates
   `log p(θ | y) ≈ log p(y | θ̂, θ) + log p(θ̂ | θ) - ½ log |H| + log p(θ)`
   per cell — but with θ = (σ, α, ρ) instead of (σ_occ, σ_pos, ρ).

ABI: bump `TULPA_ABI_VERSION` in `inst/include/tulpa/model_data.h`.
Model packages (tulpaObs, numdenom) auto-check on first NUTS call so
mismatches surface cleanly.

### R-side touch points

- `R/nested_laplace_joint.R`:
  - `.parse_copy_spec` (single-block path, ~L800–870) and the
    multi-block analogue (~L2080–2105): accept `alpha_grid` natively;
    translate legacy `sigma_pos_grid` once with a deprecation.
  - `.joint_block_axis_grid` (L2118–2157): emit `(sigma, alpha[, rho])`
    for the copy block. Non-copy blocks unchanged.
  - `.alpha_grid_moments` (L957–1009): becomes a one-liner — α is a
    grid axis, so `out$alpha_vec <- tg[, "alpha"]`, and the weighted
    median / 2.5–97.5 quantiles are computed directly off the joint
    log-marginal via `.nl_wtd_quantile`. The current ratio
    `sp / so` path, the foreign-axis filter, the per-axis Laplace
    fallback (`.joint_alpha_log_params`, L1048+), and the delta-method
    SD machinery (`.joint_alpha_sd_log_delta`, L1085+) all become
    dead code. Delete in the same change.
  - `.joint_attach_alpha_moments` (L1016): renames to
    `.joint_attach_copy_moments`; semantics simplify to "α is already
    a grid axis, just expose its weighted-quantile moments under
    `alpha` and call it a day".
- Documentation in roxygen for the joint driver: drop the
  "σ_pos / σ_occ" derived-ratio prose. Update example block specs.

### Validation plan

A single recovery study under three regimes:

| Regime | Simulated α | Goal vs rgeneric (INLA) |
|---|---|---|
| Strong copy | α = 2.0 | match RMSE on α within 5%, match coverage at 95% within 3pp |
| Moderate copy | α = 1.0 | match RMSE within 10% |
| No copy | α = 0.0 | **beat** rgeneric: posterior mass concentrates near 0 with calibrated CI; rgeneric should overshoot |

Per regime: 30 seeds, BYM2 + AR1 + IID layout (the layout already
exercised in `ddc628f`), N≈300 sites × 10 visits. Recovery script lives
in `dev_notes/_run_joint_multi_recovery.R` (already exists; extend
truth grid).

Acceptance:
- α posterior median within tolerance of truth (per table above).
- 95% CI empirical coverage in [0.85, 0.97] over 30 seeds per regime.
- At α=0: CI lower bound ≤ 0 in ≥85% of seeds; rgeneric reference does
  **not** meet this (this is the gap we are paid for).

Run before merging Win 1. Land regression as
`tests/testthat/test_joint_alpha_recovery_*.R`.

### Effort

Estimated 2–3 weeks of focused work:
- Week 1: C++ Q assembly + Newton step + log-marginal change; unit
  tests on synthetic BYM2 with known α.
- Week 2: R-side grid + prior_alpha + deprecation shim; delete dead
  derived-α machinery; roxygen + vignette pass.
- Week 3: 30-seed recovery sweep across three regimes; INLA reference
  comparison; CRAN-style check on tulpa + tulpaObs.

This is the right long-term shape. Do it.

---

## Win 2 — keep the marginal weighted-quantile summary

Already shipped (`fa6d2f3`, `bb8922c`, `60da97f`). **Do not roll back.**

The current path computes α posterior median + empirical 95% CI from
weighted quantiles on the joint hyperparameter grid:

```
per cell i: weight w_i ∝ exp(log_marginal_i)
            α_i = f(grid_i)
weighted median(α_i), 2.5%/97.5% quantiles
```

After Win 1 this becomes trivial — α is a grid axis — but the
*summarization machinery* (`.nl_wtd_quantile`, foreign-axis filtering,
empirical CI rather than `mean ± 1.96·sd`) is exactly right and stays.

Specifically preserved:
- empirical quantiles via `.nl_wtd_quantile`, not Lognormal / Gaussian
  fits on a skewed posterior;
- weighted median as the point estimate, not the grid MAP;
- 2.5% / 97.5% quantiles as the CI, not `median ± 1.96·sd_log`.

The Lognormal closed-form path (`.joint_alpha_log_params`,
`.joint_alpha_sd_log_delta`) is fine as an internal diagnostic but
**must not** be the user-visible CI. After Win 1, delete it.

This follows the global rule: marginalize derived quantities over the
joint posterior; never plug-in MAPs of components.

---

## Win 3 — expose α posterior natively (after Win 1)

After Win 1 lands, α is a direct hyperparameter of the joint model, so
the engine should expose it natively in the fit object:

```r
fit$alpha_mean      # weighted mean over joint grid
fit$alpha_median    # weighted median
fit$alpha_ci        # c(2.5%, 97.5%) empirical quantiles
fit$alpha_grid      # the alpha axis values
fit$alpha_weights   # marginal weights summed over (sigma, rho)
```

Implementation: one helper next to `.joint_attach_copy_moments` that
marginalizes the joint log-marginal over non-α axes and exposes the
α-axis weights for downstream visualization. Same machinery as
`.nl_wtd_quantile`, applied per-axis.

Removes a whole class of summarization bugs because there's no derived
quantity left:
- no ratio variance;
- no per-axis Laplace-at-mode parabola fits;
- no delta-method transfer to log scale;
- no "is the modal cell at a grid edge?" branching.

`tulpaObs::fit$alpha_*` becomes a thin pass-through from the engine.

### S3 surface

Expose under existing generic methods (`R/methods_generic.R`):
- `summary.tulpa_fit()`: add an "Copy coefficients" block listing each
  copy block's α median and 95% CI.
- `tidy.tulpa_fit()`: emit copy-coefficient rows
  (`term = "alpha[<block_name>]"`, `estimate`, `conf.low`, `conf.high`).
- `plot.tulpa_fit(..., what = "alpha")`: density of α per copy block
  from the marginal weights.

These are all in tulpa core (generic), not the model packages.

---

## Sequencing

| Phase | Work | Gate |
|---|---|---|
| 0 | This document. | — |
| 1 | C++ `Q(σ, α, ρ)` assembly + Newton + log-marginal. Unit tests on synthetic data with known α. | Local check passes; 5-seed recovery within tolerance. |
| 2 | R-side `alpha_grid` + `prior_alpha`; deprecation shim for `sigma_pos_grid`. Delete derived-α machinery. | `devtools::check()` clean on tulpa, tulpaObs, numdenom. |
| 3 | 30-seed recovery sweep, three α regimes, INLA cross-check. | Acceptance criteria above. |
| 4 | Win 3: expose `fit$alpha_*` + S3 surface. | New regression for `tidy()` / `summary()` output. |
| 5 | Vignette + NEWS update; bump ABI; release notes. | win-builder r-devel clean. |

Phases 1–3 are the value. 4 is cleanup. 5 is shipping.

---

## Out of scope (deliberately)

- Multi-block copy where multiple latent fields are copied into the
  positive arm with separate α's. Feasible after Win 1 (just add more
  axes), but not in the first ship.
- α varying by covariate (TVC-style varying copy coefficient). Same
  reasoning.
- Replacing the 2D grid with CCD over `(σ, α, ρ)` for higher-dimensional
  copy blocks. Worth doing once Win 1 ships and grid size starts to
  bite.

---

## Risk register

1. **C++ Q-matrix bug on the cross-arm coupling.** Mitigation: unit
   test the per-cell Hessian against the numerical Hessian on a 50×50
   synthetic problem before any recovery work. Same gradient-verify
   discipline as NUTS.
2. **`prior_alpha` calibration.** PC prior on copy coefficients is less
   well-established than on amplitudes. Mitigation: report sensitivity
   to `prior_alpha$scale` across {0.5, 1, 2} in the validation sweep;
   if recovery is too tight at α=0, surface as a `prior_alpha$zero`
   weight knob.
3. **Backward compat for users on `copy$sigma_pos_grid`.** Mitigation:
   shim + warning for one minor release; deprecation listed in NEWS;
   tests cover both paths until removal.
4. **rgeneric reference may differ in non-copy specifics.** Mitigation:
   match the BYM2 + AR1 + IID layout exactly; cross-check both engines
   on the *occupancy* posterior first (should agree at α≥1) before
   reading α-specific metrics.
