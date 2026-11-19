# Posterior draws from a joint nested-Laplace fit

The
[`tulpa_posterior_draws()`](https://gillescolling.com/tulpa/reference/tulpa_posterior_draws.md)
method for a joint nested-Laplace fit
([`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)).
Each draw picks an outer-grid cell `k ~ Categorical(weights)` and then
samples the inner latent vector from the inner Gaussian `N(m_k, V_k)` at
that cell, where `m_k` is the cell's inner mode and `V_k` the covariance
the cell's log-marginal was read off. The draws are i.i.d. samples from
`sum_k weights_k * N(m_k, V_k)`.

## Usage

``` r
# S3 method for class 'tulpa_nested_laplace_joint'
tulpa_posterior_draws(fit, idx = NULL, n = 1000, ...)
```

## Arguments

- fit:

  A `tulpa_nested_laplace_joint` fit (single-block or multi-block). The
  fit must have been produced with `control$store_Q = TRUE` so the
  per-grid sparse precision `Q_csc_*_per_grid` is available.

- idx:

  Optional integer vector of 1-based latent indices to return. `NULL`
  (default) returns the full latent vector. The latent vector stacks
  per-arm fixed effects, per-arm random effects, then the latent
  field(s); use `fit$arm_layout` (`beta_start`, `re_start`, `phi_start`
  / `theta_start` / `field_starts`, all 0-based) to map a sub-block to
  indices.

- n:

  Number of posterior draws (default 1000).

- ...:

  Unused; for S3 compatibility.

## Value

A numeric matrix `[n x length(idx)]` of latent draws, one row per draw,
columns named `x<idx>`. Carries `attr(., "draws_kind") = "iid"`
(consistent with the draws-provenance gate), `attr(., "cells")` – the
outer-grid cell index each row was drawn from – `attr(., "theta")`, the
hyperparameter half of the same rows continuized within each row's own
cell
([`tulpa_hyper_draws()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_draws.md))
– and `attr(., "scope") = "latent"`.

## Details

The inner Gaussian at cell `k` has precision \\M_k = Q_k + S^\top D_k
S\\, the stored precision `Q_k` plus the sum-to-zero pins `S` the sparse
inner solve folds at solve time (an intrinsic field's augmented prior,
precision `D_k`), conditioned on the hard rows `A x = 0` the inner solve
carried (an intrinsic field's level row where a fixed effect absorbs its
level, a Knorr-Held interaction's null space): \\V_k = B (B^\top M_k
B)^{-1} B^\top\\, `B` a basis of the null space of `A`. A pinned field
with no level row keeps its level, at the pin's precision. The returned
draw is `m_k` plus a draw of `N(0, V_k)`, restricted to `idx`; `m_k`
already satisfies the hard rows.

Cells with zero outer-grid weight (e.g. pruned cells) or no stored `Q`
are dropped and the remaining weights renormalized. A degenerate
single-cell grid (quadrature ESS 1) returns draws from that cell's
`N(m_1, V_1)`.

## See also

[`tulpa_posterior_draws()`](https://gillescolling.com/tulpa/reference/tulpa_posterior_draws.md),
[`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md),
[`posterior_sample()`](https://gillescolling.com/tulpa/reference/posterior_sample.md)
