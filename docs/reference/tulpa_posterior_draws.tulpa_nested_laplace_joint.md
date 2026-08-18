# Posterior draws from a joint nested-Laplace fit

The
[`tulpa_posterior_draws()`](https://gillescolling.com/tulpa/reference/tulpa_posterior_draws.md)
method for a joint nested-Laplace fit
([`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)).
Each draw picks an outer-grid cell `k ~ Categorical(weights)` and then
samples the inner latent vector from the constrained Gaussian
`N(m_k, V_k)` at that cell, where `m_k` is the cell's inner mode and
`V_k` is the inner-Laplace covariance with the ICAR / BYM2 sum-to-zero
field constraint imposed (conditioning by kriging). The draws are i.i.d.
samples from `sum_k weights_k * N(m_k, V_k)`.

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
outer-grid cell index each row was drawn from – and
`attr(., "scope") = "latent"`.

## Details

The constrained draw at cell `k` uses the sparse Cholesky of the stored
precision `Q_k` and the conditioning-by-kriging correction \$\$z_c = z -
Q_k^{-1} A^\top (A Q_k^{-1} A^\top)^{-1} A z,\$\$ where
`z ~ N(0, Q_k^{-1})` and `A` stacks the field sum-to-zero rows (one for
an ICAR / CAR field; two – structured and unstructured – for a BYM2
field; one per spatial block in a multi-block fit). The returned draw is
`m_k + z_c`, restricted to `idx`. Because `m_k` already satisfies the
constraint (the inner solve centres the field), the mean is left
unchanged and only the covariance is constrained, so the per-cell
marginal matches the inner-Laplace constrained covariance exactly.

Cells with zero outer-grid weight (e.g. pruned cells) or no stored `Q`
are dropped and the remaining weights renormalized. A degenerate
single-cell grid (quadrature ESS 1) returns draws from that cell's
`N(m_1, V_1)`.

## See also

[`tulpa_posterior_draws()`](https://gillescolling.com/tulpa/reference/tulpa_posterior_draws.md),
[`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md),
[`posterior_sample()`](https://gillescolling.com/tulpa/reference/posterior_sample.md)
