# Posterior draws from a nested-Laplace fit

Draw from the outer-grid mixture posterior of a nested-Laplace fit – the
engine analogue of `inla.posterior.sample()`. Each draw picks an
outer-grid cell `k ~ Categorical(weights)` and then samples that cell's
inner Gaussian, so the draws are i.i.d. samples from
`sum_k w_k N(m_k, V_k)`.

Sampling the mixture is the faithful primitive for marginalizing
nonlinear derived quantities (e.g. `plogis(eta_2) - plogis(eta_1)`,
expected-cover products `p * mu`): compute the derived quantity per
draw, then summarize. Collapsing the grid to a single moment-matched
Gaussian biases skewed or multimodal-over-grid posteriors.

## Usage

``` r
tulpa_posterior_draws(fit, idx = NULL, n = 1000, ...)
```

## Arguments

- fit:

  A nested-Laplace fit
  ([`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
  or
  [`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)).

- idx:

  Optional integer vector of 1-based indices into whatever a draw covers
  for this backend (see above); `NULL` (default) returns all of them.

- n:

  Number of posterior draws (default 1000).

- ...:

  Unused; for S3 compatibility.

## Value

A numeric matrix `[n x length(idx)]`, one row per draw. Carries
`attr(., "draws_kind") = "iid"` (consistent with the draws-provenance
gate), `attr(., "cells")` – the outer-grid cell index each row was drawn
from – and `attr(., "scope")`, which of the two representations above
the columns are.

## What a draw covers

It depends on which representation the backend retained, and the
returned matrix says so in its `scope` attribute.

- `tulpa_nested_laplace_joint`:

  the FULL latent vector – per-arm fixed effects, per-arm random
  effects, then the latent field(s) – because the joint fit retains each
  cell's sparse precision over that vector (`control$store_Q = TRUE`).
  `scope` is `"latent"`.

- `tulpa_nested_laplace` (single-block):

  the FIXED-EFFECT block. This backend inverts each cell's precision
  into the marginal fixed-effect block and releases the precision
  itself, so the latent field is not part of the retained per-cell
  Gaussian and cannot be sampled from the fit. `scope` is `"fixed"`.

## See also

[`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md),
[`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md),
[`posterior_sample()`](https://gillescolling.com/tulpa/reference/posterior_sample.md)
