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

# Default S3 method
tulpa_posterior_draws(fit, idx = NULL, n = 1000, ...)

# S3 method for class 'tulpa_nested_laplace'
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
the columns are. `attr(., "theta")` is the HYPERPARAMETER half of the
same rows, one column per outer-grid axis, continuized within each row's
own cell by
[`tulpa_hyper_draws()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_draws.md);
reading `fit$theta_grid[attr(., "cells"), ]` instead returns the bare
node coordinate, which is an atom rather than a marginal.

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

[`tulpa_hyper_draws()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_draws.md),
[`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md),
[`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md),
[`posterior_sample()`](https://gillescolling.com/tulpa/reference/posterior_sample.md)

## Examples

``` r
# \donttest{
set.seed(1)
S <- 30L                                   # spatial units in a chain
nb <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
nn <- lengths(nb)
field <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4))))
idx <- rep(seq_len(S), each = 6L); n <- length(idx); x <- rnorm(n)
y <- rbinom(n, 1L, plogis(-0.2 + 0.6 * x + field[idx]))
prior <- list(type = "icar", n_spatial_units = S, spatial_idx = idx,
              adj_row_ptr = c(0L, cumsum(nn)), adj_col_idx = unlist(nb) - 1L,
              n_neighbors = nn, tau_grid = c(0.5, 1, 2, 4, 8))
fit <- tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = prior,
                            family = "binomial",
                            control = list(progress = FALSE))
dr <- tulpa_posterior_draws(fit, n = 200)
colMeans(dr)                # fixed effects, mixed over the outer grid
#>       beta1           x 
#> -0.02097457  0.57813071 
summary(attr(dr, "theta"))  # the hyperparameter half of the same draws
#>       tau         
#>  Min.   : 0.4106  
#>  1st Qu.: 3.9219  
#>  Median : 5.7068  
#>  Mean   : 5.9962  
#>  3rd Qu.: 8.3608  
#>  Max.   :11.3051  
# }
```
