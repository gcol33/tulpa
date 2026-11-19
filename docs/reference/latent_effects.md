# Latent block effects per coordinate

The posterior of one
[`latent()`](https://gillescolling.com/tulpa/reference/latent.md) block
([`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md),
[`tgmrf_cpp()`](https://gillescolling.com/tulpa/reference/tgmrf_cpp.md))
of a nested-Laplace fit, one row per coordinate of the block: each outer
cell contributes the coordinate's conditional mean and marginal variance
under the cell's latent Gaussian, and the rows are the exact mean, SD
and 2.5% / 97.5% quantiles of that Gaussian mixture over the cell
weights. Each coordinate's value is its contribution to the linear
predictor of the observations `obs_idx` maps to it.

## Usage

``` r
latent_effects(object, block = 1L)
```

## Arguments

- object:

  A nested-Laplace `tulpa_fit` carrying at least one
  [`latent()`](https://gillescolling.com/tulpa/reference/latent.md)
  block.

- block:

  Which latent block: its position among the fit's
  [`latent()`](https://gillescolling.com/tulpa/reference/latent.md)
  blocks (default 1), or the `name` given to
  [`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md).

## Value

A data frame with one row per coordinate: `index`, `estimate`, `sd`,
`conf.low`, `conf.high` and `source` (`"mixture"`, or `"mode"` where the
fit retained no per-cell variance).

## See also

[`field_effects()`](https://gillescolling.com/tulpa/reference/field_effects.md)
for a spatial field,
[`ranef()`](https://gillescolling.com/tulpa/reference/ranef.md) for
random effects,
[`temporal()`](https://gillescolling.com/tulpa/reference/temporal.md)
for a temporal field.

## Examples

``` r
# \donttest{
set.seed(1)
K <- 12
W <- matrix(0, K, K)
for (i in 1:K) { j <- if (i < K) i + 1 else 1; W[i, j] <- W[j, i] <- 1 }
d <- data.frame(region = rep(1:K, each = 10), x = rnorm(120))
d$y <- rpois(120, exp(0.5 + 0.4 * d$x + sin(2 * pi * d$region / K)))
blk <- tgmrf(Q = function(theta) Matrix::Diagonal(120, exp(theta[1])),
             prior = function(theta) dnorm(theta[1], 2, 1.5, log = TRUE),
             init = c(log_tau = 2), bounds = list(lower = -1, upper = 7),
             obs_idx = seq_len(120), name = "obs_iid")
fit <- tulpa(y ~ x + spatial(region) + latent(blk), data = d,
             family = "poisson", spatial = list(type = "icar", adjacency = W),
             mode = "auto", control = list(progress = FALSE))
head(latent_effects(fit, "obs_iid"))
#>   index      estimate        sd   conf.low conf.high  source
#> 1     1 -0.0406908127 0.2022865 -0.4873533 0.3498855 mixture
#> 2     2  0.0070854950 0.1946794 -0.3920110 0.4154124 mixture
#> 3     3  0.0834161171 0.2079128 -0.2871426 0.5647573 mixture
#> 4     4 -0.0658436023 0.1944762 -0.5028362 0.2950211 mixture
#> 5     5  0.0007732515 0.1939808 -0.4008558 0.4035905 mixture
#> 6     6 -0.0350130729 0.2023950 -0.4787040 0.3599066 mixture
# }
```
