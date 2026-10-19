# Extract latent factor posteriors from fit

Extract latent factor posteriors from fit

## Usage

``` r
latent_factors(fit, ...)

# Default S3 method
latent_factors(fit, summary = TRUE, probs = c(0.025, 0.5, 0.975), ...)
```

## Arguments

- fit:

  A fit sampled with latent factors: its draw matrix (`$draws` or
  `$samples`) holds columns named `factor[observation,factor]` and
  `$.internal$latent_info` records the layout. The default method reads
  any fit, of any class, that stores them this way.

- ...:

  Passed to methods.

- summary:

  Logical; if TRUE, return summary statistics. If FALSE, return full
  posterior draws.

- probs:

  Quantiles for summary. Default is c(0.025, 0.5, 0.975).

## Value

If `summary = TRUE`, a data frame with columns for observation, factor,
and summary statistics. If `summary = FALSE`, a matrix of posterior
draws.

## Examples

``` r
# A fit sampled with latent factors comes from a model package (tulpaRatio
# reads the latent_factor() spec). The accessor reads two things off it:
# draw columns named factor[observation,factor] and the latent layout. Here
# both are simulated for three observations on two factors.
set.seed(1)
cols <- sprintf("factor[%d,%d]", rep(1:3, 2), rep(1:2, each = 3))
draws <- matrix(rnorm(200 * 6, mean = rep(c(-1, 0, 1), 2), sd = 0.3),
                200, 6, byrow = TRUE, dimnames = list(NULL, cols))
fit <- structure(
  list(draws = draws,
       .internal = list(latent_info = list(type = "latent", n_factors = 2L,
                                           n_obs = 3L))),
  class = "tulpa_fit")

latent_factors(fit)
#>   observation factor          mean        sd       q2.5          q50      q97.5
#> 1           1      1 -1.0038420351 0.3155887 -1.6878861 -1.018331764 -0.4234243
#> 2           2      1 -0.0055566152 0.2833776 -0.6571490 -0.004646812  0.5299950
#> 3           3      1  0.9837739415 0.3218027  0.3606486  0.965475809  1.6272686
#> 4           1      2 -1.0201955825 0.3251058 -1.5504414 -1.050918737 -0.3429517
#> 5           2      2 -0.0006869875 0.2996354 -0.6103986 -0.003331038  0.5344809
#> 6           3      2  1.0046982486 0.3104367  0.4031134  0.994218538  1.6496387
dim(latent_factors(fit, summary = FALSE))
#> [1] 200   6
```
