# Extract spatial range and variance from a fitted spatial model

Summarises the posterior of spatial hyperparameters. For sampler-tier
fits this reads the raw hyperparameter draws; for a nested-Laplace
spatial fit it summarises the outer hyperparameter grid. Works with
ICAR, BYM2, GP (NNGP), CAR, SPDE, and SVC spatial types.

## Usage

``` r
spatial_range(object, probs = c(0.025, 0.975))
```

## Arguments

- object:

  A `tulpa_fit` object fitted with a spatial component.

- probs:

  Quantile probabilities for the summary (default 0.025, 0.975).

## Value

A data.frame with rows for each spatial hyperparameter and columns
`mean`, `sd`, and one quantile column per entry of `probs` (named from
the probability, e.g. `q2.5`, `q97.5` for the defaults).

## Details

A range is reported in the units of the coordinates as supplied, also
when the field was fitted on standardized coordinates
(`scale_coords = TRUE`, the default of
[`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md)
and
[`spatial_svc()`](https://gillescolling.com/tulpa/reference/spatial_svc.md)).
A `mode = "laplace"` fit conditions on the field's hyperparameters
rather than inferring them, so it carries no posterior for this function
to summarise.

Physical range moments are determined by the model's prior tail and
likelihood limit. A divergent mean or SD is reported as `Inf`.
`object$moment_existence` carries the reason. Quantiles remain
available; the summary's `log_moments` attribute gives finite log-range
means and SDs.

## Examples

``` r
# \donttest{
set.seed(1)
S <- 12L
adj <- matrix(0, S, S)
for (i in 1:(S - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1
d <- data.frame(region = factor(rep(1:S, each = 5)), x = rnorm(S * 5))
field <- as.numeric(scale(cumsum(rnorm(S, 0, 0.5))))
d$y <- rpois(nrow(d), exp(0.3 + 0.4 * d$x + field[d$region]))
fit <- tulpa(y ~ x + spatial(region), data = d, family = "poisson",
             spatial = spatial_car(adj, level = "group", group_var = "region"))
#> [nested-laplace] 1/9 cells (11%) | elapsed 0s | ETA >=0s | 0.00s/cells
#> [nested-laplace] 8/8 cells (100%) | elapsed 0s | ETA done | 0.00s/cells
spatial_range(fit)
#>            mean        sd      q2.5   q97.5
#> sigma 0.6660589 0.2017354 0.3649507 1.14908
# }
```
