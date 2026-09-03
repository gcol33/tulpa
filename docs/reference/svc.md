# Extract spatially-varying coefficients from a fitted model

Extract posterior distributions of spatially-varying coefficients (SVCs)
from a fitted tulpa model with SVC specification.

## Usage

``` r
svc(object, terms = NULL, summary = FALSE, probs = c(0.025, 0.5, 0.975), ...)

# S3 method for class 'tulpa_fit'
svc(object, terms = NULL, summary = FALSE, probs = c(0.025, 0.5, 0.975), ...)
```

## Arguments

- object:

  A `tulpa_fit` object fitted with `svc` argument

- terms:

  Which SVC terms to extract. If NULL (default), extracts all.

- summary:

  Logical; if TRUE, return summary statistics instead of full posterior
  draws.

- probs:

  Quantiles to compute if `summary = TRUE`.

- ...:

  Ignored

## Value

A `tulpa_svc_posterior` object containing:

- `draws`: Array of posterior draws (draws x locations x terms)

- `coords`: Coordinate matrix

- `term_names`: Names of SVC terms

## See also

[`spatial_svc()`](https://gillescolling.com/tulpa/reference/spatial_svc.md),
[`plot.tulpa_svc_posterior()`](https://gillescolling.com/tulpa/reference/plot.tulpa_svc_posterior.md)

## Examples

``` r
# \donttest{
set.seed(303)
n <- 25L
df <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
bsurf <- 0.9 * sin(2.8 * df$lon) + 0.7 * cos(2.2 * df$lat)
df$count <- rpois(n, exp(0.2 + (0.8 + bsurf) * df$x))

# The varying slope on `x` is a spatial field; SVC is exact-mode only.
fit <- tulpa(
  count ~ x,
  data = df,
  family = "poisson",
  spatial = spatial_svc(~ lon + lat, terms = ~ x - 1, nn = 5L),
  mode = "exact",
  control = list(n_iter = 80L, n_warmup = 40L, seed = 1L)
)

svc_post <- svc(fit)
summary(svc_post)
# }
```
