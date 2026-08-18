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
if (FALSE) { # \dontrun{
# Generate synthetic spatial data (not run - SVC not fully supported)
set.seed(303)
n <- 40
df <- data.frame(
  lon = runif(n, 0, 10),
  lat = runif(n, 0, 10),
  depth = rnorm(n),
  count = rpois(n, 20),
  effort = rgamma(n, shape = 4, rate = 1)
)

# Fit model with SVC
fit <- tulpa(
  count | effort ~ depth,
  data = df,
  family = tulpaRatio::tulpa_poisson_gamma(),
  svc = spatial_svc(~ lon + lat, terms = c(1, 2)),
  iter = 200, warmup = 100, chains = 1
)

# Extract SVC posteriors
svc_post <- svc(fit)
summary(svc_post)
} # }
```
