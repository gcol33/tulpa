# Extract spatiotemporal effects from fitted model

Extract posterior distributions of spatiotemporal interaction effects
from a fitted tulpa model.

## Usage

``` r
spatiotemporal_effects(
  object,
  format = c("array", "long", "summary"),
  probs = c(0.025, 0.5, 0.975),
  ...
)

# S3 method for class 'tulpa_fit'
spatiotemporal_effects(
  object,
  format = c("array", "long", "summary"),
  probs = c(0.025, 0.5, 0.975),
  ...
)
```

## Arguments

- object:

  A `tulpa_fit` object fitted with `spatiotemporal` argument

- format:

  Output format: `"array"` (default, S x T x draws), `"long"` (data
  frame with s, t, draw, value columns), or `"summary"` (posterior
  summaries).

- probs:

  Quantiles to compute if `format = "summary"`.

- ...:

  Ignored

## Value

Spatiotemporal effects in requested format

## Examples

``` r
# \donttest{
# After fitting a model with spatiotemporal interaction...
# st_effects <- spatiotemporal_effects(fit)
# summary(st_effects)
# }
```
