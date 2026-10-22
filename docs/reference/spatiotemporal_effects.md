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

# Default S3 method
spatiotemporal_effects(
  object,
  format = c("array", "long", "summary"),
  probs = c(0.025, 0.5, 0.975),
  ...
)

# S3 method for class 'tulpa_st_summary'
print(x, ...)
```

## Arguments

- object:

  A fit carrying a `$spatiotemporal` Knorr-Held interaction block
  (`n_spatial`, `n_times`) and its draws in
  `$.internal$spatiotemporal_draws`.
  [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) itself
  has no `spatiotemporal =` argument; such a block comes from a model
  package's own fitter (see
  [`spatiotemporal()`](https://gillescolling.com/tulpa/reference/spatiotemporal.md)
  for what the engine itself fits), and the default method reads any
  fit, of any class, that stores it this way.

- format:

  Output format: `"array"` (default, S x T x draws), `"long"` (data
  frame with s, t, draw, value columns), or `"summary"` (posterior
  summaries).

- probs:

  Quantiles to compute if `format = "summary"`.

- ...:

  Ignored

- x:

  A `tulpa_st_summary` table.

## Value

For `format = "array"`, an S x T x draws numeric array carrying
`n_spatial`, `n_times` and `n_draws` attributes; for `"long"`, a data
frame with columns `s`, `t`, `draw` and `value`; for `"summary"`, a
`tulpa_st_summary` data frame with columns `s`, `t`, `mean`, one column
per entry of `probs` and `sd`, which has
[`print()`](https://rdrr.io/r/base/print.html) and
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) methods.

## Examples

``` r
# A Knorr-Held interaction block is attached by a model package's own
# fitter (see spatiotemporal() for what the engine fits). The accessor reads
# its layout and an n_draws x (S * T) draw matrix stored s-fastest: column
# (t - 1) * S + s holds region s at time t. Here both are simulated for
# S = 3 regions over T = 4 times.
set.seed(1)
S <- 3L; n_t <- 4L
st_mean <- outer(c(-0.5, 0, 0.5), seq(-0.3, 0.3, length.out = n_t))
draws <- matrix(rnorm(100 * S * n_t, mean = rep(c(st_mean), each = 100),
                      sd = 0.1), nrow = 100)
fit <- structure(
  list(spatiotemporal = list(n_spatial = S, n_times = n_t),
       .internal = list(spatiotemporal_draws = draws)),
  class = "tulpa_fit")

st_effects <- spatiotemporal_effects(fit, format = "summary")
head(st_effects)
dim(spatiotemporal_effects(fit, format = "array"))
```
