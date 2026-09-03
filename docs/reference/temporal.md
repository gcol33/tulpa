# Extract temporal effects from a fitted model

Extract posterior distributions of temporal effects from a fitted tulpa
model with temporal specification.

## Usage

``` r
temporal(
  object,
  component = "all",
  summary = FALSE,
  probs = c(0.025, 0.5, 0.975),
  ...
)

# S3 method for class 'tulpa_fit'
temporal(
  object,
  component = "all",
  summary = FALSE,
  probs = c(0.025, 0.5, 0.975),
  ...
)
```

## Arguments

- object:

  A `tulpa_fit` object fitted with `temporal` argument

- component:

  Which component to extract for multi-scale models: `"all"` (default),
  `"trend"`, `"seasonal"`, or `"short_term"`.

- summary:

  Logical; if TRUE, return summary statistics instead of full posterior
  draws.

- probs:

  Quantiles to compute if `summary = TRUE`.

- ...:

  Ignored

## Value

A `tulpa_temporal_posterior` object

## Details

`temporal()` is overloaded. Given a fitted model it is the accessor
described here. Given a one-sided formula (or a named `formula =` /
`structure =` argument) it is instead the inline varying-coefficient
field constructor used in a
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) model
formula, the temporal mirror of
[`spatial()`](https://gillescolling.com/tulpa/reference/spatial.md):
`temporal(formula = ~ 1 + x || time, structure = "rw1")` declares a
smooth temporal level (the intercept column) plus a temporally varying
slope on each covariate column. `structure` is one of `"rw1"` (default),
`"rw2"`, or `"ar1"`; only the double bar `||` (independent fields) is
supported.

## See also

[`temporal_multiscale()`](https://gillescolling.com/tulpa/reference/temporal_multiscale.md),
[`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md)

## Examples

``` r
# \donttest{
set.seed(131)
df <- data.frame(year = 1:40, x = rnorm(40))
df$count <- rpois(40, exp(1 + 0.2 * df$x))

fit <- tulpa(
  count ~ x,
  data = df,
  family = "poisson",
  temporal = temporal_multiscale("year", trend = "rw2", seasonal = 12),
  mode = "exact",
  control = list(n_iter = 200L, n_warmup = 100L, seed = 1L)
)

# Extract all temporal effects
temp_post <- temporal(fit)
summary(temp_post)
# }
```
