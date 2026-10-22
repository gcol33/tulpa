# Extract temporal effects from a fitted model

Extract posterior distributions of temporal effects from a fitted tulpa
model with temporal specification.

## Usage

``` r
# S3 method for class 'tulpa_temporal_field_fit'
print(x, ...)

# S3 method for class 'tulpa_temporal_field'
print(x, ...)

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

# S3 method for class 'tulpa_temporal_posterior'
print(x, ...)

# S3 method for class 'tulpa_temporal_posterior'
summary(object, probs = c(0.025, 0.5, 0.975), ...)

# S3 method for class 'tulpa_temporal_summary'
print(x, ...)
```

## Arguments

- x:

  A `temporal()` field specification, the fit
  [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)
  returns for one, a `tulpa_temporal_posterior` or a
  `tulpa_temporal_summary`.

- ...:

  Ignored

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

## Value

A `tulpa_temporal_posterior` object

[`summary()`](https://rdrr.io/r/base/summary.html) on a
`tulpa_temporal_posterior` returns a `tulpa_temporal_summary` data frame
with one row per time point (per component for multi-scale fits),
holding the posterior mean, SD and requested quantiles of the temporal
effect.

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

On a sampler fit the posterior is the field's own draws. On a
nested-Laplace fit it is the Gaussian mixture the outer grid defines:
each hyperparameter cell contributes its conditional mode and marginal
variance, weighted by the cell's posterior weight, and
[`summary()`](https://rdrr.io/r/base/summary.html) reports that
mixture's exact mean, SD and quantiles. The `draws` of such a fit are
sampled from the same mixture, each from its cell's joint Gaussian, so
they carry the correlation between time points as well as each point's
marginal, and a quantity derived from them (a trend, a difference
between two times) has the right spread. A field so large that its
per-cell covariance was not retained (more than 4 million values over
the grid) is drawn with independent time points within each cell
instead. The `time` column of the summary holds the time values the
field was fitted on.

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
