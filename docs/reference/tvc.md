# Extract temporally-varying coefficients from a fitted model

Extract posterior distributions of temporally-varying coefficients
(TVCs) from a fitted tulpa model with TVC specification.

## Usage

``` r
tvc(object, terms = NULL, summary = FALSE, probs = c(0.025, 0.5, 0.975), ...)

# S3 method for class 'tulpa_fit'
tvc(object, terms = NULL, summary = FALSE, probs = c(0.025, 0.5, 0.975), ...)
```

## Arguments

- object:

  A `tulpa_fit` object fitted with `tvc` argument

- terms:

  Which TVC terms to extract. If NULL (default), extracts all.

- summary:

  Logical; if TRUE, return summary statistics instead of full posterior
  draws.

- probs:

  Quantiles to compute if `summary = TRUE`.

- ...:

  Ignored

## Value

A `tulpa_tvc_posterior` object containing:

- `draws`: Array of posterior draws (draws x times x terms)

- `time_levels`: Time point labels

- `term_names`: Names of TVC terms

## See also

[`temporal_tvc()`](https://gillescolling.com/tulpa/reference/temporal_tvc.md),
[`plot.tulpa_tvc_posterior()`](https://gillescolling.com/tulpa/reference/plot.tulpa_tvc_posterior.md)

## Examples

``` r
# \donttest{
set.seed(160)
n_t <- 10L; reps <- 5L
walk <- cumsum(rnorm(n_t, 0, 0.35)); walk <- walk - mean(walk)
year <- rep(seq_len(n_t), each = reps)
df <- data.frame(year = year, x = rnorm(length(year)))
df$count <- rpois(nrow(df), exp(0.3 + (0.5 + walk[year]) * df$x))

# The slope on `x` walks in time; TVC is exact-mode only.
fit <- tulpa(
  count ~ x,
  data = df,
  family = "poisson",
  temporal = temporal_tvc("year", terms = ~ x - 1, structure = "rw1"),
  mode = "exact",
  control = list(n_iter = 200L, n_warmup = 100L, seed = 1L)
)

tvc_post <- tvc(fit)
summary(tvc_post)
plot(tvc_post, "x")
# }
```
