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
if (FALSE) { # \dontrun{
# Generate synthetic data (not run - TVC experimental)
set.seed(160)
df <- data.frame(
  year = rep(2015:2024, each = 5),
  x = rnorm(50),
  count = rpois(50, lambda = 18),
  effort = rgamma(50, shape = 3.5, rate = 1)
)

# Fit model with TVC
fit <- tulpa(
  count | effort ~ x,
  data = df,
  family = tulpaRatio::tulpa_poisson_gamma(),
  tvc = temporal_tvc("year", terms = c(1, 2)),
  backend = "hmc",
  iter = 200,
  warmup = 100,
  chains = 1
)

# Extract TVC posteriors
tvc_post <- tvc(fit)
summary(tvc_post)

# Plot temporal evolution
plot(tvc_post, "x")
} # }
```
