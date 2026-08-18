# Extract latent factor posteriors from fit

Extract latent factor posteriors from fit

## Usage

``` r
latent_factors(fit, summary = TRUE, probs = c(0.025, 0.5, 0.975))
```

## Arguments

- fit:

  A tulpa_fit object

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
if (FALSE) { # \dontrun{
# `fit` is a ratio / multi-arm model fitted with latent factors by a consumer
# package (tulpaRatio owns the two-arm formula and the negbin/negbin ratio
# family and reads the latent_factor() spec):

# Get summary
factors <- latent_factors(fit)
head(factors)

# Get full posterior draws
factor_draws <- latent_factors(fit, summary = FALSE)
} # }
```
