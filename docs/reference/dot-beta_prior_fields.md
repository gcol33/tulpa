# Resolve a supplied fixed-effect prior to its Gaussian `(mean, sd)` fields

The one place that decides whether a `beta_prior` argument can be
expressed as a Gaussian prior on the fixed effects at all.
[`.normalize_beta_prior()`](https://gillescolling.com/tulpa/reference/dot-normalize_beta_prior.md)
(per-coefficient) and
[`.beta_prior_ridge_sd()`](https://gillescolling.com/tulpa/reference/dot-beta_prior_ridge_sd.md)
(scalar ridge) both resolve through it and differ only in the shape they
recycle the fields to, so a prior one accepts is a prior the other
accepts.

## Usage

``` r
.beta_prior_fields(beta_prior)
```

## Arguments

- beta_prior:

  A list (or `tulpa_prior` object) carrying `sd` and optionally `mean`.

## Value

`list(mean, sd)`, unrecycled.

## Details

A prior with no `sd` is an input the fitters cannot express:
substituting the default for it would replace the user's modelling
statement with a different one and leave no trace on the posterior.
