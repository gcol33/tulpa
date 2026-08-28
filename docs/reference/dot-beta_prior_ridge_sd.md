# Resolve a `beta_prior = list(mean, sd)` to a single ridge SD.

For the fitters whose fixed-effect prior is a mean-zero Gaussian ridge
with a scalar SD (EP, multinomial, ordinal): validates the unified
`beta_prior` object, enforces a mean of 0, and returns the positive
scalar SD. Keeps the shared prior interface
(`beta_prior = list(mean, sd)`) while rejecting the options those
fitters do not implement (a non-zero mean, a per-coefficient SD).
Validation of the supplied object itself is
[`.beta_prior_fields()`](https://gillescolling.com/tulpa/reference/dot-beta_prior_fields.md),
shared with
[`.normalize_beta_prior()`](https://gillescolling.com/tulpa/reference/dot-normalize_beta_prior.md).

## Usage

``` r
.beta_prior_ridge_sd(beta_prior, default_sd = .tulpa_prior_sd("ridge"))
```

## Arguments

- beta_prior:

  `list(mean, sd)`; `mean` must be 0, `sd` a positive scalar. `NULL`
  takes the engine default.

- default_sd:

  SD used when `beta_prior` is `NULL`. A supplied prior is resolved
  through
  [`.beta_prior_fields()`](https://gillescolling.com/tulpa/reference/dot-beta_prior_fields.md)
  and never falls back to this.
