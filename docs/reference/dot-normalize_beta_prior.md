# Normalize an optional fixed-effect Gaussian prior

Validates `beta_prior` and recycles scalar `mean` / `sd` to length `p`.
Returns `NULL` (use the built-in weak prior) or `list(mean, sd)` with
both vectors of length `p`. Shared by
[`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
and the EM driver so the validation rules live in one place.

## Usage

``` r
.normalize_beta_prior(beta_prior, p)
```

## Arguments

- beta_prior:

  `NULL`, or a list with `sd` (required) and optional `mean`.

- p:

  Number of fixed effects (`ncol(X)`).
