# Validate a GLM design bundle (`y`, `X`, `n_trials`) at a fitter's front door.

Shared by the flagship drivers
([`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md),
[`tulpa_gibbs()`](https://gillescolling.com/tulpa/reference/tulpa_gibbs.md),
the `re_cov` fitters) so `nrow(X) == length(y)` and
`length(n_trials) == length(y)` are enforced in one place – otherwise a
mismatched `n_trials` (`as.integer(NULL)` -\> `integer(0)`) reaches the
C++ kernel silently.

## Usage

``` r
.validate_glm_design(y, X, n_trials, where)
```

## Arguments

- y, X:

  Response vector and design matrix.

- n_trials:

  Binomial denominators, or `NULL` (defaults to 1 per row).

- where:

  Caller name for the error message.

## Value

List `list(N, n_trials)` with `n_trials` coerced to a length-`N` integer
vector.
