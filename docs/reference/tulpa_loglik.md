# Streaming pointwise log-likelihood

Wrap a pointwise log-likelihood for
[`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)
without materializing the whole `[n_draws x n_obs]` matrix. A plain
matrix is wrapped directly; a block generator (a function of an integer
column vector returning the `[n_draws x length(cols)]` submatrix) lets
the criteria accumulators stream over observation blocks, so an
EVA-scale `[200 x 1.16M]` log-likelihood is consumed a few thousand
columns at a time.

## Usage

``` r
tulpa_loglik(x, n_obs = NULL, n_draws = NULL)
```

## Arguments

- x:

  Either a numeric `[n_draws x n_obs]` matrix, an existing
  `tulpa_loglik`, or a function `f(cols)` returning the
  `[n_draws x length(cols)]` submatrix for the integer column indices
  `cols`.

- n_obs, n_draws:

  Required when `x` is a generator function; the column and row counts
  of the implied matrix.

## Value

A `tulpa_loglik` object: a list with `get(cols)`, `n_obs`, `n_draws`,
and `materialized`.

## See also

[`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)
