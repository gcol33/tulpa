# Build model matrices from a parsed formula

Takes a parsed formula and data frame, and constructs:

- The fixed-effects design matrix X

- An offset vector (or NULL) extracted from `offset(...)` terms

- RE group index vectors

- RE slope matrices (if applicable)

## Usage

``` r
tulpa_build_model_data(parsed, data)
```

## Arguments

- parsed:

  A `tulpa_parsed_formula` object

- data:

  A data frame

## Value

A list with:

- `y`: response vector (or NULL). A `cbind(successes, failures)`
  response is resolved to the successes column.

- `n_trials`: binomial denominators when the response was written as
  `cbind(successes, failures)`, otherwise NULL

- `X`: fixed-effects design matrix

- `offset`: numeric vector or NULL

- `re_terms`: list of RE data structures (group indices, slope matrices)
