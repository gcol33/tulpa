# Validate a 0-based/1-based random-effect index against its group count.

`re_idx` addresses one grouping factor: `0` marks "no random effect" and
`1..n_re_groups` a group. An out-of-range id would index out of bounds
in the C++ kernel, so bound it in R.

## Usage

``` r
.validate_re_idx(re_idx, n_re_groups, N, where)
```

## Arguments

- re_idx:

  Per-observation group index (length `N`).

- n_re_groups:

  Number of groups.

- N:

  Expected length.

- where:

  Caller name for the error message.

## Value

`re_idx` coerced to an integer vector.
