# Pack an RE term's covariance specification for the C++ kernel

Maps one `re_list` element to the value the multi-term Laplace kernels
consume: a length-`n_coefs` marginal-SD vector for a diagonal /
uncorrelated term, or a packed lower-triangular Cholesky of length
`n_coefs (n_coefs + 1) / 2` for a correlated one. A correlated term is
signalled by `r$L` (a lower-triangular Cholesky factor, `Sigma = L L'`)
or `r$cov` (the covariance matrix); when present these take precedence
over `r$sigma`.

## Usage

``` r
.re_cov_pack(r)
```

## Arguments

- r:

  One element of a `re_list` (see
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)).

## Value

The packed numeric vector.
