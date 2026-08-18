# Interpret an RE term's covariance specification

Maps one `re_list` element to the two representations the Laplace path
needs: `pack`, the value the C++ kernel consumes (a length-`n_coefs`
marginal-SD vector for a diagonal / uncorrelated term, or a packed
lower-triangular Cholesky of length `n_coefs (n_coefs + 1) / 2` for a
correlated one), and `Q`, the `n_coefs x n_coefs` RE precision
`Sigma^{-1}` used to build the marginal fixed-effect SE. A correlated
term is signalled by `r$L` (a lower- triangular Cholesky factor,
`Sigma = L L'`) or `r$cov` (the covariance matrix); when present these
take precedence over `r$sigma`.

## Usage

``` r
.re_cov_spec(r)
```

## Arguments

- r:

  One element of a `re_list` (see
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)).

## Value

`list(pack, Q, diagonal)`.
