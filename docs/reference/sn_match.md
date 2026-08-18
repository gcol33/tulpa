# Match three cumulants to a skew-normal parameterisation

Inverts the skew-normal moment formulas to convert `(mu, sigma, gamma)`
into `(xi, omega, alpha)` parameters. Returns `NULL` with a warning when
`|gamma|` exceeds the skew-normal ceiling (~0.9953) – in that regime the
third cumulant cannot be matched by any skew-normal and the caller
should fall back to direct-quadrature quantiles.

## Usage

``` r
sn_match(mu, sigma, gamma)
```

## Arguments

- mu:

  Posterior mean (numeric, length 1).

- sigma:

  Posterior standard deviation (positive numeric, length 1).

- gamma:

  Posterior skewness (numeric, length 1).

## Value

Named list with elements `xi`, `omega`, `alpha`; or `NULL` if
`|gamma| >= .SN_GAMMA_MAX`.

## References

Azzalini, A. (1985). A class of distributions which includes the normal
ones. *Scand. J. Statist.* **12**: 171-178.
