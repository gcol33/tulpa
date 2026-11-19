# Skew-normal distribution matched to three cumulants

`sn_match()` inverts the skew-normal moment formulas to convert
`(mu, sigma, gamma)` into `(xi, omega, alpha)` parameters. It returns
`NULL` with a warning when `|gamma|` exceeds the skew-normal ceiling
(~0.9953) – in that regime the third cumulant cannot be matched by any
skew-normal and the caller should fall back to direct-quadrature
quantiles.

`sn_cdf()` is the cumulative distribution function of the univariate
skew-normal, \$\$F(q; \xi, \omega, \alpha) = \Phi(z) - 2 T(z,
\alpha),\$\$ where \\z = (q - \xi) / \omega\\ and \\T\\ is Owen's T
function. Owen's T is evaluated by base R numerical quadrature; no extra
dependencies are required.

`sn_quantile()` inverts the skew-normal CDF by Newton iteration on
`sn_cdf()`, with a Brent-style bracket fallback when Newton fails to
converge.

## Usage

``` r
sn_match(mu, sigma, gamma)

sn_cdf(q, sn)

sn_quantile(p, sn, tol = 1e-10, max_iter = 60L)
```

## Arguments

- mu:

  Posterior mean (numeric, length 1).

- sigma:

  Posterior standard deviation (positive numeric, length 1).

- gamma:

  Posterior skewness (numeric, length 1).

- q:

  Numeric vector of quantiles.

- sn:

  Skew-normal parameter list from `sn_match()` with elements `xi`,
  `omega`, `alpha`.

- p:

  Numeric vector of probabilities in \\\[0, 1\]\\.

- tol:

  Absolute tolerance on the CDF residual (default `1e-10`).

- max_iter:

  Maximum Newton iterations (default `60`).

## Value

`sn_match()`: a named list with elements `xi`, `omega`, `alpha`; or
`NULL` if `|gamma| >= .SN_GAMMA_MAX`. `sn_cdf()`: a numeric vector of
CDF values, same length as `q`. `sn_quantile()`: a numeric vector of
quantiles, same length as `p`.

## References

Azzalini, A. (1985). A class of distributions which includes the normal
ones. *Scand. J. Statist.* **12**: 171-178.
