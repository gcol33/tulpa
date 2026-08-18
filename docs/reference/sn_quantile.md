# Skew-normal quantile

Inverse of the skew-normal CDF via Newton iteration on
[`sn_cdf()`](https://gillescolling.com/tulpa/reference/sn_cdf.md) with a
Brent-style bracket fallback when Newton fails to converge.

## Usage

``` r
sn_quantile(p, sn, tol = 1e-10, max_iter = 60L)
```

## Arguments

- p:

  Numeric vector of probabilities in \\\[0, 1\]\\.

- sn:

  Skew-normal parameter list from
  [`sn_match()`](https://gillescolling.com/tulpa/reference/sn_match.md).

- tol:

  Absolute tolerance on the CDF residual (default `1e-10`).

- max_iter:

  Maximum Newton iterations (default `60`).

## Value

Numeric vector of quantiles, same length as `p`.
