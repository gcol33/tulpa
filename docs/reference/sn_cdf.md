# Skew-normal CDF

Cumulative distribution function of the univariate skew-normal, \$\$F(q;
\xi, \omega, \alpha) = \Phi(z) - 2 T(z, \alpha),\$\$ where \\z = (q -
\xi) / \omega\\ and \\T\\ is Owen's T function. Owen's T is evaluated by
base R numerical quadrature; no extra dependencies are required.

## Usage

``` r
sn_cdf(q, sn)
```

## Arguments

- q:

  Numeric vector of quantiles.

- sn:

  Skew-normal parameter list from
  [`sn_match()`](https://gillescolling.com/tulpa/reference/sn_match.md)
  with elements `xi`, `omega`, `alpha`.

## Value

Numeric vector of CDF values, same length as `q`.
