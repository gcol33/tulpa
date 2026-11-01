# Draw Polya-Gamma random variates

Vectorized `PG(b, z)` draws via the Polson, Scott & Windle (2013)
sampler (`tulpa::rpg_vec()`), the kernel every Polya-Gamma Gibbs fitter
in this package draws its auxiliary weights from. Exposed as a door for
consumer packages fitting their own Polya-Gamma Gibbs models, which
cannot reach an RNG through `LinkingTo` the way a likelihood kernel
would.

## Usage

``` r
tulpa_rpg(b, z)
```

## Arguments

- b:

  Integer vector of PG shape parameters (trial counts); one draw per
  element.

- z:

  Numeric vector of tilting parameters, the same length as `b`.

## Value

A numeric vector of `PG(b, z)` draws, the same length as `b`.

## Examples

``` r
tulpa_rpg(rep(1L, 5), rnorm(5))
#> [1] 0.06843288 0.10272653 0.17211161 0.07873079 0.31075252
```
