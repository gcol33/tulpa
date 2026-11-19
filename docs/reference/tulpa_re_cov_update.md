# Closed-form covariance update of a grouped random-effect EM

The covariance update every tulpa EM over a grouped random effect uses
([`tulpa_re_em()`](https://gillescolling.com/tulpa/reference/tulpa_re_em.md),
and compiled consumers through `<tulpa/re_cov_update.h>`): the
group-averaged posterior second moment of one covariance block,
symmetrized, projected to its diagonal for an uncorrelated block, with
its eigenvalues floored at `sigma_floor^2`.

## Usage

``` r
tulpa_re_cov_update(M, correlated = TRUE, sigma_floor = 1e-04)
```

## Arguments

- M:

  The `c x c` group-averaged posterior second moment
  `mean_g(E[b_g] E[b_g]' + Cov(b_g))` of the block.

- correlated:

  `FALSE` keeps only the diagonal.

- sigma_floor:

  Floor on every standard deviation, applied to the eigenvalues.

## Value

The updated `c x c` covariance.

## Examples

``` r
M <- matrix(c(0.5, 0.2, 0.2, 1e-9), 2, 2)
tulpa_re_cov_update(M)
#>           [,1]       [,2]
#> [1,] 0.5076867 0.17808688
#> [2,] 0.1780869 0.06246951
tulpa_re_cov_update(M, correlated = FALSE, sigma_floor = 0.01)
#>      [,1]  [,2]
#> [1,]  0.5 0e+00
#> [2,]  0.0 1e-04
```
