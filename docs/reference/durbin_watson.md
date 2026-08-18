# Durbin-Watson test for temporal autocorrelation

Tests first-order autocorrelation in temporally-ordered residuals.

## Usage

``` r
durbin_watson(object, alternative = c("two.sided", "greater", "less"))
```

## Arguments

- object:

  A numeric vector of temporally-ordered residuals

- alternative:

  `"two.sided"`, `"greater"` (positive autocorr), or `"less"`

## Value

An `htest` object with DW statistic, lag-1 r, and p-value
