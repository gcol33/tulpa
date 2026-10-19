# Durbin-Watson test for temporal autocorrelation

Tests first-order autocorrelation in temporally-ordered residuals.

## Usage

``` r
durbin_watson(
  object,
  alternative = c("two.sided", "greater", "less"),
  time = NULL,
  resid_type = "pearson"
)
```

## Arguments

- object:

  A fitted model, or a numeric vector of temporally-ordered residuals. A
  fit's residuals are in its data's row order, so pass `time` unless the
  rows already are in time order.

- alternative:

  `"two.sided"`, `"greater"` (positive autocorr), or `"less"`

- time:

  Optional time index, one per residual; the residuals are put in
  increasing `time` order before the statistic is formed. `NULL`
  (default) takes them in the order given.

- resid_type:

  Residual type if extracting from a model (default `"pearson"`), as for
  [`moran_i()`](https://gillescolling.com/tulpa/reference/moran_i.md).

## Value

An `htest` object with DW statistic, lag-1 r, and p-value

## Examples

``` r
set.seed(1)
e <- as.numeric(arima.sim(list(ar = 0.6), n = 100))
durbin_watson(e)
#> 
#>  Durbin-Watson test for temporal autocorrelation
#> 
#> data:  e
#> DW = 0.8137, lag-1 r = 0.59315, p-value = 3.002e-09
#> alternative hypothesis: two.sided
#> 
# Residuals recorded out of time order:
tt <- sample(100)
durbin_watson(e[tt], time = tt)
#> 
#>  Durbin-Watson test for temporal autocorrelation
#> 
#> data:  e[tt]
#> DW = 0.8137, lag-1 r = 0.59315, p-value = 3.002e-09
#> alternative hypothesis: two.sided
#> 
```
