# Moran's I test for spatial autocorrelation in residuals

Tests whether residuals exhibit spatial structure after model fitting.
Supports inverse-distance and k-nearest-neighbour weight matrices. No
external dependencies (uses normal approximation for inference).

## Usage

``` r
moran_i(
  object,
  coords,
  weights = c("inverse", "knn"),
  k = 10L,
  resid_type = "pearson",
  alternative = c("two.sided", "greater", "less")
)
```

## Arguments

- object:

  A fitted model, or a numeric vector of residuals

- coords:

  N x 2 coordinate matrix (required)

- weights:

  Weight scheme: `"inverse"` or `"knn"`

- k:

  Number of neighbours for knn (default 10)

- resid_type:

  Residual type if extracting from model (default `"pearson"`)

- alternative:

  `"two.sided"`, `"greater"`, or `"less"`

## Value

An `htest` object with Moran's I, expected I, and p-value

## Examples

``` r
set.seed(1)
coords <- cbind(runif(50), runif(50))
resid  <- rnorm(50)
moran_i(resid, coords)
#> 
#>  Moran's I (inverse-distance weights)
#> 
#> data:  resid
#> Moran's I = -0.07788, Expected I = -0.020408, p-value = 0.6056
#> alternative hypothesis: two.sided
#> 
```
