# Map standardised CCD coordinates to physical hyperparameters

Converts standardised CCD z-coordinates (in \\\mathbb{R}^k\\) to
physical hyperparameters \\\theta\\ via the affine map \\\theta =
\hat\theta + L \cdot z\\ with optional log-scale transform per
component. `L` is typically a Cholesky factor of the negative Hessian
inverse evaluated at the (working) mode \\\hat\theta\\ – i.e. it scales
`z` to one posterior standard deviation per axis.

## Usage

``` r
ccd_to_theta(z, theta_hat, L, log_scale = FALSE)
```

## Arguments

- z:

  Matrix `[n_points x k]` of standardised coordinates from
  [`ccd_grid()`](https://gillescolling.com/tulpa/reference/ccd_grid.md).

- theta_hat:

  Numeric vector of length `k`: centre of the design, in either physical
  or log-space (per `log_scale`).

- L:

  Numeric `[k x k]` matrix: scale/rotation applied to z. Pass `diag(sd)`
  for a diagonal axis-aligned grid where `sd` is the per-axis posterior
  SD; pass a Cholesky factor to capture correlations between
  hyperparameters.

- log_scale:

  Logical (or logical vector of length k). If `TRUE` for component j,
  `theta_hat[j]` and column j of the affine transform live on log-scale
  and are exponentiated afterward (useful for positive parameters like
  tau). Default `FALSE` everywhere.

## Value

Numeric matrix `[n_points x k]` of physical theta-values.

## Examples

``` r
g <- ccd_grid(2)
ccd_to_theta(g$z, theta_hat = c(0, 1), L = diag(c(0.5, 0.2)))
#>             [,1]      [,2]
#>  [1,]  0.0000000 1.0000000
#>  [2,]  0.7071068 1.0000000
#>  [3,] -0.7071068 1.0000000
#>  [4,]  0.0000000 1.2828427
#>  [5,]  0.0000000 0.7171573
#>  [6,] -0.5000000 0.8000000
#>  [7,]  0.5000000 0.8000000
#>  [8,] -0.5000000 1.2000000
#>  [9,]  0.5000000 1.2000000
```
