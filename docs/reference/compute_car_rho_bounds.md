# Compute valid bounds for rho in proper CAR

For proper CAR, rho must be in the range (1/lambda_min, 1/lambda_max)
where lambda are the eigenvalues of D^(-1)W. In practice, we typically
restrict to (0, 1) for interpretability (positive spatial
autocorrelation).

## Usage

``` r
compute_car_rho_bounds(adjacency)
```

## Arguments

- adjacency:

  Adjacency matrix

## Value

Named vector with `lower` and `upper` bounds for rho
