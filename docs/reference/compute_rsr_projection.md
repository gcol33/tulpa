# Compute RSR projection matrix

Compute the orthogonal projection matrix P_perp = I - P_X that projects
the spatial effect into the space orthogonal to the covariates.

## Usage

``` r
compute_rsr_projection(X)
```

## Arguments

- X:

  Design matrix of covariates to orthogonalize against

## Value

Projection matrix (n x n)
