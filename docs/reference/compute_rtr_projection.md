# Compute RTR projection matrix

Compute the orthogonal projection matrix P_perp = I - P_X that projects
the temporal effect into the space orthogonal to the covariates.

## Usage

``` r
compute_rtr_projection(X)
```

## Arguments

- X:

  Design matrix of covariates to orthogonalize against

## Value

Projection matrix (n x n)
