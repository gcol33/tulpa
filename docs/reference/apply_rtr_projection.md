# Apply RTR projection to temporal effect

Project temporal effect into the space orthogonal to covariates. Called
during posterior computation.

## Usage

``` r
apply_rtr_projection(f, P_perp)
```

## Arguments

- f:

  Temporal effect vector (length n)

- P_perp:

  Projection matrix from compute_rtr_projection

## Value

Projected temporal effect (length n)
