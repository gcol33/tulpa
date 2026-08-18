# Apply RSR projection to spatial effect

Project spatial effect into the space orthogonal to covariates. Called
during posterior computation.

## Usage

``` r
apply_rsr_projection(w, P_perp)
```

## Arguments

- w:

  Spatial effect vector (length n)

- P_perp:

  Projection matrix from compute_rsr_projection

## Value

Projected spatial effect (length n)
