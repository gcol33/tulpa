# Build the sparse FEM stiffness / projector from a spatial_spde spec

Prefers the stored `Matrix` objects (`spatial$G`, `spatial$A`); falls
back to rebuilding them from the pre-extracted CSC slots so a spec
carrying only the slots still assembles.

## Usage

``` r
.spde_fem_matrices(spatial)
```
