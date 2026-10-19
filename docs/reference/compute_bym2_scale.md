# Compute BYM2 scaling factor

Compute the scaling factor for BYM2 following Riebler et al. (2016).
This makes the spatial fraction parameter interpretable. On a
disconnected graph each connected component is scaled separately and an
isolated node (an island) gets unit variance (Freni-Sterrantino et al.
2018); the value returned is then the reference scale of the largest
component, and `.bym2_component_scaling()` carries the per-node
remainder.

## Usage

``` r
compute_bym2_scale(adjacency)
```

## Arguments

- adjacency:

  Adjacency matrix

## Value

Scaling factor (scalar)
