# Compute nearest neighbors for NNGP

Compute the k nearest neighbors for each observation using Euclidean
distance. Returns in a format suitable for the NNGP likelihood.

## Usage

``` r
compute_nngp_neighbors(coords, k)
```

## Arguments

- coords:

  N x 2 matrix of coordinates

- k:

  Number of nearest neighbors

## Value

List with:

- `nn_idx`: N x k matrix of neighbor indices (0 for obs with fewer
  neighbors)

- `nn_dist`: N x k matrix of distances to neighbors

- `nn_order`: Ordering of observations for NNGP (by coordinate)
