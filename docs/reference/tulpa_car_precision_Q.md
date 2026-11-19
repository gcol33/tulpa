# CAR precision matrix of an adjacency graph

The conditional autoregressive precision `Q = tau * (D - rho * W)`, with
`W` the adjacency matrix and `D` the diagonal matrix of neighbour
counts. At `rho = 1` it is the intrinsic (ICAR) precision, the graph
Laplacian, whose null space holds one constant direction per connected
component; it is returned without an identifying augmentation, so a
caller imposes the sum-to-zero constraints itself. For `rho` strictly
inside the range set by the eigenvalues of `D^{-1/2} W D^{-1/2}` (in
particular `0 <= rho < 1` on a graph without isolated nodes) it is a
proper, full-rank precision.

## Usage

``` r
tulpa_car_precision_Q(adjacency, rho = 1, tau = 1)
```

## Arguments

- adjacency:

  Square symmetric 0/1 adjacency matrix (dense or sparse).

- rho:

  Spatial dependence parameter.

- tau:

  Precision scale.

## Value

A sparse symmetric precision matrix (`dsCMatrix`).

## See also

[`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md),
[`spatial_car_proper()`](https://gillescolling.com/tulpa/reference/spatial_car_proper.md),
[`tulpa_latent_ascent()`](https://gillescolling.com/tulpa/reference/tulpa_latent_ascent.md)

## Examples

``` r
A <- matrix(0, 4, 4); A[cbind(1:3, 2:4)] <- 1; A <- A + t(A)
tulpa_car_precision_Q(A)              # path-graph Laplacian
#> 4 x 4 sparse Matrix of class "dsCMatrix"
#>                 
#> [1,]  1 -1  .  .
#> [2,] -1  2 -1  .
#> [3,]  . -1  2 -1
#> [4,]  .  . -1  1
tulpa_car_precision_Q(A, rho = 0.9)   # proper CAR
#> 4 x 4 sparse Matrix of class "dsCMatrix"
#>                         
#> [1,]  1.0 -0.9  .    .  
#> [2,] -0.9  2.0 -0.9  .  
#> [3,]  .   -0.9  2.0 -0.9
#> [4,]  .    .   -0.9  1.0
```
