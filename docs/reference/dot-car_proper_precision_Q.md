# Proper-CAR precision Q = tau \* (D - rho W) at fixed (tau, rho).

D is the diagonal degree matrix (neighbour counts) and W the adjacency,
so Q has the same nonzero structure as the ICAR precision with rho
scaling the off-diagonals. Matches `tulpa::add_car_proper_prior` / the
nested CAR_proper precision builder.

## Usage

``` r
.car_proper_precision_Q(spatial, tau, rho)
```
