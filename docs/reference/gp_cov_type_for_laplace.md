# Map a spatial_gp covariance spec to the Laplace cov_type integer

The Laplace NNGP kernel (`laplace_core.cpp`) supports three covariance
functions: 0 = exponential, 1 = Matern(nu=1.5), 2 = Matern(nu=2.5).
Anything else is rejected with a clear error rather than silently
falling back to a different covariance.

## Usage

``` r
gp_cov_type_for_laplace(spatial)
```
