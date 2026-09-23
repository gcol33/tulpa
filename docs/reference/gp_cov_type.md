# Map a spatial_gp covariance spec to the engine's cov_type code

One code per kernel, read identically by every path that forms a
covariance from it – the Laplace NNGP scatter, the Polya-Gamma sweep,
the exact-NUTS NNGP/SVC kernels and the field predictor
(`tulpa::CovType` in `inst/include/tulpa/types.h`, dispatched by
`cov_value` in `inst/include/tulpa/cov_kernel.h`): 0 = exponential, 1 =
Matern(nu = 1.5), 4 = Matern(nu = 2.5). The smoothness travels IN the
code, so no path has to read a `nu` beside it to know which kernel it is
evaluating. Anything else is rejected here rather than silently fitted
as a different covariance.

## Usage

``` r
gp_cov_type(spatial)
```
