# BYM2 spatial structure

Specify a Besag-York-Mollie 2 (BYM2) spatial random effect. BYM2
decomposes the spatial effect into a structured (ICAR) component and an
unstructured (IID) component, with a mixing parameter controlling the
proportion of variance attributable to spatial structure.

BYM2 is preferred over plain CAR when you want to:

- Distinguish structured vs unstructured spatial variation

- Have an interpretable spatial fraction parameter

- Use the scaling from Riebler et al. (2016)

## Usage

``` r
spatial_bym2(
  adjacency,
  level = c("group", "obs"),
  group_var = NULL,
  shared = NULL,
  scale_factor = NULL,
  parameterization = c("standard", "collapsed")
)
```

## Arguments

- adjacency:

  Symmetric adjacency matrix (`[n_units x n_units]`).

- level:

  Either `"group"` (one effect per level of `group_var`) or `"obs"` (one
  effect per row of the data; `nrow(data)` must equal
  `nrow(adjacency)`).

- group_var:

  Name of the grouping variable in the data; required when
  `level = "group"`.

- shared:

  Optional shared-effect handle (see model docs).

- scale_factor:

  Scaling factor for the ICAR component. If NULL (default), computed
  from the adjacency matrix following Riebler et al. On a graph with
  several connected components each component is scaled to unit
  generalized variance separately and an isolated node (island) gets
  unit variance on its structured part, following Freni-Sterrantino et
  al. (2018). A supplied value is one scale for the whole graph.

- parameterization:

  `"standard"` (default) or `"collapsed"` (deprecated).

## Value

A `tulpa_spatial` object

## References

Riebler, A., Sorbye, S. H., Simpson, D., & Rue, H. (2016). An intuitive
Bayesian spatial model for disease mapping that accounts for scaling.
Statistical Methods in Medical Research, 25(4), 1145-1165.

Freni-Sterrantino, A., Ventrucci, M., & Rue, H. (2018). A note on
intrinsic conditional autoregressive models for disconnected graphs.
Spatial and Spatio-temporal Epidemiology, 26, 25-34.

## Examples

``` r
# Create adjacency matrix for 10 regions (chain structure)
adj <- matrix(0, 10, 10)
for (i in 1:9) {
  adj[i, i+1] <- adj[i+1, i] <- 1
}

# Create BYM2 spatial structure
bym2 <- spatial_bym2(adj, level = "group", group_var = "region")
print(bym2)
#> tulpa spatial specification
#> ===========================
#> 
#> Type: BYM2 
#> Level: group 
#> Spatial units: 10 
#> Shared: Yes (enters both processes) 
#> Group variable: region 
#> Scale factor: 0.8169 

# \donttest{
# Disease mapping with BYM2 spatial smoothing
set.seed(456)
n_regions <- 10
epi_data <- data.frame(
  region = factor(rep(1:n_regions, each = 4)),
  age = rnorm(n_regions * 4, 50, 10)
)
epi_data$cases <- rbinom(nrow(epi_data), size = 100, prob = 0.15)

fit <- tulpa(
  cases ~ age + spatial(region),
  spatial = spatial_bym2(adj, level = "group", group_var = "region"),
  data = epi_data,
  family = "binomial",
  n_trials = rep(100L, nrow(epi_data)),
  mode = "laplace"
)
summary(fit)
#> tulpa fit  (laplace, tier 2) 
#> Call: tulpa(formula = cases ~ age + spatial(region), data = epi_data, family = "binomial", mode = "laplace", n_trials = rep(100L, nrow(epi_data)), spatial = spatial_bym2(adj, level = "group", group_var = "region"))
#>   inference mode: structured 
#>   selected because: User-specified backend: laplace 
#>   family: binomial 
#>   n_obs: 40 
#> 
#> Fixed effects:
#>              estimate std.error     2.5 %   97.5 %
#> (Intercept) -1.957898  0.331077 -2.606797 -1.30900
#> age          0.006391  0.004661 -0.002745  0.01553
#> 
#> Reliability (iid draws):
# }
```
