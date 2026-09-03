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
  from the adjacency matrix following Riebler et al.

- parameterization:

  `"standard"` (default) or `"collapsed"` (deprecated).

## Value

A `tulpa_spatial` object

## References

Riebler, A., Sorbye, S. H., Simpson, D., & Rue, H. (2016). An intuitive
Bayesian spatial model for disease mapping that accounts for scaling.
Statistical Methods in Medical Research, 25(4), 1145-1165.

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
# }
```
