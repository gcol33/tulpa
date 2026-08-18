# Spatiotemporal interaction specifications for tulpa

Functions to specify spatiotemporal interaction effects for tulpa
models. These capture dependencies that arise when spatial patterns vary
over time, or when temporal trends differ across space.

Specify a spatiotemporal interaction effect for tulpa models. The
interaction captures structured or unstructured deviation from the
additive spatial + temporal model.

## Usage

``` r
spatiotemporal(
  spatial,
  temporal,
  type = c("I", "II", "III", "IV", "iid", "separable"),
  shared = NULL
)
```

## Arguments

- spatial:

  A spatial specification from
  [`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md),
  [`spatial_bym2()`](https://gillescolling.com/tulpa/reference/spatial_bym2.md),
  or
  [`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md).

- temporal:

  A temporal specification from
  [`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md),
  [`temporal_rw2()`](https://gillescolling.com/tulpa/reference/temporal_rw2.md),
  [`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md),
  or
  [`temporal_gp()`](https://gillescolling.com/tulpa/reference/temporal_gp.md).

- type:

  Interaction type:

  - `"I"` or `"iid"`: Unstructured interaction (IID)

  - `"II"`: Structured time at each location

  - `"III"`: Structured space at each time point

  - `"IV"`: Fully structured (Kronecker product of spatial and temporal)

  - `"separable"`: Separable covariance (Kronecker product)

- shared:

  Logical; if TRUE (default), spatiotemporal effect enters both all
  processes. Set to FALSE for process-specific effects (triggers warning
  about potential confounding).

## Value

A `tulpa_spatiotemporal` object

## Details

Spatiotemporal interactions extend the basic additive model:

\$\$\eta\_{st} = X\beta + f_s(space) + f_t(time)\$\$

to include interactions:

\$\$\eta\_{st} = X\beta + f_s(space) + f_t(time) + \delta\_{st}\$\$

where \\\delta\_{st}\\ captures space-time interactions.

**Interaction Types (following Knorr-Held, 2000):**

- **Type I**: Unstructured interaction - IID \\\delta\_{st} \sim N(0,
  \sigma^2)\\

- **Type II**: Structured time, unstructured space - temporal structure
  at each location

- **Type III**: Structured space, unstructured time - spatial structure
  at each time

- **Type IV**: Structured space AND time - full Kronecker interaction

**Separable Models:**

- **Separable**: Covariance is Kronecker product \\C\_{st} = C_s \otimes
  C_t\\

- **Non-separable**: GP with joint space-time metric

**Type I (IID)**

Independent random effect for each space-time combination:
\$\$\delta\_{st} \stackrel{iid}{\sim} N(0, \sigma^2\_\delta)\$\$

This is the simplest form, requiring S\*T parameters but capturing no
structured interaction.

**Type II (Temporal structure per location)**

Each location has its own temporal random effect: \$\$\delta\_{\cdot
t}^{(s)} \sim RW(\sigma^2)\$\$

This captures location-specific temporal trends but assumes independence
across locations.

**Type III (Spatial structure per time point)**

Each time point has its own spatial random effect: \$\$\delta\_{s
\cdot}^{(t)} \sim ICAR(\tau)\$\$

This captures time-specific spatial patterns but assumes independence
across time points.

**Type IV (Full structure)**

Kronecker product of spatial and temporal precision matrices:
\$\$Q\_\delta = Q_s \otimes Q_t\$\$

This is the most constrained model, assuming the interaction has the
same structure as the marginal effects.

**Separable**

For GP-based effects, assumes separable covariance: \$\$C(\mathbf{s}\_1,
t_1; \mathbf{s}\_2, t_2) = C_s(\mathbf{s}\_1, \mathbf{s}\_2) \cdot
C_t(t_1, t_2)\$\$

## References

Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time
variation in disease risk. Statistics in Medicine, 19(17-18), 2555-2567.

## See also

[`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md),
[`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md),
[`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md),
[`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md)

## Examples

``` r
# Create adjacency matrix for 10 regions
adj <- matrix(0, 10, 10)
for (i in 1:9) adj[i, i+1] <- adj[i+1, i] <- 1

# Type I: Unstructured interaction
st1 <- spatiotemporal(
  spatial = spatial_car(adj, level = "group", group_var = "region"),
  temporal = temporal_rw1("year"),
  type = "I"
)
print(st1)
#> tulpa Spatiotemporal Interaction Specification
#> ===============================================
#> 
#> Interaction type: Type I: Unstructured (IID) 
#> 
#> Spatial component:
#>   Type: tulpa_spatial 
#>   Group variable: region 
#> 
#> Temporal component:
#>   Type: rw1 
#>   Time variable: year 
#> 
#> Shared: Yes (enters both processes) 

# Type IV: Fully structured interaction
st4 <- spatiotemporal(
  spatial = spatial_car(adj, level = "group", group_var = "region"),
  temporal = temporal_rw1("year"),
  type = "IV"
)
print(st4)
#> tulpa Spatiotemporal Interaction Specification
#> ===============================================
#> 
#> Interaction type: Type IV: Fully structured (Kronecker) 
#> 
#> Spatial component:
#>   Type: tulpa_spatial 
#>   Group variable: region 
#> 
#> Temporal component:
#>   Type: rw1 
#>   Time variable: year 
#> 
#> Shared: Yes (enters both processes) 

if (FALSE) { # \dontrun{
# Generate synthetic spatiotemporal data (not run - experimental)
set.seed(123)
n_regions <- 10
n_years <- 8
df <- expand.grid(
  region = 1:n_regions,
  year = 2015:(2015 + n_years - 1)
)
df$x <- rnorm(nrow(df))
df$count <- rpois(nrow(df), lambda = 20)
df$effort <- rgamma(nrow(df), shape = 4, rate = 1)

# Fit model with spatiotemporal interaction
fit <- tulpa(
  count | effort ~ x,
  data = df,
  family = tulpaRatio::tulpa_poisson_gamma(),
  spatiotemporal = spatiotemporal(
    spatial = spatial_car(adj, level = "group", group_var = "region"),
    temporal = temporal_rw1("year"),
    type = "IV"
  ),
  iter = 200, warmup = 100, chains = 1
)
summary(fit)
} # }
```
