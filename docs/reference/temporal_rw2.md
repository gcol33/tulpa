# RW2 temporal structure (second-order random walk)

Specify a second-order random walk temporal random effect. RW2 penalizes
deviations from linearity, resulting in smoother trends than RW1:
`phi[t] - 2*phi[t-1] + phi[t-2] ~ N(0, sigma^2)`.

## Usage

``` r
temporal_rw2(
  time_var,
  group_var = NULL,
  cyclic = FALSE,
  shared = NULL,
  parameterization = c("auto", "centered")
)
```

## Arguments

- time_var:

  A formula (`~ time`) or single character string naming the time
  variable in the data.

- group_var:

  Optional formula (`~ g`) or character string naming a grouping
  variable. When supplied, a separate random walk is fit per group;
  `NULL` (default) fits a single walk shared across all observations.

- cyclic:

  Logical. If `TRUE`, the random walk wraps around so the last time
  point is a neighbour of the first (cyclic boundary, e.g. month of
  year). Default `FALSE`.

- shared:

  Whether the temporal effect is shared across processes in a
  multi-process model. `NULL` (default) shares the effect; `FALSE` fits
  process-specific effects and emits a warning about unshared
  confounding.

- parameterization:

  Sampler coordinates: `"auto"` uses field information to scale field
  amplitudes; `"centered"` samples effect coordinates.

## Value

A `tulpa_temporal` object

## Details

RW2 produces smoother curves than RW1 because it penalizes the second
derivative (curvature) rather than the first derivative (slope). It
requires at least 3 time points.

The precision matrix is rank T-2 (two constraints needed).

The SD sigma of the walk's second differences carries a PC prior with
`P(sigma > 3) = 0.01`, the same on the nested-Laplace and the sampler
routes.

## Examples

``` r
# Create temporal RW2 specification
temporal_rw2("year")
#> tulpa temporal specification
#> ============================
#> 
#> Type: RW2 (Second-order Random Walk) 
#> Time variable: year 
#> Shared: Yes (enters both processes) 

# \donttest{
# Smooth temporal trend
set.seed(126)
df <- data.frame(
  year = rep(1:20, each = 4),
  x = rnorm(80)
)
trend <- sin(seq(0, 2, length.out = 20))
df$count <- rpois(80, exp(1 + 0.3 * df$x + trend[df$year]))

fit <- tulpa(
  count ~ x,
  data = df,
  family = "poisson",
  temporal = temporal_rw2("year"),
  mode = "auto"
)
#> [nested-laplace] 1/9 cells (11%) | elapsed 0s | ETA >=0s | 0.00s/cells
#> [nested-laplace] 5/5 cells (100%) | elapsed 0s | ETA done | 0.00s/cells
#> [nested-laplace] 1/5 cells (20%) | elapsed 0s | ETA >=0s | 0.00s/cells
#> [nested-laplace] 5/5 cells (100%) | elapsed 0s | ETA done | 0.00s/cells
summary(fit)
#> tulpa fit  (nested_laplace, tier 2) 
#> Call: tulpa(formula = count ~ x, data = df, family = "poisson", mode = "auto", temporal = temporal_rw2("year"))
#>   inference mode: structured 
#>   selected because: temporal rw2 field; nested-Laplace integration 
#>   family: poisson 
#>   n_obs: 80 
#> 
#> Fixed effects:
#>             estimate std.error  2.5 % 97.5 %
#> (Intercept)   1.6658   0.05081 1.5592 1.7583
#> x             0.2924   0.05272 0.1893 0.3959
#>   interval read: skew_map_cell 
#>   interval declined: skew_correct: gamma_3 is retained at the MAP cell only, so the mixture components carry no per-cell skew to compose; the coefficients it declines keep the read they would have had 
#> 
#> Hyperparameters:
#>  parameter mean  sd median ci_lo  ci_hi
#>        tau  Inf Inf   1010 56.61 101345
#> 
#> Fixed-effect prior: normal(0, 2.5)
#> 
#> Reliability (iid draws):
#>   outer pareto_k -1.574 (good); grid ESS 5.821 of 13 cells; inner skew 0.05647 (good)
#>   verdict: reliable (both layers good)
# }
```
