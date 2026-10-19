# AR1 temporal structure (first-order autoregressive)

Specify a first-order autoregressive temporal random effect. AR1 models
temporal correlation where each time point depends on the previous one:
`phi[t] = rho * phi[t-1] + epsilon[t]`.

Unlike RW1/RW2, AR1 is a proper (stationary) model with an estimated
autocorrelation parameter rho.

## Usage

``` r
temporal_ar1(time_var, group_var = NULL, shared = NULL, rho_prior = NULL)
```

## Arguments

- time_var:

  A formula (`~ time`) or single character string naming the time
  variable in the data.

- group_var:

  Optional formula (`~ g`) or character string naming a grouping
  variable. When supplied, a separate random walk is fit per group;
  `NULL` (default) fits a single walk shared across all observations.

- shared:

  Whether the temporal effect is shared across processes in a
  multi-process model. `NULL` (default) shares the effect; `FALSE` fits
  process-specific effects and emits a warning about unshared
  confounding.

- rho_prior:

  Prior for the AR(1) autocorrelation. Default `NULL` is a
  Uniform(-1, 1) prior. Supply a `prior_beta(alpha, beta)` to place a
  Beta(alpha, beta) prior on `u = (rho + 1)/2` for a more informative
  prior (e.g. `prior_beta(5, 2)` favours positive autocorrelation).
  Honoured on both the exact-sampler and nested-Laplace paths.

## Value

A `tulpa_temporal` object

## Details

The AR1 process has marginal variance sigma^2 / (1 - rho^2) and
correlation between time points t and s of rho^\|t-s\|.

The precision matrix is tridiagonal and full rank, so no constraints are
needed.

## Examples

``` r
# Create temporal AR1 specification
temporal_ar1("year")
#> tulpa temporal specification
#> ============================
#> 
#> Type: AR1 (First-order Autoregressive) 
#> Time variable: year 
#> Shared: Yes (enters both processes) 
temporal_ar1("year", group = "site")
#> tulpa temporal specification
#> ============================
#> 
#> Type: AR1 (First-order Autoregressive) 
#> Time variable: year 
#> Group variable: site 
#> Shared: Yes (enters both processes) 

# \donttest{
# AR1 temporal correlation
set.seed(127)
df <- data.frame(
  year = rep(1:20, each = 3),
  x = rnorm(60)
)
f <- as.numeric(arima.sim(list(ar = 0.7), 20, sd = 0.4))
df$count <- rpois(60, exp(1 + 0.3 * df$x + f[df$year]))

fit <- tulpa(
  count ~ x,
  data = df,
  family = "poisson",
  temporal = temporal_ar1("year"),
  mode = "auto"
)
summary(fit)
#> tulpa fit  (nested_laplace, tier 2) 
#> Call: tulpa(formula = count ~ x, data = df, family = "poisson", mode = "auto", temporal = temporal_ar1("year"))
#>   inference mode: structured 
#>   selected because: temporal ar1 field; nested-Laplace integration 
#>   family: poisson 
#>   n_obs: 60 
#> 
#> Fixed effects:
#>             estimate std.error   2.5 % 97.5 %
#> (Intercept)   1.1134   0.49232 0.08823 2.0181
#> x             0.3297   0.07599 0.18296 0.4808
#>   interval read: skew_map_cell 
#>   interval declined: skew_correct: gamma_3 is retained at the MAP cell only, so the mixture components carry no per-cell skew to compose; the coefficients it declines keep the read they would have had 
#> 
#> Hyperparameters:
#>  parameter   mean     sd median  ci_lo  ci_hi
#>        tau 8.4592 4.8630 7.1050 2.2867 24.286
#>        rho 0.6618 0.2235 0.7196 0.1365  1.193
#> 
#> Reliability (iid draws):
#>   outer pareto_k NA (NA); grid ESS 6.996 of 25 cells; inner skew 0.006854 (good)
#>   verdict: outer integration not assessed; inner Laplace good 
# }
```
