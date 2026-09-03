# AR1 temporal structure (First-order Autoregressive)

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
temporal_ar1("year", group = "site")

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
# }
```
