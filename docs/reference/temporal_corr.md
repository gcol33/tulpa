# Extract temporal correlation parameters from a fitted model

Returns posterior summary for temporal hyperparameters (tau, rho for
AR1, sigma/lengthscale for temporal GP).

## Usage

``` r
temporal_corr(object, probs = c(0.025, 0.975))
```

## Arguments

- object:

  A `tulpa_fit` object fitted with a temporal component.

- probs:

  Quantile probabilities (default 0.025, 0.975).

## Value

A data.frame with rows for each temporal hyperparameter.

## Examples

``` r
# \donttest{
set.seed(127)
df <- data.frame(year = rep(1:20, each = 3), x = rnorm(60))
f <- as.numeric(arima.sim(list(ar = 0.7), 20, sd = 0.4))
df$count <- rpois(60, exp(1 + 0.3 * df$x + f[df$year]))
fit <- tulpa(count ~ x, data = df, family = "poisson",
             temporal = temporal_ar1("year"))
temporal_corr(fit)
# }
```
