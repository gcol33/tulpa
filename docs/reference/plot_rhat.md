# Plot Rhat Convergence Diagnostic

Creates a visual display of Rhat values for all parameters, with
color-coding to highlight convergence issues. Rhat values \> 1.01
indicate potential convergence problems.

## Usage

``` r
plot_rhat(fit, threshold = 1.01, pars = NULL)
```

## Arguments

- fit:

  A `tulpa_fit` object.

- threshold:

  Rhat threshold for warnings (default: 1.01).

- pars:

  Character vector of parameter names to include. If NULL (default),
  includes all main parameters (excludes high-dimensional RE/spatial).

## Value

A ggplot object (if ggplot2 available) or base R plot (invisible).

## Details

Color coding:

- Green: Rhat \< 1.01 (converged)

- Yellow: 1.01 \<= Rhat \< 1.05 (borderline)

- Red: Rhat \>= 1.05 (not converged)

## See also

[`plot_ess()`](https://gillescolling.com/tulpa/reference/plot_ess.md),
[`diagnostic_summary()`](https://gillescolling.com/tulpa/reference/diagnostic_summary.md),
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)

## Examples

``` r
# Diagnostic plots require a fitted tulpa model
# See tulpa() examples for fitting models

# \donttest{
set.seed(123)
df <- data.frame(x = rnorm(60))
df$y <- rpois(60, exp(0.5 + 0.3 * df$x))
fit <- tulpa(y ~ x, data = df, family = "poisson", mode = "hmc",
             control = list(n_iter = 500L, warmup = 250L, n_chains = 2L,
                            seed = 1L))
plot_rhat(fit)
# }
```
