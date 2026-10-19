# Tidy fixed-effect table (broom-compatible)

Tidy fixed-effect table (broom-compatible)

## Usage

``` r
# S3 method for class 'tulpa_fit'
tidy(x, conf.level = 0.95, ...)
```

## Arguments

- x:

  A `tulpa_fit` object.

- conf.level:

  Interval level (default 0.95).

- ...:

  Ignored.

## Value

Data frame: term, estimate, std.error, conf.low, conf.high.

## Examples

``` r
# \donttest{
set.seed(1)
df <- data.frame(x = rnorm(100), g = factor(rep(1:10, 10)))
df$y <- rpois(100, exp(0.3 * df$x))
fit <- tulpa(y ~ x + (1 | g), data = df, family = "poisson")
#> Warning: tulpa(): the 're_cov_gibbs' chain has not converged -- x: Rhat 1.01, bulk-ESS 67. Its estimates and intervals are not yet the posterior. Run longer (control$n_iter / control$warmup), or use mode = 'hmc'; see check_diagnostics().
tidy(fit)
#>          term   estimate std.error   conf.low  conf.high
#> 1 (Intercept) -0.3850430 0.2033565 -0.7866048 0.00483819
#> 2           x  0.4237101 0.1235747  0.1703145 0.66292298
# }
```
