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
tidy(fit)
#>          term   estimate std.error   conf.low  conf.high
#> 1 (Intercept) -0.3803811 0.2160954 -0.8402536 0.04342843
#> 2           x  0.4379263 0.1281955  0.1841656 0.69509922
# }
```
