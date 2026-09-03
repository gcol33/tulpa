# Model-level summary statistics (broom-compatible)

Model-level summary statistics (broom-compatible)

## Usage

``` r
# S3 method for class 'tulpa_fit'
glance(x, ...)
```

## Arguments

- x:

  A `tulpa_fit` object.

- ...:

  Ignored.

## Value

Single-row data frame.

## Examples

``` r
# \donttest{
set.seed(1)
df <- data.frame(x = rnorm(100))
df$y <- rpois(100, exp(0.3 * df$x))
fit <- tulpa(y ~ x, data = df, family = "poisson")
glance(fit)
# }
```
