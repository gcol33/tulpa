# Fixed-effect coefficients (lme4-compatible)

The fixed-effect point estimates, equivalent to
[`coef()`](https://rdrr.io/r/stats/coef.html) on a `tulpa_fit`. Provided
under the `fixef` name for code written against the `lme4` / `nlme`
interface: `lme4::fixef(fit)` and `nlme::fixef(fit)` dispatch here too
when either package is installed, so a `tulpa_fit` can be dropped into
an `lme4`-shaped workflow.

## Usage

``` r
fixef(object, ...)

# S3 method for class 'tulpa_fit'
fixef(object, ...)
```

## Arguments

- object:

  A `tulpa_fit` object.

- ...:

  Ignored.

## Value

Named numeric vector of fixed-effect estimates.

## Details

Note the deliberate difference from
[`lme4::coef.merMod`](https://rdrr.io/pkg/lme4/man/merMod-class.html),
which returns per-group sums of the fixed and random effects. On a
`tulpa_fit`, [`coef()`](https://rdrr.io/r/stats/coef.html) returns the
fixed effects alone and `fixef()` is its synonym; the random effects are
[`ranef()`](https://gillescolling.com/tulpa/reference/ranef.md).

## See also

[`coef.tulpa_fit()`](https://gillescolling.com/tulpa/reference/coef.tulpa_fit.md),
[`ranef()`](https://gillescolling.com/tulpa/reference/ranef.md)

## Examples

``` r
# \donttest{
set.seed(1)
df <- data.frame(x = rnorm(80))
df$y <- rpois(80, exp(0.5 + 0.3 * df$x))
fit <- tulpa(y ~ x, data = df, family = "poisson")
fixef(fit)
# }
```
