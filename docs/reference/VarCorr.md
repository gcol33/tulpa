# Random-effect variances and correlations

Summarize the random-effect covariance of a fit: the standard deviation
of each random-effect coefficient, the correlations between them within
a term, and the covariance matrix itself.

Whether the covariance was estimated, sampled, or merely conditioned on
is reported alongside it, because the three are different claims. A fit
from `mode = "laplace"` conditions on `sigma_re`, so its values are the
ones supplied; `mode = "eb"` estimates them; a sampler tier integrates
them.

## Usage

``` r
VarCorr(x, sigma = 1, ...)

# S3 method for class 'tulpa_fit'
VarCorr(x, sigma = 1, ...)
```

## Arguments

- x:

  A `tulpa_fit`.

- sigma:

  Ignored, for compatibility with the `VarCorr` generic.

- ...:

  Ignored.

## Value

A data frame with one row per random-effect coefficient: `term`, `coef`,
`sd`, and `source` (one of `"estimated"`, `"sampled"`, `"conditioned"`).
Correlated terms additionally carry the covariance matrices in the
`"cov"` attribute, one per term, each with a `"correlation"` attribute.
Returns an empty data frame when the fit has no random effects.

## See also

[`ranef()`](https://gillescolling.com/tulpa/reference/ranef.md) for the
per-level deviations,
[`tulpa_eb()`](https://gillescolling.com/tulpa/reference/tulpa_eb.md) to
estimate the covariance rather than condition on it.

## Examples

``` r
# \donttest{
set.seed(1)
G <- 30L; per <- 10L; n <- G * per
grp <- rep(seq_len(G), each = per); x <- rnorm(n)
b <- rnorm(G, 0, 0.8)
d <- data.frame(y = rpois(n, exp(0.3 + 0.5 * x + b[grp])), x = x,
                g = factor(grp))
fit <- tulpa(y ~ x + (1 | g), data = d, family = "poisson", mode = "eb")
VarCorr(fit)
# }
```
