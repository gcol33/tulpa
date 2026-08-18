# Multinomial (nominal K-class) logistic regression via Laplace

Fits a baseline-category multinomial logit model: for a K-level
unordered response, classes `1..K-1` each get their own linear predictor
and class `K` is the baseline. The coupled multinomial likelihood is
solved by a Newton step to the penalized mode (a Gaussian ridge prior on
the coefficients) and summarized by a Laplace approximation, reusing the
native multinomial kernel.

## Usage

``` r
tulpa_multinomial(
  formula,
  data,
  beta_prior = list(mean = 0, sd = 10),
  control = list()
)
```

## Arguments

- formula:

  Model formula; the response must be a factor (or coercible to one)
  with \>= 3 levels. The baseline is the last level.

- data:

  A data frame.

- beta_prior:

  Fixed-effect prior as `list(mean, sd)`: a mean-zero (`mean = 0`)
  Gaussian ridge on every coefficient with scalar SD `sd` (default
  `list(mean = 0, sd = 10)`). A finite SD keeps the mode finite under
  separation.

- control:

  List of numerical knobs: `max_iter` (default 100), `tol` (default
  1e-8), `n_draws` (posterior draws, default 2000), `seed`.

## Value

A `tulpa_fit` (subclass `tulpa_multinomial`) with `coef` (named
`class:term`), `vcov`, `draws`, `log_marginal`, `classes`, `baseline`,
and the standard generic-method support.

## See also

[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) for
single-process GLMMs.

## Examples

``` r
# \donttest{
set.seed(1)
n <- 300L; x <- rnorm(n)
eta <- cbind(0.5 + 1.0 * x, -0.3 - 0.8 * x)          # classes 1, 2 vs baseline 3
P <- cbind(exp(eta), 1); P <- P / rowSums(P)
y <- factor(apply(P, 1, function(pr) sample.int(3L, 1L, prob = pr)))
fit <- tulpa_multinomial(y ~ x, data = data.frame(y = y, x = x))
coef(fit)
#> 1:(Intercept)           1:x 2:(Intercept)           2:x 
#>    0.68664932    0.73377877   -0.08564341   -0.70336296 
# }
```
