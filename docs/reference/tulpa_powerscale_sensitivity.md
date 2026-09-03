# Power-scaling prior / likelihood sensitivity

Local power-scaling sensitivity (Kallioinen et al. 2024): how much each
fixed-effect posterior moves when the prior or the likelihood is raised
to a power `alpha` near 1. The existing draws are importance-reweighted
by `exp((alpha - 1) * log_component)` (PSIS-smoothed via
[`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md);
no refits), and the sensitivity is the gradient of the cumulative
Jensen-Shannon distance between the base and power-scaled posteriors
with respect to `log2(alpha)`.

Values above `threshold` flag sensitivity. High on **both** the prior
and likelihood components indicates potential prior-data conflict; high
prior with low likelihood indicates a strong prior / weak likelihood.

When the fit recorded per-draw hyperparameter log-prior values at
draw-synthesis time (`$hyper_log_prior_draws`, stored by the
nested-Laplace mixture paths such as
[`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)
and the [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)
random-slope redirect), a `hyperparameter` column reports the
power-scaling sensitivity of the hyperparameter prior by the same
reweighting; `NA` otherwise.

## Usage

``` r
tulpa_powerscale_sensitivity(
  fit,
  data,
  prior = NULL,
  lower_alpha = 0.99,
  upper_alpha = 1.01,
  threshold = 0.05
)
```

## Arguments

- fit:

  A `tulpa_fit` fitted through
  [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)
  (fixed-effect / GLMM; spatial / temporal-field fits are rejected).

- data:

  The data frame the model was fit to.

- prior:

  The Gaussian fixed-effect prior the fit used, as `list(mean =, sd =)`
  (scalars recycled). Required for the prior component; omit to compute
  the likelihood component only.

- lower_alpha, upper_alpha:

  Power-scaling grid endpoints for the gradient (defaults 0.99 / 1.01,
  as in priorsense).

- threshold:

  Sensitivity flag threshold (default 0.05).

## Value

A data frame with one row per fixed-effect parameter and columns
`variable`, `prior`, `hyperparameter`, `likelihood`, `diagnosis`.

## References

Kallioinen, Paananen, Buerkner & Vehtari (2024). Detecting and
diagnosing prior and likelihood sensitivity with power-scaling.
*Statistics and Computing* 34:57. Nguyen & Vreeken (2015).
Non-parametric Jensen-Shannon divergence. ECML PKDD.

## See also

[`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md),
[`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md).

## Examples

``` r
# \donttest{
set.seed(1)
d <- data.frame(x = rnorm(150))
d$y <- rpois(150, exp(0.5 + 0.7 * d$x))
fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace",
             beta_prior = list(mean = 0, sd = 5))
tulpa_powerscale_sensitivity(fit, data = d, prior = list(mean = 0, sd = 5))
# }
```
