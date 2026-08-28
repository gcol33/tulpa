# Expectation-Propagation fit for a GLM

Fits a generalized linear model with a Gaussian coefficient prior by
Expectation Propagation: the posterior is approximated by a Gaussian
whose per-observation sites match the moments of the tilted distribution
(via Gauss-Hermite quadrature). EP is exact for a Gaussian likelihood
and typically more accurate than Laplace on skewed likelihoods, since it
matches marginal moments rather than the mode curvature.

## Usage

``` r
tulpa_ep(
  formula,
  data,
  family = "binomial",
  phi = 1,
  phi2 = NULL,
  n_trials = NULL,
  beta_prior = .tulpa_default_beta_prior("ep"),
  control = list()
)
```

## Arguments

- formula:

  Model formula.

- data:

  A data frame.

- family:

  Character family name (see
  [`family_names()`](https://gillescolling.com/tulpa/reference/family_names.md)).

- phi:

  Dispersion / precision passed to the family (held fixed).

- phi2:

  Optional second dispersion (Student-t degrees of freedom for
  `family = "t"`; default 4 when `NULL`).

- n_trials:

  Binomial denominators (length `nrow(data)`), or `NULL` (= 1).

- beta_prior:

  Fixed-effect prior as `list(mean, sd)`: a mean-zero (`mean = 0`)
  Gaussian on every coefficient with SD `sd` (default the engine
  default, `prior_normal(0, 2.5)`). EP's site parameterisation assumes a
  mean-zero coefficient prior, so a non-zero `mean` errors – use a
  sampler (`mode = "mala"`) for a shifted prior.

- control:

  List: `max_sweeps` (default 50), `tol` (default 1e-6), `damping`
  (default 0.8), `n_quad` (Gauss-Hermite nodes, default 20), `n_draws`
  (default 2000), `seed`.

## Value

A `tulpa_fit` (subclass `tulpa_ep`) with `coefficients` (posterior
mean), `vcov`, `draws`, `log_marginal` (the EP approximation),
`converged`.

## References

Minka (2001). Expectation Propagation for approximate Bayesian
inference. UAI. Rasmussen & Williams (2006). Gaussian Processes for
Machine Learning, Algorithm 3.5.

## See also

[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) (Laplace
/ sampler tiers),
[`pathfinder()`](https://gillescolling.com/tulpa/reference/pathfinder.md)
(VI).

## Examples

``` r
# \donttest{
set.seed(1)
d <- data.frame(x = rnorm(200))
d$y <- rbinom(200, 1, plogis(-0.3 + 0.8 * d$x))
fit <- tulpa_ep(y ~ x, data = d, family = "binomial")
coef(fit)
#> (Intercept)           x 
#>  -0.4432535   0.4887355 
# }
```
