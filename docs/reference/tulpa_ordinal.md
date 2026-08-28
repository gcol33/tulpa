# Ordinal (ordered K-class) cumulative-logit regression via Laplace

Fits a proportional-odds cumulative-logit model for an ordered factor
response: `P(y <= j | x) = plogis(c_j - x'beta)` with ordered cutpoints
`c_1 < ... < c_{K-1}`. The penalized mode (a ridge prior on the
coefficients and cutpoints) is found by L-BFGS and summarized by a
Laplace approximation. There is no separate intercept – the cutpoints
carry the baseline levels.

## Usage

``` r
tulpa_ordinal(
  formula,
  data,
  link = c("logit", "probit"),
  beta_prior = .tulpa_default_beta_prior("ordinal"),
  cut_prior_sd = 10,
  control = list()
)
```

## Arguments

- formula:

  Model formula; the response must be an ordered (or coercible) factor
  with \>= 3 levels. An intercept in `formula` is dropped.

- data:

  A data frame.

- link:

  Cumulative link: `"logit"` (proportional odds, default) or `"probit"`.

- beta_prior:

  Fixed-effect prior as `list(mean, sd)`: a mean-zero (`mean = 0`)
  Gaussian ridge on every coefficient with scalar SD `sd` (default the
  engine default, `prior_normal(0, 2.5)`).

- cut_prior_sd:

  SD of the mean-zero Gaussian ridge prior on the cutpoint parameters
  (default 10).

- control:

  List of numerical knobs: `max_iter` (default 200), `n_draws` (default
  2000), `seed`.

## Value

A `tulpa_fit` (subclass `tulpa_ordinal`) with `coef` (covariate
effects), `cutpoints`, `vcov`, `draws`, `log_marginal`, `levels`.

## See also

[`tulpa_multinomial()`](https://gillescolling.com/tulpa/reference/tulpa_multinomial.md)
for the nominal (unordered) case.

## Examples

``` r
# \donttest{
set.seed(1)
n <- 400L; x <- rnorm(n)
cuts <- c(-1, 0.5, 2); eta <- 0.8 * x
Fm <- plogis(outer(-eta, cuts, "+")); P <- cbind(Fm, 1) - cbind(0, Fm)
y <- ordered(apply(P, 1, function(pr) sample.int(4L, 1L, prob = pr)))
fit <- tulpa_ordinal(y ~ x, data = data.frame(y = y, x = x))
fit$coefficients; fit$cutpoints
#>         x 
#> 0.8391392 
#>        1|2        2|3        3|4 
#> -1.0608612  0.4512822  1.9829574 
# }
```
