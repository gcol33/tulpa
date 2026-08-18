# K-fold cross-validation for a tulpa fit

Splits the data into `K` folds, refits the model on each `K - 1`
training partition (via the fit's stored
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) call),
and accumulates the held-out fold's pointwise log predictive density
\\\log \frac{1}{S}\sum_s p(y_i \mid \eta_i^{(s)})\\ over the training
posterior draws. The summed `elpd_kfold` is directly comparable to the
`elpd_loo` from
[`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)
– the exact refit counterpart to the PSIS-LOO approximation, for when
the Pareto k-hat gate flags LOO as unreliable.

Fixed-effect / GLMM fits only: subsetting the observations breaks a
spatial or temporal field, so those fits are rejected (use PSIS-LOO via
[`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)).
Held-out random-effect groups contribute at their prior mean
(population-level held-out prediction), matching
[`predict()`](https://rdrr.io/r/stats/predict.html).

## Usage

``` r
tulpa_kfold(object, data, K = 10L, folds = NULL, n_trials = NULL, seed = NULL)
```

## Arguments

- object:

  A `tulpa_fit` fitted through
  [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) (must
  carry `$call`).

- data:

  The data frame the model was fit to.

- K:

  Number of folds (default 10).

- folds:

  Optional integer vector of fold ids, length `nrow(data)`; a random
  balanced partition is drawn when `NULL`.

- n_trials:

  Optional binomial denominators (length `nrow(data)`); defaults to the
  trials stored on the fit, else 1 (Bernoulli). Each fold's refit
  receives the training rows' trials, and the held-out density is scored
  at the test rows' trials.

- seed:

  Optional seed for the random partition.

## Value

A list with `elpd_kfold` (summed held-out elpd), `se_elpd_kfold` (its
standard error), `pointwise` (per-observation held-out elpd), `folds`,
and `K`.

## See also

[`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)
for PSIS-LOO / WAIC on a single fit.

## Examples

``` r
# \donttest{
set.seed(1)
d <- data.frame(x = rnorm(120))
d$y <- rpois(120, exp(0.4 + 0.6 * d$x))
fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
cv  <- tulpa_kfold(fit, data = d, K = 5, seed = 1)
cv$elpd_kfold
#> [1] -186.7652
# }
```
