# Selective refit of high-Pareto-k observations (reloo)

PSIS-LOO with exact refits where the importance sampling is unreliable:
observations whose Pareto k-hat exceeds `k_threshold` are re-scored by
refitting the model without that observation (through the fit's stored
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) call, as
in
[`tulpa_kfold()`](https://gillescolling.com/tulpa/reference/tulpa_kfold.md))
and evaluating the exact held-out log predictive density. All other
observations keep their PSIS-LOO value, so the cost is one refit per
flagged observation rather than per fold.

The same restrictions as
[`tulpa_kfold()`](https://gillescolling.com/tulpa/reference/tulpa_kfold.md)
apply: fixed-effect / GLMM fits only (subsetting breaks a spatial /
temporal field), and held-out random-effect groups contribute at their
prior mean.

## Usage

``` r
tulpa_reloo(
  object,
  data,
  k_threshold = .nl_diag("k_usable"),
  n_trials = NULL,
  ndraws = NULL
)
```

## Arguments

- object:

  A `tulpa_fit` fitted through
  [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) (must
  carry `$call`).

- data:

  The data frame the model was fit to.

- k_threshold:

  Pareto k-hat above which an observation is refit (default 0.7, the
  standard PSIS reliability gate).

- n_trials:

  Optional binomial denominators (length `nrow(data)`); defaults to the
  trials stored on the fit, else 1.

- ndraws:

  Number of posterior draws used for the PSIS-LOO baseline (defaults to
  all stored draws, or 400 on the draw-free Laplace tier).

## Value

A list with `elpd_loo` (corrected), `se_elpd_loo`, `looic`, `pointwise`
(per-observation elpd, exact at the refit observations), `reloo_idx`
(indices refit), `pareto_k` (the original k-hat values), and
`k_threshold`.

## See also

[`tulpa_kfold()`](https://gillescolling.com/tulpa/reference/tulpa_kfold.md)
for the full refit-CV;
[`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)
for PSIS-LOO / WAIC without refits.

## Examples

``` r
# \donttest{
set.seed(1)
d <- data.frame(x = rnorm(120))
d$y <- rpois(120, exp(0.4 + 0.6 * d$x))
fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
rl  <- tulpa_reloo(fit, data = d)
rl$elpd_loo
# }
```
