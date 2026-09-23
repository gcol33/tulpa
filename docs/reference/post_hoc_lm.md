# Fit a post-hoc linear model on estimated parameters

Useful for exploring drivers of occupancy/detection/abundance variation
after model fitting. Fits a weighted linear model and optionally
generates bootstrap confidence intervals.

## Usage

``` r
post_hoc_lm(
  formula,
  data,
  weights = NULL,
  n_boot = 1000L,
  probs = c(0.025, 0.975)
)
```

## Arguments

- formula:

  Model formula (e.g., `psi_hat ~ trait1 + trait2`).

- data:

  A data.frame with response and predictors.

- weights:

  Optional weights (e.g., inverse of standard errors).

- n_boot:

  Number of bootstrap replicates for CI (default 1000, 0 to skip).

- probs:

  Quantile probabilities for bootstrap CI (default 0.025, 0.975).

## Value

A list of class `"post_hoc_lm"` with:

- summary:

  data.frame of coefficient estimates and CIs

- lm_fit:

  the underlying `lm` object

- boot_coefs:

  matrix of bootstrap coefficient samples (if `n_boot > 0`)

- R2:

  R-squared from the fitted model

## Examples

``` r
# Explore drivers of per-site estimates after fitting a model.
site <- data.frame(
  psi_hat = c(0.2, 0.5, 0.8, 0.4, 0.6, 0.3),
  se      = c(0.05, 0.04, 0.06, 0.05, 0.03, 0.05),
  trait   = c(1.0, 2.5, 3.8, 1.9, 3.1, 1.2)
)
fit <- post_hoc_lm(psi_hat ~ trait, data = site,
                   weights = 1 / site$se^2, n_boot = 200L)
fit
#> Post-hoc linear model
#> R-squared: 0.979
#> 
#>         term   estimate  std.error statistic      p.value    conf.low conf.high
#>  (Intercept) 0.03719919 0.03517469  1.057556 0.3498961569 -0.02738938 0.1066280
#>        trait 0.18719050 0.01375560 13.608312 0.0001688337  0.15826786 0.2128769
```
