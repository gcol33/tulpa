# DIC, CPO, WAIC and PSIS-LOO on a fit

Generic front doors onto the two criteria
[`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)
computes that the loo package owns no generic for. WAIC and PSIS-LOO
have theirs
([`loo::waic()`](https://mc-stan.org/loo/reference/waic.html),
[`loo::loo()`](https://mc-stan.org/loo/reference/loo.html)), so a model
package registers methods on those rather than on new names that would
mask them.

## Usage

``` r
pointwise_loglik(object, ...)

# Default S3 method
pointwise_loglik(object, ...)

# S3 method for class 'tulpa_fit'
pointwise_loglik(object, ndraws = NULL, ...)

dic(object, ...)

# Default S3 method
dic(object, loglik_at_mean = NULL, ...)

# S3 method for class 'tulpa_fit'
dic(object, ...)

cpo(object, ...)

# Default S3 method
cpo(object, ...)

# S3 method for class 'tulpa_fit'
cpo(object, ...)

# S3 method for class 'tulpa_fit'
waic(x, ...)

# S3 method for class 'tulpa_fit'
loo(x, ...)
```

## Arguments

- object, x:

  A pointwise log-likelihood matrix (draws x observations), or a fitted
  model object a method is registered for, such as a `tulpa_fit`.

- ...:

  For `dic()` and `cpo()`, passed to
  [`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)
  (e.g. `group`, `chunk_size`). For `waic()` and `loo()`, passed to
  loo's matrix methods (e.g. `cores`, `save_psis`).

- ndraws:

  Number of posterior draws the matrix is evaluated at. Defaults to all
  stored draws, or 400 on the draw-free Laplace tier; a smaller number
  subsamples the stored ones. Read by the `tulpa_fit` method of
  `pointwise_loglik()`, which is the door the criteria below reach the
  matrix through.

- loglik_at_mean:

  Length-`n_obs` vector of pointwise log-likelihoods at the posterior
  mean of the parameters. Required for DIC's plug-in deviance; without
  it the DIC fields are `NA`.

## Value

`dic()` and `cpo()` return a `tulpa_criteria` object. `waic()` returns a
loo `waic` object and `loo()` a loo `psis_loo` object.

## Details

`pointwise_loglik()` is the one door onto the `[n_draws x n_obs]` matrix
itself, the input every criterion above is computed from.
[`compare_models()`](https://gillescolling.com/tulpa/reference/compare_models.md)
and
[`model_average()`](https://gillescolling.com/tulpa/reference/model_average.md)
call it (through an internal wrapper) rather than assuming the engine's
own fit layout, so a model package that registers
`pointwise_loglik.<its fit class>()` – alongside its own `waic()` /
`loo()` / `dic()` / `cpo()` methods – reaches model comparison and
averaging too.

The default methods take a draws x observations pointwise log-likelihood
matrix, the same input
[`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)
takes. A model package registers a method taking its own fit object,
builds the matrix from the posterior, and delegates here.

The `tulpa_fit` methods build that matrix from the fit itself, from the
same source `compare_models(criterion = "waic")` reads: a `log_lik` the
backend stored with its draws, else the family density evaluated at
posterior draws of the in-sample linear predictor (sampler draws, the
outer-grid mixture of a nested-Laplace fit, or Gaussian draws at the
fixed-effect mode and covariance). The draws are pinned to a fixed
internal seed, so repeated calls return the same numbers and leave the
session RNG untouched. `dic()` plugs in the posterior mean of the linear
predictor, so its `p_dic` counts effective parameters in the linear
predictor's parameterization; where the fit stored its `log_lik` and
carries no linear predictor draws, the DIC fields are `NA` and `dbar` is
still reported. A fit with no pointwise log-likelihood (no stored
response, a family that is not one built-in family name, or a linear
predictor the fit cannot reproduce) is refused with an error naming the
fit's class and the reason.

[`loo::waic()`](https://mc-stan.org/loo/reference/waic.html) and
[`loo::loo()`](https://mc-stan.org/loo/reference/loo.html) dispatch to
the `tulpa_fit` methods once loo is loaded, and return loo's own `waic`
/ `psis_loo` objects, so
[`loo::loo_compare()`](https://mc-stan.org/loo/reference/loo_compare.html)
reads them. On an MCMC chain fit `loo()` passes relative effective
sample sizes computed over the fit's chains; on an i.i.d. or
approximation fit the draws are independent and `r_eff` is 1.

## See also

[`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)
for every criterion at once and for what the LOO unit means;
[`compare_models()`](https://gillescolling.com/tulpa/reference/compare_models.md)
to rank several fits.

## Examples

``` r
set.seed(1)
y  <- rnorm(40)
mu <- matrix(rnorm(200 * 40, sd = 0.2), 200, 40)
ll <- dnorm(matrix(y, 200, 40, byrow = TRUE), mean = mu, log = TRUE)
cpo(ll)
#> tulpa model criteria  (200 draws x 40 observations)
#>   LOOIC          107.8  (SE 7.3)
#>   elpd_loo       -53.9  (SE 3.7)
#>   p_loo            1.4
#>   LPML           -53.9
# \donttest{
d <- data.frame(x = rnorm(120))
d$y <- rpois(120, exp(0.4 + 0.6 * d$x))
fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
dic(fit)
#> tulpa model criteria  (400 draws x 120 observations)
#>   DIC            391.9
#>   p_DIC            2.0
cpo(fit)$lpml
#> [1] -196.2028
if (requireNamespace("loo", quietly = TRUE)) loo::waic(fit)
#> Warning: 
#> 1 (0.8%) p_waic estimates greater than 0.4. We recommend trying loo instead.
#> 
#> Computed from 400 by 120 log-likelihood matrix.
#> 
#>           Estimate   SE
#> elpd_waic   -196.2  9.9
#> p_waic         2.4  0.6
#> waic         392.4 19.8
#> 
#> 1 (0.8%) p_waic estimates greater than 0.4. We recommend trying loo instead. 
# }
```
