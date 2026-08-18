# DIC and CPO

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
dic(object, ...)

# Default S3 method
dic(object, loglik_at_mean = NULL, ...)

cpo(object, ...)

# Default S3 method
cpo(object, ...)
```

## Arguments

- object:

  A pointwise log-likelihood matrix (draws x observations), or a fitted
  model object a method is registered for.

- ...:

  Passed to
  [`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)
  (e.g. `group`, `chunk_size`).

- loglik_at_mean:

  Length-`n_obs` vector of pointwise log-likelihoods at the posterior
  mean of the parameters. Required for DIC's plug-in deviance; without
  it the DIC fields are `NA`.

## Value

A `tulpa_criteria` object.

## Details

The default methods take a draws x observations pointwise log-likelihood
matrix, the same input
[`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)
takes. A model package registers a method taking its own fit object,
builds the matrix from the posterior, and delegates here.

## See also

[`tulpa_criteria()`](https://gillescolling.com/tulpa/reference/tulpa_criteria.md)
for every criterion at once and for what the LOO unit means.

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
```
