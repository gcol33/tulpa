# The fit's log-scale goodness quantity

What this returns depends on what the fit computed, and the returned
object names it in a `quantity` attribute:

## Usage

``` r
# S3 method for class 'tulpa_fit'
logLik(object, ...)
```

## Arguments

- object:

  A `tulpa_fit` object.

- ...:

  Ignored.

## Value

A `logLik` object with `quantity` and `conditioned_on` attributes, and a
`declined` attribute naming the reason when no value could be read.

## Details

- `"log_posterior_mean"`:

  a sampler fit: the mean over the draws of the log joint posterior
  density, prior included, in that backend's own parameterization: the
  unconstrained coordinates with their Jacobians for the ModelData
  samplers (`hmc`, `ess`, `sghmc`, `sgld`, `mclmc`, `smc`, `vi`), and
  the natural scale each quantity is sampled on for the RE-covariance
  and Polya-Gamma Gibbs samplers. Values are therefore comparable across
  fits of the same backend only.

- `"log_evidence"`:

  a deterministic fit that estimated no hyperparameter from the data:
  the log marginal probability of the data under the model as specified.
  A hyperparameter the fit integrated counts, and so does one the caller
  supplied (a `phi`, a `sigma_re`, an outer grid laid at one value),
  which is part of the model rather than an estimate. A Laplace fit of a
  model with nothing to integrate reports its log marginal likelihood
  here, which already is that quantity.

- `"log_marginal_likelihood"`:

  a deterministic fit that estimated some hyperparameters by maximising
  over the same data (empirical Bayes, or `estimate_phi = TRUE`): the
  log marginal likelihood at those estimates. The `conditioned_on`
  attribute names them.

- `"log_likelihood"`:

  a fit that maximised over every parameter it reports, with the random
  effects integrated out
  ([`agq_fit()`](https://gillescolling.com/tulpa/reference/agq_fit.md)):
  the maximised log-likelihood, with `df` the number of maximised
  parameters.

On a nested-Laplace fit the value is the log evidence of its outer grid,
`log sum_k exp(log_marginal_k + log_cell_k)`, where `log_marginal`
already carries each axis's hyperprior density on its integration
coordinate and `log_cell_k` is the cell's absolute volume there. Every
default outer axis carries a proper prior (a PC prior on each scale and
range, a uniform on each bounded axis), so the value is the evidence
under that prior and does not move with where the nodes were laid or how
many there are, provided the grid covers the posterior. A fit carrying
an axis with no proper prior (a flat random-effect covariance prior, a
dispersion whose family has no PC prior yet, an axis on a block that
does not carry its coordinates) reports `NA` with
`declined = "improper_hyperprior"` and names the axes in a
`declined_axes` attribute. A central-composite (CCD) design reproduces
moments and carries no cell volume, so a fit integrated on one reports
`NA` with a `declined` attribute.

A sampler fit whose producer kept no per-draw log posterior reports `NA`
with `quantity = "log_posterior_mean"` and
`declined = "no_log_posterior_recorded"`. The Polya-Gamma Gibbs routes
that recentre a proper field every sweep (the NNGP and multiscale-GP
fields, and the negative-binomial kernel's iid random-effect block)
leave no written density invariant and record none; a fit carrying
neither draws nor a log marginal declines with
`"no_goodness_quantity_recorded"`. A value of `NA` always carries a
`declined` attribute.

Values with a different `quantity` or `conditioned_on` are not on one
scale, and `compare_models(criterion = "loglik")` refuses such a set.
Only a `"log_likelihood"` is what AIC and BIC penalise, so
[`AIC.tulpa_fit()`](https://gillescolling.com/tulpa/reference/AIC.tulpa_fit.md)
and
[`BIC.tulpa_fit()`](https://gillescolling.com/tulpa/reference/AIC.tulpa_fit.md)
refuse every other quantity; compare those fits by their evidence, or by
`compare_models(criterion = "waic")` / `"loo"`, which score the
pointwise predictive density.
