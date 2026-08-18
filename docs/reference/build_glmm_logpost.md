# Build a GLMM log-posterior (and gradient) from a model-data bundle.

Build a GLMM log-posterior (and gradient) from a model-data bundle.

## Usage

``` r
build_glmm_logpost(
  bundle,
  family,
  sigma_re = NULL,
  n_trials = NULL,
  phi = 1,
  beta_prior = list(mean = 0, sd = 2.5),
  weights = NULL,
  phi2 = NULL
)
```

## Arguments

- bundle:

  Output of
  [`tulpa_build_model_data()`](https://gillescolling.com/tulpa/reference/tulpa_build_model_data.md)
  (needs `y`, `X`, `offset`, `re_terms`, `n_obs`, `n_fixed`).

- family:

  Character family name (see
  [`family_names()`](https://gillescolling.com/tulpa/reference/family_names.md)).

- sigma_re:

  Numeric vector of random-effect SDs, one per RE term. Length must
  equal `length(bundle$re_terms)`. Ignored when there are no RE terms.

- n_trials:

  Binomial denominators (or `NULL`).

- phi:

  Dispersion/precision passed to the family.

- beta_prior:

  `list(mean, sd)` Gaussian prior on the fixed effects (scalars,
  recycled). Default `list(mean = 0, sd = 2.5)`.

- weights:

  Optional per-observation likelihood weights (length `n_obs`): each
  row's log-likelihood and score contribution is scaled by its weight.

- phi2:

  Optional second dispersion (Student-t degrees of freedom).

## Value

A list with:

- `log_posterior(theta)` – scalar log-posterior (up to a constant).

- `grad_log_posterior(theta)` – gradient vector.

- `dim` – length of `theta`.

- `init` – a zero starting vector of length `dim`.

- `unpack(theta)` – list(`beta`, `u` = list of n_groups x n_coefs
  matrices).

- `param_names` – character labels aligned with `theta`.
