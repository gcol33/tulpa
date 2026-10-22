# Prior specification for tulpa models

Specify priors for model parameters. Supports both PC (penalized
complexity) priors for variance components and standard distributions
for other parameters.

A `tulpa_priors` object is what the simulation tools draw parameters
from:
[`prior_predict()`](https://gillescolling.com/tulpa/reference/prior_predict.md)
and
[`tulpa_simulate()`](https://gillescolling.com/tulpa/reference/tulpa_simulate.md)
with `theta = NULL`. The model fitters do not read one.
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) carries
its own defaults, set per backend through its `beta_prior`, `re_prior`
and `hyperprior` arguments:

- Fixed effects: `prior_normal(0, 2.5)` on every backend – the same
  default as `beta` here.

- Random-effect SDs: a PC prior with `P(sigma > 3) = 0.01` (plus an
  LKJ(2) prior on a correlated block's correlations) on the routes that
  integrate or maximize over them (nested Laplace, `re_cov_nested`,
  `mode = "eb"`); half-Cauchy(2.5) on the samplers
  (`re_prior$sigma_re_scale`, and `re_prior$prior_sigma_scale` for
  `mode = "gibbs"`); an inverse-Wishart under
  `control$re_cov = "gibbs"`.

The `sigma`, `phi`, `rho_temporal` and `rho_spatial` defaults below are
therefore the simulation defaults, not the priors a fit ran under.

## Usage

``` r
tulpa_priors(
  beta = NULL,
  sigma = NULL,
  phi = NULL,
  rho_temporal = NULL,
  rho_spatial = NULL
)

# S3 method for class 'tulpa_priors'
print(x, ...)

# S3 method for class 'tulpa_prior'
print(x, ...)

# S3 method for class 'tulpa_prior_normal'
format(x, indent = "", ...)

# S3 method for class 'tulpa_prior_half_normal'
format(x, indent = "", ...)

# S3 method for class 'tulpa_prior_half_cauchy'
format(x, indent = "", ...)

# S3 method for class 'tulpa_prior_gamma'
format(x, indent = "", ...)

# S3 method for class 'tulpa_prior_exponential'
format(x, indent = "", ...)

# S3 method for class 'tulpa_prior_beta'
format(x, indent = "", ...)

# S3 method for class 'tulpa_prior_pc'
format(x, indent = "", ...)

# S3 method for class 'tulpa_prior'
format(x, indent = "", ...)
```

## Arguments

- beta:

  Prior for fixed effects. Default: `prior_normal(0, 2.5)`.

- sigma:

  Prior for random effect SDs. Default: PC prior with P(sigma \> 1) =
  0.01.

- phi:

  Prior for overdispersion parameter. Default: PC prior with P(phi
  \> 10) = 0.01.

- rho_temporal:

  Prior for temporal autocorrelation. Default: `prior_beta(2, 2)`
  centered at 0.5.

- rho_spatial:

  Prior for spatial proportion (BYM2). Default: `prior_beta(1, 1)`
  (uniform).

- x:

  A `tulpa_priors` specification, or a single prior built by a
  `prior_*()` constructor.

- ...:

  Ignored.

- indent:

  String prefixed to each formatted line.

## Value

A `tulpa_priors` object

[`format()`](https://rdrr.io/r/base/format.html) returns the prior as a
character string.

## Details

PC priors (Simpson et al., 2017) provide principled regularization that:

- Favors simpler models (smaller variance components)

- Has interpretable parameters (tail probabilities)

- Prevents overfitting with sparse data

For other parameters, standard distributions are available via helper
functions:
[`prior_normal()`](https://gillescolling.com/tulpa/reference/prior_normal.md),
[`prior_half_normal()`](https://gillescolling.com/tulpa/reference/prior_half_normal.md),
[`prior_half_cauchy()`](https://gillescolling.com/tulpa/reference/prior_half_cauchy.md),
[`prior_gamma()`](https://gillescolling.com/tulpa/reference/prior_gamma.md),
[`prior_beta()`](https://gillescolling.com/tulpa/reference/prior_beta.md),
[`prior_exponential()`](https://gillescolling.com/tulpa/reference/prior_exponential.md).

## References

Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sorbye, S. H.
(2017). Penalising model component complexity: A principled, practical
approach to constructing priors. Statistical Science, 32(1), 1-28.

## Examples

``` r
# Default priors
tulpa_priors()

# Custom fixed effect prior
tulpa_priors(beta = prior_normal(0, 1))

# Tighter random effect prior
tulpa_priors(sigma = prior_pc(U = 0.5, alpha = 0.01))

# Half-Cauchy for random effect SD
tulpa_priors(sigma = prior_half_cauchy(2.5))

# Informative prior for temporal correlation
tulpa_priors(rho_temporal = prior_beta(5, 2))  # Prior mode at ~0.8
```
