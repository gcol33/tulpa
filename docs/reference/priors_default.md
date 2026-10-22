# Show default priors for a tulpa family

Display the default
[`tulpa_priors()`](https://gillescolling.com/tulpa/reference/tulpa_priors.md)
specification: the priors the simulation tools
([`prior_predict()`](https://gillescolling.com/tulpa/reference/prior_predict.md),
[`tulpa_simulate()`](https://gillescolling.com/tulpa/reference/tulpa_simulate.md))
draw from, and a starting point for customizing them.
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) does not
read a `tulpa_priors` object; the no-family printout closes with the
defaults a
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) fit does
use, which differ for the random-effect SDs (see
[`tulpa_priors()`](https://gillescolling.com/tulpa/reference/tulpa_priors.md)).

## Usage

``` r
priors_default(family = NULL, spatial = FALSE, temporal = FALSE)
```

## Arguments

- family:

  `NULL` (default) for the general printout, or a `tulpa_family` object
  – a model package's family constructor, such as a tulpaRatio ratio
  family, or one built with
  [`tulpa_family()`](https://gillescolling.com/tulpa/reference/tulpa_family.md)
  – for the parameters that family carries. A family NAME (`"poisson"`)
  is not a `tulpa_family` and is refused; the built-in
  [`family_names()`](https://gillescolling.com/tulpa/reference/family_names.md)
  share the general printout.

- spatial:

  Logical; if TRUE, include spatial priors. Default FALSE.

- temporal:

  Logical; if TRUE, include temporal priors. Default FALSE.

## Value

Invisibly returns a `tulpa_priors` object with the defaults. Primarily
called for its side effect of printing.

## Details

The family-specific overdispersion lines name the ratio families a model
package registers (`negbin_negbin`, `negbin_gamma`, `poisson_gamma`,
`gamma_gamma`, `beta_binomial`); they are not among tulpa's own
[`family_names()`](https://gillescolling.com/tulpa/reference/family_names.md).

Default priors in
[`tulpa_priors()`](https://gillescolling.com/tulpa/reference/tulpa_priors.md)
follow these principles:

- **Fixed effects (beta)**: Normal(0, 2.5) - weakly informative, allows
  coefficients roughly in \[-5, 5\] on the link scale.

- **Random effect SD (sigma)**: PC prior with P(sigma \> 1) = 0.01 -
  favors simpler models with smaller variance components.

- **Overdispersion (phi)**: PC prior with P(phi \> 10) = 0.01 on the NB2
  size `phi` (larger `phi` is less overdispersion; `phi -> Inf` is the
  Poisson limit). The default keeps `phi` finite, allowing
  overdispersion while penalising extreme values.

- **Temporal correlation (rho)**: Beta(2, 2) - symmetric prior centered
  at 0.5, appropriate for AR(1) correlation.

- **Spatial mixing (rho_spatial)**: Beta(1, 1) = Uniform(0, 1) - no
  prior preference for structured vs. unstructured spatial variation.

## See also

[`tulpa_priors()`](https://gillescolling.com/tulpa/reference/tulpa_priors.md)
for creating custom priors

## Examples

``` r
# Defaults for all families (no family argument)
priors_default()

# Family-specific defaults take a tulpa_family object. Model packages
# (e.g. tulpaRatio) register rich families; a minimal one is enough here.
fam <- tulpa_family(
  name = "poisson_gamma",
  simulate_fn = function(eta, params, n_obs, ...) rpois(n_obs, exp(eta[[1]]))
)
priors_default(fam)

# Including spatial parameters
priors_default(fam, spatial = TRUE)

# Use as a starting point for customization
my_priors <- priors_default(fam)
my_priors$beta <- prior_normal(0, 1)  # Tighter prior on fixed effects
```
