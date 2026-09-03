# Show default priors for a tulpa family

Display the default prior specifications used for each model family.
Useful for understanding what priors are applied before fitting and as a
starting point for customization.

## Usage

``` r
priors_default(family = NULL, spatial = FALSE, temporal = FALSE)
```

## Arguments

- family:

  A tulpa family object (e.g. a ratio family constructor from a model
  package such as tulpaRatio). If NULL (default), shows defaults for all
  families.

- spatial:

  Logical; if TRUE, include spatial priors. Default FALSE.

- temporal:

  Logical; if TRUE, include temporal priors. Default FALSE.

## Value

Invisibly returns a `tulpa_priors` object with the defaults. Primarily
called for its side effect of printing.

## Details

Default priors in tulpa follow these principles:

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
