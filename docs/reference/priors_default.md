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
#> Default priors for tulpa models
#> ================================
#> 
#> These defaults apply to all families unless overridden.
#> 
#> Fixed effects (beta):
#>   Normal(0, 2.5)
#>   Interpretation: Coefficients roughly in [-5, 5] on link scale
#>   Customization: prior_normal(mean, sd)
#> 
#> Random effect SD (sigma):
#>   PC prior: P(sigma > 1) = 0.01
#>   Interpretation: Favors smaller variance components
#>   Customization: prior_pc(U, alpha) or prior_half_normal(sd)
#> 
#> Overdispersion (phi) [negbin/poisson_gamma only]:
#>   PC prior: P(phi > 10) = 0.01
#>   Interpretation: NB2 size; keeps phi finite (allows overdispersion),
#>                   phi -> Inf is the Poisson limit
#>   Customization: prior_pc(U, alpha) or prior_gamma(shape, rate)
#> 
#> Family-specific notes:
#>   negbin_negbin: Uses phi for both processes
#>   binomial: No overdispersion parameter (unless beta_binomial)
#>   poisson_gamma: Uses phi for gamma shape parameter

# Family-specific defaults take a tulpa_family object. Model packages
# (e.g. tulpaRatio) register rich families; a minimal one is enough here.
fam <- tulpa_family(
  name = "poisson_gamma",
  simulate_fn = function(eta, params, n_obs, ...) rpois(n_obs, exp(eta[[1]]))
)
priors_default(fam)
#> Default priors for poisson_gamma family
#> ===================================== 
#> 
#> Fixed effects (beta):
#>   Normal(0.00, 2.50)
#>   Used for: All regression coefficients in all processes
#> 
#> Random effect SD (sigma):
#>   PC prior: P(x > 1.00) = 0.010
#>     => Exponential(4.605)
#>   Used for: Standard deviation of group-level effects
#> 
#> Overdispersion (phi):
#>   PC prior: P(x > 10.00) = 0.010
#>     => Exponential(0.461)
#>   Used for: Gamma shape parameter for effort/exposure
#> 
#> To customize, use tulpa_priors():
#>   priors <- tulpa_priors(
#>     beta = prior_normal(0, 1),
#>     sigma = prior_pc(U = 0.5, alpha = 0.01)
#>   )

# Including spatial parameters
priors_default(fam, spatial = TRUE)
#> Default priors for poisson_gamma family
#> ===================================== 
#> 
#> Fixed effects (beta):
#>   Normal(0.00, 2.50)
#>   Used for: All regression coefficients in all processes
#> 
#> Random effect SD (sigma):
#>   PC prior: P(x > 1.00) = 0.010
#>     => Exponential(4.605)
#>   Used for: Standard deviation of group-level effects
#> 
#> Overdispersion (phi):
#>   PC prior: P(x > 10.00) = 0.010
#>     => Exponential(0.461)
#>   Used for: Gamma shape parameter for effort/exposure
#> 
#> Spatial mixing (rho_spatial):
#>   Beta(1.00, 1.00)  [mean = 0.50]
#>   Used for: BYM2 mixing proportion (structured vs. unstructured)
#> 
#> To customize, use tulpa_priors():
#>   priors <- tulpa_priors(
#>     beta = prior_normal(0, 1),
#>     sigma = prior_pc(U = 0.5, alpha = 0.01)
#>   )

# Use as a starting point for customization
my_priors <- priors_default(fam)
#> Default priors for poisson_gamma family
#> ===================================== 
#> 
#> Fixed effects (beta):
#>   Normal(0.00, 2.50)
#>   Used for: All regression coefficients in all processes
#> 
#> Random effect SD (sigma):
#>   PC prior: P(x > 1.00) = 0.010
#>     => Exponential(4.605)
#>   Used for: Standard deviation of group-level effects
#> 
#> Overdispersion (phi):
#>   PC prior: P(x > 10.00) = 0.010
#>     => Exponential(0.461)
#>   Used for: Gamma shape parameter for effort/exposure
#> 
#> To customize, use tulpa_priors():
#>   priors <- tulpa_priors(
#>     beta = prior_normal(0, 1),
#>     sigma = prior_pc(U = 0.5, alpha = 0.01)
#>   )
my_priors$beta <- prior_normal(0, 1)  # Tighter prior on fixed effects
```
