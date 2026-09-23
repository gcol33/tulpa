# Prior specification for tulpa models

Specify priors for model parameters. Supports both PC (penalized
complexity) priors for variance components and standard distributions
for other parameters.

## Usage

``` r
tulpa_priors(
  beta = NULL,
  sigma = NULL,
  phi = NULL,
  rho_temporal = NULL,
  rho_spatial = NULL
)
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

## Value

A `tulpa_priors` object

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
#> tulpa prior specification
#> =========================
#> 
#> Fixed effects (beta):
#>   Normal(0.00, 2.50)
#> 
#> Random effect SD (sigma):
#>   PC prior: P(x > 1.00) = 0.010
#>     => Exponential(4.605)
#> 
#> Overdispersion (phi):
#>   PC prior: P(x > 10.00) = 0.010
#>     => Exponential(0.461)
#> 
#> Temporal autocorrelation (rho_temporal):
#>   Beta(2.00, 2.00)  [mean = 0.50]
#> 
#> Spatial proportion (rho_spatial):
#>   Beta(1.00, 1.00)  [mean = 0.50]

# Custom fixed effect prior
tulpa_priors(beta = prior_normal(0, 1))
#> tulpa prior specification
#> =========================
#> 
#> Fixed effects (beta):
#>   Normal(0.00, 1.00)
#> 
#> Random effect SD (sigma):
#>   PC prior: P(x > 1.00) = 0.010
#>     => Exponential(4.605)
#> 
#> Overdispersion (phi):
#>   PC prior: P(x > 10.00) = 0.010
#>     => Exponential(0.461)
#> 
#> Temporal autocorrelation (rho_temporal):
#>   Beta(2.00, 2.00)  [mean = 0.50]
#> 
#> Spatial proportion (rho_spatial):
#>   Beta(1.00, 1.00)  [mean = 0.50]

# Tighter random effect prior
tulpa_priors(sigma = prior_pc(U = 0.5, alpha = 0.01))
#> tulpa prior specification
#> =========================
#> 
#> Fixed effects (beta):
#>   Normal(0.00, 2.50)
#> 
#> Random effect SD (sigma):
#>   PC prior: P(x > 0.50) = 0.010
#>     => Exponential(9.210)
#> 
#> Overdispersion (phi):
#>   PC prior: P(x > 10.00) = 0.010
#>     => Exponential(0.461)
#> 
#> Temporal autocorrelation (rho_temporal):
#>   Beta(2.00, 2.00)  [mean = 0.50]
#> 
#> Spatial proportion (rho_spatial):
#>   Beta(1.00, 1.00)  [mean = 0.50]

# Half-Cauchy for random effect SD
tulpa_priors(sigma = prior_half_cauchy(2.5))
#> tulpa prior specification
#> =========================
#> 
#> Fixed effects (beta):
#>   Normal(0.00, 2.50)
#> 
#> Random effect SD (sigma):
#>   Half-Cauchy(2.50)
#> 
#> Overdispersion (phi):
#>   PC prior: P(x > 10.00) = 0.010
#>     => Exponential(0.461)
#> 
#> Temporal autocorrelation (rho_temporal):
#>   Beta(2.00, 2.00)  [mean = 0.50]
#> 
#> Spatial proportion (rho_spatial):
#>   Beta(1.00, 1.00)  [mean = 0.50]

# Informative prior for temporal correlation
tulpa_priors(rho_temporal = prior_beta(5, 2))  # Prior mode at ~0.8
#> tulpa prior specification
#> =========================
#> 
#> Fixed effects (beta):
#>   Normal(0.00, 2.50)
#> 
#> Random effect SD (sigma):
#>   PC prior: P(x > 1.00) = 0.010
#>     => Exponential(4.605)
#> 
#> Overdispersion (phi):
#>   PC prior: P(x > 10.00) = 0.010
#>     => Exponential(0.461)
#> 
#> Temporal autocorrelation (rho_temporal):
#>   Beta(5.00, 2.00)  [mean = 0.71]
#> 
#> Spatial proportion (rho_spatial):
#>   Beta(1.00, 1.00)  [mean = 0.50]
```
