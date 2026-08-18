# Validate the dispersion parameter `phi` for a family.

Errors when `family` carries a dispersion / precision parameter and
`phi` is not a positive finite scalar. A no-op for families without
dispersion (binomial, poisson) and for unknown / model-package families.
Shared by the
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) front
door and
[`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
so the rule lives in one place .

## Usage

``` r
.validate_family_phi(family, phi)
```
