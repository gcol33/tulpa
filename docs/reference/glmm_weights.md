# Compute GLM working weights for Laplace Hessian

Thin wrapper over the family-ops registry
([`family_weight()`](https://gillescolling.com/tulpa/reference/family_weight.md))
so the weight formulas live in exactly one place (`R/family_loglik.R`).

## Usage

``` r
glmm_weights(eta, family, n_trials = NULL, phi = 1, phi2 = NULL)
```
