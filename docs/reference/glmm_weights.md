# Compute GLM working weights for Laplace Hessian

Thin wrapper over the family-ops registry so the weight formulas live in
exactly one place (`R/family_loglik.R`). This is the weight the ENGINE's
Laplace Hessian carries, which is what every caller here rebuilds `H`
from, so which of the two curvatures it returns is decided by the
compiled dispatch rather than chosen independently:
`cpp_family_working_weight_is_observed()` names the families whose
compiled working weight IS the observed curvature, and for those the
y-free expected form is a different function (gcol33/tulpa#824).

## Usage

``` r
glmm_weights(eta, family, n_trials = NULL, phi = 1, phi2 = NULL, y = NULL)
```

## Details

`neg_binomial_2` is the one family where that bites: its compiled branch
returns `(y + phi) phi mu / (mu + phi)^2` while the registry's y-free
`weight` is the expected `mu phi / (mu + phi)`, which at a response away
from the mean differ by tens of percent. Every other family either has
no separate observed form or is one the compiled side answers with the
expected weight, so `y` changes nothing and may be omitted.
