# Negative Hessian of the zero-inflated mixture in eta space.

Returns an `n x 3` matrix with columns `count` (-d2/d eta_count^2), `zi`
(-d2/d logit_zi^2) and `cross` (-d2/d eta_count d logit_zi), i.e. the
distinct entries of the symmetric 2 x 2 block per observation.

## Usage

``` r
zi_neg_hessian(eta, logit_zi, y, family, n_trials = NULL, phi = 1, phi2 = NULL)
```

## Details

This is the observed negative Hessian, exact wherever the base family
registers `obs_weight`. At y \> 0 the mixture is additively separable,
so the count block is the base family's own curvature and the cross term
is zero; the coupling lives entirely in the y = 0 branch, where both
components can explain the zero.
