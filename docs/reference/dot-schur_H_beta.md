# Schur complement of the latent block out of the joint Hessian

Given the fixed-effect design `X`, the combined latent design `D` (the
iid RE indicator block stacked with the spatial-field design), the
latent prior precision `Q_latent`, and the GLM weights `W`, returns the
marginal fixed- effect precision
`H_beta = X'WX - (X'WD) (D'WD + Q_latent)^{-1} (X'WD)'`. Shared by the
SPDE and NNGP marginal-SE paths.

## Usage

``` r
.schur_H_beta(X, D, Q_latent, W, constr = NULL)
```

## Details

`constr` (optional, `ncol(D) x m`) holds hard linear constraints
`constr' u = 0` on the latent – the sum-to-zero of an intrinsic field.
The latent is then integrated over the constrained subspace only, so the
Schur complement takes the constrained latent covariance
`P^{-1} - P^{-1} A (A' P^{-1} A)^{-1} A' P^{-1}` (conditioning by
kriging, Rue & Held 2005) in place of `P^{-1}`, `P = D'WD + Q_latent`.
Without it the augmentation's finite variance on the intrinsic field's
level, which the intercept aliases, lands in the intercept's standard
error (gcol33/tulpa#901).
