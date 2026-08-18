# Schur complement of the latent block out of the joint Hessian.

Given the fixed-effect design `X`, the combined latent design `D` (the
iid RE indicator block stacked with the spatial-field design), the
latent prior precision `Q_latent`, and the GLM weights `W`, returns the
marginal fixed- effect precision
`H_beta = X'WX - (X'WD) (D'WD + Q_latent)^{-1} (X'WD)'`. Shared by the
SPDE and NNGP marginal-SE paths.

## Usage

``` r
.schur_H_beta(X, D, Q_latent, W)
```
