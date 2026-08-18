# Marginal H_beta for a BYM2 field. The latent is the two-block `[phi (structured), theta (unstructured)]` convolution; each enters the linear predictor through the field indicator scaled by its `d_fac` (`sigma * sqrt(rho) * scale_factor` for phi, `sigma * sqrt(1 - rho)` for theta), with `phi` carrying the augmented ICAR precision `.icar_precision_Q` and `theta ~ N(0, I)`. The conditional Laplace kernel hardcodes `sigma = 1`, `rho = 0.5`.

Marginal H_beta for a BYM2 field. The latent is the two-block
`[phi (structured), theta (unstructured)]` convolution; each enters the
linear predictor through the field indicator scaled by its `d_fac`
(`sigma * sqrt(rho) * scale_factor` for phi, `sigma * sqrt(1 - rho)` for
theta), with `phi` carrying the augmented ICAR precision
`.icar_precision_Q` and `theta ~ N(0, I)`. The conditional Laplace
kernel hardcodes `sigma = 1`, `rho = 0.5`.

## Usage

``` r
.marginal_H_beta_bym2(
  mode,
  X,
  spatial,
  family,
  phi,
  n_trials,
  weights = NULL,
  offset = NULL,
  sigma_spatial = 1,
  rho = 0.5,
  re_idx = NULL,
  n_re_groups = 0L,
  sigma_re = 1
)
```
