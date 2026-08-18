# Assemble the fractional rSPDE precision and obs map at a given (range, sigma)

Shared by every fractional-`nu` fit path (Laplace single-point, nested,
NUTS, marginal SEs). Maps `(range, sigma)` to the SPDE hyperparameters
`kappa = sqrt(8 nu) / range`, builds the latent precision
`Q = Pl' C^{-1} Pl` and field shift `Pr` (field `u = Pr x`) from the
validated R oracle `.spde_rational_assemble()`, then **normalizes the
field to marginal variance `sigma^2`**. The rational construction is
correct in spectral shape but its overall scale carries kappa-dependent
constants (the `l_max` normalization, the per-factor conditioning
rescalings); without the variance normalization the implied field
variance is not `sigma^2` and varies with the range, which biases the
nested `(range, sigma)` integration. The normalization rescales `Pr`
(hence `A_eff = A Pr`) by `sigma / sqrt(mean marginal variance)`; `Q` is
untouched, so `logdet_Q` remains the correct prior normalizer for the
variance-normalized model.

## Usage

``` r
.spde_assemble_at(spatial, range, sigma, order = 2L)
```

## Arguments

- spatial:

  A validated `spatial_spde` spec with fractional `nu`.

- range, sigma:

  Spatial range and marginal SD.

- order:

  Rational approximation order. Default 2 (the rSPDE convention), which
  keeps the rational precision's condition number tractable.

## Value

A list with `Q`, `Pr`, `A_eff`, `Pl` (all CSC, sized to the non-orphan
submesh), `keep` (1-based indices of the retained mesh nodes),
`n_mesh_full` (the full mesh size), `kappa`, `var_scale`, `l_max`, and
`logdet_Q`.
