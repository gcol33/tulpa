# Fractional rSPDE single-point Laplace fit at a fixed (range, sigma)

The fractional counterpart of the integer `cpp_laplace_fit_spde` branch
in
[`laplace_spde_at()`](https://gillescolling.com/tulpa/reference/laplace_spde_at.md).
Assembles `(Q, A_eff)` via
[`.spde_assemble_at()`](https://gillescolling.com/tulpa/reference/dot-spde_assemble_at.md)
and runs the precomputed C++ solve, whose latent is the auxiliary
weights `x`; the returned `mode` mesh block is mapped back to the field
`u = Pr x` so every downstream consumer reads field-space mesh effects
exactly as on the integer path. The log-marginal is the well-conditioned
`B` / determinant-lemma marginal (`cpp_spde_fractional_logmarginal()`)
for the RE-free case (comparable across the `(range, sigma)` grid); a
fit with an iid RE block keeps the precomputed precision-space marginal.

## Usage

``` r
.spde_laplace_fractional_at(
  y,
  n_trials,
  X,
  spatial,
  family,
  phi,
  range,
  sigma,
  re_idx,
  n_re_groups,
  sigma_re,
  max_iter,
  tol,
  n_threads,
  offset,
  weights = NULL,
  order = 2L
)
```
