# Rational Approximation Coefficients for a Fractional SPDE

Returns the coefficient descriptor for an SPDE Matern field of operator
order `alpha = nu + 1` (in 2D). For integer `nu` the construction is
exact and no rational approximation is needed. For fractional `nu` it
returns the rational-SPDE roots from the BRASIL best-rational
approximation (see Details).

## Usage

``` r
rational_spde_coefficients(nu, m = 4L, lambda_range = c(1e-04, 10000))
```

## Arguments

- nu:

  Matern smoothness parameter. A positive number; may be fractional
  (e.g. 0.5, 1.5, 2.5).

- m:

  Rational approximation order (number of numerator / denominator
  factors) used for fractional `nu`. Higher `m` lowers the approximation
  error of the field's spectral density. Default 4.

- lambda_range:

  The generalized-eigenvalue spectrum `c(l_min, l_max)` of the FEM
  operator `CiL = C^{-1}(kappa^2 C + G)` over which the rational
  approximation is fitted. The approximation acts on the normalized
  interval `[l_min / l_max, 1]`. Used for fractional `nu` only.

## Value

For integer `nu`, a list with `is_integer = TRUE`, the operator order
`alpha`, `beta = alpha / 2`, `m = 0`, and empty `poles` / `weights`. For
fractional `nu`, a list with `is_integer = FALSE`, `alpha`, `beta`, the
rSPDE roots `rb` (denominator factors, drive `Pl`) and `rc` (numerator
factors, drive `Pr`), the integer power `m_beta`, the remaining
fractional exponent `beta_rem`, the scale constant `scale`, the rational
order `m`, and the approximation `error`.

## Details

For integer `nu` (1, 2, 3, ...) the operator order `alpha = nu + 1` is
an integer and the precision is assembled directly from integer powers
of the FEM operator `L = kappa^2 C + G` – an exact construction.

For fractional `nu` the field uses the operator-based rational SPDE
approximation (Bolin & Kirchner 2020). With `beta = alpha / 2` and
`m_beta = max(1, floor(beta))`, the field-mode variance of the assembled
precision `Q = Pl' C^{-1} Pl` (field `u = Pr x`, `x ~ N(0, Q^{-1})`)
tracks the Matern spectral density `l^{-2 beta}` when
`prod(1 - l rc) / (l^{m_beta - 1} prod(1 - l rb))` approximates
`l^{-beta_rem}` on the scaled spectrum,
`beta_rem = beta - (m_beta - 1)`. The roots come from the
degree-`(m, m)` best uniform (minimax) rational approximation of
`x^{-beta_rem}`, computed by the BRASIL algorithm (Hofreither 2021); its
numerator zeros map to `rc = 1 / zero` and denominator poles to
`rb = 1 / pole`. The field assembly from these roots is
`.spde_rational_assemble()`; it is validated against the Matern spectral
density in `test-spde-rational.R`.

## References

Bolin, D. & Kirchner, K. (2020). The rational SPDE approach for Gaussian
random fields with general smoothness. Journal of Computational and
Graphical Statistics, 29(2), 274-285.

Hofreither, C. (2021). An algorithm for best rational approximation
based on barycentric rational interpolation. Numerical Algorithms, 88,
365-388.
