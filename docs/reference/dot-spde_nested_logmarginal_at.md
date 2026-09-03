# Numerically stable Laplace log-marginal for a fractional rSPDE at (range, sigma)

Delegates the well-conditioned `B` / matrix-determinant-lemma marginal
to C++ (`cpp_spde_fractional_logmarginal()`). The precision-space
`0.5(log|Q| - log|H|)` is corrupted in a range-dependent way by the
rational precision's wide spectrum (cond(Q) ~ 1e13+), so the marginal is
formed through the obs-space
`B = (A_eff Pl^{-1}) C (A_eff Pl^{-1})' + X X'/tau_beta`, built through
the operator factor `Pl` (cond = sqrt cond(Q)), never an explicit `Q`
inverse. Gaussian is the exact conjugate marginal; non-gaussian uses the
det-lemma at the precomputed Laplace mode. `phi` follows the one engine
convention (the Gaussian residual variance) and is converted at the
kernel calls below.

## Usage

``` r
.spde_nested_logmarginal_at(
  spatial,
  range,
  sigma,
  y,
  X,
  family,
  phi,
  n_trials,
  order,
  max_iter,
  tol,
  n_threads,
  offset,
  tau_beta = 1e-04
)
```

## Value

A list with `log_marginal`, `n_iter`, `converged`, and the assembly.
