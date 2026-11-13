# Mode-centred central composite design over outer hyperparameters

Places the nodes of a central composite design (CCD) at the mode of an
outer log-marginal and scales them by its curvature, the outer
integration INLA uses (Rue, Martino and Chopin 2009). `eval_logm` is the
caller's inner solve: the log marginal at one physical hyperparameter
vector. Each axis is integrated in an unconstrained coordinate with a
flat prior there, the log of a positive scale or the value of a bounded
parameter, and its physical box bounds the mode-find and clamps the
nodes. The design reuses
[`ccd_grid()`](https://gillescolling.com/tulpa/reference/ccd_grid.md),
[`ccd_to_theta()`](https://gillescolling.com/tulpa/reference/ccd_grid.md)
and
[`ccd_weights()`](https://gillescolling.com/tulpa/reference/ccd_grid.md),
and the outer PSIS k-hat of the Gaussian proposal comes from
[`tulpa_batched_pareto_k()`](https://gillescolling.com/tulpa/reference/tulpa_batched_pareto_k.md).

## Usage

``` r
tulpa_ccd_outer_grid(
  eval_logm,
  axes,
  f0_mult = 1.1,
  diagnose_k = TRUE,
  k_samples = .nl_diag("ccd_k_samples"),
  max_modefind = 40L
)
```

## Arguments

- eval_logm:

  `function(theta)` returning the log marginal (a scalar) at the
  physical hyperparameter vector `theta` (axes in `axes` order), or a
  non-finite value where it is not defined.

- axes:

  List of axes, each `list(name, log_scale, lower, upper, start)`:
  `log_scale = TRUE` integrates `log(theta)`; `lower` / `upper` is the
  physical box; `start` the physical starting value of the mode-find.

- f0_mult:

  Multiplier on the design radius `sqrt(k)`.

- diagnose_k:

  Compute the outer PSIS k-hat.

- k_samples:

  Proposal draws for the k-hat.

- max_modefind:

  Iteration cap of the bounded mode-find (each step is one inner solve
  per coordinate).

## Value

A list with `ok`; when `TRUE` also `nodes` (`n_node x k` physical
values, columns named), `dnode` (design weights), `u_hat` and `L` (mode
and Cholesky factor of the outer covariance in the unconstrained
coordinates), `kind` (per-node label), `n_points`, `pareto_k`; when
`FALSE`, `reason`.

## Details

The design declines (returning `ok = FALSE` with the reason) when there
are fewer than two axes (a one-dimensional grid is cheaper and as
accurate), when the mode-find fails or lands on the box, or when the
outer curvature is not positive definite or is ill-conditioned (a weakly
identified axis); the caller then integrates on its tensor grid.

## See also

[`tulpa_hyper_grid()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_grid.md)
(`design = "ccd"`), which builds the axes from its axis specs.

## Examples

``` r
# A bivariate Gaussian log marginal in (log tau, rho).
f <- function(th) -0.5 * ((log(th[1]) - 1)^2 / 0.04 + (th[2] - 0.4)^2 / 0.01)
axes <- list(list(name = "tau", log_scale = TRUE, lower = 0.3, upper = 30,
                  start = 3),
             list(name = "rho", log_scale = FALSE, lower = 0.05,
                  upper = 0.95, start = 0.5))
cc <- tulpa_ccd_outer_grid(f, axes, diagnose_k = FALSE)
cc$nodes
```
