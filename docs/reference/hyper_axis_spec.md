# Describe one outer-grid hyperparameter axis

Builds a single axis spec for
[`tulpa_hyper_grid()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_grid.md).
Carries the candidate values, the optional log-prior, and the metadata
(log-scale, bounds, refinable flag) that the generic refinement /
consistency passes need.

## Usage

``` r
hyper_axis_spec(
  name,
  grid,
  log_prior = NULL,
  log_scale = FALSE,
  bounds = NULL,
  refinable = FALSE,
  atom_mass = NULL,
  slab_bounds = NULL,
  log_prior_coord = c("integration", "natural"),
  extend = TRUE
)
```

## Arguments

- name:

  Character. Axis label, used as the column name of the grid matrix and
  in posterior summaries.

- grid:

  Numeric vector of length \>= 1. The per-axis candidate values (the
  outer integration nodes on this axis). The full outer grid is the
  Cartesian product across axes.

- log_prior:

  Optional `function(x)` returning the scalar log prior density at axis
  value `x`. `NULL` (default) is a flat / improper prior (zero log-prior
  contribution). `log_prior_coord` says which coordinate the function is
  a density on.

- log_scale:

  Logical. Does the axis live naturally on a log scale (`sigma`, `tau`,
  `lengthscale`, ...)? Drives geometric vs arithmetic spacing in
  refinement and log-axis quantile fits. Default `FALSE`.

- bounds:

  Numeric vector of length 2 giving the natural support `(lower, upper)`
  of the axis, e.g. `c(0, Inf)` for `sigma`, `c(0, 1)` for a BYM2 mixing
  coefficient. `NULL` (default) is unbounded. A finite endpoint is
  treated as OPEN: it is the value the parameterisation degenerates at,
  so refinement clamps new points to the interior and the axis's
  quadrature cells and reported support are closed strictly inside it
  rather than half a node step past the outermost node.

- refinable:

  Logical. When `TRUE`, the axis participates in the adaptive-grid and
  var-of-means consistency passes (when those are enabled at the driver
  level). Spatial prior amplitudes (`sigma`) are typically left at the
  user-specified grid (`refinable = FALSE`); the copy coefficient
  `alpha` and per-arm dispersion `phi` typically opt in. Default
  `FALSE`.

- atom_mass:

  Numeric in `[0, 1)`, or `NULL` (default). Prior probability of the
  zero level on a log-scale axis. A `0` is a point mass, not a point of
  the log continuum, so its prior share has to be declared rather than
  inherited from the node count; the continuum nodes then share
  `1 - atom_mass` by quadrature weight. Required when `grid` contains a
  `0` on a `log_scale` axis. A `log_prior` on such an axis is a density
  on the continuum: it is not evaluated at the zero level and it shapes
  the continuum's share without moving the declared split, so
  `atom_mass` is the prior probability the fit integrates whatever the
  density is (gcol33/tulpa#624, gcol33/tulpa#626).

- slab_bounds:

  Numeric `c(lower, upper)`, or `NULL` (default). Fixed support of the
  continuum part of the prior. A flat measure on a log axis is improper,
  so the truncation bounds are what make it a proper density and they
  are therefore a prior choice: the weights normalise over `slab_bounds`
  and refinement is not allowed to move outside it. `NULL` leaves the
  node span acting as the support, which makes the prior depend on where
  refinement put the outermost node.

- log_prior_coord:

  One of `"integration"` (default) or `"natural"`, naming the coordinate
  `log_prior` is a density on. The contribution is added to the cell's
  `log_marginal`, which the integrator weights by cell widths measured
  on the axis's INTEGRATION coordinate (`log x` on a `log_scale` axis),
  so `"integration"` is carried through as written and is what the flat
  default's zero contribution is flat on. `"natural"` declares a density
  on `x` itself – a PC prior, `dexp`, `dgamma` – and the engine adds the
  change of variables `log(x)` on a `log_scale` axis. Every path that
  meets a declared density carries it across in one place
  (`.hyper_prior_carry()`). Inert on a linear axis, where the two
  coordinates coincide (gcol33/tulpa#623).

- extend:

  Logical. On a refinable axis, may the passes place a node beyond the
  outermost value in `grid`? `TRUE` (default) lets refinement follow the
  posterior out past the declared span, which is what an axis the engine
  placed wants. `FALSE` confines every new node to the interior of the
  declared span: the axis is integrated more finely over exactly the
  range given, and a mode outside it shows up as mass at the edge
  instead of moving the range. Inert when `refinable = FALSE`.

## Value

An object of class `tulpa_hyper_axis_spec` (a validated list with the
fields listed above).

## See also

[`tulpa_hyper_grid()`](https://gillescolling.com/tulpa/reference/tulpa_hyper_grid.md).
