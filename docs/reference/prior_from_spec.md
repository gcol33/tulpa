# Build a `prior` list for [`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md) from a tulpa spec object

Validates a `tulpa_temporal` or `tulpa_spatial` specification against
`data`, then converts it to the prior list shape consumed by
[`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md).
Mainly an internal helper for callers that already have a fitted spec;
users typically pass `spec` + `data` directly to
[`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
instead.

Supported spec types:

- `tulpa_temporal` with `type in {"rw1", "rw2", "ar1"}`

- `tulpa_spatial`, areal
  `type in {"car", "icar", "car_proper", "bym2"}`. An areal spec returns
  the graph fields only – `type`, `spatial_idx`, `n_spatial_units`,
  `adj_row_ptr`, `adj_col_idx`, `n_neighbors` – plus `scale_factor` for
  bym2 and the eigenvalue-derived `rho_bounds` for proper CAR. No grid
  field is returned for any areal type: the outer axes are built later
  by the family's registry `defaults()` closure, and proper CAR's
  `(tau, rho)` grid is built there from those bounds.

- `tulpa_gp` / `tulpa_hsgp`, continuous
  `type in {"gp", "nngp", "hsgp"}`: validated against `data` and routed
  to the `nngp` / `hsgp` nested kernel through the shared converter.

SPDE is the one continuous field not built here – it carries its own
(range, sigma) FEM integrator; call
[`fit_spde()`](https://gillescolling.com/tulpa/reference/fit_spde.md)
with the
[`spatial_spde()`](https://gillescolling.com/tulpa/reference/spatial_spde.md)
spec.

## Usage

``` r
prior_from_spec(spec, data)
```

## Arguments

- spec:

  A `tulpa_temporal` or `tulpa_spatial` object.

- data:

  Data frame the spec resolves time/group/site indices against.

## Value

A `prior` list ready for
[`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md).
