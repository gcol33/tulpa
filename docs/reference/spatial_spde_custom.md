# SPDE Spatial Field from Custom Matrices

Specify a continuous Matern spatial field using externally-provided FEM
matrices. Use this with meshes from fmesher, rSPDE, or any other source.

## Usage

``` r
spatial_spde_custom(
  C,
  G,
  A,
  nu = 1,
  prior_range = NULL,
  prior_sigma = NULL,
  coords = NULL
)
```

## Arguments

- C:

  Mass matrix (n_mesh x n_mesh sparse matrix, e.g. from
  `fmesher::fm_fem()$c0`).

- G:

  Stiffness matrix (n_mesh x n_mesh sparse matrix, e.g. from
  `fmesher::fm_fem()$g1`).

- A:

  Projection matrix (n_obs x n_mesh sparse matrix, e.g. from
  [`fmesher::fm_basis()`](https://inlabru-org.github.io/fmesher/reference/fm_basis.html)).

- nu:

  Matern smoothness parameter. A positive number; integer values give
  the exact FEM construction, fractional values the BRASIL rational SPDE
  approximation (supported by
  [`fit_spde()`](https://gillescolling.com/tulpa/reference/fit_spde.md);).
  Default 1.

- prior_range:

  PC prior on the spatial range, `c(U, alpha)` with P(range \< U) =
  alpha. `NULL` (the default) anchors it on `coords` as
  [`spatial_spde()`](https://gillescolling.com/tulpa/reference/spatial_spde.md)
  does, and needs them.

- prior_sigma:

  PC prior on the marginal standard deviation, `c(U, alpha)` with
  P(sigma \> U) = alpha. `NULL` (the default) is `c(3, 0.01)`.

- coords:

  Observation coordinates (an `n_obs x 2` matrix), read only to anchor
  the default range prior. Not needed when `prior_range` is given.

## Value

A `tulpa_spatial` object with type `"spde"`.
