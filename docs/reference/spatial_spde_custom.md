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
  prior_range = c(0.5, 0.5),
  prior_sigma = c(1, 0.5)
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

  Prior for the spatial range. Default `c(0.5, 0.5)`.

- prior_sigma:

  Prior for the marginal standard deviation. Default `c(1, 0.5)`.

## Value

A `tulpa_spatial` object with type `"spde"`.
