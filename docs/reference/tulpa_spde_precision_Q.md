# Rebuild the SPDE/Matern field precision matrix

Rebuilds the sparse precision matrix `Q(kappa, tau)` of an SPDE/Matern
field from its mesh geometry and a `(kappa, tau)` hyperparameter pair,
mirroring the compiled builder (`src/spde_qbuilder.h`) that
[`tulpa_nuts_spde()`](https://gillescolling.com/tulpa/reference/tulpa_nuts_spde.md)
and
[`fit_spde()`](https://gillescolling.com/tulpa/reference/fit_spde.md)
use internally. Intended for a caller running its own outer-grid loop
over `(range, sigma)` around a coupled model tulpa's own SPDE front
doors
([`fit_spde()`](https://gillescolling.com/tulpa/reference/fit_spde.md),
[`spatial_spde()`](https://gillescolling.com/tulpa/reference/spatial_spde.md))
do not fit directly – e.g. a per-species community layer sharing one
spatial field.

## Usage

``` r
tulpa_spde_precision_Q(spatial, kappa, tau_spde)
```

## Arguments

- spatial:

  A
  [`spatial_spde()`](https://gillescolling.com/tulpa/reference/spatial_spde.md)
  /
  [`spatial_spde_custom()`](https://gillescolling.com/tulpa/reference/spatial_spde_custom.md)
  spec, carrying `n_mesh`, `C0_diag`, `G` and `nu`.

- kappa:

  Spatial frequency `kappa = sqrt(8*nu) / range`.

- tau_spde:

  SPDE precision scale (see
  [`tulpa_spde_log_hyperprior()`](https://gillescolling.com/tulpa/reference/tulpa_spde_log_hyperprior.md)'s
  `(range, sigma)` -\> `(kappa, tau)` mapping via `.spde_kappa_tau()`).

## Value

A sparse `n_mesh x n_mesh` precision matrix.

## See also

[`tulpa_spde_log_hyperprior()`](https://gillescolling.com/tulpa/reference/tulpa_spde_log_hyperprior.md),
[`fit_spde()`](https://gillescolling.com/tulpa/reference/fit_spde.md),
[`spatial_spde()`](https://gillescolling.com/tulpa/reference/spatial_spde.md)

## Examples

``` r
n <- 6L
G <- Matrix::bandSparse(n, k = c(-1, 0, 1),
                        diagonals = list(rep(-1, n - 1), c(1, rep(2, n - 2), 1),
                                         rep(-1, n - 1)))
sp <- spatial_spde_custom(Matrix::Diagonal(n), G, Matrix::Diagonal(n),
                          prior_range = c(1, 0.5), prior_sigma = c(1, 0.01))
Q <- tulpa_spde_precision_Q(sp, kappa = sqrt(8) / 1, tau_spde = 1)
dim(Q)
```
