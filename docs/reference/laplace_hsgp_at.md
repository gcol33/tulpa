# HSGP Laplace at given hyperparameters

Single-point Laplace for a Hilbert-space GP field at a fixed
`(sigma2, lengthscale)`, the conditional counterpart of the nested HSGP
integrator. Reuses the shared `make_hsgp_block` factory + dense spec
solver (DENSE_BASIS scatter) via `cpp_laplace_fit_hsgp`, so the mode
equals the nested kernel at that one grid cell. `sigma2` / `lengthscale`
default to the spec's fields (or `1`) and are recorded on the result.

## Usage

``` r
laplace_hsgp_at(
  y,
  n_trials,
  X,
  spatial,
  family = "binomial",
  phi = 1,
  sigma2 = NULL,
  lengthscale = NULL,
  re_idx = NULL,
  n_re_groups = 0L,
  sigma_re = 1,
  max_iter = 100L,
  tol = 1e-06,
  n_threads = 1L,
  offset = NULL,
  weights = NULL
)
```
