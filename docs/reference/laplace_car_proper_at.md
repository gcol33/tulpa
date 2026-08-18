# Proper-CAR Laplace at given hyperparameters

Single-point Laplace for a proper-CAR areal field at a fixed
`(tau, rho)`, the conditional counterpart of the nested CAR_proper
integrator. Reuses the shared `make_car_proper_latent_blocks` factory +
dense spec solver via `cpp_laplace_fit_car_proper`, so the mode +
log-marginal equal the nested kernel at that one grid cell. `tau` /
`rho` default to the spec's fields (or `tau = 1`, `rho` = midpoint of
the eigenvalue-derived `rho_bounds`) and are recorded on the result.

## Usage

``` r
laplace_car_proper_at(
  y,
  n_trials,
  X,
  spatial,
  family = "binomial",
  phi = 1,
  tau = NULL,
  rho = NULL,
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
