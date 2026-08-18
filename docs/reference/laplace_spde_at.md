# SPDE Laplace at given hyperparameters

Single-point Laplace approximation for an SPDE spatial field at a fixed
(range, sigma). Used by both `dispatch_laplace_spatial` (single-point
path) and `fit_spde` (single-point branch) so the call site stays a
single source of truth.

## Usage

``` r
laplace_spde_at(
  y,
  n_trials,
  X,
  spatial,
  family = "binomial",
  phi = 1,
  range = NULL,
  sigma = NULL,
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

## Arguments

- y:

  Response vector.

- n_trials:

  Trial sizes (binomial).

- X:

  Fixed-effects design matrix.

- spatial:

  A `tulpa_spatial` object of type `"spde"`.

- family:

  Distribution family.

- phi:

  Dispersion parameter (negbin / gamma only).

- range:

  Spatial range (NULL -\> use `spatial$prior_range[1]`).

- sigma:

  Marginal SD (NULL -\> use `spatial$prior_sigma[1]`).

- max_iter:

  Newton iterations.

- tol:

  Newton tolerance.

- n_threads:

  OpenMP threads.

- weights:

  Optional per-observation likelihood weights (length `length(y)`),
  scaling each row's log-density, score and Fisher curvature.

## Value

The raw `cpp_laplace_fit_spde` result list (mode, log_det_Q,
log_marginal, n_iter, converged), augmented with `range`, `sigma`, and
the spatial spec for downstream prediction.
