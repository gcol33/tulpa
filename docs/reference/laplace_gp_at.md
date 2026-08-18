# NNGP Laplace at given hyperparameters

Single-point Laplace approximation for a Matern/exponential GP spatial
field at fixed (sigma2_gp, phi_gp). Used by `dispatch_laplace_spatial`
when `spatial$type == "gp"`. The neighbor structure is read straight off
the validated spec – call `validate_gp(spatial, data)` first if
constructing manually.

## Usage

``` r
laplace_gp_at(
  y,
  n_trials,
  X,
  spatial,
  family = "binomial",
  phi = 1,
  sigma2_gp = NULL,
  phi_gp = NULL,
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

  A `tulpa_gp` spec, validated (i.e., `neighbor_info` populated).

- family:

  Distribution family.

- phi:

  Dispersion parameter (negbin / gamma only).

- sigma2_gp:

  Marginal variance (NULL -\> 1.0).

- phi_gp:

  Range / decay parameter (NULL -\> 1.0).

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

The raw `cpp_laplace_fit_gp` result list, augmented with `sigma2_gp`,
`phi_gp`, and the spatial spec.
