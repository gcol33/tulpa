# Dispatch spatial Laplace to the correct C++ backend

`weights` is the per-observation likelihood weight. It reaches the same
`BuiltinFamilyResponse::weights` channel the non-spatial route uses,
which scales each row's log-density, score and Fisher curvature by the
same `w_i`, so the mode these kernels return and the marginal precision
`.marginal_H_beta_*()` builds at it describe one model
(gcol33/tulpa#385).

## Usage

``` r
dispatch_laplace_spatial(
  y,
  n_trials,
  X,
  re_idx,
  n_re_groups,
  sigma_re,
  spatial,
  family,
  phi,
  max_iter,
  tol,
  n_threads,
  offset = NULL,
  weights = NULL
)
```
