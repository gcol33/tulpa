# Dispatch a fit to the backend chosen by the mode/tier system.

The routing spine. Resolves `mode` to a concrete backend via
[`select_inference_mode()`](https://gillescolling.com/tulpa/reference/select_inference_mode.md),
asserts the backend is R-reachable (fails loudly otherwise), then calls
its fitter with `fitter_args`. The caller supplies `fitter_args`
matching the backend's input contract
(`BACKEND_REGISTRY$<backend>$input`).

The selected mode/tier/backend are stamped onto the returned fit
(without overwriting any the fitter already set), so the inference
contract is always visible in the output.

## Usage

``` r
tulpa_dispatch(
  mode,
  fitter_args = list(),
  family = NULL,
  n_obs = NULL,
  has_spatial = FALSE,
  has_temporal = FALSE,
  has_latent = FALSE,
  spatial_type = NULL,
  temporal = NULL
)
```

## Arguments

- mode:

  User-specified mode (`"auto"`, a tier, or a backend name).

- fitter_args:

  Named list of arguments forwarded to the backend fitter.

- family, n_obs, has_spatial, has_temporal, has_latent, spatial_type,
  temporal:

  Model characteristics forwarded to
  [`select_inference_mode()`](https://gillescolling.com/tulpa/reference/select_inference_mode.md).

## Value

The fitter's result, with `inference_mode`, `inference_tier`, `backend`,
and `selection_reason` ensured.
