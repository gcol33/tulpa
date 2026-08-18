# Select inference mode and backend

Implements the mode selection logic for tulpa. Accepts either tier names
(auto, exact, structured, optimized) or backend names (hmc, ess, pg,
laplace, vi).

When mode is "auto", selects between Tier 1 (Exact) and Tier 2
(Structured) based on model characteristics. Never selects Tier 3
(Optimized) automatically.

## Usage

``` r
select_inference_mode(
  mode,
  family,
  n_obs,
  has_spatial = FALSE,
  has_temporal = FALSE,
  has_latent = FALSE,
  spatial_type = NULL,
  temporal = NULL,
  has_re = FALSE
)
```

## Arguments

- mode:

  User-specified mode or backend name

- family:

  Model family object

- n_obs:

  Number of observations

- has_spatial:

  Whether model has spatial effects

- has_temporal:

  Whether model has temporal effects

- has_latent:

  Whether model has latent factors

- has_re:

  Whether the model has any random-effect term (`(1 | g)` or
  `(1 + x | g)`), scalar or slope alike

## Value

List with:

- mode: The selected mode name

- backend: The selected backend

- tier: The tier number (1, 2, or 3)

- tier_name: The tier name

- reason: Explanation for the selection
