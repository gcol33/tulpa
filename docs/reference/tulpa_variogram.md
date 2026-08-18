# Empirical semivariogram of residuals

Computes the empirical semivariogram in distance bins for visual
assessment of remaining spatial structure in residuals.

## Usage

``` r
tulpa_variogram(
  object,
  coords,
  n_bins = 15L,
  max_dist = NULL,
  resid_type = "pearson"
)
```

## Arguments

- object:

  A fitted model, or a numeric vector of residuals

- coords:

  N x 2 coordinate matrix (required)

- n_bins:

  Number of distance bins (default 15)

- max_dist:

  Maximum distance (default: half the maximum pairwise distance)

- resid_type:

  Residual type if extracting from model (default `"pearson"`)

## Value

A `tulpa_variogram` data.frame with columns `dist`, `gamma`, `n_pairs`
