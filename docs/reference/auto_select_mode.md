# Auto-select mode (Tier 1 or Tier 2 only)

Implements the "auto" mode selection. Chooses the most reliable method
that is expected to finish for the given model.

**Critical rule**: Auto never selects Tier 3 (Optimized).

## Usage

``` r
auto_select_mode(
  family,
  n_obs,
  has_spatial,
  has_temporal,
  has_latent,
  temporal = NULL,
  spatial_type = NULL,
  has_re = FALSE
)
```
