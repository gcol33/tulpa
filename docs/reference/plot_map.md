# Plot spatial predictions as a map

Create publication-ready maps from a tulpa spatial fit: the fitted
response surface or the prediction uncertainty (credible-interval
width), on the response scale.

## Usage

``` r
plot_map(
  x,
  coords = NULL,
  what = c("fitted", "uncertainty"),
  summary = c("median", "mean", "q2.5", "q97.5", "sd"),
  newdata = NULL,
  title = NULL,
  palette = "viridis",
  points = FALSE,
  point_color = "grey30",
  point_size = 0.5,
  na_color = "transparent",
  crs = NULL,
  legend_title = NULL,
  ...
)
```

## Arguments

- x:

  A `tulpa_fit` object with spatial structure, or a data frame
  containing predictions with coordinate columns.

- coords:

  A data frame or matrix with spatial coordinates (columns named
  'x'/'y', 'X'/'Y', 'lon'/'lat', 'longitude'/'latitude', or
  'Easting'/'Northing'). Required if `x` is a data frame.

- what:

  What to plot: "fitted" (default, the response-scale point prediction)
  or "uncertainty" (the 95% credible-interval width).

- summary:

  Which summary statistic for `what = "fitted"`: "median" (default) /
  "mean" (both the point prediction), "q2.5" / "q97.5" (the credible
  bounds), or "sd" (the link-scale standard error).

- newdata:

  Optional data frame with prediction locations and covariates. If NULL
  and `x` is a tulpa_fit, uses fitted values at observed locations.

- title:

  Plot title. If NULL, auto-generated based on `what`.

- palette:

  Color palette: "viridis" (default), "magma", "plasma", "inferno",
  "cividis", "mako", "rocket", or a custom vector of colors.

- points:

  Logical; if TRUE, overlay observation points. Default FALSE.

- point_color:

  Color for observation points. Default "grey30".

- point_size:

  Size for observation points. Default 0.5.

- na_color:

  Color for NA values. Default "transparent".

- crs:

  Coordinate reference system (proj4 string or EPSG code). If NULL, uses
  planar coordinates.

- legend_title:

  Title for the color legend. If NULL, auto-generated.

- ...:

  Additional arguments passed to ggplot2 theme functions.

## Value

A ggplot2 object that can be further customized.

## Details

This function provides a streamlined workflow for visualizing spatial
predictions from tulpa models. It handles:

- Extracting predictions from tulpa_fit objects

- Converting to appropriate spatial format (stars/sf)

- Creating publication-quality maps with sensible defaults

- Uncertainty visualization via credible interval width

For custom maps or more control, extract predictions using
[`predict()`](https://rdrr.io/r/stats/predict.html) and use ggplot2
directly with `geom_stars()` or `geom_sf()`.

## Required packages

This function requires `ggplot2`. For raster-style maps, `stars` and
`sf` are also needed. Install with:

    install.packages(c("ggplot2", "stars", "sf"))

## See also

[`predict.tulpa_fit()`](https://gillescolling.com/tulpa/reference/predict.tulpa_fit.md)
for predictions at new locations

## Examples

``` r
# \donttest{
if (requireNamespace("ggplot2", quietly = TRUE)) {
  set.seed(123)
  n_sites <- 20
  df <- data.frame(
    y = rbinom(n_sites, 20, 0.4),
    elevation = rnorm(n_sites),
    site = factor(seq_len(n_sites)),
    lon = runif(n_sites),
    lat = runif(n_sites)
  )
  adj <- matrix(0, n_sites, n_sites)
  for (i in 1:(n_sites - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1
  fit <- tulpa(
    y ~ elevation + spatial(site),
    data = df,
    family = "binomial",
    n_trials = rep(20L, n_sites),
    spatial = spatial_car(adj, group_var = "site"),
    mode = "laplace"
  )
  # Areal (CAR/ICAR) fits carry no point coordinates, so pass `coords`:
  cc <- df[, c("lon", "lat")]
  plot_map(fit, coords = cc)                      # fitted response surface
  plot_map(fit, what = "uncertainty", coords = cc)
  plot_map_panel(fit, coords = cc)                # both side by side
}
# }
```
