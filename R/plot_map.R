#' Plot spatial predictions as a map
#'
#' @description
#' Create publication-ready maps from tulpa spatial predictions. Supports
#' plotting ratio estimates, uncertainty (credible interval width), and
#' individual process predictions (numerator/denominator).
#'
#' @param x A `tulpa_fit` object with spatial structure, or a data frame
#'   containing predictions with coordinate columns.
#' @param coords A data frame or matrix with spatial coordinates (columns
#'   named 'x'/'y', 'X'/'Y', 'lon'/'lat', 'longitude'/'latitude', or
#'   'Easting'/'Northing'). Required if `x` is a data frame.
#' @param what What to plot: "ratio" (default), "numerator", "denominator",
#'   or "uncertainty".
#' @param summary Which summary statistic: "median" (default), "mean",
#'   "q2.5", "q97.5", or "sd".
#' @param newdata Optional data frame with prediction locations and covariates.
#'   If NULL and `x` is a tulpa_fit, uses fitted values at observed locations.
#' @param title Plot title. If NULL, auto-generated based on `what`.
#' @param palette Color palette: "viridis" (default), "magma", "plasma",
#'   "inferno", "cividis", "mako", "rocket", or a custom vector of colors.
#' @param points Logical; if TRUE, overlay observation points. Default FALSE.
#' @param point_color Color for observation points. Default "grey30".
#' @param point_size Size for observation points. Default 0.5.
#' @param na_color Color for NA values. Default "transparent".
#' @param crs Coordinate reference system (proj4 string or EPSG code).
#'   If NULL, uses planar coordinates.
#' @param legend_title Title for the color legend. If NULL, auto-generated.
#' @param ... Additional arguments passed to ggplot2 theme functions.
#'
#' @return A ggplot2 object that can be further customized.
#'
#' @details
#' This function provides a streamlined workflow for visualizing spatial
#' predictions from tulpa models. It handles:
#'
#' - Extracting predictions from tulpa_fit objects
#' - Converting to appropriate spatial format (stars/sf)
#' - Creating publication-quality maps with sensible defaults
#' - Uncertainty visualization via credible interval width
#'
#' For custom maps or more control, extract predictions using `ratio()` or
#' `predict()` and use ggplot2 directly with `geom_stars()` or `geom_sf()`.
#'
#' @section Required packages:
#' This function requires `ggplot2`. For raster-style maps, `stars` and `sf`
#' are also needed. Install with:
#' ```
#' install.packages(c("ggplot2", "stars", "sf"))
#' ```
#'
#' @examples
#' # plot_map requires a fitted spatial tulpa model
#' # See spatial_car() examples for fitting spatial models
#'
#' \dontrun{
#' # Fit a spatial ratio model through a model package (tulpaRatio owns the
#' # two-arm formula and the poisson/gamma ratio family):
#' set.seed(123)
#' n_sites <- 20
#' n <- 60
#' df <- data.frame(
#'   count = rpois(n, lambda = 8),
#'   effort = rgamma(n, shape = 4, rate = 1),
#'   elevation = rnorm(n),
#'   site = factor(rep(1:n_sites, length.out = n)),
#'   x = rep(runif(n_sites), length.out = n),
#'   y = rep(runif(n_sites), length.out = n)
#' )
#' # Create simple adjacency matrix
#' adj <- matrix(0, n_sites, n_sites)
#' for (i in 1:(n_sites-1)) adj[i, i+1] <- adj[i+1, i] <- 1
#' fit <- tulpa(
#'   count | effort ~ elevation + (1 | site),
#'   data = df,
#'   family = tulpaRatio::tulpa_poisson_gamma(),
#'   spatial = spatial_car(adj, group_var = "site"),
#'   mode = "laplace"
#' )
#' # plot_map(fit)  # requires ggplot2
#' }
#'
#' @seealso \code{ratio()} for extracting ratio posteriors,
#'   \code{predict.tulpa_fit()} for predictions at new locations
#'
#' @export
plot_map <- function(x,
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
                     ...) {


  what <- match.arg(what)
  summary <- match.arg(summary)


  # Check for required packages
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for plot_map(). Install with:\n",
         "  install.packages('ggplot2')", call. = FALSE)
  }

  # Extract predictions based on input type
  if (inherits(x, "tulpa_fit")) {
    plot_data <- prepare_map_data_from_fit(x, what, summary, newdata)
    obs_coords <- extract_coords_from_fit(x)
  } else if (is.data.frame(x)) {
    if (is.null(coords)) {
      stop("'coords' must be provided when 'x' is a data frame", call. = FALSE)
    }
    plot_data <- prepare_map_data_from_df(x, coords, what, summary)
    obs_coords <- NULL
  } else {
    stop("'x' must be a tulpa_fit object or a data frame", call. = FALSE)
  }

  # Auto-generate title if not provided
  if (is.null(title)) {
    title <- switch(what,
      fitted = "Fitted Values",
      uncertainty = "Prediction Uncertainty (95% CI Width)"
    )
  }

  # Auto-generate legend title if not provided
  if (is.null(legend_title)) {
    legend_title <- switch(what,
      fitted = switch(summary,
        median = "Median",
        mean = "Mean",
        q2.5 = "2.5%",
        q97.5 = "97.5%",
        sd = "SD"
      ),
      uncertainty = "CI Width"
    )
  }

  # Create the map
  create_map(
    plot_data = plot_data,
    obs_coords = if (points) obs_coords else NULL,
    title = title,
    palette = palette,
    point_color = point_color,
    point_size = point_size,
    na_color = na_color,
    crs = crs,
    legend_title = legend_title,
    ...
  )
}


#' Prepare map data from tulpa_fit object
#'
#' @keywords internal
prepare_map_data_from_fit <- function(fit, what, summary, newdata) {

  if (!is.null(newdata)) {
    # Predictions at new locations
    pred <- predict(fit, newdata = newdata, summary = TRUE)
    coords <- extract_coords(newdata)
  } else {
    # Fitted values at observed locations
    coords <- extract_coords_from_fit(fit)

    if (what == "ratio" || what == "uncertainty") {
      r <- ratio(fit, summary = TRUE)
      pred <- r
    } else if (what == "numerator") {
      pred <- fitted(fit, type = "numerator")
    } else {
      pred <- fitted(fit, type = "denominator")
    }
  }

  # Extract the appropriate summary statistic
  if (what == "uncertainty") {
    # CI width = q97.5 - q2.5
    if ("q97.5" %in% names(pred) && "q2.5" %in% names(pred)) {
      value <- pred$q97.5 - pred$q2.5
    } else if ("q975" %in% names(pred) && "q025" %in% names(pred)) {
      value <- pred$q975 - pred$q025
    } else {
      stop("Cannot compute uncertainty: quantiles not found in predictions",
           call. = FALSE)
    }
  } else {
    col_name <- switch(summary,
      median = c("median", "q50", "q0.5"),
      mean = "mean",
      q2.5 = c("q2.5", "q025"),
      q97.5 = c("q97.5", "q975"),
      sd = "sd"
    )
    # Find matching column
    matched <- intersect(col_name, names(pred))
    if (length(matched) == 0) {
      stop("Summary statistic '", summary, "' not found in predictions",
           call. = FALSE)
    }
    value <- pred[[matched[1]]]
  }

  data.frame(
    x = coords$x,
    y = coords$y,
    value = value
  )
}


#' Prepare map data from data frame
#'
#' @keywords internal
prepare_map_data_from_df <- function(x, coords, what, summary) {

  coords_df <- extract_coords(coords)

  # Determine which column to use
  if (what == "uncertainty") {
    # Look for CI width or compute from quantiles
    if ("ci_width" %in% names(x)) {
      value <- x$ci_width
    } else if ("q97.5" %in% names(x) && "q2.5" %in% names(x)) {
      value <- x$q97.5 - x$q2.5
    } else {
      stop("Cannot find uncertainty column. Provide 'ci_width' or quantiles.",
           call. = FALSE)
    }
  } else {
    col_name <- switch(summary,
      median = c("median", "q50", "q0.5"),
      mean = c("mean", "estimate"),
      q2.5 = c("q2.5", "q025", "lower"),
      q97.5 = c("q97.5", "q975", "upper"),
      sd = c("sd", "se", "std_error")
    )

    # Also check for what-specific columns
    what_cols <- switch(what,
      ratio = c("ratio", "ratio_est"),
      numerator = c("numerator", "mu_num", "fitted_num"),
      denominator = c("denominator", "mu_denom", "fitted_denom")
    )

    all_candidates <- c(col_name, what_cols)
    matched <- intersect(all_candidates, names(x))

    if (length(matched) == 0) {
      stop("Cannot find appropriate column for '", what, "' with summary '",
           summary, "'", call. = FALSE)
    }
    value <- x[[matched[1]]]
  }

  data.frame(
    x = coords_df$x,
    y = coords_df$y,
    value = value
  )
}


#' Extract coordinates from various formats
#'
#' @keywords internal
extract_coords <- function(x) {

  if (is.matrix(x)) {
    x <- as.data.frame(x)
  }

  # Try various coordinate column names
  x_names <- c("x", "X", "lon", "longitude", "Longitude", "Easting", "easting")
  y_names <- c("y", "Y", "lat", "latitude", "Latitude", "Northing", "northing")

  x_col <- intersect(x_names, names(x))
  y_col <- intersect(y_names, names(x))

  if (length(x_col) == 0 || length(y_col) == 0) {
    # Try first two columns if numeric
    if (ncol(x) >= 2 && is.numeric(x[[1]]) && is.numeric(x[[2]])) {
      return(data.frame(x = x[[1]], y = x[[2]]))
    }
    stop("Cannot identify coordinate columns. Use 'x'/'y', 'lon'/'lat', etc.",
         call. = FALSE)
  }

  data.frame(x = x[[x_col[1]]], y = x[[y_col[1]]])
}


#' Extract coordinates from tulpa_fit object
#'
#' @keywords internal
extract_coords_from_fit <- function(fit) {

  # Check various locations where coords might be stored
  if (!is.null(fit$coords)) {
    return(extract_coords(fit$coords))
  }

  if (!is.null(fit$data)) {
    coords <- tryCatch(
      extract_coords(fit$data),
      error = function(e) NULL
    )
    if (!is.null(coords)) return(coords)
  }

  if (!is.null(fit$stan_data$coords)) {
    return(extract_coords(fit$stan_data$coords))
  }

  # Return NULL if no coords found (will skip point overlay)
  NULL
}


#' Create the ggplot2 map
#'
#' @keywords internal
create_map <- function(plot_data, obs_coords, title, palette,
                       point_color, point_size, na_color, crs,
                       legend_title, ...) {

  # Determine if we can use stars (for gridded data) or need points
  use_stars <- can_use_stars(plot_data)

  if (use_stars && requireNamespace("stars", quietly = TRUE) &&
      requireNamespace("sf", quietly = TRUE)) {
    p <- create_raster_map(plot_data, crs, palette, na_color, legend_title)
  } else {
    p <- create_point_map(plot_data, palette, na_color, legend_title)
  }

  # Add observation points if requested
  if (!is.null(obs_coords)) {
    if (requireNamespace("sf", quietly = TRUE) && !is.null(crs)) {
      obs_sf <- sf::st_as_sf(obs_coords, coords = c("x", "y"), crs = crs)
      p <- p + ggplot2::geom_sf(data = obs_sf,
                                 color = point_color,
                                 size = point_size)
    } else {
      p <- p + ggplot2::geom_point(data = obs_coords,
                                    ggplot2::aes(x = .data$x, y = .data$y),
                                    color = point_color,
                                    size = point_size)
    }
  }

  # Add title and theme
  p <- p +
    ggplot2::labs(title = title, x = "Easting", y = "Northing") +
    ggplot2::theme_bw(base_size = 12) +
    ggplot2::coord_equal()

  p
}


#' Check if data is gridded (can use stars)
#'
#' @keywords internal
can_use_stars <- function(plot_data) {
  # Check if coordinates form a regular grid
  x_vals <- unique(plot_data$x)
  y_vals <- unique(plot_data$y)

  # If nx * ny approximately equals nrow, it's likely a grid
  expected_n <- length(x_vals) * length(y_vals)
  actual_n <- nrow(plot_data)

  # Allow some tolerance for missing cells

  abs(expected_n - actual_n) / expected_n < 0.1
}


#' Create raster-style map using stars
#'
#' @keywords internal
create_raster_map <- function(plot_data, crs, palette, na_color, legend_title) {

  # Convert to stars
  if (!is.null(crs)) {
    stars_obj <- stars::st_as_stars(plot_data, crs = crs)
  } else {
    stars_obj <- stars::st_as_stars(plot_data)
  }

  # Get palette function
  scale_fill <- get_palette_scale(palette, na_color, legend_title)

  ggplot2::ggplot() +
    stars::geom_stars(data = stars_obj,
                      ggplot2::aes(x = .data$x, y = .data$y, fill = .data$value)) +
    scale_fill
}


#' Create point-based map
#'
#' @keywords internal
create_point_map <- function(plot_data, palette, na_color, legend_title) {

  scale_fill <- get_palette_scale(palette, na_color, legend_title, geom = "point")

  ggplot2::ggplot(plot_data, ggplot2::aes(x = .data$x, y = .data$y,
                                           color = .data$value)) +
    ggplot2::geom_point() +
    scale_fill
}


#' Get appropriate color scale
#'
#' @keywords internal
get_palette_scale <- function(palette, na_color, legend_title, geom = "fill") {

  viridis_palettes <- c("viridis", "magma", "plasma", "inferno",
                        "cividis", "mako", "rocket", "turbo")

  # Check for viridis palette (single string)
  is_viridis <- length(palette) == 1 && palette %in% viridis_palettes

  if (geom == "fill") {
    if (is_viridis) {
      ggplot2::scale_fill_viridis_c(option = palette, na.value = na_color,
                                     name = legend_title)
    } else if (is.character(palette) && length(palette) > 1) {
      ggplot2::scale_fill_gradientn(colors = palette, na.value = na_color,
                                     name = legend_title)
    } else {
      ggplot2::scale_fill_viridis_c(na.value = na_color, name = legend_title)
    }
  } else {
    if (is_viridis) {
      ggplot2::scale_color_viridis_c(option = palette, na.value = na_color,
                                      name = legend_title)
    } else if (is.character(palette) && length(palette) > 1) {
      ggplot2::scale_color_gradientn(colors = palette, na.value = na_color,
                                      name = legend_title)
    } else {
      ggplot2::scale_color_viridis_c(na.value = na_color, name = legend_title)
    }
  }
}


#' Plot multiple maps in a grid
#'
#' @description
#' Create a multi-panel figure with ratio estimate and uncertainty maps
#' side by side.
#'
#' @param x A `tulpa_fit` object with spatial structure.
#' @param newdata Optional data frame with prediction locations.
#' @param ncol Number of columns in the grid. Default 2.
#' @param ... Additional arguments passed to `plot_map()`.
#'
#' @return A patchwork object (if patchwork is installed) or a list of ggplots.
#'
#' @examples
#' # See plot_map() examples for fitting a spatial model
#' # plot_map_panel(fit)
#' # plot_map_panel(fit, ncol = 1)
#'
#' @export
plot_map_panel <- function(x, newdata = NULL, ncol = 2, ...) {

  p1 <- plot_map(x, what = "ratio", newdata = newdata,
                 title = "Ratio Estimate", ...)
  p2 <- plot_map(x, what = "uncertainty", newdata = newdata,
                 title = "Uncertainty (95% CI Width)", ...)

  if (requireNamespace("patchwork", quietly = TRUE)) {
    patchwork::wrap_plots(p1, p2, ncol = ncol)
  } else {
    message("Install 'patchwork' for combined plots. Returning list.")
    list(estimate = p1, uncertainty = p2)
  }
}
