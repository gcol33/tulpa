#' Spatially varying coefficient structure
#'
#' @description
#' Specify a spatially varying coefficient (SVC): one or more fixed-effect
#' coefficients are allowed to vary smoothly over space, with the variation
#' governed by a Gaussian process (NNGP or HSGP approximation).
#'
#' @param coords A formula (`~ lon + lat`) or character vector of length 2
#'   naming the two coordinate variables in the data.
#' @param terms Which coefficients vary over space. A formula, an integer vector
#'   of design-matrix column indices, or a character vector of term names.
#'   Default `1` (the intercept).
#' @param cov Covariance function. One of `"exponential"`, `"matern"`,
#'   `"gaussian"`, or `"spherical"`.
#' @param nn Number of nearest neighbours used in the NNGP approximation
#'   (`approx = "nngp"`).
#' @param shared Whether the effect is shared across processes in a
#'   multi-process model. `NULL` (default) shares it; `FALSE` fits
#'   process-specific effects and emits a warning.
#' @param scale_coords Logical. Standardize coordinates before fitting
#'   (default `TRUE`).
#' @param approx Spatial approximation. `"nngp"` (nearest-neighbour GP) or
#'   `"hsgp"` (Hilbert-space GP).
#' @param m Number of basis functions per dimension for the HSGP approximation
#'   (`approx = "hsgp"`).
#' @param c_boundary Boundary-extension factor for the HSGP domain
#'   (`approx = "hsgp"`).
#'
#' @return A `tulpa_svc` object (also of class `tulpa_spatial`).
#'
#' @seealso [spatial_gp()] for a spatial random effect (rather than a varying
#'   coefficient).
#'
#' @examples
#' # Intercept that varies smoothly over space
#' spatial_svc(~ lon + lat)
#'
#' @export
spatial_svc <- function(coords,
                        terms = 1,
                        cov = c("exponential", "matern", "gaussian", "spherical"),
                        nn = 15,
                        shared = NULL,
                        scale_coords = TRUE,
                        approx = c("nngp", "hsgp"),
                        m = 6,
                        c_boundary = 1.5) {

  cov <- match.arg(cov)
  approx <- match.arg(approx)

  # Parse coordinate specification
  if (inherits(coords, "formula")) {
    coord_vars <- all.vars(coords)
    if (length(coord_vars) != 2) {
      stop("`coords` formula must specify exactly 2 coordinate variables",
           call. = FALSE)
    }
  } else if (is.character(coords) && length(coords) == 2) {
    coord_vars <- coords
  } else {
    stop("`coords` must be a formula (~ lon + lat) or character vector of length 2",
         call. = FALSE)
  }

  # Parse terms specification
  if (inherits(terms, "formula")) {
    # Will be resolved against design matrix later
    terms_spec <- list(type = "formula", formula = terms)
  } else if (is.numeric(terms)) {
    terms_spec <- list(type = "index", indices = as.integer(terms))
  } else if (is.character(terms)) {
    terms_spec <- list(type = "names", names = terms)
  } else {
    stop("`terms` must be a formula, integer vector, or character vector",
         call. = FALSE)
  }

  # Validate nn (only relevant for NNGP)
  if (approx == "nngp") {
    if (!is.numeric(nn) || length(nn) != 1 || nn < 1) {
      stop("`nn` must be a positive integer", call. = FALSE)
    }
    nn <- as.integer(nn)
  }

  # Validate HSGP parameters
  if (approx == "hsgp") {
    if (!is.numeric(m) || length(m) != 1 || m < 3 || m > 50) {
      stop("`m` must be an integer between 3 and 50", call. = FALSE)
    }
    m <- as.integer(m)
    if (!is.numeric(c_boundary) || length(c_boundary) != 1 || c_boundary < 1.0) {
      stop("`c_boundary` must be a number >= 1.0", call. = FALSE)
    }
  }

  if (isFALSE(shared)) .warn_nonshared("SVCs", "spatially-varying effects")

  structure(
    list(
      type = "svc",
      coord_vars = coord_vars,
      terms_spec = terms_spec,
      cov = cov,
      nn = nn,
      shared = shared,
      scale_coords = scale_coords,
      approx = approx,
      m = m,
      c_boundary = c_boundary,
      # Filled in during validation
      n_obs = NULL,
      n_svc = NULL,
      svc_indices = NULL,
      svc_names = NULL,
      coords_matrix = NULL,
      neighbor_info = NULL
    ),
    class = c("tulpa_svc", "tulpa_spatial", "list")
  )
}


#' Print method for tulpa_svc
#'
#' @param x A tulpa_svc object
#' @param ... Ignored
#'
#' @return The input `x`, returned invisibly. Called for the side effect of
#'   printing the spatially-varying-coefficient specification to the console.
#'
#' @export
print.tulpa_svc <- function(x, ...) {
  cat("tulpa spatially-varying coefficients\n")
  cat("=====================================\n\n")

  cat("Coordinates:", paste(x$coord_vars, collapse = ", "), "\n")
  cat("Covariance:", x$cov, "\n")
  cat("Neighbors (NNGP):", x$nn, "\n")
  cat("Shared:", if (!isFALSE(x$shared)) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_svc)) {
    cat("\nSVC terms:", x$n_svc, "\n")
    if (!is.null(x$svc_names)) {
      cat("  ", paste(x$svc_names, collapse = ", "), "\n")
    }
  } else {
    cat("\nTerms: ")
    if (x$terms_spec$type == "formula") {
      cat(deparse(x$terms_spec$formula), "\n")
    } else if (x$terms_spec$type == "index") {
      cat("columns ", paste(x$terms_spec$indices, collapse = ", "), "\n")
    } else {
      cat(paste(x$terms_spec$names, collapse = ", "), "\n")
    }
  }

  if (!is.null(x$n_obs)) {
    cat("Observations:", x$n_obs, "\n")
  }

  invisible(x)
}


#' Validate SVC specification against data and design matrix
#'
#' @param svc tulpa_svc object
#' @param data Data frame
#' @param X Design matrix (to resolve term names)
#'
#' @return Updated tulpa_svc object with computed neighbor structure
#' @keywords internal
validate_svc <- function(svc, data, X) {
  if (is.null(svc)) return(NULL)
  if (!inherits(svc, "tulpa_svc")) return(svc)  # Other spatial types

  N <- nrow(data)
  p <- ncol(X)

  # Check coordinate columns exist
  for (cv in svc$coord_vars) {
    if (!(cv %in% names(data))) {
      stop(sprintf("Coordinate variable '%s' not found in data", cv),
           call. = FALSE)
    }
  }

  # Extract coordinates
  coords <- cbind(
    data[[svc$coord_vars[1]]],
    data[[svc$coord_vars[2]]]
  )

  # Check for missing coordinates
  if (any(is.na(coords))) {
    stop("Coordinate columns contain missing values", call. = FALSE)
  }

  # Scale coordinates if requested
  if (svc$scale_coords) {
    coords <- scale(coords)
  }

  # Resolve SVC terms
  coef_names <- colnames(X)
  if (is.null(coef_names)) {
    coef_names <- paste0("V", seq_len(p))
  }

  if (svc$terms_spec$type == "index") {
    svc_indices <- svc$terms_spec$indices
    if (any(svc_indices < 1 | svc_indices > p)) {
      stop(sprintf("SVC term indices must be between 1 and %d", p),
           call. = FALSE)
    }
    svc_names <- coef_names[svc_indices]

  } else if (svc$terms_spec$type == "names") {
    svc_names <- svc$terms_spec$names
    svc_indices <- match(svc_names, coef_names)
    if (any(is.na(svc_indices))) {
      missing <- svc_names[is.na(svc_indices)]
      stop(sprintf("SVC terms not found in design matrix: %s",
                   paste(missing, collapse = ", ")), call. = FALSE)
    }

  } else if (svc$terms_spec$type == "formula") {
    # Parse formula to get terms
    fmla <- svc$terms_spec$formula
    tt <- terms(fmla)
    term_labels <- attr(tt, "term.labels")
    has_intercept <- attr(tt, "intercept") == 1

    svc_names <- character(0)
    if (has_intercept) {
      svc_names <- c(svc_names, "(Intercept)")
    }
    svc_names <- c(svc_names, term_labels)

    svc_indices <- match(svc_names, coef_names)
    if (any(is.na(svc_indices))) {
      missing <- svc_names[is.na(svc_indices)]
      stop(sprintf("SVC terms not found in design matrix: %s",
                   paste(missing, collapse = ", ")), call. = FALSE)
    }
  }

  # Compute nearest neighbors (NNGP only)
  approx <- svc$approx %||% "nngp"
  if (approx == "nngp") {
    nn <- min(svc$nn, N - 1)
    neighbor_info <- compute_nngp_neighbors(coords, nn)
  } else {
    neighbor_info <- NULL
  }

  # Update SVC object
  svc$n_obs <- N
  svc$n_svc <- length(svc_indices)
  svc$svc_indices <- svc_indices
  svc$svc_names <- svc_names
  svc$coords_matrix <- coords
  svc$neighbor_info <- neighbor_info

  # Set spatial parameters for parameter layout
  if (approx == "hsgp") {
    m <- svc$m %||% 6L
    svc$n_spatial <- length(svc_indices) * as.integer(m)^2  # m^2 basis per SVC term
  } else {
    svc$n_spatial <- svc$n_obs * svc$n_svc  # N effects per SVC term
  }

  svc
}


#' Compute nearest neighbors for NNGP
#'
#' @description
#' Compute the k nearest neighbors for each observation using Euclidean
#' distance. Returns in a format suitable for the NNGP likelihood.
#'
#' @param coords N x 2 matrix of coordinates
#' @param k Number of nearest neighbors
#'
#' @return List with:
#'   - `nn_idx`: N x k matrix of neighbor indices (0 for obs with fewer neighbors)
#'   - `nn_dist`: N x k matrix of distances to neighbors
#'   - `nn_order`: Ordering of observations for NNGP (by coordinate)
#'
#' @keywords internal
compute_nngp_neighbors <- function(coords, k) {
  N <- nrow(coords)

  # Order observations (improves NNGP conditioning)
  # Use maximum-minimum distance ordering for better numerical properties
  order_idx <- order(coords[, 1], coords[, 2])

  # Reorder coordinates
  coords_ordered <- coords[order_idx, , drop = FALSE]

  # Compute neighbors for each observation
  nn_idx <- matrix(0L, nrow = N, ncol = k)
  nn_dist <- matrix(Inf, nrow = N, ncol = k)

  # Phase 1.3: Precompute pairwise distances among neighbors
  nn_neighbor_dist <- array(0, dim = c(N, k, k))

  for (i in 2:N) {
    # Only consider previous observations (in ordering) as potential neighbors
    n_candidates <- min(i - 1, k)

    if (n_candidates > 0) {
      # Compute distances to all previous observations
      dists <- sqrt(
        (coords_ordered[1:(i-1), 1] - coords_ordered[i, 1])^2 +
        (coords_ordered[1:(i-1), 2] - coords_ordered[i, 2])^2
      )

      # Find k nearest
      if (length(dists) <= k) {
        nn_order <- order(dists)
        nn_idx[i, seq_len(length(dists))] <- nn_order
        nn_dist[i, seq_len(length(dists))] <- dists[nn_order]
      } else {
        nn_order <- order(dists)[1:k]
        nn_idx[i, ] <- nn_order
        nn_dist[i, ] <- dists[nn_order]
      }

      # Phase 1.3: Compute pairwise distances among neighbors
      n_neighbors <- sum(nn_idx[i, ] > 0)
      if (n_neighbors > 1) {
        neighbor_indices <- nn_idx[i, 1:n_neighbors]
        neighbor_coords <- coords_ordered[neighbor_indices, , drop = FALSE]
        for (j1 in 1:n_neighbors) {
          for (j2 in 1:n_neighbors) {
            if (j1 == j2) {
              nn_neighbor_dist[i, j1, j2] <- 0
            } else {
              nn_neighbor_dist[i, j1, j2] <- sqrt(
                (neighbor_coords[j1, 1] - neighbor_coords[j2, 1])^2 +
                (neighbor_coords[j1, 2] - neighbor_coords[j2, 2])^2
              )
            }
          }
        }
      }
    }
  }

  list(
    nn_idx = nn_idx,
    nn_dist = nn_dist,
    nn_neighbor_dist = nn_neighbor_dist,  # Phase 1.3: cached pairwise distances
    nn_order = order_idx,
    nn_order_inv = order(order_idx),  # Inverse permutation
    k = k
  )
}


#' Extract spatially-varying coefficients from a fitted model
#'
#' @description
#' Extract posterior distributions of spatially-varying coefficients (SVCs)
#' from a fitted tulpa model with SVC specification.
#'
#' @param object A `tulpa_fit` object fitted with `svc` argument
#' @param terms Which SVC terms to extract. If NULL (default), extracts all.
#' @param summary Logical; if TRUE, return summary statistics instead of
#'   full posterior draws.
#' @param probs Quantiles to compute if `summary = TRUE`.
#' @param ... Ignored
#'
#' @return A `tulpa_svc_posterior` object containing:
#' - `draws`: Array of posterior draws (draws x locations x terms)
#' - `coords`: Coordinate matrix
#' - `term_names`: Names of SVC terms
#'
#' @examples
#' \dontrun{
#' # Generate synthetic spatial data (not run - SVC not fully supported)
#' set.seed(303)
#' n <- 40
#' df <- data.frame(
#'   lon = runif(n, 0, 10),
#'   lat = runif(n, 0, 10),
#'   depth = rnorm(n),
#'   count = rpois(n, 20),
#'   effort = rgamma(n, shape = 4, rate = 1)
#' )
#'
#' # Fit model with SVC
#' fit <- tulpa(
#'   count | effort ~ depth,
#'   data = df,
#'   family = tulpaRatio::tulpa_poisson_gamma(),
#'   svc = spatial_svc(~ lon + lat, terms = c(1, 2)),
#'   iter = 200, warmup = 100, chains = 1
#' )
#'
#' # Extract SVC posteriors
#' svc_post <- svc(fit)
#' summary(svc_post)
#' }
#'
#' @seealso [spatial_svc()], [plot.tulpa_svc_posterior()]
#'
#' @export
svc <- function(object, terms = NULL, summary = FALSE,
                probs = c(0.025, 0.5, 0.975), ...) {
  UseMethod("svc")
}


#' @rdname svc
#' @export
svc.tulpa_fit <- function(object, terms = NULL, summary = FALSE,
                           probs = c(0.025, 0.5, 0.975), ...) {
  .extract_varying_coef(
    object, terms, summary, probs,
    slot = "svc",
    info_class = "tulpa_svc",
    not_fitted_msg = paste0(
      "Model was not fitted with spatially-varying coefficients.\n",
      "Use `svc` argument in tulpa() to specify SVCs."),
    draws_field = "svc_draws",
    names_field = "svc_names",
    build_result = function(info, draws, term_names) {
      structure(
        list(
          draws = draws,
          coords = info$coords_matrix,
          term_names = term_names,
          n_obs = info$n_obs,
          n_svc = length(term_names),
          n_draws = dim(draws)[1],
          cov = info$cov
        ),
        class = "tulpa_svc_posterior"
      )
    }
  )
}


#' Summary method for tulpa_svc_posterior
#'
#' @param object A tulpa_svc_posterior object
#' @param probs Quantiles to compute
#' @param ... Ignored
#'
#' @return A `tulpa_svc_summary` data frame with one row per location and term,
#'   holding the observation index, term name, coordinates, and the posterior
#'   mean, SD, and requested quantiles of each spatially-varying coefficient.
#'
#' @export
summary.tulpa_svc_posterior <- function(object, probs = c(0.025, 0.5, 0.975), ...) {
  .summary_varying_coef(
    object, probs,
    n_terms = object$n_svc,
    lead_cols = function(j) data.frame(
      obs = seq_len(object$n_obs),
      term = object$term_names[j],
      coord_1 = object$coords[, 1],
      coord_2 = object$coords[, 2]
    ),
    summary_class = "tulpa_svc_summary"
  )
}


#' Print method for tulpa_svc_posterior
#'
#' @param x A tulpa_svc_posterior object
#' @param ... Ignored
#'
#' @return The input `x`, returned invisibly. Called for the side effect of
#'   printing a summary of the spatially-varying-coefficient posterior to the
#'   console.
#'
#' @export
print.tulpa_svc_posterior <- function(x, ...) {
  .print_varying_coef(
    x, "Spatially-varying",
    axis_label = "Locations", axis_value = x$n_obs,
    meta_label = "Covariance function", meta_value = x$cov,
    viz = "spatial"
  )
}


#' Plot method for tulpa_svc_posterior
#'
#' @param x A tulpa_svc_posterior object
#' @param term Which term to plot (name or index). Default: first term.
#' @param type Plot type: "mean" (default), "sd", or quantile (e.g., "q50")
#' @param ... Additional arguments passed to plotting functions
#'
#' @return A `ggplot` object when ggplot2 is installed; otherwise `NULL`
#'   invisibly, after drawing a base-graphics map. Called for the side effect of
#'   mapping the selected spatially-varying coefficient.
#'
#' @export
plot.tulpa_svc_posterior <- function(x, term = 1, type = "mean", ...) {

  if (is.character(term)) {
    term_idx <- match(term, x$term_names)
    if (is.na(term_idx)) {
      stop("Term not found: ", term, call. = FALSE)
    }
  } else {
    term_idx <- term
  }

  term_name <- x$term_names[term_idx]
  draws <- x$draws[, , term_idx]

  # Compute summary to plot
  if (type == "mean") {
    values <- colMeans(draws)
    title <- paste("SVC:", term_name, "(posterior mean)")
  } else if (type == "sd") {
    values <- apply(draws, 2, sd)
    title <- paste("SVC:", term_name, "(posterior SD)")
  } else if (grepl("^q[0-9]+", type)) {
    prob <- as.numeric(gsub("q", "", type)) / 100
    values <- apply(draws, 2, quantile, probs = prob)
    title <- paste0("SVC: ", term_name, " (", type, ")")
  } else {
    stop("Unknown plot type: ", type, call. = FALSE)
  }

  coords <- x$coords

  # Use ggplot2 if available
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    df <- data.frame(
      x = coords[, 1],
      y = coords[, 2],
      value = values
    )

    p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$x, y = .data$y, color = .data$value)) +
      ggplot2::geom_point(size = 2, ...) +
      ggplot2::scale_color_viridis_c() +
      ggplot2::labs(
        title = title,
        x = "Coordinate 1",
        y = "Coordinate 2",
        color = "Effect"
      ) +
      ggplot2::coord_fixed() +
      theme_tulpa()

    return(p)
  }

  # Base R fallback
  col_ramp <- colorRampPalette(c("blue", "white", "red"))
  n_colors <- 100
  colors <- col_ramp(n_colors)

  val_range <- range(values)
  val_scaled <- (values - val_range[1]) / diff(val_range)
  val_scaled[is.na(val_scaled)] <- 0.5
  point_colors <- colors[pmax(1, pmin(n_colors, ceiling(val_scaled * n_colors)))]

  plot(coords[, 1], coords[, 2],
       col = point_colors,
       pch = 19,
       xlab = "Coordinate 1",
       ylab = "Coordinate 2",
       main = title,
       asp = 1,
       ...)

  invisible(NULL)
}


