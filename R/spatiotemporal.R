#' Spatiotemporal interaction specifications for tulpa
#'
#' @description
#' Functions to specify spatiotemporal interaction effects for tulpa models.
#' These capture dependencies that arise when spatial patterns vary over time,
#' or when temporal trends differ across space.
#'
#' @details
#' Spatiotemporal interactions extend the basic additive model:
#'
#' \deqn{\eta_{st} = X\beta + f_s(space) + f_t(time)}
#'
#' to include interactions:
#'
#' \deqn{\eta_{st} = X\beta + f_s(space) + f_t(time) + \delta_{st}}
#'
#' where \eqn{\delta_{st}} captures space-time interactions.
#'
#' **Interaction Types (following Knorr-Held, 2000):**
#'
#' - **Type I**: Unstructured interaction - IID \eqn{\delta_{st} \sim N(0, \sigma^2)}
#' - **Type II**: Structured time, unstructured space - temporal structure at each location
#' - **Type III**: Structured space, unstructured time - spatial structure at each time
#' - **Type IV**: Structured space AND time - full Kronecker interaction
#'
#' **Separable Models:**
#'
#' - **Separable**: Covariance is Kronecker product \eqn{C_{st} = C_s \otimes C_t}
#' - **Non-separable**: GP with joint space-time metric
#'
#' @name spatiotemporal
#' @references
#' Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time variation
#' in disease risk. Statistics in Medicine, 19(17-18), 2555-2567.
#'
#' @keywords models
NULL


#' Spatiotemporal interaction specification
#'
#' @description
#' Specify a spatiotemporal interaction effect for tulpa models.
#' The interaction captures structured or unstructured deviation from
#' the additive spatial + temporal model.
#'
#' @param spatial A spatial specification from [spatial_car()], [spatial_bym2()],
#'   or [spatial_gp()].
#' @param temporal A temporal specification from [temporal_rw1()], [temporal_rw2()],
#'   [temporal_ar1()], or [temporal_gp()].
#' @param type Interaction type:
#'   - `"I"` or `"iid"`: Unstructured interaction (IID)
#'   - `"II"`: Structured time at each location
#'   - `"III"`: Structured space at each time point
#'   - `"IV"`: Fully structured (Kronecker product of spatial and temporal)
#'   - `"separable"`: Separable covariance (Kronecker product)
#' @param shared Logical; if TRUE (default), spatiotemporal effect enters both
#'   all processes. Set to FALSE for process-specific effects
#'   (triggers warning about potential confounding).
#'
#' @return A `tulpa_spatiotemporal` object
#'
#' @details
#' **Type I (IID)**
#'
#' Independent random effect for each space-time combination:
#' \deqn{\delta_{st} \stackrel{iid}{\sim} N(0, \sigma^2_\delta)}
#'
#' This is the simplest form, requiring S*T parameters but capturing no
#' structured interaction.
#'
#' **Type II (Temporal structure per location)**
#'
#' Each location has its own temporal random effect:
#' \deqn{\delta_{\cdot t}^{(s)} \sim RW(\sigma^2)}
#'
#' This captures location-specific temporal trends but assumes independence
#' across locations.
#'
#' **Type III (Spatial structure per time point)**
#'
#' Each time point has its own spatial random effect:
#' \deqn{\delta_{s \cdot}^{(t)} \sim ICAR(\tau)}
#'
#' This captures time-specific spatial patterns but assumes independence
#' across time points.
#'
#' **Type IV (Full structure)**
#'
#' Kronecker product of spatial and temporal precision matrices:
#' \deqn{Q_\delta = Q_s \otimes Q_t}
#'
#' This is the most constrained model, assuming the interaction has the
#' same structure as the marginal effects.
#'
#' **Separable**
#'
#' For GP-based effects, assumes separable covariance:
#' \deqn{C(\mathbf{s}_1, t_1; \mathbf{s}_2, t_2) = C_s(\mathbf{s}_1, \mathbf{s}_2) \cdot C_t(t_1, t_2)}
#'
#' @examples
#' # Create adjacency matrix for 10 regions
#' adj <- matrix(0, 10, 10)
#' for (i in 1:9) adj[i, i+1] <- adj[i+1, i] <- 1
#'
#' # Type I: Unstructured interaction
#' st1 <- spatiotemporal(
#'   spatial = spatial_car(adj, level = "group", group_var = "region"),
#'   temporal = temporal_rw1("year"),
#'   type = "I"
#' )
#' print(st1)
#'
#' # Type IV: Fully structured interaction
#' st4 <- spatiotemporal(
#'   spatial = spatial_car(adj, level = "group", group_var = "region"),
#'   temporal = temporal_rw1("year"),
#'   type = "IV"
#' )
#' print(st4)
#'
#' \dontrun{
#' # Generate synthetic spatiotemporal data (not run - experimental)
#' set.seed(123)
#' n_regions <- 10
#' n_years <- 8
#' df <- expand.grid(
#'   region = 1:n_regions,
#'   year = 2015:(2015 + n_years - 1)
#' )
#' df$x <- rnorm(nrow(df))
#' df$count <- rpois(nrow(df), lambda = 20)
#' df$effort <- rgamma(nrow(df), shape = 4, rate = 1)
#'
#' # Fit model with spatiotemporal interaction
#' fit <- tulpa(
#'   count | effort ~ x,
#'   data = df,
#'   family = tulpa_poisson_gamma(),
#'   spatiotemporal = spatiotemporal(
#'     spatial = spatial_car(adj, level = "group", group_var = "region"),
#'     temporal = temporal_rw1("year"),
#'     type = "IV"
#'   ),
#'   iter = 200, warmup = 100, chains = 1
#' )
#' summary(fit)
#' }
#'
#' @seealso [spatial_car()], [spatial_gp()], [temporal_rw1()], [temporal_ar1()]
#'
#' @export
spatiotemporal <- function(spatial,
                           temporal,
                           type = c("I", "II", "III", "IV", "iid", "separable"),
                           shared = NULL) {

  # Validate spatial specification

  if (!inherits(spatial, c("tulpa_spatial", "tulpa_gp", "tulpa_multiscale"))) {
    stop("`spatial` must be a tulpa spatial specification ",
         "(from spatial_car, spatial_bym2, spatial_gp, etc.)", call. = FALSE)
  }

  # Validate temporal specification
  if (!inherits(temporal, "tulpa_temporal")) {
    stop("`temporal` must be a tulpa temporal specification ",
         "(from temporal_rw1, temporal_rw2, temporal_ar1, temporal_gp, etc.)",
         call. = FALSE)
  }

  # Normalize type
  type <- match.arg(type)
  if (type == "iid") type <- "I"

  # Check compatibility
  if (type == "separable") {
    # Separable requires GP for at least one of spatial/temporal
    is_spatial_gp <- inherits(spatial, c("tulpa_gp", "tulpa_multiscale"))
    is_temporal_gp <- inherits(temporal, "tulpa_temporal_gp")

    if (!is_spatial_gp && !is_temporal_gp) {
      warning(
        "Separable interaction works best with GP spatial or temporal effects.\n",
        "Consider using type = 'IV' for CAR/RW combinations.",
        call. = FALSE
      )
    }
  }

  # Check for Kronecker interaction with proper CAR
  if (type == "IV" && inherits(spatial, "tulpa_spatial")) {
    if (!is.null(spatial$proper) && spatial$proper) {
      warning(
        "Type IV interaction with proper CAR may have identifiability issues.\n",
        "Consider using ICAR (proper = FALSE) instead.",
        call. = FALSE
      )
    }
  }

  # Warning for non-shared effects
  if (isFALSE(shared)) {
    warning(
      "Non-shared spatiotemporal effects (shared = FALSE) means effects are not shared across processes.\n",
      "Consider whether space-time interactions should be shared between\n",
      "processes if shared confounding structure is expected.",
      call. = FALSE
    )
  }

  structure(
    list(
      type = type,
      spatial = spatial,
      temporal = temporal,
      shared = shared,
      # Filled in during validation
      n_spatial = NULL,
      n_times = NULL,
      n_params = NULL,
      spatial_group_var = spatial$group_var,
      temporal_time_var = temporal$time_var
    ),
    class = c("tulpa_spatiotemporal", "list")
  )
}


#' Print method for tulpa_spatiotemporal
#'
#' @param x A tulpa_spatiotemporal object
#' @param ... Ignored
#'
#' @export
print.tulpa_spatiotemporal <- function(x, ...) {
  cat("tulpa Spatiotemporal Interaction Specification\n")
  cat("===============================================\n\n")

  type_desc <- switch(x$type,
    "I" = "Type I: Unstructured (IID)",
    "II" = "Type II: Structured time, unstructured space",
    "III" = "Type III: Structured space, unstructured time",
    "IV" = "Type IV: Fully structured (Kronecker)",
    "separable" = "Separable covariance"
  )
  cat("Interaction type:", type_desc, "\n\n")

  cat("Spatial component:\n")
  cat("  Type:", class(x$spatial)[1], "\n")
  if (!is.null(x$spatial$group_var)) {
    cat("  Group variable:", x$spatial$group_var, "\n")
  }
  if (!is.null(x$n_spatial)) {
    cat("  Spatial units:", x$n_spatial, "\n")
  }

  cat("\nTemporal component:\n")
  cat("  Type:", x$temporal$type, "\n")
  cat("  Time variable:", x$temporal$time_var, "\n")
  if (!is.null(x$n_times)) {
    cat("  Time points:", x$n_times, "\n")
  }

  cat("\nShared:", if (x$shared) "Yes (enters both processes)" else "No", "\n")

  if (!is.null(x$n_params)) {
    cat("Total interaction parameters:", x$n_params, "\n")
  }

  invisible(x)
}


#' Validate spatiotemporal specification against data
#'
#' @param st tulpa_spatiotemporal object
#' @param data Data frame
#'
#' @return Updated tulpa_spatiotemporal object with validated components
#' @keywords internal
validate_spatiotemporal <- function(st, data) {
  if (is.null(st)) return(NULL)

  # Validate spatial component
  if (inherits(st$spatial, "tulpa_hsgp")) {
    # HSGP spatial: validate coords, set n_spatial = m^2 (basis functions)
    st$spatial_is_hsgp <- TRUE
    coord_vars <- st$spatial$coord_vars
    coords <- as.matrix(data[, coord_vars, drop = FALSE])
    st$spatial$coords_matrix <- coords
    st$spatial$n_obs <- nrow(coords)
    m <- st$spatial$m %||% 6L
    st$n_spatial <- as.integer(m)^2  # m^2 basis functions replace S spatial units
  } else if (inherits(st$spatial, "tulpa_gp")) {
    st$spatial <- validate_gp(st$spatial, data)
    st$n_spatial <- st$spatial$n_spatial
  } else if (inherits(st$spatial, "tulpa_multiscale")) {
    st$spatial <- validate_gp(st$spatial, data)
    st$n_spatial <- st$spatial$n_spatial
  } else {
    validate_spatial(st$spatial, data)
    st$n_spatial <- st$spatial$n_spatial
  }

  # Validate temporal component
  if (inherits(st$temporal, "tulpa_temporal_gp")) {
    st$temporal <- validate_temporal_gp(st$temporal, data)
    st$n_times <- st$temporal$n_times
  } else if (inherits(st$temporal, "tulpa_temporal_multiscale")) {
    st$temporal <- validate_temporal_multiscale(st$temporal, data)
    st$n_times <- st$temporal$n_times
  } else {
    st$temporal <- validate_temporal(st$temporal, data)
    st$n_times <- st$temporal$n_times
  }

  # Compute number of interaction parameters based on type
  S <- st$n_spatial
  T <- st$n_times

  st$n_params <- switch(st$type,
    "I" = S * T,                    # IID: one per space-time combo
    "II" = S * T,                   # Temporal structure per location
    "III" = S * T,                  # Spatial structure per time point
    "IV" = S * T,                   # Fully structured
    "separable" = S * T             # Separable GP
  )

  # Build space-time indexing
  st$st_index <- build_st_index(st, data)

  st
}


#' Build space-time index mapping
#'
#' @param st Validated spatiotemporal object
#' @param data Data frame
#'
#' @return List with space-time indexing information
#' @keywords internal
build_st_index <- function(st, data) {
  N <- nrow(data)
  S <- st$n_spatial
  T <- st$n_times

  # Get spatial index for each observation
  if (isTRUE(st$spatial_is_hsgp)) {
    # HSGP-ST: no spatial grouping, spatial mapping via Phi basis matrix
    s_idx <- rep(1L, N)  # placeholder — C++ uses Phi instead
  } else if (!is.null(st$spatial$group_var)) {
    spatial_var <- st$spatial$group_var
    if (!(spatial_var %in% names(data))) {
      stop(sprintf("Spatial group variable '%s' not found in data", spatial_var),
           call. = FALSE)
    }
    s_vals <- data[[spatial_var]]
    if (is.factor(s_vals)) {
      s_idx <- as.integer(s_vals)
    } else {
      s_factor <- as.factor(s_vals)
      s_idx <- as.integer(s_factor)
    }
  } else {
    # Observation-level spatial (GP)
    s_idx <- seq_len(N)
  }

  # Get temporal index for each observation
  time_var <- st$temporal$time_var
  if (!(time_var %in% names(data))) {
    stop(sprintf("Time variable '%s' not found in data", time_var),
         call. = FALSE)
  }
  t_vals <- data[[time_var]]
  if (is.factor(t_vals)) {
    t_idx <- as.integer(t_vals)
  } else {
    unique_times <- sort(unique(t_vals))
    t_factor <- factor(t_vals, levels = unique_times)
    t_idx <- as.integer(t_factor)
  }

  # Compute flattened index: st_flat[n] = (s_idx[n] - 1) * T + t_idx[n]
  # This gives column-major ordering (s varies slowest)
  st_flat <- (s_idx - 1L) * T + t_idx

  list(
    s_idx = s_idx,
    t_idx = t_idx,
    st_flat = st_flat,
    N = N,
    S = S,
    T = T
  )
}


#' Prepare spatiotemporal data for HMC backend
#'
#' @param st Validated tulpa_spatiotemporal object
#' @param data Data frame
#'
#' @return List with C++ compatible spatiotemporal data
#' @keywords internal
prepare_spatiotemporal_for_hmc <- function(st, data) {
  if (is.null(st)) {
    return(list(
      has_spatiotemporal = FALSE,
      type = "none"
    ))
  }

  idx <- st$st_index
  S <- idx$S
  T <- idx$T

  # Prepare spatial precision structure
  spatial_Q_info <- prepare_spatial_precision(st$spatial)

  # Prepare temporal precision structure
  temporal_Q_info <- prepare_temporal_precision(st$temporal)

  result <- list(
    has_spatiotemporal = TRUE,
    type = st$type,
    shared = st$shared,
    n_spatial = S,
    n_times = T,
    n_params = st$n_params,

    # Observation indexing
    s_idx = idx$s_idx,
    t_idx = idx$t_idx,
    st_flat = idx$st_flat,

    # Spatial structure
    spatial_type = class(st$spatial)[1],
    spatial_Q = spatial_Q_info,

    # Temporal structure
    temporal_type = st$temporal$type,
    temporal_Q = temporal_Q_info,
    temporal_cyclic = isTRUE(st$temporal$cyclic)
  )

  # HSGP-ST fields
  if (isTRUE(st$spatial_is_hsgp)) {
    result$spatial_is_hsgp <- TRUE
    result$hsgp_m <- st$spatial$m %||% 6L
    result$hsgp_c <- st$spatial$c %||% 1.5
    result$hsgp_coords <- st$spatial$coords_matrix
    result$hsgp_scale_coords <- st$spatial$scale_coords %||% TRUE
  }

  result
}


#' Prepare spatial precision matrix info
#'
#' @param spatial tulpa_spatial object
#'
#' @return List with spatial precision structure
#' @keywords internal
prepare_spatial_precision <- function(spatial) {
  if (is.null(spatial)) return(NULL)

  if (inherits(spatial, "tulpa_hsgp")) {
    # HSGP: no adjacency or neighbor structure needed — spatial precision is spectral
    return(list(
      type = "hsgp",
      n = spatial$m^2,
      adj_row_ptr = integer(0),
      adj_col_idx = integer(0)
    ))
  }

  if (inherits(spatial, c("tulpa_gp", "tulpa_multiscale"))) {
    # GP: return neighbor info for NNGP
    list(
      type = "gp",
      n = spatial$n_spatial,
      nn = spatial$nn,
      coords = as.vector(t(spatial$coords_matrix)),
      nn_idx = as.vector(t(spatial$neighbor_info$nn_idx)),
      nn_dist = as.vector(t(spatial$neighbor_info$nn_dist)),
      nn_order = spatial$neighbor_info$nn_order,
      nn_order_inv = spatial$neighbor_info$nn_order_inv
    )
  } else {
    # CAR/BYM2: return adjacency structure
    adj <- as.matrix(spatial$adjacency)
    n <- nrow(adj)

    # Build CSR format (0-based row_ptr, 1-based col_idx — C++ does -1)
    row_ptr <- integer(n + 1)  # initialized to 0 (0-based CSR)
    col_idx <- integer(0)

    for (i in seq_len(n)) {
      neighbors <- which(adj[i, ] > 0)
      col_idx <- c(col_idx, neighbors)
      row_ptr[i + 1] <- row_ptr[i] + length(neighbors)
    }

    n_neighbors <- diff(row_ptr)

    # Compute Q_inv and L_Q for precision mass matrix
    Q <- diag(n_neighbors) - adj
    Q_reg <- Q + 0.01 * diag(n)
    Q_inv <- solve(Q_reg)
    L_Q_lower <- t(chol(Q_reg))  # Lower Cholesky

    list(
      type = if (spatial$type == "bym2") "bym2" else "car",
      n = n,
      adj_row_ptr = row_ptr,
      adj_col_idx = col_idx,
      n_neighbors = n_neighbors,
      proper = isTRUE(spatial$proper),
      scale_factor = spatial$scale_factor,
      Q_inv = as.numeric(Q_inv),
      L_Q = as.numeric(L_Q_lower)
    )
  }
}


#' Prepare temporal precision matrix info
#'
#' @param temporal tulpa_temporal object
#'
#' @return List with temporal precision structure
#' @keywords internal
prepare_temporal_precision <- function(temporal) {
  if (is.null(temporal)) return(NULL)

  T <- temporal$n_times
  type <- temporal$type
  cyclic <- isTRUE(temporal$cyclic)

  # Build temporal precision matrix Q_time for Kronecker mass
  Q_time <- NULL
  Q_time_inv_flat <- NULL
  L_time_flat <- NULL
  if (T >= 3) {
    if (type == "rw1") {
      # RW1: Q[i,i] = 2 (interior), 1 (boundary), Q[i,i+1] = Q[i+1,i] = -1
      Q_time <- matrix(0, T, T)
      for (i in 1:(T-1)) {
        Q_time[i, i] <- Q_time[i, i] + 1
        Q_time[i+1, i+1] <- Q_time[i+1, i+1] + 1
        Q_time[i, i+1] <- -1
        Q_time[i+1, i] <- -1
      }
    } else if (type == "rw2") {
      # RW2: second-order differences
      D <- matrix(0, T-2, T)
      for (i in 1:(T-2)) {
        D[i, i] <- 1; D[i, i+1] <- -2; D[i, i+2] <- 1
      }
      Q_time <- t(D) %*% D
    } else if (type == "ar1") {
      # AR1 with rho=0.5 as default (approximate)
      Q_time <- diag(T)
      for (i in 1:(T-1)) {
        Q_time[i, i+1] <- -0.5
        Q_time[i+1, i] <- -0.5
      }
    }
    if (!is.null(Q_time)) {
      Q_time_reg <- Q_time + 0.01 * diag(T)
      Q_time_inv <- solve(Q_time_reg)
      L_time_lower <- t(chol(Q_time_reg))
      Q_time_inv_flat <- as.numeric(Q_time_inv)
      L_time_flat <- as.numeric(L_time_lower)
    }
  }

  list(
    type = type,
    T = T,
    cyclic = cyclic,
    Q_time_inv = Q_time_inv_flat,
    L_time = L_time_flat
  )
}


#' Extract spatiotemporal effects from fitted model
#'
#' @description
#' Extract posterior distributions of spatiotemporal interaction effects
#' from a fitted tulpa model.
#'
#' @param object A `tulpa_fit` object fitted with `spatiotemporal` argument
#' @param format Output format: `"array"` (default, S x T x draws), `"long"`
#'   (data frame with s, t, draw, value columns), or `"summary"` (posterior summaries).
#' @param probs Quantiles to compute if `format = "summary"`.
#' @param ... Ignored
#'
#' @return Spatiotemporal effects in requested format
#'
#' @examples
#' \donttest{
#' # After fitting a model with spatiotemporal interaction...
#' # st_effects <- spatiotemporal_effects(fit)
#' # summary(st_effects)
#' }
#'
#' @export
spatiotemporal_effects <- function(object,
                                   format = c("array", "long", "summary"),
                                   probs = c(0.025, 0.5, 0.975),
                                   ...) {
  UseMethod("spatiotemporal_effects")
}


#' @rdname spatiotemporal_effects
#' @export
spatiotemporal_effects.tulpa_fit <- function(object,
                                              format = c("array", "long", "summary"),
                                              probs = c(0.025, 0.5, 0.975),
                                              ...) {

  format <- match.arg(format)

  # Check if model has spatiotemporal effects
  if (is.null(object$spatiotemporal)) {
    stop("Model was not fitted with spatiotemporal interaction.\n",
         "Use `spatiotemporal` argument in tulpa() to specify interaction.",
         call. = FALSE)
  }

  st_info <- object$spatiotemporal
  S <- st_info$n_spatial
  T <- st_info$n_times

  # Get interaction draws
  st_draws <- object$.internal$spatiotemporal_draws

  if (is.null(st_draws)) {
    stop("Spatiotemporal draws not found in model output", call. = FALSE)
  }

  n_draws <- dim(st_draws)[1]

  if (format == "array") {
    # Reshape to S x T x draws
    result <- array(NA_real_, dim = c(S, T, n_draws))
    for (d in seq_len(n_draws)) {
      result[, , d] <- matrix(st_draws[d, ], nrow = S, ncol = T, byrow = FALSE)
    }

    attr(result, "n_spatial") <- S
    attr(result, "n_times") <- T
    attr(result, "n_draws") <- n_draws
    class(result) <- c("tulpa_st_array", "array")
    return(result)

  } else if (format == "long") {
    # Create long-format data frame
    result <- expand.grid(
      draw = seq_len(n_draws),
      t = seq_len(T),
      s = seq_len(S)
    )
    result$value <- as.vector(st_draws)
    result <- result[, c("s", "t", "draw", "value")]

    class(result) <- c("tulpa_st_long", "data.frame")
    return(result)

  } else {
    # Compute summary statistics
    st_mat <- matrix(NA_real_, nrow = S * T, ncol = 3 + length(probs))
    colnames(st_mat) <- c("s", "t", "mean", paste0("q", probs * 100))

    for (i in seq_len(S)) {
      for (j in seq_len(T)) {
        idx <- (i - 1) * T + j
        st_mat[idx, "s"] <- i
        st_mat[idx, "t"] <- j
        st_mat[idx, "mean"] <- mean(st_draws[, idx])

        qs <- quantile(st_draws[, idx], probs = probs)
        for (k in seq_along(probs)) {
          st_mat[idx, 3 + k] <- qs[k]
        }
      }
    }

    result <- as.data.frame(st_mat)
    result$sd <- apply(st_draws, 2, sd)

    attr(result, "n_spatial") <- S
    attr(result, "n_times") <- T
    attr(result, "n_draws") <- n_draws
    class(result) <- c("tulpa_st_summary", "data.frame")
    return(result)
  }
}


#' Plot method for spatiotemporal effects
#'
#' @param x Spatiotemporal effects object
#' @param type Plot type: `"heatmap"` (default), `"time_series"`, or `"spatial_map"`
#' @param ... Additional arguments passed to plotting functions
#'
#' @importFrom graphics image matplot
#' @importFrom grDevices hcl.colors
#'
#' @export
plot.tulpa_st_summary <- function(x, type = "heatmap", ...) {

  S <- attr(x, "n_spatial")
  T <- attr(x, "n_times")

  if (type == "heatmap") {
    # Create matrix of means
    mean_mat <- matrix(x$mean, nrow = S, ncol = T, byrow = FALSE)

    if (requireNamespace("ggplot2", quietly = TRUE)) {
      df <- data.frame(
        s = rep(seq_len(S), T),
        t = rep(seq_len(T), each = S),
        value = as.vector(mean_mat)
      )

      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$t, y = .data$s, fill = .data$value)) +
        ggplot2::geom_tile() +
        ggplot2::scale_fill_gradient2(
          low = "blue", mid = "white", high = "red",
          midpoint = 0
        ) +
        ggplot2::labs(
          title = "Spatiotemporal Interaction Effects",
          x = "Time",
          y = "Space",
          fill = "Effect"
        ) +
        theme_tulpa()

      return(p)
    }

    # Base R fallback
    image(seq_len(T), seq_len(S), t(mean_mat),
          xlab = "Time", ylab = "Space",
          main = "Spatiotemporal Interaction Effects",
          col = hcl.colors(100, "RdBu", rev = TRUE),
          ...)

  } else if (type == "time_series") {
    # Plot time series for each spatial unit
    mean_mat <- matrix(x$mean, nrow = S, ncol = T, byrow = FALSE)

    if (requireNamespace("ggplot2", quietly = TRUE)) {
      df <- data.frame(
        s = factor(rep(seq_len(S), T)),
        t = rep(seq_len(T), each = S),
        value = as.vector(mean_mat)
      )

      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$t, y = .data$value, color = .data$s, group = .data$s)) +
        ggplot2::geom_line(alpha = 0.6) +
        ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
        ggplot2::labs(
          title = "Spatiotemporal Effects by Location",
          x = "Time",
          y = "Interaction Effect",
          color = "Location"
        ) +
        theme_tulpa()

      return(p)
    }

    # Base R fallback
    matplot(seq_len(T), t(mean_mat), type = "l", lty = 1,
            xlab = "Time", ylab = "Interaction Effect",
            main = "Spatiotemporal Effects by Location", ...)
    abline(h = 0, lty = 2, col = "gray50")
  }

  invisible(NULL)
}


#' Non-separable spatiotemporal GP
#'
#' @description
#' Specify a non-separable Gaussian Process for spatiotemporal effects.
#' Unlike separable models where the covariance factors as \eqn{C_s \otimes C_t},
#' non-separable models allow for direct space-time interaction in the covariance.
#'
#' @param coords A one-sided formula specifying coordinate columns (e.g.,
#'   `~ lon + lat`), or a character vector of length 2.
#' @param time_var Name of the time variable in data.
#' @param cov_space Spatial covariance: `"exponential"` (default), `"matern"`,
#'   `"gaussian"`, or `"spherical"`.
#' @param cov_time Temporal covariance: `"exponential"` (default), `"matern"`,
#'   or `"gaussian"`.
#' @param nonsep_type Non-separability type:
#'   - `"product"`: \eqn{C_{st} = C_s \cdot C_t} (separable, for reference)
#'   - `"sum"`: \eqn{C_{st} = C_s + C_t}
#'   - `"gneiting"`: Gneiting (2002) non-separable class
#'   - `"cressie_huang"`: Cressie-Huang (1999) non-separable class
#' @param nn Number of nearest neighbors for NNGP approximation. Default 15.
#' @param shared Logical; if TRUE (default), effect enters both processes.
#'
#' @return A `tulpa_st_gp` object
#'
#' @details
#' The non-separable covariance functions allow for more flexible space-time

#' dependence:
#'
#' **Gneiting class:**
#' \deqn{C(h, u) = \frac{\sigma^2}{(a|u|^{2\alpha} + 1)^{\tau}} \exp\left(-\frac{c\|h\|^{2\gamma}}{(a|u|^{2\alpha} + 1)^{\beta\gamma}}\right)}
#'
#' where h is spatial lag, u is temporal lag, and parameters control the
#' space-time interaction.
#'
#' **Cressie-Huang class:**
#' Constructed via Fourier transform methods to ensure positive definiteness.
#'
#' @references
#' Gneiting, T. (2002). Nonseparable, stationary covariance functions for
#' space-time data. Journal of the American Statistical Association, 97(458), 590-600.
#'
#' Cressie, N., & Huang, H. C. (1999). Classes of nonseparable, spatio-temporal
#' stationary covariance functions. Journal of the American Statistical
#' Association, 94(448), 1330-1340.
#'
#' @examples
#' # Non-separable spatiotemporal GP
#' st_gp <- spatiotemporal_gp(
#'   ~ lon + lat,
#'   time_var = "year",
#'   nonsep_type = "gneiting"
#' )
#' print(st_gp)
#'
#' @export
spatiotemporal_gp <- function(coords,
                              time_var,
                              cov_space = c("exponential", "matern", "gaussian", "spherical"),
                              cov_time = c("exponential", "matern", "gaussian"),
                              nonsep_type = c("product", "sum", "gneiting", "cressie_huang"),
                              nn = 15,
                              shared = NULL) {

  cov_space <- match.arg(cov_space)
  cov_time <- match.arg(cov_time)
  nonsep_type <- match.arg(nonsep_type)

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

  if (!is.character(time_var) || length(time_var) != 1) {
    stop("`time_var` must be a single character string", call. = FALSE)
  }

  # Validate nn
  if (!is.numeric(nn) || length(nn) != 1 || nn < 1) {
    stop("`nn` must be a positive integer", call. = FALSE)
  }
  nn <- as.integer(nn)

  # Warning for non-shared
  if (isFALSE(shared)) {
    warning(
      "Non-shared spatiotemporal GP effects (shared = FALSE) means effects are not shared across processes.",
      call. = FALSE
    )
  }

  structure(
    list(
      type = "st_gp",
      coord_vars = coord_vars,
      time_var = time_var,
      cov_space = cov_space,
      cov_time = cov_time,
      nonsep_type = nonsep_type,
      nn = nn,
      shared = shared,
      # Filled during validation
      n_obs = NULL,
      n_spatial = NULL,
      n_times = NULL,
      coords_matrix = NULL,
      time_values = NULL,
      neighbor_info = NULL
    ),
    class = c("tulpa_st_gp", "tulpa_spatiotemporal", "list")
  )
}


#' Print method for tulpa_st_gp
#'
#' @param x A tulpa_st_gp object
#' @param ... Ignored
#'
#' @export
print.tulpa_st_gp <- function(x, ...) {
  cat("tulpa Non-Separable Spatiotemporal GP\n")
  cat("======================================\n\n")

  cat("Coordinates:", paste(x$coord_vars, collapse = ", "), "\n")
  cat("Time variable:", x$time_var, "\n")
  cat("Spatial covariance:", x$cov_space, "\n")
  cat("Temporal covariance:", x$cov_time, "\n")

  nonsep_desc <- switch(x$nonsep_type,
    product = "Product (separable)",
    sum = "Sum",
    gneiting = "Gneiting non-separable",
    cressie_huang = "Cressie-Huang non-separable"
  )
  cat("Non-separability:", nonsep_desc, "\n")
  cat("Neighbors (NNGP):", x$nn, "\n")
  cat("Shared:", if (x$shared) "Yes" else "No", "\n")

  if (!is.null(x$n_obs)) {
    cat("\nObservations:", x$n_obs, "\n")
  }

  invisible(x)
}


#' Validate non-separable spatiotemporal GP
#'
#' @param st_gp tulpa_st_gp object
#' @param data Data frame
#'
#' @return Updated object with computed structure
#' @keywords internal
validate_st_gp <- function(st_gp, data) {
  if (is.null(st_gp)) return(NULL)
  if (!inherits(st_gp, "tulpa_st_gp")) return(st_gp)

  N <- nrow(data)

  # Check coordinate columns exist
  for (cv in st_gp$coord_vars) {
    if (!(cv %in% names(data))) {
      stop(sprintf("Coordinate variable '%s' not found in data", cv),
           call. = FALSE)
    }
  }

  # Check time variable exists
  if (!(st_gp$time_var %in% names(data))) {
    stop(sprintf("Time variable '%s' not found in data", st_gp$time_var),
         call. = FALSE)
  }

  # Extract coordinates
  coords <- cbind(
    data[[st_gp$coord_vars[1]]],
    data[[st_gp$coord_vars[2]]]
  )

  # Check for missing coordinates
  if (any(is.na(coords))) {
    stop("Coordinate columns contain missing values", call. = FALSE)
  }

  # Extract and scale time values
  time_vals <- data[[st_gp$time_var]]
  if (!is.numeric(time_vals)) {
    if (inherits(time_vals, c("Date", "POSIXt"))) {
      time_vals <- as.numeric(time_vals)
    } else {
      stop("Time variable must be numeric or a date/time type", call. = FALSE)
    }
  }

  if (any(is.na(time_vals))) {
    stop("Time variable contains missing values", call. = FALSE)
  }

  # Scale coordinates and time
  coords_scaled <- scale(coords)
  time_scaled <- as.vector(scale(time_vals))

  st_gp$n_obs <- N
  st_gp$n_spatial <- length(unique(paste(coords[, 1], coords[, 2], sep = "_")))
  st_gp$n_times <- length(unique(time_vals))
  st_gp$coords_matrix <- coords_scaled
  st_gp$time_values <- time_scaled

  # Compute space-time neighbors for NNGP
  # We order by time first, then space within time
  nn <- min(st_gp$nn, N - 1)
  st_gp$neighbor_info <- compute_st_nngp_neighbors(coords_scaled, time_scaled, nn)
  st_gp$n_params <- N

  st_gp
}


#' Compute nearest neighbors for spatiotemporal NNGP
#'
#' @param coords N x 2 matrix of coordinates
#' @param time N-vector of time values
#' @param k Number of nearest neighbors
#'
#' @return List with neighbor structure
#' @keywords internal
compute_st_nngp_neighbors <- function(coords, time, k) {
  N <- nrow(coords)

  # Order by time first (ensures temporal causality in conditioning)
  time_order <- order(time, coords[, 1], coords[, 2])

  coords_ordered <- coords[time_order, , drop = FALSE]
  time_ordered <- time[time_order]

  # Compute neighbors in space-time
  nn_idx <- matrix(0L, nrow = N, ncol = k)
  nn_dist_space <- matrix(Inf, nrow = N, ncol = k)
  nn_dist_time <- matrix(Inf, nrow = N, ncol = k)

  for (i in 2:N) {
    n_candidates <- min(i - 1, k)

    if (n_candidates > 0) {
      # Compute space-time distances to previous observations
      # Using simple Euclidean distance in scaled space-time
      space_dists <- sqrt(
        (coords_ordered[1:(i-1), 1] - coords_ordered[i, 1])^2 +
        (coords_ordered[1:(i-1), 2] - coords_ordered[i, 2])^2
      )
      time_dists <- abs(time_ordered[1:(i-1)] - time_ordered[i])

      # Combined distance (can be weighted)
      combined_dists <- sqrt(space_dists^2 + time_dists^2)

      if (length(combined_dists) <= k) {
        nn_order <- order(combined_dists)
        nn_idx[i, seq_along(combined_dists)] <- nn_order
        nn_dist_space[i, seq_along(combined_dists)] <- space_dists[nn_order]
        nn_dist_time[i, seq_along(combined_dists)] <- time_dists[nn_order]
      } else {
        nn_order <- order(combined_dists)[1:k]
        nn_idx[i, ] <- nn_order
        nn_dist_space[i, ] <- space_dists[nn_order]
        nn_dist_time[i, ] <- time_dists[nn_order]
      }
    }
  }

  list(
    nn_idx = nn_idx,
    nn_dist_space = nn_dist_space,
    nn_dist_time = nn_dist_time,
    nn_order = time_order,
    nn_order_inv = order(time_order),
    k = k
  )
}
