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
#' No tulpa backend fits an interaction term, so this constructor errors. The
#' additive space-time model is fitted by `tulpa(spatial = , temporal = )` and
#' by [fit_st_nested()].
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
#' @return Nothing: the call always signals an error.
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
#' @seealso [spatial_car()], [spatial_gp()], [temporal_rw1()], [temporal_ar1()]
#'
#' @export
spatiotemporal <- function(spatial,
                           temporal,
                           type = c("I", "II", "III", "IV", "iid", "separable"),
                           shared = NULL) {
  stop("Spatiotemporal interaction terms are not fitted by any tulpa backend: ",
       "no solver carries a Knorr-Held type I-IV interaction, and tulpa() has ",
       "no `spatiotemporal =` argument. The additive space-time model ",
       "`X beta + u_spatial + v_temporal` is fitted by tulpa(spatial = , ",
       "temporal = ) through the joint nested-Laplace path, or directly by ",
       "fit_st_nested().", call. = FALSE)
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
  n_t <- st_info$n_times

  # Get interaction draws
  st_draws <- object$.internal$spatiotemporal_draws

  if (is.null(st_draws)) {
    stop("Spatiotemporal draws not found in model output", call. = FALSE)
  }

  n_draws <- dim(st_draws)[1]

  if (format == "array") {
    # Reshape to S x T x draws
    result <- array(NA_real_, dim = c(S, n_t, n_draws))
    for (d in seq_len(n_draws)) {
      result[, , d] <- matrix(st_draws[d, ], nrow = S, ncol = n_t, byrow = FALSE)
    }

    attr(result, "n_spatial") <- S
    attr(result, "n_times") <- n_t
    attr(result, "n_draws") <- n_draws
    class(result) <- c("tulpa_st_array", "array")
    return(result)

  } else if (format == "long") {
    # Create long-format data frame. Each draw's row is stored s-fastest
    # (matrix(nrow = S, ncol = T) column-major, as in the array format), so
    # enumerate s before t to keep the labels aligned when S != T.
    result <- expand.grid(
      draw = seq_len(n_draws),
      s = seq_len(S),
      t = seq_len(n_t)
    )
    result$value <- as.vector(st_draws)
    result <- result[, c("s", "t", "draw", "value")]

    class(result) <- c("tulpa_st_long", "data.frame")
    return(result)

  } else {
    # Compute summary statistics
    st_mat <- matrix(NA_real_, nrow = S * n_t, ncol = 3 + length(probs))
    colnames(st_mat) <- c("s", "t", "mean", paste0("q", probs * 100))

    for (i in seq_len(S)) {
      for (j in seq_len(n_t)) {
        # Draws are stored s-fastest (column-major S x T), matching the array /
        # long formats; index the (s = i, t = j) column accordingly so labels
        # stay aligned when S != T.
        idx <- (j - 1) * S + i
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
    attr(result, "n_times") <- n_t
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
#' @return A `ggplot` object when ggplot2 is installed; otherwise `NULL`
#'   invisibly, after drawing a base-graphics plot. Called for the side effect
#'   of visualizing the spatiotemporal interaction effects.
#'
#' @importFrom graphics image matplot barplot abline
#' @importFrom grDevices hcl.colors
#'
#' @export
plot.tulpa_st_summary <- function(x, type = "heatmap", ...) {

  S <- attr(x, "n_spatial")
  n_t <- attr(x, "n_times")

  if (type == "heatmap") {
    # Create matrix of means
    mean_mat <- matrix(x$mean, nrow = S, ncol = n_t, byrow = FALSE)

    if (requireNamespace("ggplot2", quietly = TRUE)) {
      df <- data.frame(
        s = rep(seq_len(S), n_t),
        t = rep(seq_len(n_t), each = S),
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
    image(seq_len(n_t), seq_len(S), t(mean_mat),
          xlab = "Time", ylab = "Space",
          main = "Spatiotemporal Interaction Effects",
          col = hcl.colors(100, "RdBu", rev = TRUE),
          ...)

  } else if (type == "time_series") {
    # Plot time series for each spatial unit
    mean_mat <- matrix(x$mean, nrow = S, ncol = n_t, byrow = FALSE)

    if (requireNamespace("ggplot2", quietly = TRUE)) {
      df <- data.frame(
        s = factor(rep(seq_len(S), n_t)),
        t = rep(seq_len(n_t), each = S),
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
    matplot(seq_len(n_t), t(mean_mat), type = "l", lty = 1,
            xlab = "Time", ylab = "Interaction Effect",
            main = "Spatiotemporal Effects by Location", ...)
    abline(h = 0, lty = 2, col = "gray50")

  } else if (type == "spatial_map") {
    # The summary is indexed by spatial unit (s = 1..S) x time (t = 1..T) with
    # no geographic coordinates, so "spatial_map" is the time-averaged effect
    # profile over the spatial units, not a coordinate map.
    mean_mat <- matrix(x$mean, nrow = S, ncol = n_t, byrow = FALSE)
    spatial_mean <- rowMeans(mean_mat)

    if (requireNamespace("ggplot2", quietly = TRUE)) {
      df <- data.frame(s = seq_len(S), value = spatial_mean)
      p <- ggplot2::ggplot(df, ggplot2::aes(x = .data$s, y = .data$value)) +
        ggplot2::geom_col(fill = "steelblue") +
        ggplot2::geom_hline(yintercept = 0, linetype = "dashed",
                            color = "gray50") +
        ggplot2::labs(
          title = "Time-averaged spatial effect",
          x = "Spatial unit", y = "Mean interaction effect"
        ) +
        theme_tulpa()
      return(p)
    }

    # Base R fallback
    barplot(spatial_mean, names.arg = seq_len(S),
            xlab = "Spatial unit", ylab = "Mean interaction effect",
            main = "Time-averaged spatial effect", ...)
    abline(h = 0, lty = 2, col = "gray50")

  } else {
    stop(sprintf(paste0(
      "Unknown plot `type` = '%s'. Use 'heatmap', 'time_series', or ",
      "'spatial_map'."), type), call. = FALSE)
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
#' No tulpa backend fits a joint space-time covariance, so this constructor
#' errors. A spatial GP alongside a temporal field is fitted by
#' `tulpa(spatial = spatial_gp(...), temporal = ...)`.
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
#' @return Nothing: the call always signals an error.
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
#' @export
spatiotemporal_gp <- function(coords,
                              time_var,
                              cov_space = c("exponential", "matern", "gaussian", "spherical"),
                              cov_time = c("exponential", "matern", "gaussian"),
                              nonsep_type = c("product", "sum", "gneiting", "cressie_huang"),
                              nn = 15,
                              shared = NULL) {
  stop("Non-separable spatiotemporal GP fields are not fitted by any tulpa ",
       "backend: no solver carries a joint space-time covariance, and tulpa() ",
       "has no `spatiotemporal =` argument. A spatial GP alongside a temporal ",
       "field is fitted by tulpa(spatial = spatial_gp(...), temporal = ...).",
       call. = FALSE)
}
