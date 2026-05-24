#' Diagnostic Plotting Functions for tulpa Models
#'
#' @description
#' Visual diagnostic tools for MCMC convergence assessment. All functions
#' provide base R fallbacks when ggplot2/bayesplot are unavailable.
#'
#' @name plot_diagnostics
#' @keywords internal
NULL

# Consistent color scheme for diagnostic plots
tulpa_diag_colors <- list(

  good = "#2E7D32",      # Green - pass

  warn = "#F57F17",      # Yellow/orange - caution
  bad = "#C62828",       # Red - fail
  divergent = "#D32F2F", # Red for divergent transitions
  chain = c("#1b9e77", "#d95f02", "#7570b3", "#e7298a",
            "#66a61e", "#e6ab02", "#a6761d", "#666666")
)

# Internal theme: minimal without grid lines (cleaner diagnostic plots)
theme_tulpa <- function() {
  ggplot2::theme_minimal() +
    ggplot2::theme(panel.grid = ggplot2::element_blank())
}


#' Plot Rhat Convergence Diagnostic
#'
#' @description
#' Creates a visual display of Rhat values for all parameters, with color-coding
#' to highlight convergence issues. Rhat values > 1.01 indicate potential
#' convergence problems.
#'
#' @param fit A `tulpa_fit` object.
#' @param threshold Rhat threshold for warnings (default: 1.01).
#' @param pars Character vector of parameter names to include. If NULL (default),
#'   includes all main parameters (excludes high-dimensional RE/spatial).
#'
#' @return A ggplot object (if ggplot2 available) or base R plot (invisible).
#'
#' @details
#' Color coding:
#' - Green: Rhat < 1.01 (converged)
#' - Yellow: 1.01 <= Rhat < 1.05 (borderline)
#' - Red: Rhat >= 1.05 (not converged)
#'
#' @examples
#' # Diagnostic plots require a fitted tulpa model
#' # See tulpa() examples for fitting models
#'
#' \dontrun{
#' # Fit a model (slow, not run on CRAN)
#' set.seed(123)
#' n <- 50
#' df <- data.frame(
#'   count = rpois(n, lambda = 8),
#'   effort = rgamma(n, shape = 4, rate = 1),
#'   depth = rnorm(n),
#'   site = factor(rep(1:10, each = 5))
#' )
#' fit <- tulpa(count | effort ~ depth + (1|site), data = df,
#'              family = tulpa_poisson_gamma(),
#'              iter = 200, warmup = 100, chains = 1)
#' # plot_rhat(fit)
#' }
#'
#' @seealso [plot_ess()], [diagnostic_summary()], [mcmc_diagnostics()]
#' @export
plot_rhat <- function(fit, threshold = 1.01, pars = NULL) {
  if (!inherits(fit, "tulpa_fit")) {
    stop("fit must be a tulpa_fit object", call. = FALSE)
  }

  # Get diagnostics

  diag <- mcmc_diagnostics(fit, pars = pars)

  # Filter to main parameters if not specified
  if (is.null(pars)) {
    main_pars <- select_main_params(diag$parameter)
    diag <- diag[diag$parameter %in% main_pars, ]
  }

  # Remove NA values

  diag <- diag[!is.na(diag$rhat), ]

  if (nrow(diag) == 0) {
    message("No Rhat values available for selected parameters")
    return(invisible(NULL))
  }

  # Assign colors based on thresholds
  diag$color <- ifelse(
    diag$rhat < threshold, tulpa_diag_colors$good,
    ifelse(diag$rhat < 1.05, tulpa_diag_colors$warn, tulpa_diag_colors$bad)
  )

  # Sort by Rhat value

  diag <- diag[order(diag$rhat, decreasing = TRUE), ]
  diag$parameter <- factor(diag$parameter, levels = diag$parameter)

  # Plot with ggplot2 if available

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    p <- ggplot2::ggplot(diag, ggplot2::aes(x = .data$rhat, y = .data$parameter)) +
      ggplot2::geom_point(ggplot2::aes(color = .data$color), size = 3) +
      ggplot2::geom_vline(xintercept = threshold, linetype = "dashed",
                          color = tulpa_diag_colors$warn, linewidth = 0.8) +
      ggplot2::geom_vline(xintercept = 1, linetype = "solid",
                          color = "gray50", linewidth = 0.5) +
      ggplot2::scale_color_identity() +
      ggplot2::labs(
        title = "Rhat Convergence Diagnostic",
        x = expression(hat(R)),
        y = NULL,
        caption = paste0("Threshold: ", threshold, " (dashed line)")
      ) +
      theme_tulpa() +
      ggplot2::theme(
        plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"),
        panel.grid.major.y = ggplot2::element_line(color = "gray90")
      )

    return(p)
  }

  # Base R fallback
  plot_rhat_base(diag, threshold)
  invisible(NULL)
}


#' Base R Rhat plot
#' @keywords internal
plot_rhat_base <- function(diag, threshold) {
  n_pars <- nrow(diag)
  old_par <- graphics::par(mar = c(4, 8, 3, 1))
  on.exit(graphics::par(old_par))

  graphics::plot(
    diag$rhat, seq_len(n_pars),
    xlim = c(min(0.99, min(diag$rhat)), max(1.1, max(diag$rhat))),
    ylim = c(0.5, n_pars + 0.5),
    pch = 19, col = diag$color, cex = 1.5,
    xlab = expression(hat(R)),
    ylab = "",
    yaxt = "n",
    main = "Rhat Convergence Diagnostic"
  )

  graphics::axis(2, at = seq_len(n_pars), labels = diag$parameter, las = 2, cex.axis = 0.8)
  graphics::abline(v = threshold, lty = 2, col = tulpa_diag_colors$warn, lwd = 2)
  graphics::abline(v = 1, lty = 1, col = "gray50")
}


#' Plot Effective Sample Size Diagnostic
#'
#' @description
#' Creates a visual display of effective sample size (ESS) for all parameters,
#' expressed as a ratio of ESS to total samples. Low ESS indicates high
#' autocorrelation.
#'
#' @param fit A `tulpa_fit` object.
#' @param type Type of ESS: "bulk" (default) or "tail".
#' @param threshold Minimum acceptable ESS (default: 400).
#' @param pars Character vector of parameter names to include.
#'
#' @return A ggplot object (if ggplot2 available) or base R plot (invisible).
#'
#' @details
#' ESS/iter ratio interpretation:
#' - Green: ESS >= threshold
#' - Yellow: threshold/2 <= ESS < threshold
#' - Red: ESS < threshold/2
#'
#' @examples
#' # See plot_rhat() examples for fitting a model
#' # plot_ess(fit)
#' # plot_ess(fit, type = "tail")
#'
#' @seealso [plot_rhat()], [diagnostic_summary()], [mcmc_diagnostics()]
#' @export
plot_ess <- function(fit, type = c("bulk", "tail"), threshold = 400, pars = NULL) {
  if (!inherits(fit, "tulpa_fit")) {
    stop("fit must be a tulpa_fit object", call. = FALSE)
  }

  type <- match.arg(type)

  # Get diagnostics
  diag <- mcmc_diagnostics(fit, pars = pars)

  # Filter to main parameters if not specified
  if (is.null(pars)) {
    main_pars <- select_main_params(diag$parameter)
    diag <- diag[diag$parameter %in% main_pars, ]
  }

  # Select ESS column
  ess_col <- if (type == "bulk") "ess_bulk" else "ess_tail"
  diag$ess <- diag[[ess_col]]

  # Remove NA values
  diag <- diag[!is.na(diag$ess), ]

  if (nrow(diag) == 0) {
    message("No ESS values available for selected parameters")
    return(invisible(NULL))
  }

  # Assign colors based on thresholds
  diag$color <- ifelse(
    diag$ess >= threshold, tulpa_diag_colors$good,
    ifelse(diag$ess >= threshold / 2, tulpa_diag_colors$warn, tulpa_diag_colors$bad)
  )

  # Sort by ESS (ascending so worst are at top)
  diag <- diag[order(diag$ess), ]
  diag$parameter <- factor(diag$parameter, levels = diag$parameter)

  # Plot with ggplot2 if available
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    p <- ggplot2::ggplot(diag, ggplot2::aes(x = .data$ess, y = .data$parameter)) +
      ggplot2::geom_point(ggplot2::aes(color = .data$color), size = 3) +
      ggplot2::geom_vline(xintercept = threshold, linetype = "dashed",
                          color = tulpa_diag_colors$warn, linewidth = 0.8) +
      ggplot2::scale_color_identity() +
      ggplot2::labs(
        title = paste0("Effective Sample Size (", type, ")"),
        x = "ESS",
        y = NULL,
        caption = paste0("Threshold: ", threshold, " (dashed line)")
      ) +
      theme_tulpa() +
      ggplot2::theme(
        plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"),
        panel.grid.major.y = ggplot2::element_line(color = "gray90")
      )

    return(p)
  }

  # Base R fallback
  plot_ess_base(diag, threshold, type)
  invisible(NULL)
}


#' Base R ESS plot
#' @keywords internal
plot_ess_base <- function(diag, threshold, type) {
  n_pars <- nrow(diag)
  old_par <- graphics::par(mar = c(4, 8, 3, 1))
  on.exit(graphics::par(old_par))

  graphics::plot(
    diag$ess, seq_len(n_pars),
    xlim = c(0, max(diag$ess) * 1.1),
    ylim = c(0.5, n_pars + 0.5),
    pch = 19, col = diag$color, cex = 1.5,
    xlab = "ESS",
    ylab = "",
    yaxt = "n",
    main = paste0("Effective Sample Size (", type, ")")
  )

  graphics::axis(2, at = seq_len(n_pars), labels = diag$parameter, las = 2, cex.axis = 0.8)
  graphics::abline(v = threshold, lty = 2, col = tulpa_diag_colors$warn, lwd = 2)
}


#' Plot Autocorrelation Functions
#'
#' @description
#' Creates autocorrelation function (ACF) plots for selected parameters.
#' High autocorrelation indicates slow mixing and low effective sample size.
#'
#' @param fit A `tulpa_fit` object.
#' @param pars Character vector of parameter names. If NULL, selects worst-mixing
#'   parameters based on ESS.
#' @param lags Maximum number of lags to compute (default: 25).
#' @param n_pars Maximum number of parameters to plot (default: 6).
#'
#' @return A ggplot object (if ggplot2 available) or base R plot (invisible).
#'
#' @details
#' Ideal ACF plots show rapid decay to zero. Slow decay indicates high
#' autocorrelation, which reduces effective sample size and may indicate
#' poor mixing.
#'
#' @examples
#' # See plot_rhat() examples for fitting a model
#' # plot_acf(fit)
#' # plot_acf(fit, pars = c("beta_num[1]", "sigma_re"))
#'
#' @seealso [plot_ess()], [mcmc_diagnostics()]
#' @export
plot_acf <- function(fit, pars = NULL, lags = 25, n_pars = 6) {
  if (!inherits(fit, "tulpa_fit")) {
    stop("fit must be a tulpa_fit object", call. = FALSE)
  }

  # Get draws
  draws <- fit$draws
  if (is.null(draws)) {
    stop("No draws available in fit object", call. = FALSE)
  }

  # Select parameters
  if (is.null(pars)) {
    # Select worst ESS parameters
    diag <- mcmc_diagnostics(fit)
    main_pars <- select_main_params(diag$parameter)
    diag <- diag[diag$parameter %in% main_pars, ]
    diag <- diag[order(diag$ess_bulk), ]
    pars <- head(diag$parameter, n_pars)
  } else {
    pars <- grep_params(pars, colnames(draws))
    pars <- head(pars, n_pars)
  }

  if (length(pars) == 0) {
    message("No parameters selected for ACF plot")
    return(invisible(NULL))
  }

  # Compute ACF for each parameter
  acf_data <- lapply(pars, function(p) {
    x <- draws[, p]
    acf_obj <- stats::acf(x, lag.max = lags, plot = FALSE)
    data.frame(
      parameter = p,
      lag = acf_obj$lag[-1],
      acf = acf_obj$acf[-1],
      stringsAsFactors = FALSE
    )
  })
  acf_df <- do.call(rbind, acf_data)

  # Plot with ggplot2 if available
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    # Compute confidence bounds (approximate 95% CI)
    n_samples <- nrow(draws)
    ci_bound <- qnorm(0.975) / sqrt(n_samples)

    p <- ggplot2::ggplot(acf_df, ggplot2::aes(x = .data$lag, y = .data$acf)) +
      ggplot2::geom_hline(yintercept = 0, color = "gray50") +
      ggplot2::geom_hline(yintercept = c(-ci_bound, ci_bound),
                          linetype = "dashed", color = "blue", alpha = 0.5) +
      ggplot2::geom_segment(ggplot2::aes(xend = .data$lag, yend = 0),
                            color = tulpa_diag_colors$chain[1], linewidth = 0.8) +
      ggplot2::geom_point(color = tulpa_diag_colors$chain[1], size = 2) +
      ggplot2::facet_wrap(~ parameter, scales = "free_y") +
      ggplot2::labs(
        title = "Autocorrelation Functions",
        x = "Lag",
        y = "ACF"
      ) +
      theme_tulpa() +
      ggplot2::theme(
        plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"),
        strip.text = ggplot2::element_text(face = "bold")
      )

    return(p)
  }

  # Base R fallback
  plot_acf_base(draws, pars, lags)
  invisible(NULL)
}


#' Base R ACF plot
#' @keywords internal
plot_acf_base <- function(draws, pars, lags) {
  n_pars <- length(pars)
  n_col <- min(3, n_pars)
  n_row <- ceiling(n_pars / n_col)

  old_par <- graphics::par(mfrow = c(n_row, n_col), mar = c(4, 4, 3, 1))
  on.exit(graphics::par(old_par))

  for (p in pars) {
    x <- draws[, p]
    stats::acf(x, lag.max = lags, main = p, col = tulpa_diag_colors$chain[1])
  }
}


#' Plot Bivariate Parameter Posteriors (Pairs Plot)
#'
#' @description
#' Creates a pairs plot showing bivariate relationships between parameters.
#' Divergent transitions (if present) are highlighted to help identify
#' problematic posterior regions.
#'
#' @param fit A `tulpa_fit` object.
#' @param pars Character vector of parameter names. If NULL, selects main
#'   variance parameters.
#' @param highlight_divergent Logical; highlight divergent transitions in red
#'   (default: TRUE).
#' @param n_pars Maximum number of parameters (default: 5).
#' @param alpha Point transparency (default: 0.3).
#'
#' @return A ggplot object (if ggplot2/GGally available) or base R plot.
#'
#' @details
#' Pairs plots help identify:
#' - Strong correlations between parameters (potential non-identifiability)
#' - Multimodality
#' - Regions where divergences cluster (indicating problematic geometry)
#'
#' @examples
#' # See plot_rhat() examples for fitting a model
#' # plot_pairs(fit)
#' # plot_pairs(fit, pars = c("sigma_re", "phi_num", "phi_denom"))
#'
#' @seealso [plot_divergences()], [mcmc_diagnostics()]
#' @export
plot_pairs <- function(fit, pars = NULL, highlight_divergent = TRUE,
                       n_pars = 5, alpha = 0.3) {
  if (!inherits(fit, "tulpa_fit")) {
    stop("fit must be a tulpa_fit object", call. = FALSE)
  }

  # Get draws
  draws <- fit$draws
  if (is.null(draws)) {
    stop("No draws available in fit object", call. = FALSE)
  }

  # Select parameters
  if (is.null(pars)) {
    # Select variance parameters and key fixed effects
    all_pars <- colnames(draws)
    var_pars <- grep("^(sigma|phi|tau|rho)", all_pars, value = TRUE)
    beta_pars <- grep("^beta_(num|denom)\\[1\\]", all_pars, value = TRUE)
    pars <- c(beta_pars, var_pars)
    pars <- head(pars, n_pars)
  } else {
    pars <- grep_params(pars, colnames(draws))
    pars <- head(pars, n_pars)
  }

  if (length(pars) < 2) {
    message("Need at least 2 parameters for pairs plot")
    return(invisible(NULL))
  }

  # Subset draws
  draws_subset <- as.data.frame(draws[, pars, drop = FALSE])

  # Add divergence indicator if available
  has_divergent <- FALSE
  if (highlight_divergent && fit$backend == "hmc") {
    div_idx <- fit$diagnostics$divergent_idx
    if (!is.null(div_idx) && length(div_idx) > 0) {
      draws_subset$divergent <- FALSE
      draws_subset$divergent[div_idx] <- TRUE
      has_divergent <- any(draws_subset$divergent)
    }
  }

  # Use bayesplot for pairs if available. Fall back to the base R renderer if
  # bayesplot is absent or rejects the inputs.
  if (requireNamespace("bayesplot", quietly = TRUE)) {
    draws_array <- get_draws_array(fit)$draws
    draws_array <- draws_array[, , pars, drop = FALSE]

    np <- if (has_divergent) {
      .tulpa_divergent_np(fit, dim(draws_array)[1L], dim(draws_array)[2L])
    } else {
      NULL
    }

    p <- tryCatch(
      bayesplot::mcmc_pairs(
        draws_array,
        pars = pars,
        np = np,
        off_diag_args = list(size = 0.5, alpha = alpha)
      ),
      error = function(e) NULL
    )
    if (!is.null(p)) return(p)
  }

  # Base R pairs plot fallback
  plot_pairs_base(draws_subset, pars, has_divergent, alpha)
  invisible(NULL)
}


# Build a bayesplot `np`-style data frame flagging divergent transitions, keyed
# by (Iteration, Chain) to align with an [iter, chain, param] draws array.
# Divergence row indices in `fit$diagnostics$divergent_idx` are mapped from the
# pooled chain-major draws onto the per-chain (iteration, chain) grid.
.tulpa_divergent_np <- function(fit, n_iter, n_chain) {
  np <- data.frame(
    Iteration = rep(seq_len(n_iter), times = n_chain),
    Chain     = rep(seq_len(n_chain), each = n_iter),
    Parameter = factor("divergent__"),
    Value     = 0
  )
  div_idx <- fit$diagnostics$divergent_idx
  if (is.null(div_idx) || length(div_idx) == 0L) return(np)

  cid <- fit$chain_id
  if (!is.null(cid) && length(cid) >= max(div_idx)) {
    chain <- cid[div_idx]
    iter  <- ave(seq_along(cid), cid, FUN = seq_along)[div_idx]
  } else {
    per   <- if (is.matrix(fit$draws)) nrow(fit$draws) %/% n_chain else n_iter
    chain <- (div_idx - 1L) %/% per + 1L
    iter  <- (div_idx - 1L) %% per + 1L
  }
  hit <- chain >= 1L & chain <= n_chain & iter >= 1L & iter <= n_iter
  for (k in which(hit)) {
    np$Value[np$Chain == chain[k] & np$Iteration == iter[k]] <- 1
  }
  np
}


#' Base R pairs plot
#' @keywords internal
plot_pairs_base <- function(draws_df, pars, has_divergent, alpha) {
  # Subset to just the parameters
  plot_data <- draws_df[, pars, drop = FALSE]

  # Color vector
  if (has_divergent) {
    cols <- ifelse(draws_df$divergent,
                   grDevices::adjustcolor(tulpa_diag_colors$divergent, alpha.f = 0.8),
                   grDevices::adjustcolor("black", alpha.f = alpha))
    pch <- ifelse(draws_df$divergent, 19, 1)
  } else {
    cols <- grDevices::adjustcolor("black", alpha.f = alpha)
    pch <- 1
  }

  graphics::pairs(
    plot_data,
    pch = pch,
    col = cols,
    main = "Parameter Pairs Plot",
    upper.panel = function(x, y, ...) {
      graphics::points(x, y, ...)
      # Add correlation
      r <- cor(x, y)
      usr <- graphics::par("usr")
      graphics::text(mean(usr[1:2]), mean(usr[3:4]),
                     sprintf("r = %.2f", r), cex = 1.2, col = "gray40")
    }
  )
}


#' Plot Divergent Transitions
#'
#' @description
#' Creates visualizations to investigate divergent transitions. Parallel
#' coordinates and scatter plots highlight where in parameter space
#' divergences occur.
#'
#' @param fit A `tulpa_fit` object (HMC backend).
#' @param pars Character vector of parameter names. If NULL, uses variance
#'   parameters.
#' @param type Plot type: "parcoord" (parallel coordinates) or "scatter".
#'
#' @return A ggplot object or base R plot (invisible).
#'
#' @details
#' Divergent transitions indicate regions of high posterior curvature that

#' the sampler cannot efficiently explore. Common causes:
#' - Very narrow funnels (hierarchical models)
#' - Strong correlations
#' - Multi-modality
#'
#' @examples
#' # See plot_rhat() examples for fitting a model
#' # plot_divergences(fit)
#' # plot_divergences(fit, type = "scatter")
#'
#' @seealso [plot_pairs()], [n_divergent()]
#' @export
plot_divergences <- function(fit, pars = NULL, type = c("parcoord", "scatter")) {
  if (!inherits(fit, "tulpa_fit")) {
    stop("fit must be a tulpa_fit object", call. = FALSE)
  }

  if (fit$backend != "hmc") {
    message("Divergence plots only available for HMC backend")
    return(invisible(NULL))
  }

  type <- match.arg(type)

  # Check for divergent transitions
  n_div <- n_divergent(fit)
  if (n_div == 0) {
    message("No divergent transitions to plot")
    return(invisible(NULL))
  }

  div_idx <- fit$diagnostics$divergent_idx
  if (is.null(div_idx) || length(div_idx) == 0) {
    message("Divergent indices not available")
    return(invisible(NULL))
  }

  # Get draws
  draws <- fit$draws

  # Select parameters
  if (is.null(pars)) {
    all_pars <- colnames(draws)
    var_pars <- grep("^(sigma|phi|tau|rho)", all_pars, value = TRUE)
    pars <- head(var_pars, 6)
  } else {
    pars <- grep_params(pars, colnames(draws))
    pars <- head(pars, 6)
  }

  if (length(pars) < 2) {
    message("Need at least 2 parameters for divergence plot")
    return(invisible(NULL))
  }

  # Create data frame with divergence indicator
  draws_df <- as.data.frame(draws[, pars, drop = FALSE])
  draws_df$divergent <- FALSE
  draws_df$divergent[div_idx] <- TRUE

  if (type == "parcoord") {
    plot_divergences_parcoord(draws_df, pars)
  } else {
    plot_divergences_scatter(draws_df, pars)
  }
}


#' Parallel coordinates plot for divergences
#' @keywords internal
plot_divergences_parcoord <- function(draws_df, pars) {
  # Scale parameters to [0, 1] for comparison
  scaled <- draws_df[, pars, drop = FALSE]
  for (p in pars) {
    rng <- range(scaled[[p]])
    if (diff(rng) > 0) {
      scaled[[p]] <- (scaled[[p]] - rng[1]) / diff(rng)
    } else {
      scaled[[p]] <- 0.5
    }
  }
  scaled$divergent <- draws_df$divergent

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    # Reshape for ggplot
    scaled$id <- seq_len(nrow(scaled))
    long_data <- reshape(
      scaled,
      direction = "long",
      varying = pars,
      v.names = "value",
      timevar = "parameter",
      times = pars
    )
    long_data$parameter <- factor(long_data$parameter, levels = pars)

    # Plot non-divergent first, then divergent on top
    p <- ggplot2::ggplot(long_data, ggplot2::aes(x = .data$parameter, y = .data$value,
                                                  group = .data$id)) +
      ggplot2::geom_line(data = long_data[!long_data$divergent, ],
                         alpha = 0.05, color = "gray50") +
      ggplot2::geom_line(data = long_data[long_data$divergent, ],
                         alpha = 0.7, color = tulpa_diag_colors$divergent,
                         linewidth = 0.8) +
      ggplot2::labs(
        title = "Divergent Transitions (Parallel Coordinates)",
        subtitle = paste0(sum(draws_df$divergent), " divergent transitions (red)"),
        x = "Parameter",
        y = "Scaled Value"
      ) +
      theme_tulpa() +
      ggplot2::theme(
        plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"),
        axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
      )

    return(p)
  }

  # Base R fallback
  plot_divergences_parcoord_base(scaled, pars, draws_df$divergent)
}


#' Base R parallel coordinates
#' @keywords internal
plot_divergences_parcoord_base <- function(scaled, pars, divergent) {
  n_pars <- length(pars)
  n_samples <- nrow(scaled)

  old_par <- graphics::par(mar = c(6, 4, 4, 2))
  on.exit(graphics::par(old_par))

  graphics::plot(
    NULL, NULL,
    xlim = c(1, n_pars),
    ylim = c(0, 1),
    xlab = "", ylab = "Scaled Value",
    xaxt = "n",
    main = "Divergent Transitions (Parallel Coordinates)"
  )
  graphics::axis(1, at = seq_len(n_pars), labels = pars, las = 2, cex.axis = 0.7)

  # Draw non-divergent lines
  for (i in which(!divergent)) {
    graphics::lines(seq_len(n_pars), as.numeric(scaled[i, pars]),
                    col = grDevices::adjustcolor("gray50", alpha.f = 0.02))
  }

  # Draw divergent lines on top
  for (i in which(divergent)) {
    graphics::lines(seq_len(n_pars), as.numeric(scaled[i, pars]),
                    col = grDevices::adjustcolor(tulpa_diag_colors$divergent, alpha.f = 0.7),
                    lwd = 1.5)
  }
}


#' Scatter plot matrix for divergences
#' @keywords internal
plot_divergences_scatter <- function(draws_df, pars) {
  # Use first 2-4 parameters for scatter matrix
  pars <- head(pars, 4)

  if (requireNamespace("ggplot2", quietly = TRUE) && length(pars) >= 2) {
    # Simple 2D scatter of first two parameters
    p <- ggplot2::ggplot(draws_df, ggplot2::aes(x = .data[[pars[1]]], y = .data[[pars[2]]])) +
      ggplot2::geom_point(data = draws_df[!draws_df$divergent, ],
                          alpha = 0.1, size = 1, color = "gray50") +
      ggplot2::geom_point(data = draws_df[draws_df$divergent, ],
                          alpha = 0.8, size = 2, color = tulpa_diag_colors$divergent) +
      ggplot2::labs(
        title = "Divergent Transitions",
        subtitle = paste0(sum(draws_df$divergent), " divergent (red)"),
        x = pars[1],
        y = pars[2]
      ) +
      theme_tulpa() +
      ggplot2::theme(
        plot.title = ggplot2::element_text(hjust = 0.5, face = "bold")
      )

    return(p)
  }

  # Base R fallback
  plot_pairs_base(draws_df, pars, has_divergent = TRUE, alpha = 0.2)
}


#' Plot Energy Diagnostic (E-BFMI)
#'
#' @description
#' Creates overlaid histograms of marginal energy and energy transition,
#' along with the E-BFMI statistic. Low E-BFMI indicates poor exploration
#' of the posterior.
#'
#' @param fit A `tulpa_fit` object (HMC backend).
#'
#' @return A ggplot object (if ggplot2 available) or base R plot.
#'
#' @details
#' E-BFMI (Energy Bayesian Fraction of Missing Information) compares the
#' distribution of energy levels to energy transitions. Values below 0.3
#' indicate the sampler may not be exploring the full posterior.
#'
#' @examples
#' # See plot_rhat() examples for fitting a model
#' # plot_energy(fit)
#'
#' @seealso [diagnostic_summary()], [check_diagnostics()]
#' @export
plot_energy <- function(fit) {
  if (!inherits(fit, "tulpa_fit")) {
    stop("fit must be a tulpa_fit object", call. = FALSE)
  }

  if (fit$backend != "hmc") {
    message("Energy plots only available for HMC backend")
    return(invisible(NULL))
  }

  # Try to get energy from diagnostics
  diag <- fit$diagnostics
  energy <- diag$energy
  if (is.null(energy) || length(energy) < 10) {
    message("Energy values not available in fit object")
    return(invisible(NULL))
  }

  # Compute E-BFMI
  energy_diff <- diff(energy)
  e_bfmi <- var(energy_diff) / var(energy)

  # Create data for plotting
  n <- length(energy)
  plot_data <- data.frame(
    value = c(energy, energy_diff),
    type = rep(c("Marginal E", "E transition"), c(n, n - 1))
  )

  # Status message
  status <- if (e_bfmi < 0.3) "WARNING: Low E-BFMI" else "OK"

  if (requireNamespace("ggplot2", quietly = TRUE)) {
    p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = .data$value, fill = .data$type)) +
      ggplot2::geom_histogram(alpha = 0.6, position = "identity", bins = 50) +
      ggplot2::scale_fill_manual(values = c("Marginal E" = tulpa_diag_colors$chain[1],
                                            "E transition" = tulpa_diag_colors$chain[2])) +
      ggplot2::labs(
        title = "HMC Energy Diagnostic",
        subtitle = sprintf("E-BFMI = %.3f (%s)", e_bfmi, status),
        x = "Energy",
        y = "Count",
        fill = NULL
      ) +
      theme_tulpa() +
      ggplot2::theme(
        plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"),
        legend.position = "bottom"
      )

    return(p)
  }

  # Base R fallback
  plot_energy_base(energy, energy_diff, e_bfmi, status)
  invisible(NULL)
}


#' Base R energy plot
#' @keywords internal
plot_energy_base <- function(energy, energy_diff, e_bfmi, status) {
  old_par <- graphics::par(mar = c(4, 4, 4, 2))
  on.exit(graphics::par(old_par))

  # Compute breaks
  all_vals <- c(energy, energy_diff)
  breaks <- seq(min(all_vals), max(all_vals), length.out = 50)

  # Marginal energy histogram
  h1 <- graphics::hist(energy, breaks = breaks, plot = FALSE)
  h2 <- graphics::hist(energy_diff, breaks = breaks, plot = FALSE)

  graphics::plot(
    h1,
    col = grDevices::adjustcolor(tulpa_diag_colors$chain[1], alpha.f = 0.6),
    main = sprintf("HMC Energy Diagnostic\nE-BFMI = %.3f (%s)", e_bfmi, status),
    xlab = "Energy",
    ylim = c(0, max(c(h1$counts, h2$counts)) * 1.1)
  )
  graphics::plot(
    h2,
    col = grDevices::adjustcolor(tulpa_diag_colors$chain[2], alpha.f = 0.6),
    add = TRUE
  )
  graphics::legend(
    "topright",
    legend = c("Marginal E", "E transition"),
    fill = c(tulpa_diag_colors$chain[1], tulpa_diag_colors$chain[2]),
    bty = "n"
  )
}


#' Comprehensive Diagnostic Summary
#'
#' @description
#' Provides a comprehensive diagnostic report for a tulpa model, combining
#' convergence metrics, divergence information, and actionable recommendations.
#'
#' @param fit A `tulpa_fit` object.
#' @param quiet Logical; if TRUE, suppress printed output (default: FALSE).
#'
#' @return A list with class `tulpa_diagnostic_summary` containing:
#' \describe{
#'   \item{status}{Overall status: "PASS", "WARN", or "FAIL"}
#'   \item{n_divergent}{Number of divergent transitions}
#'   \item{divergent_pct}{Percentage of divergent transitions}
#'   \item{worst_rhat}{Data frame of parameters with worst Rhat}
#'   \item{worst_ess}{Data frame of parameters with worst ESS}
#'   \item{e_bfmi}{E-BFMI value (HMC only)}
#'   \item{recommendations}{Character vector of recommendations}
#' }
#'
#' @examples
#' # See plot_rhat() examples for fitting a model
#' # ds <- diagnostic_summary(fit)
#' # print(ds)
#'
#' @seealso [check_diagnostics()], [mcmc_diagnostics()], [plot_diagnostics()]
#' @export
diagnostic_summary <- function(fit, quiet = FALSE) {
  if (!inherits(fit, "tulpa_fit")) {
    stop("fit must be a tulpa_fit object", call. = FALSE)
  }

  backend <- fit$backend %||% "unknown"
  recommendations <- character(0)
  status <- "PASS"

  # Initialize result
  result <- list(
    backend = backend,
    n_divergent = 0,
    divergent_pct = 0,
    worst_rhat = NULL,
    worst_ess = NULL,
    e_bfmi = NA,
    recommendations = character(0),
    status = "PASS"
  )

  # Get MCMC diagnostics
  if (backend %in% c("hmc", "pg")) {
    diag <- tryCatch(mcmc_diagnostics(fit), error = function(e) NULL)

    if (!is.null(diag)) {
      # Filter to main parameters
      main_pars <- select_main_params(diag$parameter)
      diag <- diag[diag$parameter %in% main_pars, ]

      # Worst Rhat
      bad_rhat <- diag[!is.na(diag$rhat) & diag$rhat > 1.01, ]
      if (nrow(bad_rhat) > 0) {
        bad_rhat <- bad_rhat[order(bad_rhat$rhat, decreasing = TRUE), ]
        result$worst_rhat <- head(bad_rhat[, c("parameter", "rhat")], 5)
        recommendations <- c(recommendations,
          "Rhat > 1.01: Run more iterations or chains")
        status <- "WARN"
      }

      # Worst ESS
      low_ess <- diag[!is.na(diag$ess_bulk) & diag$ess_bulk < 400, ]
      if (nrow(low_ess) > 0) {
        low_ess <- low_ess[order(low_ess$ess_bulk), ]
        result$worst_ess <- head(low_ess[, c("parameter", "ess_bulk", "ess_tail")], 5)
        recommendations <- c(recommendations,
          "ESS < 400: Run more iterations or use thinning")
        status <- "WARN"
      }
    }
  }

  # Check divergences (HMC only)
  if (backend == "hmc") {
    n_div <- n_divergent(fit)
    result$n_divergent <- n_div

    # Calculate percentage
    n_samples <- nrow(fit$draws) %||% 0
    if (n_samples > 0) {
      result$divergent_pct <- 100 * n_div / n_samples
    }

    if (n_div > 0) {
      if (result$divergent_pct > 10) {
        status <- "FAIL"
        recommendations <- c(recommendations,
          "Many divergences: Check model specification or reparameterize")
      } else if (result$divergent_pct > 1) {
        status <- "WARN"
        recommendations <- c(recommendations,
          "Some divergences: Use plot_divergences() to investigate")
      } else {
        recommendations <- c(recommendations,
          "Few divergences: May be acceptable, but verify results")
      }
    }

    # E-BFMI
    energy <- fit$diagnostics$energy
    if (!is.null(energy) && length(energy) > 10) {
      energy_diff <- diff(energy)
      result$e_bfmi <- var(energy_diff) / var(energy)

      if (result$e_bfmi < 0.3) {
        status <- if (status == "FAIL") "FAIL" else "WARN"
        recommendations <- c(recommendations,
          "Low E-BFMI: Sampler may not be exploring the full posterior")
      }
    }
  }

  # Laplace-specific notes
  if (backend == "laplace") {
    recommendations <- c(recommendations,
      "Laplace provides approximate inference. Consider HMC for full uncertainty.")

    # Check convergence
    converged <- fit$laplace_result$converged %||% fit$.internal$converged
    if (!is.null(converged) && !converged) {
      status <- "WARN"
      recommendations <- c(recommendations,
        "Laplace optimization did not converge")
    }
  }

  result$recommendations <- recommendations
  result$status <- status

  class(result) <- c("tulpa_diagnostic_summary", "list")

  # Print if not quiet

if (!quiet) {
    print(result)
  }

  invisible(result)
}


#' Print method for diagnostic summary
#'
#' @param x A tulpa_diagnostic_summary object.
#' @param ... Ignored.
#'
#' @export
print.tulpa_diagnostic_summary <- function(x, ...) {
  # Status color (ANSI codes)
  status_color <- switch(
    x$status,
    "PASS" = "\033[32m",  # green
    "WARN" = "\033[33m",  # yellow
    "FAIL" = "\033[31m",  # red
    ""
  )
  reset <- "\033[0m"

  cat("\n")
  cat("=== tulpa Diagnostic Summary ===\n")
  cat("\n")
  cat("Backend:", x$backend, "\n")
  cat("Status: ", status_color, x$status, reset, "\n", sep = "")
  cat("\n")

  # Divergences
  if (x$backend == "hmc") {
    cat("Divergent transitions:", x$n_divergent)
    if (x$n_divergent > 0) {
      cat(sprintf(" (%.1f%%)", x$divergent_pct))
    }
    cat("\n")

    if (!is.na(x$e_bfmi)) {
      ebfmi_status <- if (x$e_bfmi >= 0.3) "OK" else "LOW"
      cat(sprintf("E-BFMI: %.3f (%s)\n", x$e_bfmi, ebfmi_status))
    }
    cat("\n")
  }

  # Worst Rhat
  if (!is.null(x$worst_rhat) && nrow(x$worst_rhat) > 0) {
    cat("Parameters with Rhat > 1.01:\n")
    x$worst_rhat$rhat <- round(x$worst_rhat$rhat, 3)
    print(x$worst_rhat, row.names = FALSE)
    cat("\n")
  }

  # Worst ESS
  if (!is.null(x$worst_ess) && nrow(x$worst_ess) > 0) {
    cat("Parameters with ESS < 400:\n")
    x$worst_ess$ess_bulk <- round(x$worst_ess$ess_bulk, 0)
    x$worst_ess$ess_tail <- round(x$worst_ess$ess_tail, 0)
    print(x$worst_ess, row.names = FALSE)
    cat("\n")
  }

  # Recommendations
  if (length(x$recommendations) > 0) {
    cat("Recommendations:\n")
    for (rec in x$recommendations) {
      cat("  -", rec, "\n")
    }
  }

  invisible(x)
}


#' Multi-Panel Diagnostic Figure
#'
#' @description
#' Creates a combined diagnostic figure with Rhat, ESS, trace plot, and
#' energy/ACF panels. Requires the `patchwork` package for layout.
#'
#' @param fit A `tulpa_fit` object.
#' @param pars Character vector of parameter names for trace plot. If NULL,
#'   uses the parameter with worst Rhat.
#'
#' @return A combined plot (ggplot + patchwork) or NULL if requirements not met.
#'
#' @details
#' Creates a 2x2 grid:
#' - Top left: Rhat plot
#' - Top right: ESS plot
#' - Bottom left: Trace for worst parameter
#' - Bottom right: Energy (HMC) or ACF (other)
#'
#' @examples
#' # See plot_rhat() examples for fitting a model
#' # plot_diagnostics(fit)
#'
#' @seealso [diagnostic_summary()], [plot_rhat()], [plot_ess()]
#' @export
plot_diagnostics <- function(fit, pars = NULL) {
  if (!inherits(fit, "tulpa_fit")) {
    stop("fit must be a tulpa_fit object", call. = FALSE)
  }

  # Check for required packages
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    message("plot_diagnostics() requires ggplot2. Using individual plot functions instead.")
    message("Install with: install.packages('ggplot2')")
    return(invisible(NULL))
  }

  if (!requireNamespace("patchwork", quietly = TRUE)) {
    message("plot_diagnostics() requires patchwork for layout.")
    message("Install with: install.packages('patchwork')")
    message("Using sequential plots instead...")

    # Fall back to sequential plots
    print(plot_rhat(fit))
    print(plot_ess(fit))
    return(invisible(NULL))
  }

  # Create individual plots
  p_rhat <- plot_rhat(fit) + ggplot2::ggtitle("Rhat")
  p_ess <- plot_ess(fit) + ggplot2::ggtitle("ESS (bulk)")

  # Trace plot for worst parameter
  diag <- tryCatch(mcmc_diagnostics(fit), error = function(e) NULL)
  if (!is.null(diag)) {
    main_pars <- select_main_params(diag$parameter)
    diag <- diag[diag$parameter %in% main_pars, ]
    worst_par <- diag$parameter[which.max(diag$rhat)]
  } else {
    worst_par <- colnames(fit$draws)[1]
  }

  # Create trace plot
  draws_array <- get_draws_array(fit)$draws
  if (worst_par %in% dimnames(draws_array)[[3]]) {
    trace_data <- data.frame(
      iteration = seq_len(dim(draws_array)[1]),
      value = draws_array[, 1, worst_par]
    )

    p_trace <- ggplot2::ggplot(trace_data, ggplot2::aes(x = .data$iteration, y = .data$value)) +
      ggplot2::geom_line(color = tulpa_diag_colors$chain[1], alpha = 0.7) +
      ggplot2::labs(title = paste("Trace:", worst_par), x = "Iteration", y = "Value") +
      theme_tulpa() +
      ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"))
  } else {
    p_trace <- ggplot2::ggplot() +
      ggplot2::annotate("text", x = 0.5, y = 0.5, label = "No trace available") +
      ggplot2::theme_void()
  }

  # Bottom right: Energy or ACF
  if (fit$backend == "hmc" && !is.null(fit$diagnostics$energy)) {
    p_br <- plot_energy(fit)
  } else {
    p_br <- plot_acf(fit, n_pars = 1)
  }

  # Handle NULL plots
  if (is.null(p_rhat)) p_rhat <- ggplot2::ggplot() + ggplot2::theme_void()
  if (is.null(p_ess)) p_ess <- ggplot2::ggplot() + ggplot2::theme_void()
  if (is.null(p_br)) p_br <- ggplot2::ggplot() + ggplot2::theme_void()

  # Combine with patchwork
  combined <- (p_rhat + p_ess) / (p_trace + p_br) +
    patchwork::plot_annotation(
      title = "tulpa Model Diagnostics",
      theme = ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5, face = "bold", size = 14))
    )

  combined
}


#' Geweke Convergence Test
#'
#' @description
#' Performs Geweke's convergence diagnostic, comparing the mean of the first
#' portion of a chain to the last portion. Useful for single-chain diagnostics.
#'
#' @param fit A `tulpa_fit` object.
#' @param frac1 Fraction of chain for first window (default: 0.1).
#' @param frac2 Fraction of chain for second window (default: 0.5).
#' @param pars Character vector of parameter names (default: all main params).
#'
#' @return A data frame with columns: parameter, z_score, p_value.
#'
#' @details
#' The Geweke test computes a z-score comparing the means of early and late
#' portions of a chain. Large z-scores (|z| > 2) indicate the chain has not
#' converged.
#'
#' @examples
#' # See plot_rhat() examples for fitting a model
#' # geweke_test(fit)
#'
#' @seealso [mcmc_diagnostics()], [check_diagnostics()]
#' @export
geweke_test <- function(fit, frac1 = 0.1, frac2 = 0.5, pars = NULL) {
  if (!inherits(fit, "tulpa_fit")) {
    stop("fit must be a tulpa_fit object", call. = FALSE)
  }

  draws <- fit$draws
  if (is.null(draws)) {
    stop("No draws available in fit object", call. = FALSE)
  }

  # Select parameters
  if (is.null(pars)) {
    pars <- select_main_params(colnames(draws))
  } else {
    pars <- grep_params(pars, colnames(draws))
  }

  n <- nrow(draws)
  n1 <- floor(frac1 * n)
  n2 <- floor(frac2 * n)

  results <- lapply(pars, function(p) {
    x <- draws[, p]

    # First window
    x1 <- x[1:n1]
    mean1 <- mean(x1)
    var1 <- spectrum0_ar(x1) / n1

    # Second window (last portion)
    x2 <- x[(n - n2 + 1):n]
    mean2 <- mean(x2)
    var2 <- spectrum0_ar(x2) / n2

    # Z-score
    z <- (mean1 - mean2) / sqrt(var1 + var2)
    p_val <- 2 * (1 - pnorm(abs(z)))

    data.frame(
      parameter = p,
      z_score = z,
      p_value = p_val,
      stringsAsFactors = FALSE
    )
  })

  result <- do.call(rbind, results)
  class(result) <- c("tulpa_geweke", "data.frame")
  result
}


#' Spectrum at frequency zero (AR fit)
#' @keywords internal
spectrum0_ar <- function(x) {
  # Fit AR model to estimate spectral density at frequency 0
  n <- length(x)
  if (n < 10) return(var(x))

  # Use AR(p) with AIC selection
  ar_fit <- tryCatch(
    ar(x, aic = TRUE, method = "burg"),
    error = function(e) NULL
  )

  if (is.null(ar_fit) || ar_fit$var.pred <= 0) {
    return(var(x))
  }

  # Spectral density at frequency 0
  ar_fit$var.pred / (1 - sum(ar_fit$ar))^2
}


#' Print method for Geweke test
#'
#' @param x A tulpa_geweke object.
#' @param ... Ignored.
#'
#' @export
print.tulpa_geweke <- function(x, ...) {
  cat("Geweke Convergence Diagnostic\n")
  cat("=============================\n\n")

  x$z_score <- round(x$z_score, 3)
  x$p_value <- round(x$p_value, 4)

  print.data.frame(x, row.names = FALSE)

  # Flag potential issues
  bad <- abs(x$z_score) > 2
  if (any(bad)) {
    cat("\nWarning:", sum(bad), "parameter(s) have |z| > 2 (potential non-convergence)\n")
  }

  invisible(x)
}


# Resolve a vector of parameter patterns against available draw names. Exact
# names (including bracketed-index entries like `u[3]`) match literally; any
# remaining patterns are treated as regular expressions, falling back to a
# fixed-substring match if the pattern is not valid regex.
grep_params <- function(pattern, names) {
  if (is.null(pattern)) return(names)
  out <- character(0)
  for (p in pattern) {
    exact <- names[names == p]
    if (length(exact)) {
      out <- c(out, exact)
      next
    }
    m <- tryCatch(grep(p, names, value = TRUE), error = function(e) NULL)
    if (is.null(m) || length(m) == 0L) m <- grep(p, names, value = TRUE, fixed = TRUE)
    out <- c(out, m)
  }
  unique(out)
}


#' Number of divergent transitions
#'
#' Counts divergent transitions recorded by an HMC/NUTS fit, reading whichever
#' field the backend populated (`$diagnostics$n_divergent`,
#' `$diagnostics$divergent_idx`, `$diagnostics$divergent`, or the top-level
#' `$n_divergent` / `$divergent`).
#'
#' @param fit A `tulpa_fit` object.
#' @return Integer count of divergent transitions (0 if none are recorded).
#' @seealso [plot_divergences()], [check_diagnostics()]
#' @export
n_divergent <- function(fit) {
  d <- fit$diagnostics
  if (!is.null(d$n_divergent))   return(as.integer(d$n_divergent))
  if (!is.null(d$divergent_idx)) return(length(d$divergent_idx))
  if (!is.null(d$divergent))     return(sum(as.logical(d$divergent), na.rm = TRUE))
  if (!is.null(fit$n_divergent)) return(as.integer(fit$n_divergent))
  if (!is.null(fit$divergent))   return(sum(as.logical(fit$divergent), na.rm = TRUE))
  0L
}


#' Quick convergence check
#'
#' Runs the core convergence diagnostics on a fit and reports whether Rhat,
#' bulk-ESS, and divergence thresholds are all met. A terse companion to
#' [diagnostic_summary()] for use in scripts and tests.
#'
#' @param fit A `tulpa_fit` object.
#' @param rhat_threshold Maximum acceptable Rhat (default 1.01).
#' @param ess_threshold Minimum acceptable bulk-ESS (default 400).
#' @param quiet Logical; if TRUE, suppress messages (default FALSE).
#'
#' @return Invisibly, `TRUE` if all checks pass, otherwise `FALSE`.
#'
#' @examples
#' # See plot_rhat() examples for fitting a model
#' # check_diagnostics(fit)
#'
#' @seealso [diagnostic_summary()], [mcmc_diagnostics()], [n_divergent()]
#' @export
check_diagnostics <- function(fit, rhat_threshold = 1.01, ess_threshold = 400,
                              quiet = FALSE) {
  if (!inherits(fit, "tulpa_fit")) {
    stop("fit must be a tulpa_fit object", call. = FALSE)
  }

  issues <- character(0)
  diag <- tryCatch(mcmc_diagnostics(fit), error = function(e) NULL)
  if (!is.null(diag)) {
    main_pars <- select_main_params(diag$parameter)
    diag <- diag[diag$parameter %in% main_pars, , drop = FALSE]
    n_bad_rhat <- sum(diag$rhat > rhat_threshold, na.rm = TRUE)
    if (n_bad_rhat > 0) {
      issues <- c(issues, sprintf("%d parameter(s) with Rhat > %s",
                                  n_bad_rhat, format(rhat_threshold)))
    }
    n_low_ess <- sum(diag$ess_bulk < ess_threshold, na.rm = TRUE)
    if (n_low_ess > 0) {
      issues <- c(issues, sprintf("%d parameter(s) with bulk-ESS < %d",
                                  n_low_ess, ess_threshold))
    }
  }

  n_div <- n_divergent(fit)
  if (n_div > 0) {
    issues <- c(issues, sprintf("%d divergent transition(s)", n_div))
  }

  ok <- length(issues) == 0L
  if (!quiet) {
    if (ok) {
      message("Convergence checks passed (Rhat, bulk-ESS, divergences).")
    } else {
      message("Convergence warnings:")
      for (msg in issues) message("  - ", msg)
    }
  }
  invisible(ok)
}


