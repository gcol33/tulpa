# Reusable outer hyperparameter-grid integrator (gcol33/tulpa#33).
#
# `tulpa_hyper_grid()` is the generic counterpart of the per-family outer grids
# in `tulpa_nested_laplace()`, `tulpa_nested_laplace_joint()`, and
# `tulpa_re_cov_nested()`: build a Cartesian outer grid over user-supplied
# axes, call a user-supplied inner-fit callback at every cell, normalise the
# resulting log-marginals to weights, and assemble posterior moments + per-axis
# weighted quantiles + (optionally) the law-of-total-covariance fixed-effect
# posterior.
#
# Design notes:
#   * Axis metadata (log_scale, bounds, refinable) lives in the spec, not in
#     hardcoded by-name lookups. This is what `tulpa_nested_laplace_joint()`'s
#     legacy refinement helpers will switch to in Step 3 of the #33 refactor.
#   * The inner-fit callback owns the per-cell math entirely: it can be a
#     `tulpa_laplace()` call, a custom Newton, an inla-style nested marginal,
#     etc. The driver only needs `log_marginal` per cell and optionally a
#     fixed-effect (mode, marginal-covariance) pair for the law-of-total-cov
#     pooling and the mixture-draw synthesis.
#   * Summary code is shared with the existing nested-Laplace family via
#     `.nl_normalise_weights_safe()`, `.nl_axis_quantiles()`,
#     `.nl_refit_axis_sd_laplace()` from `R/nested_laplace.R`.

#' Outer hyperparameter-grid integration with a user-supplied inner fit
#'
#' @description
#' Generic driver for nested-Laplace-style outer integration over a small
#' hyperparameter block. The user supplies per-axis specs (values + optional
#' log-prior + log-scale / bounds / refinable metadata) and an `inner_fit`
#' callback that, at every hyperparameter cell, returns the inner log marginal
#' and optionally the fixed-effect posterior mode + marginal covariance.
#' The driver builds the Cartesian outer grid, normalises the log-marginals to
#' integration weights, reports per-axis posterior moments and weighted
#' quantiles, and (when the inner fit supplies them) law-of-total-covariance
#' fixed-effect posterior + mixture draws.
#'
#' This factors the per-family outer-grid plumbing in `tulpa_nested_laplace()`
#' / `tulpa_nested_laplace_joint()` / `tulpa_re_cov_nested()` into one
#' callback-driven entry point: downstream consumers (occupancy / N-mixture /
#' cover hurdle families in tulpaObs, custom user families) drop in their own
#' per-cell inner fit and get the outer integration for free.
#'
#' @param hyper_specs A list of axis specs. Each spec is either a
#'   `hyper_axis_spec()` object or a plain list with the same fields (the
#'   driver auto-wraps); see [hyper_axis_spec()]. The outer grid is the
#'   Cartesian product of the per-axis grids; the joint log-prior is the sum
#'   of per-axis `log_prior` contributions (axes with `log_prior = NULL` are
#'   flat / improper).
#' @param inner_fit `function(hypers)` returning a list with:
#'   * `log_marginal` -- scalar; the inner-solve log marginal at this cell.
#'     Non-finite values are mapped to `-Inf` (the cell gets zero weight).
#'   * `beta_mean` -- numeric vector of fixed-effect posterior mean at the
#'     cell. Required when `combine != "none"`.
#'   * `beta_cov` -- `p x p` numeric matrix of fixed-effect marginal
#'     covariance at the cell. Required when `combine = "law_of_total_cov"`.
#'   `hypers` is a named numeric vector with the current cell's axis values
#'   (names match `vapply(hyper_specs, `[[`, character(1), "name")`).
#'   Errors thrown by `inner_fit` are caught and treated as a failed cell
#'   (`log_marginal = -Inf`, no beta contribution).
#' @param combine How to pool per-cell fixed-effect posteriors into a single
#'   posterior over the betas. One of:
#'   * `"law_of_total_cov"` (default) -- compute
#'     `E[Cov(beta | theta)] + Cov(E[beta | theta])` from the per-cell
#'     `(beta_mean, beta_cov)`; synthesise `n_draws` posterior draws by
#'     mixture sampling. Requires `beta_mean` and `beta_cov` per cell.
#'   * `"weighted_mean_only"` -- pool only the per-cell `beta_mean` into the
#'     weighted posterior mean. `beta_cov` is ignored; no draws.
#'   * `"none"` -- do not assemble a fixed-effect posterior. Only the
#'     hyperparameter posterior is returned. `beta_mean` / `beta_cov` from
#'     `inner_fit` are ignored if supplied.
#' @param n_draws Number of posterior draws of the fixed effects to
#'   synthesise from the cell mixture (default 2000). Used only when
#'   `combine = "law_of_total_cov"`. `0` disables draw synthesis (the
#'   law-of-total-cov mean and covariance are still returned).
#' @param seed Optional integer seed for the draw synthesis.
#' @param beta_names Optional character vector naming the fixed-effect
#'   coordinates. When `NULL` (default) the names are taken from the first
#'   successful cell's `beta_mean` (or `beta1`, `beta2`, ... if it is
#'   unnamed).
#'
#' @return A list of class `c("tulpa_hyper_grid", "tulpa_fit", "list")` with:
#'   * `theta_grid` -- numeric matrix `[n_cells x n_axes]` of outer-grid
#'     hyperparameter values; columns named after the axes.
#'   * `theta_names` -- character vector of axis names.
#'   * `log_marginal` -- numeric `[n_cells]`; per-cell log integrand
#'     `inner$log_marginal + log_prior(theta_cell)`. Non-finite / failed cells
#'     are `-Inf`.
#'   * `log_prior` -- numeric `[n_cells]`; per-cell log-prior contribution
#'     (the sum across axes of `axis$log_prior(theta_cell[axis])`). `0` when
#'     all axes have `log_prior = NULL`.
#'   * `weights` -- numeric `[n_cells]` summing to 1 (or all `NA` with a
#'     warning when no cell carries finite mass).
#'   * `theta_mean`, `theta_sd` -- named numeric vectors; weighted posterior
#'     mean and SD per axis. SDs refit via the 3-point Laplace-at-mode
#'     parabola where possible (gcol33/tulpa#20).
#'   * `theta_median`, `theta_ci_lo`, `theta_ci_hi` -- named numeric vectors;
#'     weighted-quantile median and 2.5 / 97.5\% empirical CI per axis
#'     (the recommended summary for right-skewed scale-like axes).
#'   * `beta`, `beta_cov`, `draws` -- fixed-effect posterior, present per
#'     `combine`:
#'     - `combine = "law_of_total_cov"`: weighted mean, total covariance
#'       (`E[V] + V[E]`), `[n_draws x p]` mixture draws.
#'     - `combine = "weighted_mean_only"`: weighted mean only; `beta_cov`
#'       and `draws` are `NULL`.
#'     - `combine = "none"`: all three are `NULL`.
#'   * `means`, `param_names`, `process_info`, `n_samples`, `n_params`,
#'     `N` -- the `tulpa_fit` accessor surface, populated when a
#'     fixed-effect posterior is assembled.
#'   * `hyper_specs` -- echoed list of `hyper_axis_spec` objects (the
#'     normalised form).
#'   * `combine`, `n_failed`, `n_grid` -- diagnostic fields.
#'
#' @seealso [hyper_axis_spec()] for the axis-spec constructor;
#'   [tulpa_nested_laplace_joint()] for the family-specific outer-grid
#'   driver that this helper generalises.
#' @export
tulpa_hyper_grid <- function(hyper_specs, inner_fit,
                             combine = c("law_of_total_cov",
                                         "weighted_mean_only",
                                         "none"),
                             n_draws = 2000L, seed = NULL,
                             beta_names = NULL) {
  combine <- match.arg(combine)
  if (!is.function(inner_fit)) {
    stop("`inner_fit` must be a function(hypers) -> list(log_marginal, ...).",
         call. = FALSE)
  }
  if (!is.null(seed)) set.seed(as.integer(seed))

  specs <- .hyper_axis_specs_normalise(hyper_specs)
  axis_names <- vapply(specs, `[[`, character(1), "name")
  n_ax <- length(specs)

  # Cartesian outer grid. expand.grid varies the first axis fastest, which
  # matches `.nl_axis_quantiles` and the joint driver's grid layout.
  axes <- stats::setNames(lapply(specs, `[[`, "grid"), axis_names)
  gr <- do.call(expand.grid,
                c(axes, list(KEEP.OUT.ATTRS = FALSE,
                              stringsAsFactors = FALSE)))
  theta_grid <- as.matrix(gr)
  colnames(theta_grid) <- axis_names
  n_cells <- nrow(theta_grid)

  # Per-cell joint log-prior: sum of per-axis log_prior contributions. Axes
  # with `log_prior = NULL` contribute 0.
  log_prior_cell <- rep(0, n_cells)
  for (j in seq_len(n_ax)) {
    pj <- specs[[j]]$log_prior
    if (is.null(pj)) next
    contrib <- vapply(theta_grid[, j], function(v) {
      out <- tryCatch(pj(v), error = function(e) NA_real_)
      if (length(out) != 1L || !is.finite(out)) NA_real_ else as.numeric(out)
    }, numeric(1))
    if (any(is.na(contrib))) {
      warning(sprintf("Axis '%s': `log_prior` returned non-finite values at %d ",
                      axis_names[j], sum(is.na(contrib))),
              "cell(s); those cells will be dropped (weight 0).",
              call. = FALSE)
    }
    log_prior_cell <- log_prior_cell + contrib
  }

  # Per-cell inner fit. We allocate the outputs once we know the beta
  # dimension from the first successful cell, so the caller is free to vary
  # the contract -- only the FIRST successful cell pins p.
  log_marg_inner <- rep(-Inf, n_cells)
  beta_dim <- NA_integer_
  beta_means <- vector("list", n_cells)
  beta_covs  <- vector("list", n_cells)
  beta_names_from_fit <- beta_names
  n_failed <- 0L

  for (i in seq_len(n_cells)) {
    hypers <- stats::setNames(as.numeric(theta_grid[i, ]), axis_names)
    fit_i <- tryCatch(inner_fit(hypers), error = function(e) e)
    if (inherits(fit_i, "error") || !is.list(fit_i)) {
      n_failed <- n_failed + 1L
      next
    }
    lm <- fit_i$log_marginal
    if (length(lm) != 1L || !is.finite(lm)) {
      n_failed <- n_failed + 1L
      next
    }
    log_marg_inner[i] <- as.numeric(lm)
    if (combine == "none") next

    bm <- fit_i$beta_mean
    if (is.null(bm)) {
      stop(sprintf("Cell %d: `inner_fit` returned NULL `beta_mean` but ", i),
           sprintf("`combine = '%s'` requires it.", combine), call. = FALSE)
    }
    bm <- as.numeric(bm)
    if (is.na(beta_dim)) {
      beta_dim <- length(bm)
      if (is.null(beta_names_from_fit)) {
        nm <- names(fit_i$beta_mean)
        beta_names_from_fit <- if (!is.null(nm) && all(nzchar(nm))) nm
                                else paste0("beta", seq_len(beta_dim))
      }
    } else if (length(bm) != beta_dim) {
      stop(sprintf("Cell %d: `beta_mean` length %d disagrees with first ", i,
                   length(bm)),
           sprintf("successful cell's length %d.", beta_dim),
           call. = FALSE)
    }
    beta_means[[i]] <- bm

    if (combine == "law_of_total_cov") {
      bc <- fit_i$beta_cov
      if (is.null(bc)) {
        stop(sprintf("Cell %d: `inner_fit` returned NULL `beta_cov` but ", i),
             "`combine = 'law_of_total_cov'` requires it.", call. = FALSE)
      }
      bc <- as.matrix(bc)
      if (!identical(dim(bc), c(beta_dim, beta_dim))) {
        stop(sprintf("Cell %d: `beta_cov` is %d x %d but `beta_mean` has ", i,
                     nrow(bc), ncol(bc)),
             sprintf("length %d.", beta_dim), call. = FALSE)
      }
      beta_covs[[i]] <- bc
    }
  }

  # Bake the prior into log_marginal once, matching the joint driver
  # convention (`res$log_marginal` is the integrand log-density that the
  # softmax + per-axis quantile helpers consume directly).
  log_marginal <- log_marg_inner + log_prior_cell

  # Normalise weights (finite-safe). All-NA when no cell carries mass.
  weights <- .nl_normalise_weights_safe(log_marginal, what = "hyper-grid cells")
  weights_for_summary <- weights
  weights_for_summary[is.na(weights_for_summary)] <- 0

  # Hyperparameter posterior moments. Mean / SD via weighted-by-cell moments;
  # median / CI via the same weighted-quantile machinery the nested-Laplace
  # family uses.
  theta_mean <- stats::setNames(rep(NA_real_, n_ax), axis_names)
  theta_sd   <- stats::setNames(rep(NA_real_, n_ax), axis_names)
  if (sum(weights_for_summary) > 0) {
    theta_mean[] <- as.numeric(crossprod(weights_for_summary, theta_grid))
    theta_sd[]   <- sqrt(pmax(0,
      as.numeric(crossprod(weights_for_summary, theta_grid^2)) -
        theta_mean^2))
  }
  qs <- .nl_axis_quantiles(theta_grid, log_marginal)
  theta_median <- qs$median
  theta_ci_lo  <- qs$ci_lo
  theta_ci_hi  <- qs$ci_hi

  # Laplace-at-mode SD refit on each axis where the 3-point parabolic fit
  # succeeds. The helper reads `theta_grid` / `log_marginal` / `theta_sd`.
  res_partial <- list(theta_grid = theta_grid,
                      log_marginal = log_marginal,
                      theta_sd = theta_sd)
  res_partial <- .nl_refit_axis_sd_laplace(res_partial)
  theta_sd <- res_partial$theta_sd

  # Fixed-effect posterior assembly.
  beta_mean_out <- NULL
  beta_cov_out  <- NULL
  draws_out     <- NULL
  param_names   <- NULL
  N_obs         <- NA_integer_

  if (combine != "none") {
    if (is.na(beta_dim)) {
      warning("No cell produced a finite `inner_fit` result; ",
              "fixed-effect posterior unavailable.", call. = FALSE)
    } else {
      beta_mat <- matrix(NA_real_, n_cells, beta_dim)
      have_beta <- !vapply(beta_means, is.null, logical(1))
      if (any(have_beta)) {
        beta_mat[have_beta, ] <- do.call(rbind, beta_means[have_beta])
      }
      w_eff <- weights_for_summary
      w_eff[!have_beta] <- 0
      w_total <- sum(w_eff)
      if (w_total > 0) {
        w_eff <- w_eff / w_total
        # NA rows (cells without a beta_mean) carry w_eff = 0, but their
        # NA values still poison the crossprod -- pool on the have_beta
        # subset explicitly.
        beta_mean_out <- as.numeric(crossprod(w_eff[have_beta],
                                              beta_mat[have_beta, ,
                                                       drop = FALSE]))
        names(beta_mean_out) <- beta_names_from_fit
        param_names <- beta_names_from_fit
      } else {
        warning("No cell carries both finite mass and a `beta_mean`; ",
                "fixed-effect posterior unavailable.", call. = FALSE)
      }

      if (combine == "law_of_total_cov" && !is.null(beta_mean_out)) {
        # Law of total covariance: Cov(beta) = E[Cov(beta|theta)]
        #                                  + Cov(E[beta|theta]).
        E_V <- matrix(0, beta_dim, beta_dim)
        V_E <- matrix(0, beta_dim, beta_dim)
        for (i in which(have_beta)) {
          w_i <- w_eff[i]
          if (w_i <= 0) next
          if (!is.null(beta_covs[[i]])) {
            E_V <- E_V + w_i * beta_covs[[i]]
          }
          d_i <- beta_mat[i, ] - beta_mean_out
          V_E <- V_E + w_i * tcrossprod(d_i)
        }
        beta_cov_out <- E_V + V_E
        rownames(beta_cov_out) <- colnames(beta_cov_out) <- beta_names_from_fit

        if (n_draws > 0L) {
          draws_out <- .re_cov_nested_beta_draws(
            beta_nodes      = beta_mat,
            beta_cov_nodes  = beta_covs,
            w               = w_eff,
            n_draws         = as.integer(n_draws),
            beta_names      = beta_names_from_fit
          )
        }
      }
    }
  }

  out <- list(
    theta_grid   = theta_grid,
    theta_names  = axis_names,
    log_marginal = log_marginal,
    log_prior    = log_prior_cell,
    weights      = weights,
    theta_mean   = theta_mean,
    theta_sd     = theta_sd,
    theta_median = theta_median,
    theta_ci_lo  = theta_ci_lo,
    theta_ci_hi  = theta_ci_hi,
    beta         = beta_mean_out,
    beta_cov     = beta_cov_out,
    draws        = draws_out,
    means        = beta_mean_out,
    param_names  = param_names,
    process_info = if (!is.null(param_names))
      list(list(name = "fixed_effects", p = length(param_names),
                coef_names = param_names))
      else NULL,
    n_samples    = if (!is.null(draws_out)) nrow(draws_out) else 0L,
    n_params     = if (!is.null(param_names)) length(param_names) else 0L,
    N            = NA_integer_,
    hyper_specs  = specs,
    combine      = combine,
    n_failed     = n_failed,
    n_grid       = n_cells
  )
  class(out) <- c("tulpa_hyper_grid", "tulpa_fit", "list")
  out
}
