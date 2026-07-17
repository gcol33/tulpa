# Reusable outer hyperparameter-grid integrator.
#
# `tulpa_hyper_grid()` is the generic counterpart of the per-family outer grids
# in `tulpa_nested_laplace()`, `tulpa_nested_laplace_joint()`, and
# `tulpa_re_cov_nested()`: build a Cartesian outer grid over user-supplied
# axes, call a user-supplied inner-fit callback at every cell, normalise the
# resulting log-marginals to weights, and assemble posterior moments + per-axis
# weighted quantiles + (optionally) the law-of-total-covariance fixed-effect
# posterior.
#
# Adaptive grid refinement and the var-of-means consistency pass live in the
# generic spec-driven helpers in `R/hyper_grid_refine.R`. The same engine is
# used by `tulpa_nested_laplace_joint()` (Step 3 of the refactor) via a
# per-driver `kernel_fn` closure -- hyper_grid wraps the user's `inner_fit`;
# the joint driver wraps `backend$call_kernel`.

# Build the kernel_fn closure refinement calls. Wraps the user's inner_fit so
# refinement can call it on a batch of new cells the same way the initial
# Cartesian pass does. `state` is a small env carrying beta_dim / beta_names
# pinned by the FIRST successful cell -- subsequent cells must match.
.hyper_grid_make_kernel_fn <- function(inner_fit, axis_names, combine, state) {
  store_beta <- combine != "none"
  store_cov  <- combine == "law_of_total_cov"

  function(new_cells, warm_start = NULL, store_extras = store_beta) {
    n_new <- nrow(new_cells)
    log_marg <- rep(-Inf, n_new)
    extras   <- if (isTRUE(store_extras)) vector("list", n_new) else NULL
    for (i in seq_len(n_new)) {
      hypers <- stats::setNames(as.numeric(new_cells[i, ]), axis_names)
      fit_i <- tryCatch(inner_fit(hypers), error = function(e) e)
      if (inherits(fit_i, "error") || !is.list(fit_i)) {
        state$n_failed <- state$n_failed + 1L
        next
      }
      lm <- fit_i$log_marginal
      if (length(lm) != 1L || !is.finite(lm)) {
        state$n_failed <- state$n_failed + 1L
        next
      }
      log_marg[i] <- as.numeric(lm)
      if (!store_beta) next

      bm <- fit_i$beta_mean
      if (is.null(bm)) {
        stop("inner_fit returned NULL `beta_mean` but `combine = '",
             combine, "'` requires it.", call. = FALSE)
      }
      bm <- as.numeric(bm)
      if (is.na(state$beta_dim)) {
        state$beta_dim <- length(bm)
        if (is.null(state$beta_names)) {
          nm <- names(fit_i$beta_mean)
          state$beta_names <- if (!is.null(nm) && all(nzchar(nm))) nm
                              else paste0("beta", seq_len(state$beta_dim))
        }
      } else if (length(bm) != state$beta_dim) {
        stop("inner_fit returned `beta_mean` of length ", length(bm),
             " disagreeing with first successful cell's length ",
             state$beta_dim, ".", call. = FALSE)
      }
      cell_extra <- list(beta_mean = bm)
      if (store_cov) {
        bc <- fit_i$beta_cov
        if (is.null(bc)) {
          stop("inner_fit returned NULL `beta_cov` but ",
               "`combine = 'law_of_total_cov'` requires it.", call. = FALSE)
        }
        bc <- as.matrix(bc)
        if (!identical(dim(bc), c(state$beta_dim, state$beta_dim))) {
          stop("inner_fit returned `beta_cov` ", nrow(bc), " x ", ncol(bc),
               " but `beta_mean` has length ", state$beta_dim, ".",
               call. = FALSE)
        }
        cell_extra$beta_cov <- bc
      }
      if (!is.null(extras)) extras[[i]] <- cell_extra
    }
    list(log_marginal = log_marg, extras = extras)
  }
}

# Per-cell joint log-prior closure -- vapplies each axis's log_prior over the
# new cells. Returns NULL when every axis has log_prior = NULL (so the caller
# can skip the prior contribution altogether).
.hyper_grid_make_hp_fn <- function(specs) {
  has_any <- any(vapply(specs, function(s) !is.null(s$log_prior), logical(1)))
  if (!has_any) return(NULL)
  function(new_cells) {
    contrib <- rep(0, nrow(new_cells))
    for (j in seq_along(specs)) {
      pj <- specs[[j]]$log_prior
      if (is.null(pj)) next
      axis_name <- specs[[j]]$name
      vals <- new_cells[, axis_name]
      cj <- vapply(vals, function(v) {
        out <- tryCatch(pj(v), error = function(e) NA_real_)
        if (length(out) != 1L || !is.finite(out)) NA_real_ else as.numeric(out)
      }, numeric(1))
      cj[is.na(cj)] <- -Inf
      contrib <- contrib + cj
    }
    contrib
  }
}

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
#' @param control Optional list of refinement / tuning knobs:
#'   * `adaptive_grid` (`FALSE`) -- run the boundary / interior refinement
#'     pass on every axis whose spec has `refinable = TRUE`. New cells are
#'     appended along the refining axis paired with the boundary modal cell's
#'     other-axis values, carrying a marginal-scale calibration so they
#'     contribute on the right scale.
#'   * `adaptive_grid_edge_thresh` (`0.02`) -- per-axis trigger threshold.
#'   * `adaptive_grid_max_passes` (`1L`) -- cap on refinement passes.
#'   * `var_of_means_consistency` (`FALSE`) -- run a post-integration
#'     consistency pass: for refinable axes whose joint-grid var-of-means
#'     undershoots the Laplace-at-mode SD by more than `tolerance`, append
#'     Laplace-guided slice points at `theta_mean +/- {0.7, 1.5} * theta_sd`
#'     pinned at the modal cell. One kernel call per axis.
#'   * `var_of_means_tolerance` (`0.7`) -- consistency-pass trigger ratio.
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
#' parabola where possible.
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
#'   * `combine`, `n_failed`, `n_grid`, `refining_axis` -- diagnostic fields.
#'   * `adaptive_grid_info`, `var_of_means_consistency_info` -- present when
#'     the corresponding refinement pass fired.
#'
#' @seealso [hyper_axis_spec()] for the axis-spec constructor;
#'   [tulpa_nested_laplace_joint()] for the family-specific outer-grid
#'   driver that this helper generalises.
#' @examples
#' \dontrun{
#' # Integrate an inner fit over a hyperparameter grid: inner_fit(theta) returns
#' # a per-cell fit and hyper_specs names the axes. tulpa_nested_laplace() is the
#' # packaged driver built on this.
#' res <- tulpa_hyper_grid(hyper_specs, inner_fit)
#' }
#' @export
tulpa_hyper_grid <- function(hyper_specs, inner_fit,
                             combine = c("law_of_total_cov",
                                         "weighted_mean_only",
                                         "none"),
                             n_draws = 2000L, seed = NULL,
                             beta_names = NULL,
                             control = list()) {
  combine <- match.arg(combine)
  if (!is.function(inner_fit)) {
    stop("`inner_fit` must be a function(hypers) -> list(log_marginal, ...).",
         call. = FALSE)
  }
  .seed_scoped(seed)

  specs <- .hyper_axis_specs_normalise(hyper_specs)
  axis_names <- vapply(specs, `[[`, character(1), "name")

  adaptive_grid             <- isTRUE(control$adaptive_grid)
  adaptive_grid_edge_thresh <- control$adaptive_grid_edge_thresh %||% 0.02
  adaptive_grid_max_passes  <- control$adaptive_grid_max_passes %||% 1L
  var_of_means_consistency  <- isTRUE(control$var_of_means_consistency)
  var_of_means_tolerance    <- control$var_of_means_tolerance %||% 0.7

  # State shared across kernel_fn invocations: pins beta dim / names on the
  # first successful cell and accumulates n_failed across passes.
  state <- new.env(parent = emptyenv())
  state$beta_dim   <- NA_integer_
  state$beta_names <- beta_names
  state$n_failed   <- 0L
  kernel_fn <- .hyper_grid_make_kernel_fn(inner_fit, axis_names, combine, state)
  hp_fn     <- .hyper_grid_make_hp_fn(specs)

  # Initial Cartesian outer grid.
  axes <- stats::setNames(lapply(specs, `[[`, "grid"), axis_names)
  gr <- do.call(expand.grid,
                c(axes, list(KEEP.OUT.ATTRS = FALSE,
                              stringsAsFactors = FALSE)))
  theta_grid <- as.matrix(gr)
  colnames(theta_grid) <- axis_names

  init <- kernel_fn(theta_grid, warm_start = NULL,
                    store_extras = combine != "none")
  log_prior_cell <- if (is.null(hp_fn)) rep(0, nrow(theta_grid))
                    else hp_fn(theta_grid)
  log_marginal   <- init$log_marginal + log_prior_cell
  extras_list    <- init$extras
  refining_axis  <- rep("", nrow(theta_grid))

  adaptive_info <- NULL
  if (adaptive_grid) {
    refined <- .hyper_adaptive_refine_pass(
      theta_grid    = theta_grid,
      log_marginal  = log_marginal,
      extras        = extras_list,
      refining_axis = refining_axis,
      specs         = specs,
      kernel_fn     = kernel_fn,
      edge_thresh   = adaptive_grid_edge_thresh,
      max_passes    = adaptive_grid_max_passes,
      hp_fn         = hp_fn
    )
    theta_grid    <- refined$theta_grid
    log_marginal  <- refined$log_marginal
    extras_list   <- refined$extras
    refining_axis <- refined$refining_axis
    adaptive_info <- refined$info
  }

  # Initial weighted moments (also needed by the consistency pass).
  weights <- .nl_normalise_weights_safe(log_marginal, what = "hyper-grid cells")
  weights_for_summary <- weights
  weights_for_summary[is.na(weights_for_summary)] <- 0

  theta_mean <- stats::setNames(rep(NA_real_, length(axis_names)), axis_names)
  theta_sd   <- stats::setNames(rep(NA_real_, length(axis_names)), axis_names)
  if (sum(weights_for_summary) > 0) {
    theta_mean[] <- as.numeric(crossprod(weights_for_summary, theta_grid))
    theta_sd[]   <- sqrt(pmax(0,
      as.numeric(crossprod(weights_for_summary, theta_grid^2)) -
        theta_mean^2))
  }
  res_partial <- list(theta_grid = theta_grid,
                      log_marginal = log_marginal,
                      theta_sd = theta_sd)
  res_partial <- .nl_refit_axis_sd_laplace(res_partial,
                                            refining = refining_axis)
  theta_sd <- res_partial$theta_sd

  consistency_info <- NULL
  if (var_of_means_consistency) {
    consistency <- .hyper_consistency_pass(
      theta_grid    = theta_grid,
      log_marginal  = log_marginal,
      extras        = extras_list,
      refining_axis = refining_axis,
      specs         = specs,
      theta_mean    = theta_mean,
      theta_sd      = theta_sd,
      kernel_fn     = kernel_fn,
      tolerance     = var_of_means_tolerance,
      hp_fn         = hp_fn,
      weights       = weights_for_summary
    )
    if (consistency$n_added > 0L) {
      theta_grid    <- consistency$theta_grid
      log_marginal  <- consistency$log_marginal
      extras_list   <- consistency$extras
      refining_axis <- consistency$refining_axis
      # Re-derive weights / moments / Laplace-SD on the merged grid.
      weights <- .nl_normalise_weights_safe(log_marginal,
                                             what = "hyper-grid cells")
      weights_for_summary <- weights
      weights_for_summary[is.na(weights_for_summary)] <- 0
      if (sum(weights_for_summary) > 0) {
        theta_mean[] <- as.numeric(crossprod(weights_for_summary, theta_grid))
        theta_sd[]   <- sqrt(pmax(0,
          as.numeric(crossprod(weights_for_summary, theta_grid^2)) -
            theta_mean^2))
      }
      res_partial$theta_grid   <- theta_grid
      res_partial$log_marginal <- log_marginal
      res_partial$theta_sd     <- theta_sd
      res_partial <- .nl_refit_axis_sd_laplace(res_partial,
                                                refining = refining_axis)
      theta_sd <- res_partial$theta_sd
    }
    consistency_info <- consistency$info
  }

  # Per-axis moment recalibration when slice cells are present (slice cells
  # for axis Y are pinned at modal non-Y values; including them in axis X's
  # marginal collapses X to a point).
  rc <- .hyper_recalibrate_axis_moments(theta_grid, log_marginal,
                                         refining_axis, theta_mean, theta_sd)
  theta_mean <- rc$theta_mean
  theta_sd   <- rc$theta_sd

  qs <- .nl_axis_quantiles(theta_grid, log_marginal, refining_axis)
  theta_median <- qs$median
  theta_ci_lo  <- qs$ci_lo
  theta_ci_hi  <- qs$ci_hi

  # Fixed-effect posterior from extras_list.
  beta_mean_out <- NULL; beta_cov_out <- NULL; draws_out <- NULL
  hyper_lp_out  <- NULL
  param_names   <- NULL
  if (combine != "none") {
    if (is.na(state$beta_dim)) {
      warning("No cell produced a finite `inner_fit` result; ",
              "fixed-effect posterior unavailable.", call. = FALSE)
    } else {
      beta_dim <- state$beta_dim
      have_beta <- !vapply(extras_list, function(e)
        is.null(e) || is.null(e$beta_mean), logical(1))
      beta_mat <- matrix(NA_real_, length(extras_list), beta_dim)
      if (any(have_beta)) {
        beta_mat[have_beta, ] <- do.call(rbind,
          lapply(extras_list[have_beta], `[[`, "beta_mean"))
      }
      w_eff <- weights_for_summary
      w_eff[!have_beta] <- 0
      w_total <- sum(w_eff)
      if (w_total > 0) {
        w_eff <- w_eff / w_total
        beta_mean_out <- as.numeric(crossprod(w_eff[have_beta],
                                              beta_mat[have_beta, ,
                                                       drop = FALSE]))
        names(beta_mean_out) <- state$beta_names
        param_names <- state$beta_names

        if (combine == "law_of_total_cov") {
          E_V <- matrix(0, beta_dim, beta_dim)
          V_E <- matrix(0, beta_dim, beta_dim)
          for (i in which(have_beta)) {
            wi <- w_eff[i]; if (wi <= 0) next
            bc <- extras_list[[i]]$beta_cov
            if (!is.null(bc)) E_V <- E_V + wi * bc
            d_i <- beta_mat[i, ] - beta_mean_out
            V_E <- V_E + wi * tcrossprod(d_i)
          }
          beta_cov_out <- E_V + V_E
          rownames(beta_cov_out) <- colnames(beta_cov_out) <- state$beta_names

          if (n_draws > 0L) {
            beta_cov_list <- lapply(extras_list, function(e)
              if (is.null(e)) NULL else e$beta_cov)
            ds <- .re_cov_nested_beta_draws(
              beta_nodes      = beta_mat,
              beta_cov_nodes  = beta_cov_list,
              w               = w_eff,
              n_draws         = as.integer(n_draws),
              beta_names      = state$beta_names
            )
            draws_out <- ds$draws
            # Per-draw hyperparameter log-prior at the drawn cell, for the
            # input power-scaling (tulpa_powerscale_sensitivity).
            if (!is.null(ds)) hyper_lp_out <- log_prior_cell[ds$picks]
          }
        }
      } else {
        warning("No cell carries both finite mass and a `beta_mean`; ",
                "fixed-effect posterior unavailable.", call. = FALSE)
      }
    }
  }

  out <- list(
    theta_grid     = theta_grid,
    theta_names    = axis_names,
    log_marginal   = log_marginal,
    log_prior      = log_prior_cell,
    weights        = weights,
    theta_mean     = theta_mean,
    theta_sd       = theta_sd,
    theta_median   = theta_median,
    theta_ci_lo    = theta_ci_lo,
    theta_ci_hi    = theta_ci_hi,
    beta           = beta_mean_out,
    beta_cov       = beta_cov_out,
    draws          = draws_out,
    hyper_log_prior_draws = hyper_lp_out,
    means          = beta_mean_out,
    param_names    = param_names,
    process_info   = if (!is.null(param_names))
      list(list(name = "fixed_effects", p = length(param_names),
                coef_names = param_names))
      else NULL,
    n_samples      = if (!is.null(draws_out)) nrow(draws_out) else 0L,
    n_params       = if (!is.null(param_names)) length(param_names) else 0L,
    N              = NA_integer_,
    hyper_specs    = specs,
    combine        = combine,
    n_failed       = state$n_failed,
    n_grid         = nrow(theta_grid),
    refining_axis  = refining_axis,
    adaptive_grid_info            = adaptive_info,
    var_of_means_consistency_info = consistency_info
  )
  class(out) <- c("tulpa_hyper_grid", "tulpa_fit", "list")
  out
}
