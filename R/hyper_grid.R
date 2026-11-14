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

  # A caller's `inner_fit` has no cheap pass, so every cell is solved and
  # `screen` is not read.
  function(new_cells, warm_start = NULL, store_extras = store_beta,
           screen = NULL) {
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
      if (!is.null(fit_i$extras)) {
        cell_extra$extras <- lapply(fit_i$extras, as.numeric)
      }
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
      # The declared coordinate is carried in one place, shared with the axis
      # quadrature and the joint driver's fold.
      cj <- .hyper_prior_carry(vals, pj, specs[[j]]$log_scale,
                               specs[[j]]$log_prior_coord %||% "integration")
      # A density on the axis describes its continuum. The zero level of an
      # axis that declares an `atom_mass` is a point mass outside that
      # continuum -- the reason its prior probability is declared rather than
      # integrated -- so the density is not read there and the declared split
      # stands (gcol33/tulpa#624, gcol33/tulpa#626).
      cj[.hyper_is_atom_level(vals, specs[[j]])] <- 0
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
#'   * `extras` -- optional named list of numeric vectors (one length per
#'     name across cells): per-cell quantities the fit reports as their
#'     posterior mean over the grid (a latent field's mode, a per-observation
#'     offset), returned in `extras_mean` under the same weights as `beta`.
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
#' @param design `"grid"` (default) integrates on the Cartesian product of the
#'   axis grids. `"ccd"` places a mode-centred central composite design
#'   ([tulpa_ccd_outer_grid()]) over the axes whose grid has more than one
#'   level (log-scale axes in the log, the grid's range as the box, the others
#'   held at their single level), weighting each node by its design weight
#'   times its marginal; when the design declines, or no node is finite, the
#'   tensor grid is used and `ccd_declined` says why. The refinement passes in
#'   `control` apply to the tensor grid only.
#' @param ccd_start Optional named physical starting values of the CCD
#'   mode-find; an axis left out starts at the centre of its grid (in its
#'   integration coordinate).
#' @param control Optional list of refinement / tuning knobs:
#'   * `adaptive_grid` (`FALSE`) -- run the boundary / interior refinement
#'     pass on every axis whose spec has `refinable = TRUE`. New cells are
#'     appended along the refining axis paired with the boundary modal cell's
#'     other-axis values. Each new cell is measured by the part of its row's
#'     base cells it takes over, so the grid integrates the same prior measure
#'     with a finer resolution where the mass is.
#'   * `adaptive_grid_edge_thresh` (`0.02`) -- per-axis trigger threshold.
#'   * `adaptive_grid_max_passes` (`1L`) -- cap on refinement passes.
#'   * `var_of_means_consistency` (`FALSE`) -- run a post-integration
#'     consistency pass: for refinable axes whose marginal has collapsed onto
#'     too few nodes to carry a spread, bisect the gaps between adjacent nodes
#'     that carry the axis's mass, each new point a level laid in every row of
#'     the other axes that holds the posterior, and repeat until the axis reaches `var_of_means_min_ess`
#'     or has taken `.nl_diag("axis_refine_nodes")` new levels inside its
#'     span. An end whose read places more than `.nl_diag("edge_tail_mass")`
#'     of the axis past it gains a level beyond it, at most
#'     `.nl_diag("edge_extend_max")` per end. One kernel call per round.
#'   * `var_of_means_min_ess` (`.nl_diag("axis_sd_ess")`) -- the quadrature
#'     effective sample size an axis marginal has to reach for the pass to
#'     leave it alone. Read off the weights, so the trigger is not one SD
#'     estimator compared against the other.
#'
#' @return A list of class `c("tulpa_fit", "list")` with:
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
#'     mean and SD per axis. Each SD comes from the estimator its axis's own
#'     resolution calls for -- the weighted spread of the axis marginal where
#'     it is resolved, the 3-point parabola at the modal node where the
#'     marginal has collapsed onto too few nodes to have a spread.
#'   * `theta_sd_source`, `theta_sd_ess`, `theta_sd_stencil_declined` --
#'     per axis, which estimator produced the SD, the quadrature effective
#'     sample size that decided it, and (where the parabola was wanted and
#'     could not be formed) why it declined.
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
#'   * `extras_mean` -- named list, the weighted mean of each `extras`
#'     quantity `inner_fit` returned (`NULL` when it returned none).
#'   * `mixture` -- `list(weights, modes, covs)` over the cells carrying mass,
#'     the per-cell fixed-effect posteriors `beta` / `beta_cov` pool (`covs`
#'     `NULL` unless `combine = "law_of_total_cov"`).
#'   * `design`, `ccd_declined`, `pareto_k` -- the integration design used,
#'     the reason a requested CCD fell back to the grid (`NULL` otherwise),
#'     and the CCD's outer PSIS k-hat (`NA` on the grid).
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
#' # Integrate a Gaussian linear model's residual SD: inner_fit(hypers) returns
#' # the per-cell log marginal and the conditional fixed-effect posterior, and
#' # the axis spec names the grid. tulpa_nested_laplace() is the packaged driver
#' # built on this.
#' set.seed(1)
#' n <- 40L; X <- cbind(1, rnorm(n)); y <- as.numeric(X %*% c(0.5, -1) + rnorm(n))
#' XtX <- crossprod(X); Xty <- crossprod(X, y)
#' inner_fit <- function(hypers) {
#'   sigma <- as.numeric(hypers["sigma"])
#'   V <- chol2inv(chol(XtX / sigma^2 + diag(1e-4, 2)))
#'   mu <- as.numeric(V %*% (Xty / sigma^2))
#'   list(log_marginal = -n * log(sigma) - 0.5 * sum((y - X %*% mu)^2) / sigma^2 +
#'          0.5 * as.numeric(determinant(V)$modulus),
#'        beta_mean = stats::setNames(mu, c("b0", "b1")), beta_cov = V)
#' }
#' specs <- list(hyper_axis_spec("sigma", grid = exp(seq(log(0.5), log(2),
#'                                                       length.out = 9)),
#'                               log_scale = TRUE, bounds = c(0, Inf)))
#' res <- tulpa_hyper_grid(specs, inner_fit, n_draws = 500L, seed = 1L)
#' res$theta_median
#' res$beta
#' @export
tulpa_hyper_grid <- function(hyper_specs, inner_fit,
                             combine = c("law_of_total_cov",
                                         "weighted_mean_only",
                                         "none"),
                             n_draws = 2000L, seed = NULL,
                             beta_names = NULL,
                             design = c("grid", "ccd"), ccd_start = NULL,
                             control = list()) {
  combine <- match.arg(combine)
  design  <- match.arg(design)
  if (!is.function(inner_fit)) {
    stop("`inner_fit` must be a function(hypers) -> list(log_marginal, ...).",
         call. = FALSE)
  }
  .seed_scoped(seed)

  specs <- .hyper_axis_specs_normalise(hyper_specs)
  axis_names <- vapply(specs, `[[`, character(1), "name")

  tulpa_check_control(control, .CONTROL_KEYS$hyper_grid, "tulpa_hyper_grid")
  .device_session <- .device_start(control)
  on.exit(cpp_device_end(.device_session), add = TRUE)
  adaptive_grid             <- isTRUE(control$adaptive_grid)
  adaptive_grid_edge_thresh <- control$adaptive_grid_edge_thresh %||% 0.02
  adaptive_grid_max_passes  <- control$adaptive_grid_max_passes %||% 1L
  var_of_means_consistency  <- isTRUE(control$var_of_means_consistency)
  var_of_means_min_ess      <- control$var_of_means_min_ess %||%
                               .nl_diag("axis_sd_ess")

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

  ccd_declined <- NULL
  if (design == "ccd") {
    ccd <- .hyper_grid_ccd(specs, axis_names, kernel_fn, hp_fn, ccd_start)
    if (isTRUE(ccd$ok)) {
      return(.hyper_grid_ccd_result(ccd, specs, axis_names, combine, state,
                                    n_draws))
    }
    ccd_declined <- ccd$reason
  }

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
    # Refinement grows theta_grid, so the initial-grid log-prior vector is stale:
    # recompute on the FINAL grid or ds$picks (into the full grid) indexes past
    # its end -> NA into hyper_log_prior_draws / the returned $log_prior. (The
    # refine pass already folds hp_fn into log_marginal for the new cells; this
    # only refreshes the standalone per-cell vector.)
    log_prior_cell <- if (is.null(hp_fn)) rep(0, nrow(theta_grid))
                      else hp_fn(theta_grid)
  }

  # Initial weighted moments.
  log_quad <- .hyper_log_quad_weights(theta_grid, specs,
                                      refining = refining_axis,
                                      log_marginal = log_marginal)
  weights <- .nl_normalise_weights_safe(
    log_marginal, what = "hyper-grid cells", log_quad = log_quad)
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

  consistency_info <- NULL
  if (var_of_means_consistency) {
    consistency <- .hyper_consistency_pass(
      theta_grid    = theta_grid,
      log_marginal  = log_marginal,
      extras        = extras_list,
      refining_axis = refining_axis,
      specs         = specs,
      kernel_fn     = kernel_fn,
      min_ess       = var_of_means_min_ess,
      hp_fn         = hp_fn
    )
    if (consistency$n_added > 0L) {
      theta_grid    <- consistency$theta_grid
      log_marginal  <- consistency$log_marginal
      extras_list   <- consistency$extras
      refining_axis <- consistency$refining_axis
      # The consistency pass appends cells, so the per-cell log-prior
      # vector is stale; recompute on the merged grid or ds$picks indexes past
      # its end -> NA into the returned $log_prior (as the refine pass does).
      log_prior_cell <- if (is.null(hp_fn)) rep(0, nrow(theta_grid))
                        else hp_fn(theta_grid)
      # Re-derive weights / moments on the merged grid.
      log_quad <- .hyper_log_quad_weights(theta_grid, specs,
                                          refining = refining_axis,
                                          log_marginal = log_marginal)
      weights <- .nl_normalise_weights_safe(
        log_marginal, what = "hyper-grid cells", log_quad = log_quad)
      weights_for_summary <- weights
      weights_for_summary[is.na(weights_for_summary)] <- 0
      if (sum(weights_for_summary) > 0) {
        theta_mean[] <- as.numeric(crossprod(weights_for_summary, theta_grid))
        theta_sd[]   <- sqrt(pmax(0,
          as.numeric(crossprod(weights_for_summary, theta_grid^2)) -
            theta_mean^2))
      }
    }
    consistency_info <- consistency$info
  }

  # The reported per-axis SD, each from the estimator its own resolution calls
  # for. Last, so it reads the final grid and the final measure.
  res_partial <- .nl_attach_axis_sd(
    list(theta_grid = theta_grid, log_marginal = log_marginal,
         log_quad = log_quad, theta_sd = theta_sd))
  theta_sd <- res_partial$theta_sd

  # Each axis's support as the caller DECLARED it, so the outer cell edges the
  # interval extends to land inside it. An axis with no
  # `bounds` (and not `log_scale`) reports NA and keeps the coordinate the edge
  # rule guesses from the values.
  qs <- .nl_axis_quantiles(
    theta_grid, log_marginal, refining_axis, log_quad = log_quad,
    domains = vapply(specs, function(s)
      .nl_domain_of_bounds(s$bounds, s$log_scale), character(1)))
  theta_median <- qs$median
  theta_ci_lo  <- qs$ci_lo
  theta_ci_hi  <- qs$ci_hi

  pool <- .hyper_grid_pool(extras_list, weights_for_summary, combine, state,
                           n_draws)
  beta_mean_out <- pool$beta; beta_cov_out <- pool$beta_cov
  draws_out <- pool$draws; param_names <- pool$param_names
  hyper_lp_out <- if (is.null(pool$picks)) NULL else log_prior_cell[pool$picks]

  # The caller's specs declare each axis's prior: a density folded through
  # `log_prior`, or a uniform on a declared `slab_bounds`. An axis carrying
  # neither has no proper prior to read an evidence under.
  hg_declared <- unlist(lapply(specs, function(s) {
    if (!is.null(s$log_prior) || !is.null(s$slab_bounds)) s$name
  }))
  hg_evidence <- .nl_attach_evidence(
    list(log_marginal = log_marginal, log_hyperprior_axes = hg_declared,
         refining_axis = refining_axis),
    theta_grid, specs)

  out <- list(
    theta_grid     = theta_grid,
    theta_names    = axis_names,
    log_marginal   = log_marginal,
    log_prior      = log_prior_cell,
    weights        = weights,
    log_evidence   = hg_evidence$log_evidence,
    log_evidence_declined = hg_evidence$log_evidence_declined,
    log_evidence_declined_axes = hg_evidence$log_evidence_declined_axes,
    theta_mean     = theta_mean,
    theta_sd       = theta_sd,
    theta_sd_source = res_partial$theta_sd_source,
    theta_sd_ess    = res_partial$theta_sd_ess,
    theta_sd_stencil_declined = res_partial$theta_sd_stencil_declined,
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
    extras_mean    = pool$extras_mean,
    mixture        = pool$mixture,
    design         = "grid",
    ccd_declined   = ccd_declined,
    pareto_k       = NA_real_,
    hyper_specs    = specs,
    combine        = combine,
    n_failed       = state$n_failed,
    n_grid         = nrow(theta_grid),
    refining_axis  = refining_axis,
    adaptive_grid_info            = adaptive_info,
    var_of_means_consistency_info = consistency_info
  )
  out <- .device_attach(out)
  class(out) <- c("tulpa_fit", "list")
  .hyper_attach_moments(out)
}


# Pool the per-cell fixed-effect posteriors and extras under the normalised
# cell weights `w` (NA-free): the weighted mean, the law of total covariance
# (E[V] + V[E]) and mixture draws per `combine`, each `extras` quantity's
# weighted mean, and the mixture itself. A cell without a `beta_mean` carries
# no weight here.
.hyper_grid_pool <- function(extras_list, w, combine, state, n_draws) {
  out <- list(beta = NULL, beta_cov = NULL, draws = NULL, picks = NULL,
              param_names = NULL, extras_mean = NULL, mixture = NULL)
  if (combine == "none") return(out)
  if (is.na(state$beta_dim)) {
    warning("No cell produced a finite `inner_fit` result; ",
            "fixed-effect posterior unavailable.", call. = FALSE)
    return(out)
  }
  beta_dim <- state$beta_dim
  have_beta <- !vapply(extras_list, function(e)
    is.null(e) || is.null(e$beta_mean), logical(1))
  beta_mat <- matrix(NA_real_, length(extras_list), beta_dim)
  if (any(have_beta)) {
    beta_mat[have_beta, ] <- do.call(rbind,
      lapply(extras_list[have_beta], `[[`, "beta_mean"))
  }
  w_eff <- w
  w_eff[!have_beta] <- 0
  w_total <- sum(w_eff)
  if (!(w_total > 0)) {
    warning("No cell carries both finite mass and a `beta_mean`; ",
            "fixed-effect posterior unavailable.", call. = FALSE)
    return(out)
  }
  w_eff <- w_eff / w_total
  beta_mean <- as.numeric(crossprod(w_eff[have_beta],
                                    beta_mat[have_beta, , drop = FALSE]))
  names(beta_mean) <- state$beta_names
  out$beta <- beta_mean
  out$param_names <- state$beta_names
  live <- which(have_beta & w_eff > 0)
  covs <- if (combine == "law_of_total_cov")
    lapply(extras_list[live], `[[`, "beta_cov") else NULL
  out$mixture <- list(weights = w_eff[live],
                      modes = beta_mat[live, , drop = FALSE], covs = covs)
  ex_names <- unique(unlist(lapply(extras_list[live], function(e) names(e$extras))))
  if (length(ex_names)) {
    out$extras_mean <- stats::setNames(lapply(ex_names, function(nm) {
      vals <- lapply(extras_list[live], function(e) e$extras[[nm]])
      as.numeric(crossprod(w_eff[live], do.call(rbind, vals)))
    }), ex_names)
  }
  if (combine == "law_of_total_cov") {
    E_V <- matrix(0, beta_dim, beta_dim)
    V_E <- matrix(0, beta_dim, beta_dim)
    for (i in which(have_beta)) {
      wi <- w_eff[i]; if (wi <= 0) next
      bc <- extras_list[[i]]$beta_cov
      if (!is.null(bc)) E_V <- E_V + wi * bc
      d_i <- beta_mat[i, ] - beta_mean
      V_E <- V_E + wi * tcrossprod(d_i)
    }
    out$beta_cov <- E_V + V_E
    rownames(out$beta_cov) <- colnames(out$beta_cov) <- state$beta_names
    if (n_draws > 0L) {
      ds <- .re_cov_nested_beta_draws(
        beta_nodes     = beta_mat,
        beta_cov_nodes = lapply(extras_list, function(e)
          if (is.null(e)) NULL else e$beta_cov),
        w              = w_eff,
        n_draws        = as.integer(n_draws),
        beta_names     = state$beta_names)
      out$draws <- ds$draws
      if (!is.null(ds)) out$picks <- ds$picks
    }
  }
  out
}

# The mode-centred CCD over the multi-level axes (tulpa_ccd_outer_grid()), the
# single-level axes held at their value. The outer integrand is the cell's log
# marginal plus its log prior, as on the grid.
.hyper_grid_ccd <- function(specs, axis_names, kernel_fn, hp_fn, ccd_start) {
  levels <- lapply(specs, function(sp) sort(unique(as.numeric(sp$grid))))
  vary <- which(lengths(levels) > 1L)
  fixed <- vapply(levels, `[`, 0.0, 1L)
  axes <- lapply(vary, function(j) {
    lv <- levels[[j]]; ls <- isTRUE(specs[[j]]$log_scale) && min(lv) > 0
    st <- ccd_start[[axis_names[j]]]
    if (is.null(st)) st <- if (ls) exp(mean(log(range(lv)))) else mean(range(lv))
    list(name = axis_names[j], log_scale = ls, lower = min(lv),
         upper = max(lv), start = as.numeric(st))
  })
  full <- function(theta_vary) { th <- fixed; th[vary] <- theta_vary; th }
  eval_logm <- function(theta_vary) {
    row <- matrix(full(theta_vary), 1L, dimnames = list(NULL, axis_names))
    lm <- kernel_fn(row, store_extras = FALSE)$log_marginal
    if (!is.null(hp_fn)) lm <- lm + hp_fn(row)
    lm
  }
  cc <- tulpa_ccd_outer_grid(eval_logm, axes)
  if (!isTRUE(cc$ok)) return(cc)
  grid <- t(apply(cc$nodes, 1L, full))
  colnames(grid) <- axis_names
  ev <- kernel_fn(grid, store_extras = TRUE)
  lm <- ev$log_marginal
  if (!is.null(hp_fn)) lm <- lm + hp_fn(grid)
  if (!any(is.finite(lm))) return(list(ok = FALSE, reason = "no finite CCD node"))
  logw <- log(cc$dnode) + lm
  w <- exp(logw - max(logw[is.finite(logw)]))
  w[!is.finite(w)] <- 0
  w <- w / sum(w)
  list(ok = TRUE, grid = grid, log_marginal = lm, weights = w,
       extras = ev$extras, pareto_k = cc$pareto_k, kind = cc$kind)
}

# The result of a CCD integration in tulpa_hyper_grid()'s shape: the
# hyperparameter moments read off the node weights (the CCD carries no tensor
# quadrature, so no quantiles), and the fixed-effect / extras pooling the grid
# path uses.
.hyper_grid_ccd_result <- function(ccd, specs, axis_names, combine, state,
                                   n_draws) {
  w <- ccd$weights
  theta_mean <- stats::setNames(as.numeric(crossprod(w, ccd$grid)), axis_names)
  theta_sd <- stats::setNames(sqrt(pmax(0, as.numeric(crossprod(w, ccd$grid^2)) -
                                          theta_mean^2)), axis_names)
  pool <- .hyper_grid_pool(ccd$extras, w, combine, state, n_draws)
  na <- stats::setNames(rep(NA_real_, length(axis_names)), axis_names)
  out <- list(
    theta_grid = ccd$grid, theta_names = axis_names,
    log_marginal = ccd$log_marginal, weights = w,
    theta_mean = theta_mean, theta_sd = theta_sd,
    theta_median = na, theta_ci_lo = na, theta_ci_hi = na,
    beta = pool$beta, beta_cov = pool$beta_cov, draws = pool$draws,
    means = pool$beta, param_names = pool$param_names,
    process_info = if (!is.null(pool$param_names))
      list(list(name = "fixed_effects", p = length(pool$param_names),
                coef_names = pool$param_names)) else NULL,
    n_samples = if (!is.null(pool$draws)) nrow(pool$draws) else 0L,
    n_params = length(pool$param_names), N = NA_integer_,
    extras_mean = pool$extras_mean, mixture = pool$mixture,
    design = "ccd", ccd_declined = NULL, pareto_k = ccd$pareto_k,
    ccd_kind = ccd$kind, hyper_specs = specs, combine = combine,
    n_failed = state$n_failed, n_grid = nrow(ccd$grid))
  out <- .device_attach(out)
  class(out) <- c("tulpa_fit", "list")
  out
}
