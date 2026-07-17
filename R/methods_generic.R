# ============================================================================
# Generic S3 methods for tulpa_fit objects
#
# A tulpa_fit arrives in one of two posterior shapes:
#   * Sampler tier (mala/pathfinder/gibbs/nuts): a `$draws` matrix, columns in
#     [fixed, random] order. Summaries are empirical (column mean/sd/quantiles).
#   * Laplace tier: a `$mode` vector (fixed effects then random-effect values)
#     and `$H_beta`, the fixed-effect posterior precision. Fixed-effect
#     summaries are the Gaussian approximation -- mean = mode, sd =
#     sqrt(diag(H_beta^-1)), quantiles from qnorm. No Monte Carlo draws.
#
# tulpa() attaches `$n_fixed`, `$fixed_names`, `$param_names`, and `$re_layout`
# (see .tulpa_param_layout) so both shapes report meaningful names and a
# fixed/random split. The accessors below read that layout; model packages that
# define their own *.<class> methods are unaffected.
# ============================================================================

# Canonical parameter layout from a model-data bundle. Returns the fixed-effect
# count and names, the full [fixed, random] name vector (random part in the
# group-major, coef-within-group order the Laplace mode and the sampler draws
# both use), and a per-term random-effect layout for ranef().
#' @keywords internal
.tulpa_param_layout <- function(bundle) {
  n_fixed     <- bundle$n_fixed %||% ncol(bundle$X) %||% 0L
  fixed_names <- bundle$fixed_names %||% colnames(bundle$X) %||%
    paste0("beta", seq_len(n_fixed))

  re_terms  <- bundle$re_terms %||% list()
  re_names  <- character(0)
  re_layout <- vector("list", length(re_terms))

  for (k in seq_along(re_terms)) {
    rt <- re_terms[[k]]
    coef_labels <- c(if (isTRUE(rt$has_intercept)) "(Intercept)",
                     rt$slope_names %||% character(0))
    if (length(coef_labels) == 0L) coef_labels <- "(Intercept)"
    gv   <- rt$group_var %||% paste0("g", k)
    levs <- rt$levels %||% as.character(seq_len(rt$n_groups %||% 0L))

    nm <- character(0)
    for (lev in levs) {
      for (cl in coef_labels) {
        nm <- c(nm, if (identical(cl, "(Intercept)")) {
          sprintf("%s[%s]", gv, lev)
        } else {
          sprintf("%s.%s[%s]", gv, cl, lev)
        })
      }
    }
    re_names <- c(re_names, nm)
    re_layout[[k]] <- list(group_var = gv, levels = levs,
                           coef_labels = coef_labels,
                           n_groups = rt$n_groups %||% length(levs),
                           n_coefs = rt$n_coefs %||% length(coef_labels))
  }

  list(n_fixed     = n_fixed,
       fixed_names = fixed_names,
       param_names = c(fixed_names, re_names),
       re_layout   = re_layout)
}


# Fixed-effect posterior draws (n_samples x n_fixed), normalized across sampler
# shapes: the generic `$draws` matrix (columns [fixed, random]) restricted to
# the fixed block, or the Gibbs `$beta` matrix (fixed effects only). NULL on the
# Laplace tier, which carries no draws.
#' @keywords internal
.fixed_draws_mat <- function(object) {
  if (is.matrix(object$draws) && nrow(object$draws) >= 1L &&
      ncol(object$draws) >= 1L) {
    p <- min(object$n_fixed %||% ncol(object$draws), ncol(object$draws))
    return(object$draws[, seq_len(p), drop = FALSE])
  }
  if (is.matrix(object$beta) && ncol(object$beta) >= 1L) return(object$beta)
  NULL
}

# Random-effect posterior draws (n_samples x n_re): the `$draws` tail past the
# fixed block, or the Gibbs `$re` matrix. NULL when none.
#' @keywords internal
.re_draws_mat <- function(object) {
  nf <- object$n_fixed %||% 0L
  if (is.matrix(object$draws) && ncol(object$draws) > nf) {
    return(object$draws[, (nf + 1L):ncol(object$draws), drop = FALSE])
  }
  if (is.matrix(object$re) && ncol(object$re) >= 1L) return(object$re)
  NULL
}

# TRUE when the fit carries posterior draws of the fixed effects (sampler tier).
#' @keywords internal
.has_draws <- function(object) !is.null(.fixed_draws_mat(object))


# Grid-marginalized fixed-effect mean and covariance for a nested-Laplace fit,
# via the law of total variance over the hyperparameter grid:
#   mean = sum_g w_g mu_g
#   cov  = sum_g w_g (V_g + mu_g mu_g') - mean mean'
# mu_g / V_g are the per-grid fixed-effect mode / covariance retained under
# keep_grid_hessians (V_g = solve(grid_hessians[[g]])); w_g are the normalized
# grid weights. NULL when the per-grid pieces were not retained.
#' @keywords internal
.nested_fixed_moments <- function(object) {
  H <- object$grid_hessians
  M <- object$grid_modes
  if (is.null(H) || is.null(M) || is.null(object$weights)) return(NULL)
  w <- object$weights / sum(object$weights)
  p <- length(M[[1]])
  m <- numeric(p)
  S <- matrix(0, p, p)
  for (g in seq_along(w)) {
    Vg <- tryCatch(solve(H[[g]]), error = function(e) matrix(NA_real_, p, p))
    mu <- M[[g]]
    m <- m + w[g] * mu
    S <- S + w[g] * (Vg + tcrossprod(mu))
  }
  list(mean = m, cov = S - tcrossprod(m))
}


# Fixed-effect coefficient table (estimate, std.error, conf.low, conf.high),
# normalized across posterior shapes. The single source the coefficient-facing
# methods (summary, coef, confint, vcov, tidy) read from.
#' @keywords internal
.fit_fixed_table <- function(object, level = 0.95) {
  a <- (1 - level) / 2

  fd <- .fixed_draws_mat(object)
  if (!is.null(fd)) {
    idx <- seq_len(ncol(fd))
    nm  <- (object$fixed_names %||% object$param_names %||%
              colnames(fd) %||% paste0("param", idx))[idx]
    return(data.frame(
      term      = nm,
      estimate  = colMeans(fd),
      std.error = apply(fd, 2, stats::sd),
      conf.low  = apply(fd, 2, stats::quantile, a),
      conf.high = apply(fd, 2, stats::quantile, 1 - a),
      row.names = NULL, stringsAsFactors = FALSE
    ))
  }

  # Nested-Laplace fit: the posterior is a mixture over the hyperparameter grid.
  # When the per-grid fixed-effect pieces are retained (keep_grid_hessians, the
  # tulpa() default for nested fits), the grid-marginalized mean and covariance
  # come from the law of total variance (.nested_fixed_moments). Otherwise only
  # the marginalized mean is available, so SE is reported as NA rather than a
  # misleadingly small between-grid-only value.
  if (is.matrix(object$modes) && !is.null(object$weights)) {
    p   <- object$n_fixed %||% ncol(object$modes)
    idx <- seq_len(p)
    nm  <- (object$fixed_names %||% object$param_names %||%
              paste0("beta", idx))[idx]
    mom <- .nested_fixed_moments(object)
    if (!is.null(mom)) {
      est <- mom$mean[idx]
      se  <- sqrt(pmax(diag(mom$cov)[idx], 0))
      z   <- stats::qnorm(1 - a)
      return(data.frame(
        term = nm, estimate = est, std.error = se,
        conf.low = est - z * se, conf.high = est + z * se,
        row.names = NULL, stringsAsFactors = FALSE
      ))
    }
    w   <- object$weights / sum(object$weights)
    est <- as.numeric(crossprod(w, object$modes[, idx, drop = FALSE]))
    return(data.frame(
      term = nm, estimate = est,
      std.error = NA_real_, conf.low = NA_real_, conf.high = NA_real_,
      row.names = NULL, stringsAsFactors = FALSE
    ))
  }

  if (!is.null(object$mode) && !is.null(object$H_beta)) {
    p   <- object$n_fixed %||% nrow(object$H_beta)
    idx <- seq_len(p)
    est <- object$mode[idx]
    V   <- tryCatch(solve(object$H_beta), error = function(e) {
      warning("H_beta is singular; standard errors set to NA.", call. = FALSE)
      matrix(NA_real_, p, p)
    })
    se <- sqrt(pmax(diag(V)[idx], 0))
    z  <- stats::qnorm(1 - a)
    nm <- (object$fixed_names %||% object$param_names %||%
             paste0("beta", idx))[idx]
    return(data.frame(
      term      = nm,
      estimate  = est,
      std.error = se,
      conf.low  = est - z * se,
      conf.high = est + z * se,
      row.names = NULL, stringsAsFactors = FALSE
    ))
  }

  # Gaussian fit carrying the full-parameter covariance directly (e.g. AGQ,
  # whose inverse observed information is $cov rather than a precision $H_beta):
  # the fixed-effect block of $cov gives real SEs without draws or H_beta.
  est_all <- object$mode %||% object$means
  if (!is.null(est_all) && is.matrix(object$cov) && !anyNA(object$cov)) {
    p   <- object$n_fixed %||% length(est_all)
    idx <- seq_len(p)
    est <- as.numeric(est_all)[idx]
    se  <- sqrt(pmax(diag(object$cov)[idx], 0))
    z   <- stats::qnorm(1 - a)
    nm  <- (object$fixed_names %||% object$param_names %||%
              names(est_all) %||% paste0("beta", idx))[idx]
    return(data.frame(
      term = nm, estimate = est, std.error = se,
      conf.low = est - z * se, conf.high = est + z * se,
      row.names = NULL, stringsAsFactors = FALSE
    ))
  }

  if (!is.null(object$means)) {
    est <- as.numeric(object$means)
    nm  <- object$param_names %||% names(object$means) %||%
      paste0("param", seq_along(est))
    return(data.frame(
      term = nm[seq_along(est)], estimate = est,
      std.error = NA_real_, conf.low = NA_real_, conf.high = NA_real_,
      row.names = NULL, stringsAsFactors = FALSE
    ))
  }

  stop("Cannot summarize this tulpa_fit: it carries no $draws, $mode/$H_beta, ",
       "$cov, or $means.", call. = FALSE)
}


#' Fixed-effect coefficients
#'
#' @param object A `tulpa_fit` object.
#' @param ... Ignored.
#' @return Named numeric vector of fixed-effect posterior means (the Laplace
#'   mode for the Laplace tier). Random effects come from [ranef()].
#' @export
coef.tulpa_fit <- function(object, ...) {
  tab <- .fit_fixed_table(object)
  stats::setNames(tab$estimate, tab$term)
}

# Compact fit display, robust across posterior shapes: reads only fields the
# accessors above normalize, so sampler-, Laplace-, and nested-tier fits all
# print without NA/NULL noise.
#' @export
print.tulpa_fit <- function(x, ...) {
  header <- "tulpa fit"
  if (!is.null(x$backend)) header <- paste0(header, "  (", x$backend, ")")
  cat(header, "\n")
  n_obs <- x$N %||% x$n_obs
  if (!is.null(n_obs)) cat("  n_obs:", n_obs, "\n")
  fd <- .fixed_draws_mat(x)
  if (!is.null(fd)) cat("  posterior draws:", nrow(fd), "\n")
  tab <- tryCatch(.fit_fixed_table(x), error = function(e) NULL)
  if (!is.null(tab) && nrow(tab)) {
    cat("\nFixed effects:\n")
    print(data.frame(estimate  = round(tab$estimate, 4),
                     std.error = round(tab$std.error, 4),
                     row.names = tab$term))
  } else if (!is.null(x$means)) {
    cat("\nPosterior means:\n")
    print(round(x$means, 4))
  }
  if (is.numeric(x$sigma) && length(x$sigma) == 1L) {
    cat("\nsigma:", round(x$sigma, 4), "\n")
  }
  invisible(x)
}

#' Posterior summary of the fixed effects
#'
#' @param object A `tulpa_fit` object.
#' @param level Credible-interval level (default 0.95).
#' @param ... Ignored.
#' @return Data frame: estimate, std.error, and lower/upper credible bounds, one
#'   row per fixed effect. Sampler tiers report empirical quantiles; the Laplace
#'   tier reports the Gaussian approximation.
#' @export
summary.tulpa_fit <- function(object, level = 0.95, ...) {
  tab <- .fit_fixed_table(object, level = level)
  a <- (1 - level) / 2
  out <- data.frame(
    estimate  = tab$estimate,
    std.error = tab$std.error,
    `conf.low`  = tab$conf.low,
    `conf.high` = tab$conf.high,
    row.names = tab$term, check.names = FALSE
  )
  names(out)[3:4] <- sprintf("%.1f%%", 100 * c(a, 1 - a))
  out
}

#' Credible intervals for the fixed effects
#'
#' @param object A `tulpa_fit` object.
#' @param parm Parameter names or indices (default: all fixed effects).
#' @param level Interval level (default 0.95).
#' @param ... Ignored.
#' @return Matrix with lower and upper columns.
#' @export
confint.tulpa_fit <- function(object, parm = NULL, level = 0.95, ...) {
  tab <- .fit_fixed_table(object, level = level)
  a <- (1 - level) / 2
  ci <- as.matrix(tab[, c("conf.low", "conf.high")])
  rownames(ci) <- tab$term
  colnames(ci) <- sprintf("%.1f%%", 100 * c(a, 1 - a))
  if (!is.null(parm)) ci <- ci[parm, , drop = FALSE]
  ci
}

#' Variance-covariance matrix of the fixed effects
#'
#' @param object A `tulpa_fit` object.
#' @param ... Ignored.
#' @return Fixed-effect variance-covariance matrix (empirical for sampler tiers,
#'   `H_beta^-1` for the Laplace tier).
#' @export
vcov.tulpa_fit <- function(object, ...) {
  fd  <- .fixed_draws_mat(object)
  mom <- .nested_fixed_moments(object)
  if (!is.null(fd)) {
    V <- stats::cov(fd)
  } else if (!is.null(mom)) {
    p <- object$n_fixed %||% nrow(mom$cov)
    V <- mom$cov[seq_len(p), seq_len(p), drop = FALSE]
  } else if (!is.null(object$H_beta)) {
    p <- object$n_fixed %||% nrow(object$H_beta)
    V <- solve(object$H_beta)[seq_len(p), seq_len(p), drop = FALSE]
  } else if (is.matrix(object$cov) && !anyNA(object$cov)) {
    p <- object$n_fixed %||% nrow(object$cov)
    V <- object$cov[seq_len(p), seq_len(p), drop = FALSE]
  } else {
    stop("No posterior draws, grid moments, H_beta, or cov available for vcov().",
         call. = FALSE)
  }
  nm <- (object$fixed_names %||% object$param_names)[seq_len(ncol(V))]
  if (!is.null(nm)) dimnames(V) <- list(nm, nm)
  V
}

#' Log-likelihood at the posterior mean
#'
#' @param object A `tulpa_fit` object.
#' @param ... Ignored.
#' @return A `logLik` object.
#' @export
logLik.tulpa_fit <- function(object, ...) {
  ll <- if (!is.null(object$log_prob)) {
    mean(object$log_prob, na.rm = TRUE)
  } else if (!is.null(object$log_marginal)) {
    # A nested-Laplace fit stores one log-marginal per hyperparameter grid point;
    # the integrated log evidence is the log-sum-exp over the grid. This assumes
    # equal cell weights: exact for the tensor / uniform hyper-grid the generic
    # driver builds, and an approximation for a design-weighted (CCD) grid, whose
    # raw quadrature weights are not recoverable from the stored posterior
    # weights. A single-point (e.g. conditional Laplace) marginal passes through
    # unchanged. Non-finite cells (an inner Newton diverging in a grid corner
    # returns +Inf / NaN) are dropped from the log-sum-exp, matching the
    # finite-guarded weight grid; an unguarded max() would otherwise collapse
    # the evidence to NaN.
    lm <- object$log_marginal
    lm <- lm[is.finite(lm)]
    if (length(lm) == 0L) {
      NA_real_
    } else if (length(lm) > 1L) {
      mx <- max(lm)
      mx + log(sum(exp(lm - mx)))
    } else {
      lm
    }
  } else NA_real_
  attr(ll, "df")   <- object$n_fixed %||% length(object$mode) %||%
    length(object$means)
  attr(ll, "nobs") <- object$N %||% NA_integer_
  class(ll) <- "logLik"
  ll
}

# The tidy / glance generics come from `generics` (the shared broom-ecosystem
# home), re-exported here -- defining our own would mask broom's when both are
# attached.

#' @importFrom generics tidy
#' @export
generics::tidy

#' @importFrom generics glance
#' @export
generics::glance

#' Tidy fixed-effect table (broom-compatible)
#'
#' @param x A `tulpa_fit` object.
#' @param conf.level Interval level (default 0.95).
#' @param ... Ignored.
#' @return Data frame: term, estimate, std.error, conf.low, conf.high.
#' @examples
#' \donttest{
#' set.seed(1)
#' df <- data.frame(x = rnorm(100), g = factor(rep(1:10, 10)))
#' df$y <- rpois(100, exp(0.3 * df$x))
#' fit <- tulpa(y ~ x + (1 | g), data = df, family = "poisson")
#' tidy(fit)
#' }
#' @export
tidy.tulpa_fit <- function(x, conf.level = 0.95, ...) {
  .fit_fixed_table(x, level = conf.level)
}

#' Model-level summary statistics (broom-compatible)
#'
#' @param x A `tulpa_fit` object.
#' @param ... Ignored.
#' @return Single-row data frame.
#' @examples
#' \donttest{
#' set.seed(1)
#' df <- data.frame(x = rnorm(100))
#' df$y <- rpois(100, exp(0.3 * df$x))
#' fit <- tulpa(y ~ x, data = df, family = "poisson")
#' glance(fit)
#' }
#' @export
glance.tulpa_fit <- function(x, ...) {
  data.frame(
    n_fixed     = x$n_fixed %||% NA_integer_,
    n_samples   = x$n_samples %||% { fd <- .fixed_draws_mat(x); if (!is.null(fd)) nrow(fd) else NA_integer_ },
    logLik      = as.numeric(logLik(x)),
    n_divergent = if (!is.null(x$divergent)) sum(x$divergent) else NA_integer_,
    mean_accept = x$mean_accept %||% (if (!is.null(x$accept_prob)) mean(x$accept_prob) else NA_real_),
    converged   = x$converged %||% NA,
    stringsAsFactors = FALSE
  )
}

#' Random-effect summaries
#'
#' @param object A `tulpa_fit` object.
#' @param ... Ignored.
#' @return Data frame with one row per random-effect coefficient: term, group,
#'   level, estimate, and (sampler tier only) sd and credible bounds.
#' @examples
#' \donttest{
#' set.seed(1)
#' df <- data.frame(x = rnorm(100), g = factor(rep(1:10, 10)))
#' df$y <- rpois(100, exp(0.3 * df$x))
#' fit <- tulpa(y ~ x + (1 | g), data = df, family = "poisson")
#' ranef(fit)
#' }
#' @export
ranef <- function(object, ...) UseMethod("ranef")

#' @rdname ranef
#' @export
ranef.tulpa_fit <- function(object, ...) {
  layout <- object$re_layout
  n_fixed <- object$n_fixed %||% 0L
  if (is.null(layout) || length(layout) == 0L) return(data.frame())

  re_names <- (object$param_names %||% character(0))
  re_names <- if (length(re_names) > n_fixed) re_names[-seq_len(n_fixed)] else character(0)

  re <- .re_draws_mat(object)
  if (!is.null(re)) {
    nm <- re_names[seq_len(ncol(re))]
    return(data.frame(
      term = nm,
      estimate = colMeans(re),
      sd = apply(re, 2, stats::sd),
      conf.low = apply(re, 2, stats::quantile, 0.025),
      conf.high = apply(re, 2, stats::quantile, 0.975),
      row.names = NULL, stringsAsFactors = FALSE
    ))
  }

  if (!is.null(object$mode) && length(object$mode) > n_fixed) {
    blups <- object$mode[(n_fixed + 1L):length(object$mode)]
    nm <- re_names[seq_len(length(blups))]
    return(data.frame(
      term = nm, estimate = blups,
      sd = NA_real_, conf.low = NA_real_, conf.high = NA_real_,
      row.names = NULL, stringsAsFactors = FALSE
    ))
  }
  data.frame()
}

#' Plot fixed-effect posteriors
#'
#' @param x A `tulpa_fit` object.
#' @param type One of `"trace"`, `"density"`, `"pairs"`. The Laplace tier has no
#'   draws, so it always shows the Gaussian densities of the fixed effects.
#' @param ... Passed to plotting functions.
#' @return The input `x`, returned invisibly. Called for the side effect of
#'   producing base-graphics plots of the fixed-effect posteriors.
#' @export
plot.tulpa_fit <- function(x, type = c("density", "trace", "pairs"), ...) {
  type <- match.arg(type)

  if (!.has_draws(x)) {
    tab <- .fit_fixed_table(x)
    np <- nrow(tab)
    old_par <- graphics::par(mfrow = c(min(np, 4), 1), mar = c(4, 4, 1, 1))
    on.exit(graphics::par(old_par))
    for (j in seq_len(min(np, 4))) {
      m <- tab$estimate[j]; s <- tab$std.error[j]
      xs <- seq(m - 4 * s, m + 4 * s, length.out = 200)
      plot(xs, stats::dnorm(xs, m, s), type = "l", xlab = tab$term[j], ylab = "density")
      graphics::abline(v = m, col = "red", lty = 2)
    }
    return(invisible(x))
  }

  sub <- .fixed_draws_mat(x)
  p <- min(ncol(sub), 4L)
  sub <- sub[, seq_len(p), drop = FALSE]
  nms <- (x$fixed_names %||% x$param_names %||% colnames(sub))[seq_len(p)]

  if (type == "trace") {
    old_par <- graphics::par(mfrow = c(min(p, 4), 1), mar = c(2, 4, 1, 1))
    on.exit(graphics::par(old_par))
    for (j in seq_len(min(p, 4))) {
      plot(sub[, j], type = "l", ylab = nms[j], xlab = "")
    }
  } else if (type == "density") {
    old_par <- graphics::par(mfrow = c(min(p, 4), 1), mar = c(4, 4, 1, 1))
    on.exit(graphics::par(old_par))
    for (j in seq_len(min(p, 4))) {
      d <- stats::density(sub[, j]); plot(d, main = "", xlab = nms[j])
      graphics::abline(v = mean(sub[, j]), col = "red", lty = 2)
    }
  } else if (type == "pairs" && p > 1) {
    graphics::pairs(sub, labels = nms, pch = ".", col = grDevices::rgb(0, 0, 0, 0.2))
  }
  invisible(x)
}


# Rebuild the fixed-effect design matrix for `newdata` from the fit's formula,
# matching how tulpa_build_model_data() built it (model.matrix on the parsed
# fixed-effects formula, response dropped).
#' @keywords internal
.tulpa_fixed_design <- function(object, newdata) {
  parsed <- tulpa_parse_formula(object$formula)
  tt <- stats::delete.response(stats::terms(parsed$fixed_formula))
  mf <- stats::model.frame(tt, data = newdata, na.action = stats::na.pass)
  X  <- stats::model.matrix(tt, mf)
  attr(X, "offset") <- stats::model.offset(mf)
  X
}

# FEM projector from the fitted SPDE mesh to arbitrary coordinates. Requires a
# mesh-backed spec built with a coordinate formula; a custom C/G/A spec has no
# mesh to re-project through.
#' @keywords internal
.spde_A_at <- function(object, newdata) {
  sp <- object$spatial
  if (is.null(sp$mesh) || is.null(sp$coord_formula)) {
    stop("Spatial-field prediction at new coordinates needs an SPDE spec built ",
         "with a mesh and a coordinate formula, e.g. spatial_spde(~ x + y, ",
         "data); a custom C/G/A spec cannot project to new points. Pass ",
         "include_field = FALSE for the fixed-effect (population) prediction.",
         call. = FALSE)
  }
  vars <- all.vars(sp$coord_formula)
  miss <- setdiff(vars, names(newdata))
  if (length(miss)) {
    stop("newdata is missing the coordinate column(s): ",
         paste(miss, collapse = ", "), call. = FALSE)
  }
  coords <- cbind(newdata[[vars[1]]], newdata[[vars[2]]])
  tulpaMesh::fem_matrices(sp$mesh, obs_coords = coords)$A
}

# Project a fitted SPDE mesh-node field to arbitrary coordinates (kriging):
# A_new %*% w_hat with w_hat the posterior-mean node field.
#' @keywords internal
.spde_field_at <- function(object, newdata) {
  as.numeric(.spde_A_at(object, newdata) %*% object$spatial_effects)
}

# Krige a fitted HSGP field to coordinates (predict()). The field is
# grid-marginalised in C++ (cpp_hsgp_field_predict): for each hyperparameter
# grid cell the per-cell latent (modes tail) is spectral-scaled and projected
# through the Laplacian basis evaluated at the new coordinates, then weighted by
# the cell's posterior weight -- never a plug-in at the posterior mean. Returns
# NULL (decline, leaving eta at the fixed-effect prediction) when the fit is not
# a plain single-block HSGP nested fit (e.g. it carries an extra latent block),
# so the caller can fall through cleanly.
#' @keywords internal
.hsgp_field_at <- function(object, newdata) {
  sp <- object$spatial
  if (is.null(sp) || !identical(sp$type, "hsgp") || is.null(object$modes) ||
      is.null(object$weights) || is.null(object$sigma2_grid) ||
      is.null(object$lengthscale_grid) || is.null(sp$coords_matrix)) {
    return(NULL)
  }
  m       <- as.integer(sp[["m"]])
  Mtot    <- m * m
  p_fixed <- object$n_fixed %||% ncol(object$model_matrix)
  modes   <- object$modes
  # The HSGP basis coefficients are the latent block immediately after the fixed
  # effects. Any additional latent tail (an iid RE) means this simple slice is
  # not the field, so decline rather than mis-index.
  if (ncol(modes) != p_fixed + Mtot) return(NULL)
  beta_grid <- modes[, p_fixed + seq_len(Mtot), drop = FALSE]

  coords_train <- as.matrix(sp$coords_matrix)
  if (is.null(newdata)) {
    coords_new <- coords_train
  } else {
    cv <- sp$coord_vars
    miss <- setdiff(cv, names(newdata))
    if (length(miss)) {
      stop("newdata is missing the coordinate column(s): ",
           paste(miss, collapse = ", "), call. = FALSE)
    }
    coords_new <- as.matrix(newdata[, cv, drop = FALSE])
    # Reapply the training coordinate standardisation (scale() recorded the
    # training centre / scale as attributes on coords_matrix) so the prediction
    # basis matches the fitted one.
    if (isTRUE(sp$scale_coords)) {
      ctr <- attr(coords_train, "scaled:center")
      scl <- attr(coords_train, "scaled:scale")
      if (!is.null(ctr) && !is.null(scl)) {
        coords_new <- sweep(sweep(coords_new, 2, ctr, "-"), 2, scl, "/")
      }
    }
  }
  # Strip attributes so the coords reach C++ as plain numeric matrices.
  ct <- matrix(as.numeric(coords_train), nrow(coords_train), 2)
  cn <- matrix(as.numeric(coords_new),   nrow(coords_new),   2)
  as.numeric(cpp_hsgp_field_predict(
    ct, cn, m, as.numeric(sp[["c"]]),
    beta_grid, as.numeric(object$sigma2_grid),
    as.numeric(object$lengthscale_grid), as.numeric(object$weights)))
}

# Krige a fitted GP / NNGP field to coordinates (predict()). At the training
# locations (newdata = NULL) the field is the grid-marginalised posterior mode
# at each observation's location (exact, no re-kriging). At new coordinates the
# field is the NNGP conditional mean given the fitted field at the location's
# nearest training locations, computed per hyperparameter-grid cell in C++
# (cpp_gp_field_predict, reusing the fit's covariance + conditional kernels) and
# weighted over the grid. Returns NULL (decline) when the fit is not a plain
# single-block GP / NNGP nested fit.
#' @keywords internal
.gp_field_at <- function(object, newdata) {
  sp <- object$spatial
  if (is.null(sp) || !(sp$type %in% c("gp", "nngp")) || is.null(object$modes) ||
      is.null(object$weights) || is.null(object$theta_grid) ||
      is.null(sp$unique_coords)) {
    return(NULL)
  }
  tn <- object$theta_names
  si <- match("sigma2", tn); pj <- match("phi_gp", tn)
  if (is.na(si) || is.na(pj)) return(NULL)
  sigma2_grid <- object$theta_grid[, si]
  phi_grid    <- object$theta_grid[, pj]

  nloc    <- sp$n_unique %||% nrow(sp$unique_coords)
  p_fixed <- object$n_fixed %||% ncol(object$model_matrix)
  modes   <- object$modes
  # The field-at-locations latent is the block after the fixed effects; an extra
  # latent tail (an iid RE) means this slice is not the field, so decline.
  if (ncol(modes) != p_fixed + nloc) return(NULL)
  field_grid <- modes[, p_fixed + seq_len(nloc), drop = FALSE]
  uc <- matrix(as.numeric(sp$unique_coords), nrow(sp$unique_coords), 2)

  if (is.null(newdata)) {
    # Training locations: the grid-marginalised field at each unique location,
    # mapped back to observations by obs_to_loc (exact -- no re-kriging).
    field_loc <- as.numeric(as.numeric(object$weights) %*% field_grid)
    otl <- sp$obs_to_loc
    if (is.null(otl)) return(NULL)
    return(field_loc[otl])
  }

  cv   <- sp$coord_vars
  miss <- setdiff(cv, names(newdata))
  if (length(miss)) {
    stop("newdata is missing the coordinate column(s): ",
         paste(miss, collapse = ", "), call. = FALSE)
  }
  ncoord <- as.matrix(newdata[, cv, drop = FALSE])
  if (isTRUE(sp$scale_coords)) {
    ctr <- attr(sp$coords_matrix, "scaled:center")
    scl <- attr(sp$coords_matrix, "scaled:scale")
    if (!is.null(ctr) && !is.null(scl)) {
      ncoord <- sweep(sweep(ncoord, 2, ctr, "-"), 2, scl, "/")
    }
  }
  nc  <- matrix(as.numeric(ncoord), nrow(ncoord), 2)
  nn  <- as.integer(object$prior$nn %||% sp$nn %||% 10L)
  cty <- as.integer(object$prior$cov_type %||% 0L)
  as.numeric(cpp_gp_field_predict(nc, uc, field_grid, sigma2_grid, phi_grid,
                                  as.numeric(object$weights), nn, cty))
}

# The variance-convention dispersion of an SPDE fit, for the R-side working
# weights: a front-door tulpa() fit stores `$phi` (the variance); a direct
# fit_spde() fit stores the kernel-convention `$phi_kernel` (gaussian /
# lognormal: the residual SD).
#' @keywords internal
.spde_phi_variance <- function(object) {
  if (!is.null(object$phi)) return(object$phi)
  pk <- object$phi_kernel %||% 1.0
  if (identical(object$family, "gaussian") ||
      identical(object$family, "lognormal")) pk^2 else pk
}

# Linear-predictor SE at query points for an SPDE fit with the field included:
# Var(x*' beta + a*' w) = c' H^{-1} c with c = [x*, a*] and H the joint
# (beta, field) posterior precision at the fitted hyperparameters -- the
# working-weight cross term X'WA, the field precision Q(range, sigma), and
# the kernel's weak fixed-effect ridge (sigma_beta = 100). Conditional on the
# hyperparameters: a nested fit's grid spread in (range, sigma) is not
# propagated, so the SE is mildly optimistic when the hyperparameter posterior
# is wide. Integer-nu, no-RE fits only; everything else declines loudly.
#' @keywords internal
.spde_linpred_se <- function(object, X_new, A_new) {
  sp <- object$spatial
  if (.spde_nu_is_fractional(sp$nu)) {
    stop("field SE is not available for fractional-nu SPDE fits yet.",
         call. = FALSE)
  }
  X <- object$model_matrix
  if (is.null(X)) {
    stop("field SE needs the training design ($model_matrix); fit through ",
         "tulpa().", call. = FALSE)
  }
  p      <- ncol(X)
  n_mesh <- sp$n_mesh
  mode   <- object$mode
  if (is.null(mode) || length(mode) != p + n_mesh) {
    stop("field SE needs the joint (beta, field) mode with no extra ",
         "random-effect block; this fit's latent layout does not match.",
         call. = FALSE)
  }
  range_val <- object$range %||% object$nested$range_mean
  sigma_val <- object$sigma %||% object$nested$sigma_mean
  if (is.null(range_val) || is.null(sigma_val)) {
    stop("field SE needs the fitted (range, sigma); none stored.",
         call. = FALSE)
  }

  A    <- as(sp$A, "CsparseMatrix")
  beta <- mode[seq_len(p)]
  w    <- mode[p + seq_len(n_mesh)]
  eta  <- as.numeric(X %*% beta) + as.numeric(A %*% w) +
    (object$offset %||% 0)
  W <- glmm_weights(eta, object$family, object$n_trials,
                    .spde_phi_variance(object))

  kappa    <- sqrt(8 * sp$nu) / range_val
  tau_spde <- 1 / (sqrt(4 * pi) * kappa * sigma_val)
  Q <- .spde_precision_Q(sp, kappa, tau_spde)

  XtWX <- crossprod(X, W * X) + diag(1e-4, p)   # kernel ridge sigma_beta = 100
  WA   <- W * A
  AtWA <- Matrix::crossprod(A, WA)
  XtWA <- Matrix::crossprod(Matrix::Matrix(X, sparse = TRUE), WA)
  H <- rbind(
    cbind(Matrix::Matrix(XtWX, sparse = TRUE), XtWA),
    cbind(Matrix::t(XtWA), AtWA + Q)
  )
  Cq <- rbind(
    Matrix::t(Matrix::Matrix(X_new, sparse = TRUE)),
    Matrix::t(as(A_new, "CsparseMatrix"))
  )
  # Per-cell field SE sqrt(colSums(Cq (.) H^{-1} Cq)) is streamed column-by-
  # column in C++ (cpp_spde_field_se): the joint precision is factorized once,
  # then each query column is solved on its own, so the dense working set stays
  # O(p + n_mesh) rather than the (p + n_mesh) x n_cells dense H^{-1} Cq a large
  # prediction grid would otherwise form.
  if (ncol(Cq) == 0L) return(numeric(0))
  cpp_spde_field_se(
    as(as(H,  "generalMatrix"), "CsparseMatrix"),
    as(as(Cq, "generalMatrix"), "CsparseMatrix")
  )
}

#' Fitted values (population level)
#'
#' @description
#' In-sample mean response from the fixed effects and the observation offset
#' (`E[y] = g^{-1}(X beta + offset)`, trial-scaled for binomial). Random
#' effects are held at their prior mean of zero; group-level effects are in
#' [ranef()]. `y - fitted(object)` equals
#' `residuals(object, type = "response")`.
#'
#' @param object A `tulpa_fit` object (must carry `$model_matrix`).
#' @param ... Ignored.
#' @return Numeric vector of fitted mean responses, length `nobs`.
#' @export
fitted.tulpa_fit <- function(object, ...) {
  X <- object$model_matrix
  if (is.null(X)) {
    stop("fitted() needs the fixed-effect design ($model_matrix); refit with ",
         "tulpa().", call. = FALSE)
  }
  beta <- coef(object)
  eta  <- as.numeric(X %*% beta[colnames(X)]) + (object$offset %||% 0)
  family_response_mean(eta, object$family, n_trials = object$n_trials,
                       phi = object$phi %||% 1.0)
}

#' Residuals from a tulpa fit
#'
#' @description
#' Population-level residuals from the fixed-effect fitted mean: `"response"`
#' is `y - E[y | eta]` on the response scale (trial-scaled for binomial,
#' offset included); `"pearson"` additionally scales by the family standard
#' deviation `sqrt(Var(y | eta))` at the fitted linear predictor. Random
#' effects are held at zero, matching [fitted()].
#'
#' @param object A `tulpa_fit` object carrying `$y` and `$model_matrix`.
#' @param type `"pearson"` (default) or `"response"`.
#' @param ... Ignored.
#' @return Numeric vector of length `nobs(object)`.
#' @export
residuals.tulpa_fit <- function(object, type = c("pearson", "response"), ...) {
  type <- match.arg(type)
  y <- object$y
  if (is.null(y)) {
    stop("residuals() needs the response ($y); refit with tulpa().",
         call. = FALSE)
  }
  X <- object$model_matrix
  if (is.null(X)) {
    stop("residuals() needs the fixed-effect design ($model_matrix); refit ",
         "with tulpa().", call. = FALSE)
  }
  beta <- coef(object)
  eta  <- as.numeric(X %*% beta[colnames(X)]) + (object$offset %||% 0)
  mu   <- family_response_mean(eta, object$family, n_trials = object$n_trials,
                               phi = object$phi %||% 1.0)
  r <- as.numeric(y) - mu
  if (type == "pearson") {
    v <- family_variance(eta, object$family, n_trials = object$n_trials,
                         phi = object$phi %||% 1.0, phi2 = object$phi2)
    r <- r / sqrt(pmax(v, .Machine$double.eps))
  }
  r
}

#' Number of observations in a tulpa fit
#'
#' @param object A `tulpa_fit` object.
#' @param ... Ignored.
#' @return Integer observation count.
#' @export
nobs.tulpa_fit <- function(object, ...) {
  n <- object$N %||% (if (!is.null(object$y)) length(object$y) else NULL)
  if (is.null(n)) {
    stop("nobs() needs an observation count ($N or $y) on the fit.",
         call. = FALSE)
  }
  as.integer(n)
}

#' Predict at new covariate values (population level)
#'
#' @description
#' Prediction of the linear predictor at `newdata`, on the link or response
#' scale. The fixed-effect part is `X beta` with credible bounds from the
#' fixed-effect covariance ([vcov()]). For a fit carrying a continuous spatial
#' field, the posterior-mean field is interpolated (kriged) to the `newdata`
#' coordinates and added to the linear predictor by default, so `predict()`
#' gives the conditional (location-specific) prediction. Three continuous field
#' families are supported: an SPDE Matern field (`spatial_spde()`), projected
#' through the mesh; a Hilbert-space GP field (`spatial_gp(approx = "hsgp")`),
#' where the Laplacian basis is re-evaluated at the new coordinates (with the
#' training centring / boundary); and a GP / NNGP field (`spatial_gp()`),
#' interpolated by the NNGP conditional mean at each new location's nearest
#' training locations. The HSGP and GP/NNGP fields are marginalised over the
#' hyperparameter grid (not plugged in at the posterior mean). Ordinary random
#' effects are held at zero (population level); add group effects from [ranef()]
#' when needed.
#'
#' @param object A `tulpa_fit` object.
#' @param newdata Data frame of covariates (and, for an SPDE fit, the coordinate
#'   columns named in the spec's coordinate formula). If `NULL`, predicts at the
#'   training design (requires `$model_matrix`).
#' @param type `"link"` (linear predictor) or `"response"` (mean scale).
#' @param se.fit If `TRUE`, also return the link-scale standard error and
#'   credible bounds. With an included SPDE field the SE propagates the joint
#'   (fixed-effect, field) posterior precision at the fitted hyperparameters
#'   -- including the cross term -- conditional on `(range, sigma)` (a nested
#'   fit's hyperparameter-grid spread is not propagated, so the bound is
#'   mildly optimistic when that posterior is wide). Integer-nu, no-RE SPDE
#'   fits only; other layouts decline with an explanation.
#' @param level Credible-interval level (default 0.95).
#' @param include_field For a continuous-spatial fit (SPDE, HSGP, or GP/NNGP),
#'   add the kriged field to the prediction (default `TRUE`). `FALSE` gives the
#'   fixed-effect (population) prediction. Ignored for fits with no continuous
#'   field. For an HSGP or GP/NNGP fit the field is added to the point
#'   prediction but its uncertainty is not yet propagated into `se.fit` (the
#'   interval reflects the fixed-effect covariance only).
#' @param ... Ignored.
#' @return If `se.fit = FALSE`, a numeric vector. If `se.fit = TRUE`, a data
#'   frame with `fit`, `se.fit` (link scale), `lower`, `upper` on the requested
#'   scale.
#' @export
predict.tulpa_fit <- function(object, newdata = NULL,
                              type = c("link", "response"),
                              se.fit = FALSE, level = 0.95,
                              include_field = TRUE, ...) {
  type <- match.arg(type)
  beta <- coef(object)

  X <- if (is.null(newdata)) object$model_matrix else .tulpa_fixed_design(object, newdata)
  if (is.null(X)) {
    stop("predict() needs `newdata` (or a fit carrying $model_matrix).",
         call. = FALSE)
  }
  # Observation offset, matching fitted()/residuals(): the fit's stored offset at
  # the training design, or the offset() term re-evaluated on newdata.
  off <- if (is.null(newdata)) (object$offset %||% 0) else (attr(X, "offset") %||% 0)
  miss <- setdiff(names(beta), colnames(X))
  if (length(miss)) {
    stop("newdata cannot reproduce fixed-effect column(s): ",
         paste(miss, collapse = ", "), call. = FALSE)
  }
  X <- X[, names(beta), drop = FALSE]

  eta <- as.numeric(X %*% beta) + off

  # Kriged SPDE field. At training data (newdata = NULL) reuse the fitted
  # projector; at new coordinates re-project the mesh-node field through the
  # spec's mesh. Only mesh-backed SPDE fits carry a field here; every other fit
  # leaves `eta` at the fixed-effect (population) prediction. With se.fit the
  # joint (beta, field) posterior precision propagates the field uncertainty
  # (see .spde_linpred_se).
  is_spde_field <- isTRUE(include_field) &&
    identical(object$spatial$type, "spde") &&
    !is.null(object$spatial_effects)
  spde_se <- NULL
  if (is_spde_field) {
    A_new <- if (is.null(newdata)) object$spatial$A
             else .spde_A_at(object, newdata)
    eta <- eta + as.numeric(A_new %*% object$spatial_effects)
    if (se.fit) {
      spde_se <- .spde_linpred_se(object, X, A_new)
    }
  }

  # Kriged HSGP field: add the grid-marginalised Laplacian-basis field at the
  # prediction coordinates. Field-uncertainty SE propagation is not yet wired
  # here (unlike SPDE), so with se.fit the interval reflects the fixed-effect
  # covariance only -- the point prediction still includes the field.
  if (isTRUE(include_field) && identical(object$spatial$type, "hsgp") &&
      !is.null(object$modes)) {
    hsgp_fld <- .hsgp_field_at(object, newdata)
    if (!is.null(hsgp_fld)) eta <- eta + hsgp_fld
  }

  # Kriged GP / NNGP field: the NNGP conditional mean at the prediction
  # coordinates (grid-marginalised). Same se.fit caveat as HSGP.
  if (isTRUE(include_field) && !is.null(object$spatial) &&
      object$spatial$type %in% c("gp", "nngp") && !is.null(object$modes)) {
    gp_fld <- .gp_field_at(object, newdata)
    if (!is.null(gp_fld)) eta <- eta + gp_fld
  }

  if (!se.fit) {
    return(if (type == "response") {
      family_mean(eta, object$family, phi = object$phi %||% 1.0)
    } else eta)
  }

  se <- if (!is.null(spde_se)) {
    spde_se
  } else {
    V <- vcov(object)
    sqrt(pmax(rowSums((X %*% V) * X), 0))
  }
  z  <- stats::qnorm(1 - (1 - level) / 2)
  lo <- eta - z * se
  hi <- eta + z * se
  if (type == "response") {
    ph  <- object$phi %||% 1.0
    eta <- family_mean(eta, object$family, phi = ph)
    lo  <- family_mean(lo, object$family, phi = ph)   # monotone inverse links: endpoints map through
    hi  <- family_mean(hi, object$family, phi = ph)
  }
  data.frame(fit = eta, se.fit = se, lower = lo, upper = hi)
}
