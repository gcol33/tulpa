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
  if (is.matrix(object$draws) && ncol(object$draws) >= 1L) {
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
       "or $means.", call. = FALSE)
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
  } else {
    stop("No posterior draws, grid moments, or H_beta available for vcov().",
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
    # the integrated log evidence is the log-sum-exp over the grid. A single-point
    # (e.g. conditional Laplace) marginal passes through unchanged.
    lm <- object$log_marginal
    if (length(lm) > 1L) {
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

#' Tidy fixed-effect table (broom-compatible)
#'
#' @param x A `tulpa_fit` object.
#' @param conf.level Interval level (default 0.95).
#' @param ... Ignored.
#' @return Data frame: term, estimate, std.error, conf.low, conf.high.
#' @export
tidy <- function(x, ...) UseMethod("tidy")

#' @rdname tidy
#' @export
tidy.tulpa_fit <- function(x, conf.level = 0.95, ...) {
  .fit_fixed_table(x, level = conf.level)
}

#' Model-level summary statistics (broom-compatible)
#'
#' @param x A `tulpa_fit` object.
#' @param ... Ignored.
#' @return Single-row data frame.
#' @export
glance <- function(x, ...) UseMethod("glance")

#' @rdname glance
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
  stats::model.matrix(tt, data = newdata)
}

#' Fitted values (population level)
#'
#' @description
#' In-sample mean response from the fixed effects (`E[y] = g^{-1}(X beta)`).
#' Random effects are held at their prior mean of zero; group-level effects are
#' in [ranef()].
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
  family_mean(as.numeric(X %*% beta[colnames(X)]), object$family)
}

#' Predict at new covariate values (population level)
#'
#' @description
#' Fixed-effect prediction: the linear predictor `X beta` at `newdata`, on the
#' link or response scale, with credible bounds from the fixed-effect
#' covariance ([vcov()]). Random effects are held at zero (population-level
#' prediction); add group effects from [ranef()] when needed.
#'
#' @param object A `tulpa_fit` object.
#' @param newdata Data frame of covariates. If `NULL`, predicts at the training
#'   design (requires `$model_matrix`).
#' @param type `"link"` (linear predictor) or `"response"` (mean scale).
#' @param se.fit If `TRUE`, also return the link-scale standard error and
#'   credible bounds.
#' @param level Credible-interval level (default 0.95).
#' @param ... Ignored.
#' @return If `se.fit = FALSE`, a numeric vector. If `se.fit = TRUE`, a data
#'   frame with `fit`, `se.fit` (link scale), `lower`, `upper` on the requested
#'   scale.
#' @export
predict.tulpa_fit <- function(object, newdata = NULL,
                              type = c("link", "response"),
                              se.fit = FALSE, level = 0.95, ...) {
  type <- match.arg(type)
  beta <- coef(object)

  X <- if (is.null(newdata)) object$model_matrix else .tulpa_fixed_design(object, newdata)
  if (is.null(X)) {
    stop("predict() needs `newdata` (or a fit carrying $model_matrix).",
         call. = FALSE)
  }
  miss <- setdiff(names(beta), colnames(X))
  if (length(miss)) {
    stop("newdata cannot reproduce fixed-effect column(s): ",
         paste(miss, collapse = ", "), call. = FALSE)
  }
  X <- X[, names(beta), drop = FALSE]

  eta <- as.numeric(X %*% beta)
  if (!se.fit) {
    return(if (type == "response") family_mean(eta, object$family) else eta)
  }

  V  <- vcov(object)
  se <- sqrt(pmax(rowSums((X %*% V) * X), 0))
  z  <- stats::qnorm(1 - (1 - level) / 2)
  lo <- eta - z * se
  hi <- eta + z * se
  if (type == "response") {
    eta <- family_mean(eta, object$family)
    lo  <- family_mean(lo, object$family)   # monotone inverse links: endpoints map through
    hi  <- family_mean(hi, object$family)
  }
  data.frame(fit = eta, se.fit = se, lower = lo, upper = hi)
}
