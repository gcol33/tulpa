# ============================================================================
# Generic S3 methods for tulpa_fit objects
# Model packages inherit these via class = c("model_fit", "tulpa_fit")
# ============================================================================

#' Extract coefficients from posterior means
#' @param object A `tulpa_fit` object with `$means` and `$process_info`.
#' @param ... Ignored.
#' @return Named list of coefficient vectors per process, or single vector.
#' @export
coef.tulpa_fit <- function(object, ...) {
  if (is.null(object$process_info)) {
    # Simple fit: return means directly
    return(object$means)
  }
  result <- list()
  offset <- 0
  for (pi in object$process_info) {
    idx <- offset + seq_len(pi$p)
    vals <- object$means[idx]
    names(vals) <- pi$coef_names
    result[[pi$name]] <- vals
    offset <- offset + pi$p
  }
  result
}

#' Confidence intervals from posterior quantiles
#' @param object A `tulpa_fit` object with `$draws`.
#' @param parm Parameter names or indices.
#' @param level Confidence level (default 0.95).
#' @param ... Ignored.
#' @return Matrix with lower and upper columns.
#' @export
confint.tulpa_fit <- function(object, parm = NULL, level = 0.95, ...) {
  alpha <- (1 - level) / 2
  probs <- c(alpha, 1 - alpha)
  draws <- object$draws
  if (is.null(parm)) {
    n_fixed <- if (!is.null(object$process_info)) {
      sum(vapply(object$process_info, function(pi) pi$p, integer(1)))
    } else ncol(draws)
    parm <- seq_len(n_fixed)
  }
  ci <- t(apply(draws[, parm, drop = FALSE], 2, quantile, probs = probs))
  colnames(ci) <- paste0(100 * probs, "%")
  ci
}

#' Variance-covariance matrix from posterior
#' @param object A `tulpa_fit` object with `$draws`.
#' @param ... Ignored.
#' @return Variance-covariance matrix.
#' @export
vcov.tulpa_fit <- function(object, ...) {
  n_fixed <- if (!is.null(object$process_info)) {
    sum(vapply(object$process_info, function(pi) pi$p, integer(1)))
  } else ncol(object$draws)
  cov(object$draws[, seq_len(n_fixed), drop = FALSE])
}

#' Log-likelihood (at posterior mean)
#' @param object A `tulpa_fit` object with `$log_prob`.
#' @param ... Ignored.
#' @return A `logLik` object.
#' @export
logLik.tulpa_fit <- function(object, ...) {
  ll <- if (!is.null(object$log_prob)) mean(object$log_prob, na.rm = TRUE) else NA_real_
  n_fixed <- if (!is.null(object$process_info)) {
    sum(vapply(object$process_info, function(pi) pi$p, integer(1)))
  } else length(object$means)
  attr(ll, "df") <- n_fixed
  attr(ll, "nobs") <- object$N %||% NA_integer_
  class(ll) <- "logLik"
  ll
}

#' Posterior summary table
#' @param object A `tulpa_fit` object with `$draws`.
#' @param ... Ignored.
#' @return Data frame with mean, sd, quantiles.
#' @export
summary.tulpa_fit <- function(object, ...) {
  draws <- object$draws
  n_params <- ncol(draws)
  result <- data.frame(
    mean = colMeans(draws),
    sd = apply(draws, 2, sd),
    q2.5 = apply(draws, 2, quantile, 0.025),
    q50 = apply(draws, 2, quantile, 0.50),
    q97.5 = apply(draws, 2, quantile, 0.975)
  )
  if (!is.null(object$param_names)) {
    n_named <- min(length(object$param_names), n_params)
    rownames(result)[seq_len(n_named)] <- object$param_names[seq_len(n_named)]
  }
  result
}

#' Tidy model output (broom-compatible)
#' @param x A `tulpa_fit` object.
#' @param conf.level Confidence level (default 0.95).
#' @param ... Ignored.
#' @return Data frame with term, estimate, std.error, conf.low, conf.high.
#' @export
tidy <- function(x, ...) UseMethod("tidy")

#' @rdname tidy
#' @export
tidy.tulpa_fit <- function(x, conf.level = 0.95, ...) {
  alpha <- (1 - conf.level) / 2
  draws <- x$draws
  if (!is.null(x$process_info)) {
    rows <- list()
    offset <- 0
    for (pi in x$process_info) {
      for (j in seq_len(pi$p)) {
        d <- draws[, offset + j]
        rows[[length(rows) + 1]] <- data.frame(
          process = pi$name, term = pi$coef_names[j],
          estimate = mean(d), std.error = sd(d),
          conf.low = quantile(d, alpha), conf.high = quantile(d, 1 - alpha),
          stringsAsFactors = FALSE)
      }
      offset <- offset + pi$p
    }
    do.call(rbind, rows)
  } else {
    n <- ncol(draws)
    nms <- colnames(draws) %||% paste0("param", seq_len(n))
    data.frame(
      term = nms,
      estimate = colMeans(draws),
      std.error = apply(draws, 2, sd),
      conf.low = apply(draws, 2, quantile, alpha),
      conf.high = apply(draws, 2, quantile, 1 - alpha),
      stringsAsFactors = FALSE)
  }
}

#' Model-level summary statistics (broom-compatible)
#' @param x A `tulpa_fit` object.
#' @param ... Ignored.
#' @return Single-row data frame.
#' @export
glance <- function(x, ...) UseMethod("glance")

#' @rdname glance
#' @export
glance.tulpa_fit <- function(x, ...) {
  data.frame(
    n_params = x$n_params %||% ncol(x$draws),
    n_samples = x$n_samples %||% nrow(x$draws),
    n_divergent = if (!is.null(x$divergent)) sum(x$divergent) else NA_integer_,
    mean_accept = if (!is.null(x$accept_prob)) mean(x$accept_prob) else NA_real_,
    stringsAsFactors = FALSE)
}

#' Extract random effects
#' @param object A `tulpa_fit` object.
#' @param ... Ignored.
#' @return Data frame with RE summaries.
#' @export
ranef <- function(object, ...) UseMethod("ranef")

#' @rdname ranef
#' @export
ranef.tulpa_fit <- function(object, ...) {
  draws <- object$draws
  n_fixed <- if (!is.null(object$process_info)) {
    sum(vapply(object$process_info, function(pi) pi$p, integer(1)))
  } else return(data.frame())
  n_extra <- length(object$det_visit_names %||% character(0))
  re_start <- n_fixed + n_extra + 1
  if (re_start > ncol(draws)) return(data.frame())

  re_draws <- draws[, re_start:ncol(draws), drop = FALSE]
  re_names <- colnames(re_draws) %||% paste0("re[", seq_len(ncol(re_draws)), "]")
  data.frame(
    term = re_names,
    mean = colMeans(re_draws),
    sd = apply(re_draws, 2, sd),
    q2.5 = apply(re_draws, 2, quantile, 0.025),
    q97.5 = apply(re_draws, 2, quantile, 0.975),
    stringsAsFactors = FALSE)
}

#' Plot posterior diagnostics
#' @param x A `tulpa_fit` object.
#' @param type One of `"trace"`, `"density"`, `"pairs"`.
#' @param ... Passed to plotting functions.
#' @export
plot.tulpa_fit <- function(x, type = c("trace", "density", "pairs"), ...) {
  type <- match.arg(type)
  draws <- x$draws
  n_fixed <- if (!is.null(x$process_info)) {
    min(ncol(draws), sum(vapply(x$process_info, function(pi) pi$p, integer(1))))
  } else min(ncol(draws), 4)
  draws_sub <- draws[, seq_len(n_fixed), drop = FALSE]
  nms <- (x$param_names %||% colnames(draws))[seq_len(n_fixed)]

  if (type == "trace") {
    np <- ncol(draws_sub)
    old_par <- par(mfrow = c(min(np, 4), 1), mar = c(2, 4, 1, 1))
    on.exit(par(old_par))
    for (j in seq_len(min(np, 4))) plot(draws_sub[, j], type = "l", ylab = nms[j], xlab = "")
  } else if (type == "density") {
    np <- ncol(draws_sub)
    old_par <- par(mfrow = c(min(np, 4), 1), mar = c(2, 4, 1, 1))
    on.exit(par(old_par))
    for (j in seq_len(min(np, 4))) {
      d <- density(draws_sub[, j]); plot(d, main = "", xlab = nms[j])
      abline(v = mean(draws_sub[, j]), col = "red", lty = 2)
    }
  } else if (type == "pairs" && n_fixed > 1) {
    pairs(draws_sub, labels = nms, pch = ".", col = rgb(0, 0, 0, 0.2))
  }
  invisible(x)
}
