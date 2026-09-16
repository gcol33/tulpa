# categorical_accessors.R
# ------------------------------------------------------------------------------
# Observation-level accessors for a categorical response: the baseline-category
# multinomial logit (tulpa_multinomial) and the cumulative-link ordinal model
# (tulpa_ordinal). Both carry class `tulpa_categorical`, a K-level response and
# a fixed-effect design, and both are linear in their parameter vector on the
# link scale, so one set of methods serves them:
#
#   link design   A_c (N x P), one per link column c:  eta_c = A_c theta
#   probabilities P (N x K) from the link columns, per model
#
# The parameter vector `theta` is the fit's own layout (`param_names`):
# multinomial [class 1 terms, ..., class K-1 terms]; ordinal [beta, cutpoints].
# ------------------------------------------------------------------------------

# Covariate design at the training data or rebuilt at `newdata`, restricted to
# the columns the fit estimated (the ordinal model carries no intercept).
#' @keywords internal
.categorical_design <- function(object, newdata) {
  X0 <- object$model_matrix
  if (is.null(newdata)) return(X0)
  X <- .tulpa_fixed_design(object, newdata)
  miss <- setdiff(colnames(X0), colnames(X))
  if (length(miss)) {
    stop("newdata cannot reproduce design column(s): ",
         paste(miss, collapse = ", "), call. = FALSE)
  }
  X[, colnames(X0), drop = FALSE]
}

# The link-scale design, one N x P matrix per link column: the K - 1 class
# logits against the baseline for the multinomial, the single location
# predictor `x' beta` for the ordinal model (its cutpoints enter the
# probabilities, not the location).
#' @keywords internal
.categorical_link_design <- function(object, X) {
  P <- ncol(object$draws)
  p <- ncol(X)
  K1 <- object$n_classes - 1L
  block <- function(start) {
    A <- matrix(0, nrow(X), P)
    A[, start + seq_len(p)] <- X
    A
  }
  if (inherits(object, "tulpa_multinomial")) {
    A <- lapply(seq_len(K1), function(j) block((j - 1L) * p))
    names(A) <- object$classes[seq_len(K1)]
    return(A)
  }
  list(eta = block(0L))
}

# N x K class probabilities at one parameter vector `theta`.
#' @keywords internal
.categorical_probs <- function(object, A, theta) {
  eta <- vapply(A, function(a) as.numeric(a %*% theta), numeric(nrow(A[[1L]])))
  eta <- matrix(eta, nrow(A[[1L]]))
  if (inherits(object, "tulpa_multinomial")) {
    Pm <- .multinomial_class_probs(eta)
    colnames(Pm) <- object$classes
    return(Pm)
  }
  K1 <- object$n_classes - 1L
  cuts <- theta[length(theta) - K1 + seq_len(K1)]
  Pm <- .ordinal_class_probs(as.numeric(eta), cuts, .ordinal_link_cdf(object$link))
  colnames(Pm) <- object$levels
  Pm
}

# Pointwise log-likelihood [nrow(theta) x N]: log P(y_i = observed class) at
# each row of the parameter matrix `theta`, at the training design.
#' @keywords internal
.categorical_loglik <- function(object, theta) {
  A   <- .categorical_link_design(object, object$model_matrix)
  N   <- nrow(object$model_matrix)
  obs <- cbind(seq_len(N), as.integer(object$y))
  ll  <- matrix(0, nrow(theta), N)
  for (s in seq_len(nrow(theta))) {
    ll[s, ] <- log(.categorical_probs(object, A, theta[s, ])[obs])
  }
  ll
}

# The response levels of a categorical fit, in class-index order.
#' @keywords internal
.categorical_levels <- function(object) {
  if (inherits(object, "tulpa_multinomial")) object$classes else object$levels
}

#' Observation-level accessors for categorical fits
#'
#' @description
#' [fitted()], [residuals()], [predict()] and [posterior_predict()] for a
#' categorical response: a [tulpa_multinomial()] fit (`family = "multinomial"`)
#' or a [tulpa_ordinal()] fit (`family = "ordinal"`). The response is one of K
#' classes, so the response-scale quantities are N x K matrices with one column
#' per class.
#'
#' * `fitted()` returns the class probabilities at the posterior mode.
#' * `residuals()` returns, per class, the indicator of the observed class minus
#'   its fitted probability (`"response"`), or that difference divided by the
#'   indicator's standard deviation `sqrt(p (1 - p))` (`"pearson"`). A
#'   categorical response has no single scalar residual; each class column is
#'   the Bernoulli residual of that class's indicator.
#' * `predict()` returns the link scale -- the K - 1 baseline-category logits of
#'   the multinomial model (N x (K - 1)), or the ordinal location predictor
#'   `x' beta` (length N) -- or the class probabilities (`type = "response"`).
#' * `posterior_predict()` draws one class per observation for each posterior
#'   draw and returns the class indices, with the response levels attached;
#'   [simulate.tulpa_fit()] turns them into factor columns.
#'
#' The pointwise log-likelihood behind [cpo()], [dic()], `loo::waic()` and
#' `loo::loo()` is the log probability of each observed class at each stored
#' posterior draw; [dic()] evaluates it at the posterior-mean parameters.
#'
#' @param object A `tulpa_multinomial` or `tulpa_ordinal` fit.
#' @param newdata Optional data frame of covariates; `NULL` uses the training
#'   design.
#' @param type For `residuals()`, `"pearson"` (default) or `"response"`. For
#'   `predict()`, `"link"` (default) or `"response"`.
#' @param se.fit For `predict()`, also return standard errors and credible
#'   bounds. On the link scale the standard error is exact for the Gaussian
#'   (Laplace) posterior, `sqrt(a' V a)`, with Gaussian bounds; on the response
#'   scale the standard error and the bounds are the posterior SD and quantiles
#'   of each class probability over the fit's posterior draws.
#' @param level Credible-interval level (default 0.95).
#' @param ndraws Number of posterior draws for `posterior_predict()`; defaults
#'   to all stored draws.
#' @param seed Optional integer seed (RNG state is restored on exit).
#' @param ... Ignored.
#' @return `fitted()`: an N x K probability matrix. `residuals()`: an N x K
#'   matrix. `predict()`: a matrix (or, for the ordinal link scale, a vector);
#'   with `se.fit = TRUE` a list of `fit`, `se.fit`, `lower`, `upper` of that
#'   shape. `posterior_predict()`: an `ndraws x N` integer matrix of class
#'   indices with attributes `levels` and `ordered`.
#' @seealso [tulpa_multinomial()], [tulpa_ordinal()]
#' @name categorical_accessors
NULL

#' @rdname categorical_accessors
#' @export
fitted.tulpa_categorical <- function(object, ...) {
  A <- .categorical_link_design(object, object$model_matrix)
  .categorical_probs(object, A, as.numeric(object$means))
}

#' @rdname categorical_accessors
#' @export
residuals.tulpa_categorical <- function(object, type = c("pearson", "response"),
                                        ...) {
  type <- match.arg(type)
  Pm <- fitted(object)
  Y  <- outer(as.integer(object$y), seq_len(ncol(Pm)), "==") * 1
  r  <- Y - Pm
  if (type == "pearson") r <- r / sqrt(pmax(Pm * (1 - Pm), .Machine$double.eps))
  r
}

#' @rdname categorical_accessors
#' @export
predict.tulpa_categorical <- function(object, newdata = NULL,
                                      type = c("link", "response"),
                                      se.fit = FALSE, level = 0.95, ...) {
  type  <- match.arg(type)
  X     <- .categorical_design(object, newdata)
  A     <- .categorical_link_design(object, X)
  theta <- as.numeric(object$means)
  shape <- function(cols) if (length(cols) == 1L) cols[[1L]]
                          else do.call(cbind, cols)
  a <- (1 - level) / 2

  if (type == "link") {
    fit <- shape(lapply(A, function(Ac) as.numeric(Ac %*% theta)))
    if (!se.fit) return(fit)
    V  <- vcov(object)
    cn <- colnames(object$draws)
    if (!is.null(cn) && all(cn %in% rownames(V))) V <- V[cn, cn, drop = FALSE]
    if (nrow(V) != ncol(object$draws)) {
      stop("predict(): the covariance does not cover the fit's parameter ",
           "vector.", call. = FALSE)
    }
    se <- shape(lapply(A, function(Ac) sqrt(pmax(rowSums((Ac %*% V) * Ac), 0))))
    z  <- stats::qnorm(1 - a)
    return(list(fit = fit, se.fit = se, lower = fit - z * se,
                upper = fit + z * se))
  }

  fit <- .categorical_probs(object, A, theta)
  if (!se.fit) return(fit)
  D <- object$draws
  S <- nrow(D)
  draws_p <- array(0, c(S, nrow(fit), ncol(fit)))
  for (s in seq_len(S)) draws_p[s, , ] <- .categorical_probs(object, A, D[s, ])
  summarise <- function(f) {
    m <- apply(draws_p, c(2L, 3L), f)
    dimnames(m) <- dimnames(fit)
    m
  }
  list(fit = fit, se.fit = summarise(stats::sd),
       lower = summarise(function(v) stats::quantile(v, a, names = FALSE)),
       upper = summarise(function(v) stats::quantile(v, 1 - a, names = FALSE)))
}

#' @rdname categorical_accessors
#' @export
posterior_predict.tulpa_categorical <- function(object, newdata = NULL,
                                                ndraws = NULL, seed = NULL,
                                                ...) {
  .seed_scoped(seed)
  X <- .categorical_design(object, newdata)
  A <- .categorical_link_design(object, X)
  D <- object$draws
  if (!is.null(ndraws) && ndraws < nrow(D)) {
    D <- D[sample.int(nrow(D), ndraws), , drop = FALSE]
  }
  lev <- .categorical_levels(object)
  K   <- length(lev)
  out <- matrix(0L, nrow(D), nrow(X))
  for (s in seq_len(nrow(D))) {
    C <- t(apply(.categorical_probs(object, A, D[s, ]), 1L, cumsum))
    out[s, ] <- pmin(1L + as.integer(rowSums(stats::runif(nrow(X)) > C)), K)
  }
  attr(out, "levels")  <- lev
  attr(out, "ordered") <- inherits(object, "tulpa_ordinal")
  out
}

#' @rdname categorical_accessors
#' @export
pp_check.tulpa_categorical <- function(object, ndraws = 50, ...) {
  if (!requireNamespace("bayesplot", quietly = TRUE)) {
    stop("Package 'bayesplot' is required for pp_check. Install with:\n",
         "  install.packages('bayesplot')", call. = FALSE)
  }
  y <- object$y
  if (is.null(y)) {
    stop("pp_check() needs the observed response; this fit stores no $y.",
         call. = FALSE)
  }
  yrep <- posterior_predict(object, ndraws = max(ndraws, 100L))
  lev  <- .categorical_levels(object)
  y_int <- as.integer(factor(y, levels = lev))
  if (nrow(yrep) > ndraws) {
    yrep <- yrep[sample.int(nrow(yrep), ndraws), , drop = FALSE]
  }
  bayesplot::ppc_bars(y_int, yrep, ...) +
    ggplot2::ggtitle("Posterior predictive check") +
    ggplot2::xlab(paste(lev, collapse = " / "))
}
