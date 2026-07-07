# ordinal.R
# ------------------------------------------------------------------------------
# Ordinal (ordered K-class) cumulative-logit / proportional-odds regression via a
# Laplace fit. For an ordered response the cumulative probabilities are
#   P(y <= j | x) = logit^{-1}(c_j - x'beta),   j = 1..K-1,
# with ordered cutpoints c_1 < ... < c_{K-1} and NO fixed-effect intercept (the
# cutpoints absorb the level). The penalized mode is found with L-BFGS on the
# negative log-posterior (a ridge prior on beta and the cutpoints) and summarized
# by a Laplace approximation from the numerical Hessian at the mode. Ordering is
# guaranteed by the reparameterization c_1 = g_1, c_k = c_{k-1} + exp(g_k).
# ------------------------------------------------------------------------------

# Negative log-posterior in the unconstrained (beta, gamma) parameterization.
.ordinal_nll <- function(par, X, cls, K, tau_b, tau_c) {
  p  <- ncol(X); K1 <- K - 1L
  beta <- par[seq_len(p)]
  gam  <- par[(p + 1L):(p + K1)]
  cuts <- cumsum(c(gam[1L], exp(gam[-1L])))       # ordered cutpoints
  eta  <- as.numeric(X %*% beta)
  Fmat <- stats::plogis(outer(-eta, cuts, "+"))   # [n x K1], F_{ij} = F(c_j - eta_i)
  Fhi  <- cbind(Fmat, 1)
  Flo  <- cbind(0, Fmat)
  n    <- length(eta)
  pobs <- Fhi[cbind(seq_len(n), cls)] - Flo[cbind(seq_len(n), cls)]
  pobs <- pmax(pobs, 1e-12)
  -(sum(log(pobs)) - 0.5 * tau_b * sum(beta^2) - 0.5 * tau_c * sum(gam^2))
}

#' Ordinal (ordered K-class) cumulative-logit regression via Laplace
#'
#' @description
#' Fits a proportional-odds cumulative-logit model for an ordered factor
#' response: `P(y <= j | x) = plogis(c_j - x'beta)` with ordered cutpoints
#' `c_1 < ... < c_{K-1}`. The penalized mode (a ridge prior on the coefficients
#' and cutpoints) is found by L-BFGS and summarized by a Laplace approximation.
#' There is no separate intercept -- the cutpoints carry the baseline levels.
#'
#' @param formula Model formula; the response must be an ordered (or coercible)
#'   factor with >= 3 levels. An intercept in `formula` is dropped.
#' @param data A data frame.
#' @param beta_prior_sd,cut_prior_sd SDs of the mean-zero Gaussian ridge priors
#'   on the coefficients and the cutpoint parameters (defaults 10 and 10).
#' @param control List of numerical knobs: `max_iter` (default 200), `n_draws`
#'   (default 2000), `seed`.
#'
#' @return A `tulpa_fit` (subclass `tulpa_ordinal`) with `coef` (covariate
#'   effects), `cutpoints`, `vcov`, `draws`, `log_marginal`, `levels`.
#'
#' @seealso [tulpa_multinomial()] for the nominal (unordered) case.
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 400L; x <- rnorm(n)
#' cuts <- c(-1, 0.5, 2); eta <- 0.8 * x
#' Fm <- plogis(outer(-eta, cuts, "+")); P <- cbind(Fm, 1) - cbind(0, Fm)
#' y <- ordered(apply(P, 1, function(pr) sample.int(4L, 1L, prob = pr)))
#' fit <- tulpa_ordinal(y ~ x, data = data.frame(y = y, x = x))
#' fit$coefficients; fit$cutpoints
#' }
#' @export
tulpa_ordinal <- function(formula, data, beta_prior_sd = 10, cut_prior_sd = 10,
                          control = list()) {
  max_iter <- as.integer(control$max_iter %||% 200L)
  n_draws  <- as.integer(control$n_draws %||% 2000L)

  mf <- stats::model.frame(formula, data)
  y  <- stats::model.response(mf)
  if (!is.factor(y)) y <- factor(y, ordered = TRUE)
  K  <- nlevels(y)
  if (K < 3L) {
    stop("tulpa_ordinal() needs a response with >= 3 ordered levels; for 2 use ",
         "family = \"binomial\".", call. = FALSE)
  }
  K1  <- K - 1L
  cls <- as.integer(y)
  X   <- stats::model.matrix(stats::terms(mf), mf)
  X   <- X[, setdiff(colnames(X), "(Intercept)"), drop = FALSE]   # cutpoints carry it
  p   <- ncol(X)
  if (p < 1L) stop("tulpa_ordinal() needs at least one predictor.", call. = FALSE)

  tau_b <- 1 / beta_prior_sd^2
  tau_c <- 1 / cut_prior_sd^2

  # Init: zero effects, evenly spread cutpoints on the logit scale.
  c0   <- stats::qlogis(seq_len(K1) / K)
  par0 <- c(rep(0, p), c(c0[1L], log(pmax(diff(c0), 1e-3))))

  opt <- stats::optim(par0, .ordinal_nll, method = "BFGS", hessian = TRUE,
                      control = list(maxit = max_iter),
                      X = X, cls = cls, K = K, tau_b = tau_b, tau_c = tau_c)

  par  <- opt$par
  beta <- par[seq_len(p)]
  gam  <- par[(p + 1L):(p + K1)]
  cuts <- cumsum(c(gam[1L], exp(gam[-1L])))
  names(beta) <- colnames(X)
  names(cuts) <- paste0(levels(y)[-K], "|", levels(y)[-1L])

  # Laplace marginal from the numerical Hessian of the negative log-joint.
  H <- opt$hessian
  ch <- tryCatch(chol(H), error = function(e) chol(H + diag(1e-6, nrow(H))))
  logdetH <- 2 * sum(log(diag(ch)))
  dim_b <- length(par)
  log_marginal <- -opt$value + 0.5 * dim_b * log(2 * pi) - 0.5 * logdetH

  V <- chol2inv(ch)
  pn <- c(colnames(X), names(cuts))
  dimnames(V) <- list(pn, pn)
  if (!is.null(control$seed)) set.seed(as.integer(control$seed))
  draws <- .ps_rmvnorm(n_draws, setNames(par, pn), V)

  fit <- list(
    coefficients = beta, cutpoints = cuts, vcov = V, draws = draws,
    means = setNames(par, pn), param_names = pn,
    log_marginal = log_marginal, converged = opt$convergence == 0,
    levels = levels(y), n_classes = K,
    family = "ordinal", formula = formula, backend = "ordinal_laplace",
    inference_tier = 2L, inference_mode = "structured", draws_kind = "iid"
  )
  class(fit) <- c("tulpa_ordinal", "tulpa_fit")
  fit
}

#' @export
vcov.tulpa_ordinal <- function(object, ...) object$vcov

#' @export
coef.tulpa_ordinal <- function(object, ...) object$coefficients

#' @export
print.tulpa_ordinal <- function(x, ...) {
  cat(sprintf("Ordinal cumulative logit (%d ordered levels), Laplace fit\n",
              x$n_classes))
  cat(sprintf("log marginal: %.2f\n\nCoefficients:\n", x$log_marginal))
  print(round(x$coefficients, 4))
  cat("\nCutpoints:\n"); print(round(x$cutpoints, 4))
  invisible(x)
}
