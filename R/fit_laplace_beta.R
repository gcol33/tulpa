#' Fit a beta-regression model via Laplace, estimating the precision
#'
#' @description
#' Thin wrapper around [tulpa_laplace()] for `family = "beta"`. The
#' Laplace engine treats `phi` as fixed per fit (same contract as gamma
#' and neg_binomial_2); this wrapper does an outer 1-D optimisation of
#' the Laplace-approximated log-marginal over `phi`, then refits at the
#' optimum to return betas and Hessian.
#'
#' The mean-precision parameterisation is `y ~ Beta(mu * phi, (1 - mu) * phi)`
#' with default logit link; `y` must be strictly in `(0, 1)`.
#'
#' @param y Response in `(0, 1)`.
#' @param X Fixed-effects design matrix.
#' @param re_list,spatial,weights,offset,max_iter,tol,n_threads,beta_prior
#'   Passed to [tulpa_laplace()] verbatim. `beta_prior` places a Gaussian
#'   penalty on the fixed effects (a list with `sd`, optional `mean`; see
#'   [tulpa_laplace()]). It is included in the Laplace log-marginal that the
#'   precision `phi` is optimised against, so the penalised model is fit
#'   consistently across the outer `phi` search. Not supported with `spatial`
#'   (the spatial solver carries its own prior).
#' @param phi_init Optional starting value for the precision. If `NULL`,
#'   a method-of-moments warm start is used.
#' @param phi_bounds Numeric length-2 vector with lower/upper bounds on
#'   `phi` for the outer optimisation. Default `c(0.1, 1e4)`.
#' @param outer_tol Tolerance for the outer optimisation. Default 1e-4.
#'
#' @return The list returned by [tulpa_laplace()] at the optimum,
#'   augmented with `phi` (the optimised precision) and
#'   `phi_log_marginal` (the optimisation trace, for diagnostics).
#'
#' @export
tulpa_laplace_beta <- function(y, X,
                               re_list   = list(),
                               spatial   = NULL,
                               weights   = NULL,
                               offset    = NULL,
                               max_iter  = 100L,
                               tol       = 1e-6,
                               n_threads = 1L,
                               beta_prior = NULL,
                               phi_init   = NULL,
                               phi_bounds = c(0.1, 1e4),
                               outer_tol  = 1e-4) {

  stopifnot(is.numeric(y), is.matrix(X), nrow(X) == length(y))
  if (any(!is.finite(y)) || min(y) <= 0 || max(y) >= 1) {
    stop("`y` must be strictly in (0, 1) for tulpa_laplace_beta().",
         call. = FALSE)
  }
  if (length(phi_bounds) != 2L || phi_bounds[1] <= 0 ||
      phi_bounds[2] <= phi_bounds[1]) {
    stop("`phi_bounds` must be a positive, increasing length-2 vector.",
         call. = FALSE)
  }

  fit_at <- function(phi) {
    tulpa_laplace(
      y = y, n_trials = NULL, X = X, re_list = re_list,
      family = "beta", phi = phi, spatial = spatial,
      weights = weights, offset = offset,
      max_iter = max_iter, tol = tol, n_threads = n_threads,
      return_hessian = FALSE, beta_prior = beta_prior
    )
  }

  if (is.null(phi_init)) {
    # Method-of-moments warm start from the intercept-only mean.
    mu0  <- mean(y)
    v0   <- stats::var(y)
    phi0 <- max(mu0 * (1 - mu0) / max(v0, 1e-8) - 1, 1)
    phi_init <- min(max(phi0, phi_bounds[1] * 1.1), phi_bounds[2] * 0.9)
  }

  trace <- list()
  obj <- function(log_phi) {
    phi <- exp(log_phi)
    fit <- tryCatch(fit_at(phi), error = function(e) NULL)
    lm  <- if (is.null(fit)) -Inf else fit$log_marginal
    trace[[length(trace) + 1L]] <<- c(phi = phi, log_marginal = lm)
    if (!is.finite(lm)) 1e10 else -lm
  }

  op <- stats::optim(
    par     = log(phi_init),
    fn      = obj,
    method  = "Brent",
    lower   = log(phi_bounds[1]),
    upper   = log(phi_bounds[2]),
    control = list(reltol = outer_tol)
  )
  phi_hat <- exp(op$par)

  fit <- tulpa_laplace(
    y = y, n_trials = NULL, X = X, re_list = re_list,
    family = "beta", phi = phi_hat, spatial = spatial,
    weights = weights, offset = offset,
    max_iter = max_iter, tol = tol, n_threads = n_threads,
    return_hessian = TRUE, beta_prior = beta_prior
  )
  fit$phi <- phi_hat
  fit$phi_trace <- do.call(rbind, trace)
  fit$phi_converged <- op$convergence == 0L
  fit
}
