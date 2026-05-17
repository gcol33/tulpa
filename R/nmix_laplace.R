#' Laplace fit of the Royle (2004) N-mixture model
#'
#' @description
#' Maximum-likelihood fit (non-spatial, fixed effects only) of the
#' Royle (2004) N-mixture model
#' \deqn{N_i \sim \mathrm{Poisson}(\lambda_i), \qquad
#'       y_{ij} | N_i \sim \mathrm{Binomial}(N_i, p_{ij}),}
#' with abundance linear predictor \eqn{\log \lambda_i = X_\lambda^{(i)}
#' \beta_\lambda} and detection linear predictor
#' \eqn{\mathrm{logit}\, p_{ij} = X_p^{(ij)} \beta_p}.
#'
#' Optimisation uses inner Newton with the marginal observed Fisher
#' information matrix as the curvature, with a complete-data Fisher
#' fallback at iterates where the observed-info Hessian is not PSD. Both
#' gradients and Hessian are analytical (no numerical differentiation),
#' so the fit converges in a small number of iterations and is typically
#' 20-50x faster than `unmarked::pcount()`'s BFGS-with-numerical-derivatives
#' path on equivalent problems.
#'
#' This is the non-spatial entry point; a spatial nested-Laplace path
#' that places a latent prior block on the abundance arm is a separate
#' entry, to be added (see `tulpa_nested_laplace_nmix()` once shipped).
#'
#' @param y Integer vector of observed counts, one entry per visit (long form).
#' @param site_idx Integer vector, same length as `y`, 1-based site index
#'   indicating which site each visit belongs to.
#' @param X_lambda Numeric matrix `[n_sites x p_lambda]` of abundance covariates.
#' @param X_p Numeric matrix `[n_obs x p_p]` of detection covariates (long form,
#'   row order matches `y`).
#' @param beta_lambda_init Optional numeric `[p_lambda]` warm start. Defaults
#'   to `c(log(mean(y) + 0.1), 0, 0, ...)`.
#' @param beta_p_init Optional numeric `[p_p]` warm start. Defaults to zeros.
#' @param K_max Marginal-sum truncation. Defaults to `max(y) + 100` (matches
#'   `unmarked::pcount`). The returned `boundary_weight` per site flags any
#'   site whose posterior over N puts non-trivial mass on `K_max`; raise
#'   `K_max` if any such weight exceeds ~1e-4.
#' @param max_iter Newton iteration budget (default 100).
#' @param tol Gradient-norm convergence tolerance (default 1e-6).
#' @param verbose Print per-iteration `(log_lik, grad_norm, boundary_max)`.
#'
#' @return A list of class `tulpa_nmix_fit`:
#'   * `beta_lambda`, `beta_p` -- MLE coefficient vectors
#'   * `log_lik` -- marginal log-likelihood at the mode
#'   * `vcov` -- variance-covariance matrix (marginal observed Fisher inverse)
#'   * `vcov_ok` -- whether the final observed-info Cholesky succeeded
#'   * `H_obs` -- final marginal observed Fisher information matrix
#'   * `n_iter` -- Newton iterations consumed
#'   * `converged` -- whether the convergence tolerance was reached
#'   * `grad_norm` -- gradient norm at termination
#'   * `mean_N`, `var_N` -- per-site posterior mean and variance of N | y
#'   * `boundary_weight` -- per-site posterior weight on `N = K_max`
#'   * `call` -- matched call
#'
#' @references
#' Royle, J. A. (2004). N-mixture models for estimating population size from
#'   spatially replicated counts. *Biometrics* 60, 108-115.
#' Dennis, E. B., Morgan, B. J. T., Ridout, M. S. (2015). Computational aspects
#'   of N-mixture models. *Biometrics* 71, 237-246.
#'
#' @export
tulpa_nmix_laplace <- function(y,
                               site_idx,
                               X_lambda,
                               X_p,
                               beta_lambda_init = NULL,
                               beta_p_init = NULL,
                               K_max = NULL,
                               max_iter = 100L,
                               tol = 1e-6,
                               verbose = FALSE) {
  y        <- as.integer(y)
  site_idx <- as.integer(site_idx)
  if (!is.matrix(X_lambda)) stop("`X_lambda` must be a numeric matrix.", call. = FALSE)
  if (!is.matrix(X_p))      stop("`X_p` must be a numeric matrix.", call. = FALSE)
  if (length(y) != length(site_idx)) {
    stop("length(y) must equal length(site_idx).", call. = FALSE)
  }
  if (length(y) != nrow(X_p)) {
    stop("length(y) must equal nrow(X_p).", call. = FALSE)
  }
  if (any(y < 0L) || anyNA(y)) {
    stop("`y` must be nonnegative integers with no NA.", call. = FALSE)
  }
  n_sites <- nrow(X_lambda)
  if (min(site_idx) < 1L || max(site_idx) > n_sites) {
    stop("`site_idx` values must lie in [1, nrow(X_lambda)].", call. = FALSE)
  }

  p_lambda <- ncol(X_lambda)
  p_p      <- ncol(X_p)
  if (is.null(beta_lambda_init)) {
    beta_lambda_init <- c(log(mean(y) + 0.1), rep(0, p_lambda - 1L))
  }
  if (is.null(beta_p_init)) {
    beta_p_init <- rep(0, p_p)
  }
  if (length(beta_lambda_init) != p_lambda) {
    stop("length(beta_lambda_init) must equal ncol(X_lambda).", call. = FALSE)
  }
  if (length(beta_p_init) != p_p) {
    stop("length(beta_p_init) must equal ncol(X_p).", call. = FALSE)
  }

  if (is.null(K_max)) {
    K_max <- as.integer(max(y) + 100L)
  } else {
    K_max <- as.integer(K_max)
    if (K_max < max(y)) {
      stop("`K_max` must be >= max(y).", call. = FALSE)
    }
  }

  fit <- cpp_nmix_laplace_fixed(
    y                = y,
    site_idx         = site_idx,
    X_lambda_R       = X_lambda,
    X_p_R            = X_p,
    beta_lambda_init = as.numeric(beta_lambda_init),
    beta_p_init      = as.numeric(beta_p_init),
    K_max            = K_max,
    max_iter         = as.integer(max_iter),
    tol              = as.numeric(tol),
    verbose          = isTRUE(verbose)
  )

  # Name coefficients from input matrices.
  nm_lam <- colnames(X_lambda)
  nm_p   <- colnames(X_p)
  if (is.null(nm_lam)) nm_lam <- paste0("lam_", seq_len(p_lambda))
  if (is.null(nm_p))   nm_p   <- paste0("p_", seq_len(p_p))
  names(fit$beta_lambda) <- nm_lam
  names(fit$beta_p)      <- nm_p
  rownames(fit$vcov) <- colnames(fit$vcov) <- c(nm_lam, nm_p)
  rownames(fit$H_obs) <- colnames(fit$H_obs) <- c(nm_lam, nm_p)

  fit$K_max <- K_max
  fit$n_sites <- n_sites
  fit$n_obs <- length(y)
  fit$call <- match.call()
  if (!fit$converged) {
    warning(sprintf(
      "tulpa_nmix_laplace did not converge in %d iterations (grad_norm = %.2e).",
      max_iter, fit$grad_norm
    ), call. = FALSE)
  }
  max_bw <- max(fit$boundary_weight, na.rm = TRUE)
  if (is.finite(max_bw) && max_bw > 1e-4) {
    warning(sprintf(
      "Max posterior weight on N = K_max is %.2e at %d sites; raise K_max.",
      max_bw, sum(fit$boundary_weight > 1e-4)
    ), call. = FALSE)
  }
  class(fit) <- c("tulpa_nmix_fit", "list")
  fit
}

#' @export
print.tulpa_nmix_fit <- function(x, ...) {
  cat("tulpa N-mixture Laplace fit\n")
  cat(sprintf("  n_sites = %d   n_obs = %d   K_max = %d\n",
              x$n_sites, x$n_obs, x$K_max))
  cat(sprintf("  log_lik = %.4f   n_iter = %d   converged = %s\n",
              x$log_lik, x$n_iter, x$converged))
  cat("\nabundance (log lambda):\n")
  print(x$beta_lambda)
  cat("\ndetection (logit p):\n")
  print(x$beta_p)
  invisible(x)
}
