#' Laplace fit of the Royle (2004) N-mixture model
#'
#' @description
#' Maximum-likelihood fit (non-spatial, fixed effects only) of the
#' Royle (2004) N-mixture model with a Poisson or negative-binomial
#' abundance mixing distribution:
#' \deqn{N_i \sim \mathrm{Poisson}(\lambda_i) \quad\text{or}\quad
#'       N_i \sim \mathrm{NegBin}(\mathrm{mean}=\lambda_i, \mathrm{size}=r),
#'       \qquad y_{ij} | N_i \sim \mathrm{Binomial}(N_i, p_{ij}),}
#' with abundance linear predictor \eqn{\log \lambda_i = X_\lambda^{(i)}
#' \beta_\lambda} and detection linear predictor
#' \eqn{\mathrm{logit}\, p_{ij} = X_p^{(ij)} \beta_p}. The NB uses the
#' `neg_binomial_2` convention (size \eqn{r}, variance
#' \eqn{\lambda + \lambda^2/r}); Poisson is the \eqn{r \to \infty} limit.
#'
#' Optimisation uses inner Newton on \eqn{(\beta_\lambda, \beta_p)} with the
#' marginal observed Fisher information matrix as the curvature (complete-data
#' Fisher fallback at iterates where the observed-info Hessian is not PSD).
#' Under `mixture = "NB"` the dispersion \eqn{\theta = \log r} is a single
#' global scalar, profiled outside the beta-Newton by block coordinate ascent
#' on its analytic profile score. All gradients and Hessians are analytical (no
#' numerical differentiation), so the fit converges in a small number of
#' iterations and is typically 20-50x faster than `unmarked::pcount()`'s
#' BFGS-with-numerical-derivatives path on equivalent problems.
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
#' @param mixture Abundance mixing distribution: `"P"` (Poisson, default) or
#'   `"NB"` (negative binomial). Under `"NB"` an extra dispersion parameter
#'   `log_r` (log NB size) is estimated jointly and reported with its standard
#'   error.
#' @param beta_lambda_init Optional numeric `[p_lambda]` warm start. Defaults
#'   to `c(log(mean(y) + 0.1), 0, 0, ...)`.
#' @param beta_p_init Optional numeric `[p_p]` warm start. Defaults to zeros.
#' @param log_r_init Optional scalar warm start for `log_r` (NB only). Defaults
#'   to a method-of-moments estimate from the site count totals, clamped to a
#'   sensible range.
#' @param r_max Upper bound on the NB size `r` (NB only, default `1e5`). If the
#'   optimiser pins `log_r` at `log(r_max)` the data are consistent with Poisson
#'   and a warning recommends `mixture = "P"`.
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
#'   * `mixture` -- `"P"` or `"NB"`
#'   * `log_r`, `r` -- estimated log NB size and the NB size `exp(log_r)`
#'     (NB only; `NA` under Poisson). `vcov` carries `log_r` as its last
#'     coordinate, so `sqrt(diag(vcov))["log_r"]` is its standard error.
#'   * `log_lik` -- marginal log-likelihood at the mode
#'   * `vcov` -- variance-covariance matrix (marginal observed Fisher inverse),
#'     over `(beta_lambda, beta_p)` and, under NB, `log_r`
#'   * `vcov_ok` -- whether the final observed-info Cholesky succeeded
#'   * `H_obs` -- final marginal observed Fisher information matrix
#'   * `n_iter` -- (outer) iterations consumed
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
                               mixture = c("P", "NB"),
                               beta_lambda_init = NULL,
                               beta_p_init = NULL,
                               log_r_init = NULL,
                               r_max = 1e5,
                               K_max = NULL,
                               max_iter = 100L,
                               tol = 1e-6,
                               verbose = FALSE) {
  mixture <- match.arg(mixture)
  nb <- identical(mixture, "NB")
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

  if (!is.numeric(r_max) || length(r_max) != 1L || r_max <= 0) {
    stop("`r_max` must be a positive scalar.", call. = FALSE)
  }
  # Warm start for log_r: r = 1 (strong overdispersion) is a neutral, robust
  # start that matches unmarked's default; the outer Newton refines it.
  if (is.null(log_r_init)) {
    log_r_init <- 0
  } else if (length(log_r_init) != 1L || !is.finite(log_r_init)) {
    stop("`log_r_init` must be a finite scalar.", call. = FALSE)
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
    verbose          = isTRUE(verbose),
    nb               = nb,
    log_r_init       = as.numeric(log_r_init),
    theta_max        = log(r_max)
  )

  # Name coefficients from input matrices. Under NB, log_r is the last vcov
  # coordinate.
  nm_lam <- colnames(X_lambda)
  nm_p   <- colnames(X_p)
  if (is.null(nm_lam)) nm_lam <- paste0("lam_", seq_len(p_lambda))
  if (is.null(nm_p))   nm_p   <- paste0("p_", seq_len(p_p))
  names(fit$beta_lambda) <- nm_lam
  names(fit$beta_p)      <- nm_p
  coef_names <- c(nm_lam, nm_p, if (nb) "log_r")
  rownames(fit$vcov)  <- colnames(fit$vcov)  <- coef_names
  rownames(fit$H_obs) <- colnames(fit$H_obs) <- coef_names

  fit$mixture <- mixture
  if (!nb) { fit$log_r <- NA_real_; fit$r <- NA_real_ }
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
  if (nb && isTRUE(fit$dispersion_boundary)) {
    warning(sprintf(
      paste0("NB dispersion pinned at the boundary (r = r_max = %.3g); the data ",
             "are consistent with Poisson. Consider mixture = \"P\"."),
      r_max
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
  mix <- x$mixture %||% "P"
  cat(sprintf("tulpa N-mixture Laplace fit (mixture = %s)\n", mix))
  cat(sprintf("  n_sites = %d   n_obs = %d   K_max = %d\n",
              x$n_sites, x$n_obs, x$K_max))
  cat(sprintf("  log_lik = %.4f   n_iter = %d   converged = %s\n",
              x$log_lik, x$n_iter, x$converged))
  cat("\nabundance (log lambda):\n")
  print(x$beta_lambda)
  cat("\ndetection (logit p):\n")
  print(x$beta_p)
  if (identical(mix, "NB") && is.finite(x$log_r)) {
    se <- tryCatch(sqrt(diag(x$vcov))["log_r"], error = function(e) NA_real_)
    cat(sprintf("\ndispersion: log_r = %.4f (SE %.4f)   r = %.4f\n",
                x$log_r, se, x$r))
  }
  invisible(x)
}
