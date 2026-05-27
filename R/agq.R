#' Adaptive Gauss-Hermite quadrature for one-RE GLMMs
#'
#' @description
#' Marginal-likelihood approximation for a generalised linear mixed
#' model with one cluster-level intercept random effect. Adaptive
#' Gauss-Hermite quadrature (AGQ) generalises Laplace by replacing the
#' single-point Gaussian integral around the cluster's posterior mode
#' with `n_quad`-point quadrature. AGQ at `n_quad = 1` recovers the
#' Laplace approximation; higher `n_quad` reduces approximation error,
#' especially for clusters with few observations or non-Gaussian
#' likelihoods.
#'
#' Scoped to the `lme4::glmer(..., nAGQ = N)` use case: one intercept-
#' only RE term, families `binomial`, `poisson`, or `gaussian`. For
#' multi-RE or random-slope models, use [tulpa_laplace()] (Laplace at
#' the joint mode) or HMC.
#'
#' @section Tier:
#' Tier 2 (Structured). AGQ is exact in the limit `n_quad -> infinity`
#' but at finite `n_quad` it is a controlled approximation — same
#' epistemic class as Laplace.
#'
#' @param y Response vector.
#' @param X Fixed-effects design matrix (`n_obs x p`).
#' @param group Integer cluster labels (1-based, length `n_obs`).
#' @param n_groups Number of clusters (default `max(group)`).
#' @param family One of `"binomial"`, `"poisson"`, `"gaussian"`.
#' @param n_trials Trial sizes (binomial only; default `rep(1, n_obs)`).
#' @param sigma_eps Residual SD (gaussian only; default `1`).
#' @param n_quad Number of Gauss-Hermite quadrature nodes per cluster.
#'   `1` recovers Laplace; common choices are `5` or `7`. Default `7`.
#' @param beta_init Initial fixed-effects (default zeros).
#' @param sigma_init Initial RE SD (default `1`).
#' @param max_iter Optimiser iteration cap (default `200`).
#' @param tol Convergence tolerance (default `1e-6`).
#' @param verbose Print optimiser summary (default `FALSE`).
#'
#' @return A list with class `tulpa_fit` carrying:
#'   * `means`: named vector `c(beta, log_sigma_re)`.
#'   * `mode`: same as `means` (Gaussian fit at the optimum).
#'   * `cov`: covariance matrix of `(beta, log_sigma_re)` from inverse
#'     observed information.
#'   * `sigma_re`: estimated RE SD.
#'   * `log_marginal`: AGQ-approximated log marginal likelihood at the
#'     fitted values.
#'   * `n_quad`: quadrature order used.
#'   * `n_iter`, `converged`: optimiser diagnostics.
#'   * `inference_mode`: `"structured"`.
#'   * `inference_tier`: `2L`.
#'   * `backend`: `"agq"`.
#'
#' @references
#' Pinheiro, J. C., & Bates, D. M. (1995). Approximations to the
#' log-likelihood function in the nonlinear mixed-effects model.
#' *Journal of Computational and Graphical Statistics*, 4(1), 12–35.
#'
#' @seealso [tulpa_laplace()] for the joint-mode Laplace path
#'   (handles multi-RE and spatial structure).
#'
#' @examples
#' \dontrun{
#'   set.seed(1)
#'   n_g <- 30; n_per <- 5; n <- n_g * n_per
#'   group <- rep(seq_len(n_g), each = n_per)
#'   x <- rnorm(n)
#'   X <- cbind(1, x)
#'   u <- rnorm(n_g, 0, 0.8)
#'   eta <- 0.3 + 0.7 * x + u[group]
#'   y <- rbinom(n, 1, plogis(eta))
#'
#'   # Compare Laplace (n_quad = 1) and AGQ-7.
#'   fit_lap <- agq_fit(y, X, group, family = "binomial", n_quad = 1)
#'   fit_agq <- agq_fit(y, X, group, family = "binomial", n_quad = 7)
#'   c(fit_lap$log_marginal, fit_agq$log_marginal)
#' }
#'
#' @export
agq_fit <- function(y, X, group,
                    n_groups = max(group),
                    family = c("binomial", "poisson", "gaussian"),
                    n_trials = NULL,
                    sigma_eps = 1.0,
                    n_quad = 7L,
                    beta_init = NULL,
                    sigma_init = 1.0,
                    max_iter = 200L,
                    tol = 1e-6,
                    verbose = FALSE) {

  family <- match.arg(family)
  if (!is.matrix(X)) X <- as.matrix(X)
  storage.mode(X) <- "double"
  n_obs <- length(y)
  p <- ncol(X)
  if (nrow(X) != n_obs) stop("nrow(X) != length(y).", call. = FALSE)
  if (length(group) != n_obs) {
    stop("length(group) != length(y).", call. = FALSE)
  }
  group <- as.integer(group)
  if (any(group < 1L) || any(group > n_groups)) {
    stop("`group` indices must lie in 1..n_groups.", call. = FALSE)
  }
  if (n_quad < 1L) stop("`n_quad` must be >= 1.", call. = FALSE)
  if (is.null(n_trials)) n_trials <- rep(1L, n_obs)
  if (is.null(beta_init)) beta_init <- rep(0, p)

  # Intercept-only RE: route through the shared compiled GLMM oracle, so the
  # family density (binomial / poisson / gaussian) has a single C++ source of
  # truth shared with tulpa_re_aghq() and the Gibbs sweep. The RE design is the
  # 1x1 intercept Z = 1; its covariance is one log-SD coordinate (log-Cholesky
  # of a scalar), so par = c(beta, log_sigma_re) and at n_quad = 1 this is the
  # joint Laplace. Empty groups (no obs) are handled in the oracle: they pay
  # only the N(0, sigma^2) prior. The gaussian residual VARIANCE is phi =
  # sigma_eps^2; binomial / poisson ignore phi.
  Z <- matrix(1, n_obs, 1L)
  orc <- cpp_glmm_oracle_make(family, sigma_eps^2, as.numeric(y),
                              as.numeric(n_trials), X, Z, group, n_groups)
  negf <- function(par)
    -cpp_aghq_objective(par, orc, 1L, FALSE, as.integer(n_quad), 1.0)

  par_init <- c(beta_init, log(sigma_init))
  opt <- stats::optim(par = par_init, fn = negf, method = "BFGS", hessian = TRUE,
                      control = list(maxit = max_iter, reltol = tol))

  sigma_hat <- exp(opt$par[p + 1L])

  cov_par <- tryCatch(solve(opt$hessian),
                      error = function(e) {
                        warning("AGQ Hessian singular; covariance set to NA.",
                                call. = FALSE)
                        matrix(NA_real_, p + 1L, p + 1L)
                      })

  par_names <- c(if (!is.null(colnames(X))) colnames(X) else paste0("beta", seq_len(p)),
                 "log_sigma_re")
  names(opt$par) <- par_names
  if (!any(is.na(cov_par))) {
    rownames(cov_par) <- colnames(cov_par) <- par_names
  }

  if (verbose) {
    cat(sprintf("agq_fit: n_quad = %d, log_marg = %.3f, converged = %s\n",
                n_quad, -opt$value, opt$convergence == 0L))
  }

  fit <- list(
    draws = matrix(NA_real_, 0L, p + 1L,
                   dimnames = list(NULL, par_names)),
    means = opt$par,
    mode = opt$par,
    cov = cov_par,
    sigma_re = sigma_hat,
    log_marginal = -opt$value,
    n_quad = as.integer(n_quad),
    n_iter = as.integer(opt$counts["function"]),
    converged = (opt$convergence == 0L),
    n_params = p + 1L,
    inference_mode = "structured",
    inference_tier = 2L,
    backend = "agq",
    param_names = par_names
  )
  class(fit) <- c("tulpa_agq_fit", "tulpa_fit")
  fit
}


#' Gauss-Hermite quadrature (probabilist's, exp(-z^2/2) weight)
#'
#' Computes nodes and weights for `\int f(z) (2\pi)^{-1/2} exp(-z^2/2) dz`
#' via Golub-Welsch on the Hermite Jacobi matrix. Sums of `w_k f(z_k)`
#' approximate `E_{Z ~ N(0,1)}[f(Z)]`.
#'
#' @param n Number of quadrature nodes.
#' @return List with `nodes` and `weights` (each length `n`).
#' @keywords internal
gauss_hermite_prob <- function(n) {
  n <- as.integer(n)
  if (n < 1L) stop("n must be >= 1.", call. = FALSE)
  if (n == 1L) return(list(nodes = 0, weights = 1))
  # Probabilist's Hermite three-term recurrence:
  #   z H_n = H_{n+1} + n H_{n-1}
  # ⇒ symmetric tridiagonal Jacobi has α_k = 0, β_k = sqrt(k).
  beta <- sqrt(seq_len(n - 1L))
  Jm <- matrix(0, n, n)
  Jm[cbind(seq_len(n - 1L), seq_len(n - 1L) + 1L)] <- beta
  Jm[cbind(seq_len(n - 1L) + 1L, seq_len(n - 1L))] <- beta
  ev <- eigen(Jm, symmetric = TRUE)
  nodes <- ev$values
  # First component squared of each eigenvector × ∫ weight = 1
  # (probabilist's measure normalises to 1).
  weights <- ev$vectors[1L, ]^2
  ord <- order(nodes)
  list(nodes = nodes[ord], weights = weights[ord])
}
