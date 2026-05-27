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

  # Pre-split observations by cluster — one closure capture, reused
  # at every optim iteration.
  obs_by_g <- split(seq_len(n_obs), group)
  if (length(obs_by_g) < n_groups) {
    # Pad empty groups (no obs) — they contribute log p(y_g | u_g) = 0
    # but still pay the prior log N(u_g | 0, sigma^2).
    for (g in seq_len(n_groups)) {
      if (is.null(obs_by_g[[as.character(g)]])) {
        obs_by_g[[as.character(g)]] <- integer(0L)
      }
    }
    obs_by_g <- obs_by_g[as.character(seq_len(n_groups))]
  }

  # Intercept-only RE: route through the shared compiled AGHQ engine. `theta` is
  # the fixed effects beta; the 1x1 RE covariance is one log-SD coordinate
  # (log-Cholesky of a scalar), so par = c(beta, log_sigma_re). At n_quad = 1
  # this is the joint Laplace. The family likelihood is the only model-specific
  # piece and lives in the per-group builder (Z = 1 intercept).
  builder <- function(beta) {
    eta_fixed <- as.numeric(X %*% beta)
    list(
      grad_hess = function(g, b) {
        rows <- obs_by_g[[g]]
        if (length(rows) == 0L)
          return(list(logL = 0, grad = 0, negH = matrix(0, 1L, 1L)))
        eta <- eta_fixed[rows] + b[1L]
        ntg <- if (family == "binomial") n_trials[rows] else NULL
        si  <- .agq_score_info(eta, y[rows], ntg, family, sigma_eps)
        list(logL = sum(.agq_loglik_elt(eta, y[rows], ntg, family, sigma_eps)),
             grad = sum(si$d1),
             negH = matrix(-sum(si$d2), 1L, 1L))
      },
      node_ll = function(g, B) {
        rows <- obs_by_g[[g]]
        if (length(rows) == 0L) return(rep(0, nrow(B)))
        ntg <- if (family == "binomial") n_trials[rows] else NULL
        ETA <- eta_fixed[rows] + matrix(B[, 1L], length(rows), nrow(B), byrow = TRUE)
        colSums(.agq_loglik_elt(ETA, y[rows], ntg, family, sigma_eps))
      })
  }

  orc <- cpp_aghq_make_rclosure_oracle(builder, n_groups, 1L, p)
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


#' Per-observation conditional log-likelihood at a linear predictor
#'
#' Elementwise log-likelihood contribution (works on a vector or a matrix of
#' `eta`, so the quadrature-node sweep reuses it). Drops the additive constants
#' that do not depend on `eta` (binomial coefficient, Poisson `lgamma`).
#' @keywords internal
.agq_loglik_elt <- function(eta, y, n_trials, family, sigma_eps) {
  if (family == "binomial") {
    y * eta - n_trials * log1pexp(eta)
  } else if (family == "poisson") {
    y * eta - exp(eta)
  } else if (family == "gaussian") {
    # Explicit form (not dnorm) so a matrix `eta` keeps its dimensions.
    -0.5 * log(2 * pi) - log(sigma_eps) - 0.5 * ((y - eta) / sigma_eps)^2
  } else {
    stop("Unsupported family in AGQ: ", family, call. = FALSE)
  }
}

#' Per-observation score and observed information wrt the linear predictor
#'
#' First and second `eta`-derivatives of `.agq_loglik_elt()` (`d2` is the second
#' derivative, i.e. minus the observed information). Drives the per-group mode.
#' @keywords internal
.agq_score_info <- function(eta, y, n_trials, family, sigma_eps) {
  if (family == "binomial") {
    mu <- 1 / (1 + exp(-eta))
    list(d1 = y - n_trials * mu, d2 = -n_trials * mu * (1 - mu))
  } else if (family == "poisson") {
    lam <- exp(eta)
    list(d1 = y - lam, d2 = -lam)
  } else if (family == "gaussian") {
    list(d1 = (y - eta) / sigma_eps^2, d2 = rep(-1 / sigma_eps^2, length(eta)))
  } else {
    stop("Unsupported family in AGQ: ", family, call. = FALSE)
  }
}


#' Stable log(1 + exp(x))
#' @keywords internal
log1pexp <- function(x) {
  ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))
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


#' Numerically stable log-sum-exp on a vector
#' @keywords internal
logsumexp <- function(x) {
  m <- max(x)
  m + log(sum(exp(x - m)))
}
