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
#' @param X Fixed-effects design matrix (`n_obs × p`).
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

  # Gauss-Hermite quadrature nodes (probabilist's: ∫ f(z) exp(-z²/2) dz / sqrt(2π))
  gh <- gauss_hermite_prob(n_quad)
  z_nodes <- gh$nodes
  w_nodes <- gh$weights      # weights such that sum_k w_k f(z_k) ≈ E[f(Z)], Z ~ N(0,1)

  neg_log_marg <- function(par) {
    beta <- par[seq_len(p)]
    sigma_re <- exp(par[p + 1L])

    eta_fixed <- as.numeric(X %*% beta)
    log_marg <- 0
    for (g in seq_len(n_groups)) {
      idx <- obs_by_g[[g]]
      eta_g_fixed <- eta_fixed[idx]
      y_g <- y[idx]
      n_g <- if (family == "binomial") n_trials[idx] else NULL

      # Find cluster mode u_hat_g via 1-D Newton (cheap; max ~5 iter).
      mode_info <- find_cluster_mode(eta_g_fixed, y_g, n_g, sigma_re,
                                     family, sigma_eps,
                                     max_iter = 50L, tol = 1e-8)
      u_hat <- mode_info$u_hat
      s_g <- 1 / mode_info$neg_hess   # posterior cluster variance

      # AGQ: u_g_k = u_hat + sqrt(s_g) * z_k, weight rescaled.
      sd_g <- sqrt(s_g)
      log_int_terms <- numeric(n_quad)
      for (k in seq_len(n_quad)) {
        u_k <- u_hat + sd_g * z_nodes[k]
        log_lik_k <- cond_log_lik(eta_g_fixed + u_k, y_g, n_g,
                                  family, sigma_eps)
        log_prior_k <- dnorm(u_k, 0, sigma_re, log = TRUE)
        # Probabilist's weight w_k already integrates against N(0,1);
        # transform from N(0,1) on z to N(u_hat, s_g) on u contributes
        # the extra sqrt(s_g) * sqrt(2π) factor in the log:
        log_int_terms[k] <- log(w_nodes[k]) + log_lik_k + log_prior_k +
          0.5 * z_nodes[k]^2 + log(sd_g) + 0.5 * log(2 * pi)
      }
      log_marg <- log_marg + logsumexp(log_int_terms)
    }
    -log_marg
  }

  par_init <- c(beta_init, log(sigma_init))
  opt <- stats::optim(
    par = par_init, fn = neg_log_marg,
    method = "BFGS",
    hessian = TRUE,
    control = list(maxit = max_iter, reltol = tol)
  )

  beta_hat <- opt$par[seq_len(p)]
  log_sigma_hat <- opt$par[p + 1L]
  sigma_hat <- exp(log_sigma_hat)

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


#' Per-cluster conditional log-likelihood evaluator
#' @keywords internal
cond_log_lik <- function(eta, y, n_trials, family, sigma_eps) {
  if (length(y) == 0L) return(0)
  if (family == "binomial") {
    # log Bernoulli/binomial pmf via log-sum-exp form:
    # y * eta - n * log(1 + exp(eta)), plus the binomial coefficient.
    # We can drop the binomial coefficient since it is constant in (beta, u).
    sum(y * eta - n_trials * log1pexp(eta))
  } else if (family == "poisson") {
    # y * eta - exp(eta) - lgamma(y + 1) — drop lgamma (const in eta).
    sum(y * eta - exp(eta))
  } else if (family == "gaussian") {
    sum(dnorm(y, eta, sigma_eps, log = TRUE))
  } else {
    stop("Unsupported family in AGQ: ", family, call. = FALSE)
  }
}


#' Stable log(1 + exp(x))
#' @keywords internal
log1pexp <- function(x) {
  ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))
}


#' Find cluster random-effect mode + curvature via 1-D Newton
#' @keywords internal
find_cluster_mode <- function(eta_fixed, y, n_trials, sigma_re,
                              family, sigma_eps,
                              max_iter = 50L, tol = 1e-8) {
  u <- 0
  for (iter in seq_len(max_iter)) {
    eta <- eta_fixed + u
    grad_lik <- if (length(y) == 0L) 0 else {
      if (family == "binomial") {
        mu <- 1 / (1 + exp(-eta))
        sum(y - n_trials * mu)
      } else if (family == "poisson") {
        sum(y - exp(eta))
      } else if (family == "gaussian") {
        sum(y - eta) / sigma_eps^2
      }
    }
    neg_hess_lik <- if (length(y) == 0L) 0 else {
      if (family == "binomial") {
        mu <- 1 / (1 + exp(-eta))
        sum(n_trials * mu * (1 - mu))
      } else if (family == "poisson") {
        sum(exp(eta))
      } else if (family == "gaussian") {
        length(y) / sigma_eps^2
      }
    }
    grad <- grad_lik - u / sigma_re^2
    neg_hess <- neg_hess_lik + 1 / sigma_re^2
    step <- grad / neg_hess
    u_new <- u + step
    if (abs(step) < tol) {
      u <- u_new
      break
    }
    u <- u_new
  }
  list(u_hat = u, neg_hess = neg_hess)
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
