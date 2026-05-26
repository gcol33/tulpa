# Analytic-derivative validation for the negative-binomial N-mixture kernel.
#
# These complement the statistical NB tests in test-nmix-laplace.R (convergence,
# multi-seed recovery, unmarked cross-check). Here we check the kernel math
# directly against finite differences -- the part most prone to silent error and
# the part the unmarked comparison only probes indirectly:
#   1. the analytic theta = log r score vs a central difference of the marginal
#      log-likelihood;
#   2. the assembled joint observed information H_obs (which carries the theta
#      row/col and the lambda<->theta cross-term) vs a numerical Hessian of the
#      analytic gradient at the fitted mode.
# Both are self-contained (no unmarked, no parameterization assumptions).

simulate_nmix_nb <- function(seed,
                             n_sites = 300,
                             J = 6,
                             beta_lambda = c(log(6), 0.4),
                             beta_p = c(qlogis(0.5), -0.3),
                             r_true = 3) {
  set.seed(seed)
  elev <- rnorm(n_sites)
  wind <- matrix(rnorm(n_sites * J), n_sites, J)
  X_lambda <- cbind(intercept = 1, elev = elev)
  lambda   <- exp(as.vector(X_lambda %*% beta_lambda))
  X_p <- cbind(intercept = 1, wind = as.vector(t(wind)))
  site_idx <- as.integer(rep(seq_len(n_sites), each = J))
  eta_p_mat <- matrix(0, n_sites, J)
  for (s in seq_len(n_sites)) eta_p_mat[s, ] <- beta_p[1] + beta_p[2] * wind[s, ]
  p_mat <- plogis(eta_p_mat)
  N <- rnbinom(n_sites, mu = lambda, size = r_true)
  y_mat <- matrix(0L, n_sites, J)
  for (s in seq_len(n_sites)) for (j in seq_len(J)) {
    y_mat[s, j] <- rbinom(1, N[s], p_mat[s, j])
  }
  list(
    y = as.integer(as.vector(t(y_mat))), site_idx = site_idx,
    X_lambda = X_lambda, X_p = X_p,
    beta_lambda_true = beta_lambda, beta_p_true = beta_p,
    r_true = r_true, log_r_true = log(r_true)
  )
}

test_that("analytic theta score matches a central difference of the marginal log-lik", {
  dat <- simulate_nmix_nb(seed = 11)
  Xl <- dat$X_lambda; Xp <- dat$X_p
  # Arbitrary (non-mode) parameter point so the score is well away from 0.
  bl <- c(log(5), 0.3); bp <- c(qlogis(0.45), -0.25); theta <- log(2.5)
  eta_lam <- as.vector(Xl %*% bl)
  eta_p   <- as.vector(Xp %*% bp)
  K_max   <- max(dat$y) + 100L

  ll <- function(th) {
    tulpa:::cpp_nmix_total_log_lik(
      dat$y, dat$site_idx, eta_p, eta_lam, K_max, r = exp(th)
    )$log_lik
  }
  res <- tulpa:::cpp_nmix_total_log_lik(
    dat$y, dat$site_idx, eta_p, eta_lam, K_max, r = exp(theta)
  )
  h <- 1e-5
  num_grad <- (ll(theta + h) - ll(theta - h)) / (2 * h)
  expect_lt(abs(res$grad_theta - num_grad), 1e-4)

  # Poisson limit (r = Inf): the dispersion score is identically 0.
  res_pois <- tulpa:::cpp_nmix_total_log_lik(
    dat$y, dat$site_idx, eta_p, eta_lam, K_max, r = Inf
  )
  expect_identical(res_pois$grad_theta, 0)
})

test_that("joint H_obs equals the numerical Hessian of the analytic gradient at the mode", {
  dat <- simulate_nmix_nb(seed = 13)
  fit <- tulpa_nmix_laplace(
    y = dat$y, site_idx = dat$site_idx,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    mixture = "NB", max_iter = 80L, tol = 1e-8
  )
  skip_if_not(fit$converged && fit$vcov_ok && !isTRUE(fit$dispersion_boundary))

  Xl <- dat$X_lambda; Xp <- dat$X_p
  K_max <- fit$K_max
  p_lam <- ncol(Xl); p_p <- ncol(Xp)

  # Analytic gradient wrt (beta_lambda, beta_p, theta) via the kernel's first
  # derivatives, mapped through the design matrices.
  grad_at <- function(par) {
    bl <- par[seq_len(p_lam)]
    bp <- par[p_lam + seq_len(p_p)]
    th <- par[p_lam + p_p + 1L]
    res <- tulpa:::cpp_nmix_total_log_lik(
      dat$y, dat$site_idx,
      as.vector(Xp %*% bp), as.vector(Xl %*% bl), K_max, r = exp(th)
    )
    c(as.vector(crossprod(Xl, res$grad_eta_lambda)),
      as.vector(crossprod(Xp, res$grad_eta_p)),
      res$grad_theta)
  }

  par_hat <- c(unname(fit$beta_lambda), unname(fit$beta_p), fit$log_r)
  expect_lt(max(abs(grad_at(par_hat))), 1e-4)   # gradient ~ 0 at the mode

  # H_obs is the observed Fisher information = -Hessian(logL); the central
  # difference of the analytic gradient gives that Hessian directly.
  np <- length(par_hat); h <- 1e-4
  num_hess <- matrix(0, np, np)
  for (j in seq_len(np)) {
    pp <- par_hat; pp[j] <- pp[j] + h
    pm <- par_hat; pm[j] <- pm[j] - h
    num_hess[, j] <- (grad_at(pp) - grad_at(pm)) / (2 * h)
  }
  num_obs_info <- -(num_hess + t(num_hess)) / 2   # symmetrize
  H <- unname(fit$H_obs)
  expect_equal(dim(H), c(np, np))
  # Entries span orders of magnitude across the n_sites-weighted blocks; scale
  # the tolerance per entry.
  expect_lt(max(abs(H - num_obs_info) / pmax(abs(H), 1)), 5e-3)
})
