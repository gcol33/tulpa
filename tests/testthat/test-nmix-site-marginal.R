# Correctness of tulpa_nmix_site_marginal() -- the composable per-site N-mixture
# marginal callback that a grouped random-effect integrator consumes.
#
# These are derivative-level checks (the part most prone to silent error and the
# one a downstream recovery test only probes indirectly):
#   1. the marginal value agrees with the fixed-effects fitter at the same betas;
#   2. the eta-level gradients (both arms) match central finite differences;
#   3. the per-site observed-information block -- including the abundance /
#      detection coupling and (NB) the dispersion row -- matches the numerical
#      Hessian of the per-site marginal log-likelihood.
# All self-contained (no unmarked, no parameterization assumptions).

simulate_nmix_panel <- function(seed, mixture = "P",
                                 n_sites = 80, J = 4,
                                 beta_lambda = c(log(5), 0.5),
                                 beta_p = c(qlogis(0.45), -0.4),
                                 r_true = 4) {
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
  N <- if (identical(mixture, "NB")) rnbinom(n_sites, mu = lambda, size = r_true)
       else rpois(n_sites, lambda)
  y_mat <- matrix(0L, n_sites, J)
  for (s in seq_len(n_sites)) for (j in seq_len(J)) {
    y_mat[s, j] <- rbinom(1, N[s], p_mat[s, j])
  }
  list(
    y = as.integer(as.vector(t(y_mat))), site_idx = site_idx,
    X_lambda = X_lambda, X_p = X_p,
    beta_lambda = beta_lambda, beta_p = beta_p, r_true = r_true
  )
}

test_that("marginal value matches the fixed-effects fitter at the same betas (Poisson + NB)", {
  for (mix in c("P", "NB")) {
    dat <- simulate_nmix_panel(seed = 101, mixture = mix)
    fit <- tulpa_nmix_laplace(
      y = dat$y, site_idx = dat$site_idx,
      X_lambda = dat$X_lambda, X_p = dat$X_p,
      mixture = mix, max_iter = 80L, tol = 1e-8
    )
    skip_if_not(fit$converged)
    marg <- tulpa_nmix_site_marginal(
      y = dat$y, site_idx = dat$site_idx,
      X_lambda = dat$X_lambda, X_p = dat$X_p,
      mixture = mix, K_max = fit$K_max
    )
    r_fit <- if (mix == "NB") fit$r else Inf
    ev <- marg$eval_beta(fit$beta_lambda, fit$beta_p, r = r_fit)
    expect_equal(ev$log_lik, fit$log_lik, tolerance = 1e-8)
    # Per-site contributions sum to the total.
    expect_equal(sum(ev$log_lik_site), ev$log_lik, tolerance = 1e-10)
    # At the mode the beta-gradient (X' eta-grad) is ~ 0.
    g_lam <- as.vector(crossprod(dat$X_lambda, ev$grad_eta_lambda))
    g_p   <- as.vector(crossprod(dat$X_p, ev$grad_eta_p))
    expect_lt(max(abs(c(g_lam, g_p))), 1e-3)
  }
})

test_that("eta-gradients match central finite differences on both arms (Poisson + NB)", {
  for (mix in c("P", "NB")) {
    dat <- simulate_nmix_panel(seed = 202, mixture = mix, n_sites = 30, J = 3)
    marg <- tulpa_nmix_site_marginal(
      y = dat$y, site_idx = dat$site_idx,
      X_lambda = dat$X_lambda, X_p = dat$X_p, mixture = mix
    )
    r_use <- if (mix == "NB") 3.2 else Inf
    # Off-mode predictors so the gradient is well away from 0.
    eta_lambda <- as.vector(dat$X_lambda %*% c(log(4), 0.3))
    eta_p      <- as.vector(dat$X_p %*% c(qlogis(0.5), -0.2))
    ev <- marg$eval(eta_lambda, eta_p, r = r_use)
    h <- 1e-5
    ll <- function(el, ep) marg$eval(el, ep, r = r_use)$log_lik

    # Abundance arm: perturb a few sites.
    for (s in c(1L, 7L, 15L)) {
      ep <- eta_lambda; ep[s] <- ep[s] + h
      em <- eta_lambda; em[s] <- em[s] - h
      num <- (ll(ep, eta_p) - ll(em, eta_p)) / (2 * h)
      expect_lt(abs(ev$grad_eta_lambda[s] - num), 1e-4)
    }
    # Detection arm: perturb a few visits.
    for (o in c(2L, 11L, 25L)) {
      ep <- eta_p; ep[o] <- ep[o] + h
      em <- eta_p; em[o] <- em[o] - h
      num <- (ll(eta_lambda, ep) - ll(eta_lambda, em)) / (2 * h)
      expect_lt(abs(ev$grad_eta_p[o] - num), 1e-4)
    }
  }
})

test_that("per-site observed-info block matches the numerical Hessian (Poisson + NB)", {
  for (mix in c("P", "NB")) {
    dat <- simulate_nmix_panel(seed = 303, mixture = mix, n_sites = 40, J = 4)
    # Isolate one site with a non-trivial count history into a single-site
    # marginal, so the total log-lik IS that site's marginal and we can
    # differentiate its eta vector directly.
    s_pick <- which(tapply(dat$y, dat$site_idx, max) >= 2)[1]
    obs <- which(dat$site_idx == s_pick)
    J <- length(obs)
    marg1 <- tulpa_nmix_site_marginal(
      y = dat$y[obs], site_idx = rep(1L, J),
      X_lambda = dat$X_lambda[s_pick, , drop = FALSE],
      X_p = dat$X_p[obs, , drop = FALSE], mixture = mix
    )
    r_use <- if (mix == "NB") 3.0 else Inf
    eta_lambda <- log(3.5)
    eta_p <- qlogis(seq(0.35, 0.6, length.out = J))
    ev <- marg1$eval(eta_lambda, eta_p, r = r_use)
    B  <- marg1$obs_info_block(1L, ev)          # analytic observed info (-Hessian)
    expect_equal(dim(B), c(1L + J, 1L + J))

    # Numerical Hessian of the per-site log-lik over (eta_lambda, eta_p).
    par <- c(eta_lambda, eta_p)
    llf <- function(v) marg1$eval(v[1], v[-1], r = r_use)$log_lik
    d <- length(par); h <- 1e-4
    H <- matrix(0, d, d)
    for (i in seq_len(d)) for (j in seq_len(d)) {
      pp <- par; pp[i] <- pp[i] + h; pp[j] <- pp[j] + h
      pm <- par; pm[i] <- pm[i] + h; pm[j] <- pm[j] - h
      mp <- par; mp[i] <- mp[i] - h; mp[j] <- mp[j] + h
      mm <- par; mm[i] <- mm[i] - h; mm[j] <- mm[j] - h
      H[i, j] <- (llf(pp) - llf(pm) - llf(mp) + llf(mm)) / (4 * h * h)
    }
    num_obs_info <- -(H + t(H)) / 2
    expect_lt(max(abs(B - num_obs_info)), 5e-3)
    # The abundance/detection coupling (off-diagonal first row) is non-zero.
    expect_gt(max(abs(B[1, -1])), 1e-6)
  }
})

test_that("edge cases: empty-visit site and inadmissible K_max", {
  dat <- simulate_nmix_panel(seed = 404, mixture = "P", n_sites = 10, J = 3)
  # Drop every visit of site 5 -> that site has no data.
  obs5 <- which(dat$site_idx == 5L)
  keep <- setdiff(seq_along(dat$y), obs5)
  marg0 <- tulpa_nmix_site_marginal(
    y = dat$y[keep], site_idx = dat$site_idx[keep],
    X_lambda = dat$X_lambda, X_p = dat$X_p[keep, , drop = FALSE],
    mixture = "P"
  )
  eta_lambda <- as.vector(dat$X_lambda %*% c(log(4), 0.2))
  eta_p <- as.vector(dat$X_p[keep, ] %*% c(qlogis(0.5), -0.1))
  ev <- marg0$eval(eta_lambda, eta_p)
  expect_equal(ev$log_lik_site[5], 0)            # no-visit site contributes 0
  expect_equal(length(marg0$obs_by_site[[5]]), 0L)
  B5 <- marg0$obs_info_block(5L, ev)
  expect_equal(dim(B5), c(1L, 1L))
  expect_equal(as.numeric(B5), 0)                # no data -> no information

  # K_max at its floor (== max(y)) is admissible: the N-sum range [max(y), K_max]
  # is non-empty and every site stays finite. (The constructor rejects any
  # K_max < max(y), so sub-floor inadmissibility is unreachable by construction.)
  floor_marg <- tulpa_nmix_site_marginal(
    y = dat$y, site_idx = dat$site_idx,
    X_lambda = dat$X_lambda, X_p = dat$X_p, mixture = "P",
    K_max = max(dat$y)
  )
  ev_floor <- floor_marg$eval(as.vector(dat$X_lambda %*% c(log(4), 0)),
                              as.vector(dat$X_p %*% c(0, 0)))
  expect_equal(ev_floor$n_K_inadmissible, 0L)
  expect_true(all(is.finite(ev_floor$log_lik_site)))
})
