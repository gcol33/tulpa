# Default PC + LKJ hyperprior for the RE-covariance integrator.
#
# The correctness property is the change-of-variables Jacobian from the natural
# (sigma, R) scale to the log-Cholesky theta the integrator works in. We check
# the closed-form Jacobian against a finite-difference Jacobian of the actual
# coordinate map (independent ground truth -- the map uses only the log-Cholesky
# packing and basic algebra, not the derived formula), then check that
# re_cov_pc_lkj_prior() assembles the PC term, the LKJ term, and that verified
# Jacobian correctly.

# Central-difference Jacobian (no numDeriv dependency).
.num_jac <- function(f, x, h = 1e-5) {
  m <- length(f(x))
  J <- matrix(0, m, length(x))
  for (j in seq_along(x)) {
    xp <- x; xp[j] <- xp[j] + h
    xm <- x; xm[j] <- xm[j] - h
    J[, j] <- (f(xp) - f(xm)) / (2 * h)
  }
  J
}

# Coordinate map theta -> (sigma_1..sigma_c, strict-lower correlations).
.theta_to_sigma_R <- function(theta, c_re) {
  L   <- tulpa:::.re_logchol_to_L(theta, c_re)
  S   <- L %*% t(L)
  sig <- sqrt(diag(S))
  R   <- S / outer(sig, sig)
  c(sig, R[lower.tri(R)])
}

# Closed-form log|det d(sigma, R)/d theta| from the derivation.
.analytic_logdet <- function(theta, c_re) {
  L   <- tulpa:::.re_logchol_to_L(theta, c_re)
  sum((c_re + 2L - seq_len(c_re)) * log(diag(L))) -
    c_re * sum(log(sqrt(rowSums(L^2))))
}


test_that("the closed-form Jacobian matches finite differences (c = 2, 3)", {
  set.seed(11L)
  for (c_re in c(2L, 3L)) {
    k <- c_re * (c_re + 1L) / 2L
    for (rep in seq_len(5L)) {
      # random theta: log-diagonal modest, off-diagonals modest
      theta <- rnorm(k, sd = 0.4)
      num <- log(abs(det(.num_jac(function(t) .theta_to_sigma_R(t, c_re),
                                  theta))))
      ana <- .analytic_logdet(theta, c_re)
      expect_equal(ana, num, tolerance = 1e-5,
                   label = sprintf("c=%d rep=%d Jacobian", c_re, rep))
    }
  }
})


test_that("re_cov_pc_lkj_prior assembles PC + LKJ + Jacobian correctly", {
  set.seed(22L)
  prior_sigma <- c(2.5, 0.05); eta <- 3
  lambda <- -log(prior_sigma[2L]) / prior_sigma[1L]
  for (c_re in c(2L, 3L)) {
    k     <- c_re * (c_re + 1L) / 2L
    prior <- re_cov_pc_lkj_prior(c_re, prior_sigma = prior_sigma, eta = eta)
    for (rep in seq_len(4L)) {
      theta <- rnorm(k, sd = 0.4)
      L     <- tulpa:::.re_logchol_to_L(theta, c_re)
      sig   <- sqrt(rowSums(L^2))
      logdetR <- 2 * sum(log(diag(L))) - 2 * sum(log(sig))
      expected <- sum(log(lambda) - lambda * sig) +    # PC on each SD
        (eta - 1) * logdetR +                          # LKJ on correlation
        .analytic_logdet(theta, c_re)                  # change of variables
      expect_equal(prior(theta), expected, tolerance = 1e-10,
                   label = sprintf("c=%d rep=%d assembled prior", c_re, rep))
    }
  }
})


test_that("c = 1 reduces to the exact scalar PC change of variables", {
  # c = 1: theta = log(sigma); p_theta(theta) = lambda exp(-lambda sigma) * sigma.
  prior <- re_cov_pc_lkj_prior(1L, prior_sigma = c(3, 0.05), eta = 2)
  lambda <- -log(0.05) / 3
  for (theta in c(-1, -0.2, 0.5, 1.3)) {
    sigma <- exp(theta)
    expect_equal(prior(theta),
                 log(lambda) - lambda * sigma + log(sigma),
                 tolerance = 1e-12)
  }
})


test_that("PC tail probability and LKJ shape behave as specified", {
  # PC: P(sigma > U) = alpha by construction of lambda = -log(alpha)/U.
  U <- 2.5; alpha <- 0.05
  lambda <- -log(alpha) / U
  expect_equal(exp(-lambda * U), alpha, tolerance = 1e-12)

  # LKJ: larger eta down-weights strong correlation. Take theta with a strong
  # intercept-slope correlation, hold the scales fixed, vary only eta. Lower
  # Cholesky of [[1, 0.9], [0.9, 1]] is [[1, 0], [0.9, sqrt(1 - 0.9^2)]].
  L <- matrix(c(1, 0, 0.9, sqrt(1 - 0.9^2)), 2, 2, byrow = TRUE)  # rho = 0.9
  theta <- tulpa:::.re_L_to_logchol(L, 2L)
  stopifnot(abs(cov2cor(L %*% t(L))[2, 1] - 0.9) < 1e-10)        # guard rho = 0.9
  p_flat <- re_cov_pc_lkj_prior(2L, eta = 1)            # uniform on R
  p_reg  <- re_cov_pc_lkj_prior(2L, eta = 4)            # favours weak corr
  expect_lt(p_reg(theta), p_flat(theta))                # eta=4 penalises rho=0.9
})


test_that("a diagonal block uses log-SD coords with no LKJ term", {
  # correlated = FALSE: theta_i = log sigma_i, prior = PC(sigma_i) + Jacobian
  # sum_i theta_i, and no correlation factor regardless of eta.
  prior_sigma <- c(2.5, 0.05)
  lambda <- -log(prior_sigma[2L]) / prior_sigma[1L]
  for (nc in c(1L, 2L, 3L)) {
    p2 <- re_cov_pc_lkj_prior(nc, prior_sigma = prior_sigma, eta = 2,
                              correlated = FALSE)
    p9 <- re_cov_pc_lkj_prior(nc, prior_sigma = prior_sigma, eta = 9,
                              correlated = FALSE)
    set.seed(nc)
    th <- rnorm(nc, sd = 0.4)
    sig <- exp(th)
    expected <- sum(log(lambda) - lambda * sig) + sum(th)
    expect_equal(p2(th), expected, tolerance = 1e-12,
                 label = sprintf("nc=%d diagonal prior", nc))
    # eta is irrelevant for a diagonal block
    expect_equal(p9(th), p2(th), tolerance = 1e-12)
  }
})


test_that("the joint multi-block prior sums independent per-block priors", {
  prior_sigma <- c(3, 0.05); eta <- 2
  layout <- list(
    list(nc = 2L, full = TRUE,  k = 3L, label = "g"),
    list(nc = 1L, full = FALSE, k = 1L, label = "h")
  )
  joint <- tulpa:::.re_cov_joint_prior(layout, prior_sigma, eta)
  p_g <- re_cov_pc_lkj_prior(2L, prior_sigma = prior_sigma, eta = eta,
                             correlated = TRUE)
  p_h <- re_cov_pc_lkj_prior(1L, prior_sigma = prior_sigma, eta = eta,
                             correlated = FALSE)
  set.seed(99L)
  theta <- rnorm(4L, sd = 0.4)   # 3 (g) + 1 (h)
  expect_equal(joint(theta), p_g(theta[1:3]) + p_h(theta[4]),
               tolerance = 1e-12)
})


test_that("supplied prior_sigma / eta change the integrated posterior", {
  skip_on_cran()
  # End-to-end: the default prior is wired into tulpa_re_cov_nested and a
  # tighter PC prior shrinks the variance-component summaries.
  set.seed(5L)
  G <- 30L; npg <- 10L; N <- G * npg
  grp <- rep(seq_len(G), each = npg)
  x <- rnorm(N); X <- cbind(1, x); Z <- cbind(1, x)
  Sigma <- matrix(c(0.8^2, 0.3 * 0.8 * 0.6, 0.3 * 0.8 * 0.6, 0.6^2), 2)
  u <- t(t(chol(Sigma)) %*% matrix(rnorm(2 * G), 2))
  eta_lin <- as.numeric(X %*% c(0, 0.4)) + rowSums(Z * u[grp, ])
  y <- rbinom(N, 1L, plogis(eta_lin))
  rt <- list(idx = grp, n_groups = G, n_coefs = 2L, Z = Z)

  wide   <- tulpa_re_cov_nested(y, rep(1L, N), X, rt, family = "binomial",
                                prior_sigma = c(5, 0.05))
  narrow <- tulpa_re_cov_nested(y, rep(1L, N), X, rt, family = "binomial",
                                prior_sigma = c(0.6, 0.05))   # strong shrinkage
  s1_wide <- wide$posterior$median[wide$posterior$parameter == "sigma_1"]
  s1_narr <- narrow$posterior$median[narrow$posterior$parameter == "sigma_1"]
  expect_lt(s1_narr, s1_wide)
})
