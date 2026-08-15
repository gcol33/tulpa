# tulpa_re_aghq()'s mode/theta cross-Hessian block (gcol33/tulpa#398).
#
# `blup_cross[[m]]` is `Bf`, the group-mode/theta cross-Hessian a downstream
# community-model consumer (tulpaObs) needs to draw a group's BLUP JOINTLY
# with theta instead of independently -- drawing them independently drops
# Cov(theta, b_s) and miscalibrates the posterior (gcol33/tulpaObs#226). Only a
# prebuilt native `oracle` (an analytic `theta_score`) can supply it; the
# R-closure bridge (`make_site` / `make_group`) cannot and must decline with
# NA, never a silent 0 (a 0 cross term reads as "theta and b_s are
# independent", which is the exact miscalibration this exists to avoid).

l1pe <- function(x) ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))

test_that("the R-closure bridge declines blup_cross (NA, not a silent 0)", {
  skip_on_cran()
  set.seed(5)
  ng <- 10L; nper <- 6L; n <- ng * nper
  group <- rep(seq_len(ng), each = nper)
  x <- rnorm(n); X <- cbind(1, x); nt <- rep(1, n)
  beta_true <- c(0.2, 0.4)
  b_true <- rnorm(ng, 0, 0.5)
  eta <- as.numeric(X %*% beta_true) + b_true[group]
  y <- rbinom(n, 1, plogis(eta))

  make_site <- function(theta) {
    eta_fixed <- as.numeric(X %*% theta)
    list(eta_re = eta_fixed,
         deriv = function(rows, e) {
           p <- plogis(e)
           list(logL = y[rows] * e - nt[rows] * l1pe(e),
                d1 = y[rows] - nt[rows] * p, d2 = -nt[rows] * p * (1 - p))
         },
         lmat = function(rows, ETA) y[rows] * ETA - nt[rows] * l1pe(ETA))
  }
  re_terms <- list(list(idx = group, n_groups = ng, n_coefs = 1L))
  Sigma0 <- list(matrix(0.25, 1, 1))

  fit <- tulpa_re_aghq(beta_true, re_terms, Sigma0, make_site = make_site,
                       n_obs = n, n_quad = 5L)
  expect_false(is.null(fit))
  expect_false(fit$blup_cross_available)
  expect_true(all(is.na(fit$blup_cross[[1]])))
  expect_equal(dim(fit$blup_cross[[1]]), c(ng, 2L, 1L))
})

test_that("a native oracle's blup_cross matches the closed-form binomial GLMM cross term", {
  skip_on_cran()
  set.seed(7)
  ng <- 14L; nper <- 9L; n <- ng * nper
  group <- rep(seq_len(ng), each = nper)
  x <- as.numeric(scale(rnorm(n))); X <- cbind(1, x)
  Z <- matrix(1, n, 1)
  beta_true <- c(0.3, 0.5); sig_true <- 0.6
  b_true <- rnorm(ng, 0, sig_true)
  eta <- as.numeric(X %*% beta_true) + b_true[group]
  y <- rbinom(n, 1, plogis(eta))
  nt <- rep(1, n)

  orc <- tulpa:::cpp_glmm_oracle_make("binomial", 1, as.numeric(y),
                                      as.numeric(nt), X, Z, group, ng)
  re_terms <- list(list(n_groups = ng, n_coefs = 1L, correlated = FALSE))
  Sigma0 <- list(matrix(sig_true^2, 1, 1))

  fit <- tulpa_re_aghq(beta_true, re_terms, Sigma0, oracle = orc, n_quad = 7L)
  expect_false(is.null(fit))
  expect_true(fit$blup_cross_available)

  BC <- fit$blup_cross[[1]]                    # ng x n_theta x 1
  expect_equal(dim(BC), c(ng, 2L, 1L))
  theta_hat <- fit$theta
  b_hat <- fit$blup[[1]][, 1]
  p_hat <- plogis(as.numeric(X %*% theta_hat) + b_hat[group])

  # Closed form: ell_g(b) = sum_i [y_i eta_i - log(1+exp(eta_i))], so
  # d^2 ell_g / dtheta db = -X_g' diag(p(1-p)) Z_g, i.e.
  # Bf_g = -d^2 ell_g/dtheta db = X_g' diag(p(1-p)) Z_g (n_theta x d).
  for (g in seq_len(ng)) {
    rows <- which(group == g)
    Xg <- X[rows, , drop = FALSE]; Zg <- Z[rows, , drop = FALSE]
    w <- p_hat[rows] * (1 - p_hat[rows])
    Bf_analytic <- crossprod(Xg, w * Zg)
    Bf_engine <- matrix(BC[g, , ], nrow = 2, ncol = 1)
    expect_equal(Bf_engine, Bf_analytic, tolerance = 1e-8, ignore_attr = TRUE)
  }
})

test_that("Cinv %*% t(Bf) reproduces the first-order db_hat/dtheta the joint BLUP draw needs", {
  skip_on_cran()
  # The consumer draws b_s | theta ~ N(blup_s - Cinv_s %*% t(Bf_s) %*% (theta_draw
  # - theta_hat), Cinv_s) -- a first-order Taylor correction of the group's mode
  # around a perturbed theta. Verify -Cinv %*% t(Bf) against an independent
  # finite-difference re-solve of the group's own Laplace mode at a perturbed
  # theta (holding the fitted Sigma fixed), i.e. this checks the joint-draw
  # formula's ingredient directly, not just the raw cross-Hessian value.
  set.seed(9)
  ng <- 12L; nper <- 9L; n <- ng * nper
  group <- rep(seq_len(ng), each = nper)
  x <- as.numeric(scale(rnorm(n))); X <- cbind(1, x)
  Z <- matrix(1, n, 1)
  beta_true <- c(0.3, 0.5); sig_true <- 0.6
  b_true <- rnorm(ng, 0, sig_true)
  eta <- as.numeric(X %*% beta_true) + b_true[group]
  y <- rbinom(n, 1, plogis(eta))
  nt <- rep(1, n)

  orc <- tulpa:::cpp_glmm_oracle_make("binomial", 1, as.numeric(y),
                                      as.numeric(nt), X, Z, group, ng)
  re_terms <- list(list(n_groups = ng, n_coefs = 1L, correlated = FALSE))
  Sigma0 <- list(matrix(sig_true^2, 1, 1))
  fit <- tulpa_re_aghq(beta_true, re_terms, Sigma0, oracle = orc, n_quad = 7L)
  expect_false(is.null(fit))

  g_test <- 3L
  rows <- which(group == g_test)
  Xg <- X[rows, , drop = FALSE]; Zg <- Z[rows, , drop = FALSE]; yg <- y[rows]
  sig2_hat <- fit$Sigma_list[[1]][1, 1]
  mode_at <- function(theta) {
    bb <- 0
    for (it in 1:100) {
      p <- plogis(as.numeric(Xg %*% theta) + Zg %*% bb)
      grad <- sum(yg - p) - bb / sig2_hat
      negH <- sum(p * (1 - p)) + 1 / sig2_hat
      bb <- bb + grad / negH
    }
    bb
  }
  theta_hat <- fit$theta
  h <- 1e-4
  db_dtheta_fd <- vapply(seq_along(theta_hat), function(j) {
    tp <- theta_hat; tp[j] <- tp[j] + h
    tm <- theta_hat; tm[j] <- tm[j] - h
    (mode_at(tp) - mode_at(tm)) / (2 * h)
  }, numeric(1))

  Cinv_g <- fit$blup_var[[1]][g_test, 1]
  Bf_g <- matrix(fit$blup_cross[[1]][g_test, , ], nrow = 2, ncol = 1)
  db_dtheta_engine <- as.numeric(-Cinv_g %*% t(Bf_g))
  # Both sides are finite differences (db_dtheta_fd over theta at h = 1e-4, the
  # engine's Bf itself an internal FD over b), and the true quantity here is
  # small (~1e-5), so an absolute rather than expect_equal's relative tolerance
  # is the right budget; the observed gap is ~2e-10.
  expect_lt(max(abs(db_dtheta_engine - db_dtheta_fd)), 1e-7)
})
