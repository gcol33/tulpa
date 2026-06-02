# Recovery tests for tulpa_re_aghq(): the callback-driven adaptive-GHQ
# refinement of a grouped random-effect covariance. The marginal is supplied by
# `make_site`; here it is a plain binomial GLMM so the engine can be checked in
# isolation (a custom latent-state marginal is exercised downstream in tulpaObs).

l1pe <- function(x) ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))

# Binomial-GLMM site closures for design X, response y, trials nt.
make_binom_site <- function(X, y, nt) {
  function(theta) {
    eta_fixed <- as.numeric(X %*% theta)
    list(
      eta_re = eta_fixed,
      deriv = function(rows, eta) {
        p <- plogis(eta)
        list(logL = y[rows] * eta - nt[rows] * l1pe(eta),
             d1 = y[rows] - nt[rows] * p,
             d2 = -nt[rows] * p * (1 - p))
      },
      lmat = function(rows, ETA) y[rows] * ETA - nt[rows] * l1pe(ETA))
  }
}

test_that("intercept-only AGHQ recovers sigma and beats Laplace (n_quad=1)", {
  skip_on_cran()
  skip_if_fast()
  truth <- 0.9
  one <- function(seed, n_quad) {
    set.seed(seed)
    ng <- 30L; n_per <- 8L; N <- ng * n_per
    group <- rep(seq_len(ng), each = n_per)
    x <- rnorm(N); X <- cbind(1, x); nt <- rep(3L, N)
    u <- rnorm(ng, 0, truth)
    y <- rbinom(N, nt, plogis(0.3 + 0.7 * x + u[group]))
    fit <- tulpa_re_aghq(
      theta0 = c(0, 0),
      re_terms = list(list(idx = group, n_groups = ng, n_coefs = 1L)),
      Sigma0 = list(matrix(0.25, 1, 1)),
      make_site = make_binom_site(X, y, nt), n_obs = N, n_quad = n_quad)
    sqrt(fit$Sigma_list[[1]][1, 1])
  }
  lap  <- vapply(1:8, one, numeric(1), n_quad = 1L)
  aghq <- vapply(1:8, one, numeric(1), n_quad = 9L)
  # AGHQ removes attenuation: closer to truth than Laplace, and near-unbiased.
  expect_lt(abs(mean(aghq) - truth), abs(mean(lap) - truth))
  expect_lt(abs(mean(aghq) - truth), 0.12)
})

test_that("correlated (1 + x | g) block: full Sigma + structure recovered", {
  skip_on_cran()
  skip_if_fast()
  set.seed(11)
  ng <- 50L; n_per <- 20L; N <- ng * n_per
  group <- rep(seq_len(ng), each = n_per)
  x <- rnorm(N); X <- cbind(1, x); nt <- rep(4L, N)
  Sig <- matrix(c(0.6, 0.3, 0.3, 0.4), 2, 2)
  U <- matrix(rnorm(ng * 2), ng, 2) %*% chol(Sig)
  eta <- 0.2 + (-0.4) * x + U[group, 1] + U[group, 2] * x
  y <- rbinom(N, nt, plogis(eta))

  fit <- tulpa_re_aghq(
    theta0 = c(0, 0),
    re_terms = list(list(idx = group, n_groups = ng, n_coefs = 2L,
                         Z = X, correlated = TRUE)),
    Sigma0 = list(diag(0.25, 2)),
    make_site = make_binom_site(X, y, nt), n_obs = N, n_quad = 7L)

  expect_false(is.null(fit))
  S <- fit$Sigma_list[[1]]
  expect_equal(dim(S), c(2L, 2L))
  # marginal SDs in the right ballpark (truth sqrt(0.6), sqrt(0.4)).
  expect_lt(abs(sqrt(S[1, 1]) - sqrt(0.6)), 0.25)
  expect_lt(abs(sqrt(S[2, 2]) - sqrt(0.4)), 0.25)
  # positive correlation recovered (truth ~ 0.61), off the boundary.
  rho <- S[1, 2] / sqrt(S[1, 1] * S[2, 2])
  expect_gt(rho, 0.2); expect_lt(rho, 0.95)
  # BLUPs track the simulated effects, fixed-effect SEs finite.
  expect_gt(cor(fit$blup[[1]][, 1], U[, 1]), 0.6)
  expect_gt(cor(fit$blup[[1]][, 2], U[, 2]), 0.4)
  expect_true(all(is.finite(fit$theta_se)))
})

test_that("LKJ penalty pulls a weakly-identified correlation off the boundary", {
  skip_on_cran()
  skip_if_fast()
  # Small per-group n: the unregularized correlation can run toward +-1.
  one <- function(eta) {
    set.seed(303)
    ng <- 40L; n_per <- 10L; N <- ng * n_per
    group <- rep(seq_len(ng), each = n_per)
    x <- rnorm(N); X <- cbind(1, x); nt <- rep(3L, N)
    Sig <- matrix(c(0.6, 0.3, 0.3, 0.4), 2, 2)
    U <- matrix(rnorm(ng * 2), ng, 2) %*% chol(Sig)
    y <- rbinom(N, nt, plogis(0.2 - 0.4 * x + U[group, 1] + U[group, 2] * x))
    fit <- tulpa_re_aghq(
      theta0 = c(0, 0),
      re_terms = list(list(idx = group, n_groups = ng, n_coefs = 2L,
                           Z = X, correlated = TRUE)),
      Sigma0 = list(diag(0.25, 2)),
      make_site = make_binom_site(X, y, nt), n_obs = N, n_quad = 7L,
      lkj_eta = eta)
    S <- fit$Sigma_list[[1]]
    S[1, 2] / sqrt(S[1, 1] * S[2, 2])
  }
  rho_ml  <- one(1)     # no penalty
  rho_reg <- one(1.5)   # default LKJ
  # The regularized correlation is no larger in magnitude (pulled toward 0).
  expect_lte(abs(rho_reg), abs(rho_ml) + 1e-6)
  expect_lt(abs(rho_reg), 0.99)
})
