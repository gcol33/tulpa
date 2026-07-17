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

test_that(".aghq_sigma_penalty: value matches the PC log-prior, grad matches FD", {
  # Two blocks: one scalar (targeted), one 2x2 (untargeted). n_theta = 3 fixed.
  re_terms <- list(list(n_groups = 12L, n_coefs = 1L, correlated = FALSE),
                   list(n_groups = 12L, n_coefs = 2L, correlated = TRUE))
  layout   <- .re_cov_block_layout(.as_re_terms_list(re_terms), NULL)
  n_theta  <- 3L
  # off -> NULL (the byte-identical path).
  expect_null(.aghq_sigma_penalty(NULL, layout, n_theta))
  # PC prior on block 1 only.
  U <- 1; alpha <- 0.05
  pen <- .aghq_sigma_penalty(list(blocks = 1L, prior_sigma = c(U, alpha)),
                             layout, n_theta)
  # A par vector: 3 theta, then re_par = [log sigma1 | logChol of the 2x2].
  set.seed(7)
  par <- c(rnorm(3), log(0.4), 0.1, -0.2, 0.3)
  # value == the scalar-block PC log-prior at log sigma1 (independent formula:
  # log(lambda) - lambda*sigma + log(sigma), lambda = -log(alpha)/U).
  lam  <- -log(alpha) / U; ls <- par[n_theta + 1L]; sig <- exp(ls)
  expect_equal(pen$val(par), log(lam) - lam * sig + ls, tolerance = 1e-10)
  # only the targeted block moves the value: perturbing an untargeted coord is a
  # no-op.
  par2 <- par; par2[n_theta + 3L] <- par2[n_theta + 3L] + 1
  expect_equal(pen$val(par2), pen$val(par), tolerance = 1e-10)
  # gradient matches a central finite difference of val, and is zero off the
  # targeted coordinate.
  g <- pen$grad(par); h <- 1e-6
  fd <- vapply(seq_along(par), function(j) {
    pp <- par; pp[j] <- pp[j] + h; pm <- par; pm[j] <- pm[j] - h
    (pen$val(pp) - pen$val(pm)) / (2 * h)
  }, numeric(1))
  expect_equal(g, fd, tolerance = 1e-5)
  expect_equal(g[-(n_theta + 1L)], rep(0, length(par) - 1L))
  # bad specs error.
  expect_error(.aghq_sigma_penalty(list(prior_sigma = 1), layout, n_theta),
               "c\\(U, alpha\\)")
  expect_error(.aghq_sigma_penalty(list(blocks = 5L, prior_sigma = c(1, .05)),
                                   layout, n_theta), "index the RE blocks")
})

test_that("sigma_prior pulls a collapsing scalar RE SD off the 0 boundary, unbiased when identified", {
  skip_on_cran()
  # Few groups + a small true SD: the ML variance component attenuates toward 0.
  fit_sd <- function(truth, ng, sigma_prior) {
    set.seed(404)
    n_per <- 6L; N <- ng * n_per
    group <- rep(seq_len(ng), each = n_per)
    x <- rnorm(N); X <- cbind(1, x); nt <- rep(3L, N)
    u <- rnorm(ng, 0, truth)
    y <- rbinom(N, nt, plogis(0.2 + 0.5 * x + u[group]))
    fit <- tulpa_re_aghq(
      theta0 = c(0, 0),
      re_terms = list(list(idx = group, n_groups = ng, n_coefs = 1L)),
      Sigma0 = list(matrix(0.25, 1, 1)),
      make_site = make_binom_site(X, y, nt), n_obs = N, n_quad = 5L,
      sigma_prior = sigma_prior)
    sqrt(fit$Sigma_list[[1]][1, 1])
  }
  pc <- c(1, 0.05)   # P(sigma > 1) = 0.05, interior mode ~ 0.33
  # Collapse-prone (small truth, few groups): the prior keeps the SD off 0.
  sd_ml  <- fit_sd(0.15, 6L, NULL)
  sd_reg <- fit_sd(0.15, 6L, pc)
  expect_gt(sd_reg, sd_ml)
  expect_gt(sd_reg, 0.05)
  # Well-identified (large truth, many groups): the weak prior barely moves it.
  sd_ml2  <- fit_sd(1.1, 60L, NULL)
  sd_reg2 <- fit_sd(1.1, 60L, pc)
  expect_lt(abs(sd_reg2 - sd_ml2), 0.15)
})
