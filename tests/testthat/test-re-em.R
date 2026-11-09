# tulpa_re_em(): Laplace-EM on the per-group oracle tulpa_re_aghq() uses.

.re_em_glmm <- function(family, seed, d_re = 1L, ng = 20L, nper = 8L) {
  set.seed(seed)
  n <- ng * nper
  group <- rep(seq_len(ng), each = nper)
  x <- as.numeric(scale(rnorm(n)))
  X <- cbind(1, x)
  Z <- if (d_re == 2L) cbind(1, x) else matrix(1, n, 1)
  beta <- c(0.3, 0.5)
  Sig <- if (d_re == 2L) matrix(c(0.40, 0.10, 0.10, 0.30), 2, 2) else matrix(0.5, 1, 1)
  B <- matrix(rnorm(ng * d_re), ng, d_re) %*% chol(Sig)
  eta <- as.numeric(X %*% beta) + rowSums(Z * B[group, , drop = FALSE])
  n_trials <- rep(1, n)
  y <- switch(family,
    binomial = stats::rbinom(n, 1, stats::plogis(eta)),
    poisson  = stats::rpois(n, exp(pmin(eta, 5))),
    gaussian = eta + stats::rnorm(n, 0, 0.5))
  phi <- if (family == "gaussian") 0.25 else 1
  orc <- tulpa:::cpp_glmm_oracle_make(family, phi, as.numeric(y),
                                      as.numeric(n_trials), X, Z, group, ng)
  list(orc = orc, beta = beta, Sig = Sig,
       re_terms = list(list(n_groups = ng, n_coefs = d_re,
                            correlated = d_re > 1L)))
}

test_that("tulpa_re_em matches the Laplace maximum on a Gaussian GLMM", {
  # With a Gaussian conditional the Laplace marginal is exact, so the EM fixed
  # point is the Laplace maximum that tulpa_re_aghq() reaches at n_quad = 1.
  fx <- .re_em_glmm("gaussian", 11L)
  em <- tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), oracle = fx$orc,
                    tol = 1e-12)
  aq <- tulpa_re_aghq(fx$beta, fx$re_terms, list(fx$Sig), oracle = fx$orc,
                      n_quad = 1L)
  expect_true(em$converged)
  expect_equal(em$theta, aq$theta, tolerance = 1e-4)
  expect_equal(em$Sigma_list[[1]], aq$Sigma_list[[1]], tolerance = 1e-3)
  expect_equal(em$log_marginal, aq$log_marginal, tolerance = 1e-6)
  expect_equal(em$theta_se, sqrt(diag(aq$theta_cov)), tolerance = 1e-2)
  # The EM never decreases the Laplace marginal.
  expect_true(all(diff(em$log_marginal_trace) > -1e-8))
})

test_that("tulpa_re_em fits a correlated Poisson RE and reports BLUPs", {
  fx <- .re_em_glmm("poisson", 12L, d_re = 2L)
  em <- tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), oracle = fx$orc)
  expect_true(em$converged)
  expect_true(all(is.finite(em$theta_se)))
  expect_equal(dim(em$blup[[1]]), c(20L, 2L))
  expect_length(em$blup_cov_g, 20L)
  expect_true(all(em$group_ok))
  expect_true(all(diff(em$log_marginal_trace) > -1e-6))
  # The exact-marginal fit on the same oracle lands close by.
  aq <- tulpa_re_aghq(fx$beta, fx$re_terms, list(fx$Sig), oracle = fx$orc,
                      n_quad = 1L)
  expect_equal(em$theta, aq$theta, tolerance = 0.05)
})

test_that("tulpa_re_em projects an uncorrelated term and applies the floor", {
  fx <- .re_em_glmm("binomial", 13L)
  em <- tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), oracle = fx$orc,
                    sigma_floor = 0.3)
  expect_gte(sqrt(em$Sigma_list[[1]][1, 1]), 0.3 - 1e-8)
  expect_error(tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), oracle = fx$orc,
                           sigma_floor = 0), "sigma_floor")
})

test_that("tulpa_re_em does not depend on the thread count", {
  # The binomial density's coefficient goes through R's math library, which
  # checks R's C stack; the oracle evaluates it per observation at construction
  # so its group loop stays off R on worker threads (#1156).
  for (family in c("binomial", "poisson", "gaussian")) {
    fx <- .re_em_glmm(family, 14L, d_re = 2L)
    e1 <- tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), oracle = fx$orc,
                      sigma_floor = 0.3, n_threads = 1L)
    e2 <- tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), oracle = fx$orc,
                      sigma_floor = 0.3, n_threads = 4L)
    expect_identical(e1$theta, e2$theta, info = family)
    expect_identical(e1$Sigma_list, e2$Sigma_list, info = family)
    expect_identical(e1$log_marginal, e2$log_marginal, info = family)
  }
})

test_that("the latent theta rule reaches the marginal rule's theta on a Gaussian GLMM", {
  # theta enters the linear predictor of a Gaussian conditional, so the
  # per-group log-determinant does not depend on it and the joint mode in theta
  # is the Laplace-marginal maximum.
  fx <- .re_em_glmm("gaussian", 11L)
  m <- tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), oracle = fx$orc, tol = 1e-12)
  l <- tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), oracle = fx$orc, tol = 1e-12,
                   theta_update = "latent")
  expect_true(l$converged)
  expect_false(l$stuck)
  expect_equal(l$theta, m$theta, tolerance = 1e-5)
  expect_equal(l$Sigma_list[[1]], m$Sigma_list[[1]], tolerance = 1e-5)
  expect_equal(l$theta_se, m$theta_se, tolerance = 1e-4)
  expect_equal(l$log_marginal, m$log_marginal, tolerance = 1e-8)
})

test_that("a make_group joint closure drives the latent rule like the native oracle", {
  fx <- .re_em_glmm("poisson", 12L, d_re = 2L)
  set.seed(12L)
  ng <- 20L; nper <- 8L; n <- ng * nper
  group <- rep(seq_len(ng), each = nper)
  x <- as.numeric(scale(rnorm(n))); X <- cbind(1, x); Z <- cbind(1, x)
  B <- matrix(rnorm(ng * 2L), ng, 2L) %*% chol(fx$Sig)
  y <- stats::rpois(n, exp(pmin(as.numeric(X %*% fx$beta) + rowSums(Z * B[group, ]), 5)))
  orc <- tulpa:::cpp_glmm_oracle_make("poisson", 1, as.numeric(y), rep(1, n), X, Z,
                                      group, ng)
  rows <- split(seq_len(n), group)
  make_group <- function(theta) {
    eta0 <- as.numeric(X %*% theta)
    joint <- function(g, b) {
      r <- rows[[g]]; Zg <- Z[r, , drop = FALSE]; Xg <- X[r, , drop = FALSE]
      mu <- exp(eta0[r] + as.numeric(Zg %*% b))
      list(logL = sum(stats::dpois(y[r], mu, log = TRUE)),
           grad = as.numeric(crossprod(Zg, y[r] - mu)), negH = crossprod(Zg, mu * Zg),
           theta_grad = as.numeric(crossprod(Xg, y[r] - mu)),
           theta_info = crossprod(Xg, mu * Xg), cross = crossprod(Xg, mu * Zg))
    }
    list(grad_hess = function(g, b) joint(g, b)[c("logL", "grad", "negH")],
         node_ll = function(g, Bn) apply(Bn, 1L, function(b) joint(g, b)$logL),
         joint = joint)
  }
  ln <- tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), oracle = orc,
                    theta_update = "latent")
  lr <- tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), make_group = make_group,
                    theta_update = "latent")
  expect_true(lr$converged)
  expect_equal(lr$theta, ln$theta, tolerance = 1e-8)
  expect_equal(lr$Sigma_list, ln$Sigma_list, tolerance = 1e-8)
  expect_equal(lr$theta_se, ln$theta_se, tolerance = 1e-8)
  # The native cross block is a central difference of the theta score.
  expect_equal(lr$blup_cross_g, ln$blup_cross_g, tolerance = 1e-6)
  expect_true(lr$blup_cross_available)
  # Without `joint` the closure has no theta score for the latent rule.
  bare <- function(theta) make_group(theta)[c("grad_hess", "node_ll")]
  expect_error(tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), make_group = bare,
                           theta_update = "latent"), "theta score")
  # A warm start at the converged modes lands on the same fit.
  lw <- tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), make_group = make_group,
                    theta_update = "latent", b0 = lr$blup[[1]])
  expect_equal(lw$theta, lr$theta, tolerance = 1e-6)
})

test_that("tulpa_re_em takes a per-coordinate theta prior and checks b0", {
  fx <- .re_em_glmm("binomial", 13L)
  a <- tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), oracle = fx$orc,
                   theta_prior_sd = c(Inf, 0.05))
  expect_lt(abs(a$theta[2]), 0.1)
  expect_error(tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), oracle = fx$orc,
                           theta_prior_sd = c(1, 2, 3)), "theta_prior_sd")
  expect_error(tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), oracle = fx$orc,
                           b0 = matrix(0, 3, 1)), "b0")
})

test_that("the AGHQ moment step leaves a Gaussian fit unchanged and moves a binary one", {
  # A Gaussian conditional makes each group's posterior exactly Gaussian, so
  # its quadrature moments are the Laplace ones and the step is one more EM
  # update, which at a converged fixed point moves nothing.
  fx <- .re_em_glmm("gaussian", 11L)
  l <- tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), oracle = fx$orc,
                   theta_update = "latent", tol = 1e-14, max_iter = 2000L)
  a <- tulpa_re_em(fx$beta, fx$re_terms, list(fx$Sig), oracle = fx$orc,
                   theta_update = "latent", aghq_debias = 7L, tol = 1e-14,
                   max_iter = 2000L)
  expect_true(a$aghq_debiased)
  expect_false(l$aghq_debiased)
  expect_equal(a$Sigma_list, l$Sigma_list, tolerance = 1e-6)
  expect_equal(a$theta_se, l$theta_se, tolerance = 1e-6)
  expect_equal(a$blup_cov_g, l$blup_cov_g, tolerance = 1e-6)
  # Binary data at eight observations per group: the moments are not Gaussian.
  fb <- .re_em_glmm("binomial", 13L)
  lb <- tulpa_re_em(fb$beta, fb$re_terms, list(fb$Sig), oracle = fb$orc,
                    theta_update = "latent")
  ab <- tulpa_re_em(fb$beta, fb$re_terms, list(fb$Sig), oracle = fb$orc,
                    theta_update = "latent", aghq_debias = 9L)
  expect_true(ab$aghq_debiased)
  expect_gt(abs(ab$Sigma_list[[1]][1, 1] - lb$Sigma_list[[1]][1, 1]), 1e-4)
  expect_identical(ab$theta, lb$theta)
  expect_error(tulpa_re_em(fb$beta, fb$re_terms, list(fb$Sig), oracle = fb$orc,
                           aghq_debias = 5L), "latent")
})

test_that("tulpa_re_cov_update projects, symmetrizes and floors", {
  M <- matrix(c(0.5, 0.2, 0.2, 1e-9), 2, 2)
  S <- tulpa_re_cov_update(M, sigma_floor = 0.01)
  expect_true(isSymmetric(S))
  expect_gte(min(eigen(S, symmetric = TRUE)$values), 1e-4 - 1e-12)
  D <- tulpa_re_cov_update(M, correlated = FALSE, sigma_floor = 0.01)
  expect_equal(D, diag(c(0.5, 1e-4)))
  expect_equal(tulpa_re_cov_update(diag(c(2, 3))), diag(c(2, 3)))
  expect_error(tulpa_re_cov_update(matrix(1, 2, 3)), "square")
})

test_that("the latent rule carries a group whose log-likelihood is not concave at the start to its mode", {
  # Each group's log-likelihood is a weighted Cauchy log-density in x = theta + b,
  # concave only within one unit of its mode. Group 11's mode sits 3.5 units
  # from the start, where its curvature has the wrong sign: the plain Newton
  # direction for that group descends its own log-likelihood, and the joint
  # line search, which scores the sum over groups, accepts it.
  w <- 200
  m <- c(seq(-0.3, 0.3, length.out = 10L), 3.5)
  ng <- length(m)
  make_group <- function(theta) {
    joint <- function(g, b) {
      r <- theta[1L] + b[1L] - m[g]
      h <- matrix(w * 2 * (1 - r^2) / (1 + r^2)^2, 1L, 1L)
      gr <- -w * 2 * r / (1 + r^2)
      list(logL = -w * log1p(r^2), grad = gr, negH = h,
           theta_grad = gr, theta_info = h, cross = h)
    }
    list(grad_hess = function(g, b) joint(g, b)[c("logL", "grad", "negH")],
         node_ll = function(g, Bn) apply(Bn, 1L, function(b) joint(g, b)$logL),
         joint = joint)
  }
  re_terms <- list(list(n_groups = ng, n_coefs = 1L, correlated = TRUE))
  fit <- tulpa_re_em(0, re_terms, list(matrix(0.09)), make_group = make_group,
                     theta_update = "latent")
  x <- fit$theta[1L] + fit$blup[[1L]][, 1L]
  expect_false(isTRUE(fit$stuck))
  expect_lt(abs(x[ng] - m[ng]), 0.1)
  expect_lt(max(abs(x[-ng] - m[-ng])), 0.1)
})
