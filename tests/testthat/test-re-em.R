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
