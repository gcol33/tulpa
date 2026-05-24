# Exact-target Gibbs estimation of a random-effect covariance Sigma
# (Bias-1 fix: sample the joint posterior p(beta, b, Sigma | y) so the
# variance components are not shrunk by the Laplace/PQL approximation of the
# non-Gaussian random-effect conditional).

sim_corr_gibbs <- function(seed, G = 50L, npg = 15L, beta = c(-0.3, 0.7),
                           Sigma = matrix(c(0.8^2, 0.5 * 0.8 * 0.6,
                                            0.5 * 0.8 * 0.6, 0.6^2), 2)) {
  set.seed(seed)
  N <- G * npg
  grp <- rep(seq_len(G), each = npg)
  x <- rnorm(N)
  X <- cbind(1, x)
  Z <- cbind(1, x)
  u <- t(t(chol(Sigma)) %*% matrix(rnorm(2 * G), nrow = 2))
  eta <- as.numeric(X %*% beta) + rowSums(Z * u[grp, ])
  y <- rbinom(N, 1L, plogis(eta))
  list(y = y, X = X, Z = Z, grp = grp, G = G, N = N, Sigma = Sigma)
}


test_that(".rinvwishart reproduces the inverse-Wishart mean", {
  skip_on_cran()
  set.seed(1L)
  nu <- 20L; p <- 2L
  Lambda <- matrix(c(2, 0.5, 0.5, 1), 2)
  draws <- replicate(8000L, tulpa:::.rinvwishart(nu, Lambda), simplify = FALSE)
  Sbar <- Reduce(`+`, draws) / length(draws)
  # E[Sigma] = Lambda / (nu - p - 1)
  expect_equal(Sbar, Lambda / (nu - p - 1L), tolerance = 0.05)
  # every draw is SPD
  expect_true(all(vapply(draws[1:200],
    function(S) min(eigen(S, symmetric = TRUE, only.values = TRUE)$values) > 0,
    logical(1))))
})


test_that("tulpa_re_cov_gibbs returns a well-formed posterior", {
  skip_on_cran()
  d <- sim_corr_gibbs(1L)
  re_term <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z)
  res <- tulpa_re_cov_gibbs(d$y, rep(1L, d$N), d$X, re_term,
                            family = "binomial",
                            n_iter = 1500L, n_burnin = 800L, seed = 11L)

  expect_equal(res$n_kept, 1500L)
  expect_length(res$Sigma_draws, 1500L)
  expect_equal(dim(res$beta_draws), c(1500L, 2L))

  expect_setequal(res$posterior$parameter,
                  c("sigma_1", "sigma_2", "rho_12",
                    "Sigma_11", "Sigma_12", "Sigma_22"))
  expect_true(all(res$posterior$ci_lo <= res$posterior$median + 1e-8))
  expect_true(all(res$posterior$median <= res$posterior$ci_hi + 1e-8))

  # Sigma_mean SPD; rho in range
  expect_equal(res$Sigma_mean, t(res$Sigma_mean), tolerance = 1e-10)
  expect_true(all(eigen(res$Sigma_mean, symmetric = TRUE,
                        only.values = TRUE)$values > 0))
  rho <- res$posterior[res$posterior$parameter == "rho_12", ]
  expect_gte(rho$ci_lo, -1)
  expect_lte(rho$ci_hi, 1)

  # adapted acceptance rates land in a sane Metropolis window
  expect_gt(res$accept$beta, 0.05)
  expect_lt(res$accept$beta, 0.95)
  expect_gt(res$accept$b, 0.05)
  expect_lt(res$accept$b, 0.95)
})


test_that("Gibbs debias recovers Sigma with covering intervals", {
  skip_on_cran()
  truth <- c(sigma_1 = 0.8, sigma_2 = 0.6, rho_12 = 0.5)
  covered <- c(sigma_1 = 0L, sigma_2 = 0L, rho_12 = 0L)
  med <- matrix(NA_real_, 4L, 3L, dimnames = list(NULL, names(truth)))
  for (s in seq_len(4L)) {
    d <- sim_corr_gibbs(200L + s)
    re_term <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z)
    res <- tulpa_re_cov_gibbs(d$y, rep(1L, d$N), d$X, re_term,
                              family = "binomial",
                              n_iter = 1500L, n_burnin = 800L,
                              seed = 500L + s)
    for (nm in names(truth)) {
      row <- res$posterior[res$posterior$parameter == nm, ]
      med[s, nm] <- row$median
      if (truth[[nm]] >= row$ci_lo && truth[[nm]] <= row$ci_hi)
        covered[[nm]] <- covered[[nm]] + 1L
    }
  }
  # 95% intervals cover truth in the large majority of the 4 seeds
  expect_gte(covered[["sigma_1"]], 3L)
  expect_gte(covered[["sigma_2"]], 3L)
  expect_gte(covered[["rho_12"]], 2L)

  # Point recovery: the posterior-median scales sit near truth (no gross
  # downward shrink). Averaged over seeds, within ~25% of the true scales.
  expect_lt(abs(mean(med[, "sigma_1"]) - 0.8) / 0.8, 0.25)
  expect_lt(abs(mean(med[, "sigma_2"]) - 0.6) / 0.6, 0.25)
})
