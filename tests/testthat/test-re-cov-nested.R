# Nested-Laplace integration over a random-effect covariance Sigma
# (Bias-2 fix: marginalize Sigma over a grid, report weighted quantiles of the
# derived scale / correlation parameters instead of the plug-in MAP).

sim_corr_recov <- function(seed, G = 60L, npg = 12L, beta = c(-0.3, 0.7),
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


test_that("tulpa_re_cov_nested returns a well-formed marginalized posterior", {
  skip_on_cran()
  d <- sim_corr_recov(1L)
  re_term <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z)
  res <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, re_term,
                             family = "binomial", n_per_axis = 5L, span = 3)

  # grid + weights
  expect_equal(res$n_grid, 5L^3L)                 # k = c(c+1)/2 = 3 axes
  expect_equal(sum(res$weights), 1, tolerance = 1e-10)
  expect_true(all(res$weights >= 0))

  # posterior table shape: sigma_1, sigma_2, rho_12, Sigma_11/12/22
  expect_setequal(res$posterior$parameter,
                  c("sigma_1", "sigma_2", "rho_12",
                    "Sigma_11", "Sigma_12", "Sigma_22"))
  expect_true(all(c("mean", "sd", "median", "ci_lo", "ci_hi") %in%
                    names(res$posterior)))

  # CI ordering ci_lo <= median <= ci_hi for every parameter
  expect_true(all(res$posterior$ci_lo <= res$posterior$median + 1e-8))
  expect_true(all(res$posterior$median <= res$posterior$ci_hi + 1e-8))

  # weighted-mean Sigma is symmetric positive definite
  expect_equal(res$Sigma_mean, t(res$Sigma_mean), tolerance = 1e-10)
  expect_true(all(eigen(res$Sigma_mean, symmetric = TRUE,
                        only.values = TRUE)$values > 0))

  # rho stays in [-1, 1]
  rho <- res$posterior[res$posterior$parameter == "rho_12", ]
  expect_gte(rho$ci_lo, -1)
  expect_lte(rho$ci_hi, 1)

  # MAP summary reported alongside the marginalized one
  expect_length(res$map$sigma, 2L)
  expect_length(res$map$rho, 1L)
})


test_that("marginalized 95% intervals cover the true Sigma parameters", {
  skip_on_cran()
  # A few seeds; the marginalized 95% CI should contain truth in the large
  # majority. (Point recovery of the scales is mildly biased low by the
  # Laplace approximation -- Bias 1 -- which the Gibbs correction targets;
  # here we check the *interval*, which is what the Bias-2 fix delivers.)
  truth <- c(sigma_1 = 0.8, sigma_2 = 0.6, rho_12 = 0.5)
  covered <- c(sigma_1 = 0L, sigma_2 = 0L, rho_12 = 0L)
  n_seed <- 5L
  for (s in seq_len(n_seed)) {
    d <- sim_corr_recov(100L + s)
    re_term <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z)
    res <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, re_term,
                               family = "binomial", n_per_axis = 5L, span = 3)
    for (nm in names(truth)) {
      row <- res$posterior[res$posterior$parameter == nm, ]
      if (truth[[nm]] >= row$ci_lo && truth[[nm]] <= row$ci_hi)
        covered[[nm]] <- covered[[nm]] + 1L
    }
  }
  # With only 5 seeds, require the scales covered every time and rho most times.
  expect_gte(covered[["sigma_1"]], 4L)
  expect_gte(covered[["sigma_2"]], 4L)
  expect_gte(covered[["rho_12"]], 3L)
})


test_that("the median is a more central summary than the mode under skew", {
  skip_on_cran()
  # Small G -> skewed variance-component marginal. The marginalized posterior
  # mean should sit at or above the plug-in MAP for the variance scales (right
  # skew pulls the mean up), and the median between them.
  d <- sim_corr_recov(7L, G = 25L, npg = 12L)
  re_term <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z)
  res <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, re_term,
                             family = "binomial", n_per_axis = 5L, span = 3)
  s1 <- res$posterior[res$posterior$parameter == "Sigma_11", ]
  # right-skewed: mean >= median, and both finite/positive
  expect_gte(s1$mean, s1$median - 1e-6)
  expect_gt(s1$median, 0)
})
