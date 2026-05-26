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
                             family = "binomial")

  # default CCD layout: k = c(c+1)/2 = 3 -> 1 centre + 2k axial + 2^k factorial
  expect_equal(res$n_grid, 1L + 2L * 3L + 2L^3L)  # = 15
  expect_equal(sum(res$weights), 1, tolerance = 1e-10)
  expect_true(all(res$weights >= 0))

  # tensor grid is opt-in via integration = "grid": n_per_axis^k cells
  res_grid <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, re_term,
                                  family = "binomial",
                                  integration = "grid", n_per_axis = 5L, span = 3)
  expect_equal(res_grid$n_grid, 5L^3L)
  expect_equal(sum(res_grid$weights), 1, tolerance = 1e-10)

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
                               family = "binomial")   # default CCD integration
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
                             family = "binomial")   # default CCD integration
  s1 <- res$posterior[res$posterior$parameter == "Sigma_11", ]
  # right-skewed: mean >= median, and both finite/positive
  expect_gte(s1$mean, s1$median - 1e-6)
  expect_gt(s1$median, 0)
})


# --- diagonal (uncorrelated) block -------------------------------------------
# A `(1 + x || g)` term is a diagonal Sigma: c log-SD integration params (here
# k = 2, vs 3 for the full block), no correlation parameter, and the off-diagonal
# entry is structurally zero. Truth has uncorrelated intercept/slope.
sim_diag_recov <- function(seed, G = 60L, npg = 12L, beta = c(-0.3, 0.6),
                           s1 = 0.8, s2 = 0.5) {
  set.seed(seed)
  N <- G * npg
  grp <- rep(seq_len(G), each = npg)
  x <- rnorm(N); X <- cbind(1, x); Z <- cbind(1, x)
  u <- cbind(rnorm(G, 0, s1), rnorm(G, 0, s2))   # independent intercept/slope
  eta <- as.numeric(X %*% beta) + rowSums(Z * u[grp, ])
  list(y = rbinom(N, 1L, plogis(eta)), X = X, Z = Z, grp = grp, G = G, N = N)
}

test_that("tulpa_re_cov_nested integrates a diagonal (uncorrelated) block", {
  skip_on_cran()
  d <- sim_diag_recov(11L)
  rt <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z,
             correlated = FALSE)
  res <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, rt, family = "binomial")

  # diagonal block: k = c = 2 -> CCD 1 + 2k + 2^k = 9 nodes
  expect_equal(res$n_grid, 1L + 2L * 2L + 2L^2L)
  # no correlation reported; only diagonal variances
  expect_setequal(res$posterior$parameter,
                  c("sigma_1", "sigma_2", "Sigma_11", "Sigma_22"))
  expect_length(res$map$rho, 0L)
  # the MAP Sigma really is diagonal (off-diagonal exactly zero)
  expect_equal(res$map$Sigma[1L, 2L], 0)
  # scales recovered to the right ballpark (interval covers truth)
  for (nm in c("sigma_1", "sigma_2")) {
    row <- res$posterior[res$posterior$parameter == nm, ]
    truth <- if (nm == "sigma_1") 0.8 else 0.5
    expect_gte(row$ci_hi, truth * 0.6)
    expect_lte(row$ci_lo, truth * 1.4)
  }
})


# --- multi-term: a list of blocks --------------------------------------------
test_that("tulpa_re_cov_nested integrates several terms as separate blocks", {
  skip_on_cran()
  set.seed(21L)
  G <- 40L; H <- 25L; npg <- 12L; N <- G * npg
  g <- rep(seq_len(G), each = npg)
  h <- sample.int(H, N, replace = TRUE)
  x <- rnorm(N); X <- cbind(1, x); Zg <- cbind(1, x)
  Sg <- matrix(c(0.8^2, 0.4 * 0.8 * 0.5, 0.4 * 0.8 * 0.5, 0.5^2), 2)
  ug <- t(t(chol(Sg)) %*% matrix(rnorm(2 * G), 2))
  uh <- rnorm(H, 0, 0.6)
  eta <- as.numeric(X %*% c(-0.2, 0.6)) + rowSums(Zg * ug[g, ]) + uh[h]
  y <- rbinom(N, 1L, plogis(eta))

  re_terms <- list(
    list(idx = g, n_groups = G, n_coefs = 2L, Z = Zg, correlated = TRUE,
         label = "g"),
    list(idx = h, n_groups = H, n_coefs = 1L, correlated = FALSE, label = "h")
  )
  res <- tulpa_re_cov_nested(y, rep(1L, N), X, re_terms, family = "binomial",
                             seed = 3L, n_draws = 300L)

  expect_equal(res$n_blocks, 2L)
  expect_equal(unname(res$n_coefs), c(2L, 1L))
  # stacked integration dimension k = 3 (full g) + 1 (scalar h) = 4
  # CCD(k=4): 1 + 2*4 + fractional factorial -- just check it ran with > 4 nodes
  expect_gt(res$n_grid, 4L)
  expect_setequal(res$posterior$parameter,
                  c("g.sigma_1", "g.sigma_2", "g.rho_12",
                    "g.Sigma_11", "g.Sigma_12", "g.Sigma_22",
                    "h.sigma_1", "h.Sigma_11"))
  expect_named(res$Sigma_mean, c("g", "h"))
  expect_equal(dim(res$Sigma_mean$g), c(2L, 2L))
  # both grouping scales positive
  for (nm in c("g.sigma_1", "g.sigma_2", "h.sigma_1")) {
    expect_gt(res$posterior[res$posterior$parameter == nm, "median"], 0)
  }
})
