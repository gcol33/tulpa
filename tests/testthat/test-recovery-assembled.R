# Parameter recovery for the ASSEMBLED estimators the front door dispatches
# (gcol33/tulpa#101): the individual kernels (mala / imh_laplace / pathfinder /
# smc) recover analytic truth in isolation, but the composed estimators -- the
# generic production NUTS that samples the variance components JOINTLY, EM+Laplace,
# and Tier-3 VI -- were validated for plumbing only. These add recovery / coverage.

# --------------------------------------------------------------------------- #
# (1) Generic production NUTS (mode = "hmc"): joint variance-component sampling. #
# This is the Tier-1 "exact MCMC" leg of the synthesis. A random-intercept       #
# Gaussian GLMM with known (beta, sigma_re) is fit through tulpa(mode = "hmc"),  #
# which samples (beta, {u_g}, log_sigma_re) jointly -- a distinct estimator from #
# conditioning on a fixed sigma_re. We assert the posterior recovers beta and    #
# that the marginalized sigma_re CI is calibrated.                               #
# --------------------------------------------------------------------------- #

test_that("generic NUTS recovers (beta, sigma_re) with calibrated CIs", {
  skip_on_cran()
  skip_if_fast()

  beta_true <- c(0.3, -0.6); sigma_re_true <- 0.8; sigma_obs <- 0.5
  G <- 20L; per <- 18L; n <- G * per
  n_seed <- 8L

  b_lo <- b_hi <- matrix(NA_real_, n_seed, 2L)
  b_hat <- matrix(NA_real_, n_seed, 2L)
  sre_hat <- numeric(n_seed); sre_cov <- logical(n_seed)

  for (s in seq_len(n_seed)) {
    set.seed(40L + s)
    grp <- factor(rep(seq_len(G), each = per))
    x   <- rnorm(n)
    u   <- rnorm(G, 0, sigma_re_true)
    y   <- beta_true[1] + beta_true[2] * x + u[as.integer(grp)] +
           rnorm(n, 0, sigma_obs)
    d   <- data.frame(y = y, x = x, g = grp)
    fit <- tulpa(y ~ x + (1 | g), data = d, family = "gaussian", mode = "hmc",
                 control = list(n_iter = 700L, n_warmup = 350L, seed = s))
    dr  <- fit$draws
    b_hat[s, ] <- colMeans(dr[, c("(Intercept)", "x"), drop = FALSE])
    b_lo[s, ]  <- apply(dr[, c("(Intercept)", "x"), drop = FALSE], 2L,
                        stats::quantile, 0.025)
    b_hi[s, ]  <- apply(dr[, c("(Intercept)", "x"), drop = FALSE], 2L,
                        stats::quantile, 0.975)
    sre <- exp(dr[, "log_sigma_re"])
    sre_hat[s] <- mean(sre)
    q <- stats::quantile(sre, c(0.025, 0.975))
    sre_cov[s] <- sigma_re_true >= q[1] && sigma_re_true <= q[2]
  }

  # Across-seed bias: the joint sampler is approximately unbiased for beta.
  expect_lt(abs(mean(b_hat[, 1]) - beta_true[1]), 0.20)
  expect_lt(abs(mean(b_hat[, 2]) - beta_true[2]), 0.15)
  # beta CI coverage (the slope is well-identified; the intercept is noisier).
  cov_b2 <- mean(b_lo[, 2] <= beta_true[2] & beta_true[2] <= b_hi[, 2])
  expect_gte(cov_b2, 0.7)
  # The variance component recovers and its marginalized CI is calibrated -- the
  # only way to confirm JOINT sampling of sigma_re (not conditioning) is unbiased.
  expect_lt(abs(median(sre_hat) / sigma_re_true - 1), 0.30)
  expect_gte(mean(sre_cov), 0.6)
})

# --------------------------------------------------------------------------- #
# (2) Tier-3 VI (mode = "vi"): recovery vs truth (not just self-consistency).   #
# Lower stakes -- auto never selects Tier-3 -- but it is an exposed fitter. VI   #
# is a Gaussian approximation that under-disperses, so the well-identified slope #
# is the reliable recovery target; the intercept is confounded with the RE mean  #
# and is left looser.                                                            #
# --------------------------------------------------------------------------- #

test_that("Tier-3 VI recovers the well-identified slope", {
  skip_on_cran()
  skip_if_fast()
  beta_true <- c(0.3, -0.6); sigma_re_true <- 0.8; sigma_obs <- 0.5
  G <- 20L; per <- 18L; n <- G * per
  slope <- numeric(6L)
  for (s in seq_len(6L)) {
    set.seed(60L + s)
    grp <- factor(rep(seq_len(G), each = per)); x <- rnorm(n)
    u   <- rnorm(G, 0, sigma_re_true)
    y   <- beta_true[1] + beta_true[2] * x + u[as.integer(grp)] +
           rnorm(n, 0, sigma_obs)
    fit <- tulpa(y ~ x + (1 | g), data = data.frame(y = y, x = x, g = grp),
                 family = "gaussian", mode = "vi", control = list(seed = s))
    slope[s] <- mean(fit$draws[, "x"])
  }
  expect_lt(abs(mean(slope) - beta_true[2]), 0.20)   # slope recovers vs truth
})

