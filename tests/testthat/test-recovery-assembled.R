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
  skip_if_not_slow()

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
                 control = list(n_iter = 700L, warmup = 350L, seed = s))
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
  skip_if_not_slow()
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


# --------------------------------------------------------------------------- #
# (3) EM+Laplace + MI/Gibbs corrections: recovery, bias reduction, and the       #
# latent-state variance the raw EM conditions away.                              #
# Model: a single-season occupancy model with KNOWN detection. Sites have        #
# occupancy psi_i = plogis(b0 + b1 x_i); a site is detected on >= 1 of J visits   #
# with per-visit probability p. The E-step posterior-weights latent occupancy at #
# the zero-detection sites; the M-step fits the occupancy submodel with the      #
# E-step weight as the binomial RESPONSE (the documented pseudo-binomial         #
# encoding -- no fractional `weights` argument, so this path is unaffected by    #
# gcol33/tulpa#108). The raw Laplace M-step conditions on the soft weights, so   #
# it under-counts occupancy at the zero-detection sites and leaves the intercept #
# biased (and the CI carries no latent-occupancy uncertainty, V_between == 0).   #
# MI/Gibbs draw the hard occupancy states and pool via Rubin: V_between > 0 and   #
# the pooled intercept moves back toward truth -- the bias reduction the         #
# corrections exist for.                                                         #
# --------------------------------------------------------------------------- #

test_that("EM+Laplace occupancy recovery; MI/Gibbs reduce intercept bias", {
  skip_if_not_slow()

  b0 <- 0.2; b1 <- -0.8; p_det <- 0.45; J <- 4L; G <- 500L
  q0 <- (1 - p_det)^J                             # P(0 detections | occupied)

  sim_occ <- function(seed) {
    set.seed(seed); x <- rnorm(G)
    z <- rbinom(G, 1L, plogis(b0 + b1 * x))
    h <- ifelse(z == 1L, rbinom(G, J, p_det), 0L) # detections out of J visits
    list(x = x, h = h)
  }
  em_occ <- function(d, correction = "none") {
    X <- cbind(1, d$x)
    e_step <- function(fits, ...) {
      psi <- if (length(fits) == 0L) rep(0.5, G)
             else plogis(as.numeric(X %*% fits$psi$mode))
      w <- ifelse(d$h > 0L, 1, psi * q0 / (psi * q0 + (1 - psi)))
      list(weights = as.numeric(w))
    }
    m_step_encode <- function(weights, ...) list(
      psi = list(y = weights, n_trials = rep(1L, G), X = X, family = "binomial")
    )
    tulpa_em_laplace(e_step, m_step_encode, correction = correction,
                     max_iter = 200L, tol = 1e-6, damping = 0,
                     n_imputations = 20L, n_gibbs = 12L, verbose = FALSE)
  }

  n_seed <- 8L
  raw_b0 <- raw_b1 <- numeric(n_seed)
  mi_b0  <- mi_b1  <- mi_vb  <- numeric(n_seed)
  gi_b0  <- gi_b1  <- gi_vb  <- numeric(n_seed)
  for (s in seq_len(n_seed)) {
    d  <- sim_occ(50L + s)
    nm <- em_occ(d, "none")$fits$psi$mode
    mi <- em_occ(d, "mi"); gi <- em_occ(d, "gibbs")
    raw_b0[s] <- nm[1];               raw_b1[s] <- nm[2]
    mi_b0[s]  <- mi$pooled$psi$mean[1]; mi_b1[s] <- mi$pooled$psi$mean[2]
    gi_b0[s]  <- gi$pooled$psi$mean[1]; gi_b1[s] <- gi$pooled$psi$mean[2]
    mi_vb[s]  <- mi$pooled$psi$V_between[1]
    gi_vb[s]  <- gi$pooled$psi$V_between[1]
  }

  # (a) the well-identified slope is recovered on the raw EM path.
  expect_lt(abs(stats::median(raw_b1) - b1), 0.30)

  # (b) MI/Gibbs recover the intercept that the raw EM leaves biased.
  expect_lt(abs(stats::median(mi_b0) - b0), 0.15)
  expect_lt(abs(stats::median(gi_b0) - b0), 0.15)
  expect_lt(abs(stats::median(mi_b1) - b1), 0.25)
  expect_lt(abs(stats::median(gi_b1) - b1), 0.25)

  # (c) the corrections reduce the raw EM's intercept bias.
  raw_err <- stats::median(abs(raw_b0 - b0))
  expect_lt(stats::median(abs(mi_b0 - b0)), raw_err)
  expect_lt(stats::median(abs(gi_b0 - b0)), raw_err)

  # (d) the corrections carry a strictly positive between-imputation variance --
  # the latent-occupancy uncertainty the raw Laplace EM conditions away
  # (V_between == 0 there), the basis for the corrected CI's coverage >= raw EM.
  expect_true(all(mi_vb >= 0) && all(gi_vb >= 0))
  expect_gt(stats::median(mi_vb), 0)
  expect_gt(stats::median(gi_vb), 0)
})

