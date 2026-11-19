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

skip_on_cran()

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
# (3) EM+Laplace + MI/Gibbs corrections: recovery on all three paths, and the    #
# latent-state variance only the corrections carry.                              #
# Model: a single-season occupancy model with KNOWN detection. Sites have        #
# occupancy psi_i = plogis(b0 + b1 x_i); a site is detected on >= 1 of J visits   #
# with per-visit probability p. The E-step posterior-weights latent occupancy at #
# the zero-detection sites; the M-step fits the occupancy submodel as the        #
# weighted Bernoulli those soft labels define -- two rows per site, y = 1 at     #
# weight w and y = 0 at weight 1 - w (gcol33/tulpa#383).                         #
#                                                                                #
# That encoding maximizes the exact expected complete-data log-likelihood, so    #
# the raw EM point estimate is already at the MLE and MI / Gibbs have no point   #
# bias to remove: measured over these 8 seeds the median |b0 - truth| is 0.0817  #
# raw and 0.0863 under MI. What the corrections add is the latent-state          #
# uncertainty the raw Laplace conditions away. Under MI, V_between is            #
# identically 0 on the raw path and the pooled V_total exceeds V_within. Gibbs   #
# samples the joint posterior p(beta, z | y), so its posterior SD exceeds the    #
# complete-data SE at the EM weights: on seeds 51-53 it read 0.104, 0.108, 0.113 #
# for b0 against complete-data SEs of 0.094, 0.094, 0.096 and Louis SEs of       #
# 0.105, 0.103, 0.110.                                                           #
# --------------------------------------------------------------------------- #

test_that("EM+Laplace occupancy recovery; MI/Gibbs add the latent-state variance", {
  skip_if_not_slow()

  b0 <- 0.2; b1 <- -0.8; p_det <- 0.45; J <- 4L; G <- 500L
  q0 <- (1 - p_det)^J                             # P(0 detections | occupied)

  sim_occ <- function(seed) {
    set.seed(seed); x <- rnorm(G)
    z <- rbinom(G, 1L, plogis(b0 + b1 * x))
    h <- ifelse(z == 1L, rbinom(G, J, p_det), 0L) # detections out of J visits
    list(x = x, h = h)
  }
  em_occ <- function(d, correction = "none", n_gibbs = 2000L) {
    X <- cbind(1, d$x)
    e_step <- function(fits, ...) {
      psi <- if (length(fits) == 0L) rep(0.5, G)
             else plogis(as.numeric(X %*% fits$psi$mode))
      w <- ifelse(d$h > 0L, 1, psi * q0 / (psi * q0 + (1 - psi)))
      list(weights = as.numeric(w))
    }
    m_step_encode <- function(weights, ...) list(
      psi = list(y = rep(c(1, 0), each = G), n_trials = rep(1L, 2L * G),
                 X = rbind(X, X), weights = c(weights, 1 - weights),
                 family = "binomial")
    )
    tulpa_em_laplace(e_step, m_step_encode, correction = correction,
                     max_iter = 200L, tol = 1e-6, damping = 0,
                     n_imputations = 20L, n_gibbs = n_gibbs, verbose = FALSE)
  }

  n_seed <- 8L
  raw_b0 <- raw_b1 <- numeric(n_seed)
  mi_b0  <- mi_b1  <- mi_vb  <- mi_vw <- mi_vt <- numeric(n_seed)
  gi_b0  <- gi_b1  <- gi_sd  <- cd_se <- numeric(n_seed)
  for (s in seq_len(n_seed)) {
    d  <- sim_occ(50L + s)
    raw <- em_occ(d, "none")$fits$psi
    nm <- raw$mode
    mi <- em_occ(d, "mi"); gi <- em_occ(d, "gibbs")
    raw_b0[s] <- nm[1];               raw_b1[s] <- nm[2]
    cd_se[s]  <- raw$se[1]
    mi_b0[s]  <- mi$pooled$psi$mean[1]; mi_b1[s] <- mi$pooled$psi$mean[2]
    gi_b0[s]  <- gi$means[["psi:beta1"]]; gi_b1[s] <- gi$means[["psi:beta2"]]
    gi_sd[s]  <- gi$sds[["psi:beta1"]]
    mi_vb[s]  <- mi$pooled$psi$V_between[1]
    mi_vw[s]  <- mi$pooled$psi$V_within[1]; mi_vt[s] <- mi$pooled$psi$V_total[1]
  }

  # (a) all three paths recover both coefficients. Measured medians over these
  # seeds: raw (0.194, -0.715), MI (0.197, -0.701) against truth (0.2, -0.8).
  for (v in list(raw_b0, mi_b0, gi_b0)) expect_lt(abs(stats::median(v) - b0), 0.10)
  for (v in list(raw_b1, mi_b1, gi_b1)) expect_lt(abs(stats::median(v) - b1), 0.20)

  # (b) the corrections do not degrade the raw EM's point estimate. The raw
  # M-step already maximizes the exact expected complete-data log-likelihood,
  # so this is a no-regression bound, not a bias reduction.
  expect_lt(stats::median(abs(mi_b0 - b0)), stats::median(abs(raw_b0 - b0)) + 0.05)
  expect_lt(stats::median(abs(gi_b0 - b0)), stats::median(abs(raw_b0 - b0)) + 0.05)

  # (c) the corrections carry the latent-occupancy uncertainty the raw Laplace
  # EM conditions away: MI through a strictly positive between-imputation
  # variance (V_between == 0 on the raw path), Gibbs through a posterior SD
  # above the complete-data SE at the EM weights.
  expect_true(all(mi_vb >= 0))
  expect_gt(stats::median(mi_vb), 0)
  expect_true(all(mi_vt > mi_vw))
  expect_true(all(gi_sd > cd_se))
})

