# test-nested-laplace-var-unit-trials.R
# Validates the var_unit_trials predictive-variance rescaling in the
# multi-block nested-Laplace dispatch, together with the fitted_eta /
# fitted_eta_var grid outputs it rides on.
#
# Math under test (binomial latent Gaussian model). The fitting Hessian in the
# latent space is
#     H_fit = sum_i n_i w_i a_i a_i' + Q,      w_i = mu_i (1 - mu_i),
# where a_i is row i's loading vector and Q the prior precision. Inflating every
# trial count to a constant M (as an occupancy M-step does, so the binomial mode
# equals the E-step weighted mean) scales the likelihood block by M:
#     H_fit = M G + Q,     G = sum_i w_i a_i a_i'.
# The per-row predictive variance a_i' H^{-1} a_i is therefore deflated relative
# to the natural unit-trial Hessian H_unit = G + Q. Since M G + Q >= G + Q in the
# Loewner order for M >= 1, (M G + Q)^{-1} <= (G + Q)^{-1}, hence
#     var_unit(i) = a_i'(G + Q)^{-1} a_i  >=  a_i'(M G + Q)^{-1} a_i = var_fit(i),
# with the ratio bounded in (1, M]: it tends to M as Q -> 0 (likelihood-
# dominated) and to 1 as Q -> inf (prior-dominated). var_unit_trials = TRUE reads
# the variance off H_unit. The mode (and hence fitted_eta and the evidence) is
# the M-inflated fit and is left untouched -- only the curvature is rescaled.

sim_iid <- function(seed = 1L, n_units = 25L, reps = 5L, M = 1L,
                    beta = c(-0.3, 0.6), sigma_u = 0.7) {
  set.seed(seed)
  N <- n_units * reps
  unit_idx <- rep(seq_len(n_units), each = reps)
  x <- rnorm(N)
  X <- cbind(1, x)
  u <- rnorm(n_units, 0, sigma_u)
  eta <- as.numeric(X %*% beta) + u[unit_idx]
  list(
    y        = rbinom(N, size = M, prob = plogis(eta)),
    X        = X,
    n_trials = rep(as.integer(M), N),
    unit_idx = as.integer(unit_idx),
    n_units  = n_units,
    reps     = reps,
    N        = N
  )
}

fit_vut <- function(d, var_unit_trials, sigma_grid = c(0.3, 0.7, 1.2)) {
  prior <- list(list(
    type       = "iid",
    obs_idx    = d$unit_idx,
    n_units    = as.integer(d$n_units),
    sigma_grid = sigma_grid
  ))
  suppressWarnings(tulpa_nested_laplace(
    y = d$y, n_trials = d$n_trials, X = d$X,
    prior = prior, family = "binomial",
    max_iter = 50L, tol = 1e-7, n_threads = 1L,
    var_unit_trials = var_unit_trials
  ))
}

test_that("multi-block fit exposes fitted_eta / fitted_eta_var on the grid", {
  d  <- sim_iid(seed = 11L, M = 1L)
  f  <- fit_vut(d, FALSE)
  ng <- nrow(f$theta_grid)

  expect_true(is.matrix(f$fitted_eta))
  expect_true(is.matrix(f$fitted_eta_var))
  expect_equal(dim(f$fitted_eta), c(ng, d$N))
  expect_equal(dim(f$fitted_eta_var), c(ng, d$N))
  expect_true(all(is.finite(f$fitted_eta)))
  expect_true(all(is.finite(f$fitted_eta_var)))
  expect_true(all(f$fitted_eta_var >= 0))
})

test_that("var_unit_trials is a no-op when trials are already unit", {
  d  <- sim_iid(seed = 11L, M = 1L)
  f0 <- fit_vut(d, FALSE)
  f1 <- fit_vut(d, TRUE)

  # The flag never touches the mode, so the mode-derived fitted_eta and the
  # evidence are identical regardless of the flag.
  expect_equal(f0$fitted_eta, f1$fitted_eta, tolerance = 1e-10)
  expect_equal(f0$log_marginal, f1$log_marginal, tolerance = 1e-10)

  # With n_trials == 1, n_trials_unit == n_trials, so H_unit == H_fit and the
  # rescaled variance reproduces the default to solver precision. This is a
  # genuine exactness check on the unit-trial Hessian reassembly + back-solve
  # path against the resident-factor path the default uses.
  expect_equal(f0$fitted_eta_var, f1$fitted_eta_var, tolerance = 1e-5)
})

test_that("var_unit_trials inflates predictive variance to the unit-trial scale", {
  M  <- 30L
  d  <- sim_iid(seed = 12L, M = M)
  f0 <- fit_vut(d, FALSE)   # variance off the M-inflated fitting Hessian
  f1 <- fit_vut(d, TRUE)    # variance off the natural unit-trial Hessian

  # Curvature only: mode and evidence unchanged.
  expect_equal(f0$fitted_eta, f1$fitted_eta, tolerance = 1e-10)
  expect_equal(f0$log_marginal, f1$log_marginal, tolerance = 1e-10)

  v0 <- as.numeric(f0$fitted_eta_var)
  v1 <- as.numeric(f1$fitted_eta_var)
  keep <- is.finite(v0) & is.finite(v1) & v0 > 1e-10
  expect_true(any(keep))

  # Loewner: the unit-trial variance is never smaller than the M-inflated one.
  expect_true(all(v1[keep] >= v0[keep] - 1e-8))

  # Ratio bounded in (1, M]: -> M as the prior vanishes, -> 1 as it dominates.
  ratio <- v1[keep] / v0[keep]
  expect_true(all(ratio <= M + 1e-6))
  expect_true(all(ratio >= 1 - 1e-6))

  # The inflation is real, not a rounding no-op, with M = 30 informative trials.
  expect_gt(median(ratio), 1.5)
})

test_that("var_unit_trials yields finite predictive variance for held-out rows", {
  M    <- 20L
  reps <- 5L
  d    <- sim_iid(seed = 13L, M = M, reps = reps)

  # Hold out the first rep of every unit (n_trials = 0). Each held-out unit is
  # still observed in its remaining reps, so its random effect stays identified
  # and the held-out row is a genuine out-of-sample prediction.
  held <- seq(1L, d$N, by = reps)
  d$n_trials[held] <- 0L
  d$y[held]        <- 0L

  f <- fit_vut(d, TRUE)
  expect_true(all(is.finite(f$fitted_eta[, held])))
  expect_true(all(is.finite(f$fitted_eta_var[, held])))
  expect_true(all(f$fitted_eta_var[, held] >= 0))
})
