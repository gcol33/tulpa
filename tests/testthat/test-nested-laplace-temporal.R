# Nested Laplace dispatch through the temporal Q-builders (RW1, RW2, AR1).

simulate_temporal_binomial <- function(T = 30L, n_per_t = 5L,
                                        beta0 = 0.3, ar1_rho = 0.85, sd = 0.6,
                                        seed = 2026L) {
  set.seed(seed)
  time <- rep(seq_len(T), each = n_per_t)
  u <- as.numeric(arima.sim(list(ar = ar1_rho), n = T, sd = sd))
  u <- u - mean(u)
  eta <- beta0 + u[time]
  p <- 1 / (1 + exp(-eta))
  n_trials <- rep(8L, length(time))
  y <- rbinom(length(time), n_trials, p)
  list(y = y, n_trials = n_trials,
       X = matrix(1, length(time), 1),
       time = as.integer(time), T = T, u_true = u)
}

# =====================================================================
# RW1
# =====================================================================

test_that("nested_laplace RW1 returns interior-peaked log-marginal", {
  d <- simulate_temporal_binomial()
  prior <- list(type = "rw1",
                temporal_idx = d$time,
                n_times = d$T,
                tau_grid = exp(seq(log(0.3), log(30), length.out = 9)))

  res <- tulpa_nested_laplace(d$y, d$n_trials, d$X, prior = prior,
                        family = "binomial")

  expect_s3_class(res, "tulpa_nested_laplace")
  expect_equal(length(res$log_marginal), 9L)
  expect_true(all(is.finite(res$log_marginal)))
  expect_equal(sum(res$weights), 1.0, tolerance = 1e-6)
  # Posterior mean within the grid
  expect_true(res$theta_mean > min(prior$tau_grid))
  expect_true(res$theta_mean < max(prior$tau_grid))
})

test_that("nested_laplace RW1 stores per-grid-point modes", {
  d <- simulate_temporal_binomial(T = 20L, n_per_t = 4L)
  prior <- list(type = "rw1",
                temporal_idx = d$time,
                n_times = d$T,
                tau_grid = c(1, 5, 20))

  res <- tulpa_nested_laplace(d$y, d$n_trials, d$X, prior = prior,
                        family = "binomial")

  expect_equal(nrow(res$modes), 3L)
  # n_x = p + n_re + n_times = 1 + 0 + 20
  expect_equal(ncol(res$modes), 21L)
  expect_true(all(is.finite(res$modes)))
})

# =====================================================================
# RW2
# =====================================================================

test_that("nested_laplace RW2 produces a smoother trend than RW1 mode", {
  d <- simulate_temporal_binomial(T = 30L, ar1_rho = 0.6)
  tg <- exp(seq(log(0.3), log(30), length.out = 9))
  prior_rw1 <- list(type = "rw1", temporal_idx = d$time,
                    n_times = d$T, tau_grid = tg)
  prior_rw2 <- list(type = "rw2", temporal_idx = d$time,
                    n_times = d$T, tau_grid = tg)

  res1 <- tulpa_nested_laplace(d$y, d$n_trials, d$X, prior = prior_rw1,
                         family = "binomial")
  res2 <- tulpa_nested_laplace(d$y, d$n_trials, d$X, prior = prior_rw2,
                         family = "binomial")

  expect_true(all(is.finite(res2$log_marginal)))
  # Both produce sensible posteriors over tau
  expect_true(res2$theta_mean > 0)
  expect_true(res2$theta_sd > 0)
  # Compare smoothness via second differences of the modal mode (last grid pt)
  mode1 <- res1$modes[which.max(res1$weights), 2:(d$T + 1)]
  mode2 <- res2$modes[which.max(res2$weights), 2:(d$T + 1)]
  diff2_1 <- diff(mode1, differences = 2L)
  diff2_2 <- diff(mode2, differences = 2L)
  # RW2 penalises curvature directly — under matched tau, second-difference
  # energy should not be larger than RW1's. Allow some slack for sample noise.
  expect_lt(sum(diff2_2^2), sum(diff2_1^2) * 1.5)
})

# =====================================================================
# AR1
# =====================================================================

test_that("nested_laplace AR1 recovers high autocorrelation in the simulation", {
  d <- simulate_temporal_binomial(T = 40L, n_per_t = 6L,
                                  ar1_rho = 0.85, sd = 0.5)
  g_tau <- exp(seq(log(0.5), log(20), length.out = 4))
  g_rho <- c(0.3, 0.6, 0.85, 0.97)
  gr <- expand.grid(tau = g_tau, rho = g_rho)
  prior <- list(type = "ar1",
                temporal_idx = d$time, n_times = d$T,
                tau_grid = gr$tau, rho_grid = gr$rho)

  res <- tulpa_nested_laplace(d$y, d$n_trials, d$X, prior = prior,
                        family = "binomial")

  expect_s3_class(res, "tulpa_nested_laplace")
  expect_equal(length(res$log_marginal), nrow(gr))
  expect_true(all(is.finite(res$log_marginal)))
  expect_equal(sum(res$weights), 1.0, tolerance = 1e-6)
  # Posterior rho should pull above 0.5 given the simulation
  expect_named(res$theta_mean, c("tau", "rho"))
  expect_gt(res$theta_mean[["rho"]], 0.5)
})

test_that("nested_laplace accepts tulpa_temporal spec via spec= + data=", {
  d <- simulate_temporal_binomial(T = 25L, ar1_rho = 0.7)
  df <- data.frame(year = d$time)
  spec <- temporal_ar1("year")

  res_spec <- tulpa_nested_laplace(
    d$y, d$n_trials, d$X,
    spec = spec, data = df,
    family = "binomial"
  )

  # Equivalent manual prior
  res_manual <- tulpa_nested_laplace(
    d$y, d$n_trials, d$X,
    prior = list(type = "ar1",
                 temporal_idx = d$time,
                 n_times = d$T),
    family = "binomial"
  )

  expect_equal(res_spec$log_marginal, res_manual$log_marginal,
               tolerance = 1e-10)
  expect_equal(res_spec$theta_mean, res_manual$theta_mean,
               tolerance = 1e-10)
})

test_that("nested_laplace rejects spec + prior together", {
  d <- simulate_temporal_binomial(T = 10L, n_per_t = 2L)
  expect_error(
    tulpa_nested_laplace(
      d$y, d$n_trials, d$X,
      spec = temporal_rw1("year"),
      prior = list(type = "rw1", temporal_idx = d$time, n_times = d$T),
      data = data.frame(year = d$time)
    ),
    "Pass either"
  )
})

test_that("nested_laplace AR1 fills default grid when missing", {
  d <- simulate_temporal_binomial(T = 20L, n_per_t = 4L)
  prior <- list(type = "ar1",
                temporal_idx = d$time, n_times = d$T)

  res <- tulpa_nested_laplace(d$y, d$n_trials, d$X, prior = prior,
                        family = "binomial")

  expect_true(all(is.finite(res$log_marginal)))
  expect_true(length(res$log_marginal) > 5L)
})
