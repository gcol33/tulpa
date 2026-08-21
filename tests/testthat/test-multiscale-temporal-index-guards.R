# Boundary checks on the multiscale temporal inputs (gcol33/tulpa#546).
#
# compute_temporal_eta reads the three arms at time_index[i] - 1 and guards each
# read, so an out-of-range index used to contribute zero rather than error --
# a mismatch between the index vector and the field length showed up as a
# quietly weaker temporal effect. The index is validated where it comes in
# instead, and n_obs / time_index are checked against each other in the kernel,
# which a LinkingTo caller filling MultiscaleTemporalData itself can reach.
#
# Structural (tier 1): no fit.

.ms_spec <- function(time_index, n_times, n_groups = NULL, group_index = NULL) {
  s <- list(type = "multiscale", time_index = as.integer(time_index),
            n_times = as.integer(n_times), trend = "rw1",
            seasonal_period = 0L, short_term = "none")
  if (!is.null(n_groups)) s$n_groups <- as.integer(n_groups)
  if (!is.null(group_index)) s$group_index <- as.integer(group_index)
  s
}

.ms_fit <- function(spec, n = 24L) {
  set.seed(2)
  x <- rnorm(n)
  tulpa_sample_glmm(as.numeric(rbinom(n, 1L, 0.5)), NULL, cbind(1, x),
                    "binomial", "hmc", temporal_spec = spec,
                    control = list(n_iter = 20L, warmup = 10L, n_chains = 1L,
                                   seed = 1L))
}

test_that("a time index outside [1, n_times] is refused, not zeroed", {
  n <- 24L
  t_ok <- rep(seq_len(6L), each = 4L)

  t_hi <- t_ok; t_hi[5] <- 7L                      # past the last time
  expect_error(.ms_fit(.ms_spec(t_hi, 6L), n), "time_index\\[5\\]")

  t_lo <- t_ok; t_lo[11] <- 0L                     # 0-based entry in a 1-based field
  expect_error(.ms_fit(.ms_spec(t_lo, 6L), n), "time_index\\[11\\]")

  t_neg <- t_ok; t_neg[2] <- -3L
  expect_error(.ms_fit(.ms_spec(t_neg, 6L), n), "must lie in")
})

test_that("n_times below one is refused", {
  expect_error(.ms_fit(.ms_spec(rep(1L, 24L), 0L)), "n_times")
})

test_that("group_index is checked for length and range", {
  n <- 24L
  t_ok <- rep(seq_len(6L), each = 4L)
  expect_error(
    .ms_fit(.ms_spec(t_ok, 6L, n_groups = 2L, group_index = rep(1L, n - 1L)), n),
    "length\\(group_index\\)")
  g_bad <- rep(1L, n); g_bad[7] <- 3L
  expect_error(
    .ms_fit(.ms_spec(t_ok, 6L, n_groups = 2L, group_index = g_bad), n),
    "group_index\\[7\\]")
})

test_that("a valid multiscale spec still builds", {
  fit <- .ms_fit(.ms_spec(rep(seq_len(6L), each = 4L), 6L))
  expect_true("log_sigma2_trend" %in% fit$param_names)
  expect_identical(sum(grepl("^trend[[]", fit$param_names)), 6L)
})

# The AR1 arm's density is the stationary one, whose 1 - rho^2 factor is floored.
# On the sampler path rho = 2 * inv_logit(logit_rho) - 1 and cannot leave the
# unit interval; this entry point takes it raw, so it rejects what the floor
# would otherwise absorb into a number that is not a density.
test_that("the multiscale log-lik entry point rejects a non-stationary rho", {
  phi <- rnorm(8)
  ok <- cpp_test_multiscale_temporal_log_lik(phi, numeric(0), phi, 1, 1, 1,
                                             0.5, "rw1", 0L, "ar1")
  expect_true(is.finite(ok))
  for (bad in c(1, -1, 1.5, -2, Inf, NaN)) {
    expect_error(
      cpp_test_multiscale_temporal_log_lik(phi, numeric(0), phi, 1, 1, 1,
                                           bad, "rw1", 0L, "ar1"),
      "rho_short")
  }
})
