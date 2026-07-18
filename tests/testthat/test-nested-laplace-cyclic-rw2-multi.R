# Guards that a cyclic RW2 temporal block is honored on the MULTI-block
# nested-Laplace path (#218). The single-block and exact-NUTS paths already
# thread `cyclic`; before the fix the multi-block driver hardcoded the RW2
# penalty and rank to the acyclic form, so `cyclic = TRUE` silently changed
# nothing once a second latent block was present. The wrap-around
# second-difference penalty and the T-1 (vs T-2) rank normalizer both move the
# inner marginal, so a correctly-wired flag must change log_marginal.

test_that("cyclic RW2 changes the multi-block nested-Laplace fit", {
  skip_on_cran()
  set.seed(7)
  n_times <- 12L
  n_units <- 8L
  reps    <- 6L
  N       <- n_times * n_units * reps

  time_idx <- rep(rep(seq_len(n_times), each = n_units), times = reps)
  unit_idx <- rep(rep(seq_len(n_units), times = n_times), times = reps)

  # A periodic seasonal curve: cyclic vs acyclic disagree most at the year seam.
  season <- 1.2 * sin(2 * pi * seq_len(n_times) / n_times)
  iota   <- rnorm(n_units, sd = 0.3)
  eta    <- -0.2 + season[time_idx] + iota[unit_idx]
  y      <- rbinom(N, 1L, plogis(eta))
  X      <- matrix(1, N, 1L)

  rw2_block <- function(cyclic) list(
    type = "rw2", temporal_idx = as.integer(time_idx),
    n_times = n_times, tau_grid = c(1, 4, 12, 30), cyclic = cyclic
  )
  iid_block <- list(
    type = "iid", obs_idx = as.integer(unit_idx),
    n_units = n_units, sigma_grid = c(0.1, 0.3, 0.6)
  )

  fit_c <- suppressWarnings(tulpa_nested_laplace(
    y = y, n_trials = rep(1L, N), X = X, family = "binomial",
    prior = list(rw2_block(TRUE), iid_block),
    control = list(max_iter = 30L, tol = 1e-6, n_threads = 1L, diagnose_k = FALSE)))
  fit_a <- suppressWarnings(tulpa_nested_laplace(
    y = y, n_trials = rep(1L, N), X = X, family = "binomial",
    prior = list(rw2_block(FALSE), iid_block),
    control = list(max_iter = 30L, tol = 1e-6, n_threads = 1L, diagnose_k = FALSE)))

  # The flag must move the inner marginal: identical log_marginal vectors would
  # mean cyclic was dropped (the pre-#218 behavior).
  expect_equal(length(fit_c$log_marginal), length(fit_a$log_marginal))
  expect_false(isTRUE(all.equal(fit_c$log_marginal, fit_a$log_marginal,
                                tolerance = 1e-8)))
})
