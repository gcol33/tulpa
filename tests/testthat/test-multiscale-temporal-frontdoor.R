# Front-door multiscale temporal (gcol33/tulpa#241 follow-up).
#
# tulpa(y ~ x, temporal = temporal_multiscale("t", ...), mode = "hmc") routes
# through the generic ModelData sampler: build_sampler_model_inputs() parses the
# validated spec onto MultiscaleTemporalData and sets has_multiscale_temporal,
# compute_param_layout() allocates the trend / seasonal / short-term arms plus
# their scales, and the generic log-posterior adds ms_temporal_eta.
#
# That flag was previously never set anywhere in tulpa -- only consumer packages
# reached the block through the ModelData ABI -- so the whole multiscale stack
# was unreachable from this package and its behaviour could only be asserted by
# calling the log-prior kernel directly. These exercise it through the actual
# construction and fitting path instead.
#
# Multiscale is sampler-only: there is no multiscale nested-Laplace kernel.

make_ms_binom <- function(n_t = 36L, reps = 8L, seed = 3L) {
  set.seed(seed)
  trend <- cumsum(rnorm(n_t, 0, 0.25)); trend <- trend - mean(trend)
  time <- rep(seq_len(n_t), each = reps); N <- length(time)
  x <- rnorm(N)
  ntr <- rep(8L, N)
  y <- rbinom(N, ntr, plogis(0.4 - 0.6 * x + trend[time]))
  list(d = data.frame(y = y, x = x, t = time, ntrials = ntr),
       trend = trend, n_t = n_t, b1 = -0.6)
}

test_that("multiscale temporal is refused by the nested-Laplace modes", {
  s <- make_ms_binom(n_t = 6L, reps = 4L)
  expect_error(
    tulpa(y ~ x, data = s$d, family = "binomial", n_trials = s$d$ntrials,
          temporal = temporal_multiscale("t", trend = "rw1", short_term = "none"),
          mode = "nested_laplace"),
    "multiscale"
  )
})

test_that("an unsupported multiscale component is refused by name", {
  d <- data.frame(t = rep(1:5, 2))
  expect_error(temporal_multiscale("t", trend = "rw3"), "should be one of")
  expect_error(temporal_multiscale("t", short_term = "arma"), "should be one of")
  # Every component switched off is refused at construction, not at fit time.
  expect_error(temporal_multiscale("t", trend = "none", short_term = "none"),
               "At least one temporal component")
})

test_that("multiscale temporal allocates and names its arms (structural)", {
  skip_if_not_slow()
  s <- make_ms_binom(n_t = 12L, reps = 4L)
  fit <- tulpa(y ~ x, data = s$d, family = "binomial", n_trials = s$d$ntrials,
               temporal = temporal_multiscale("t", trend = "rw1", seasonal = 4,
                                              short_term = "ar1"),
               mode = "hmc",
               control = list(n_iter = 120L, n_warmup = 60L, seed = 1L,
                              verbose = FALSE))
  pn <- colnames(fit$draws)
  # One scale per active arm, the AR1 correlation, and one value per node:
  # trend over n_times, seasonal over the period, short-term over n_times.
  expect_true(all(c("log_sigma2_trend", "log_sigma2_seasonal",
                    "log_sigma2_short", "logit_rho_short") %in% pn))
  expect_equal(sum(grepl("^trend\\[", pn)), 12L)
  expect_equal(sum(grepl("^seasonal\\[", pn)), 4L)
  expect_equal(sum(grepl("^short_term\\[", pn)), 12L)
  # 2 fixed + 4 hyper + 12 + 4 + 12.
  expect_equal(ncol(fit$draws), 2L + 4L + 12L + 4L + 12L)
})

test_that("a switched-off arm allocates nothing", {
  skip_if_not_slow()
  s <- make_ms_binom(n_t = 10L, reps = 4L)
  fit <- tulpa(y ~ x, data = s$d, family = "binomial", n_trials = s$d$ntrials,
               temporal = temporal_multiscale("t", trend = "rw1",
                                              short_term = "none"),
               mode = "hmc",
               control = list(n_iter = 120L, n_warmup = 60L, seed = 2L,
                              verbose = FALSE))
  pn <- colnames(fit$draws)
  expect_true("log_sigma2_trend" %in% pn)
  expect_false(any(grepl("^seasonal\\[|^short_term\\[", pn)))
  expect_false(any(c("log_sigma2_seasonal", "log_sigma2_short",
                     "logit_rho_short") %in% pn))
  expect_equal(ncol(fit$draws), 2L + 1L + 10L)
})

test_that("multiscale temporal recovers the trend through the front door", {
  skip_if_not_slow()
  s <- make_ms_binom()
  fit <- tulpa(y ~ x, data = s$d, family = "binomial", n_trials = s$d$ntrials,
               temporal = temporal_multiscale("t", trend = "rw1",
                                              short_term = "none"),
               mode = "hmc",
               control = list(n_iter = 1500L, n_warmup = 750L, seed = 5L,
                              verbose = FALSE))
  dr <- as.matrix(fit$draws)
  tcol <- grep("^trend\\[", colnames(dr))
  expect_equal(length(tcol), s$n_t)

  # The trend shape is the identified quantity: an RW1 level is confounded with
  # the intercept, so the constant is not asserted.
  th <- colMeans(dr[, tcol, drop = FALSE])
  expect_gt(cor(th, s$trend), 0.85)

  # The covariate slope is identified separately from the walk.
  expect_lt(abs(unname(coef(fit)["x"]) - s$b1), 0.2)
})
