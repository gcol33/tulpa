test_that("intrinsic TVC levels have a precision-independent standard normal density", {
  for (structure in c("rw1", "rw2")) for (n_times in c(1L, 2L, 8L)) for (n_groups in 1:2) {
    n <- 4L * n_times * n_groups
    x <- rep(c(-1, -.3, .3, 1), n_times * n_groups)
    X <- cbind(1, x)
    spec <- list(n_times = n_times, n_groups = n_groups, n_tvc = 1L,
      time_index = rep(rep(seq_len(n_times), each = 4L), n_groups),
      group_index = rep(seq_len(n_groups), each = 4L * n_times),
      tvc_indices = 2L, X_tvc = x, structure = structure)
    probe <- function(point = NULL) tulpa:::cpp_test_field_coordinates(
      y = .2 + .5 * x, X = X, tvc_spec = spec, shift = 0, point = point,
      native_point = FALSE)
    initial <- probe()
    field <- seq.int(4L, n_times * n_groups + 3L)
    for (log_tau in c(-8, 0, 8)) {
      q <- initial$q
      q[3L] <- log_tau
      zero <- probe(q)
      q[field] <- 2 / sqrt(n_times)
      level <- probe(q)
      expect_equal(mean(level$x[field]) * sqrt(n_times) * exp(.5 * log_tau),
                   2, tolerance = 1e-10)
      expect_equal(level$prior_lp + level$jacobian - zero$prior_lp - zero$jacobian,
                   -2 * n_groups, tolerance = 1e-8)
      expect_equal(level$prior_gradient[3L], zero$prior_gradient[3L], tolerance = 1e-8)
      expect_equal(level$inverse, q, tolerance = 1e-10)
      expect_equal(level$gradient, level$numerical, tolerance = 1e-5)
      expect_true(level$verified)
    }
  }
})

test_that("RW2 posterior means agree with dense Gaussian integration across seeds", {
  skip_if_not_slow()
  set.seed(1172)
  n_times <- 8L
  time <- rep(seq_len(n_times), each = 12L)
  x <- rnorm(length(time))
  truth <- .5 * sin(2 * pi * seq_len(n_times) / n_times)
  d <- data.frame(time = time, x = x,
    y = .2 + (.5 + truth[time]) * x + rnorm(length(time)))
  reference <- .dense_tvc_rw2_reference(d$y, d$x, d$time, n_times)
  refined <- .dense_tvc_rw2_reference(d$y, d$x, d$time, n_times, n_grid = 1601L)
  expect_equal(reference$slope_mean, refined$slope_mean, tolerance = 1e-8)
  expect_equal(reference$slope_sd, refined$slope_sd, tolerance = 1e-8)
  expect_equal(reference$log_tau_mean, refined$log_tau_mean, tolerance = 1e-8)
  for (seed in seq_len(5L)) {
    fit <- tulpa(y ~ x, data = d, family = "gaussian", phi = 1,
      temporal = temporal_tvc("time", terms = ~ x - 1, structure = "rw2"),
      mode = "exact", control = list(n_iter = 2000L, warmup = 1000L,
                                     n_chains = 2L, seed = seed))
    total <- fit$draws[, "x"] + tvc(fit)$draws[, , 1L]
    by_chain <- split(seq_len(nrow(total)), fit$chain_id)
    mcse <- vapply(seq_len(n_times), function(j)
      posterior::mcse_mean(vapply(by_chain, function(ii) total[ii, j],
                                  numeric(length(by_chain[[1L]])))), numeric(1))
    expect_lt(max(abs(colMeans(total) - reference$slope_mean) / mcse), 5,
              label = paste("slope MCSE distance, seed", seed))
    log_tau <- fit$draws[, "log_tau_tvc[1]"]
    mcse_tau <- posterior::mcse_mean(vapply(by_chain, function(ii) log_tau[ii],
                                            numeric(length(by_chain[[1L]]))))
    expect_lt(abs(mean(log_tau) - reference$log_tau_mean) / mcse_tau, 5,
              label = paste("log-precision MCSE distance, seed", seed))
    expect_equal(n_divergent(fit), 0L, info = paste("seed", seed))
    dg <- diagnostics(fit)
    expect_lt(max(dg$rhat), 1.05, label = paste("maximum Rhat, seed", seed))
  }
})
