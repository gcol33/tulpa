# gcol33/tulpa#651 -- the outer-grid width a fit RAN at, against the one it
# asked for.
#
# `n_threads_outer` is clamped by the grid driver to the team the OpenMP
# environment hands out, and the clamp is correct: the per-cell block cache
# sizes its slot array from `omp_get_max_threads()`, so an unclamped
# `num_threads(n)` clause would map excess workers onto slot 0. What was missing
# is any trace of it. The progress reporter prints its thread suffix only above
# one thread, so a ten-thread request clamped to one prints exactly what a
# serial run prints, and a fit launched under `OMP_NUM_THREADS=1` to pin the
# INNER threads runs the outer grid on one core for hours with nothing on the
# fit or in the log to say so.
#
# What has to hold:
#   1. the resolver reports the same cap the driver clamps against;
#   2. a clamped request says so ONCE, naming the binding cap and how to lift
#      it -- not once per grid cell, and not at all when nothing was clamped;
#   3. both numbers reach the fit, because neither alone shows a clamp;
#   4. the driver stamps the width it actually opened the region at, which is
#      the environment cap clamped further by the cell count.

test_that("the realised outer width is the request capped by the environment", {
  cap <- tulpa:::cpp_get_max_threads()
  expect_length(cap, 1L)
  expect_true(is.numeric(cap) && is.finite(cap) && cap >= 1L)

  tw <- suppressMessages(tulpa:::.nl_outer_threads(1L))
  expect_identical(tw$requested, 1L)
  expect_identical(tw$realised, 1L)

  big <- as.integer(cap) + 10L
  tw_big <- suppressMessages(tulpa:::.nl_outer_threads(big))
  expect_identical(tw_big$requested, big)
  expect_identical(tw_big$realised, as.integer(cap))
  expect_lt(tw_big$realised, tw_big$requested)

  # A non-positive request defers to the environment, which is what the driver's
  # own resolver does with one.
  expect_identical(suppressMessages(tulpa:::.nl_outer_threads(0L))$realised,
                   as.integer(cap))
})

test_that("a clamped request is reported once and names the binding cap", {
  cap <- as.integer(tulpa:::cpp_get_max_threads())
  big <- cap + 10L

  msgs <- testthat::capture_messages(tulpa:::.nl_outer_threads(big))
  # Once per resolution, not once per grid cell.
  expect_length(msgs, 1L)
  # The request, the width it will actually run at, the cap that bound it, and
  # the lever that lifts it -- a reader who has only this line has to be able to
  # act on it.
  expect_match(msgs[[1L]], "n_threads_outer", fixed = TRUE)
  expect_match(msgs[[1L]], as.character(big), fixed = TRUE)
  expect_match(msgs[[1L]], "omp_get_max_threads", fixed = TRUE)
  expect_match(msgs[[1L]], "OMP_NUM_THREADS", fixed = TRUE)
  # The front door the reader set `control` on.
  expect_match(msgs[[1L]], "tulpa_nested_laplace_joint()", fixed = TRUE)

  msgs2 <- testthat::capture_messages(
    tulpa:::.nl_outer_threads(big, fn = "my_fitter()"))
  expect_length(msgs2, 1L)
  expect_match(msgs2[[1L]], "my_fitter()", fixed = TRUE)
})

test_that("an unclamped request is silent", {
  cap <- as.integer(tulpa:::cpp_get_max_threads())
  # A serial fit says nothing, and neither does one asking for exactly what the
  # environment will give it: the message exists to report a REDUCTION.
  expect_length(testthat::capture_messages(tulpa:::.nl_outer_threads(1L)), 0L)
  expect_length(testthat::capture_messages(tulpa:::.nl_outer_threads(cap)), 0L)
})

test_that("both numbers reach the fit, and the driver's own width wins", {
  tw <- list(requested = 10L, realised = 1L)

  # Nothing from the driver: the environment-capped request stands in.
  out <- tulpa:::.nl_attach_outer_threads(list(log_marginal = 1), tw)
  expect_identical(out$n_threads_outer_requested, 10L)
  expect_identical(out$n_threads_outer_realised, 1L)

  # The driver reports the width it opened the region at, which is the cap
  # clamped further by the cell count -- a two-cell grid runs two wide however
  # many threads the environment would allow. That number is the one kept.
  out2 <- tulpa:::.nl_attach_outer_threads(
    list(n_threads_outer_realised = 2L), list(requested = 10L, realised = 8L))
  expect_identical(out2$n_threads_outer_realised, 2L)
  expect_identical(out2$n_threads_outer_requested, 10L)

  # The request alone cannot show a clamp and the realised width alone cannot
  # either; a fit carries the pair.
  expect_true(all(c("n_threads_outer_requested", "n_threads_outer_realised")
                  %in% names(out)))
})

test_that("the grid driver stamps the width it ran at", {
  src <- test_path("..", "..", "src", "nested_laplace_grid.h")
  skip_if_not(file.exists(src), "package sources not available")
  txt <- readLines(src, warn = FALSE)

  # The clamp stays -- the cache's slot count depends on it.
  expect_true(any(grepl("tulpa_omp_team_size_req(n_threads_outer, n_grid)",
                        txt, fixed = TRUE)))
  # And the clamped value leaves the driver on the result, not only on the
  # progress reporter, whose thread suffix is printed above one thread only.
  expect_true(any(grepl("out[\"n_threads_outer_realised\"]", txt,
                        fixed = TRUE)))
})

test_that("a joint fit records the outer width it ran at", {
  skip_on_cran()
  set.seed(651L)
  N <- 200L
  n_s <- 30L
  nbr <- lapply(seq_len(n_s),
                function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  n_nb <- vapply(nbr, length, integer(1))
  spatial_idx <- sample.int(n_s, N, replace = TRUE)
  w_s <- 0.6 * rnorm(n_s)
  x <- rnorm(N)
  X <- cbind(1, x)
  y <- rbinom(N, 1, plogis(as.numeric(X %*% c(-0.3, 0.5)) + w_s[spatial_idx]))

  arm <- list(y = as.numeric(y), n_trials = rep(1L, N), X = X,
              spatial_idx = as.integer(spatial_idx),
              re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1.0,
              family = "binomial", phi = 1.0)
  prior <- list(type = "bym2", n_spatial_units = n_s,
                adj_row_ptr = as.integer(c(0L, cumsum(n_nb))),
                adj_col_idx = as.integer(unlist(nbr)) - 1L,
                n_neighbors = as.integer(n_nb), scale_factor = 1.0,
                sigma_grid = c(0.3, 0.6, 1.0), rho_grid = c(0.3, 0.7))

  fit <- suppressMessages(tulpa_nested_laplace_joint(
    responses = list(a = arm), prior = prior,
    control = list(n_threads = 1L, n_threads_outer = 1L,
                   auto_recenter = FALSE, diagnose_k = FALSE,
                   diagnose_skew = FALSE,
                   adaptive_grid = FALSE,
                   var_of_means_consistency = FALSE)))

  expect_identical(fit$n_threads_outer_requested, 1L)
  expect_true(is.numeric(fit$n_threads_outer_realised))
  # A serial request is never clamped, so the two agree.
  expect_identical(as.integer(fit$n_threads_outer_realised), 1L)
})
