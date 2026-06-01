# Wall-clock fit timing on the nested-Laplace front-door fitters
# (gcol33/tulpa#48): the timer/formatter helpers, the `$timing` field on real
# fits, and the one-line print summary.

# --------------------------------------------------------------------------- #
# Helpers: timer / formatter / summary line                                    #
# --------------------------------------------------------------------------- #

test_that(".tulpa_timer accumulates phases under their labels", {
  tm <- .tulpa_timer()
  tm$mark("setup")
  tm$mark("grid")
  tm$mark("grid")          # second touch adds to the same bucket
  tm$mark("postproc")
  timing <- tm$timing()

  expect_true(is.numeric(timing))
  expect_identical(names(timing), c("total", "setup", "grid", "postproc"))
  expect_true(all(timing >= 0))
  # total is the independent start-to-now wall-clock; phases sum to ~total.
  expect_equal(sum(timing[c("setup", "grid", "postproc")]),
               unname(timing[["total"]]), tolerance = 1e-2)
})

test_that(".format_duration renders hour / minute / second bands", {
  expect_equal(.format_duration(5 * 3600 + 25 * 60), "5h 25m")
  expect_equal(.format_duration(2 * 3600 + 9 * 60), "2h 09m")     # zero-pad
  expect_equal(.format_duration(3 * 60 + 12), "3m 12s")
  expect_equal(.format_duration(4.2), "4.2s")
  expect_true(is.na(.format_duration(-1)))
  expect_true(is.na(.format_duration(c(1, 2))))
})

test_that(".timing_summary_line matches the issue's one-line format", {
  timing <- c(total = 5 * 3600 + 25 * 60, setup = 0, grid = 2 * 3600 + 9 * 60)
  expect_equal(.timing_summary_line(timing), "fit in 5h 25m (grid 2h 09m)")
  # No grid bucket -> total only, no parenthetical.
  expect_equal(.timing_summary_line(c(total = 42)), "fit in 42.0s")
  # Unusable input -> NULL so the caller can skip the line.
  expect_null(.timing_summary_line(NULL))
  expect_null(.timing_summary_line(c(setup = 10)))
})

# --------------------------------------------------------------------------- #
# Integration: real fits carry $timing                                         #
# --------------------------------------------------------------------------- #

.timing_chain_adj <- function(n_s) {
  nbr <- lapply(seq_len(n_s),
                function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  n_neighbors <- vapply(nbr, length, integer(1))
  list(
    adj_row_ptr     = as.integer(c(0L, cumsum(n_neighbors))),
    adj_col_idx     = as.integer(unlist(nbr)) - 1L,
    n_neighbors     = as.integer(n_neighbors),
    n_spatial_units = n_s
  )
}

test_that("tulpa_nested_laplace attaches a wall-clock timing breakdown", {
  set.seed(11)
  n_s <- 20L
  adj <- .timing_chain_adj(n_s)
  N <- 200L
  spatial_idx <- sample.int(n_s, N, replace = TRUE)
  X <- cbind(1, rnorm(N))
  eta <- as.numeric(X %*% c(-0.2, 0.4))
  y <- rbinom(N, 1L, plogis(eta))

  prior <- list(
    type = "icar", spatial_idx = spatial_idx,
    n_spatial_units = n_s,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    tau_grid = exp(seq(log(0.5), log(20), length.out = 5))
  )

  fit <- tulpa_nested_laplace(
    y = y, n_trials = rep(1L, N), X = X,
    prior = prior, family = "binomial"
  )

  expect_true(is.numeric(fit$timing))
  expect_true("total" %in% names(fit$timing))
  expect_true(all(c("setup", "grid", "postproc", "diagnostics") %in%
                    names(fit$timing)))
  expect_true(all(fit$timing >= 0))
  expect_gt(fit$timing[["total"]], 0)
  # Phases partition the total wall-clock to within a small slop.
  phase_sum <- sum(fit$timing[setdiff(names(fit$timing), "total")])
  expect_equal(phase_sum, unname(fit$timing[["total"]]), tolerance = 0.05)
})

test_that("print.tulpa_nested_laplace surfaces the one-line timing summary", {
  set.seed(12)
  n_s <- 16L
  adj <- .timing_chain_adj(n_s)
  N <- 150L
  spatial_idx <- sample.int(n_s, N, replace = TRUE)
  X <- cbind(1, rnorm(N))
  y <- rbinom(N, 1L, plogis(as.numeric(X %*% c(0.1, -0.3))))

  prior <- list(
    type = "icar", spatial_idx = spatial_idx,
    n_spatial_units = n_s,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    tau_grid = exp(seq(log(0.5), log(20), length.out = 4))
  )
  fit <- tulpa_nested_laplace(y = y, n_trials = rep(1L, N), X = X,
                              prior = prior, family = "binomial")

  out <- capture.output(print(fit))
  expect_true(any(grepl("nested-Laplace fit", out)))
  expect_true(any(grepl("fit in ", out)))
  expect_invisible(print(fit))
})

test_that("joint nested-Laplace fit carries the timing breakdown", {
  set.seed(13)
  n_s <- 24L
  adj <- .timing_chain_adj(n_s)
  N <- 250L
  spatial_idx <- sample.int(n_s, N, replace = TRUE)
  X <- cbind(1, rnorm(N))
  y <- rbinom(N, 1L, plogis(as.numeric(X %*% c(-0.1, 0.3))))

  arm <- list(
    y = as.numeric(y), n_trials = rep(1L, N), X = X,
    spatial_idx = spatial_idx, re_idx = rep(0, N), n_re_groups = 0L,
    sigma_re = 1.0, family = "binomial", phi = 1.0
  )
  prior <- list(
    type = "icar", n_spatial_units = adj$n_spatial_units,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors, sigma_grid = c(0.5, 1.0, 1.5)
  )

  fit <- tulpa_nested_laplace_joint(responses = list(occ = arm), prior = prior)

  expect_s3_class(fit, "tulpa_nested_laplace_joint")
  expect_true(is.numeric(fit$timing))
  expect_true(all(c("total", "setup", "grid") %in% names(fit$timing)))
  expect_gt(fit$timing[["total"]], 0)
  expect_true(all(fit$timing >= 0))

  out <- capture.output(print(fit))
  expect_true(any(grepl("joint nested-Laplace fit", out)))
  expect_true(any(grepl("fit in ", out)))
})
