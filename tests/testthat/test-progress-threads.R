# Progress-line outer-thread count (gcol33/tulpa#88).
#
# The shared fit progress reporter appends "| N threads" to its console line
# when the outer grid runs N cells at once, so "ran on N cores" is a property
# of the log itself. Two reporters mirror the same wire format:
#   * the R-side `.tulpa_iter_progress()` (counted R loops, e.g. EM-Laplace),
#     which takes the count via its `threads` argument;
#   * the C++ `tulpa_progress::GridProgress` (the outer nested-Laplace grid and
#     the parallel NUTS sampler), which derives it from the realised outer
#     width stamped by `set_width()` inside `run_nested_laplace_grid`.
# A serial loop leaves the count at 1 and the field is omitted -- non-threaded
# callers are byte-for-byte unchanged.

# --------------------------------------------------------------------------- #
# 1. R-side reporter: optional threads field                                  #
# --------------------------------------------------------------------------- #

test_that(".tulpa_iter_progress appends the threads field when threads > 1", {
  old <- options(tulpa.nl_progress = list(progress = TRUE,
                                          progress_every = 1L,
                                          progress_throttle = 0))
  on.exit(options(old), add = TRUE)
  out <- capture.output({
    prg <- tulpa:::.tulpa_iter_progress("test", total = 3L,
                                        unit = "cells", threads = 8L)
    prg$tick(); prg$tick(); prg$tick(); prg$finish()
  })
  expect_true(any(grepl("\\| 8 threads", out, fixed = FALSE)))
  # The rest of the line is the unchanged format.
  expect_true(any(grepl("^\\[test\\] 3/3 cells \\(100%\\) .* \\| 8 threads$",
                        out)))
})

test_that(".tulpa_iter_progress omits the threads field when serial", {
  old <- options(tulpa.nl_progress = list(progress = TRUE,
                                          progress_every = 1L,
                                          progress_throttle = 0))
  on.exit(options(old), add = TRUE)
  out <- capture.output({
    prg <- tulpa:::.tulpa_iter_progress("test", total = 3L, unit = "cells")
    prg$tick(); prg$tick(); prg$tick(); prg$finish()
  })
  expect_false(any(grepl("threads", out)))
  expect_true(any(grepl("^\\[test\\] 3/3 cells \\(100%\\)", out)))
})

# --------------------------------------------------------------------------- #
# 2. C++ outer grid: nested-Laplace-joint console line                        #
# --------------------------------------------------------------------------- #

.chain_adj_88 <- function(n_s) {
  nbr <- lapply(seq_len(n_s),
                function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  n_neighbors <- vapply(nbr, length, integer(1))
  list(adj_row_ptr = as.integer(c(0L, cumsum(n_neighbors))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L,
       n_neighbors = as.integer(n_neighbors),
       n_spatial_units = n_s)
}

.fit_joint_88 <- function(n_threads_outer, seed = 11L) {
  set.seed(seed)
  N <- 160L; n_s <- 16L
  adj <- .chain_adj_88(n_s)
  spatial_idx <- sample.int(n_s, N, replace = TRUE)
  rw  <- cumsum(rnorm(n_s, 0, 1 / sqrt(n_s)))
  phi <- rw - mean(rw)
  x   <- rnorm(N)
  Xocc <- cbind(1, x)
  eta_occ <- as.numeric(Xocc %*% c(-0.2, 0.4)) + phi[spatial_idx]
  occur   <- rbinom(N, 1, plogis(eta_occ))
  is_pos  <- occur == 1L
  Xpos    <- Xocc[is_pos, , drop = FALSE]
  spi_pos <- spatial_idx[is_pos]
  eta_pos <- as.numeric(Xpos %*% c(0.1, -0.3)) + 1.5 * phi[spi_pos]
  y_pos   <- rnorm(sum(is_pos), eta_pos, 0.3)

  arm_occ <- list(y = as.numeric(occur), n_trials = rep(1L, N),
                  X = Xocc, spatial_idx = spatial_idx,
                  re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1.0,
                  family = "binomial", phi = 1.0)
  arm_pos <- list(y = y_pos, n_trials = rep(1L, length(y_pos)),
                  X = Xpos, spatial_idx = spi_pos,
                  re_idx = rep(0, length(y_pos)), n_re_groups = 0L,
                  sigma_re = 1.0, family = "gaussian", phi = 0.3)
  prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                n_neighbors = adj$n_neighbors, sigma_grid = c(0.6, 1.0, 1.5))

  capture.output(
    fit <- tulpa_nested_laplace_joint(
      responses = list(
        occ = arm_occ,
        pos = modifyList(arm_pos, list(
          field_coef = list(name = "alpha", grid = c(0.5, 1.0, 1.5))))
      ),
      prior = prior,
      control = list(force_sparse = TRUE,
                     n_threads_outer = n_threads_outer,
                     var_of_means_consistency = FALSE,
                     progress = TRUE, progress.throttle = 0)
    )
  )
}

test_that("nested-laplace-joint progress line shows the outer-thread count", {
  skip_on_cran()
  n_out <- 4L
  out <- .fit_joint_88(n_threads_outer = n_out)
  joint_lines <- grep("^\\[nested-laplace-joint\\]", out, value = TRUE)
  expect_gt(length(joint_lines), 0L)
  expect_true(any(grepl(sprintf("\\| %d threads$", n_out), joint_lines)))
})

test_that("serial nested-laplace-joint fit omits the threads field", {
  skip_on_cran()
  out <- .fit_joint_88(n_threads_outer = 1L)
  joint_lines <- grep("^\\[nested-laplace-joint\\]", out, value = TRUE)
  expect_gt(length(joint_lines), 0L)
  expect_false(any(grepl("threads", joint_lines)))
})
