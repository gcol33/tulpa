# Outer-grid progress ETA + detached-console liveness (gcol33/tulpa#115).
#
# The C++ tulpa_progress::GridProgress reporter drives the nested-Laplace outer
# grid. Two regressions it fixes:
#   1. The ETA used to project the serial pilot's per-cell rate across the whole
#      grid. The pilot is the central, warm, serial cell; the parallel cells are
#      extreme-hyperparameter and run under bandwidth contention, so each costs
#      more -- the projection ran ~10x optimistic. The ETA now rests on the
#      realised per-cell wall time of completed POST-pilot cells, and the
#      pilot-only projection is flagged as a lower bound ("ETA >=") until a
#      parallel cell has finished.
#   2. The console line froze at the pilot ("1/N") in a detached / redirected
#      run: parallel ticks suppressed the console entirely (worker threads must
#      not touch the R print API), so only the serial pilot and the final
#      finish() line appeared. The master thread (thread 0 == the R main thread)
#      now emits from inside the parallel region too, so the line advances
#      cell by cell.
#
# Both are read off the captured console line, whose format is
#   [nested-laplace-joint] <done>/<total> cells (<pct>%) | elapsed .. | ETA <m><eta> | ..
# with <m> in {">=", "~", ""} (lower bound / point estimate / done).

.chain_adj_115 <- function(n_s) {
  nbr <- lapply(seq_len(n_s),
                function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  n_neighbors <- vapply(nbr, length, integer(1))
  list(adj_row_ptr = as.integer(c(0L, cumsum(n_neighbors))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L,
       n_neighbors = as.integer(n_neighbors),
       n_spatial_units = n_s)
}

# A joint occu+cover nested-Laplace fit with a deliberately wide outer grid
# (5 sigma x 5 alpha = 25 cells) so the parallel region completes several cells
# on the master thread. `progress.every = 1` + `progress.throttle = 0` emit on
# every completed cell (master only inside the parallel region) so the captured
# output carries the full done sequence.
.fit_joint_115 <- function(n_threads_outer, seed = 11L) {
  set.seed(seed)
  N <- 160L; n_s <- 16L
  adj <- .chain_adj_115(n_s)
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
                n_neighbors = adj$n_neighbors,
                sigma_grid = c(0.4, 0.7, 1.0, 1.5, 2.0))

  capture.output(
    fit <- tulpa_nested_laplace_joint(
      responses = list(
        occ = arm_occ,
        pos = modifyList(arm_pos, list(
          field_coef = list(name = "alpha",
                            grid = c(0.4, 0.7, 1.0, 1.3, 1.6))))
      ),
      prior = prior,
      control = list(force_sparse = TRUE,
                     n_threads_outer = n_threads_outer,
                     var_of_means_consistency = FALSE,
                     progress = TRUE, progress.every = 1L,
                     progress.throttle = 0)
    )
  )
}

# Pull the count progress lines and decode (done, total, eta-marker) per line.
.decode_progress <- function(out) {
  jl <- grep("^\\[nested-laplace-joint\\] [0-9]+/[0-9]+ ", out, value = TRUE)
  data.frame(
    done  = as.integer(sub("^\\[[^]]*\\] ([0-9]+)/.*",       "\\1", jl)),
    total = as.integer(sub("^\\[[^]]*\\] [0-9]+/([0-9]+).*", "\\1", jl)),
    mark  = ifelse(grepl("ETA >=", jl), ">=",
            ifelse(grepl("ETA ~",  jl), "~", "done")),
    stringsAsFactors = FALSE
  )
}

test_that("outer-grid ETA: pilot is a lower bound, later cells a point estimate", {
  skip_on_cran()
  pr <- .decode_progress(.fit_joint_115(n_threads_outer = 4L))
  expect_gt(nrow(pr), 0L)
  # The first completed cell (the cheap serial pilot) only bounds the ETA from
  # below: it is shown as "ETA >=".
  expect_true(any(pr$done == 1L & pr$mark == ">="))
  # Once a cell beyond the pilot has finished, the ETA is a realised-throughput
  # point estimate: "ETA ~".
  expect_true(any(pr$done > 1L & pr$mark == "~"))
  # The pilot lower bound is never dressed up as a point estimate.
  expect_false(any(pr$done == 1L & pr$mark == "~"))
})

test_that("outer-grid console advances through the parallel grid (not frozen)", {
  skip_on_cran()
  pr <- .decode_progress(.fit_joint_115(n_threads_outer = 4L))
  expect_gt(nrow(pr), 0L)
  total <- pr$total[1L]
  # A line whose done count is strictly between the pilot (1) and the final cell
  # proves the master thread emitted from inside the parallel region, rather
  # than the console freezing at "1/N" until finish() (gcol33/tulpa#115).
  expect_true(any(pr$done > 1L & pr$done < total))
})

test_that("serial outer-grid ETA still reduces to the plain extrapolation", {
  skip_on_cran()
  pr <- .decode_progress(.fit_joint_115(n_threads_outer = 1L))
  expect_gt(nrow(pr), 0L)
  # No outer threads -> no "| N threads" suffix, and the first emitted cell is
  # still a lower bound while later cells are point estimates.
  expect_true(any(pr$done == 1L & pr$mark == ">="))
  expect_true(any(pr$done > 1L & pr$mark == "~"))
})
