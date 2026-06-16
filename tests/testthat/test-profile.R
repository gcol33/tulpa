# Phase profiler for the sparse Laplace path: the per-phase accumulator
# (scatter / factorize / eta / ...), its cross-thread aggregation, and the
# tulpa_profile() reader. Only the SPARSE joint / single-block solvers carry
# TULPA_PROFILE_PHASE scopes, so every fit here forces a field above
# SPARSE_THRESHOLD (200) to exercise the instrumented path.

.profile_chain_adj <- function(n_s) {
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

# A joint binomial+gaussian ICAR fit over a field large enough to take the
# sparse Newton path. Returns the responses/prior/copy list for reuse.
.profile_joint_inputs <- function(N = 1500L, n_s = 250L, seed = 31) {
  set.seed(seed)
  spatial_idx <- sample.int(n_s, N, replace = TRUE)
  rw    <- cumsum(rnorm(n_s, 0, 0.6 / sqrt(n_s)))
  phi_s <- rw - mean(rw)
  x     <- rnorm(N)
  Xocc  <- cbind(1, x)
  eta_occ <- as.numeric(Xocc %*% c(-0.3, 0.5)) + phi_s[spatial_idx]
  occur   <- rbinom(N, 1L, plogis(eta_occ))
  is_pos  <- occur == 1L
  Xpos    <- Xocc[is_pos, , drop = FALSE]
  spi_pos <- spatial_idx[is_pos]
  eta_pos <- as.numeric(Xpos %*% c(0.2, -0.4)) + phi_s[spi_pos]
  y_pos   <- rnorm(sum(is_pos), eta_pos, 0.5)

  adj <- .profile_chain_adj(n_s)
  arm_occ <- list(
    y = as.numeric(occur), n_trials = rep(1L, N), X = Xocc,
    spatial_idx = spatial_idx, re_idx = rep(0, N), n_re_groups = 0L,
    sigma_re = 1.0, family = "binomial", phi = 1.0
  )
  arm_pos <- list(
    y = y_pos, n_trials = rep(1L, length(y_pos)), X = Xpos,
    spatial_idx = spi_pos, re_idx = rep(0, length(y_pos)), n_re_groups = 0L,
    sigma_re = 1.0, family = "gaussian", phi = 1.0
  )
  prior <- list(
    type = "icar", n_spatial_units = adj$n_spatial_units,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors, sigma_grid = c(0.5, 1.0)
  )
  list(
    responses = list(
      occ = arm_occ,
      pos = modifyList(arm_pos, list(
        field_coef = list(name = "alpha", grid = c(0.5, 1.0))))
    ),
    prior     = prior
  )
}

test_that("cpp_profile_reset zeroes every phase", {
  cpp_profile_reset()
  r <- cpp_profile_read()
  expect_identical(
    as.character(r$names),
    c("pattern_build", "prep", "eta", "scatter", "analyze", "factorize",
      "solve", "line_search", "log_det", "log_lik_prior")
  )
  expect_true(all(r$us == 0))
  expect_true(all(r$calls == 0L))
})

test_that("the sparse joint solver records scatter and factorize separately", {
  skip_on_cran()
  inp <- .profile_joint_inputs()

  cpp_profile_reset()
  fit <- tulpa_nested_laplace_joint(responses = inp$responses,
                                    prior = inp$prior, copy = inp$copy)
  r <- cpp_profile_read()

  expect_s3_class(fit, "tulpa_nested_laplace_joint")

  named <- function(nm) r$us[[which(r$names == nm)]]
  calls <- function(nm) r$calls[[which(r$names == nm)]]

  # The instrumented sparse path ran: both the assembly scatter and the
  # numeric factorize accumulated time and fired the same number of times
  # (one scatter + one factor per inner Newton iter, plus the final pass).
  expect_gt(named("scatter"), 0)
  expect_gt(named("factorize"), 0)
  expect_gt(calls("scatter"), 0L)
  expect_equal(calls("factorize"), calls("scatter"))
})

test_that("tulpa_profile returns the phase split and carries the fit", {
  skip_on_cran()
  inp <- .profile_joint_inputs()

  p <- tulpa_profile(
    tulpa_nested_laplace_joint(responses = inp$responses,
                               prior = inp$prior, copy = inp$copy)
  )

  expect_s3_class(p, "data.frame")
  expect_identical(names(p),
                   c("phase", "seconds", "calls", "ms_per_call", "share"))
  expect_true(all(c("scatter", "factorize") %in% p$phase))
  # Rows are ordered by descending time.
  expect_false(is.unsorted(rev(p$seconds)))
  # Shares of the timed phases sum to 1.
  expect_equal(sum(p$share[p$seconds > 0]), 1, tolerance = 1e-8)
  # ms_per_call is consistent with seconds / calls where calls > 0.
  pos <- p$calls > 0
  expect_equal(p$ms_per_call[pos], (p$seconds[pos] * 1e3) / p$calls[pos],
               tolerance = 1e-6)

  expect_s3_class(attr(p, "value"), "tulpa_nested_laplace_joint")
})

test_that("tulpa_profile resets between runs (no carry-over)", {
  skip_on_cran()
  inp <- .profile_joint_inputs()

  p1 <- tulpa_profile(
    tulpa_nested_laplace_joint(responses = inp$responses,
                               prior = inp$prior, copy = inp$copy)
  )
  # A second profiled run reports only its own work, not the sum of both.
  p2 <- tulpa_profile(
    tulpa_nested_laplace_joint(responses = inp$responses,
                               prior = inp$prior, copy = inp$copy)
  )
  s1 <- p1$calls[p1$phase == "scatter"]
  s2 <- p2$calls[p2$phase == "scatter"]
  expect_gt(s2, 0L)
  # Deterministic inputs -> same number of scatter calls, not double.
  expect_equal(s2, s1)
})
