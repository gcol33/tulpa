# Boundary validation of the areal (ICAR / BYM2 / proper-CAR) entry points
# (gcol33/tulpa#441).
#
# adj_col_idx indexes a WRITE into the dense Hessian and spatial_idx indexes the
# latent field, both through raw pointer arithmetic. An out-of-range value from
# a caller that builds its own CSR is heap corruption rather than an R-level
# error, so these entries have to reject it before the scatter runs.

.chain_adj <- function(n) {
  # 0-based CSR of a path graph on n nodes.
  row_ptr <- 0L
  col_idx <- integer(0)
  nnb <- integer(n)
  for (s in seq_len(n)) {
    nb <- c(if (s > 1) s - 1L, if (s < n) s + 1L)
    col_idx <- c(col_idx, as.integer(nb) - 1L)
    nnb[s] <- length(nb)
    row_ptr <- c(row_ptr, length(col_idx))
  }
  list(row_ptr = as.integer(row_ptr), col_idx = as.integer(col_idx),
       n_neighbors = nnb)
}

.fit_spatial <- function(adj, spatial_idx, n_units, N = 12L) {
  set.seed(4L)
  y <- rbinom(N, 1L, 0.4)
  cpp_laplace_fit_spatial(
    y = as.numeric(y), n = rep(1L, N),
    X = cbind(rep(1, N)), re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1,
    spatial_idx = as.integer(spatial_idx), n_spatial_units = as.integer(n_units),
    adj_row_ptr = adj$row_ptr, adj_col_idx = adj$col_idx,
    n_neighbors = adj$n_neighbors, tau_spatial = 1,
    family = "binomial", phi = 1, max_iter = 20L, tol = 1e-6, n_threads = 1L
  )
}

test_that("a valid areal call still fits", {
  n_units <- 6L
  adj <- .chain_adj(n_units)
  fit <- .fit_spatial(adj, rep_len(seq_len(n_units), 12L), n_units)
  expect_true(is.finite(fit$log_marginal))
})

test_that("an out-of-range neighbour column is an R error, not a write", {
  n_units <- 6L
  adj <- .chain_adj(n_units)
  bad <- adj
  bad$col_idx[1] <- as.integer(n_units)      # one past the last site
  expect_error(.fit_spatial(bad, rep_len(seq_len(n_units), 12L), n_units),
               "adj_col_idx")
  bad2 <- adj
  bad2$col_idx[2] <- -1L
  expect_error(.fit_spatial(bad2, rep_len(seq_len(n_units), 12L), n_units),
               "adj_col_idx")
})

test_that("a mis-sized CSR is rejected", {
  n_units <- 6L
  adj <- .chain_adj(n_units)
  short <- adj
  short$row_ptr <- head(adj$row_ptr, -1L)
  expect_error(.fit_spatial(short, rep_len(seq_len(n_units), 12L), n_units),
               "adj_row_ptr")

  nn <- adj
  nn$n_neighbors <- head(adj$n_neighbors, -1L)
  expect_error(.fit_spatial(nn, rep_len(seq_len(n_units), 12L), n_units),
               "n_neighbors")

  tail_mismatch <- adj
  tail_mismatch$col_idx <- head(adj$col_idx, -1L)
  expect_error(.fit_spatial(tail_mismatch, rep_len(seq_len(n_units), 12L), n_units),
               "adj_col_idx")
})

test_that("spatial_idx is checked against n_spatial_units and against n_obs", {
  n_units <- 6L
  adj <- .chain_adj(n_units)
  idx <- rep_len(seq_len(n_units), 12L)

  bad <- idx; bad[3] <- n_units + 1L
  expect_error(.fit_spatial(adj, bad, n_units), "spatial_idx")

  zero <- idx; zero[1] <- 0L           # 1-based, so 0 is out of range
  expect_error(.fit_spatial(adj, zero, n_units), "spatial_idx")

  expect_error(.fit_spatial(adj, idx[1:6], n_units), "spatial_idx")
})

test_that("the nested ICAR entry carries the same check", {
  n_units <- 6L
  adj <- .chain_adj(n_units)
  bad <- adj
  bad$col_idx[1] <- as.integer(n_units)
  set.seed(5L)
  N <- 12L
  y <- rbinom(N, 1L, 0.4)
  expect_error(
    cpp_nested_laplace_icar(
      y = as.numeric(y), n = rep(1L, N), X = cbind(rep(1, N)),
      re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1,
      spatial_idx = rep_len(seq_len(n_units), N),
      n_spatial_units = n_units,
      adj_row_ptr = bad$row_ptr, adj_col_idx = bad$col_idx,
      n_neighbors = bad$n_neighbors, tau_grid = c(0.5, 1, 2),
      family = "binomial", phi = 1, max_iter = 20L, tol = 1e-6, n_threads = 1L
    ),
    "adj_col_idx")
})
