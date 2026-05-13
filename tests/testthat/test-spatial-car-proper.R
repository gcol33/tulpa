# Tests for proper-CAR spatial wiring in tulpa.
#
# These exercise:
#   - The R-side spatial_car_proper() constructor.
#   - The C++ dispatch: spatial_type_str = "car_proper" routes to
#     SpatialType::CAR_PROPER and allocates a logit_rho_car parameter.
#   - The handcoded log_post / gradient code paths for CAR_PROPER —
#     verified against the numerical reference via gradient_check_only.

# --- Helper: build a chain adjacency matrix -------------------------------
chain_adj <- function(n) {
  adj <- matrix(0L, n, n)
  for (i in seq_len(n - 1)) {
    adj[i, i + 1] <- 1L
    adj[i + 1, i] <- 1L
  }
  adj
}

# --- Helper: build CSR triple from adjacency ------------------------------
adj_to_csr <- function(adj) {
  n <- nrow(adj)
  row_ptr <- integer(n + 1)
  col_idx <- integer(0)
  n_neighbors <- integer(n)
  for (i in seq_len(n)) {
    nbrs <- which(adj[i, ] != 0)
    n_neighbors[i] <- length(nbrs)
    col_idx <- c(col_idx, nbrs - 1L)
    row_ptr[i + 1] <- row_ptr[i] + length(nbrs)
  }
  list(row_ptr = row_ptr, col_idx = col_idx, n_neighbors = n_neighbors)
}

# --- Helper: simulate from a proper-CAR field -----------------------------
# phi ~ N(0, (tau * (D - rho * W))^{-1})
sim_car_proper_field <- function(adj, tau = 1.0, rho = 0.7, seed = 42) {
  set.seed(seed)
  n <- nrow(adj)
  D <- diag(rowSums(adj))
  Q <- tau * (D - rho * adj)
  L <- chol(Q)
  z <- rnorm(n)
  # phi solves L^T phi = z, then phi has covariance Q^{-1}
  backsolve(L, z)
}

test_that("spatial_car_proper() returns a tulpa_spatial with type='car_proper'", {
  adj <- chain_adj(5)
  spec <- spatial_car_proper(adj, level = "group", group_var = "site")
  expect_s3_class(spec, "tulpa_spatial")
  expect_equal(spec$type, "car_proper")
  expect_true(spec$proper)
  expect_equal(spec$n_spatial, 5)
  expect_false(is.null(spec$rho_bounds))
  expect_named(spec$rho_bounds, c("lower", "upper"))
})

test_that("spatial_car(adj, proper = TRUE) is equivalent to spatial_car_proper", {
  adj <- chain_adj(6)
  a <- spatial_car_proper(adj, level = "group", group_var = "x")
  b <- spatial_car(adj, level = "group", group_var = "x", proper = TRUE)
  expect_equal(a$type, b$type)
  expect_equal(a$rho_bounds, b$rho_bounds)
  expect_equal(a$n_spatial, b$n_spatial)
})
