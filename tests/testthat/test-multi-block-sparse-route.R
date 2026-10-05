# The single-arm multi-block driver behind tulpa_nested_laplace() solves past
# SPARSE_THRESHOLD latents, or with a block whose prior has only a sparse
# scatter, as a one-arm fit on the joint driver's sparse Newton
# (gcol33/tulpa#944). Below the threshold it keeps its dense inner solve. The
# two have to be one model: the dense path is checked against the joint door
# forced sparse on a small lattice, and the routed path against the joint door
# on a lattice past the threshold.

.sparse_route_lattice <- function(nr, nc) {
  adj_list <- lapply(grid_neighbours(nr, nc), sort)
  nn <- vapply(adj_list, length, integer(1))
  list(n_spatial_units = length(adj_list),
       adj_row_ptr = as.integer(c(0L, cumsum(nn))),
       adj_col_idx = as.integer(unlist(adj_list)) - 1L,
       n_neighbors = as.integer(nn))
}

.sparse_route_data <- function(nr, nc, reps = 2L, seed = 9L) {
  set.seed(seed)
  g <- .sparse_route_lattice(nr, nc)
  n_s <- g$n_spatial_units
  idx <- rep(seq_len(n_s), each = reps)
  n <- length(idx)
  x <- rnorm(n)
  X <- cbind(`(Intercept)` = 1, x = x)
  f <- as.numeric(scale(sin(seq_len(n_s) / 7))) * 0.5
  y <- rpois(n, exp(0.3 + 0.4 * x + f[idx]))
  list(graph = g, idx = idx, X = X, y = y, n = n)
}

.sparse_route_ctl <- list(axis_refine = "none", prune = FALSE,
                          diagnose_k = FALSE)

.sparse_route_declared <- function(type) {
  g <- expand.grid(s = c(0.3, 0.8), r = c(0.3, 0.7))
  switch(type,
    icar = list(tau_grid = c(0.8, 3, 10)),
    bym2 = list(sigma_grid = g$s, rho_grid = g$r),
    car_proper = list(tau_grid = 1 / g$s^2, rho_grid = g$r))
}

.sparse_route_fits <- function(d, type, joint_control) {
  blk <- c(list(type = type), d$graph, .sparse_route_declared(type))
  registry <- tulpa_nested_laplace(
    y = d$y, n_trials = rep(1L, d$n), X = d$X, family = "poisson",
    prior = list(c(blk, list(spatial_idx = d$idx))),
    control = .sparse_route_ctl)
  joint <- tulpa_nested_laplace_joint(
    responses = list(y = list(y = d$y, n_trials = rep(1L, d$n), X = d$X,
                              family = "poisson")),
    prior = list(c(blk, list(spatial_idx = list(d$idx)))),
    control = joint_control)
  list(registry = registry, joint = joint)
}

.expect_same_fit <- function(a, b, info) {
  expect_equal(unname(as.matrix(a$theta_grid)),
               unname(as.matrix(b$theta_grid)), info = info)
  expect_equal(a$log_marginal, b$log_marginal, tolerance = 1e-6, info = info)
  expect_equal(unname(coef(a)), unname(coef(b)), tolerance = 1e-6,
               info = info)
}


test_that("the dense multi-block solve equals the sparse joint Newton", {
  skip_on_cran()
  d <- .sparse_route_data(5L, 5L)
  for (type in c("icar", "bym2", "car_proper")) {
    fits <- .sparse_route_fits(
      d, type, c(.sparse_route_ctl, list(force_sparse = TRUE)))
    .expect_same_fit(fits$registry, fits$joint, info = type)
  }
})


test_that("a field past the sparse threshold fits through the joint Newton", {
  skip_on_cran()
  d <- .sparse_route_data(15L, 14L)
  for (type in c("icar", "bym2", "car_proper")) {
    fits <- .sparse_route_fits(d, type, .sparse_route_ctl)
    .expect_same_fit(fits$registry, fits$joint, info = type)
    reg <- fits$registry
    n_cells <- nrow(as.matrix(reg$theta_grid))
    expect_equal(dim(reg$fitted_eta), c(n_cells, d$n), info = type)
    expect_equal(dim(reg$fitted_eta_var), c(n_cells, d$n), info = type)
    expect_true(all(is.finite(reg$fitted_eta)), info = type)
    expect_true(all(reg$fitted_eta_var > 0), info = type)
  }
})
