# A projected areal field on the two multi-block drivers (gcol33/tulpa#940):
# an icar / bym2 / car_proper block carrying `projector` reaches arm k as
# A_k z, with z on the block's own prior. The joint driver reads it per arm,
# the single-arm one behind tulpa_nested_laplace() as one matrix.

.proj_lattice <- function(nr = 5L, nc = 5L) {
  adj_list <- lapply(grid_neighbours(nr, nc), sort)
  nn <- vapply(adj_list, length, integer(1))
  list(n_spatial_units = length(adj_list),
       adj_row_ptr = as.integer(c(0L, cumsum(nn))),
       adj_col_idx = as.integer(unlist(adj_list)) - 1L,
       n_neighbors = as.integer(nn))
}

.proj_incidence <- function(idx, n_s) {
  S <- matrix(0, length(idx), n_s)
  S[cbind(seq_along(idx), idx)] <- 1
  S
}

.proj_residual <- function(X) diag(nrow(X)) - X %*% solve(crossprod(X), t(X))

.proj_data <- function(n = 150L, seed = 5L, family = "poisson") {
  set.seed(seed)
  g <- .proj_lattice()
  n_s <- g$n_spatial_units
  idx <- sample.int(n_s, n, TRUE)
  x <- rnorm(n)
  X <- cbind(`(Intercept)` = 1, x = x)
  f <- as.numeric(scale(sin(seq_len(n_s) / 3))) * 0.6
  eta <- 0.4 + 0.5 * x + f[idx]
  y <- switch(family,
              poisson = rpois(n, exp(eta)),
              gaussian = rnorm(n, eta, 0.7))
  list(graph = g, idx = idx, X = X, y = y, n = n, n_s = n_s)
}

test_that("a projector equal to the unit incidence fits the gathered model", {
  skip_on_cran()
  d <- .proj_data()
  ctl <- list(axis_refine = "none", prune = FALSE, diagnose_k = FALSE)
  resp <- list(y = list(y = d$y, n_trials = rep(1L, d$n), X = d$X,
                        family = "poisson"))
  for (type in c("icar", "bym2")) {
    blk <- c(list(type = type), d$graph)
    gathered <- tulpa_nested_laplace_joint(
      responses = resp, prior = list(c(blk, list(spatial_idx = list(d$idx)))),
      control = ctl)
    projected <- tulpa_nested_laplace_joint(
      responses = resp,
      prior = list(c(blk, list(projector = list(
        .proj_incidence(d$idx, d$n_s))))),
      control = ctl)
    expect_equal(projected$log_marginal, gathered$log_marginal,
                 tolerance = 1e-6, info = type)
    expect_equal(coef(projected), coef(gathered), tolerance = 1e-6,
                 info = type)
    expect_equal(vcov(projected), vcov(gathered), tolerance = 1e-6,
                 info = type)
  }
})

test_that("a restricted field leaves the gaussian fixed effects at OLS", {
  skip_on_cran()
  # With A = P S and P X = 0, the field is orthogonal to the design, so the
  # fixed-effect mode at every cell is the least-squares fit whatever the
  # field does. A Matrix-sparse projector takes the same route as a dense one.
  d <- .proj_data(family = "gaussian")
  A <- .proj_residual(d$X) %*% .proj_incidence(d$idx, d$n_s)
  expect_lt(max(abs(rowSums(A))), 1e-10)
  resp <- list(y = list(y = d$y, n_trials = rep(1L, d$n), X = d$X,
                        family = "gaussian", phi = 0.49))
  blk <- c(list(type = "icar"), d$graph)
  ctl <- list(axis_refine = "none", diagnose_k = FALSE)
  fit <- tulpa_nested_laplace_joint(
    responses = resp, prior = list(c(blk, list(projector = list(A)))),
    control = ctl)
  ols <- stats::lm.fit(d$X, d$y)$coefficients
  expect_equal(unname(coef(fit)), unname(ols), tolerance = 1e-3)
  sparse_fit <- tulpa_nested_laplace_joint(
    responses = resp,
    prior = list(c(blk, list(projector = Matrix::Matrix(A, sparse = TRUE)))),
    control = ctl)
  expect_equal(sparse_fit$log_marginal, fit$log_marginal, tolerance = 1e-8)
})

test_that("a projector is refused where it would be dropped", {
  d <- .proj_data()
  resp <- list(y = list(y = d$y, n_trials = rep(1L, d$n), X = d$X,
                        family = "poisson"))
  S <- .proj_incidence(d$idx, d$n_s)
  blk <- c(list(type = "icar", projector = list(S)), d$graph)
  expect_error(
    tulpa_nested_laplace_joint(responses = resp, prior = blk),
    "does not read a block `projector`")
  expect_error(
    tulpa_nested_laplace_joint(
      responses = resp,
      prior = list(c(blk, list(spatial_idx = list(d$idx))))),
    "either `projector` or `spatial_idx`")
  short <- blk
  short$projector <- list(S[-1, ])
  expect_error(
    tulpa_nested_laplace_joint(responses = resp, prior = list(short)),
    "must be a 150 x 25 matrix")
})

test_that("the registry door reads a projector as the joint driver does", {
  skip_on_cran()
  d <- .proj_data()
  ctl <- list(axis_refine = "none", prune = FALSE, diagnose_k = FALSE)
  resp <- list(y = list(y = d$y, n_trials = rep(1L, d$n), X = d$X,
                        family = "poisson"))
  S <- .proj_incidence(d$idx, d$n_s)
  A <- .proj_residual(d$X) %*% S
  for (type in c("icar", "bym2", "car_proper")) {
    blk <- c(list(type = type), d$graph)
    gathered <- tulpa_nested_laplace(
      y = d$y, n_trials = rep(1L, d$n), X = d$X, family = "poisson",
      prior = list(c(blk, list(spatial_idx = d$idx))), control = ctl)
    via_incidence <- tulpa_nested_laplace(
      y = d$y, n_trials = rep(1L, d$n), X = d$X, family = "poisson",
      prior = c(blk, list(projector = S)), control = ctl)
    expect_equal(via_incidence$log_marginal, gathered$log_marginal,
                 tolerance = 1e-6, info = type)
    expect_equal(coef(via_incidence), coef(gathered), tolerance = 1e-6,
                 info = type)
    # The two doors lay their default grids differently, so the comparison
    # runs on one declared grid.
    g <- expand.grid(s = c(0.3, 0.7, 1.4), r = c(0.25, 0.6))
    declared <- switch(type,
      icar = list(tau_grid = c(0.5, 2, 8)),
      bym2 = list(sigma_grid = g$s, rho_grid = g$r),
      car_proper = list(tau_grid = 1 / g$s^2, rho_grid = g$r))
    restricted <- tulpa_nested_laplace(
      y = d$y, n_trials = rep(1L, d$n), X = d$X, family = "poisson",
      prior = c(blk, declared, list(projector = A)), control = ctl)
    joint <- tulpa_nested_laplace_joint(
      responses = resp,
      prior = list(c(blk, declared, list(projector = list(A)))),
      control = ctl)
    expect_equal(unname(as.matrix(restricted$theta_grid)),
                 unname(as.matrix(joint$theta_grid)), info = type)
    expect_equal(restricted$log_marginal, joint$log_marginal,
                 tolerance = 1e-6, info = type)
    expect_equal(unname(coef(restricted)), unname(coef(joint)),
                 tolerance = 1e-6, info = type)
  }
  expect_error(
    tulpa_nested_laplace(
      y = d$y, n_trials = rep(1L, d$n), X = d$X, family = "poisson",
      prior = c(list(type = "icar", projector = S, spatial_idx = d$idx),
                d$graph)),
    "either `projector` or `spatial_idx`")
  expect_error(
    tulpa_nested_laplace(
      y = d$y, n_trials = rep(1L, d$n), X = d$X, family = "poisson",
      prior = c(list(type = "icar", projector = S[-1, ]), d$graph)),
    "must be a 150 x 25 matrix")
})

test_that("a restricted single-arm gaussian fit leaves the fixed effects at OLS", {
  skip_on_cran()
  d <- .proj_data(family = "gaussian")
  A <- .proj_residual(d$X) %*% .proj_incidence(d$idx, d$n_s)
  fit <- tulpa_nested_laplace(
    y = d$y, n_trials = rep(1L, d$n), X = d$X, family = "gaussian",
    phi = 0.49, prior = c(list(type = "icar", projector = A), d$graph),
    control = list(axis_refine = "none", diagnose_k = FALSE))
  ols <- stats::lm.fit(d$X, d$y)$coefficients
  expect_equal(unname(coef(fit)), unname(ols), tolerance = 1e-3)
})
