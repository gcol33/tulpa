# A fit_st_nested() fit is placed by the joint door's placement pass: a
# precision whose posterior sits outside its family's default axis is
# re-laid around the outer mode, and the reported interval covers the
# simulated value.

.chain_adj_st <- function(n_s) {
  adj <- matrix(0, n_s, n_s)
  for (i in seq_len(n_s - 1L)) adj[i, i + 1L] <- adj[i + 1L, i] <- 1
  adj
}

# One simulated RW1 path with known precision `tau`, centred to match the
# field's sum-to-zero constraint.
.rw1_path_st <- function(n, tau) {
  v <- cumsum(c(0, rnorm(n - 1L, 0, 1 / sqrt(tau))))
  v - mean(v)
}

.st_place_fixture <- function(seed, n_s, n_t, n_per, tau_s, tau_t, sx, b1) {
  set.seed(seed)
  adj <- .chain_adj_st(n_s)
  s <- rep(seq_len(n_s), each = n_per)
  tt <- sample(n_t, length(s), TRUE)
  us <- .rw1_path_st(n_s, tau_s)
  vt <- .rw1_path_st(n_t, tau_t)
  x <- rnorm(length(s), 0, sx)
  eta <- 0.1 + b1 * x + us[s] + vt[tt]
  fit_st_nested(rbinom(length(eta), 1, plogis(eta)), cbind(1, x), s, adj, tt,
                n_t, family = "binomial", temporal_type = "rw1")
}

.covers <- function(fit, axis, truth) {
  hp <- summary(fit)$hyperparameters
  r <- hp[hp$parameter == axis, ]
  expect_lt(r$ci_lo, truth)
  expect_gt(r$ci_hi, truth)
  expect_lt(min(fit$theta_grid[, axis]), truth)
  expect_gt(max(fit$theta_grid[, axis]), truth)
}

skip_on_cran()

test_that("a spatial precision far above its default axis is placed and covered", {
  skip_on_cran()
  fit <- .st_place_fixture(7, 14L, 8L, 60L, tau_s = 60, tau_t = 2,
                           sx = 0.3, b1 = 0.3)
  expect_identical(fit$outer_grid_placement, "auto_recentered")
  .covers(fit, "b1.tau", 60)
  .covers(fit, "b2.tau", 2)
  expect_cells_solved(fit)
})

test_that("a temporal precision far below its default axis is placed and covered", {
  skip_on_cran()
  fit <- .st_place_fixture(101, 12L, 16L, 150L, tau_s = 4, tau_t = 0.03,
                           sx = 0.2, b1 = 0.2)
  expect_identical(fit$outer_grid_placement, "auto_recentered")
  .covers(fit, "b2.tau", 0.03)
  .covers(fit, "b1.tau", 4)
  expect_cells_solved(fit)
})
