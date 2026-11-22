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

test_that("an AR(1) rho heaviest on its lowest default level is followed below it", {
  skip_on_cran()
  # The default rho levels run 0, 0.4, 0.7, 0.9, 0.97. Here the lowest holds
  # 0.45 of the posterior at an ESS of 2.85: its inner marginal is too flat to
  # rail and its spacing resolves it, so nothing places it, and a collapsed
  # axis is only followed out once a mode is known -- unless its density
  # peaks on an end level, which is this axis (gcol33/tulpa#1285). A dense
  # reference (rho on tanh(seq(-3.5, 3.5, length.out = 57))) reads
  # (-0.665, 0.154, 0.876); the declared levels alone read (-0.186, 0.268,
  # 0.903).
  set.seed(1)
  n_s <- 16L; n_t <- 8L; N <- 400L
  s <- sample(n_s, N, TRUE); tt <- sample(n_t, N, TRUE)
  us <- as.numeric(scale(cumsum(rnorm(n_s))))
  vt <- as.numeric(scale(cumsum(rnorm(n_t))))
  x <- rnorm(N)
  y <- rbinom(N, 1, plogis(0.2 + 0.5 * x + 0.7 * us[s] + 0.6 * vt[tt]))
  fit <- suppressWarnings(fit_st_nested(y, cbind(1, x), s, .chain_adj_st(n_s),
                                        tt, n_t, family = "binomial",
                                        control = list(progress = FALSE)))
  expect_lt(min(fit$theta_grid[, "b2.rho"]), -0.5)
  expect_equal(c(fit$theta_ci_lo[["b2.rho"]], fit$theta_median[["b2.rho"]],
                 fit$theta_ci_hi[["b2.rho"]]),
               c(-0.665, 0.154, 0.876), tolerance = 0.03)
  expect_cells_solved(fit)
})
