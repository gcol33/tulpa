# The cheap-pass screen of tulpa_re_cov_nested()'s tensor (gcol33/tulpa#1192):
# cells scored below prune_tol of the grid's mass by a short inner Newton are
# left out without a full solve, the mass they may carry is bounded by the
# worst screening error measured on a solved cell, and past
# .NL_SCREEN$re_cov_gate_mass the cells carrying it are solved.

skip_on_cran()

screen_fixture <- function(seed = 201L) {
  set.seed(seed)
  G <- 20L; per <- 10L; n <- G * per
  grp <- rep(seq_len(G), each = per); x <- rnorm(n)
  eta <- 0.2 + 0.5 * x + rnorm(G, 0, 0.8)[grp] + rnorm(G, 0, 0.5)[grp] * x
  list(y = eta + rnorm(n, 0, 0.5), n = n, X = cbind(1, x),
       rt = list(idx = grp, n_groups = G, n_coefs = 2L, Z = cbind(1, x),
                 correlated = TRUE))
}

screen_fit <- function(d, ...) {
  tulpa_re_cov_nested(d$y, rep(1L, d$n), d$X, d$rt, family = "gaussian",
                      phi = 0.25, hyperprior = "proper",
                      control = list(diagnose_k = FALSE, ...))
}

test_that("prune = FALSE solves every cell and reports no screen", {
  d <- screen_fixture()
  off <- screen_fit(d, prune = FALSE)
  expect_false(off$screen$applied)
  expect_identical(off$screen$dropped, 0L)
  expect_identical(off$screen$bound, 0)
  expect_identical(off$n_grid, nrow(off$theta_grid))
  expect_gte(off$n_grid, 9L^3)
})

test_that("the screen leaves cells out within its gate and keeps the summaries", {
  d <- screen_fixture()
  off <- screen_fit(d, prune = FALSE)
  on  <- screen_fit(d)
  s <- on$screen
  expect_true(s$applied)
  expect_gt(s$dropped, 0L)
  expect_lt(on$n_grid, off$n_grid)
  expect_lte(s$bound, tulpa:::.nl_screen("re_cov_gate_mass") * (1 + 1e-8))
  expect_gte(s$max_error, 0)
  expect_equal(sum(on$weights), 1)

  # Every reported moment and median of the covariance stays within a few
  # hundredths of its own posterior SD of the fit that solves every cell.
  rows <- intersect(on$posterior$parameter, off$posterior$parameter)
  a <- off$posterior[match(rows, off$posterior$parameter), ]
  b <- on$posterior[match(rows, on$posterior$parameter), ]
  for (cn in c("mean", "sd", "median")) {
    expect_lt(max(abs(b[[cn]] - a[[cn]]) / a$sd), 0.05, label = cn)
  }
  expect_equal(unname(on$means), unname(off$means),
               tolerance = 0.01 * min(sqrt(diag(as.matrix(off$vcov %||% diag(2))))))
})

test_that("a screened cell is one the full grid gives almost no weight", {
  d <- screen_fixture()
  off <- screen_fit(d, prune = FALSE)
  on  <- screen_fit(d)
  key <- function(m) apply(signif(m, 10), 1L, paste, collapse = ",")
  kept <- key(off$theta_grid) %in% key(on$theta_grid)
  expect_identical(sum(kept), on$n_grid)
  expect_lte(sum(off$weights[!kept]), 10 * tulpa:::.nl_screen("re_cov_gate_mass"))
})

test_that("screening knobs are validated where they are set", {
  d <- screen_fixture()
  expect_error(screen_fit(d, prune_tol = 1.5), "prune_tol")
  expect_error(screen_fit(d, screen_iters = 0), "screen_iters")
})
