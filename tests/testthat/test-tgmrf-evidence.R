# A tgmrf() block's theta axis is laid evenly in theta and carries the user's
# prior as a density on theta, so its cell measure is its width in theta and
# the evidence of a fit carrying one is a number. Every NA evidence carries a
# reason. `n_grid` sets the block's node count.

test_that("tgmrf n_grid sets the per-axis node count and is validated", {
  Qf <- function(theta) Matrix::Diagonal(4, exp(theta[1]))
  lp <- function(theta) sum(stats::dnorm(theta, 0, 1, log = TRUE))
  blk <- tgmrf(Q = Qf, prior = lp, init = c(a = 0, b = 0),
               bounds = list(lower = c(-1, -2), upper = c(1, 2)),
               n_grid = c(3, 7))
  g <- tulpa:::.nl_block_axis_grid(blk)$grid
  expect_identical(length(unique(g[, "a"])), 3L)
  expect_identical(length(unique(g[, "b"])), 7L)
  expect_equal(sort(unique(g[, "b"])), seq(-2, 2, length.out = 7))
  expect_identical(tgmrf(Q = Qf, prior = lp, init = c(a = 0, b = 0),
                         n_grid = 4)$n_grid, c(4L, 4L))
  expect_identical(length(unique(tulpa:::.nl_block_axis_grid(
    tgmrf(Q = Qf, prior = lp, init = c(a = 0)))$grid[, "a"])),
    tulpa:::.nl_grid_par("tgmrf_axis", "n"))
  expect_error(tgmrf(Q = Qf, prior = lp, init = c(a = 0), n_grid = 1),
               "whole numbers >= 2")
  expect_error(tgmrf(Q = Qf, prior = lp, init = c(a = 0, b = 0),
                     n_grid = c(3, 4, 5)), "length 1 or")
})

test_that("an NA evidence names why", {
  tg <- matrix(c(1, 2, 3), ncol = 1L, dimnames = list(NULL, "b1.foo"))
  spec <- hyper_axis_spec(name = "b1.foo", grid = c(1, 2, 3), log_scale = FALSE)
  spec$unweighted <- TRUE
  res <- list(log_marginal = c(-1, -2, -3), log_hyperprior_axes = "b1.foo")
  out <- tulpa:::.nl_attach_evidence(res, tg, list(spec))
  expect_true(is.na(out$log_evidence))
  expect_identical(out$log_evidence_declined, "undeclared_axis_coordinate")
  expect_identical(names(out$log_evidence_declined_axes), "b1.foo")

  spec$unweighted <- NULL
  out2 <- tulpa:::.nl_attach_evidence(
    utils::modifyList(res, list(log_marginal = rep(NaN, 3))), tg, list(spec))
  expect_identical(out2$log_evidence_declined, "no_finite_cell")
})

test_that("a tulpa() fit with a latent(tgmrf()) block reports its evidence", {
  skip_if_fast()
  set.seed(3)
  K <- 12L
  W <- matrix(0, K, K)
  for (i in seq_len(K)) { j <- if (i < K) i + 1L else 1L; W[i, j] <- W[j, i] <- 1 }
  d <- data.frame(region = rep(seq_len(K), each = 10L))
  d$x <- stats::rnorm(nrow(d))
  d$y <- stats::rpois(nrow(d), exp(0.5 + 0.4 * d$x + sin(2 * pi * d$region / K)))
  d$cell <- seq_len(nrow(d))
  mk <- function(n_grid) tgmrf(
    Q = function(theta) Matrix::Diagonal(nrow(d), exp(theta[1])),
    prior = function(theta) stats::dnorm(theta[1], 2, 1.5, log = TRUE),
    init = c(log_tau = 2), bounds = list(lower = -1, upper = 7),
    obs_idx = d$cell, n_grid = n_grid)
  ev <- vapply(c(5L, 17L), function(ng) {
    f <- suppressWarnings(tulpa(
      y ~ x + spatial(region) + latent(mk(ng)), data = d, family = "poisson",
      spatial = list(type = "icar", adjacency = W), mode = "auto",
      control = list(progress = FALSE)))
    expect_null(f$log_evidence_declined)
    expect_true(is.finite(as.numeric(logLik(f))))
    f$log_evidence
  }, numeric(1))
  expect_true(all(is.finite(ev)))
  # The same prior integrated on a coarser and a finer theta axis: the
  # evidence is a property of the model, not of the node count.
  expect_lt(abs(diff(ev)), 1)
})
