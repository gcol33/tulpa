# The Gaussian beta mixture over the Laplace theta target, compared with
# dense quadrature. The generic IMH sampler receives that target explicitly.
.tgfd_block <- function(n) {
  tgmrf(
    Q = function(theta) {
      sigma <- exp(theta[1]); rho <- tanh(theta[2])
      d <- rep((1 + rho^2) / sigma^2, n); o <- rep(-rho / sigma^2, n)
      M <- Matrix::bandSparse(n, k = c(-1, 0, 1), diagonals = list(o, d, o))
      M[1, n] <- M[n, 1] <- -rho / sigma^2
      methods::as(methods::as(M, "generalMatrix"), "CsparseMatrix")
    },
    prior = function(theta) stats::dnorm(theta[1], 0, 1, log = TRUE) +
      stats::dnorm(theta[2], 0, 1, log = TRUE),
    init   = c(log_sigma = 0, atanh_rho = atanh(0.3)),
    bounds = list(lower = c(log(0.3), 0), upper = c(log(3), atanh(0.9))),
    name   = "periodic_ar1")
}

.tgfd_data <- function(blk, n, seed) {
  set.seed(seed)
  L <- Matrix::Cholesky(blk$Q(c(log(0.8), atanh(0.6))))
  z <- as.numeric(Matrix::solve(L, rnorm(n), system = "Lt"))
  rpois(n, exp(0.3 + z))
}

.tgfd_marginal_imh <- function(y, X, blk, n_iter, warmup) {
  obs <- tulpa:::.tgmrf_obs_idx(blk, length(y))
  lm <- tulpa:::.tgmrf_make_log_marginal(y, rep(1L, length(y)), X, blk, obs,
    NULL, 0L, 1, "poisson", 1, 50L, 1e-7, 1L)
  pilot <- tulpa:::.tgmrf_pilot(y, rep(1L, length(y)), X, blk, obs,
                                 family = "poisson")
  om <- tulpa:::.tgmrf_outer_mode(pilot, lm, blk)
  fit <- imh_laplace(lm$eval, om$mode, tulpa:::.spd_inverse(om$cov),
                     n_iter = n_iter, warmup = warmup)
  th <- fit$draws
  colnames(th) <- blk$theta_names
  fit$mode_theta <- om$mode
  fit$mode_status <- om$status
  fit$inference_mode <- "structured"
  fit$inference_tier <- 2L
  tulpa:::.tgmrf_with_fixed(fit, th, lm, X)
}

test_that("VI and explicit Laplace-target IMH match the Gaussian beta mixture", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  n <- 40L; blk <- .tgfd_block(n); y <- .tgfd_data(blk, n, 20260518L)
  X <- matrix(1, n, 1L)

  b2 <- blk; b2$obs_idx <- seq_len(n)
  b2$theta_grid_built <- as.matrix(expand.grid(
    log_sigma = seq(log(0.3), log(3), length.out = 41L),
    atanh_rho = seq(0, atanh(0.9), length.out = 41L)))
  dense <- tulpa_nested_laplace(
    y = y, n_trials = rep(1L, n), X = X, prior = b2, family = "poisson",
    control = list(progress = FALSE, prune = FALSE, axis_refine = "none"))
  ref <- .nested_fixed_moments(dense)
  ref_sd <- sqrt(ref$cov[1, 1])

  set.seed(1)
  fv <- tulpa_tgmrf(y = y, n_trials = rep(1L, n), X = X, block = blk,
                    family = "poisson", mode = "vi", n_draws = 800L)
  set.seed(2)
  fi <- .tgfd_marginal_imh(y, X, blk, n_iter = 3000L, warmup = 500L)
  # The IMH proposal is centred on the outer mode, not on a pilot node: it
  # sits within one spacing of the dense grid's best cell on either axis.
  expect_identical(fi$mode_status, "ok")
  k_dense <- which.max(dense$log_marginal)
  spacing <- c(log(3) - log(0.3), atanh(0.9)) / 40
  expect_lt(max(abs(fi$mode_theta - dense$theta_grid[k_dense, ]) / spacing),
            1)
  for (f in list(fv, fi)) {
    expect_identical(colnames(f$draws), c("beta_1", "log_sigma", "atanh_rho"))
    expect_identical(f$fixed_names, "beta_1")
    expect_identical(names(coef(f)), "beta_1")
    expect_equal(f$draws[, "beta_1"], f$draws_beta[, 1])
    expect_equal(unname(f$draws_theta), unname(f$draws[, 2:3]))
  }
  # VI's theta is a Gaussian fit, not the posterior, so its beta mixture is
  # held loosely. IMH samples theta without the bounds wall the quadrature is
  # laid inside; restricted to the same box, its beta draws are the bounded
  # mixture, held to Monte Carlo error. (Outside the box, rho -> 1 leaves the
  # level weakly identified and the beta tail heavy, so the unrestricted SD
  # is no stable target.)
  expect_lt(abs(mean(fv$draws_beta) - ref$mean) / ref_sd, 0.25)
  th <- fi$draws_theta
  inb <- th[, 1] >= log(0.3) & th[, 1] <= log(3) &
    th[, 2] >= 0 & th[, 2] <= atanh(0.9)
  expect_gt(mean(inb), 0.5)
  b_in <- fi$draws_beta[inb, 1]
  expect_lt(abs(mean(b_in) - ref$mean) / ref_sd, 0.15)
  expect_lt(abs(stats::sd(b_in) / ref_sd - 1), 0.15)
})


test_that("Laplace-target IMH reaches the rho -> 1 tail of its theta target", {
  # Outside the block's box the posterior keeps 6% of its mass toward
  # rho -> 1, where the level is weakly identified and the fixed effect's
  # conditional spread grows; the unbounded fixed-effect SD is 0.548 by dense
  # quadrature against 0.430 inside the box. A Gaussian proposal at the mode's
  # curvature is lighter-tailed than this posterior and read 0.452-0.457
  # (-17%) at 10000 iterations; the t proposal reads within 0.04.
  skip_if_not_slow()
  skip_if_not_installed("Matrix")
  n <- 40L; blk <- .tgfd_block(n); y <- .tgfd_data(blk, n, 20260518L)
  X <- matrix(1, n, 1L)

  b2 <- blk; b2$obs_idx <- seq_len(n)
  b2$theta_grid_built <- as.matrix(expand.grid(
    log_sigma = seq(-2.5, 2, length.out = 46L),
    atanh_rho = seq(-1.5, 4.5, length.out = 121L)))
  wide <- suppressWarnings(tulpa_nested_laplace(
    y = y, n_trials = rep(1L, n), X = X, prior = b2, family = "poisson",
    control = list(progress = FALSE, prune = FALSE, axis_refine = "none")))
  # The region holds the posterior: no mass on its rho -> 1 edge.
  expect_lt(sum(wide$weights[wide$theta_grid[, 2] > 3]), 1e-3)
  ref_sd <- sqrt(.nested_fixed_moments(wide)$cov[1, 1])

  set.seed(1)
  fi <- .tgfd_marginal_imh(y, X, blk, n_iter = 10000L, warmup = 500L)
  expect_lt(abs(stats::sd(fi$draws_beta[, 1]) / ref_sd - 1), 0.06)
})
