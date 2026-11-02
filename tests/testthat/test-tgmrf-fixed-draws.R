# The theta-sampling tgmrf adapters (mode = "imh" / "vi" / "nuts") report the
# fixed effects: one conditional draw beta | theta per theta draw, so the beta
# columns are the mixture over the sampled theta. Checked against a dense
# tensor quadrature over theta of the same Laplace body.

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

test_that("tgmrf vi / imh report fixed-effect draws matching dense quadrature", {
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
  fi <- tulpa_tgmrf(y = y, n_trials = rep(1L, n), X = X, block = blk,
                    family = "poisson", mode = "imh", n_iter = 3000L,
                    warmup = 500L)
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
