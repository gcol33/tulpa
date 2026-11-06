test_that("an incomplete precision check retains an infinite upper bound", {
  u <- matrix(seq(-3, 3, length.out = 128L), ncol = 1L)
  group <- rep(1:8, each = 16L)
  out <- list(grid_modes = lapply(u[, 1L], function(x) x),
    grid_hessians = rep(list(matrix(1, 1L)), nrow(u)))
  local_mocked_bindings(.nl_gauss_mixture_summary = function(...)
    stop("fixed-effect quantiles evaluated"), .package = "tulpa")
  partial <- tulpa:::.nl_qmc_error(u, rep(0, nrow(u)), group, out,
    rep(16L, 8L), tol = .001)
  expect_false(partial$complete)
  expect_identical(partial$max, Inf)
  expect_true(all(is.infinite(partial$fixed_quantiles)))
  expect_true(all(is.finite(partial$hyper_quantiles)))
  expect_error(tulpa:::.nl_qmc_error(u, rep(0, nrow(u)), group, out,
    rep(16L, 8L)), "fixed-effect quantiles evaluated")
})

test_that("a stopping candidate receives the complete fixed-effect bound", {
  u <- matrix(rep(qnorm((seq_len(64L) - .5) / 64L), 8L), ncol = 1L)
  group <- rep(1:8, each = 64L)
  out <- list(grid_modes = lapply(seq_len(nrow(u)), function(i) group[i] / 4),
    grid_hessians = rep(list(matrix(1, 1L)), nrow(u)))
  full <- tulpa:::.nl_qmc_error(u, rep(0, nrow(u)), group, out, rep(64L, 8L))
  candidate <- tulpa:::.nl_qmc_error(u, rep(0, nrow(u)), group, out,
    rep(64L, 8L), tol = .05)
  expect_identical(candidate, full)
  expect_true(candidate$complete)
  expect_gt(max(candidate$fixed_quantiles), .05)
  expect_equal(candidate$max, max(c(candidate$log_evidence,
    candidate$hyper_quantiles, candidate$fixed_quantiles)))
})

test_that("the point ceiling receives the complete precision check", {
  skip_on_cran()
  U <- matrix(seq(-4, 4, length.out = 33L), ncol = 1L,
    dimnames = list(NULL, "a"))
  density <- function(u) dnorm(u[, 1L], log = TRUE)
  res <- list(theta_grid = U, log_marginal = density(U),
    weights = tulpa:::.nl_normalise_weights_safe(density(U)))
  solve <- function(cells, retain) list(log_marginal = density(cells),
    log_hyperprior_axes = "a", grid_modes = as.list(cells[, 1L]),
    grid_hessians = rep(list(matrix(1, 1L)), nrow(cells)))
  fit <- tulpa:::.nl_qmc_integrate(res, solve, "identity", n = 8192L, tol = 1e-8)
  expect_equal(fit$outer_qmc$n, 8192L)
  expect_false(fit$outer_qmc$converged)
  expect_true(fit$outer_qmc$error$complete)
  expect_true(all(is.finite(fit$outer_qmc$error$fixed_quantiles)))
  history <- fit$outer_qmc$history
  expect_identical(as.logical(history[, "complete"]), c(FALSE, TRUE))
  expect_identical(unname(history[1L, "error"]), Inf)
  expect_true(is.finite(history[1L, "hyper_error"]))
  expect_equal(unname(history[2L, "error"]), fit$outer_qmc$error$max)
})
