skip_on_cran()

test_that("full-latent and outer corrections agree with coupled joint NUTS", {
  skip_if_not_slow()
  block <- .subspace_exact_block()
  d <- .subspace_exact_data()
  C <- matrix(c(.4, .75, .75, 2.1), 2)
  expect_equal(as.matrix(block$Q(c(.5))),
    kronecker(diag(4), solve(C)) * exp(-1), tolerance = 1e-12, ignore_attr = TRUE)
  reference <- tulpa_tgmrf(d$y, d$n_trials, d$X, block, family = "poisson",
    mode = "nuts_joint", control = list(n_iter = 40000L, warmup = 5000L,
      target_accept = .99, max_depth = 10L, seed = 117620L))
  expect_equal(reference$n_divergent, 0L)
  truth <- .subspace_reference_statistics(reference)
  estimates <- array(NA_real_, c(5L, 11L, 8L))
  for (r in seq_len(8L)) {
    fit <- .subspace_exact_fit(block, "exact", 117620L + r,
      n_grid = 81L, n_iter = 4000L, warmup = 1000L, n_draws = 8000L)
    expect_true(fit$subspace_debias$outer$applied)
    expect_identical(fit$inference_tier, 2L)
    estimates[, , r] <- .subspace_fit_statistics(fit, fit$joint_latent_draws)
  }
  average <- apply(estimates, c(1, 2), mean)
  mcse <- apply(estimates, c(1, 2), sd) / sqrt(dim(estimates)[3])
  error <- abs(average - truth$value) / sqrt(mcse^2 + truth$mcse^2)
  expect_true(max(error) < 5, info = paste("maximum combined MCSE", max(error)))
  expect_lt(fit$subspace_debias$outer$pareto_k, fit$subspace_debias$outer$k_threshold)
  set.seed(117629)
  latent <- tulpa_posterior_draws(fit, n = 100L)
  set.seed(117629)
  eta <- posterior_linpred(fit, ndraws = 100L)
  expected <- latent[, 1:2, drop = FALSE] %*% t(d$X) + latent[, 3:10, drop = FALSE]
  expect_equal(eta, expected, tolerance = 1e-10, ignore_attr = TRUE)
  expect_identical(attr(latent, "theta"), fit$theta_grid[attr(latent, "cells"), , drop = FALSE])
})
