test_that("Gaussian mixture quantiles invert an independent CDF", {
  set.seed(341)
  n <- 301L
  mu <- cbind(rnorm(n), 5 * rnorm(n) + 2, c(rep(-3, 150), rep(4, 151)))
  variance <- cbind(exp(rnorm(n)), exp(rnorm(n) - 3), rep(.03, n))
  w <- exp(rnorm(n)); w <- w / sum(w)
  probs <- c(.001, .025, .1, .5, .9, .975, .999)
  got <- tulpa:::.nl_gauss_mixture_summary(mu, variance, w, probs)
  expected <- vapply(seq_len(ncol(mu)), function(j)
    vapply(probs, function(p) uniroot(function(x)
      sum(w * pnorm(x, mu[, j], sqrt(variance[, j]))) - p,
      c(min(mu[, j] - 12 * sqrt(variance[, j])),
        max(mu[, j] + 12 * sqrt(variance[, j]))), tol = 1e-12)$root, numeric(1)),
    numeric(length(probs)))
  expect_equal(got$quantiles, t(expected), tolerance = 1e-10)
})

test_that("identical Gaussian components retain analytic quantiles", {
  mu <- matrix(c(rep(.7, 100), rep(-2, 100)), 100, 2)
  variance <- matrix(c(rep(.3, 100), rep(4, 100)), 100, 2)
  probs <- c(.025, .5, .975)
  got <- tulpa:::.nl_gauss_mixture_summary(mu, variance, seq_len(100), probs)
  expected <- rbind(qnorm(probs, .7, sqrt(.3)), qnorm(probs, -2, 2))
  expect_equal(got$quantiles, expected, tolerance = 1e-13)
  missing <- tulpa:::.nl_gauss_mixture_summary(mu, variance, seq_len(100), NA_real_)
  expect_true(all(is.na(missing$quantiles)))
})

test_that("degenerate Gaussian components preserve quantiles of atoms", {
  got <- tulpa:::.nl_gauss_mixture_summary(matrix(c(-2, 3)), matrix(c(0, 0)), c(.2, .8))
  expect_equal(as.numeric(got$quantiles), c(-2, 3, 3), tolerance = 1e-12)
})

test_that("a small Newton step at an atom retains the CDF quantile bracket", {
  for (variance in c(0, 1e-40)) {
    got <- tulpa:::.nl_gauss_mixture_summary(matrix(c(-5, 0, 1)),
      matrix(rep(variance, 3)), c(1 / 8, 1 / 4, 5 / 8))
    expect_equal(as.numeric(got$quantiles), c(-5, 1, 1), tolerance = 1e-12)
  }
})

test_that("mixture quantiles preserve a change of measurement units", {
  mu <- c(-3, 0, 4)
  variance <- c(.2, 1.3, .4)
  weights <- c(.2, .5, .3)
  probs <- c(.025, .5, .975)
  expected <- vapply(probs, function(p) uniroot(function(x)
    sum(weights * pnorm(x, mu, sqrt(variance))) - p,
    c(-20, 20), tol = 1e-13)$root, numeric(1))
  for (scale in c(1e-20, 1, 1e20)) {
    got <- tulpa:::.nl_gauss_mixture_summary(matrix(scale * mu),
      matrix(scale^2 * variance), weights, probs)
    expect_equal(as.numeric(got$quantiles) / scale, expected, tolerance = 1e-12)
  }
})

test_that("Gaussian mixture moments retain small within-component variance", {
  mu <- matrix(rep(1e6, 128))
  variance <- matrix(rep(1e-8, 128))
  got <- tulpa:::.nl_gauss_mixture_summary(mu, variance, rep(1, 128))
  expect_equal(got$sd, 1e-4, tolerance = 1e-14)
  state <- list(weights = rep(1 / 128, 128), grid_modes = rep(list(1e6), 128),
    grid_hessians = rep(list(matrix(1e8)), 128))
  moments <- tulpa:::.nested_fixed_moments(state)
  expect_equal(moments$cov, matrix(1e-8), tolerance = 1e-14)
})

test_that("mixture moments retain covariance between nearby large locations", {
  mu <- rbind(c(1e6, -1e6), c(1e6 + 3e-4, -1e6 + 5e-4))
  covariance <- matrix(c(1e-8, 2e-9, 2e-9, 2e-8), 2)
  weights <- c(.25, .75)
  expected <- covariance + prod(weights) * tcrossprod(mu[2, ] - mu[1, ])
  state <- list(weights = weights, grid_modes = lapply(1:2, function(i) mu[i, ]),
    grid_hessians = rep(list(solve(covariance)), 2))
  moments <- tulpa:::.nested_fixed_moments(state)
  summary <- tulpa:::.nl_gauss_mixture_summary(mu,
    matrix(rep(diag(covariance), each = 2), 2), weights)
  expect_equal(moments$mean, mu[1, ] + .75 * (mu[2, ] - mu[1, ]), tolerance = 1e-12)
  expect_equal(moments$cov, expected, tolerance = 1e-14)
  expect_equal(summary$sd, sqrt(diag(expected)), tolerance = 1e-12)
})
