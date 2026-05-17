# Tests for sn_match / sn_cdf / sn_quantile.

test_that("sn_match reproduces forward moments", {
  # Forward: (xi, omega, alpha) -> (mu, sigma, gamma); then sn_match
  # should round-trip back to the same skew-normal parameters.
  for (alpha in c(-5, -1.5, -0.3, 0.3, 1.5, 5)) {
    xi    <- 0.4
    omega <- 1.3
    delta <- alpha / sqrt(1 + alpha^2)
    b     <- sqrt(2 / pi)
    mu    <- xi + omega * delta * b
    sigma <- omega * sqrt(1 - 2 * delta^2 / pi)
    gamma <- ((4 - pi) / 2) * (delta * b)^3 / (1 - 2 * delta^2 / pi)^(3 / 2)

    sn <- sn_match(mu, sigma, gamma)
    expect_equal(sn$xi,    xi,    tolerance = 1e-10)
    expect_equal(sn$omega, omega, tolerance = 1e-10)
    expect_equal(sn$alpha, alpha, tolerance = 1e-10)
  }
})

test_that("sn_match reduces to normal at gamma = 0", {
  sn <- sn_match(2, 0.5, 0)
  expect_equal(sn$xi, 2,    tolerance = 1e-12)
  expect_equal(sn$omega, 0.5, tolerance = 1e-12)
  expect_equal(sn$alpha, 0,  tolerance = 1e-12)
})

test_that("sn_match returns NULL past skew-normal ceiling", {
  expect_warning(out <- sn_match(0, 1, 0.998), "ceiling")
  expect_null(out)
})

test_that("sn_cdf is monotone increasing and bounded in [0, 1]", {
  sn <- sn_match(0, 1, 0.4)
  q  <- seq(-6, 6, length.out = 200)
  Fq <- sn_cdf(q, sn)
  expect_true(all(diff(Fq) >= -1e-12))
  expect_true(all(Fq >= -1e-12 & Fq <= 1 + 1e-12))
})

test_that("sn_cdf matches pnorm when alpha = 0", {
  sn <- list(xi = 0.5, omega = 2, alpha = 0)
  q  <- c(-3, -1, 0, 1, 3)
  expect_equal(sn_cdf(q, sn), pnorm(q, 0.5, 2), tolerance = 1e-9)
})

test_that("sn_quantile inverts sn_cdf", {
  sn <- sn_match(0.2, 0.8, 0.6)
  p  <- c(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
  q  <- sn_quantile(p, sn)
  expect_equal(sn_cdf(q, sn), p, tolerance = 1e-8)
})

test_that("sn_quantile handles boundary p", {
  sn <- sn_match(0, 1, 0.3)
  expect_identical(sn_quantile(0, sn), -Inf)
  expect_identical(sn_quantile(1, sn),  Inf)
  expect_true(is.nan(sn_quantile(-0.1, sn)))
  expect_true(is.nan(sn_quantile( 1.1, sn)))
})

test_that("sn_quantile works across negative-skew range", {
  sn <- sn_match(-0.3, 1.2, -0.5)
  p  <- c(0.025, 0.5, 0.975)
  q  <- sn_quantile(p, sn)
  expect_equal(sn_cdf(q, sn), p, tolerance = 1e-8)
})
