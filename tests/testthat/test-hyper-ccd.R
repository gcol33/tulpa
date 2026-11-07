# Mode-centred outer CCD (tulpa_ccd_outer_grid) and tulpa_hyper_grid()'s
# design = "ccd" / extras channel.

.ccd_gauss <- function() {
  # Known outer posterior in u = (log tau, rho) ~ N(mu, S).
  mu <- c(log(5), 0.6)
  S  <- matrix(c(0.25, 0.05, 0.05, 0.02), 2, 2)
  Q  <- solve(S)
  list(mu = mu, S = S, logm = function(theta) {
    d <- c(log(theta[1]), theta[2]) - mu
    as.numeric(-0.5 * t(d) %*% Q %*% d)
  })
}
.ccd_axes <- list(
  list(name = "tau", log_scale = TRUE,  lower = 0.3, upper = 30,   start = 3),
  list(name = "rho", log_scale = FALSE, lower = 0.1, upper = 0.95, start = 0.5))

test_that("tulpa_ccd_outer_grid reproduces the Gaussian outer moments", {
  g <- .ccd_gauss()
  set.seed(1)
  cc <- tulpa_ccd_outer_grid(g$logm, .ccd_axes, k_samples = 200L)
  expect_true(cc$ok)
  expect_equal(unname(cc$u_hat), g$mu, tolerance = 1e-3)
  lm <- apply(cc$nodes, 1L, g$logm)
  w  <- cc$dnode * exp(lm - max(lm)); w <- w / sum(w)
  expect_equal(sum(w * log(cc$nodes[, 1])), g$mu[1], tolerance = 1e-2)
  expect_equal(sum(w * cc$nodes[, 2]),      g$mu[2], tolerance = 1e-2)
  # E[tau] = exp(mu1 + S11 / 2) for a log-normal margin.
  expect_equal(sum(w * cc$nodes[, 1]), exp(g$mu[1] + 0.5 * g$S[1, 1]), tolerance = 0.2)
  expect_true(is.finite(cc$pareto_k))
})

test_that("tulpa_ccd_outer_grid declines on a flat or single-axis problem", {
  flat <- tulpa_ccd_outer_grid(function(theta) 0, .ccd_axes, diagnose_k = FALSE)
  expect_false(flat$ok)
  expect_match(flat$reason, "curvature|box|mode")
  one <- tulpa_ccd_outer_grid(function(theta) -(log(theta[1]))^2, .ccd_axes[1],
                              diagnose_k = FALSE)
  expect_false(one$ok)
  expect_match(one$reason, "single axis")
})

test_that("tulpa_hyper_grid integrates on a CCD and pools extras", {
  g <- .ccd_gauss()
  inner <- function(h) {
    th <- c(h[["tau"]], h[["rho"]])
    list(log_marginal = g$logm(th), beta_mean = c(b = th[2]),
         beta_cov = matrix(0.01), extras = list(u = c(log(th[1]), th[2])))
  }
  specs <- list(
    hyper_axis_spec("tau", grid = exp(seq(log(0.3), log(30), length.out = 15)),
                    log_scale = TRUE, bounds = c(0, Inf)),
    hyper_axis_spec("rho", grid = seq(0.1, 0.95, length.out = 15)))
  gr <- tulpa_hyper_grid(specs, inner, n_draws = 0L)
  cc <- tulpa_hyper_grid(specs, inner, n_draws = 0L, design = "ccd",
                         ccd_start = list(tau = 3, rho = 0.5))
  expect_identical(gr$design, "grid")
  expect_identical(cc$design, "ccd")
  expect_null(cc$ccd_declined)
  expect_lt(cc$n_grid, gr$n_grid)
  # Both designs integrate the same posterior: E[log tau], E[rho].
  expect_equal(cc$extras_mean$u, g$mu, tolerance = 2e-2)
  expect_equal(gr$extras_mean$u, g$mu, tolerance = 2e-2)
  expect_equal(unname(cc$beta), g$mu[2], tolerance = 2e-2)
  # The pooled covariance is the law of total covariance over the mixture.
  m <- cc$mixture
  mbar <- sum(m$weights * m$modes[, 1])
  expect_equal(cc$beta_cov[1, 1],
               sum(m$weights * (0.01 + (m$modes[, 1] - mbar)^2)))
  expect_equal(sum(m$weights), 1)
  # A one-dimensional request declines to the grid and says why.
  one <- tulpa_hyper_grid(specs[1], function(h)
    list(log_marginal = -(log(h[["tau"]]) - 1)^2, beta_mean = 0, beta_cov = matrix(1)),
    n_draws = 0L, design = "ccd")
  expect_identical(one$design, "grid")
  expect_match(one$ccd_declined, "single axis")
})
