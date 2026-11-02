# The fractional-nu single-point fit and the nested grid cell read the gaussian
# residual variance phi through one kernel convention (the SD), so at one
# (range, sigma) they compute the same det-lemma marginal.

skip_on_cran()

.frac_fixture <- function(family = "poisson") {
  set.seed(4)
  n <- 200
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  sp <- spatial_spde(~ lon + lat, data = d, max_edge = c(0.1, 0.3), nu = 0.5)
  eta <- 1 + 0.5 * d$x + sin(3 * d$lon)
  d$y <- if (family == "poisson") rpois(n, exp(eta)) else eta + rnorm(n, 0, 0.5)
  list(d = d, sp = sp, X = model.matrix(~ x, d), n = n)
}

test_that("fractional single-point and nested marginals agree for gaussian", {
  skip_on_cran()
  # Both read the residual variance phi through the kernel's SD convention.
  fx <- .frac_fixture("gaussian")
  phi <- 0.25
  single <- tulpa:::.spde_laplace_fractional_at(
    y = fx$d$y, n_trials = rep(1L, fx$n), X = fx$X, spatial = fx$sp,
    family = "gaussian", phi = phi, range = 0.3, sigma = 0.5,
    re_idx = NULL, n_re_groups = 0L, sigma_re = 1, max_iter = 100L,
    tol = 1e-6, n_threads = 1L, offset = NULL)
  nested <- tulpa:::.spde_nested_logmarginal_at(
    spatial = fx$sp, range = 0.3, sigma = 0.5, y = fx$d$y, X = fx$X,
    family = "gaussian", phi = phi, n_trials = rep(1L, fx$n), order = 2L,
    max_iter = 100L, tol = 1e-6, n_threads = 1L, offset = NULL)
  expect_equal(single$log_marginal, nested$log_marginal, tolerance = 1e-6)
})
