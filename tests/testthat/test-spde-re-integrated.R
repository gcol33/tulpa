# An SPDE field beside a random intercept: the RE SD is INTEGRATED, not pinned.
#
# fit_spde()'s outer grid is (range, sigma) and its `sigma_re` is a scalar it
# conditions on, so `tulpa(y ~ x + (1 | g), spatial = spatial_spde(...))` under
# auto / nested_laplace landed on a fit whose RE was the one variance component
# it never estimated -- held at the default of 1, a number the data never
# produced, on the path whose whole point is integrating the hyperparameters
# (gcol33/tulpa#817). The generic nested driver already carries an `spde` block
# type, so the field now goes there as a block beside the RE's own `iid` block
# and both SDs sit on one outer grid.
#
# Integer nu only: fractional nu is the operator-based rational construction
# fit_spde() owns, and it refuses an RE term regardless.

make_spde_re_data <- function(seed, sd_re = 0.8, n = 150L, n_groups = 8L) {
  set.seed(seed)
  L <- cbind(lon = stats::runif(n, 0, 10), lat = stats::runif(n, 0, 10))
  S <- 0.64 * exp(-as.matrix(stats::dist(L)) / 3) + diag(1e-8, n)
  w <- as.numeric(t(chol(S)) %*% stats::rnorm(n))
  d <- data.frame(L, x = stats::rnorm(n),
                  g = rep(seq_len(n_groups), length.out = n))
  b <- stats::rnorm(n_groups, 0, sd_re)
  d$y <- stats::rpois(n, exp(0.3 + 0.5 * d$x + w + b[d$g]))
  list(d = d, b = b, sd_re = sd_re)
}

fit_spde_re <- function(d, ...) {
  suppressWarnings(tulpa(y ~ x + (1 | g), data = d, family = "poisson",
                         spatial = spatial_spde(~ lon + lat, data = d),
                         control = list(verbose = FALSE), ...))
}

test_that("an SPDE field with an RE term carries the RE SD as an outer axis", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  D <- make_spde_re_data(2)
  fit <- fit_spde_re(D$d)

  expect_identical(fit$backend, "nested_laplace")
  # Three axes: the field's (range, sigma) and the RE block's own sigma. Two
  # would mean the RE was conditioned again.
  expect_equal(ncol(fit$theta_grid), 3L)
  expect_true("b2.sigma" %in% colnames(fit$theta_grid))
  # The axis has to be a GRID, not a point -- a one-point axis is conditioning
  # wearing an axis's name.
  expect_gt(length(unique(fit$theta_grid[, "b2.sigma"])), 1L)
  expect_true("b2.sigma" %in% colnames(tulpa_hyper_draws(fit)))
})

test_that("no mode conditions the RE SD on the SPDE path any more", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  D <- make_spde_re_data(2)
  for (m in c("auto", "nested_laplace", "structured")) {
    warns <- character(0)
    fit <- withCallingHandlers(
      tulpa(y ~ x + (1 | g), data = D$d, family = "poisson", mode = m,
            spatial = spatial_spde(~ lon + lat, data = D$d),
            control = list(verbose = FALSE)),
      warning = function(w) {
        warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning")
      })
    expect_identical(fit$backend, "nested_laplace", info = m)
    expect_equal(ncol(fit$theta_grid), 3L, info = m)
    expect_false(any(grepl("sigma_re", warns)), info = m)
  }
})

test_that("a supplied sigma_re still conditions, as the degenerate axis", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  D <- make_spde_re_data(2)
  fit <- fit_spde_re(D$d, sigma_re = 0.5)
  # Same path, same axis, one node on it -- conditioning is the degenerate case
  # of integrating, not a second route.
  expect_true("b2.sigma" %in% colnames(fit$theta_grid))
  expect_equal(unique(fit$theta_grid[, "b2.sigma"]), 0.5)
})

test_that("an SPDE field with no RE term still reaches fit_spde()", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  # The redirect is narrowed, not removed: without an RE term there is nothing
  # to integrate beside the field and the dedicated integrator still owns it.
  D <- make_spde_re_data(2)
  fit <- suppressWarnings(tulpa(y ~ x, data = D$d, family = "poisson",
                                spatial = spatial_spde(~ lon + lat, data = D$d),
                                control = list(verbose = FALSE)))
  expect_identical(fit$backend, "spde")
})

test_that("fractional nu with an RE term still refuses", {
  skip_if_not_installed("fmesher")
  skip_on_cran()
  D <- make_spde_re_data(2)
  expect_error(
    tulpa(y ~ x + (1 | g), data = D$d, family = "poisson",
          mode = "nested_laplace",
          spatial = spatial_spde(~ lon + lat, data = D$d, nu = 1.5)),
    "fractional-nu SPDE")
})

test_that("the integrated RE SD tracks the simulated one", {
  skip_if_not_installed("fmesher")
  skip_if_not_slow()
  # Measured over sd_re in {0.2, 0.8, 2.0} x 8 seeds: the posterior mean of the
  # RE SD came out 0.268 / 1.175 / 2.727 -- monotone in the truth and the right
  # order of magnitude, against the flat 1 the conditioned path used whatever
  # the data said. (It sits high: 8 groups is a thin variance component and the
  # PC prior shrinks it upward, which is why this asserts ordering and a broad
  # band rather than a point.) The fixed-effect slope error barely moves --
  # 0.063 vs 0.076 at sd_re = 0.2 and 0.072 vs 0.092 at 2.0, inside the
  # seed-to-seed spread of ~0.07 -- so this is about the variance component
  # being estimated at all, not about the slope.
  est <- vapply(c(0.2, 0.8, 2.0), function(sd_re) {
    mean(vapply(1:4, function(s) {
      D <- make_spde_re_data(s, sd_re = sd_re)
      mean(tulpa_hyper_draws(fit_spde_re(D$d))[, "b2.sigma"])
    }, numeric(1)))
  }, numeric(1))

  expect_true(all(diff(est) > 0))
  expect_gt(est[1], 0.05); expect_lt(est[1], 0.7)
  expect_gt(est[3], 1.2)
})
