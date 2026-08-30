# Refinement changes the quadrature error, not the measure (gcol33/tulpa#620).
#
# The outer nodes are a quadrature rule for a prior declared before the fit, so
# a grid that gains nodes after seeing the data integrates the same measure it
# started with -- only more accurately. These tests run the whole driver on a
# fixture whose fixed-measure answer is available to any accuracy: a Gaussian
# sample with known mean, whose residual-scale marginal likelihood is exact.

# Exact log marginal of y ~ N(0, sigma^2) at one sigma, up to a constant that
# cancels in the weights.
gaussian_sigma_inner <- function(y) {
  n <- length(y); s2 <- sum(y^2)
  function(hypers) {
    sigma <- as.numeric(hypers[["sigma"]])
    if (!is.finite(sigma) || sigma <= 0) return(list(log_marginal = -Inf))
    list(log_marginal = -n * log(sigma) - 0.5 * s2 / sigma^2,
         beta_mean = c(mu = 0), beta_cov = matrix(sigma^2 / n, 1, 1))
  }
}

# The declared prior, and the support it is normalised over. Both fixed before
# the fit, so they are the same measure whatever nodes end up integrating it.
LOG_PRIOR <- function(s) stats::dexp(s, 1, log = TRUE)
SLAB <- c(0.2, 6)

# Moments of the posterior, integrated densely over the declared support.
#
# The integration coordinate of a `log_scale` axis is `log sigma`, and a spec's
# `log_prior` enters the target there: the engine weights
# `exp(log_marginal + log_prior)` by cell widths measured in `log sigma` and
# applies no volume element, the same convention the outer Pareto-k target
# carries. So the reference is a dense rule on `log sigma`, which is what the
# nodes are a quadrature rule FOR.
reference_moments <- function(y, slab = SLAB, n = 200001L) {
  u  <- seq(log(slab[1L]), log(slab[2L]), length.out = n)
  s  <- exp(u)
  inner <- gaussian_sigma_inner(y)
  lm <- vapply(s, function(v) inner(c(sigma = v))$log_marginal, numeric(1)) +
        LOG_PRIOR(s)
  w <- exp(lm - max(lm)); w <- w / sum(w)
  c(mean = sum(w * s), sd = sqrt(sum(w * s^2) - sum(w * s)^2))
}

slab_grid <- function(m, slab = SLAB) {
  exp(seq(log(slab[1L]), log(slab[2L]), length.out = m))
}

fit_sigma <- function(y, grid, control, slab = SLAB) {
  specs <- list(hyper_axis_spec("sigma", grid = grid, log_scale = TRUE,
                                bounds = c(0, Inf), refinable = TRUE,
                                log_prior = LOG_PRIOR, slab_bounds = slab))
  tulpa_hyper_grid(specs, gaussian_sigma_inner(y), combine = "none",
                   n_draws = 0L, control = control)
}

REFINE <- list(adaptive_grid = TRUE, adaptive_grid_edge_thresh = 1e-6,
               adaptive_grid_max_passes = 3L,
               var_of_means_consistency = TRUE)
PINNED <- list(adaptive_grid = FALSE, var_of_means_consistency = FALSE)

test_that("refinement moves the answer towards the measure, not away from it", {
  set.seed(620)
  y <- stats::rnorm(60, 0, 1.3)
  ref <- reference_moments(y)

  for (m in c(7L, 13L)) {
    pinned  <- fit_sigma(y, slab_grid(m), PINNED)
    refined <- fit_sigma(y, slab_grid(m), REFINE)
    # The arms differ in their node sets, which is the premise of the test.
    expect_gt(nrow(refined$theta_grid), nrow(pinned$theta_grid))
    err <- function(f) abs(f$theta_mean[["sigma"]] - ref[["mean"]])
    expect_lt(err(refined), err(pinned), label = sprintf("m = %d", m))
    expect_lt(err(refined) / ref[["mean"]], 0.03)
  }
})

test_that("two grids that resolve the measure agree on the posterior", {
  # The invariance the issue asks for: a coarse grid refined after seeing the
  # data and a fine grid pinned before it integrate ONE measure, so they report
  # one posterior. Before the nodes carried integration weights, the added
  # nodes reweighted the prior instead and the reported SD of a refinable axis
  # moved by more than 100 %.
  set.seed(620)
  y <- stats::rnorm(60, 0, 1.3)
  refined <- fit_sigma(y, slab_grid(7L), REFINE)
  fine    <- fit_sigma(y, slab_grid(21L), PINNED)
  expect_equal(unname(refined$theta_mean[["sigma"]]),
               unname(fine$theta_mean[["sigma"]]), tolerance = 0.02)
  expect_equal(unname(refined$theta_sd[["sigma"]]),
               unname(fine$theta_sd[["sigma"]]), tolerance = 0.15)
})

test_that("a finer grid integrates the declared measure more accurately", {
  set.seed(6201)
  y <- stats::rnorm(40, 0, 0.8)
  ref <- reference_moments(y)
  err <- function(m) abs(fit_sigma(y, slab_grid(m), PINNED)$theta_mean[["sigma"]] -
                         ref[["mean"]])
  e <- vapply(c(5L, 13L, 41L), err, numeric(1))
  expect_lt(e[3L], e[2L])
  expect_lt(e[2L], e[1L])
  expect_lt(e[3L] / ref[["mean"]], 0.01)
})

test_that("the nodes are a quadrature rule for the declared prior", {
  # A flat inner fit, so the weights ARE the measure: refining then has one
  # visible consequence, the same declared prior integrated more accurately.
  flat <- function(hypers) list(log_marginal = 0)
  u <- seq(log(SLAB[1L]), log(SLAB[2L]), length.out = 400001L)
  w <- exp(LOG_PRIOR(exp(u))); w <- w / sum(w)
  ref_mean <- sum(w * exp(u))

  got <- vapply(c(5L, 11L, 41L), function(m) {
    sp <- list(hyper_axis_spec("sigma", grid = slab_grid(m), log_scale = TRUE,
                               bounds = c(0, Inf), log_prior = LOG_PRIOR,
                               slab_bounds = SLAB))
    tulpa_hyper_grid(sp, flat, combine = "none",
                     n_draws = 0L)$theta_mean[["sigma"]]
  }, numeric(1))

  err <- abs(got - ref_mean)
  expect_lt(err[3L], err[2L])
  expect_lt(err[2L], err[1L])
  expect_lt(err[3L] / ref_mean, 0.002)
})

test_that("nodes outside the declared support are not proposed", {
  # The support is a prior choice, so refinement may only reduce the quadrature
  # error inside it. An axis whose posterior presses on the ceiling reports the
  # boundary mass rather than moving the support outward.
  set.seed(6202)
  y <- stats::rnorm(50, 0, 3.5)
  slab <- c(0.3, 2)
  fit <- fit_sigma(y, slab_grid(5L, slab), REFINE, slab = slab)
  expect_true(all(fit$theta_grid[, "sigma"] >= slab[1L]))
  expect_true(all(fit$theta_grid[, "sigma"] <= slab[2L]))
  expect_identical(.nl_edge_mass_axes(fit), "sigma:upper")
})

test_that("an evenly spaced grid keeps the equal weights it always had", {
  # The rule the engine applied before the nodes carried weights, so an
  # unrefined grid over a flat declared measure is unchanged.
  lev <- exp(seq(log(0.2), log(5), length.out = 6L))
  spec <- hyper_axis_spec("sigma", grid = lev, log_scale = TRUE,
                          bounds = c(0, Inf))
  w <- .hyper_axis_level_weights(lev, spec)
  expect_equal(unname(w), rep(1 / 6, 6), tolerance = 1e-12)
})

test_that("tulpa_hyper_grid refuses an unknown control knob", {
  # The consistency-pass trigger moved from a ratio against the parabola to an
  # ESS on the weights, so the retired spelling has to say so rather than be
  # ignored.
  expect_true("var_of_means_min_ess" %in% .CONTROL_KEYS$hyper_grid)
  expect_false("var_of_means_tolerance" %in% .CONTROL_KEYS$hyper_grid)
  expect_error(
    tulpa_hyper_grid(list(hyper_axis_spec("sigma", grid = c(1, 2))),
                     function(hypers) list(log_marginal = 0),
                     control = list(var_of_means_tolerance = 0.7)),
    "Unknown control knob")
})
