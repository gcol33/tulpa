# Fixed-effect standard errors on a nested-Laplace fit carrying an INTRINSIC
# latent field (ICAR, RW1, RW2, and the s(...) smoother built on RW2).
#
# An intrinsic field's constant null direction is jointly unidentified with the
# intercept. The engine identifies it by augmenting the field precision during
# the inner solve (laplace_s2z.h), which puts the level under the field's own
# tau and leaves the intercept informed by the data.
#
# The failure this guards against is specific and quiet: identify the level
# ONLY by centring the mode afterwards (latent_block.h `center_intercept`,
# which folds the removed mean into the intercept so eta is preserved). That
# gets the intercept POINT ESTIMATE right, and leaves the (intercept, level)
# direction flat in the Hessian the marginal covariance is read from -- so the
# intercept standard error silently collapses to the fixed-effect prior SD
# (sigma_beta = 100). Point estimates, field shape, log-marginals and every
# other coefficient stay correct, so nothing else in the suite moves.
#
# The assertions below are therefore on the INTERCEPT's SE specifically, and
# are scaled against that prior: an SE anywhere near 100 means the pin is not
# reaching the Hessian.

# Fixed-effect prior SD the nested kernels use (nested_laplace.cpp sigma_beta).
.SIGMA_BETA <- 100

.rw_data <- function(seed = 7, n = 400L, T = 40L) {
  set.seed(seed)
  tt <- rep(seq_len(T), length.out = n)
  x  <- rnorm(n)
  d  <- data.frame(x = x, tt = tt)
  d$y <- rpois(n, exp(0.3 + 0.5 * x + 0.4 * sin(tt / 6)))
  d
}

test_that("an RW1 field leaves the intercept informed by the data", {
  skip_on_cran()
  d <- .rw_data()
  tab <- tulpa:::.fit_fixed_table(
    tulpa(y ~ x, data = d, family = "poisson",
          temporal = temporal_rw1(time_var = "tt")))

  se_int <- tab$std.error[1]
  expect_true(is.finite(se_int) && se_int > 0)
  # The defect: the intercept learns nothing beyond its own prior.
  expect_lt(se_int, .SIGMA_BETA / 100)
  # Recovery, so a pin of the wrong size is caught too rather than any pin.
  expect_equal(tab$estimate[1], 0.3, tolerance = 0.25)
  expect_equal(tab$estimate[2], 0.5, tolerance = 0.1)

  # The slope shares the Hessian but not the aliased direction, so it was
  # correct even when the intercept was not -- it must stay correct.
  expect_lt(tab$std.error[2], 0.2)
})

test_that("RW2, a cyclic RW1 and a smoother identify their level too", {
  skip_on_cran()
  d <- .rw_data()
  se_int <- function(fit) tulpa:::.fit_fixed_table(fit)$std.error[1]

  expect_lt(se_int(tulpa(y ~ x, data = d, family = "poisson",
                         temporal = temporal_rw2(time_var = "tt"))),
            .SIGMA_BETA / 100)
  expect_lt(se_int(tulpa(y ~ x, data = d, family = "poisson",
                         temporal = temporal_rw1(time_var = "tt",
                                                 cyclic = TRUE))),
            .SIGMA_BETA / 100)

  # s(x) is an RW2 over the binned covariate, so it inherits the same path.
  set.seed(5)
  ds <- data.frame(xx = runif(400, -2, 2))
  ds$y <- rpois(400, exp(0.3 + sin(2 * ds$xx)))
  fs <- tulpa(y ~ s(xx, k = 15), data = ds, family = "poisson")
  expect_lt(se_int(fs), .SIGMA_BETA / 100)
  # The pin must not have disturbed what the smoother already did right.
  sm <- smooth_effects(fs)
  expect_equal(sum(sm$estimate), 0, tolerance = 1e-8)
})

test_that("an ICAR field leaves the intercept informed by the data", {
  skip_on_cran()
  set.seed(1)
  S <- 30L
  nb <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
  nn <- lengths(nb)
  field <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4))))
  idx <- rep(seq_len(S), each = 8L)
  n <- length(idx)
  x <- rnorm(n)
  y <- rbinom(n, 1L, plogis(-0.2 + 0.6 * x + field[idx]))
  fit <- tulpa_nested_laplace(
    y, rep(1L, n), cbind(1, x),
    prior = list(type = "icar", n_spatial_units = S, spatial_idx = idx,
                 adj_row_ptr = c(0L, cumsum(nn)), adj_col_idx = unlist(nb) - 1L,
                 n_neighbors = nn, tau_grid = c(0.5, 1, 2, 4, 8)),
    family = "binomial",
    control = list(keep_grid_hessians = TRUE, diagnose_k = FALSE))

  expect_lt(tulpa:::.fit_fixed_table(fit)$std.error[1], .SIGMA_BETA / 100)
})

test_that("the pin's two storages give the same standard errors", {
  skip_on_cran()
  # The augmentation's 11' is either written into the sparse pattern or folded
  # in at solve time (S2ZStorage / s2z_densify). Exact either way, so the fitted
  # standard errors must not depend on which side of the cutoff a field falls.
  d <- .rw_data()
  se <- function() {
    tulpa:::.fit_fixed_table(tulpa(y ~ x, data = d, family = "poisson",
                           temporal = temporal_rw1(time_var = "tt")))$std.error
  }
  old <- Sys.getenv("TULPA_S2Z_DENSIFY_MAX", unset = NA)
  on.exit(if (is.na(old)) Sys.unsetenv("TULPA_S2Z_DENSIFY_MAX")
          else Sys.setenv(TULPA_S2Z_DENSIFY_MAX = old), add = TRUE)

  Sys.setenv(TULPA_S2Z_DENSIFY_MAX = "0")
  folded <- se()
  Sys.setenv(TULPA_S2Z_DENSIFY_MAX = "10000")
  stored <- se()
  expect_equal(folded, stored, tolerance = 1e-8)
})

test_that("the RW1 intercept posterior matches an exact-MCMC fit", {
  skip_if_not_slow()
  # The reference: the sampler path identifies the same direction
  # (tulpa_priors_temporal.h), so its intercept posterior is what the
  # nested-Laplace approximation is approximating.
  d <- .rw_data()
  nl <- tulpa:::.fit_fixed_table(
    tulpa(y ~ x, data = d, family = "poisson",
          temporal = temporal_rw1(time_var = "tt")))
  nu <- tulpa:::.fit_fixed_table(
    tulpa(y ~ x, data = d, family = "poisson",
          temporal = temporal_rw1(time_var = "tt"), mode = "hmc",
          control = list(n_iter = 2000L, n_warmup = 1000L, seed = 11L)))

  expect_equal(nl$estimate[1], nu$estimate[1], tolerance = 0.05)
  expect_equal(nl$estimate[2], nu$estimate[2], tolerance = 0.05)
  # The two paths place the field level differently -- the Laplace path folds it
  # into the intercept, the sampler centres it out of eta -- so the intercept
  # SEs agree in scale rather than to sampling error. Before the pin the ratio
  # was ~2000.
  expect_lt(nl$std.error[1] / nu$std.error[1], 2)
  expect_gt(nl$std.error[1] / nu$std.error[1], 0.5)
  expect_equal(nl$std.error[2], nu$std.error[2], tolerance = 0.25)
})
