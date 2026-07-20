# Prior on the zero-inflation coefficients, beta_zi ~ N(0, zi_prior_sd^2).
#
# The prior is load-bearing rather than decorative. Where a ZI design level
# contributes no zeros, the likelihood is monotone in that level's beta_zi --
# every step toward -Inf raises it -- so the mode exists only because the prior
# is there. It also has to be the SAME prior on every backend: the Laplace path
# appends it to the fixed-effect BetaPrior tail while the samplers carry it as
# ModelData::zi_prior_sd, and if those disagree then one `ziformula` means two
# different models depending on `mode`.

zi_pois_data <- function(seed = 42L, n = 400L, p_zero = 0.3) {
  set.seed(seed)
  x <- rnorm(n)
  y <- rpois(n, exp(0.5 + 0.4 * x))
  y[runif(n) < p_zero] <- 0L
  data.frame(y = y, x = x)
}

# A ZI design where one level never produces a zero: the likelihood alone sends
# that level's coefficient to -Inf, so whatever the fit returns for it is the
# prior's doing.
zi_separated_data <- function(seed = 7L, n_per = 120L) {
  set.seed(seed)
  g <- rep(c("has_zeros", "no_zeros"), each = n_per)
  y <- integer(length(g))
  y[g == "has_zeros"] <- rpois(n_per, 2)
  y[g == "has_zeros"][runif(n_per) < 0.45] <- 0L
  # Strictly positive counts in this level, by construction.
  y[g == "no_zeros"] <- rpois(n_per, 4) + 1L
  data.frame(y = y, g = factor(g))
}

zi_coef <- function(fit) {
  cf <- coef(fit)
  cf[grepl("^zi_", names(cf))]
}


test_that("the ZI prior does not depend on whether beta_prior was supplied", {
  # The regression this pins: the ZI block's prior used to be appended only
  # when the caller passed a fixed-effect prior, so `beta_prior` -- which
  # governs the COUNT block -- silently set the ZI block's prior SD to 2.5 when
  # present and left it at the weak default of 100 when absent. The two blocks
  # are separate; supplying a beta prior equal to the built-in default must
  # therefore leave the ZI coefficient untouched.
  d <- zi_pois_data()

  fit_default <- tulpa(y ~ x, data = d, family = "poisson",
                       ziformula = ~1, mode = "laplace")
  # sd = 100 IS the built-in default fixed-effect prior, so this changes
  # nothing about the count block and must change nothing at all.
  fit_bp <- tulpa(y ~ x, data = d, family = "poisson",
                  ziformula = ~1, mode = "laplace",
                  beta_prior = list(mean = 0, sd = 100))

  expect_equal(fit_default$mode, fit_bp$mode, tolerance = 1e-10)
})


test_that("zi_prior$sd changes the zero-inflation coefficient", {
  d <- zi_pois_data()

  fit_default <- tulpa(y ~ x, data = d, family = "poisson",
                       ziformula = ~1, mode = "laplace")
  fit_tight <- tulpa(y ~ x, data = d, family = "poisson",
                     ziformula = ~1, mode = "laplace",
                     zi_prior = list(sd = 0.25))

  zi_default <- zi_coef(fit_default)
  zi_tight <- zi_coef(fit_tight)

  expect_length(zi_default, 1L)
  # A mean-zero prior an order of magnitude tighter shrinks the coefficient
  # toward 0 rather than merely perturbing it.
  expect_lt(abs(zi_tight), abs(zi_default))
  expect_gt(abs(zi_default - zi_tight), 0.1)
})


test_that("the ZI prior identifies a level that contributes no zeros", {
  d <- zi_separated_data()
  expect_true(all(d$y[d$g == "no_zeros"] > 0))

  fit <- tulpa(y ~ 1, data = d, family = "poisson",
               ziformula = ~ g, mode = "laplace")
  expect_true(fit$converged)

  cf <- coef(fit)
  slope <- cf[["zi_gno_zeros"]]
  # Unpenalized this diverges. The default N(0, 2.5^2) is what bounds it, so a
  # finite estimate on the prior's scale is the property under test.
  expect_true(is.finite(slope))
  expect_lt(abs(slope), 6 * 2.5)

  # Tightening the prior must pull it further in: confirms the bound is the
  # prior's doing and not the likelihood's.
  fit_tight <- tulpa(y ~ 1, data = d, family = "poisson",
                     ziformula = ~ g, mode = "laplace",
                     zi_prior = list(sd = 0.5))
  expect_lt(abs(coef(fit_tight)[["zi_gno_zeros"]]), abs(slope))
})


test_that("the Laplace and sampler paths carry the same ZI prior default", {
  skip_on_cran()
  # Same `ziformula`, same prior: the two backends build the beta_zi penalty
  # through different code (BetaPrior tail vs ModelData::zi_prior_sd), so this
  # pins them to one number. Compared at the mode, which the sampler recovers
  # up to MC error.
  d <- zi_pois_data(seed = 11L, n = 600L)

  fit_lap <- tulpa(y ~ x, data = d, family = "poisson",
                   ziformula = ~1, mode = "laplace")
  fit_mcmc <- tulpa(y ~ x, data = d, family = "poisson",
                    ziformula = ~1, mode = "hmc",
                    control = list(n_iter = 1500L, n_warmup = 750L, seed = 3L))

  zi_lap <- zi_coef(fit_lap)
  zi_mcmc <- mean(as.matrix(fit_mcmc$draws)[, "beta_zi[1]"])

  expect_equal(unname(zi_lap), unname(zi_mcmc), tolerance = 0.15)
})


test_that("zi_prior rejects malformed input", {
  d <- zi_pois_data(n = 200L)
  zi_fit <- function(...) {
    tulpa(y ~ x, data = d, family = "poisson", ziformula = ~1,
          mode = "laplace", ...)
  }

  expect_error(zi_fit(zi_prior = 2.5), "must be NULL or a list")
  expect_error(zi_fit(zi_prior = list()), "must supply `sd`")
  expect_error(zi_fit(zi_prior = list(sd = -1)), "must be positive")
  expect_error(zi_fit(zi_prior = list(sd = 0)), "must be positive")
  expect_error(zi_fit(zi_prior = list(sd = c(1, 2))), "must be a scalar")
  # A per-coefficient mean is not something the kernels can carry, so it is
  # refused rather than silently dropped.
  expect_error(zi_fit(zi_prior = list(mean = 1, sd = 2)), "mean.*not supported")
})


test_that("zi_prior = list(sd = Inf) removes the penalty", {
  d <- zi_pois_data()
  fit_inf <- tulpa(y ~ x, data = d, family = "poisson",
                   ziformula = ~1, mode = "laplace",
                   zi_prior = list(sd = Inf))
  fit_wide <- tulpa(y ~ x, data = d, family = "poisson",
                    ziformula = ~1, mode = "laplace",
                    zi_prior = list(sd = 1e6))
  # tau = 1/sd^2 is 0 at Inf and 1e-12 at 1e6; on a well-identified logit both
  # are effectively unpenalized and must agree.
  expect_equal(zi_coef(fit_inf), zi_coef(fit_wide), tolerance = 1e-6)
})
