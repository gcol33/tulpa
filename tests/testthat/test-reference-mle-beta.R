# Beta regression against betareg.
#
# betareg fits the same mean-precision Beta GLM (Ferrari & Cribari-Neto 2004)
# by ML, so it is the oracle for both halves of the likelihood: the mean
# coefficients at a held precision, and the precision itself. tulpa reaches the
# precision three ways -- conditioned through tulpa(), profiled by Brent in
# tulpa_laplace_beta(), and marginalized by NUTS in tulpa_nuts_beta() -- and all
# three are checked against the same MLE.

make_beta_data <- function(N = 600L, beta = c(0.3, 0.7), kappa = 30,
                           seed = 2026L) {
  set.seed(seed)
  x  <- rnorm(N)
  mu <- plogis(beta[1] + beta[2] * x)
  data.frame(y = rbeta(N, mu * kappa, (1 - mu) * kappa), x = x)
}


test_that("beta reproduces betareg at the reference precision", {
  skip_on_cran()
  skip_if_not_installed("betareg")
  d <- make_beta_data()

  ref <- betareg::betareg(y ~ x, data = d)
  co  <- summary(ref)$coefficients

  fit <- suppressMessages(
    tulpa(y ~ x, data = d, family = "beta", mode = "laplace",
          phi = co$precision["(phi)", "Estimate"],
          beta_prior = list(mean = 0, sd = REF_DIFFUSE_SD)))

  expect_agrees_with_mle(coef(fit), co$mean[, "Estimate"],
                         co$mean[, "Std. Error"], label = "beta")
})


test_that("the beta profile log-marginal peaks at betareg's precision", {
  skip_on_cran()
  skip_if_not_installed("betareg")
  d <- make_beta_data()

  ref     <- betareg::betareg(y ~ x, data = d)
  phi_mle <- summary(ref)$coefficients$precision["(phi)", "Estimate"]

  grid <- seq(20, 45, by = 0.25)
  expect_lt(abs(ref_profile_phi(y ~ x, d, "beta", grid,
                                beta_prior = list(mean = 0,
                                                  sd = REF_DIFFUSE_SD)) -
                  phi_mle) / phi_mle, 0.02)
})


test_that("tulpa_laplace_beta()'s profiled precision matches betareg's", {
  skip_on_cran()
  skip_if_not_installed("betareg")
  d <- make_beta_data()

  ref <- betareg::betareg(y ~ x, data = d)
  co  <- summary(ref)$coefficients

  fit <- tulpa_laplace_beta(d$y, cbind(1, d$x))
  expect_true(fit$phi_converged)

  # The Brent outer step maximizes the same profile the grid above walks, so it
  # lands on the MLE up to the prior's pull on the mean coefficients.
  expect_lt(abs(fit$phi - co$precision["(phi)", "Estimate"]) /
              co$precision["(phi)", "Estimate"], 0.02)
  expect_agrees_with_mle(fit$mode[seq_len(2L)], co$mean[, "Estimate"],
                         co$mean[, "Std. Error"], tol = 0.05,
                         label = "laplace_beta")
})


test_that("tulpa_nuts_beta()'s posterior mean matches betareg's MLE", {
  skip_if_not_slow()
  skip_if_not_installed("betareg")
  d <- make_beta_data()

  ref <- betareg::betareg(y ~ x, data = d)
  co  <- summary(ref)$coefficients

  fit <- tulpa_nuts_beta(d$y, cbind(1, d$x),
                         control = list(n_iter = 1500L, n_warmup = 750L,
                                        seed = 7L))
  expect_equal(sum(fit$divergent), 0L)

  pm <- colMeans(fit$draws)
  expect_agrees_with_mle(pm[seq_len(2L)], co$mean[, "Estimate"],
                         co$mean[, "Std. Error"], tol = 0.25,
                         label = "nuts_beta")

  # betareg reports the precision on its original scale; NUTS samples
  # log(phi), so the reference SE transfers by the delta method.
  phi_mle <- co$precision["(phi)", "Estimate"]
  phi_se  <- co$precision["(phi)", "Std. Error"]
  expect_agrees_with_mle(pm[["log_phi"]], log(phi_mle), phi_se / phi_mle,
                         tol = 0.25, label = "nuts_beta log_phi")
})
