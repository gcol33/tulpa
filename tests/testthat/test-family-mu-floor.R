# One floor on mu, read by the density, the score, the Newton working weight and
# both curvature ladders (laplace_family_link.h, laplace_family_curvature.h).
#
# The generic mu-space route evaluates a family through its link, and every arm
# of it floors mu before dividing by it. Two floors is the failure this file
# pins: with the density clamped at 1e-15 and the score at 1e-7, the band
# mu in (1e-15, 1e-7) evaluates the density at the true mu and the score at the
# floor, so `grad` is scaled by mu / 1e-7 and the two are derivatives of
# different functions. tulpa_laplace() line-searches on the density and steps
# along the score, so a Newton loop in that band stops where the clamped score
# is near zero rather than where the objective it reports is stationary.
#
# The band opens at eta near -5.2 for a probit binomial, an ordinary linear
# predictor for a rare-event model, which is why the existing family/link
# finite-difference grids (eta in [-0.8, 1.6], mu near 0.25 to 0.8) never
# reached it.

BAND_ETA <- list(
  binomial_probit  = c(-4, -5, -6, -7, -7.5),
  binomial_cloglog = c(-14, -18, -25, -30),
  beta_probit      = c(-4, -5, -6, -7, -7.5)
)

test_that("the score is the eta-derivative of the reported density deep in the tail", {
  # Both quantities come from the same call, so this compares the two clamps
  # against each other rather than either against an outside reference.
  fd <- function(f, e) {
    h <- max(1e-6, 1e-7 * abs(e))
    (f(e + h) - f(e - h)) / (2 * h)
  }
  cases <- list(
    list(fam = "binomial_probit",  y = 1,   n = 1L, phi = 1),
    list(fam = "binomial_cloglog", y = 1,   n = 1L, phi = 1),
    list(fam = "beta_probit",      y = 0.6, n = 1L, phi = 6)
  )
  for (cs in cases) {
    ll <- function(z) cpp_family_terms(cs$y, cs$n, z, cs$fam, cs$phi)[["log_lik"]]
    for (e in BAND_ETA[[cs$fam]]) {
      got <- cpp_family_terms(cs$y, cs$n, e, cs$fam, cs$phi)[["grad"]]
      expect_equal(got, fd(ll, e), tolerance = 1e-6,
                   info = sprintf("%s at eta = %g (mu = %.2e)", cs$fam, e,
                                  exp(ll(e))))
    }
  }
})

test_that("the worked example of the two-floor band returns the exact score", {
  # binomial_probit, y = 1, n = 1, eta = -6: mu = 9.87e-10 sits between the two
  # floors. The exact score is dnorm(-6) / pnorm(-6) = 6.158; a score clamped at
  # 1e-7 returns 6.076e-2, low by the factor mu / 1e-7 = 101, which grows
  # without bound as eta decreases.
  got <- cpp_family_terms(1, 1L, -6, "binomial_probit", 1)[["grad"]]
  expect_equal(got, dnorm(-6) / pnorm(-6), tolerance = 1e-10)
  # ... and the working weight, dmu^2 / V, carries the same factor.
  mu <- pnorm(-6)
  expect_equal(cpp_family_terms(1, 1L, -6, "binomial_probit", 1)[["neg_hess"]],
               dnorm(-6)^2 / (mu * (1 - mu)), tolerance = 1e-10)
})

test_that("the density and the score clamp mu at the same place", {
  # Straddle the analytic floor. Above it neither is clamped, below it both are,
  # and the reference computes the clamped mu itself -- so a score reading a
  # different floor misses on one side or the other.
  muf   <- 1e-15
  eta_f <- qnorm(muf)
  clamp <- function(mu) max(min(mu, 1 - muf), muf)
  for (e in c(eta_f + 0.05, eta_f - 0.05)) {
    r  <- cpp_family_terms(1, 1L, e, "binomial_probit", 1)
    mc <- clamp(pnorm(e))
    expect_equal(r[["log_lik"]], log(mc), tolerance = 1e-12,
                 info = sprintf("density at eta = %.6f", e))
    expect_equal(r[["grad"]], (1 - mc) / (mc * (1 - mc)) * dnorm(e),
                 tolerance = 1e-10,
                 info = sprintf("score at eta = %.6f", e))
  }
})

test_that("a family whose mu is not on the unit interval shares the same floor", {
  # The other arm of clamp_mu_for_family is max(mu, kMuFloor). poisson_sqrt has
  # mu = eta^2, so the floor is reachable at an ordinary eta. The working weight
  # dmu^2 / V = 4 eta^2 / mu_clamped is what pins which value was used: at
  # eta = 1e-9 it reads 4e-3 off a 1e-15 floor and 4e-8 off a 1e-10 one.
  for (e in c(4e-8, 3e-8, 1e-9)) {
    mc <- max(e^2, 1e-15)
    r  <- cpp_family_terms(0, 1L, e, "poisson_sqrt", 1)
    expect_equal(r[["log_lik"]], -mc, tolerance = 1e-18,
                 info = sprintf("poisson_sqrt density at eta = %.1e", e))
    expect_equal(r[["neg_hess"]], (2 * e)^2 / mc, tolerance = 1e-10,
                 info = sprintf("poisson_sqrt weight at eta = %.1e", e))
  }
})

test_that("cauchit's score reads the same mu its density reports", {
  # pcauchy decays like 1/|eta|, so reaching mu < 1e-7 takes |eta| > 3e6, where a
  # finite difference of the density is dominated by cancellation. The reference
  # is built from the mu the density itself reports (y = 1, n = 1 makes the
  # log-likelihood log(mu)), which is what this file is about: one mu behind
  # both. It is deliberately not pcauchy(eta) -- the engine's cauchit linkinv is
  # 0.5 + atan(eta) / pi, which loses digits to cancellation out here, and that
  # is a separate defect from the floor.
  for (e in c(-1e6, -1e7, -1e8)) {
    r  <- cpp_family_terms(1, 1L, e, "binomial_cauchit", 1)
    mu <- exp(r[["log_lik"]])
    expect_equal(r[["grad"]], (1 - mu) / (mu * (1 - mu)) * dcauchy(e),
                 tolerance = 1e-9, info = sprintf("cauchit at eta = %g", e))
  }
})

test_that("both curvature ladders are derivatives of the weight in the band", {
  # curvature_deta / curvature_deta2 feed gamma_3 and the debias correction, and
  # carry their own copy of the clamp. Finite differences bind each rung to the
  # one below it at etas inside the band.
  fd <- function(f, e) {
    h <- max(1e-6, 1e-7 * abs(e))
    (f(e + h) - f(e - h)) / (2 * h)
  }
  for (e in BAND_ETA$binomial_probit) {
    nh <- function(z) cpp_family_terms(1, 1L, z, "binomial_probit", 1)[["neg_hess"]]
    dw <- function(z) cpp_family_curvature_deta(1, 1L, z, "binomial_probit", 1)[["dw_deta"]]
    expect_equal(dw(e), fd(nh, e), tolerance = 1e-6,
                 info = sprintf("dw_deta at eta = %g", e))
    expect_equal(
      cpp_family_curvature_deta2(1, 1L, e, "binomial_probit", 1)[["d2w_deta2"]],
      fd(dw, e), tolerance = 1e-6,
      info = sprintf("d2w_deta2 at eta = %g", e))
  }
})
