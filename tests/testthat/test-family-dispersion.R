# Dispersion derivatives, checked against the registry they differentiate.
#
# These three functions per family are hand-derived, and a wrong one does not
# announce itself: it biases every empirical-Bayes dispersion estimate by a
# smooth amount and still converges. So each is finite-differenced against the
# EXACT `loglik` / `score` / `weight` registered for that family in
# .FAMILY_OPS. Differentiating the registry rather than a textbook form is the
# point -- phi is a size for neg_binomial_2, a variance for gaussian, a shape
# for gamma and a precision for beta, and a derivative correct for one
# parameterization is wrong for another.

# Central difference in phi, on the log scale where phi > 0 so the step stays
# proportional to the value.
fd_dphi <- function(f, phi, h = 1e-6) {
  hp <- phi * h
  (f(phi + hp) - f(phi - hp)) / (2 * hp)
}

# Draw a y in the family's support so the log-density is finite: the
# derivatives are evaluated at realized data, not at the mean.
dispersion_case <- function(family, phi, n = 40L, seed = 3L) {
  set.seed(seed)
  eta <- switch(family,
    neg_binomial_2 = rnorm(n, 0.6, 0.4),
    gaussian       = rnorm(n, 1.0, 0.7),
    gamma          = rnorm(n, 0.5, 0.3),
    beta           = rnorm(n, 0.2, 0.5))
  ops <- tulpa:::.FAMILY_OPS[[family]]
  y <- ops$sample(eta, NULL, phi)
  # Beta's log-density diverges at the open boundary; nudge inside it.
  if (family == "beta") y <- pmin(pmax(y, 1e-6), 1 - 1e-6)
  if (family == "gamma") y <- pmax(y, 1e-8)
  list(eta = eta, y = y, ops = ops, d = tulpa:::.family_dphi(family))
}

FAMILY_PHI <- list(
  neg_binomial_2 = c(0.8, 2.5, 9.0),
  gaussian       = c(0.3, 1.0, 4.0),
  gamma          = c(0.7, 3.0, 12.0)
)


test_that("dloglik/dphi matches a finite difference of the registered loglik", {
  for (fam in names(FAMILY_PHI)) {
    for (phi in FAMILY_PHI[[fam]]) {
      cs <- dispersion_case(fam, phi)
      analytic <- cs$d$dloglik(cs$eta, cs$y, NULL, phi)
      numeric <- fd_dphi(function(p) cs$ops$loglik(cs$eta, cs$y, NULL, p), phi)
      expect_equal(analytic, numeric, tolerance = 1e-5,
                   info = sprintf("%s, phi = %g", fam, phi))
    }
  }
})


test_that("dscore/dphi matches a finite difference of the registered score", {
  # The score derivative is what carries the mode's movement with phi
  # (dx_hat/dphi); an error here tilts the gradient without changing the fit at
  # any fixed phi, so nothing else would catch it.
  for (fam in names(FAMILY_PHI)) {
    for (phi in FAMILY_PHI[[fam]]) {
      cs <- dispersion_case(fam, phi)
      analytic <- cs$d$dscore(cs$eta, cs$y, NULL, phi)
      numeric <- fd_dphi(function(p) cs$ops$score(cs$eta, cs$y, NULL, p), phi)
      expect_equal(analytic, numeric, tolerance = 1e-5,
                   info = sprintf("%s, phi = %g", fam, phi))
    }
  }
})


test_that("dweight/dphi differentiates the weight H is actually built from", {
  # The target is NOT uniformly `.FAMILY_OPS$weight`. The Laplace Hessian uses
  # the OBSERVED curvature for neg_binomial_2 (`obs_weight`, and the branch at
  # laplace_family_curvature.h:126), while gaussian / gamma / beta build H from
  # the expected weight -- beta's `variance_fn` is defined so that dmu^2 / V is
  # exactly the Fisher information, so the two coincide there.
  #
  # Differentiating the expected form for neg_binomial_2 is wrong by a few
  # percent: enough to move the maximizer, small enough that a recovery test
  # would still converge and look plausible. Naming the right target per family
  # is the whole content of this test.
  h_weight <- list(
    neg_binomial_2 = function(ops, eta, y, phi) ops$obs_weight(eta, y, NULL, phi),
    gaussian       = function(ops, eta, y, phi) ops$weight(eta, NULL, phi),
    gamma          = function(ops, eta, y, phi) ops$weight(eta, NULL, phi)
  )
  for (fam in names(FAMILY_PHI)) {
    for (phi in FAMILY_PHI[[fam]]) {
      cs <- dispersion_case(fam, phi)
      analytic <- cs$d$dweight(cs$eta, cs$y, NULL, phi)
      numeric <- fd_dphi(
        function(p) h_weight[[fam]](cs$ops, cs$eta, cs$y, p), phi)
      expect_equal(analytic, numeric, tolerance = 1e-5,
                   info = sprintf("%s, phi = %g", fam, phi))
    }
  }
})


test_that("the neg_binomial_2 weight derivative reduces to the expected one at y = mu", {
  # Averaging the observed curvature over y returns the expected weight, so
  # their phi-derivatives must agree at y = E[y] = mu. This pins the observed
  # form against the simpler expression it generalizes, independently of the
  # finite difference above.
  eta <- seq(-0.8, 1.2, length.out = 7)
  mu <- exp(eta)
  for (phi in c(0.8, 2.5, 9.0)) {
    d <- tulpa:::.family_dphi("neg_binomial_2")
    at_mean <- d$dweight(eta, mu, NULL, phi)
    expected_form <- mu^2 / (mu + phi)^2
    expect_equal(at_mean, expected_form, tolerance = 1e-12,
                 info = sprintf("phi = %g", phi))
  }
})


test_that("the derivatives are registered only where phi is free", {
  # poisson and binomial have no dispersion at all, so estimating one is not a
  # missing feature but a category error; .family_dphi() must refuse rather
  # than return something the optimizer would happily walk along.
  expect_null(tulpa:::.family_dphi("poisson"))
  expect_null(tulpa:::.family_dphi("binomial"))

  # beta's derivatives exist but are parked: the assembled gradient is not
  # exact for it (H is the Fisher matrix there, while the mode-movement solve
  # needs the observed one). It must not be reachable by name.
  expect_null(tulpa:::.family_dphi("beta"))
  expect_false("beta" %in% tulpa:::.dispersion_families())
  expect_false(any(startsWith(tulpa:::.dispersion_families(), ".")))

  for (fam in names(FAMILY_PHI)) {
    d <- tulpa:::.family_dphi(fam)
    expect_true(is.list(d), info = fam)
    expect_setequal(names(d), c("dloglik", "dscore", "dweight"))
  }
})


test_that("every dispersion family has a registered weight to differentiate", {
  # The dweight entries differentiate .FAMILY_OPS$weight. A family listed here
  # whose registry entry lacked one would leave dweight describing nothing.
  for (fam in tulpa:::.dispersion_families()) {
    ops <- tulpa:::.FAMILY_OPS[[fam]]
    expect_true(is.function(ops$weight), info = fam)
    expect_true(is.function(ops$score), info = fam)
    expect_true(is.function(ops$loglik), info = fam)
  }
})
