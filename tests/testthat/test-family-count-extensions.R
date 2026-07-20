# Density-level validation for the count families added on top of the Poisson /
# neg_binomial_2 pair: neg_binomial_1 (linear variance) and the two
# zero-truncated families. Normalization is the load-bearing check -- a wrong
# constant or a dropped truncation term shows up there and nowhere else, since
# score and weight are both invariant to a constant offset in the log-density.
#
# These validate the family math. Parameter recovery against simulated truth
# belongs with the compiled kernels; until those carry the families, tulpa()
# refuses them (see the .R_ONLY_FAMILIES gate below) so there is nothing to
# recover through yet.

support_sum <- function(family, eta, phi, ymin) {
  yy <- ymin:20000
  sum(exp(family_loglik(rep(eta, length(yy)), yy, family, phi = phi)))
}

# Exact moments and expected curvature, taken over the support under the
# family's own density rather than assumed in closed form.
support_moments <- function(family, eta, phi, ymin) {
  yy <- ymin:20000
  pr <- exp(family_loglik(rep(eta, length(yy)), yy, family, phi = phi))
  m <- sum(pr * yy)
  h <- 1e-4
  d2 <- (family_loglik(rep(eta + h, length(yy)), yy, family, phi = phi) -
         2 * family_loglik(rep(eta, length(yy)), yy, family, phi = phi) +
         family_loglik(rep(eta - h, length(yy)), yy, family, phi = phi)) / h^2
  list(mean = m, var = sum(pr * (yy - m)^2), fisher = -sum(pr * d2))
}

grid_eta <- c(-1, 0, 1.5)
grid_phi <- c(0.5, 2, 8)


test_that("the added count families are normalized over their support", {
  for (eta in grid_eta) {
    for (phi in grid_phi) {
      expect_equal(support_sum("neg_binomial_1", eta, phi, 0), 1,
                   tolerance = 1e-9)
      expect_equal(support_sum("truncated_neg_binomial_2", eta, phi, 1), 1,
                   tolerance = 1e-9)
    }
    expect_equal(support_sum("truncated_poisson", eta, 1, 1), 1,
                 tolerance = 1e-9)
  }
})


test_that("the added count families put no mass on y = 0 when truncated", {
  for (fam in c("truncated_poisson", "truncated_neg_binomial_2")) {
    expect_true(all(is.infinite(family_loglik(grid_eta, 0, fam, phi = 2)) &
                      family_loglik(grid_eta, 0, fam, phi = 2) < 0))
  }
})


test_that("scores of the added count families match finite differences", {
  h <- 1e-6
  for (spec in list(list("neg_binomial_1", 0), list("truncated_poisson", 1),
                    list("truncated_neg_binomial_2", 1))) {
    fam <- spec[[1]]; ymin <- spec[[2]]
    for (eta in grid_eta) {
      for (phi in grid_phi) {
        y <- ymin:12
        fd <- (family_loglik(eta + h, y, fam, phi = phi) -
               family_loglik(eta - h, y, fam, phi = phi)) / (2 * h)
        expect_equal(family_score_eta(eta, y, fam, phi = phi), fd,
                     tolerance = 1e-5)
      }
    }
  }
})


test_that("truncated working weights equal the exact expected curvature", {
  # Both truncated families register the exact Fisher information, so the
  # registered weight must reproduce E[-d2 loglik / deta2] on the nose.
  for (eta in grid_eta) {
    ref <- support_moments("truncated_poisson", eta, 1, 1)
    expect_equal(family_weight(eta, "truncated_poisson"), ref$fisher,
                 tolerance = 1e-5)
    for (phi in grid_phi) {
      ref <- support_moments("truncated_neg_binomial_2", eta, phi, 1)
      expect_equal(family_weight(eta, "truncated_neg_binomial_2", phi = phi),
                   ref$fisher, tolerance = 1e-4)
    }
  }
})


test_that("neg_binomial_1 registers the moment weight, under the exact information", {
  # Unlike neg_binomial_2 -- whose moment weight coincides with the expected
  # curvature -- NB1's shape r = mu/phi moves with the mean, so the exact
  # information r^2 (trigamma(r) - E[trigamma(y + r)]) has no elementary form
  # and the registered quasi-likelihood weight sits strictly below it. Pin both
  # the registered value and the direction of the gap so a future switch to the
  # exact series is a deliberate change rather than a silent one.
  for (eta in grid_eta) {
    for (phi in grid_phi) {
      mu <- exp(eta)
      w <- family_weight(eta, "neg_binomial_1", phi = phi)
      expect_equal(w, mu / (1 + phi), tolerance = 1e-12)

      r <- mu / phi
      yy <- 0:60000
      pr <- stats::dnbinom(yy, size = r, mu = mu)
      exact <- r^2 * (trigamma(r) - sum(pr * trigamma(yy + r)))
      expect_equal(support_moments("neg_binomial_1", eta, phi, 0)$fisher,
                   exact, tolerance = 1e-5)
      expect_lt(w, exact)
    }
  }
})


test_that("variance and response_mean of the added count families are exact", {
  for (spec in list(list("neg_binomial_1", 0), list("truncated_poisson", 1),
                    list("truncated_neg_binomial_2", 1))) {
    fam <- spec[[1]]; ymin <- spec[[2]]
    for (eta in grid_eta) {
      for (phi in grid_phi) {
        ref <- support_moments(fam, eta, phi, ymin)
        expect_equal(family_variance(eta, fam, phi = phi), ref$var,
                     tolerance = 1e-8)
        expect_equal(family_response_mean(eta, fam, phi = phi), ref$mean,
                     tolerance = 1e-8)
      }
    }
  }
})


test_that("samplers of the added count families respect support and moments", {
  skip_on_cran()
  set.seed(20260720)
  for (spec in list(list("neg_binomial_1", 0), list("truncated_poisson", 1),
                    list("truncated_neg_binomial_2", 1))) {
    fam <- spec[[1]]; ymin <- spec[[2]]
    eta <- 0.7; phi <- 2
    draws <- family_sample(rep(eta, 4e5), fam, phi = phi)
    ref <- support_moments(fam, eta, phi, ymin)
    expect_gte(min(draws), ymin)
    expect_equal(mean(draws), ref$mean, tolerance = 0.02)
    expect_equal(stats::var(draws), ref$var, tolerance = 0.05)
  }
})


test_that("zero-truncated families reject a zero response", {
  for (fam in c("truncated_poisson", "truncated_neg_binomial_2")) {
    expect_error(.validate_family_counts(fam, c(0, 1, 2)), "zero-truncated")
    expect_silent(.validate_family_counts(fam, c(1, 2, 3)))
  }
  # The untruncated sibling still admits zeros.
  expect_silent(.validate_family_counts("neg_binomial_1", c(0, 1, 2)))
})


test_that("tulpa() refuses families the compiled kernels do not carry", {
  # The C++ family dispatch falls through to Poisson for an unrecognized name,
  # so an ungated family would be silently fit as a Poisson rather than error.
  d <- data.frame(y = c(1, 2, 3, 4), x = c(1, 2, 3, 4))
  for (fam in c("neg_binomial_1", "truncated_poisson",
                "truncated_neg_binomial_2")) {
    expect_error(tulpa(y ~ x, data = d, family = fam),
                 "not yet wired into the compiled kernels")
  }
})
