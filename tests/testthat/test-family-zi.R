# Zero-inflation as a composition over the base family registry. As in
# test-family-count-extensions.R, normalization is the load-bearing check:
# score and curvature are both invariant to a constant offset in the
# log-density, so only summing exp(loglik) over the support catches a wrong
# mixture constant.
#
# The hurdle degeneracy is tested explicitly. ZI over a zero-truncated base is
# the hurdle model, and the general mixture code must reproduce the two-part
# closed form without any truncated-specific branch.

zi_support_sum <- function(family, eta, z, phi, ymax = 20000) {
  yy <- 0:ymax
  sum(exp(zi_loglik(rep(eta, length(yy)), z, yy, family, phi = phi)))
}

zi_support_moments <- function(family, eta, z, phi, ymax = 20000) {
  yy <- 0:ymax
  pr <- exp(zi_loglik(rep(eta, length(yy)), z, yy, family, phi = phi))
  m  <- sum(pr * yy)
  list(mean = m, var = sum(pr * (yy - m)^2))
}

grid_eta <- c(-1, 0, 1.5)
grid_z   <- c(-1.5, 0, 0.8)
grid_phi <- c(0.5, 2, 8)

untruncated <- c("poisson", "neg_binomial_2", "neg_binomial_1")
truncated   <- c("truncated_poisson", "truncated_neg_binomial_2")


test_that("observed curvature is the exact derivative of the score", {
  # Every family registering obs_weight must reproduce -d score / d eta at the
  # realized y, not merely in expectation.
  h <- 1e-6
  for (fam in c("neg_binomial_2", "neg_binomial_1",
                "truncated_neg_binomial_2")) {
    ymin <- if (fam %in% truncated) 1 else 0
    for (eta in grid_eta) {
      for (phi in grid_phi) {
        y  <- ymin:12
        fd <- -(family_score_eta(eta + h, y, fam, phi = phi) -
                  family_score_eta(eta - h, y, fam, phi = phi)) / (2 * h)
        expect_equal(.family_obs_weight(rep(eta, length(y)), y, fam,
                                        phi = phi),
                     fd, tolerance = 1e-5)
      }
    }
  }
})


test_that("observed curvature averages to the registered weight for nb2", {
  # neg_binomial_2's registered weight is the exact Fisher information, so
  # averaging the observed curvature over y must return it. neg_binomial_1's
  # does not (its weight is the moment weight) -- pinned in
  # test-family-count-extensions.R -- so it is deliberately excluded here.
  for (eta in grid_eta) {
    for (phi in grid_phi) {
      yy <- 0:60000
      pr <- stats::dnbinom(yy, size = phi, mu = exp(eta))
      avg <- sum(pr * .family_obs_weight(rep(eta, length(yy)), yy,
                                         "neg_binomial_2", phi = phi))
      expect_equal(avg, family_weight(eta, "neg_binomial_2", phi = phi),
                   tolerance = 1e-8)
    }
  }
})


test_that("the zero-inflated mixture is normalized over its support", {
  for (fam in c(untruncated, truncated)) {
    for (eta in grid_eta) {
      for (z in grid_z) {
        for (phi in grid_phi) {
          expect_equal(zi_support_sum(fam, eta, z, phi), 1, tolerance = 1e-9)
        }
      }
    }
  }
})


test_that("zero-inflated scores match finite differences in both components", {
  h <- 1e-6
  for (fam in c(untruncated, truncated)) {
    ymin <- if (fam %in% truncated) 1 else 0
    for (eta in grid_eta) {
      for (z in grid_z) {
        for (phi in grid_phi) {
          y <- c(0, ymin:8)
          g <- zi_score_eta(eta, z, y, fam, phi = phi)
          fd_eta <- (zi_loglik(eta + h, z, y, fam, phi = phi) -
                       zi_loglik(eta - h, z, y, fam, phi = phi)) / (2 * h)
          fd_z   <- (zi_loglik(eta, z + h, y, fam, phi = phi) -
                       zi_loglik(eta, z - h, y, fam, phi = phi)) / (2 * h)
          expect_equal(g[, "count"], fd_eta, tolerance = 1e-5,
                       ignore_attr = TRUE)
          expect_equal(g[, "zi"], fd_z, tolerance = 1e-5, ignore_attr = TRUE)
        }
      }
    }
  }
})


test_that("the zero-inflated negative Hessian matches finite differences", {
  # All three distinct entries of the symmetric 2 x 2 block, including the
  # cross term that couples the count and zi predictors at y = 0.
  h <- 1e-4
  for (fam in c(untruncated, truncated)) {
    ymin <- if (fam %in% truncated) 1 else 0
    for (eta in grid_eta) {
      for (z in grid_z) {
        for (phi in grid_phi) {
          y  <- c(0, ymin:8)
          nh <- zi_neg_hessian(eta, z, y, fam, phi = phi)
          ll <- function(a, b) zi_loglik(a, b, y, fam, phi = phi)

          fd_ee <- -(ll(eta + h, z) - 2 * ll(eta, z) + ll(eta - h, z)) / h^2
          fd_zz <- -(ll(eta, z + h) - 2 * ll(eta, z) + ll(eta, z - h)) / h^2
          fd_ez <- -(ll(eta + h, z + h) - ll(eta + h, z - h) -
                       ll(eta - h, z + h) + ll(eta - h, z - h)) / (4 * h^2)

          expect_equal(nh[, "count"], fd_ee, tolerance = 1e-4,
                       ignore_attr = TRUE)
          expect_equal(nh[, "zi"], fd_zz, tolerance = 1e-4,
                       ignore_attr = TRUE)
          expect_equal(nh[, "cross"], fd_ez, tolerance = 1e-4,
                       ignore_attr = TRUE)
        }
      }
    }
  }
})


test_that("zero-inflation over a truncated base is exactly the hurdle model", {
  # The two-part hurdle closed form, written independently of the mixture code:
  #   y = 0:  log(pi)
  #   y > 0:  log(1 - pi) + loglik_truncated(y)
  # with gradient (0, 1 - pi) at y = 0 and (score_trunc, -pi) at y > 0, and a
  # negative Hessian that is block diagonal everywhere.
  for (fam in truncated) {
    for (eta in grid_eta) {
      for (z in grid_z) {
        for (phi in grid_phi) {
          y  <- 0:8
          pi <- stats::plogis(z)
          is0 <- y == 0

          ref_ll <- ifelse(is0, log(pi),
                           log1p(-pi) +
                             family_loglik(rep(eta, length(y)), y, fam,
                                           phi = phi))
          expect_equal(zi_loglik(eta, z, y, fam, phi = phi), ref_ll,
                       tolerance = 1e-10)

          g <- zi_score_eta(eta, z, y, fam, phi = phi)
          expect_equal(g[, "count"],
                       ifelse(is0, 0,
                              family_score_eta(rep(eta, length(y)), y, fam,
                                               phi = phi)),
                       tolerance = 1e-10, ignore_attr = TRUE)
          expect_equal(g[, "zi"], ifelse(is0, 1 - pi, -pi),
                       tolerance = 1e-10, ignore_attr = TRUE)

          nh <- zi_neg_hessian(eta, z, y, fam, phi = phi)
          expect_equal(nh[, "zi"], rep(pi * (1 - pi), length(y)),
                       tolerance = 1e-10, ignore_attr = TRUE)
          # The hurdle likelihood is additively separable, so the count and zi
          # predictors never couple -- unlike genuine zero inflation.
          expect_equal(nh[, "cross"], rep(0, length(y)),
                       tolerance = 1e-12, ignore_attr = TRUE)
          expect_equal(nh[is0, "count"], 0, tolerance = 1e-12,
                       ignore_attr = TRUE)
        }
      }
    }
  }
})


test_that("genuine zero inflation couples the two predictors at y = 0", {
  # Guards the converse of the hurdle test: with an untruncated base both
  # components can produce a zero, so the cross term must be non-zero. A
  # mixture implementation that always returned a block-diagonal Hessian would
  # pass every hurdle check above and still be wrong here.
  for (fam in untruncated) {
    nh <- zi_neg_hessian(0.5, 0.3, 0, fam, phi = 2)
    expect_true(abs(nh[, "cross"]) > 1e-3)
  }
  # ... and no coupling once the zero is ruled out.
  for (fam in untruncated) {
    nh <- zi_neg_hessian(0.5, 0.3, 3, fam, phi = 2)
    expect_equal(nh[, "cross"], 0, tolerance = 1e-12, ignore_attr = TRUE)
  }
})


test_that("zero-inflated moments match the mixture's exact moments", {
  for (fam in c(untruncated, truncated)) {
    for (eta in grid_eta) {
      for (z in grid_z) {
        for (phi in grid_phi) {
          ref <- zi_support_moments(fam, eta, z, phi)
          expect_equal(zi_response_mean(eta, z, fam, phi = phi), ref$mean,
                       tolerance = 1e-8)
          expect_equal(zi_variance(eta, z, fam, phi = phi), ref$var,
                       tolerance = 1e-8)
        }
      }
    }
  }
})


test_that("the zero-inflated sampler reproduces the mixture's moments", {
  skip_on_cran()
  set.seed(20260720)
  for (fam in c(untruncated, truncated)) {
    eta <- 0.7; z <- 0.2; phi <- 2
    draws <- zi_sample(rep(eta, 4e5), z, fam, phi = phi)
    ref <- zi_support_moments(fam, eta, z, phi)
    expect_gte(min(draws), 0)
    expect_equal(mean(draws), ref$mean, tolerance = 0.02)
    expect_equal(stats::var(draws), ref$var, tolerance = 0.05)
  }
})


test_that("zero inflation is refused for families with no atom at zero", {
  for (fam in c("gaussian", "gamma", "beta", "lognormal")) {
    expect_error(zi_loglik(0, 0, 1, fam), "no probability atom at zero")
  }
  expect_silent(zi_loglik(0, 0, 1, "poisson"))
})
