# The R family registry (R/family_loglik.R) against the compiled family
# dispatch (src/laplace_family_link.h), over EVERY entry of family_names().
#
# Both are full implementations of the same per-family (loglik, score,
# curvature) math. mala() / pathfinder() / imh_laplace() build their target
# from the R registry through build_glmm_logpost(), and glmm_weights() rebuilds
# the Laplace Hessian from it for the post-fit marginal SEs on GP / NNGP /
# HSGP / SPDE fits, while every other backend runs the compiled kernels -- so a
# divergence between the two makes two modes of the same engine fit different
# models for the identical formula. That is the shape of the 0.0.73 regression
# (a phi SD-vs-variance mismatch between compiled kernels), and nothing in C++
# ties the R side to it.
#
# test-family-link.R covers 4 non-canonical-link families and
# test-family-count-compiled.R 3 count families; this covers the registry
# itself, canonical entries included (gcol33/tulpa#824).

# One valid (y, eta, phi, n_trials) case per registered family. `y` stays
# inside each family's support (counts integer, the truncated pair >= 1, the
# positive families > 0, beta in (0, 1)) and spans a response well below and
# well above the mean, which is where an expected-vs-observed curvature
# difference is visible at all.
.registry_cases <- list(
  binomial                 = list(n = 10L, y = c(0, 3, 10),       phi = 1,   phi2 = NULL, eta = c(-1.2, 0, 0.7)),
  poisson                  = list(n = 1L,  y = c(0, 2, 7),        phi = 1,   phi2 = NULL, eta = c(-0.5, 0.4, 1.3)),
  neg_binomial_2           = list(n = 1L,  y = c(0, 2, 9),        phi = 1.7, phi2 = NULL, eta = c(-0.5, 0.4, 1.3)),
  neg_binomial_1           = list(n = 1L,  y = c(0, 2, 9),        phi = 1.7, phi2 = NULL, eta = c(-0.5, 0.4, 1.3)),
  truncated_poisson        = list(n = 1L,  y = c(1, 3, 8),        phi = 1,   phi2 = NULL, eta = c(-0.3, 0.6, 1.2)),
  truncated_neg_binomial_2 = list(n = 1L,  y = c(1, 3, 8),        phi = 1.4, phi2 = NULL, eta = c(-0.3, 0.6, 1.2)),
  lognormal                = list(n = 1L,  y = c(0.4, 1.6, 5),    phi = 1.3, phi2 = NULL, eta = c(-0.4, 0.3, 1.1)),
  gaussian                 = list(n = 1L,  y = c(-1.1, 0.5, 2.3), phi = 1.3, phi2 = NULL, eta = c(-0.4, 0.3, 1.1)),
  beta                     = list(n = 1L,  y = c(0.1, 0.5, 0.85), phi = 4.2, phi2 = NULL, eta = c(-0.9, 0.2, 1.0)),
  gamma                    = list(n = 1L,  y = c(0.3, 1.5, 4.1),  phi = 2.2, phi2 = NULL, eta = c(-0.4, 0.3, 1.1)),
  inverse_gaussian         = list(n = 1L,  y = c(0.3, 1.5, 4.1),  phi = 1.1, phi2 = NULL, eta = c(-0.4, 0.3, 1.1)),
  beta_binomial            = list(n = 12L, y = c(0, 5, 12),       phi = 3.0, phi2 = NULL, eta = c(-0.9, 0.2, 1.0)),
  tweedie                  = list(n = 1L,  y = c(0, 1.7, 5.5),    phi = 1.6, phi2 = 1.5,  eta = c(-0.4, 0.3, 1.1)),
  t                        = list(n = 1L,  y = c(-1.1, 0.5, 2.3), phi = 1.2, phi2 = 6,    eta = c(-0.4, 0.3, 1.1))
)

test_that("every registered family has a case in the cross-check", {
  # A family added to .FAMILY_OPS without a case here would be silently
  # uncovered, which is the situation this file exists to end.
  expect_setequal(family_names(), names(.registry_cases))
})

test_that("the R family registry agrees with the compiled kernels", {
  skip_on_cran()
  for (fam in family_names()) {
    cs <- .registry_cases[[fam]]
    # R-side phi is the residual VARIANCE for gaussian / lognormal and the SD
    # is what the kernel takes; crossing that seam is what .phi_to_kernel()
    # is for, and crossing it wrongly is gcol33/tulpa#332 / #661.
    phi_cpp <- tulpa:::.phi_to_kernel(fam, cs$phi)
    phi2    <- if (is.null(cs$phi2)) NA_real_ else cs$phi2
    for (y in cs$y) {
      for (e in cs$eta) {
        cpp <- cpp_family_terms(y, cs$n, e, fam, phi_cpp, phi2)
        info <- sprintf("%s at y = %g, eta = %g", fam, y, e)
        expect_equal(family_loglik(e, y, fam, cs$n, cs$phi, cs$phi2),
                     unname(cpp[["log_lik"]]), tolerance = 1e-10, info = info)
        expect_equal(family_score_eta(e, y, fam, cs$n, cs$phi, cs$phi2),
                     unname(cpp[["grad"]]), tolerance = 1e-10, info = info)
      }
    }
  }
})

test_that("the R working weight is the one the compiled Hessian carries", {
  skip_on_cran()
  # Two curvatures exist per family and the compiled dispatch picks one:
  # cpp_family_working_weight_is_observed() reports which. Where it says
  # observed, the registry's y-free `weight` is a DIFFERENT function and only
  # `obs_weight` reproduces the Hessian the engine used -- neg_binomial_2 is
  # the family where that bites (gcol33/tulpa#824).
  for (fam in family_names()) {
    cs   <- .registry_cases[[fam]]
    phi_cpp <- tulpa:::.phi_to_kernel(fam, cs$phi)
    phi2    <- if (is.null(cs$phi2)) NA_real_ else cs$phi2
    for (y in cs$y) {
      for (e in cs$eta) {
        info <- sprintf("%s at y = %g, eta = %g", fam, y, e)
        # Where the registry carries its own closed form for the observed
        # curvature, that ORACLE must match the compiled dispatch
        # `.family_obs_weight()` now routes through for every family.
        ops <- tulpa:::.family_ops(fam)
        if (!is.null(ops$obs_weight)) {
          ref <- if (is.null(cs$phi2)) ops$obs_weight(e, y, cs$n, cs$phi)
                 else ops$obs_weight(e, y, cs$n, cs$phi, cs$phi2)
          expect_equal(
            ref, tulpa:::.family_obs_weight(e, y, fam, cs$n, cs$phi, cs$phi2),
            tolerance = 1e-10, info = paste("oracle,", info))
        }
        expect_equal(
          tulpa:::.family_obs_weight(e, y, fam, cs$n, cs$phi, cs$phi2),
          unname(cpp_family_obs_terms(y, cs$n, e, fam, phi_cpp, phi2)[["neg_hess"]]),
          tolerance = 1e-10, info = paste("observed,", info))
        # ... and glmm_weights() returns whichever form the compiled working
        # weight is, so the Hessian it rebuilds is the fit's own.
        expect_equal(
          tulpa:::glmm_weights(e, fam, cs$n, cs$phi, cs$phi2, y = y),
          unname(cpp_family_terms(y, cs$n, e, fam, phi_cpp, phi2)[["neg_hess"]]),
          tolerance = 1e-10, info = paste("working,", info))
      }
    }
  }
})

test_that("the observed curvature is not the expected weight where they differ", {
  # The substitution .family_obs_weight() used to make for a family with no
  # registered closed form. It is exact only where the response enters the
  # log-likelihood linearly in eta; elsewhere it is a different function, and
  # for beta_binomial -- which is in .ZI_FAMILIES, so the zero-inflation
  # mixture reaches it -- the two even disagree in SIGN (gcol33/tulpa#824).
  expect_equal(tulpa:::.family_obs_weight(-0.9, 12, "beta_binomial",
                                          n_trials = 12L, phi = 3.0),
               unname(cpp_family_obs_terms(12, 12L, -0.9, "beta_binomial",
                                           3.0)[["neg_hess"]]),
               tolerance = 1e-10)
  expect_lt(tulpa:::.family_obs_weight(-0.9, 12, "beta_binomial",
                                       n_trials = 12L, phi = 3.0), 0)
  expect_gt(family_weight(-0.9, "beta_binomial", n_trials = 12L, phi = 3.0), 0)

  # Where they DO coincide identically -- a curvature carrying no y -- the two
  # stay equal, so the change is confined to the families it had to reach.
  for (fam in c("poisson", "binomial", "truncated_poisson")) {
    cs <- .registry_cases[[fam]]
    for (y in cs$y) for (e in cs$eta) {
      expect_equal(tulpa:::.family_obs_weight(e, y, fam, cs$n, cs$phi),
                   family_weight(e, fam, cs$n, cs$phi), tolerance = 1e-10,
                   info = sprintf("%s at y = %g, eta = %g", fam, y, e))
    }
  }
})

test_that("the second dispersion crosses to the compiled dispatch intact", {
  # tweedie has no default power, so a missing phi2 must still refuse rather
  # than reach the kernel as NA; `t` has one on both sides, and the two
  # defaults have to BE the same number or the compiled route quietly fits a
  # different df from the registry.
  expect_error(tulpa:::.family_obs_weight(0.3, 1.7, "tweedie"), "phi2")
  expect_error(family_weight(0.3, "tweedie"), "phi2")
  for (e in c(-0.4, 0.3, 1.1)) {
    for (y in c(-1.1, 0.5, 2.3)) {
      expect_equal(
        tulpa:::.family_obs_weight(e, y, "t", phi = 1.2),
        unname(cpp_family_obs_terms(y, 1L, e, "t", 1.2,
                                    tulpa:::.STUDENT_T_DF)[["neg_hess"]]),
        tolerance = 1e-12,
        info = sprintf("t default df at y = %g, eta = %g", y, e))
    }
  }
})

test_that(".family_obs_weight recycles to the longest argument", {
  eta <- c(-0.4, 0.3, 1.1)
  expect_length(tulpa:::.family_obs_weight(eta, 2, "poisson"), 3L)
  expect_length(tulpa:::.family_obs_weight(0.3, c(1, 2, 5), "poisson"), 3L)
  expect_equal(tulpa:::.family_obs_weight(eta, 2, "poisson"),
               vapply(eta, function(e)
                 tulpa:::.family_obs_weight(e, 2, "poisson"), numeric(1)))
})

test_that("neg_binomial_2 is the family whose Laplace weight needs the response", {
  # Pins the characterisation rather than the family list: a family qualifies
  # when the compiled working weight IS the observed curvature AND the
  # registry carries a separate (y-carrying) observed form.
  needs <- Filter(tulpa:::.glmm_weight_needs_y, family_names())
  expect_equal(needs, "neg_binomial_2")

  # Without the response it refuses rather than returning the expected form,
  # which differs by tens of percent at a y away from the mean: at phi = 1.7,
  # eta = 1.3, y = 9 the expected weight is 1.162 and the observed 2.315.
  expect_error(tulpa:::glmm_weights(1.3, "neg_binomial_2", phi = 1.7),
               "pass `y`")
  mu <- exp(1.3)
  expect_equal(tulpa:::glmm_weights(1.3, "neg_binomial_2", phi = 1.7, y = 9),
               (9 + 1.7) * 1.7 * mu / (mu + 1.7)^2, tolerance = 1e-10)
  expect_equal(family_weight(1.3, "neg_binomial_2", phi = 1.7),
               mu * 1.7 / (mu + 1.7), tolerance = 1e-10)
})
