# Cross-path equality for the per-family (loglik, grad, curvature) triplets
# (gcol33/tulpa#142 A9).
#
# The same per-family math is maintained in parallel across several kernels,
# one per backend: laplace_family_link.h (Laplace/Newton dispatch),
# glmm_oracle.h (the compiled GLMM oracle behind agq_fit / re_aghq / the Gibbs
# sweep), and laplace_likelihoods.cpp (the explicit triplets). Nothing in C++
# ties them together -- they agree by convention only. That structure produced
# the 0.0.73 bug, where the compiled kernels read `phi` as the residual SD while
# R passed the variance, so `mode = "laplace"` and `mode = "mala"` silently fit
# different models at phi != 1.
#
# These tests evaluate every kernel at a shared (y, eta, phi) and pin them
# against each other, so a future edit to one copy that misses a sibling fails
# here instead of silently changing what a backend fits.

test_that("gaussian: the SD and VARIANCE phi conventions line up exactly", {
  # THE 0.0.73 REGRESSION GUARD.
  # laplace_family_link.h and laplace_likelihoods.cpp take phi = residual SD.
  # glmm_oracle.h takes phi = residual VARIANCE. The R layer bridges this
  # (fit_laplace.R sqrt()s the variance; agq.R squares sigma_eps). If either
  # kernel's convention silently flips, these equalities break.
  for (sd in c(0.5, 1.0, 2.0)) {
    for (y in c(-1.3, 0.0, 2.7)) {
      for (eta in c(-0.8, 0.0, 1.5)) {
        link  <- cpp_family_terms(y, 1L, eta, "gaussian", sd)
        expl  <- cpp_test_laplace_gaussian(y, eta, sd)
        orac  <- cpp_glmm_elt_terms("gaussian", eta, y, 1, sd^2)

        info <- paste("sd:", sd, "y:", y, "eta:", eta)
        # SD-convention kernels agree with each other
        expect_equal(unname(link), unname(expl), tolerance = 1e-10, info = info)
        # and with the VARIANCE-convention kernel once phi is squared
        expect_equal(unname(orac), unname(link), tolerance = 1e-10, info = info)

        # and all agree with R's own normal density
        expect_equal(unname(link[["log_lik"]]),
                     stats::dnorm(y, mean = eta, sd = sd, log = TRUE),
                     tolerance = 1e-10, info = info)
      }
    }
  }
})

test_that("gaussian: passing phi on the wrong scale is detectable", {
  # Guards the guard: if the two conventions were accidentally the same, the
  # test above would pass even after a convention flip. At sd != 1 the wrong
  # scale must give a different answer.
  sd <- 2.0
  right <- cpp_glmm_elt_terms("gaussian", 0.5, 1.0, 1, sd^2)   # variance: correct
  wrong <- cpp_glmm_elt_terms("gaussian", 0.5, 1.0, 1, sd)     # SD: the 0.0.73 bug
  expect_false(isTRUE(all.equal(unname(right), unname(wrong))))
})

test_that("poisson: every kernel agrees at a shared eta", {
  for (y in c(0L, 1L, 5L, 17L)) {
    for (eta in c(-1.0, 0.0, 1.2, 2.5)) {
      link <- cpp_family_terms(y, 1L, eta, "poisson", 1.0)
      expl <- cpp_test_laplace_poisson(y, eta)
      orac <- cpp_glmm_elt_terms("poisson", eta, y, 1, 1.0)

      info <- paste("y:", y, "eta:", eta)
      expect_equal(unname(link[["log_lik"]]),  expl$log_lik,     tolerance = 1e-10, info = info)
      expect_equal(unname(link[["grad"]]),     expl$gradient,    tolerance = 1e-10, info = info)
      expect_equal(unname(link[["neg_hess"]]), expl$neg_hessian, tolerance = 1e-10, info = info)
      expect_equal(unname(orac), unname(link), tolerance = 1e-10, info = info)
      expect_equal(unname(link[["log_lik"]]),
                   stats::dpois(y, lambda = exp(eta), log = TRUE),
                   tolerance = 1e-10, info = info)
    }
  }
})

test_that("neg_binomial_2: every kernel agrees, and phi is the size everywhere", {
  # A mu/size vs mu/dispersion split between copies would show up here.
  for (y in c(0L, 3L, 12L)) {
    for (eta in c(-0.5, 0.0, 1.4)) {
      for (size in c(0.7, 2.0, 10.0)) {
        link <- cpp_family_terms(y, 1L, eta, "neg_binomial_2", size)
        expl <- cpp_test_laplace_negbin(y, eta, size)
        orac <- cpp_glmm_elt_terms("neg_binomial_2", eta, y, 1, size)

        info <- paste("y:", y, "eta:", eta, "size:", size)
        expect_equal(unname(link[["log_lik"]]),  expl$log_lik,     tolerance = 1e-10, info = info)
        expect_equal(unname(link[["grad"]]),     expl$gradient,    tolerance = 1e-10, info = info)
        expect_equal(unname(link[["neg_hess"]]), expl$neg_hessian, tolerance = 1e-10, info = info)
        expect_equal(unname(orac), unname(link), tolerance = 1e-10, info = info)
        # phi is the NB2 size: mu = exp(eta), var = mu + mu^2/size
        expect_equal(unname(link[["log_lik"]]),
                     stats::dnbinom(y, size = size, mu = exp(eta), log = TRUE),
                     tolerance = 1e-10, info = info)
      }
    }
  }
})

test_that("binomial: every kernel agrees and keeps the lchoose normalizer", {
  for (n in c(1L, 5L, 20L)) {
    for (y in unique(c(0L, 1L, n %/% 2L, n))) {
      for (eta in c(-2.0, -0.3, 0.0, 1.1)) {
        link <- cpp_family_terms(y, n, eta, "binomial", 1.0)
        expl <- cpp_test_laplace_binomial(y, n, eta)
        orac <- cpp_glmm_elt_terms("binomial", eta, y, n, 1.0)

        info <- paste("n:", n, "y:", y, "eta:", eta)
        expect_equal(unname(link), unname(orac), tolerance = 1e-10, info = info)
        expect_equal(unname(link[["log_lik"]]),  expl$log_lik,     tolerance = 1e-10, info = info)
        expect_equal(unname(link[["grad"]]),     expl$gradient,    tolerance = 1e-10, info = info)
        expect_equal(unname(link[["neg_hess"]]), expl$neg_hessian, tolerance = 1e-10, info = info)
        # A true log-density: agrees with dbinom, which carries lchoose(n, y).
        expect_equal(unname(link[["log_lik"]]),
                     stats::dbinom(y, size = n, prob = stats::plogis(eta), log = TRUE),
                     tolerance = 1e-10, info = info)
      }
    }
  }
})

test_that("the curvature of every family matches a numerical derivative", {
  # Catches a grad/curvature that drifts out of step with its own loglik --
  # the gamma copies disagree on observed vs expected information, so a kernel
  # can be self-inconsistent without any cross-path check noticing.
  cases <- list(
    list(family = "gaussian",       y = 1.2, n = 1L, phi = 1.5),
    list(family = "poisson",        y = 4,   n = 1L, phi = 1.0),
    list(family = "neg_binomial_2", y = 6,   n = 1L, phi = 2.5),
    list(family = "binomial",       y = 3,   n = 10L, phi = 1.0)
  )
  h <- 1e-5
  for (cs in cases) {
    for (eta in c(-0.6, 0.0, 0.9)) {
      ll <- function(e) unname(cpp_family_terms(cs$y, cs$n, e, cs$family, cs$phi)[["log_lik"]])
      got <- cpp_family_terms(cs$y, cs$n, eta, cs$family, cs$phi)
      info <- paste(cs$family, "eta:", eta)

      num_grad <- (ll(eta + h) - ll(eta - h)) / (2 * h)
      num_curv <- -(ll(eta + h) - 2 * ll(eta) + ll(eta - h)) / h^2
      expect_equal(unname(got[["grad"]]), num_grad, tolerance = 1e-5, info = info)
      expect_equal(unname(got[["neg_hess"]]), num_curv, tolerance = 1e-3, info = info)
    }
  }
})
