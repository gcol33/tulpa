# Truncated-family curvature under an ESTIMATED random-effect variance.
#
# The Laplace marginal is
#
#   log p(y | theta) = log p(y, x* | theta) - 0.5 log|H| + const,
#
# and the H that makes that an approximation to the integral is the observed
# negative Hessian of the joint at the mode. The mode itself does not depend on
# which curvature the Newton step uses -- the score is exact either way -- so a
# working weight that is not the observed curvature is invisible in beta and in
# any fit that holds the variance components fixed. It enters only through
# log|H|, i.e. only when something optimizes or integrates over theta.
#
# Where the two forms coincide, nothing is at stake:
#
#   * truncated_poisson: the y-term of the score is linear in eta, so the
#     curvature carries no y at all and observed = expected identically.
#   * neg_binomial_2: grad_hess_for_family() already returns the observed
#     form (neg_hess_log_lik_negbin, which carries y).
#
# truncated_neg_binomial_2 is the one family where they differ and the
# registered Newton weight is the expected form (laplace_family_link.h:375
# returns TruncationTerm::e_weight = Var(Y | Y > 0), y-free). These tests fit
# it through tulpa_eb(), where the outer Brent maximization over log(sigma)
# sees log|H| directly.

# --- simulation helpers ------------------------------------------------------

# Zero-truncated NB2 by rejection. Exact for the truncated law, and independent
# of the registry's own sampler so a fault in .rtrunc_count() cannot hide here.
rtnb2 <- function(n, mu, phi) {
  out <- integer(n)
  todo <- seq_len(n)
  while (length(todo)) {
    draw <- stats::rnbinom(length(todo), size = phi, mu = mu[todo])
    keep <- draw > 0L
    out[todo[keep]] <- draw[keep]
    todo <- todo[!keep]
  }
  out
}

sim_re_count <- function(seed, family, G = 40L, per = 10L, sigma = 0.7,
                         beta = c(0.4, 0.6), phi = 2.0) {
  set.seed(seed)
  n <- G * per
  grp <- rep(seq_len(G), each = per)
  x <- stats::rnorm(n)
  b <- stats::rnorm(G, 0, sigma)
  mu <- exp(beta[1L] + beta[2L] * x + b[grp])

  y <- switch(family,
    neg_binomial_2           = stats::rnbinom(n, size = phi, mu = mu),
    truncated_neg_binomial_2 = rtnb2(n, mu, phi),
    truncated_poisson        = {
      out <- integer(n); todo <- seq_len(n)
      while (length(todo)) {
        draw <- stats::rpois(length(todo), mu[todo])
        keep <- draw > 0L
        out[todo[keep]] <- draw[keep]
        todo <- todo[!keep]
      }
      out
    },
    stop("unhandled family in sim_re_count()")
  )

  list(y = y, X = cbind(1, x), grp = grp, G = G,
       sigma = sigma, beta = beta, phi = phi)
}

scalar_term <- function(d) list(idx = d$grp, n_groups = d$G, n_coefs = 1L)

eb_sigma <- function(d, family) {
  tulpa_eb(d$y, NULL, d$X, scalar_term(d), family = family,
           phi = d$phi)$map$sigma
}

# Mean estimate over seeds. The PC prior pulls sigma down a little, so every
# assertion below is a bias bound on this mean rather than a per-seed bound.
mean_sigma <- function(seeds, family, ...) {
  est <- vapply(seeds, function(s) {
    eb_sigma(sim_re_count(s, family, ...), family)
  }, numeric(1))
  expect_true(all(is.finite(est)))
  expect_true(all(est > 0.2 & est < 1.5))   # no boundary collapse, no runaway
  mean(est)
}


# --- 1. Which families actually have a Fisher-vs-observed distinction --------

test_that("truncated_poisson curvature carries no y, so the two forms agree", {
  # Pins the registry's claim that observed = expected for this family, which
  # is what licenses it as a control in the recovery tests below.
  for (eta in c(-1.5, -0.5, 0.3, 1.2, 2.0)) {
    for (y in c(1, 2, 5, 20)) {
      w_newton <- cpp_family_terms(y, 1L, eta, "truncated_poisson", 1.0)[["neg_hess"]]
      w_obs <- cpp_family_obs_terms(y, 1L, eta, "truncated_poisson", 1.0)[["neg_hess"]]
      expect_equal(w_newton, w_obs, tolerance = 1e-12,
                   info = sprintf("eta = %g, y = %g", eta, y))
    }
  }
})

test_that("neg_binomial_2 registers the observed curvature for Newton", {
  # The untruncated sibling carries y in its Newton weight, so it is a control
  # for the estimator rather than for the curvature choice.
  phi <- 2.0
  for (eta in c(-0.5, 0.4, 1.3)) {
    w1 <- cpp_family_terms(1, 1L, eta, "neg_binomial_2", phi)[["neg_hess"]]
    w9 <- cpp_family_terms(9, 1L, eta, "neg_binomial_2", phi)[["neg_hess"]]
    expect_false(isTRUE(all.equal(w1, w9, tolerance = 1e-8)))
    expect_equal(w1, cpp_family_obs_terms(1, 1L, eta, "neg_binomial_2", phi)[["neg_hess"]],
                 tolerance = 1e-12)
  }
})

test_that("truncated_neg_binomial_2 Newton weight is y-free and differs from observed", {
  # The gap is the quantity the recovery tests are sensitive to. It is not a
  # rounding difference: it grows linearly in y, since the observed form carries
  # (y + phi) * phi * mu / (mu + phi)^2 where the expected form carries the
  # y-free Var(Y | Y > 0).
  phi <- 2.0
  eta <- 0.3
  ys <- c(1, 2, 5, 10, 25)

  w_newton <- vapply(ys, function(y)
    cpp_family_terms(y, 1L, eta, "truncated_neg_binomial_2", phi)[["neg_hess"]],
    numeric(1))
  w_obs <- vapply(ys, function(y)
    cpp_family_obs_terms(y, 1L, eta, "truncated_neg_binomial_2", phi)[["neg_hess"]],
    numeric(1))

  # y-free: every y gives the same Newton weight.
  expect_equal(diff(range(w_newton)), 0, tolerance = 1e-12)
  # The observed form moves with y and is monotone in it.
  expect_true(all(diff(w_obs) > 0))
  # And the two disagree materially somewhere in the support that the fits
  # below actually sample.
  expect_gt(max(abs(w_obs - w_newton)) / w_newton[1L], 0.5)
})


# --- 2. Controls: the estimator recovers sigma where the choice cannot bite --

test_that("EB recovers sigma for untruncated neg_binomial_2", {
  skip_on_cran()
  expect_lt(abs(mean_sigma(1:8, "neg_binomial_2", sigma = 0.7, phi = 2.0) - 0.7), 0.1)
})

test_that("EB recovers sigma for truncated_poisson", {
  skip_on_cran()
  # Same truncated Laplace code path as the target test; observed = expected
  # here, so this isolates the curvature choice from the truncation machinery.
  expect_lt(abs(mean_sigma(1:8, "truncated_poisson", sigma = 0.7) - 0.7), 0.1)
})


# --- 3. Target: sigma under truncated_neg_binomial_2 -------------------------

test_that("EB recovers sigma for truncated_neg_binomial_2 under mild truncation", {
  skip_on_cran()
  # beta0 = 1.6 puts the mean near 5, so P(Y = 0) is small and the truncation
  # correction -- and with it any discrepancy between the two curvatures -- is
  # a small part of log|H|.
  got <- mean_sigma(1:8, "truncated_neg_binomial_2",
                    sigma = 0.7, beta = c(1.6, 0.6), phi = 2.0)
  expect_lt(abs(got - 0.7), 0.1)
})

test_that("EB recovers sigma for truncated_neg_binomial_2 under severe truncation", {
  skip_on_cran()
  # beta0 = -0.7 puts the mean near 0.5, where P(Y = 0) is roughly half the
  # untruncated mass. The truncation term then dominates the curvature, so a
  # Newton weight that is not the observed Hessian is at its most visible in
  # log|H| and therefore in the outer maximization over log(sigma).
  #
  # This is the configuration no other test in the suite reaches: every other
  # truncated fit is fixed-effects-only or holds sigma_re conditioned, and
  # neither exercises log|H| as a function of theta.
  got <- mean_sigma(1:8, "truncated_neg_binomial_2",
                    sigma = 0.7, beta = c(-0.7, 0.6), phi = 2.0)
  expect_lt(abs(got - 0.7), 0.1)
})
