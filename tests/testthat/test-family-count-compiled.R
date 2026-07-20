# The compiled kernels for neg_binomial_1 and the two zero-truncated families.
#
# test-family-count-extensions.R validates the family math against the R
# registry; this file validates that the C++ dispatch computes the SAME math
# and that fits through it recover known truth. Both halves are needed: a
# kernel can agree with the registry pointwise and still be wired into the
# fitter wrongly, and it can recover parameters while quietly disagreeing with
# the reference on curvature (which surfaces as wrong standard errors, not
# wrong estimates).

count_fams <- c("neg_binomial_1", "truncated_poisson",
                "truncated_neg_binomial_2")

fam_phi <- function(fam) if (fam == "truncated_poisson") 1 else 3

sim_count <- function(fam, n, beta, phi, seed) {
  set.seed(seed)
  x  <- stats::rnorm(n)
  mu <- exp(beta[1] + beta[2] * x)
  y <- switch(fam,
    neg_binomial_1 = stats::rnbinom(n, size = mu / phi, mu = mu),
    # Inverse-CDF draws from the zero-truncated laws: sampling the uniform
    # above P(Y = 0) conditions on Y > 0 exactly, with no rejection loop.
    truncated_poisson =
      stats::qpois(stats::runif(n, exp(-mu), 1), mu),
    truncated_neg_binomial_2 =
      stats::qnbinom(stats::runif(n, stats::dnbinom(0, size = phi, mu = mu), 1),
                     size = phi, mu = mu))
  data.frame(y = y, x = x)
}


test_that("the compiled kernels reproduce the R registry pointwise", {
  for (fam in count_fams) {
    phi <- fam_phi(fam)
    eta <- seq(-0.8, 1.8, length.out = 9)
    y   <- if (fam == "neg_binomial_1") c(0, 1, 2, 3, 5, 8, 0, 2, 4)
           else                          c(1, 2, 3, 4, 6, 9, 1, 3, 5)

    ct <- t(vapply(seq_along(eta), function(i)
      cpp_family_terms(y[i], 1L, eta[i], fam, phi), numeric(3)))
    co <- t(vapply(seq_along(eta), function(i)
      cpp_family_obs_terms(y[i], 1L, eta[i], fam, phi), numeric(2)))

    expect_equal(unname(ct[, "log_lik"]),
                 family_loglik(eta, y, fam, NULL, phi),
                 tolerance = 1e-12, info = fam)
    expect_equal(unname(ct[, "grad"]),
                 family_score_eta(eta, y, fam, NULL, phi),
                 tolerance = 1e-12, info = fam)
    # The Newton working weight is the registry's expected form.
    expect_equal(unname(ct[, "neg_hess"]),
                 rep_len(family_weight(eta, fam, NULL, phi), length(eta)),
                 tolerance = 1e-12, info = fam)
    # The observed curvature is a separate accessor and must match the
    # registry's obs_weight, which for these families differs from the above.
    expect_equal(unname(co[, "neg_hess"]),
                 .family_obs_weight(eta, y, fam, NULL, phi),
                 tolerance = 1e-10, info = fam)
  }
})


test_that("the AD density agrees with the double density and its own score", {
  # The sampler backends differentiate builtin_family_ll_ad.h, which expresses
  # each density separately from log_lik_for_family. Evaluating it at fwd::Dual
  # checks both at once: the value against the independent double
  # implementation, and the AD derivative against the analytic score. Without
  # this a formula typo in the AD branch would only surface as a slightly-off
  # posterior that the sampler-vs-Laplace tolerance could absorb.
  for (fam in count_fams) {
    phi <- fam_phi(fam)
    eta <- seq(-0.8, 1.8, length.out = 9)
    y   <- if (fam == "neg_binomial_1") c(0, 1, 2, 3, 5, 8, 0, 2, 4)
           else                          c(1, 2, 3, 4, 6, 9, 1, 3, 5)

    ad <- t(vapply(seq_along(eta), function(i)
      cpp_family_ad_terms(y[i], 1L, eta[i], fam, phi), numeric(2)))

    expect_equal(unname(ad[, "log_lik"]),
                 family_loglik(eta, y, fam, NULL, phi),
                 tolerance = 1e-10, info = fam)
    expect_equal(unname(ad[, "grad"]),
                 family_score_eta(eta, y, fam, NULL, phi),
                 tolerance = 1e-10, info = fam)
  }
})


test_that("the Newton working weight is positive where the observed one is not", {
  # The reason the two accessors exist rather than one. neg_binomial_1's
  # observed curvature turns negative once the realized y sits well above the
  # mean, which would break the PD Hessian Newton needs; the registered working
  # weight mu / (1 + phi) never does. Newton therefore takes the working
  # weight, and only the zero-inflation mixture -- which differentiates the
  # density rather than taking an expectation -- takes the observed one.
  eta <- seq(-1, 2, length.out = 10)
  obs <- vapply(eta, function(e)
    cpp_family_obs_terms(25, 1L, e, "neg_binomial_1", 3)[["neg_hess"]],
    numeric(1))
  wrk <- vapply(eta, function(e)
    cpp_family_terms(25, 1L, e, "neg_binomial_1", 3)[["neg_hess"]], numeric(1))

  expect_true(all(wrk > 0))
  expect_true(any(obs < 0))
})


test_that("the observed curvature the ZI mixture reads is positive", {
  # The mixture only ever evaluates the observed curvature at y = 0, so the
  # negativity above never reaches it. For neg_binomial_1 that value is
  # r log1p(phi) with r = mu / phi, which is positive for every eta -- this is
  # what makes the family safe to expose through ziformula despite its
  # observed curvature changing sign elsewhere.
  eta <- seq(-2, 3, length.out = 15)
  for (fam in count_fams) {
    phi <- fam_phi(fam)
    obs <- vapply(eta, function(e)
      cpp_family_obs_terms(0, 1L, e, fam, phi)[["neg_hess"]], numeric(1))
    expect_true(all(obs > 0), info = fam)
  }
  # And the closed form it should equal for neg_binomial_1.
  expect_equal(
    vapply(eta, function(e)
      cpp_family_obs_terms(0, 1L, e, "neg_binomial_1", 3)[["neg_hess"]],
      numeric(1)),
    exp(eta) / 3 * log1p(3), tolerance = 1e-10)
})


test_that("the compiled path recovers each family's generating parameters", {
  specs <- list(
    list(fam = "neg_binomial_1",           beta = c(1.4, 0.5), phi = 2),
    list(fam = "truncated_poisson",        beta = c(0.7, 0.4), phi = 1),
    list(fam = "truncated_neg_binomial_2", beta = c(1.0, 0.5), phi = 4))

  for (s in specs) {
    est <- t(vapply(1:8, function(k) {
      d <- sim_count(s$fam, 4000, s$beta, s$phi, 300 + k)
      coef(tulpa(y ~ x, data = d, family = s$fam, phi = s$phi,
                 mode = "laplace"))
    }, numeric(2)))
    expect_equal(mean(est[, 1]), s$beta[1], tolerance = 0.04, info = s$fam)
    expect_equal(mean(est[, 2]), s$beta[2], tolerance = 0.04, info = s$fam)
  }
})


test_that("a truncated fit is not the untruncated fit", {
  # Zero-truncated data has no zeros, so a Poisson fit to it overestimates the
  # mean: the truncation term is exactly what corrects for that. Without this
  # the recovery test above would also pass for a kernel that silently dropped
  # the truncation on data whose counts happen to be large.
  # The gap narrows as the mean grows (fewer zeros are being conditioned away),
  # so this uses a small mean where the correction is unmistakable: at
  # exp(0.2) ~ 1.2 the untruncated fit overshoots by ~0.35.
  d <- sim_count("truncated_poisson", 4000, c(0.2, 0.3), 1, seed = 91)
  trunc <- coef(tulpa(y ~ x, data = d, family = "truncated_poisson",
                      mode = "laplace"))
  plain <- coef(tulpa(y ~ x, data = d, family = "poisson", mode = "laplace"))

  # Absolute, not the relative tolerance expect_equal() applies by default.
  expect_lt(abs(unname(trunc[["(Intercept)"]]) - 0.2), 0.06)
  expect_gt(plain[["(Intercept)"]], trunc[["(Intercept)"]] + 0.2)
})


test_that("a hurdle model factorizes into its two independent fits", {
  # The strongest available check on the hurdle path, and independent of the
  # mixture code: the hurdle log-likelihood separates exactly into a Bernoulli
  # for zero-vs-positive (carrying only the zi coefficients) and a
  # zero-truncated count model on the positives (carrying only the count
  # coefficients). Fitting the two halves separately must therefore reproduce
  # the joint fit. A mixture that leaked information between the two blocks --
  # a wrong cross term, a mis-shared random effect -- would break this while
  # still recovering truth roughly.
  set.seed(11)
  n  <- 3000
  x  <- stats::rnorm(n)
  z  <- stats::rnorm(n)
  mu <- exp(0.9 + 0.4 * x)
  yc <- stats::qpois(stats::runif(n, exp(-mu), 1), mu)
  y  <- ifelse(stats::runif(n) < stats::plogis(-0.6 + 0.5 * z), 0, yc)
  d  <- data.frame(y = y, x = x, z = z)

  joint <- coef(tulpa(y ~ x, data = d, family = "truncated_poisson",
                      ziformula = ~ z, mode = "laplace"))

  pos   <- d[d$y > 0, ]
  count <- coef(tulpa(y ~ x, data = pos, family = "truncated_poisson",
                      mode = "laplace"))
  # The zero component models P(structural zero), so the response is the
  # zero indicator on the same orientation as the mixture's pi.
  #
  # The identity is on the likelihood, so it transfers to the MAP only when the
  # two sides carry the same prior. The joint fit gives its zi block the ZI
  # prior (`.ZI_PRIOR_SD_DEFAULT`), while a standalone binomial gives its
  # coefficients the weak built-in fixed-effect prior, so the standalone side
  # is matched to the joint one here. Unmatched, the two agree only to ~2e-4 --
  # a residual that is entirely the prior gap, and that a loose tolerance would
  # hide.
  d$is0 <- as.numeric(d$y == 0)
  zero  <- coef(tulpa(is0 ~ z, data = d, family = "binomial",
                      mode = "laplace",
                      beta_prior = list(mean = 0, sd = .ZI_PRIOR_SD_DEFAULT)))

  # Prior-matched, the factorization is exact to solver precision, so this is
  # pinned nine orders tighter than the mixture's recovery tolerance. A leaked
  # cross term would have to be smaller than the Newton tolerance to survive.
  expect_equal(unname(joint[["(Intercept)"]]), unname(count[["(Intercept)"]]),
               tolerance = 1e-4)
  expect_equal(unname(joint[["x"]]), unname(count[["x"]]), tolerance = 1e-4)
  expect_equal(unname(joint[["zi_(Intercept)"]]),
               unname(zero[["(Intercept)"]]), tolerance = 1e-9)
  expect_equal(unname(joint[["zi_z"]]), unname(zero[["z"]]), tolerance = 1e-9)
})


test_that("the hurdle factorization is exact under any matched ZI prior", {
  # The companion to the test above: the identity is a property of the
  # likelihood, not of the particular prior scale, so it must hold equally at a
  # prior far from the default. This is what separates "the priors happen to
  # agree" from "the two blocks are genuinely independent".
  set.seed(11)
  n  <- 3000
  x  <- stats::rnorm(n)
  z  <- stats::rnorm(n)
  mu <- exp(0.9 + 0.4 * x)
  yc <- stats::qpois(stats::runif(n, exp(-mu), 1), mu)
  y  <- ifelse(stats::runif(n) < stats::plogis(-0.6 + 0.5 * z), 0, yc)
  d  <- data.frame(y = y, x = x, z = z, is0 = as.numeric(y == 0))

  wide <- 100
  joint <- coef(tulpa(y ~ x, data = d, family = "truncated_poisson",
                      ziformula = ~ z, mode = "laplace",
                      zi_prior = list(sd = wide)))
  zero  <- coef(tulpa(is0 ~ z, data = d, family = "binomial",
                      mode = "laplace",
                      beta_prior = list(mean = 0, sd = wide)))

  expect_equal(unname(joint[["zi_(Intercept)"]]),
               unname(zero[["(Intercept)"]]), tolerance = 1e-9)
  expect_equal(unname(joint[["zi_z"]]), unname(zero[["z"]]), tolerance = 1e-9)
})


test_that("a hurdle model recovers its generating parameters", {
  est <- t(vapply(1:8, function(k) {
    set.seed(700 + k)
    n  <- 3000
    x  <- stats::rnorm(n)
    mu <- exp(0.9 + 0.4 * x)
    yc <- stats::qpois(stats::runif(n, exp(-mu), 1), mu)
    y  <- ifelse(stats::runif(n) < stats::plogis(-0.6), 0, yc)
    coef(tulpa(y ~ x, data = data.frame(y = y, x = x),
               family = "truncated_poisson", ziformula = ~ 1,
               mode = "laplace"))
  }, numeric(3)))

  truth <- c(0.9, 0.4, -0.6)
  for (j in 1:3) expect_equal(mean(est[, j]), truth[j], tolerance = 0.05)
})


test_that("the sampler path carries the new families through its AD density", {
  # These families now set the AD callbacks (builtin_family_has_ad), so the
  # samplers differentiate the exact density instead of falling back to the
  # numerical gradient. Agreement with Laplace is the check that the AD branch
  # computes the same log-density the double path does.
  skip_on_cran()
  d <- sim_count("truncated_poisson", 2000, c(0.8, 0.35), 1, seed = 55)

  lap <- coef(tulpa(y ~ x, data = d, family = "truncated_poisson",
                    mode = "laplace"))
  smp <- coef(tulpa(y ~ x, data = d, family = "truncated_poisson",
                    mode = "hmc",
                    control = list(n_iter = 700, n_warmup = 350,
                                   n_chains = 2, seed = 4)))

  expect_equal(unname(smp[["(Intercept)"]]), unname(lap[["(Intercept)"]]),
               tolerance = 0.06)
  expect_equal(unname(smp[["x"]]), unname(lap[["x"]]), tolerance = 0.06)
})


test_that("an unregistered family errors rather than fitting a Poisson", {
  # The dispatch used to fall through to the Poisson branch, so a family known
  # to R but not to C++ fitted silently and wrongly. .R_ONLY_FAMILIES is now
  # empty because every shipped family is wired; this pins the backstop that
  # made emptying it safe.
  expect_length(.R_ONLY_FAMILIES, 0L)
  expect_error(cpp_family_terms(1, 1L, 0.5, "not_a_real_family", 1),
               "no compiled implementation")
})
