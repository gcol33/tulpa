# Front-door likelihood-input validation (gcol33/tulpa#104).
#
# The public tulpa() entry accepts dispersion-carrying and count families, so a
# non-positive `phi` or a non-integer `y` would otherwise flow into the kernel
# and produce a NaN / silently-floored (biased) log-likelihood. These checks
# turn each such input error into a clear front-door error.

test_that(".validate_family_phi rejects non-positive phi for dispersion families", {
  for (fam in c("gaussian", "gamma", "neg_binomial_2", "inverse_gaussian", "beta")) {
    expect_error(.validate_family_phi(fam, 0),  "positive", info = fam)
    expect_error(.validate_family_phi(fam, -1), "positive", info = fam)
    expect_error(.validate_family_phi(fam, NA_real_), "positive", info = fam)
    expect_silent(.validate_family_phi(fam, 1.5))
  }
  # `<family>_<link>` codes share the base family's dispersion rule.
  expect_error(.validate_family_phi("gamma_inverse", -2), "positive")
})

test_that(".validate_family_phi is a no-op for families without dispersion", {
  expect_silent(.validate_family_phi("binomial", -1))   # phi unused
  expect_silent(.validate_family_phi("poisson", 0))     # phi unused
  expect_silent(.validate_family_phi("not_a_family", -5))
})

test_that(".validate_family_counts rejects non-integer / negative counts", {
  for (fam in c("poisson", "binomial", "neg_binomial_2")) {
    expect_error(.validate_family_counts(fam, c(0, 1, 2.5)), "integer", info = fam)
    expect_error(.validate_family_counts(fam, c(0, -1, 2)),  "integer", info = fam)
    expect_silent(.validate_family_counts(fam, c(0, 1, 2, 10)))
  }
  # NA / non-finite entries are ignored (handled elsewhere), not flagged here.
  expect_silent(.validate_family_counts("poisson", c(1, NA, 3)))
  # Continuous families do not constrain y to integers.
  expect_silent(.validate_family_counts("gaussian", c(0.5, 1.7, -2.3)))
})

test_that("tulpa() rejects phi <= 0 for a gaussian fit", {
  skip_on_cran()
  set.seed(1)
  d <- data.frame(y = rnorm(40), x = rnorm(40))
  expect_error(
    tulpa(y ~ x, data = d, family = "gaussian", phi = 0, mode = "laplace"),
    "positive")
  # A valid phi is accepted.
  fit <- tulpa(y ~ x, data = d, family = "gaussian", phi = 1, mode = "laplace")
  expect_s3_class(fit, "tulpa_fit")
})

test_that("tulpa() rejects non-integer y for a poisson fit", {
  skip_on_cran()
  set.seed(2)
  d_bad  <- data.frame(y = c(1.5, 2, 3, 4, 5, 6), x = rnorm(6))
  expect_error(
    tulpa(y ~ x, data = d_bad, family = "poisson", mode = "laplace"),
    "integer")
  d_good <- data.frame(y = rpois(40, 3), x = rnorm(40))
  fit <- tulpa(y ~ x, data = d_good, family = "poisson", mode = "laplace")
  expect_s3_class(fit, "tulpa_fit")
})

# NOTE: the integer-count check lives at the tulpa() front door, NOT in
# tulpa_laplace(). tulpa_laplace() is the engine the EM driver calls per block,
# and the documented EM M-step encodes E-step weights as a CONTINUOUS
# pseudo-binomial response (y in [0, 1]); a count check there would reject that
# legitimate internal usage. So the rule is enforced only where the response is
# user-supplied count data (tulpa()).

# `family = "Gamma"` (the stats::Gamma() spelling) reaching the registry as an
# unresolved string, rather than the "gamma" it means (gcol33/tulpa#806).
test_that(".canonical_family resolves case only as a last resort", {
  expect_equal(.canonical_family("Gamma"), "gamma")
  expect_equal(.canonical_family("GAMMA"), "gamma")
  expect_equal(.canonical_family("Gamma_log"), "gamma_log")
  # An exact registry hit, or an existing alias, is never re-cased.
  expect_equal(.canonical_family("gamma"), "gamma")
  expect_equal(.canonical_family("binomial"), "binomial")
  expect_equal(.canonical_family("negbin"), "neg_binomial_2")
  # Unrecognized at any case: passed through unchanged, still refused downstream.
  expect_equal(.canonical_family("NotAFamily"), "NotAFamily")
})

test_that("family = 'Gamma' and family = Gamma(link = 'log') fit identically", {
  skip_on_cran()
  set.seed(11)
  n <- 60L
  d <- data.frame(x = rnorm(n))
  d$y <- rgamma(n, shape = 3, rate = 3 / exp(0.2 + 0.3 * d$x))
  c_string <- coef(tulpa(y ~ x, data = d, family = "Gamma", mode = "laplace", phi = 3))
  c_object <- coef(tulpa(y ~ x, data = d, family = Gamma(link = "log"), mode = "laplace", phi = 3))
  expect_equal(unname(c_string), unname(c_object))
})

test_that("fit_spde() and tulpa_re_cov_nested() resolve family = 'Gamma' (#806)", {
  skip_on_cran()
  set.seed(12)
  G <- 15L; per <- 8L; nr <- G * per
  grp <- rep(seq_len(G), each = per); xr <- rnorm(nr)
  ygr <- rgamma(nr, shape = 3, rate = 3 / exp(0.2 + 0.3 * xr))
  rt <- list(idx = grp, n_groups = G, n_coefs = 1L, Z = matrix(1, nr, 1), correlated = FALSE)
  r1 <- tulpa_re_cov_nested(ygr, NULL, cbind(1, x = xr), rt, family = "Gamma")$Sigma_mean
  r2 <- tulpa_re_cov_nested(ygr, NULL, cbind(1, x = xr), rt, family = "gamma")$Sigma_mean
  expect_equal(r1, r2)

  expect_error(
    tulpa_re_cov_nested(ygr, NULL, cbind(1, x = xr), rt, family = "NotAFamily"),
    "Unknown family")
})
