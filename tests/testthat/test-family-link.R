# The link layer (R/family_link.R) expresses each base family in mu space and
# composes it with a link. `.FAMILY_OPS` expresses the same families directly in
# eta with the canonical link already substituted and simplified. Two
# representations of one family can drift, so the first test below binds them:
# compose each base family with its OWN canonical link and require the result to
# match the hand-written entry.
#
# That composition is not what the canonical path actually runs -- .family_ops()
# returns the hand-written entry for a canonical code, so its simplified
# arithmetic is preserved bit for bit -- it is the check that the mu-space
# primitives encode the same family.

test_that("composing a family with its own canonical link reproduces the registry", {
  set.seed(3)
  eta <- seq(-1.5, 1.5, length.out = 11)
  cases <- list(
    list(base = "gaussian",         y = 1.2,  n = 1L,  phi = 1.4),
    list(base = "lognormal",        y = 1.7,  n = 1L,  phi = 0.8),
    list(base = "binomial",         y = 3,    n = 10L, phi = 1.0),
    list(base = "poisson",          y = 4,    n = 1L,  phi = 1.0),
    list(base = "neg_binomial_2",   y = 6,    n = 1L,  phi = 2.5),
    list(base = "gamma",            y = 2.0,  n = 1L,  phi = 3.0),
    list(base = "inverse_gaussian", y = 1.4,  n = 1L,  phi = 1.1),
    list(base = "beta",             y = 0.6,  n = 1L,  phi = 6.0)
  )

  for (cs in cases) {
    ref      <- tulpa:::.FAMILY_OPS[[cs$base]]
    link     <- unname(tulpa:::.LINK_DEFAULTS[[cs$base]])
    composed <- tulpa:::.compose_family_ops(cs$base, link)
    nt       <- rep(cs$n, length(eta))
    info     <- paste(cs$base, "/", link)

    expect_equal(composed$loglik(eta, cs$y, nt, cs$phi),
                 ref$loglik(eta, cs$y, nt, cs$phi),
                 tolerance = 1e-10, info = info)
    expect_equal(composed$score(eta, cs$y, nt, cs$phi),
                 ref$score(eta, cs$y, nt, cs$phi),
                 tolerance = 1e-10, info = info)
    expect_equal(composed$weight(eta, nt, cs$phi),
                 ref$weight(eta, nt, cs$phi),
                 tolerance = 1e-10, info = info)
    expect_equal(composed$variance(eta, nt, cs$phi),
                 ref$variance(eta, nt, cs$phi),
                 tolerance = 1e-10, info = info)
  }
})

test_that("a canonical family code keeps the hand-written entry, not the composition", {
  # Identity, not equality: the simplified closures are the ones that must run,
  # so the canonical path's arithmetic is unchanged by the link layer existing.
  for (base in names(tulpa:::.LINK_DEFAULTS)) {
    expect_identical(tulpa:::.family_ops(base), tulpa:::.FAMILY_OPS[[base]],
                     info = base)
  }
})

test_that("the score is the eta-derivative of the log-likelihood under any link", {
  # The composition is only correct if score and weight are the derivatives of
  # the log-density it reports. Finite differences check that directly, at etas
  # inside each link's domain.
  h <- 1e-6
  cases <- list(
    list(family = "gamma_inverse",         y = 2.0, n = 1L,  phi = 3.0, eta = c(0.4, 0.9, 1.6)),
    list(family = "poisson_sqrt",          y = 4,   n = 1L,  phi = 1.0, eta = c(0.4, 0.9, 1.6)),
    list(family = "inverse_gaussian_1mu2", y = 1.4, n = 1L,  phi = 1.1, eta = c(0.4, 0.9, 1.6)),
    list(family = "binomial_probit",       y = 3,   n = 10L, phi = 1.0, eta = c(-0.7, 0, 0.8)),
    list(family = "binomial_cloglog",      y = 3,   n = 10L, phi = 1.0, eta = c(-0.7, 0, 0.8)),
    list(family = "poisson_identity",      y = 4,   n = 1L,  phi = 1.0, eta = c(1.5, 3.0, 5.0)),
    list(family = "gaussian_log",          y = 1.2, n = 1L,  phi = 1.5, eta = c(-0.4, 0.3, 1.0))
  )

  for (cs in cases) {
    for (e in cs$eta) {
      ll <- function(z) family_loglik(z, cs$y, cs$family, cs$n, cs$phi)
      num <- (ll(e + h) - ll(e - h)) / (2 * h)
      got <- family_score(e, cs$y, cs$family, cs$n, cs$phi)
      expect_equal(got, num, tolerance = 1e-5,
                   info = paste(cs$family, "at eta =", e))
    }
  }
})

test_that("the R link layer agrees with the compiled kernels", {
  # R and C++ must describe the same model: the R registry now feeds H_beta and
  # posterior_predict for exactly the families the engine fits.
  cases <- list(
    list(family = "gamma_inverse",         y = 2.0, n = 1L, phi = 3.0, eta = c(0.4, 1.1)),
    list(family = "inverse_gaussian_1mu2", y = 1.4, n = 1L, phi = 1.1, eta = c(0.4, 1.1)),
    list(family = "poisson_sqrt",          y = 4,   n = 1L, phi = 1.0, eta = c(0.4, 1.1)),
    list(family = "gaussian_log",          y = 1.2, n = 1L, phi = 1.5, eta = c(-0.4, 0.7))
  )
  for (cs in cases) {
    for (e in cs$eta) {
      # tulpa_laplace()'s gaussian/lognormal phi is the VARIANCE; the compiled
      # kernels take the SD. Convert at the boundary, as the front door does.
      phi_cpp <- if (cs$family %in% c("gaussian_log")) sqrt(cs$phi) else cs$phi
      cpp <- cpp_family_terms(cs$y, cs$n, e, cs$family, phi_cpp)
      expect_equal(family_loglik(e, cs$y, cs$family, cs$n, cs$phi),
                   unname(cpp[["log_lik"]]), tolerance = 1e-10,
                   info = paste(cs$family, "loglik at", e))
      expect_equal(family_score(e, cs$y, cs$family, cs$n, cs$phi),
                   unname(cpp[["grad"]]), tolerance = 1e-10,
                   info = paste(cs$family, "score at", e))
    }
  }
})

test_that("the binomial working weight carries n_trials the right way round", {
  # y ~ Bin(n, mu): Fisher information on eta is n (dmu/deta)^2 / (mu (1-mu)).
  # variance_fn used to return the RESPONSE variance n mu (1-mu) into the
  # dmu^2 / V composition, which is n^2 too small. The canonical logit link is
  # answered before that route, so only the suffixed forms were wrong, and only
  # at n > 1.
  for (link in c("probit", "cloglog", "cauchit")) {
    fam <- paste0("binomial_", link)
    lk  <- tulpa:::.LINKS[[link]]
    for (n in c(1L, 10L, 37L)) {
      for (eta in c(-0.8, 0, 0.5)) {
        mu   <- lk$linkinv(eta)
        want <- n * lk$mu_eta(eta)^2 / (mu * (1 - mu))
        expect_equal(unname(cpp_family_terms(3, n, eta, fam, 1)[["neg_hess"]]),
                     want, tolerance = 1e-10,
                     info = paste(fam, "n =", n, "eta =", eta))
        # ... and the R layer agrees with the kernel it now feeds.
        expect_equal(family_weight(eta, fam, n, 1), want, tolerance = 1e-10,
                     info = paste(fam, "R weight, n =", n))
      }
    }
  }
})

test_that("the canonical binomial weight is unchanged", {
  # The fix touches the generic route only; logit is answered before it.
  for (n in c(1L, 10L)) {
    for (eta in c(-0.8, 0, 0.5)) {
      mu <- stats::plogis(eta)
      expect_equal(unname(cpp_family_terms(3, n, eta, "binomial", 1)[["neg_hess"]]),
                   n * mu * (1 - mu), tolerance = 1e-10)
    }
  }
})

test_that("the R log-likelihood is -Inf outside a constrained link's domain", {
  for (fam in c("gamma_inverse", "inverse_gaussian_1mu2")) {
    expect_identical(family_loglik(c(-2, 0), 1.5, fam, 1L, 2.0), c(-Inf, -Inf),
                     info = fam)
  }
  expect_identical(family_loglik(c(-2, 0), 4, "poisson_sqrt", 1L, 1.0),
                   c(-Inf, -Inf))
})

test_that("an unparseable family still errors, and the message names the links", {
  expect_error(.family_or_stop("not_a_family"), "Unknown family")
  expect_error(.family_or_stop("gamma_banana"), "Unknown family")
  # A base family that takes no link suffix stays rejected in suffixed form.
  expect_error(.family_or_stop("tweedie_inverse"), "Unknown family")
  expect_error(.family_or_stop("not_a_family"), "1mu2")
})
