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
      got <- family_score_eta(e, cs$y, cs$family, cs$n, cs$phi)
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
      expect_equal(family_score_eta(e, cs$y, cs$family, cs$n, cs$phi),
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

test_that("a hurdle spelling is answered with the composition, not the registry", {
  # There is no hurdle family to alias to -- the mixture over a zero-truncated
  # base IS the hurdle likelihood -- so the useful answer is which two pieces to
  # combine, not the list of everything else on offer.
  base_of <- c(hurdle_poisson = "truncated_poisson",
               hurdle_nbinom2 = "truncated_neg_binomial_2")
  for (nm in names(base_of)) {
    msg <- tryCatch(.family_or_stop(nm), error = conditionMessage)
    expect_match(msg, "ziformula", info = nm)
    expect_match(msg, base_of[[nm]], fixed = TRUE, info = nm)
    # The generic branch must not be the one that fires: dumping the registry
    # here would answer a question the caller did not ask.
    expect_false(grepl("Unknown family", msg), info = nm)
  }
})

# --------------------------------------------------------------------------- #
# (gcol33/tulpa#454) The cloglog lower tail, on relative tolerance             #
#                                                                              #
# mu = 1 - exp(-exp(eta)) cancels as exp(eta) shrinks: the outer subtraction   #
# loses a digit per decade of |eta| and rounds to exactly 0 once exp(eta)      #
# falls below the double spacing at 1, near eta = -36.7. mu = 0 is outside the #
# support of every consumer of the link, so what reaches log_lik_mu is the     #
# clamp floor rather than the probability, and the gradient built from it is   #
# wrong by the same factor. -expm1(-a) is exact for every a > 0.               #
#                                                                              #
# Read against R's own link on RELATIVE tolerance: an absolute one passes on a #
# value with no correct digits left.                                           #
# --------------------------------------------------------------------------- #

test_that("cloglog matches stats::binomial on relative tolerance into the tail", {
  eta <- c(seq(-50, 5, by = 0.25), -745, -100, -37, -36.7, -20, -1e-8)
  R_lk <- stats::binomial("cloglog")
  lad <- cpp_link_ladder(eta, "cloglog")

  expect_true(all(lad[, "linkinv"] > 0),
              label = "no eta returns mu = 0, which is out of support")
  expect_equal(lad[, "linkinv"], R_lk$linkinv(eta), tolerance = 1e-14)
  expect_equal(lad[, "mu_eta"],  R_lk$mu.eta(eta),  tolerance = 1e-14)

  # In the deep tail mu -> exp(eta) to first order; that is the value the
  # cancelling form loses entirely.
  deep <- eta[eta < -40]
  expect_equal(lad[eta < -40, "linkinv"], exp(deep), tolerance = 1e-12)
})

# --------------------------------------------------------------------------- #
# cauchit's lower tail (gcol33/tulpa#602)                                       #
#                                                                              #
# mu = 0.5 + atan(eta) / pi recovers the tail from a subtraction against 0.5,  #
# so it loses a digit per decade of |eta|. atan(x) + atan(1/x) = pi/2 gives    #
# each tail directly: atan(-1/eta) / pi below zero, 1 - atan(1/eta) / pi above #
# it. Same defect as cloglog above, milder -- cauchit's tail decays like       #
# 1 / (pi |eta|), so the value never underflows to 0, it just runs out of      #
# correct digits.                                                              #
#                                                                              #
# `linkinv` is one function behind both binomial_cauchit and beta_cauchit, so  #
# reading cpp_link_ladder covers the two.                                      #
# --------------------------------------------------------------------------- #

test_that("cauchit matches stats::pcauchy on relative tolerance into both tails", {
  eta <- c(-1e9, -1e8, -1e7, -1e6, -1e5, -1e3, -37, -3, -0.5, 0,
           0.5, 3, 37, 1e3, 1e5, 1e6, 1e7, 1e8, 1e9)
  lad <- cpp_link_ladder(eta, "cauchit")

  expect_equal(lad[, "linkinv"], stats::pcauchy(eta), tolerance = 1e-15)
  expect_equal(lad[, "mu_eta"],  stats::dcauchy(eta), tolerance = 1e-15)

  # In the deep lower tail mu -> 1 / (pi |eta|) to first order; that is the
  # quantity the cancelling form is recovering from 0.5.
  deep <- eta[eta <= -1e5]
  expect_equal(lad[eta <= -1e5, "linkinv"], 1 / (pi * abs(deep)),
               tolerance = 1e-9)

  # The ladder is known to be sensitive: the cancelling form it replaces is
  # wrong at these etas by orders of magnitude more than the tolerance above,
  # and worse the further out it is read.
  cancelling <- 0.5 + atan(deep) / pi
  rel <- abs(cancelling - stats::pcauchy(deep)) / stats::pcauchy(deep)
  expect_gt(max(rel), 1e-9)
  expect_true(all(diff(rel[order(abs(deep))]) > 0))
})

test_that("the cauchit derivative ladder is its own finite difference", {
  eta <- c(-30, -10, -3, -0.5, 0, 0.7, 1.8, 12)
  h <- 1e-5
  lad <- function(e) cpp_link_ladder(e, "cauchit")
  for (j in seq_along(eta)) {
    e <- eta[j]
    d1 <- (lad(e + h)[1, "linkinv"] - lad(e - h)[1, "linkinv"]) / (2 * h)
    d2 <- (lad(e + h)[1, "mu_eta"]  - lad(e - h)[1, "mu_eta"])  / (2 * h)
    d3 <- (lad(e + h)[1, "mu_eta2"] - lad(e - h)[1, "mu_eta2"]) / (2 * h)
    expect_equal(unname(lad(e)[1, "mu_eta"]),  unname(d1), tolerance = 1e-6)
    expect_equal(unname(lad(e)[1, "mu_eta2"]), unname(d2), tolerance = 1e-6)
    expect_equal(unname(lad(e)[1, "mu_eta3"]), unname(d3), tolerance = 1e-6)
  }
})

test_that("the R cauchit link reads the same tails as the kernel", {
  eta <- c(-1e9, -1e6, -1e5, -3, 0, 3, 1e5, 1e6, 1e9)
  lk <- tulpa:::.LINKS$cauchit
  expect_equal(lk$linkinv(eta), unname(cpp_link_ladder(eta, "cauchit")[, "linkinv"]),
               tolerance = 1e-15)
  expect_equal(lk$mu_eta(eta), stats::dcauchy(eta), tolerance = 1e-15)

  # linkfun is the inverse, and a mu that far into the tail is where
  # tan(pi * (mu - 0.5)) loses the argument it is meant to recover.
  mu <- c(1e-12, 1e-8, 1e-4, 0.3, 0.5, 0.7, 1 - 1e-4)
  expect_equal(lk$linkinv(lk$linkfun(mu)), mu, tolerance = 1e-13)
})

test_that("the cloglog derivative ladder is its own finite difference", {
  # mu_eta2 / mu_eta3 carry the same cancellation risk one and two derivatives
  # in (-expm1(eta) is the second's stable form), and both reach a gradient.
  eta <- c(-30, -10, -3, -0.5, 0, 0.7, 1.8)
  h <- 1e-5
  lad <- function(e) cpp_link_ladder(e, "cloglog")
  for (j in seq_along(eta)) {
    e <- eta[j]
    d1 <- (lad(e + h)[1, "linkinv"] - lad(e - h)[1, "linkinv"]) / (2 * h)
    d2 <- (lad(e + h)[1, "mu_eta"]  - lad(e - h)[1, "mu_eta"])  / (2 * h)
    d3 <- (lad(e + h)[1, "mu_eta2"] - lad(e - h)[1, "mu_eta2"]) / (2 * h)
    expect_equal(unname(lad(e)[1, "mu_eta"]),  unname(d1), tolerance = 1e-6)
    expect_equal(unname(lad(e)[1, "mu_eta2"]), unname(d2), tolerance = 1e-6)
    expect_equal(unname(lad(e)[1, "mu_eta3"]), unname(d3), tolerance = 1e-6)
  }
})
