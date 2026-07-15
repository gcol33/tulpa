# Cross-scale equivalence for the shared PC prior (gcol33/tulpa#142 A4).
#
# The PC prior is an exponential on the standard deviation sigma, but samplers
# parameterize the scale six different ways. Each site used to re-derive the
# change of variables by hand, and two got it wrong (tulpa_priors_tvc.h carried
# an excess +2*log_tau; tulpa_priors_hsgp.h dropped +log(sigma)). These tests
# pin every scale against the base density plus a NUMERICAL Jacobian, so a
# re-derivation on the wrong scale fails here instead of silently tilting a
# hyperparameter posterior.

pc_lambda <- function(U, alpha) -log(alpha) / U

# sigma(x) for each sampled coordinate x
pc_sigma_of <- list(
  sigma      = function(x) x,
  log_sigma  = function(x) exp(x),
  sigma2     = function(x) sqrt(x),
  log_sigma2 = function(x) exp(x / 2),
  tau        = function(x) 1 / sqrt(x),
  log_tau    = function(x) exp(-x / 2)
)

# The sampled coordinate x corresponding to a given sigma
pc_coord_of <- list(
  sigma      = function(s) s,
  log_sigma  = function(s) log(s),
  sigma2     = function(s) s^2,
  log_sigma2 = function(s) log(s^2),
  tau        = function(s) 1 / s^2,
  log_tau    = function(s) log(1 / s^2)
)

test_that("every PC-prior scale equals the base density plus a numerical Jacobian", {
  U <- 1.0; alpha <- 0.01
  lambda <- pc_lambda(U, alpha)

  for (sigma in c(0.05, 0.3, 1.0, 2.5)) {
    got <- cpp_pc_prior_scales(sigma, U, alpha)

    for (scale in names(pc_sigma_of)) {
      x <- pc_coord_of[[scale]](sigma)
      # d(sigma)/d(x) by central difference on the sampled coordinate
      h <- 1e-6 * max(1, abs(x))
      dsig_dx <- (pc_sigma_of[[scale]](x + h) - pc_sigma_of[[scale]](x - h)) / (2 * h)
      expected <- log(lambda) - lambda * sigma + log(abs(dsig_dx))
      expect_equal(unname(got[[scale]]), expected, tolerance = 1e-6,
                   info = paste("scale:", scale, "sigma:", sigma))
    }
  }
})

# Integration range per sampled coordinate. The linear coordinates run to +Inf,
# where integrate()'s tail transformation handles the decay. The log coordinates
# are bounded, because THIS HARNESS -- not the density -- breaks out there: it
# maps x to sigma in R via exp(), which overflows to Inf past x ~ 709 and hands
# the C++ an infinite sigma, whose log-scale Jacobian is then +Inf. The compiled
# density is fine at any finite x (safe_exp() clamps at +-700, so the upper tail
# is driven to zero and stays integrable). +-50 spans sigma from 2e-22 to 5e21,
# which is the whole mass at this rate and far more than a posterior occupies.
pc_range_of <- list(
  sigma      = c(0, Inf),
  log_sigma  = c(-50, 50),
  sigma2     = c(0, Inf),
  log_sigma2 = c(-50, 50),
  tau        = c(0, Inf),
  log_tau    = c(-50, 50)
)

test_that("each PC-prior scale integrates to one (a proper density)", {
  # A wrong Jacobian (the tvc/hsgp bugs) breaks normalization: the tvc form was
  # improper as tau -> Inf. Integrating on each coordinate catches that.
  U <- 1.0; alpha <- 0.01

  for (scale in names(pc_sigma_of)) {
    dens <- function(x) {
      vapply(x, function(xi) {
        sigma <- pc_sigma_of[[scale]](xi)
        exp(unname(cpp_pc_prior_scales(sigma, U, alpha)[[scale]]))
      }, numeric(1))
    }
    rng <- pc_range_of[[scale]]
    total <- stats::integrate(dens, rng[1L], rng[2L], subdivisions = 500L,
                              rel.tol = 1e-8)$value
    expect_equal(total, 1, tolerance = 1e-5, info = paste("scale:", scale))
  }
})

test_that("the PC-prior rate is calibrated so P(sigma > U) = alpha", {
  for (U in c(0.5, 1, 3)) {
    for (alpha in c(0.01, 0.05, 0.5)) {
      lambda <- pc_lambda(U, alpha)
      # tail of the base exponential on sigma
      expect_equal(exp(-lambda * U), alpha, tolerance = 1e-12)
      # the exported base density agrees with the exponential
      got <- cpp_pc_prior_scales(U, U, alpha)
      expect_equal(unname(got[["sigma"]]),
                   stats::dexp(U, rate = lambda, log = TRUE), tolerance = 1e-10)
    }
  }
})
