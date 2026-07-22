# External-MLE reference checks -- shared scaffolding.
#
# The likelihoods in R/family_loglik.R and their compiled kernels are checked
# against independent implementations of the same likelihood: lme4, glmmTMB,
# betareg, pscl and MASS::glm.nb. A parameter-recovery test against a simulated
# truth cannot separate a likelihood bug from sampling noise at finite N; an
# agreement test against a second implementation of the same likelihood can,
# because both are looking at identical data.
#
# The comparison is on tulpa's Laplace path with a diffuse fixed-effect prior
# and the dispersion held at the reference's own estimate. Conditioned that way
# the Laplace mode IS the MLE of the same likelihood, so agreement is expected
# to several significant digits rather than to a fraction of a standard error.
# Anything looser would pass while a term of the log-density was wrong.

# A prior wide enough that the penalty it adds is negligible against the data
# term at the sample sizes used here, so the mode is the MLE to the precision
# being asserted.
REF_DIFFUSE_SD <- 100

# |estimate - reference| measured in reference standard errors. `tol` is in the
# same units, so a tolerance of 1e-3 asserts agreement to a thousandth of an SE.
expect_agrees_with_mle <- function(est, ref_est, ref_se, tol = 1e-3,
                                   label = "") {
  est     <- as.numeric(est)
  ref_est <- as.numeric(ref_est)
  ref_se  <- as.numeric(ref_se)
  for (j in seq_along(ref_est)) {
    d <- abs(est[[j]] - ref_est[[j]]) / ref_se[[j]]
    testthat::expect_lt(
      d, tol,
      label = sprintf("%s[%d]: |est - mle| / mle_se = %.3g (tol %.3g)",
                      label, j, d, tol)
    )
  }
}

# Profile the Laplace log-marginal over a dispersion grid and return the value
# it peaks at. The dispersion is conditioned on rather than estimated on this
# path, so the likelihood's dependence on it is checked by where the profile
# puts its maximum.
ref_profile_phi <- function(formula, data, family, grid, ...) {
  lm_ <- vapply(grid, function(p) {
    fit <- suppressMessages(
      tulpa(formula, data = data, family = family, mode = "laplace",
            phi = p, ...))
    fit$log_marginal
  }, numeric(1))
  grid[which.max(lm_)]
}

# Simulate y > 0 from Poisson(mu) / NB2(mu, phi) by rejection, so the marginal
# is exactly the zero-truncated law rather than a re-weighted approximation.
ref_sim_truncated_poisson <- function(mu) {
  y <- stats::rpois(length(mu), mu)
  while (any(y == 0)) {
    i <- which(y == 0)
    y[i] <- stats::rpois(length(i), mu[i])
  }
  y
}

ref_sim_truncated_nb2 <- function(mu, phi) {
  y <- stats::rnbinom(length(mu), mu = mu, size = phi)
  while (any(y == 0)) {
    i <- which(y == 0)
    y[i] <- stats::rnbinom(length(i), mu = mu[i], size = phi)
  }
  y
}

for (.nm in c("REF_DIFFUSE_SD", "expect_agrees_with_mle", "ref_profile_phi",
              "ref_sim_truncated_poisson", "ref_sim_truncated_nb2")) {
  assign(.nm, get(.nm), envir = globalenv())
}
rm(.nm)
