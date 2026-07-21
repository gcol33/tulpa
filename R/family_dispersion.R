# ============================================================================
# Dispersion derivatives: the phi half of the family registry.
#
# Estimating phi by empirical Bayes needs the outer objective differentiated
# with respect to it. Writing l(x) = sum_i w_i loglik_i(eta_i; phi) for the
# joint field, H = A' W A + P for the Laplace precision, and noting that the
# random-effect prior P does not involve phi,
#
#     m(phi) = l(x_hat) - 0.5 (x_hat - mu)' P (x_hat - mu)
#              + 0.5 log|P| - 0.5 log|H|
#
# The inner solve is stationary in x, so the mode's movement contributes
# nothing through the first two terms -- but log|H| depends on phi twice, once
# explicitly through W(eta, phi) and once through the mode. Implicit
# differentiation of the stationarity condition gives that second path:
#
#     dx_hat/dphi = H^-1 A' (w * dscore/dphi)
#
# and collecting both,
#
#     dm/dphi = sum_i w_i dloglik_i/dphi
#               - 0.5 sum_i w_i s_i [ dW_i/dphi + (dW_i/deta_i)(A dx_hat/dphi)_i ]
#
# with s_i = (A H^-1 A')_ii. The optimizer works in log phi (phi > 0), where
# dm/dlog_phi = phi dm/dphi.
#
# Everything except the three derivatives below is already built for the
# random-effect gradient: s_i and dW/deta come from laplace_gradient.R, and the
# H^-1 solves are the same ones. So a family becomes dispersion-estimable by
# adding one entry here, not by touching the optimizer.
#
# Each entry differentiates the EXACT `loglik` / `score` / `weight` registered
# in .FAMILY_OPS for that family -- not the textbook form, which can differ in
# parameterization (phi is a size for neg_binomial_2, a variance for gaussian, a
# shape for gamma, a precision for beta). test-family-dispersion.R finite-
# differences each one against its registry counterpart, which is what keeps
# them honest if a registered likelihood is ever reparameterized.
#
# Families absent from this list have no free dispersion (poisson, binomial) or
# no derivative derived yet; `.family_dphi()` returns NULL for them and the
# caller refuses to estimate rather than reporting a fixed phi as an estimate.
# ============================================================================

#' @keywords internal
.FAMILY_DPHI <- list(

  # phi = size. mu = exp(eta); Var(y) = mu + mu^2 / phi.
  neg_binomial_2 = list(
    dloglik = function(eta, y, n_trials, phi) {
      mu <- .mean_log(eta)
      digamma(y + phi) - digamma(phi) + log(phi) - log(phi + mu) +
        1 - (phi + y) / (phi + mu)
    },
    dscore = function(eta, y, n_trials, phi) {
      mu <- .mean_log(eta)
      (y - mu) * mu / (mu + phi)^2
    },
    # H is built from the OBSERVED curvature for this family --
    # `mu phi (y + phi) / (mu + phi)^2`, which is `.FAMILY_OPS$obs_weight` and
    # the branch at laplace_family_curvature.h:126 -- not from the expected
    # weight the registry calls `weight`. Differentiating the expected form
    # instead leaves the gradient wrong by a few percent, which is enough to
    # shift the maximizer and nothing else would flag.
    #
    #   W       = mu phi (y + phi) / (mu + phi)^2
    #   dW/dphi = mu [ y (mu - phi) + 2 phi mu ] / (mu + phi)^3
    #
    # At y = E[y] = mu this collapses to mu^2 / (mu + phi)^2, the expected
    # weight's derivative, which is the consistency check between the two.
    dweight = function(eta, y, n_trials, phi) {
      mu <- .mean_log(eta)
      mu * (y * (mu - phi) + 2 * phi * mu) / (mu + phi)^3
    }
  ),

  # phi = residual VARIANCE (the R-side convention; the compiled kernels take
  # the SD and are handed sqrt(phi) at the boundary).
  gaussian = list(
    dloglik = function(eta, y, n_trials, phi) {
      (y - eta)^2 / (2 * phi^2) - 1 / (2 * phi)
    },
    dscore = function(eta, y, n_trials, phi) -(y - eta) / phi^2,
    dweight = function(eta, y, n_trials, phi) rep(-1 / phi^2, length(eta))
  ),

  # phi = shape. The working weight is phi itself (constant in eta), so the
  # dW/deta path contributes nothing here and dW/dphi is 1.
  gamma = list(
    dloglik = function(eta, y, n_trials, phi) {
      mu <- .mean_log(eta)
      log(phi) + 1 - digamma(phi) + log(y) - log(mu) - y / mu
    },
    dscore = function(eta, y, n_trials, phi) {
      mu <- .mean_log(eta)
      (y - mu) / mu
    },
    dweight = function(eta, y, n_trials, phi) rep(1, length(eta))
  ),

  # NOT REGISTERED: beta. Its three derivatives are correct in isolation (they
  # finite-difference against the registry cleanly), but the assembled
  # dm/dlog_phi is not exact for it, and it is the only family where that is
  # true. The mode's movement comes from differentiating the TRUE stationarity
  # condition, so it needs the OBSERVED curvature:
  #
  #     (A' W_obs A + P) dx_hat/dphi = A' (w dscore/dphi)
  #
  # `H` is that matrix for neg_binomial_2 (which builds H from the observed
  # curvature) and for gaussian / gamma (where observed and expected coincide,
  # the weight being constant in eta). For beta they differ -- `variance_fn` is
  # constructed so dmu^2 / V is the FISHER information -- so H^-1 is the wrong
  # inverse in that one term. Measured, the assembled gradient lands within
  # ~1e-4 relative of a central difference of the log-marginal where the other
  # three reach 1e-9.
  #
  # Supporting it needs the beta observed curvature as its own registry entry
  # and a second factorization for the mode-movement solve. Until then
  # `.family_dphi("beta")` returns NULL and the caller refuses, rather than
  # optimizing along a gradient that is quietly a little bit wrong. The
  # derivatives are kept below, unregistered, so the work is not lost.
  .beta_unregistered = list(
    dloglik = function(eta, y, n_trials, phi) {
      mu <- .mean_beta(eta)
      a <- mu * phi
      b <- (1 - mu) * phi
      digamma(phi) - mu * digamma(a) - (1 - mu) * digamma(b) +
        mu * log(y) + (1 - mu) * log1p(-y)
    },
    dscore = function(eta, y, n_trials, phi) {
      mu <- .mean_beta(eta)
      a <- mu * phi
      b <- (1 - mu) * phi
      dmu <- mu * (1 - mu)
      dmu * ((log(y) - log1p(-y) - digamma(a) + digamma(b)) +
               phi * (-trigamma(a) * mu + trigamma(b) * (1 - mu)))
    },
    dweight = function(eta, y, n_trials, phi) {
      mu <- .mean_beta(eta)
      a <- mu * phi
      b <- (1 - mu) * phi
      dmu <- mu * (1 - mu)
      dmu^2 * (2 * phi * (trigamma(a) + trigamma(b)) +
                 phi^2 * (psigamma(a, 2L) * mu + psigamma(b, 2L) * (1 - mu)))
    }
  )
)


#' Dispersion derivatives for a family, or `NULL`
#'
#' @param family Family name, as registered in `.FAMILY_OPS`.
#' @return A list with `dloglik`, `dscore` and `dweight`, or `NULL` when the
#'   family has no free dispersion or none has been derived. `NULL` is a refusal
#'   to estimate, never a signal to fall back to a fixed value silently.
#' @keywords internal
.family_dphi <- function(family) {
  nm <- .family_base(family)
  # The leading-dot entries are parked derivations, not registered families;
  # looking one up by name must not resurrect it.
  if (startsWith(nm, ".")) return(NULL)
  .FAMILY_DPHI[[nm]]
}


#' Families whose dispersion can be estimated
#' @keywords internal
.dispersion_families <- function() {
  nms <- names(.FAMILY_DPHI)
  nms[!startsWith(nms, ".")]
}
