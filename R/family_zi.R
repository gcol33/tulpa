# family_zi.R
# Zero-inflation as a composition over the base family registry, not as a
# family of its own. A ZI model adds a second linear predictor -- the logit of
# the structural-zero probability -- on top of any family that has an atom at
# zero:
#
#   pi_i = plogis(logit_zi_i)          structural-zero probability
#   p0_i = P_base(Y = 0 | eta_i)
#
#   y = 0:  loglik = log(pi + (1 - pi) p0)
#   y > 0:  loglik = log(1 - pi) + loglik_base(y | eta)
#
# The hurdle model is the same mixture over a zero-truncated base. There
# p0 == 0, so the y = 0 branch collapses to log(pi), and the count-side
# gradient, count-side curvature, and the cross term all vanish identically --
# leaving exactly the two-part hurdle likelihood. Hurdle models therefore need
# no separate families: pair `ziformula` with `truncated_poisson` or
# `truncated_neg_binomial_2` and this code produces them.
#
# Derivatives are returned in the two-process eta ordering the compiled
# Laplace path uses, eta = [eta_count, logit_zi], so `zi_neg_hessian()`
# columns map onto the row-major 2 x 2 block the EtaWeightsFn contract fills.


# Families admitting a zero atom, hence a structural-zero mixture. Continuous
# families put no mass at any single point, so mixing in one is not identified.
#' @keywords internal
.ZI_FAMILIES <- .COUNT_FAMILIES


# Backends carrying the mixture. "laplace" reaches it as process 1 of the
# two-process spec; every `tulpa_sample_glmm` backend reaches it through the
# `logit_zi` callback argument, which the generic log-posterior builds from
# X_zi_flat and the beta_zi block. Derived from the registry rather than
# hardcoded so a new sampler backend inherits ZI with its fitter. A function,
# not a constant: this file sources before inference_modes.R.
#' @keywords internal
.zi_backends <- function() {
  c("laplace",
    names(BACKEND_REGISTRY)[vapply(
      BACKEND_REGISTRY,
      function(b) identical(b$fitter, "tulpa_sample_glmm"),
      logical(1))])
}


# Families whose compiled kernels carry the zero-inflated mixture. Narrower
# than .ZI_FAMILIES: the mixture's y = 0 branch differentiates through
# P(Y = 0), which needs the base family's OBSERVED curvature rather than the
# Newton working weight, and only these kernels register one
# (obs_grad_hess_for_family / has_observed_curvature in
# src/laplace_family_link.h). beta_binomial is excluded because its compiled
# weight is the moment weight. The R composition in this file covers every
# family in .ZI_FAMILIES for density work regardless.
#
# Paired with a zero-truncated base this is the hurdle model: the mixture
# degenerates exactly, so hurdle needs no family of its own.
#' @keywords internal
.ZI_COMPILED_FAMILIES <- c("poisson", "binomial", "neg_binomial_2",
                           "neg_binomial_1", "truncated_poisson",
                           "truncated_neg_binomial_2")


#' Reject a ZI request the compiled kernels cannot fit.
#'
#' @keywords internal
.validate_family_zi_compiled <- function(family) {
  if (.family_base(family) %in% .ZI_COMPILED_FAMILIES) return(invisible(TRUE))
  stop(sprintf(paste0(
    "`ziformula` with family = '%s' has no compiled kernel, so tulpa() ",
    "cannot fit it. Zero inflation is compiled for: %s. The R-level mixture ",
    "(zi_loglik() and friends) covers this family for density evaluation."),
    family, paste(.ZI_COMPILED_FAMILIES, collapse = ", ")), call. = FALSE)
}


#' Reject a ZI request against a family with no atom at zero.
#'
#' @keywords internal
.validate_family_zi <- function(family) {
  if (.family_base(family) %in% .ZI_FAMILIES) return(invisible(TRUE))
  stop(sprintf(paste0(
    "family = '%s' has no probability atom at zero, so a zero-inflation ",
    "component is not identified. `ziformula` applies to: %s."),
    family, paste(.ZI_FAMILIES, collapse = ", ")), call. = FALSE)
}


# log(1 + exp(x)) and the two logistic log-probabilities, via stats::plogis so
# the tail behaviour is R's rather than hand-rolled.
.log_pi   <- function(z) stats::plogis(z, log.p = TRUE)
.log_1mpi <- function(z) stats::plogis(-z, log.p = TRUE)

# log(exp(a) + exp(b)) without intermediate overflow.
.logspace_add <- function(a, b) {
  m <- pmax(a, b)
  ifelse(is.infinite(m) & m < 0, m, m + log1p(exp(-abs(a - b))))
}


# Recycle the three per-observation vectors to a common length and evaluate the
# base-family quantities every ZI derivative needs. `p0`, `s0` and `w0` are the
# base density, score and observed curvature AT y = 0 -- the mixture's y = 0
# branch differentiates through P_base(Y = 0), not through the realized y.
.zi_pieces <- function(eta, logit_zi, y, family, n_trials, phi, phi2) {
  n <- max(length(eta), length(logit_zi), length(y))
  eta <- rep_len(eta, n)
  z   <- rep_len(logit_zi, n)
  y   <- rep_len(y, n)
  zero <- rep_len(0, n)
  list(
    n = n, eta = eta, z = z, y = y,
    is0 = y == 0,
    pi  = stats::plogis(z),
    lp0 = family_loglik(eta, zero, family, n_trials, phi, phi2),
    s0  = family_score_eta(eta, zero, family, n_trials, phi, phi2),
    w0  = .family_obs_weight(eta, zero, family, n_trials, phi, phi2)
  )
}


#' Elementwise log-likelihood of the zero-inflated mixture.
#'
#' @keywords internal
zi_loglik <- function(eta, logit_zi, y, family, n_trials = NULL, phi = 1.0,
                      phi2 = NULL) {
  .validate_family_zi(family)
  p <- .zi_pieces(eta, logit_zi, y, family, n_trials, phi, phi2)
  ll_zero <- .logspace_add(.log_pi(p$z), .log_1mpi(p$z) + p$lp0)
  ll_pos  <- .log_1mpi(p$z) +
    family_loglik(p$eta, p$y, family, n_trials, phi, phi2)
  ifelse(p$is0, ll_zero, ll_pos)
}


#' Score of the zero-inflated mixture in the two-process eta ordering.
#'
#' Returns an `n x 2` matrix with columns `count` (d loglik / d eta_count) and
#' `zi` (d loglik / d logit_zi).
#'
#' @keywords internal
zi_score_eta <- function(eta, logit_zi, y, family, n_trials = NULL, phi = 1.0,
                         phi2 = NULL) {
  .validate_family_zi(family)
  p  <- .zi_pieces(eta, logit_zi, y, family, n_trials, phi, phi2)
  p0 <- exp(p$lp0)
  D  <- p$pi + (1 - p$pi) * p0

  g_count <- ifelse(p$is0,
                    (1 - p$pi) * p0 * p$s0 / D,
                    family_score_eta(p$eta, p$y, family, n_trials, phi, phi2))
  g_zi <- ifelse(p$is0,
                 p$pi * (1 - p$pi) * (1 - p0) / D,
                 -p$pi)
  cbind(count = g_count, zi = g_zi)
}


#' Negative Hessian of the zero-inflated mixture in eta space.
#'
#' Returns an `n x 3` matrix with columns `count` (-d2/d eta_count^2), `zi`
#' (-d2/d logit_zi^2) and `cross` (-d2/d eta_count d logit_zi), i.e. the
#' distinct entries of the symmetric 2 x 2 block per observation.
#'
#' This is the observed negative Hessian, exact wherever the base family
#' registers `obs_weight`. At y > 0 the mixture is additively separable, so the
#' count block is the base family's own curvature and the cross term is zero;
#' the coupling lives entirely in the y = 0 branch, where both components can
#' explain the zero.
#'
#' @keywords internal
zi_neg_hessian <- function(eta, logit_zi, y, family, n_trials = NULL,
                           phi = 1.0, phi2 = NULL) {
  .validate_family_zi(family)
  p  <- .zi_pieces(eta, logit_zi, y, family, n_trials, phi, phi2)
  p0 <- exp(p$lp0)
  pi <- p$pi
  D  <- pi + (1 - pi) * p0

  # y = 0: derivatives of log D, with D_eta = (1 - pi) p0 s0 and
  # D_z = pi (1 - pi) (1 - p0).
  D_eta <- (1 - pi) * p0 * p$s0
  D_z   <- pi * (1 - pi) * (1 - p0)
  D_ee  <- (1 - pi) * p0 * (p$s0^2 - p$w0)
  D_zz  <- (1 - p0) * pi * (1 - pi) * (1 - 2 * pi)
  D_ez  <- -pi * (1 - pi) * p0 * p$s0

  nh_count0 <- -(D_ee / D - (D_eta / D)^2)
  nh_zi0    <- -(D_zz / D - (D_z / D)^2)
  nh_cross0 <- -(D_ez / D - D_eta * D_z / D^2)

  cbind(
    count = ifelse(p$is0, nh_count0,
                   .family_obs_weight(p$eta, p$y, family, n_trials, phi, phi2)),
    zi    = ifelse(p$is0, nh_zi0, pi * (1 - pi)),
    cross = ifelse(p$is0, nh_cross0, 0)
  )
}


#' Response-scale mean of the zero-inflated mixture.
#'
#' @keywords internal
zi_response_mean <- function(eta, logit_zi, family, n_trials = NULL,
                             phi = 1.0) {
  .validate_family_zi(family)
  n  <- max(length(eta), length(logit_zi))
  pi <- stats::plogis(rep_len(logit_zi, n))
  (1 - pi) * family_response_mean(rep_len(eta, n), family, n_trials, phi)
}


#' Response variance of the zero-inflated mixture.
#'
#' Law of total variance over the mixture indicator: with probability `pi` the
#' response is a structural zero, otherwise it is a base-family draw.
#'
#' @keywords internal
zi_variance <- function(eta, logit_zi, family, n_trials = NULL, phi = 1.0,
                        phi2 = NULL) {
  .validate_family_zi(family)
  n   <- max(length(eta), length(logit_zi))
  eta <- rep_len(eta, n)
  pi  <- stats::plogis(rep_len(logit_zi, n))
  m   <- family_response_mean(eta, family, n_trials, phi)
  v   <- family_variance(eta, family, n_trials, phi, phi2)
  (1 - pi) * (v + m^2) - ((1 - pi) * m)^2
}


#' One draw per element from the zero-inflated mixture.
#'
#' @keywords internal
zi_sample <- function(eta, logit_zi, family, n_trials = NULL, phi = 1.0,
                      phi2 = NULL) {
  .validate_family_zi(family)
  n   <- max(length(eta), length(logit_zi))
  eta <- rep_len(eta, n)
  pi  <- stats::plogis(rep_len(logit_zi, n))
  structural <- stats::runif(n) < pi
  out <- family_sample(eta, family, n_trials, phi, phi2)
  out[structural] <- 0
  out
}
