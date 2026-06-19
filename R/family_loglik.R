# family_loglik.R
# ------------------------------------------------------------------------------
# Single source of truth for the R-level family math: the inverse-link mean,
# the log-likelihood, the score (d log-lik / d eta), and the Laplace/IRLS
# working weight (-d^2 log-lik / d eta^2 under the canonical/expected form).
#
# Each family is one entry in `.FAMILY_OPS`, so adding a family is a single
# registry entry -- no per-family branches scattered across fitters. The
# Laplace Newton loop (`glmm_weights()`), the sampler-path log-posterior
# builder, and any diagnostics all read from here, so the mean function and the
# weights cannot drift between code paths.
#
# Parameterization (matches the character-named families the design fitters
# already use): `eta` is the linear predictor, `phi` the dispersion/precision
# (residual variance for gaussian, size for neg_binomial_2, precision for beta),
# `n_trials` the binomial denominator (default 1).
# ------------------------------------------------------------------------------

# --- shared, clamped inverse-link means (used by loglik / score / weight) -----

.mean_binomial <- function(eta, ...) {
  mu <- 1 / (1 + exp(-eta))
  pmin(pmax(mu, 1e-8), 1 - 1e-8)
}
.mean_log <- function(eta, ...) pmax(exp(eta), 1e-8)   # poisson, neg_binomial_2
.mean_identity <- function(eta, ...) eta               # gaussian
.mean_beta <- function(eta, ...) {
  mu <- 1 / (1 + exp(-eta))
  pmin(pmax(mu, 1e-7), 1 - 1e-7)
}

# --- family operations registry ----------------------------------------------
# Fields per family:
#   mean(eta)                     inverse link, clamped
#   loglik(eta, y, n_trials, phi) elementwise log-density (full normalizer)
#   score(eta, y, n_trials, phi)  elementwise d log-lik / d eta
#   weight(eta, n_trials, phi)    Laplace/IRLS working weight (no y dependence)

.FAMILY_OPS <- list(
  binomial = list(
    mean = .mean_binomial,
    loglik = function(eta, y, n_trials, phi) {
      n <- if (!is.null(n_trials)) n_trials else rep(1, length(eta))
      mu <- .mean_binomial(eta)
      lchoose(n, y) + y * log(mu) + (n - y) * log1p(-mu)
    },
    score = function(eta, y, n_trials, phi) {
      n <- if (!is.null(n_trials)) n_trials else rep(1, length(eta))
      y - n * .mean_binomial(eta)
    },
    weight = function(eta, n_trials, phi) {
      n <- if (!is.null(n_trials)) n_trials else rep(1, length(eta))
      mu <- .mean_binomial(eta)
      n * mu * (1 - mu)
    }
  ),

  poisson = list(
    mean = .mean_log,
    loglik = function(eta, y, n_trials, phi) {
      mu <- .mean_log(eta)
      y * log(mu) - mu - lgamma(y + 1)
    },
    score = function(eta, y, n_trials, phi) y - .mean_log(eta),
    weight = function(eta, n_trials, phi) .mean_log(eta)
  ),

  neg_binomial_2 = list(
    mean = .mean_log,
    loglik = function(eta, y, n_trials, phi) {
      mu <- .mean_log(eta)
      lgamma(y + phi) - lgamma(phi) - lgamma(y + 1) +
        phi * (log(phi) - log(phi + mu)) +
        y * (log(mu) - log(mu + phi))
    },
    score = function(eta, y, n_trials, phi) {
      mu <- .mean_log(eta)
      phi * (y - mu) / (mu + phi)
    },
    weight = function(eta, n_trials, phi) {
      mu <- .mean_log(eta)
      mu * phi / (mu + phi)
    }
  ),

  gaussian = list(
    mean = .mean_identity,
    loglik = function(eta, y, n_trials, phi) {
      -0.5 * ((y - eta)^2 / phi + log(2 * pi * phi))
    },
    score = function(eta, y, n_trials, phi) (y - eta) / phi,
    # Laplace/IRLS working weight = -d^2 log-lik / d eta^2 = 1/phi (phi is the
    # residual variance). The dispersion MUST enter here: the marginal-SE Schur
    # (.marginal_H_beta_*) and the dense H_beta path are the only callers, and
    # both need the dispersion-correct curvature. A unit weight here treats phi
    # as 1 and inflates the gaussian fixed-effect SEs by sqrt(1/phi)
    # (gcol33/tulpa#87).
    weight = function(eta, n_trials, phi) rep(1 / phi, length(eta))
  ),

  beta = list(
    mean = .mean_beta,
    loglik = function(eta, y, n_trials, phi) {
      mu <- .mean_beta(eta)
      a <- mu * phi
      b <- (1 - mu) * phi
      lgamma(phi) - lgamma(a) - lgamma(b) +
        (a - 1) * log(y) + (b - 1) * log1p(-y)
    },
    score = function(eta, y, n_trials, phi) {
      mu <- .mean_beta(eta)
      a <- mu * phi
      b <- (1 - mu) * phi
      dmu <- mu * (1 - mu)
      phi * (log(y) - log1p(-y) - digamma(a) + digamma(b)) * dmu
    },
    weight = function(eta, n_trials, phi) {
      mu <- .mean_beta(eta)
      dmu <- mu * (1 - mu)
      tg <- trigamma(mu * phi) + trigamma((1 - mu) * phi)
      phi * phi * tg * dmu * dmu
    }
  )
)


#' Supported R-level family names.
#' @keywords internal
family_names <- function() names(.FAMILY_OPS)


# Families carrying a dispersion / precision parameter `phi` that enters the
# likelihood, score, and working weight. A non-positive `phi` makes the variance
# / log-density singular (gaussian: r^2 / (2 phi^2); gamma: lgamma(phi)), so it
# must be validated at the front door rather than flowing into a NaN log-lik
# (gcol33/tulpa#104). The `*_<link>` suffix forms (e.g. "gamma_inverse") share
# the base family's dispersion, so membership is tested on the prefix.
.PHI_FAMILIES <- c("gaussian", "gamma", "neg_binomial_2", "negative_binomial",
                   "inverse_gaussian", "beta", "interval_gaussian",
                   "truncated_gaussian")

# Count families whose likelihood is defined only at non-negative integer `y`.
# The C++ kernels cast the response to `int` (laplace_family_link.h /
# laplace_likelihoods.h), silently flooring a continuous response into a biased
# log-likelihood, so a non-integer `y` is rejected at the front door.
.COUNT_FAMILIES <- c("poisson", "binomial", "neg_binomial_2", "negative_binomial")

# Base family of a `family` / `family_<link>` code (the part before the link
# suffix), so a custom link form validates against the same rules.
.family_base <- function(family) {
  if (!is.character(family) || length(family) != 1L) return(NA_character_)
  for (fam in unique(c(.PHI_FAMILIES, .COUNT_FAMILIES, family_names()))) {
    if (identical(family, fam) ||
        startsWith(family, paste0(fam, "_"))) return(fam)
  }
  family
}

#' Validate the dispersion parameter `phi` for a family.
#'
#' Errors when `family` carries a dispersion / precision parameter and `phi` is
#' not a positive finite scalar. A no-op for families without dispersion
#' (binomial, poisson) and for unknown / model-package families. Shared by the
#' [tulpa()] front door and [tulpa_laplace()] so the rule lives in one place
#'.
#' @keywords internal
.validate_family_phi <- function(family, phi) {
  if (!(.family_base(family) %in% .PHI_FAMILIES)) return(invisible(TRUE))
  if (!is.numeric(phi) || length(phi) != 1L || !is.finite(phi) || phi <= 0) {
    stop(sprintf("`phi` must be a positive finite scalar for family = '%s'.",
                 family), call. = FALSE)
  }
  invisible(TRUE)
}

#' Validate the response `y` for a count family.
#'
#' Errors when `family` is a count family and `y` carries negative or
#' non-integer values, which the integer-casting kernels would silently floor
#' into a biased likelihood. A no-op for continuous families.
#' @keywords internal
.validate_family_counts <- function(family, y) {
  if (!(.family_base(family) %in% .COUNT_FAMILIES)) return(invisible(TRUE))
  yf <- y[is.finite(y)]
  if (length(yf) && (any(yf < 0) || any(yf != round(yf)))) {
    stop(sprintf(paste0(
      "family = '%s' requires non-negative integer counts in `y`; got negative ",
      "or non-integer values. The likelihood casts `y` to an integer, which ",
      "would silently floor a continuous response into a biased log-likelihood."),
      family), call. = FALSE)
  }
  invisible(TRUE)
}


#' Look up a family's operation set, with a clear error for unknown families.
#' @keywords internal
.family_ops <- function(family) {
  ops <- .FAMILY_OPS[[family]]
  if (is.null(ops)) {
    stop(sprintf(
      "Unknown family '%s'. Supported: %s.",
      family, paste(family_names(), collapse = ", ")
    ), call. = FALSE)
  }
  ops
}


#' Inverse-link mean for a family.
#' @keywords internal
family_mean <- function(eta, family) .family_ops(family)$mean(eta)


#' Elementwise log-likelihood for a family.
#' @keywords internal
family_loglik <- function(eta, y, family, n_trials = NULL, phi = 1.0) {
  .family_ops(family)$loglik(eta, y, n_trials, phi)
}


#' Score (d log-likelihood / d eta), elementwise.
#' @keywords internal
family_score_eta <- function(eta, y, family, n_trials = NULL, phi = 1.0) {
  .family_ops(family)$score(eta, y, n_trials, phi)
}


#' Laplace/IRLS working weight (-d^2 log-lik / d eta^2), elementwise.
#' @keywords internal
family_weight <- function(eta, family, n_trials = NULL, phi = 1.0) {
  .family_ops(family)$weight(eta, n_trials, phi)
}
