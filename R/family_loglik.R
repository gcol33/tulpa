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

# Inverse-Gaussian sampler (Michael, Schucany & Haas 1976), parameterized to
# match the `inverse_gaussian` loglik below: mean mu, shape lambda = 1 / phi.
.rinvgauss <- function(n, mu, lambda) {
  y <- stats::rnorm(n)^2
  x <- mu + mu^2 * y / (2 * lambda) -
    mu / (2 * lambda) * sqrt(4 * mu * lambda * y + mu^2 * y^2)
  u <- stats::runif(n)
  ifelse(u <= mu / (mu + x), x, mu^2 / x)
}

# --- family operations registry ----------------------------------------------
# Fields per family:
#   mean(eta, phi)                inverse link, clamped; phi reaches entries
#                                 whose response mean depends on the dispersion
#                                 (lognormal), the rest swallow it via `...`
#   loglik(eta, y, n_trials, phi) elementwise log-density (full normalizer)
#   score(eta, y, n_trials, phi)  elementwise d log-lik / d eta
#   weight(eta, n_trials, phi)    Laplace/IRLS working weight (no y dependence)
#   sample(eta, n_trials, phi)    one y draw per element (posterior predictive)
#   variance(eta, n_trials, phi)  response variance Var(y | eta), elementwise
#   response_mean(eta, n_trials)  E[y | eta] on the response scale; only where
#                                 it differs from mean() (trial-scaled families)
# Families in .PHI2_FAMILIES additionally take a trailing `phi2` (the second
# dispersion channel: Student-t df); the wrappers pass it only when supplied.

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
    },
    sample = function(eta, n_trials, phi) {
      n <- if (!is.null(n_trials)) n_trials else rep(1, length(eta))
      stats::rbinom(length(eta), size = n, prob = .mean_binomial(eta))
    },
    variance = function(eta, n_trials, phi) {
      n <- if (!is.null(n_trials)) n_trials else rep(1, length(eta))
      mu <- .mean_binomial(eta)
      n * mu * (1 - mu)
    },
    response_mean = function(eta, n_trials) {
      n <- if (!is.null(n_trials)) n_trials else rep(1, length(eta))
      n * .mean_binomial(eta)
    }
  ),

  poisson = list(
    mean = .mean_log,
    loglik = function(eta, y, n_trials, phi) {
      mu <- .mean_log(eta)
      y * log(mu) - mu - lgamma(y + 1)
    },
    score = function(eta, y, n_trials, phi) y - .mean_log(eta),
    weight = function(eta, n_trials, phi) .mean_log(eta),
    sample = function(eta, n_trials, phi) {
      stats::rpois(length(eta), .mean_log(eta))
    },
    variance = function(eta, n_trials, phi) .mean_log(eta)
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
    },
    sample = function(eta, n_trials, phi) {
      stats::rnbinom(length(eta), size = phi, mu = .mean_log(eta))
    },
    variance = function(eta, n_trials, phi) {
      mu <- .mean_log(eta)
      mu + mu^2 / phi
    }
  ),

  # Lognormal (identity link on the log scale), phi = log-scale VARIANCE
  # (mirroring the gaussian entry; the SD-parameterized compiled kernels get
  # sqrt(phi) at the front-door boundary). log(y) ~ N(eta, phi); the -log(y)
  # Jacobian is part of the density. E[y] = exp(eta + phi/2).
  lognormal = list(
    mean = function(eta, phi = 1.0, ...) exp(eta + phi / 2),
    loglik = function(eta, y, n_trials, phi) {
      ly <- log(y)
      -ly - 0.5 * log(2 * pi * phi) - (ly - eta)^2 / (2 * phi)
    },
    score = function(eta, y, n_trials, phi) (log(y) - eta) / phi,
    weight = function(eta, n_trials, phi) rep(1 / phi, length(eta)),
    sample = function(eta, n_trials, phi) {
      stats::rlnorm(length(eta), meanlog = eta, sdlog = sqrt(phi))
    },
    variance = function(eta, n_trials, phi) {
      (exp(phi) - 1) * exp(2 * eta + phi)
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
    weight = function(eta, n_trials, phi) rep(1 / phi, length(eta)),
    sample = function(eta, n_trials, phi) {
      stats::rnorm(length(eta), mean = eta, sd = sqrt(phi))
    },
    variance = function(eta, n_trials, phi) rep(phi, length(eta))
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
    },
    sample = function(eta, n_trials, phi) {
      mu <- .mean_beta(eta)
      stats::rbeta(length(eta), mu * phi, (1 - mu) * phi)
    },
    variance = function(eta, n_trials, phi) {
      mu <- .mean_beta(eta)
      mu * (1 - mu) / (phi + 1)
    }
  ),

  # Gamma (log link), phi = shape. Mean mu = exp(eta), variance mu^2 / phi.
  # Mirrors laplace_family_link.h so the R-closure backends agree with the C++
  # Laplace / ModelData-sampler path.
  gamma = list(
    mean = .mean_log,
    loglik = function(eta, y, n_trials, phi) {
      mu <- .mean_log(eta)
      phi * log(phi) - lgamma(phi) + (phi - 1) * log(y) -
        phi * log(mu) - phi * y / mu
    },
    score = function(eta, y, n_trials, phi) {
      mu <- .mean_log(eta)
      phi * (y - mu) / mu
    },
    weight = function(eta, n_trials, phi) rep(phi, length(eta)),
    sample = function(eta, n_trials, phi) {
      stats::rgamma(length(eta), shape = phi, rate = phi / .mean_log(eta))
    },
    variance = function(eta, n_trials, phi) .mean_log(eta)^2 / phi
  ),

  # Inverse Gaussian (log link), phi = dispersion. Mean mu = exp(eta),
  # variance phi * mu^3.
  inverse_gaussian = list(
    mean = .mean_log,
    loglik = function(eta, y, n_trials, phi) {
      mu <- .mean_log(eta)
      -0.5 * log(2 * pi * phi * y^3) - (y - mu)^2 / (2 * phi * mu^2 * y)
    },
    score = function(eta, y, n_trials, phi) {
      mu <- .mean_log(eta)
      (y - mu) / (phi * mu^2)
    },
    weight = function(eta, n_trials, phi) {
      mu <- .mean_log(eta)
      1 / (phi * mu)
    },
    sample = function(eta, n_trials, phi) {
      .rinvgauss(length(eta), mu = .mean_log(eta), lambda = 1 / phi)
    },
    variance = function(eta, n_trials, phi) phi * .mean_log(eta)^3
  ),

  # Beta-binomial (logit link), phi = precision (a + b), n_trials = n.
  # mu = P(success) = a / (a + b); a = mu*phi, b = (1-mu)*phi. Overdispersed
  # binomial: Var(y) = n mu(1-mu) [1 + (n-1)/(phi+1)]. The score is exact; the
  # working weight is the moment-based Fisher weight n mu(1-mu) / D with the
  # overdispersion factor D (matching grad_hess_for_family in the C++ path).
  beta_binomial = list(
    mean = .mean_beta,
    loglik = function(eta, y, n_trials, phi) {
      n  <- if (!is.null(n_trials)) n_trials else rep(1, length(eta))
      mu <- .mean_beta(eta); a <- mu * phi; b <- (1 - mu) * phi
      lchoose(n, y) + lgamma(y + a) + lgamma(n - y + b) - lgamma(n + a + b) -
        lgamma(a) - lgamma(b) + lgamma(a + b)
    },
    score = function(eta, y, n_trials, phi) {
      n  <- if (!is.null(n_trials)) n_trials else rep(1, length(eta))
      mu <- .mean_beta(eta); a <- mu * phi; b <- (1 - mu) * phi
      dmu <- mu * (1 - mu)
      phi * (digamma(y + a) - digamma(a) - digamma(n - y + b) + digamma(b)) * dmu
    },
    weight = function(eta, n_trials, phi) {
      n  <- if (!is.null(n_trials)) n_trials else rep(1, length(eta))
      mu <- .mean_beta(eta); D <- 1 + (n - 1) / (phi + 1)
      n * mu * (1 - mu) / D
    },
    sample = function(eta, n_trials, phi) {
      n  <- if (!is.null(n_trials)) n_trials else rep(1, length(eta))
      mu <- .mean_beta(eta)
      p  <- stats::rbeta(length(eta), mu * phi, (1 - mu) * phi)
      stats::rbinom(length(eta), size = n, prob = p)
    },
    variance = function(eta, n_trials, phi) {
      n  <- if (!is.null(n_trials)) n_trials else rep(1, length(eta))
      mu <- .mean_beta(eta)
      n * mu * (1 - mu) * (1 + (n - 1) / (phi + 1))
    },
    response_mean = function(eta, n_trials) {
      n <- if (!is.null(n_trials)) n_trials else rep(1, length(eta))
      n * .mean_beta(eta)
    }
  ),

  # Tweedie compound Poisson-gamma (log link), 1 < p < 2: phi = dispersion,
  # phi2 = power p (REQUIRED -- no silent default). Mean mu = exp(eta),
  # variance phi * mu^p; exact zero mass exp(-lambda) with lambda =
  # mu^(2-p) / (phi (2-p)). The positive-y density is the compound
  # Poisson-gamma series (Dunn & Smyth 2005): N ~ Poisson(lambda) gamma
  # events with shape a = (2-p)/(p-1) and rate b = mu^(1-p) / (phi (p-1)),
  # log-sum-exp'd over N around its dominating term. The score and working
  # weight are the exponential-dispersion closed forms ((y - mu)/(phi V(mu))
  # through the log link), no series needed.
  tweedie = list(
    mean = .mean_log,
    loglik = function(eta, y, n_trials, phi, phi2 = NULL) {
      p <- .tweedie_power(phi2)
      .tweedie_loglik(.mean_log(eta), y, phi, p)
    },
    score = function(eta, y, n_trials, phi, phi2 = NULL) {
      p <- .tweedie_power(phi2)
      mu <- .mean_log(eta)
      (y - mu) / (phi * mu^(p - 1))
    },
    weight = function(eta, n_trials, phi, phi2 = NULL) {
      p <- .tweedie_power(phi2)
      .mean_log(eta)^(2 - p) / phi
    },
    sample = function(eta, n_trials, phi, phi2 = NULL) {
      p  <- .tweedie_power(phi2)
      mu <- .mean_log(eta)
      lam <- mu^(2 - p) / (phi * (2 - p))
      a   <- (2 - p) / (p - 1)
      b   <- mu^(1 - p) / (phi * (p - 1))
      n_ev <- stats::rpois(length(eta), lam)
      out <- numeric(length(eta))
      pos <- n_ev > 0
      out[pos] <- stats::rgamma(sum(pos), shape = n_ev[pos] * a,
                                rate = b[pos])
      out
    },
    variance = function(eta, n_trials, phi, phi2 = NULL) {
      phi * .mean_log(eta)^.tweedie_power(phi2)
    }
  ),

  # Student-t location-scale (identity link), phi = scale, df = phi2 through
  # the second dispersion channel (default .STUDENT_T_DF, the robust drop-in).
  # The score is exact and the working weight is the constant Fisher
  # information (nu+1)/((nu+3) phi^2). Mirrors the C++ `t` branch.
  t = list(
    mean = .mean_identity,
    loglik = function(eta, y, n_trials, phi, phi2 = NULL) {
      nu <- phi2 %||% .STUDENT_T_DF; r <- (y - eta) / phi
      lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(nu * pi * phi^2) -
        0.5 * (nu + 1) * log1p(r^2 / nu)
    },
    score = function(eta, y, n_trials, phi, phi2 = NULL) {
      nu <- phi2 %||% .STUDENT_T_DF; resid <- y - eta
      (nu + 1) * resid / (nu * phi^2 + resid^2)
    },
    weight = function(eta, n_trials, phi, phi2 = NULL) {
      nu <- phi2 %||% .STUDENT_T_DF
      rep((nu + 1) / ((nu + 3) * phi^2), length(eta))
    },
    sample = function(eta, n_trials, phi, phi2 = NULL) {
      eta + phi * stats::rt(length(eta), df = phi2 %||% .STUDENT_T_DF)
    },
    variance = function(eta, n_trials, phi, phi2 = NULL) {
      nu <- phi2 %||% .STUDENT_T_DF
      rep(if (nu > 2) phi^2 * nu / (nu - 2) else Inf, length(eta))
    }
  )
)

# Families whose ops consume a second dispersion `phi2` (the Student-t degrees
# of freedom; the Tweedie power). Everything else rejects a supplied phi2
# loudly rather than silently ignoring it.
.PHI2_FAMILIES <- c("t", "tweedie")

# The Tweedie power. Required (a silently defaulted variance power would be a
# statistical decision the user never made) and restricted to the compound
# Poisson-gamma range.
.tweedie_power <- function(phi2) {
  if (is.null(phi2)) {
    stop("family = 'tweedie' requires `phi2` (the variance power p, ",
         "1 < p < 2).", call. = FALSE)
  }
  p <- as.numeric(phi2)
  if (!is.finite(p) || p <= 1 || p >= 2) {
    stop("The Tweedie power `phi2` must lie strictly in (1, 2); got ",
         format(p), ".", call. = FALSE)
  }
  p
}

# Tweedie log-density at mean mu (vectorized over observations). y = 0 has the
# exact Poisson-zero mass; y > 0 is the compound Poisson-gamma series
# log-sum-exp'd over the event count N, expanding from the dominating index
# j_max = y^(2-p) / (phi (2-p)) (Dunn & Smyth 2005) until terms fall 37 nats
# below the running maximum.
.tweedie_loglik <- function(mu, y, phi, p) {
  mu  <- pmin(rep_len(mu, length(y)), 1e10)   # broadcast; clamp eta overflow
  lam <- mu^(2 - p) / (phi * (2 - p))
  a   <- (2 - p) / (p - 1)
  b   <- mu^(1 - p) / (phi * (p - 1))
  out <- numeric(length(y))
  out[y < 0] <- -Inf                       # outside the support
  zero <- y == 0
  out[zero] <- -lam[zero]
  for (i in which(y > 0)) {
    jmax <- y[i]^(2 - p) / (phi * (2 - p))
    out[i] <- .tweedie_logf_pos(y[i], lam[i], a, b[i], jmax)
  }
  out
}

.tweedie_logf_pos <- function(y, lam, a, b, jmax) {
  logterm <- function(n) {
    n * log(lam) - lgamma(n + 1) + n * a * log(b) +
      (n * a - 1) * log(y) - lgamma(n * a)
  }
  n0 <- max(1L, as.integer(round(jmax)))
  lt <- logterm(n0)
  lmax <- lt; terms <- lt
  # Expand upward then downward until 37 nats below the peak.
  n <- n0
  repeat {
    n <- n + 1L
    lt <- logterm(n)
    terms <- c(terms, lt)
    lmax <- max(lmax, lt)
    if (lt < lmax - 37) break
  }
  n <- n0
  while (n > 1L) {
    n <- n - 1L
    lt <- logterm(n)
    terms <- c(terms, lt)
    lmax <- max(lmax, lt)
    if (lt < lmax - 37) break
  }
  lmax + log(sum(exp(terms - lmax))) - lam - b * y
}

# Default degrees of freedom for the Student-t family (matches the C++
# kStudentTDf), used when no `phi2` is supplied.
.STUDENT_T_DF <- 4


#' Supported R-level family names.
#' @keywords internal
family_names <- function() names(.FAMILY_OPS)


# Families carrying a dispersion / precision parameter `phi` that enters the
# likelihood, score, and working weight. A non-positive `phi` makes the variance
# / log-density singular (gaussian: r^2 / (2 phi^2); gamma: lgamma(phi)), so it
# must be validated at the front door rather than flowing into a NaN log-lik
# (gcol33/tulpa#104). The `*_<link>` suffix forms (e.g. "gamma_inverse") share
# the base family's dispersion, so membership is tested on the prefix.
.PHI_FAMILIES <- c("gaussian", "lognormal", "gamma", "neg_binomial_2",
                   "negative_binomial", "inverse_gaussian", "beta",
                   "beta_binomial", "t", "tweedie",
                   "interval_gaussian", "truncated_gaussian")

# Count families whose likelihood is defined only at non-negative integer `y`.
# The C++ kernels cast the response to `int` (laplace_family_link.h /
# laplace_likelihoods.h), silently flooring a continuous response into a biased
# log-likelihood, so a non-integer `y` is rejected at the front door.
.COUNT_FAMILIES <- c("poisson", "binomial", "neg_binomial_2", "negative_binomial",
                     "beta_binomial")

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


#' Reject non-finite fitting inputs.
#'
#' The design is built with `na.action = na.pass`, so a missing predictor or
#' response survives into `X` / `y`. tulpa() does not drop incomplete cases, so
#' an NA/NaN/Inf would propagate into the C++ kernels as a NaN estimate. Fail
#' loudly with the offending row instead.
#' @keywords internal
.assert_finite_model_inputs <- function(X, y) {
  if (!is.null(X)) {
    ok_row <- if (is.matrix(X)) .all_finite_rows(X) else is.finite(X)
    if (!all(ok_row)) {
      bad <- which(!ok_row)
      stop(sprintf(paste0(
        "Non-finite value(s) in the model matrix (%d row(s), first at row %d). ",
        "tulpa() does not drop incomplete cases; remove or impute NA/NaN/Inf ",
        "in the predictors before fitting."), length(bad), bad[1L]),
        call. = FALSE)
    }
  }
  if (!is.null(y)) {
    bad <- which(!is.finite(as.numeric(y)))
    if (length(bad)) {
      stop(sprintf(paste0(
        "Non-finite value(s) in the response (%d, first at row %d). Remove or ",
        "impute NA/NaN/Inf in the response before fitting."),
        length(bad), bad[1L]), call. = FALSE)
    }
  }
  invisible(TRUE)
}

# Row-wise all-finite test for a numeric matrix.
.all_finite_rows <- function(X) {
  fin <- is.finite(X)
  dim(fin) <- dim(X)
  .rowSums(fin, nrow(X), ncol(X)) == ncol(X)
}


#' Look up a family's operation set, with a clear error for unknown families.
#' @keywords internal
.family_ops <- function(family) {
  ops <- .FAMILY_OPS[[family]]
  if (is.null(ops)) {
    .family_or_stop(family)
  }
  ops
}

# Validate a family identifier against the registry (the one unknown-family
# error, shared by every front door); returns the identifier invisibly.
#' @keywords internal
.family_or_stop <- function(family) {
  if (!is.character(family) || length(family) != 1L ||
      is.null(.FAMILY_OPS[[family]])) {
    stop(sprintf(
      "Unknown family '%s'. Supported: %s.",
      as.character(family)[1L], paste(family_names(), collapse = ", ")
    ), call. = FALSE)
  }
  invisible(family)
}


# Resolve a supplied second dispersion against the family: NULL passes, a
# phi2-aware family gets it, anything else errors rather than silently
# ignoring a parameter the user thought they set.
.phi2_or_stop <- function(family, phi2) {
  if (is.null(phi2)) return(NULL)
  if (!.family_base(family) %in% .PHI2_FAMILIES) {
    stop(sprintf(
      "Family '%s' takes no second dispersion `phi2` (supported: %s).",
      family, paste(.PHI2_FAMILIES, collapse = ", ")), call. = FALSE)
  }
  if (!is.numeric(phi2) || length(phi2) != 1L || !is.finite(phi2) || phi2 <= 0) {
    stop("`phi2` must be a positive finite scalar.", call. = FALSE)
  }
  phi2
}


#' Inverse-link mean for a family. `phi` reaches entries whose response mean
#' depends on the dispersion (lognormal); the others ignore it via `...`.
#' @keywords internal
family_mean <- function(eta, family, phi = 1.0) {
  .family_ops(family)$mean(eta, phi)
}


#' Elementwise log-likelihood for a family.
#' @keywords internal
family_loglik <- function(eta, y, family, n_trials = NULL, phi = 1.0,
                          phi2 = NULL) {
  ops <- .family_ops(family)
  if (is.null(.phi2_or_stop(family, phi2))) {
    return(ops$loglik(eta, y, n_trials, phi))
  }
  ops$loglik(eta, y, n_trials, phi, phi2)
}


#' Score (d log-likelihood / d eta), elementwise.
#' @keywords internal
family_score_eta <- function(eta, y, family, n_trials = NULL, phi = 1.0,
                             phi2 = NULL) {
  ops <- .family_ops(family)
  if (is.null(.phi2_or_stop(family, phi2))) {
    return(ops$score(eta, y, n_trials, phi))
  }
  ops$score(eta, y, n_trials, phi, phi2)
}


#' Laplace/IRLS working weight (-d^2 log-lik / d eta^2), elementwise.
#' @keywords internal
family_weight <- function(eta, family, n_trials = NULL, phi = 1.0,
                          phi2 = NULL) {
  ops <- .family_ops(family)
  if (is.null(.phi2_or_stop(family, phi2))) {
    return(ops$weight(eta, n_trials, phi))
  }
  ops$weight(eta, n_trials, phi, phi2)
}


#' One response draw per element of `eta` (posterior predictive), elementwise.
#' @keywords internal
family_sample <- function(eta, family, n_trials = NULL, phi = 1.0,
                          phi2 = NULL) {
  ops <- .family_ops(family)
  if (is.null(ops$sample)) {
    stop(sprintf("Family '%s' has no sampling function registered.", family),
         call. = FALSE)
  }
  if (is.null(.phi2_or_stop(family, phi2))) {
    return(ops$sample(eta, n_trials, phi))
  }
  ops$sample(eta, n_trials, phi, phi2)
}


#' Response variance Var(y | eta) for a family, elementwise.
#' @keywords internal
family_variance <- function(eta, family, n_trials = NULL, phi = 1.0,
                            phi2 = NULL) {
  ops <- .family_ops(family)
  if (is.null(ops$variance)) {
    stop(sprintf("Family '%s' has no variance function registered.", family),
         call. = FALSE)
  }
  if (is.null(.phi2_or_stop(family, phi2))) {
    return(ops$variance(eta, n_trials, phi))
  }
  ops$variance(eta, n_trials, phi, phi2)
}


#' Response-scale mean of y given eta for a family (trial-scaled where relevant).
#' @keywords internal
family_response_mean <- function(eta, family, n_trials = NULL, phi = 1.0) {
  ops <- .family_ops(family)
  if (!is.null(ops$response_mean)) return(ops$response_mean(eta, n_trials))
  ops$mean(eta, phi)
}
