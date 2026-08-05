# ============================================================================
# Skew-normal utilities for simplified Laplace approximation (SLA).
#
# Downstream model packages (tulpaObs, tulpaglmm) build per-coefficient
# marginal posteriors with three cumulants (mu, sigma, gamma) and need to
# convert them into a skew-normal (xi, omega, alpha) parameterisation for
# quantile / CDF evaluation. All three helpers are pure R and depend only
# on base / stats.
#
# Reference: Azzalini (1985) "A class of distributions which includes the
# normal ones". Scand. J. Statist. 12: 171-178. Moment formulas as given
# in Azzalini & Capitanio (2014, Ch. 2).
# ============================================================================


# Maximum |skewness| attainable by a univariate skew-normal. Derived from
# delta = 1 in gamma = (4 - pi)/2 * (delta sqrt(2/pi))^3 / (1 - 2 delta^2/pi)^{3/2}.
.SN_GAMMA_MAX <- ((4 - pi) / 2) * (2 / (pi - 2))^(3 / 2)


# ----------------------------------------------------------------------------
# Owen's T function via base R numerical quadrature.
#
#   T(h, a) = (1 / 2 pi) * integral_0^a exp(-h^2 (1 + x^2) / 2) / (1 + x^2) dx
#
# Vectorised over (h, a) by recycling. Integrand is smooth and decays
# exponentially in h; stats::integrate() with rel.tol = 1e-10 gives ~12
# digits across the SLA-relevant range. No `sn`-package dependency.
# ----------------------------------------------------------------------------
.owens_T <- function(h, a) {
  n <- max(length(h), length(a))
  h <- rep_len(as.numeric(h), n)
  a <- rep_len(as.numeric(a), n)
  out <- numeric(n)
  for (i in seq_len(n)) {
    hi <- h[i]; ai <- a[i]
    if (!is.finite(hi) || !is.finite(ai) || ai == 0) {
      out[i] <- 0
      next
    }
    if (abs(hi) > 38) {
      # exp(-h^2/2) underflows; T -> 0
      out[i] <- 0
      next
    }
    h2 <- hi * hi
    integrand <- function(x) exp(-h2 * (1 + x * x) / 2) / (1 + x * x)
    val <- tryCatch(
      stats::integrate(integrand, 0, abs(ai), rel.tol = 1e-10,
                       stop.on.error = FALSE)$value,
      error = function(e) NA_real_
    )
    out[i] <- sign(ai) * val / (2 * pi)
  }
  out
}


#' Match three cumulants to a skew-normal parameterisation
#'
#' Inverts the skew-normal moment formulas to convert
#' \code{(mu, sigma, gamma)} into \code{(xi, omega, alpha)} parameters.
#' Returns \code{NULL} with a warning when \code{|gamma|} exceeds the
#' skew-normal ceiling (~0.9953) -- in that regime the third cumulant
#' cannot be matched by any skew-normal and the caller should fall back
#' to direct-quadrature quantiles.
#'
#' @param mu Posterior mean (numeric, length 1).
#' @param sigma Posterior standard deviation (positive numeric, length 1).
#' @param gamma Posterior skewness (numeric, length 1).
#' @return Named list with elements \code{xi}, \code{omega}, \code{alpha};
#'   or \code{NULL} if \code{|gamma| >= .SN_GAMMA_MAX}.
#' @references Azzalini, A. (1985). A class of distributions which
#'   includes the normal ones. \emph{Scand. J. Statist.} \strong{12}: 171-178.
#' @keywords internal
#' @export
sn_match <- function(mu, sigma, gamma) {
  if (!is.numeric(sigma) || length(sigma) != 1L || sigma <= 0) {
    stop("`sigma` must be a single positive numeric.", call. = FALSE)
  }
  if (!is.numeric(gamma) || length(gamma) != 1L || !is.finite(gamma)) {
    stop("`gamma` must be a single finite numeric.", call. = FALSE)
  }
  if (abs(gamma) >= .SN_GAMMA_MAX) {
    warning(
      "|gamma| = ", format(abs(gamma), digits = 5),
      " exceeds skew-normal ceiling (", format(.SN_GAMMA_MAX, digits = 5),
      "); returning NULL. Use direct-quadrature quantiles instead.",
      call. = FALSE
    )
    return(NULL)
  }

  c1       <- ((4 - pi) / 2)^(2 / 3)
  abs_g23  <- abs(gamma)^(2 / 3)
  delta_sq <- (pi / 2) * abs_g23 / (abs_g23 + c1)
  delta    <- sign(gamma) * sqrt(delta_sq)
  omega    <- sigma / sqrt(1 - 2 * delta_sq / pi)
  xi       <- mu - omega * delta * sqrt(2 / pi)
  alpha    <- delta / sqrt(1 - delta_sq)

  list(xi = xi, omega = omega, alpha = alpha)
}


#' Skew-normal CDF
#'
#' Cumulative distribution function of the univariate skew-normal,
#' \deqn{F(q; \xi, \omega, \alpha) = \Phi(z) - 2 T(z, \alpha),}
#' where \eqn{z = (q - \xi) / \omega} and \eqn{T} is Owen's T function.
#' Owen's T is evaluated by base R numerical quadrature; no extra
#' dependencies are required.
#'
#' @param q Numeric vector of quantiles.
#' @param sn Skew-normal parameter list from [sn_match()] with elements
#'   \code{xi}, \code{omega}, \code{alpha}.
#' @return Numeric vector of CDF values, same length as \code{q}.
#' @keywords internal
#' @export
sn_cdf <- function(q, sn) {
  .check_sn(sn)
  z <- (as.numeric(q) - sn$xi) / sn$omega
  stats::pnorm(z) - 2 * .owens_T(z, sn$alpha)
}


#' Skew-normal quantile
#'
#' Inverse of the skew-normal CDF via Newton iteration on
#' [sn_cdf()] with a Brent-style bracket fallback when Newton fails to
#' converge.
#'
#' @param p Numeric vector of probabilities in \eqn{[0, 1]}.
#' @param sn Skew-normal parameter list from [sn_match()].
#' @param tol Absolute tolerance on the CDF residual (default \code{1e-10}).
#' @param max_iter Maximum Newton iterations (default \code{60}).
#' @return Numeric vector of quantiles, same length as \code{p}.
#' @keywords internal
#' @export
sn_quantile <- function(p, sn, tol = 1e-10, max_iter = 60L) {
  .check_sn(sn)
  p <- as.numeric(p)
  n <- length(p)
  out <- numeric(n)

  for (i in seq_len(n)) {
    pi_ <- p[i]
    if (!is.finite(pi_) || pi_ < 0 || pi_ > 1) {
      out[i] <- NaN
      next
    }
    if (pi_ == 0) { out[i] <- -Inf; next }
    if (pi_ == 1) { out[i] <-  Inf; next }

    # Newton init from the matching normal quantile of equal mean/sd.
    mu    <- sn$xi + sn$omega * (sn$alpha / sqrt(1 + sn$alpha^2)) * sqrt(2 / pi)
    sigma <- sn$omega * sqrt(1 - 2 * (sn$alpha / sqrt(1 + sn$alpha^2))^2 / pi)
    q <- mu + sigma * stats::qnorm(pi_)

    converged <- FALSE
    for (iter in seq_len(max_iter)) {
      z   <- (q - sn$xi) / sn$omega
      F_q <- stats::pnorm(z) - 2 * .owens_T(z, sn$alpha)
      f_q <- (2 / sn$omega) * stats::dnorm(z) * stats::pnorm(sn$alpha * z)
      resid <- F_q - pi_
      if (abs(resid) < tol) { converged <- TRUE; break }
      if (f_q < .Machine$double.eps) break  # density underflow -> fall through
      q <- q - resid / f_q
    }

    if (!converged) {
      # Bracket fallback: expand outward until sn_cdf brackets pi_.
      lo <- mu - 12 * sigma
      hi <- mu + 12 * sigma
      F_lo <- sn_cdf(lo, sn); F_hi <- sn_cdf(hi, sn)
      tries <- 0L
      while (F_lo > pi_ && tries < 8L) { lo <- lo - 12 * sigma; F_lo <- sn_cdf(lo, sn); tries <- tries + 1L }
      tries <- 0L
      while (F_hi < pi_ && tries < 8L) { hi <- hi + 12 * sigma; F_hi <- sn_cdf(hi, sn); tries <- tries + 1L }
      q <- tryCatch(
        stats::uniroot(function(x) sn_cdf(x, sn) - pi_,
                       lower = lo, upper = hi, tol = tol)$root,
        error = function(e) NA_real_
      )
    }

    out[i] <- q
  }
  out
}


.check_sn <- function(sn) {
  if (!is.list(sn) ||
      !all(c("xi", "omega", "alpha") %in% names(sn)) ||
      !is.numeric(sn$xi) || !is.numeric(sn$omega) || !is.numeric(sn$alpha) ||
      sn$omega <= 0) {
    stop("`sn` must be a list with numeric scalars xi, omega (>0), alpha.",
         call. = FALSE)
  }
  invisible(TRUE)
}


# ============================================================================
# Skew-normal as an IMPORTANCE-PROPOSAL family (gcol33/tulpa#276).
#
# The outer Pareto-k-hat scores the hyperparameter posterior against a Gaussian
# proposal. A variance-component marginal stays right-skewed even after the log
# unconstraining transform, and a SYMMETRIC proposal against a skewed target has
# a heavy importance-ratio tail whatever the fit's quality -- so the k-hat reads
# unreliable for a reason living in the proposal, not in the integration. The
# skew-normal is the smallest family that removes that particular mismatch: it
# nests the Gaussian at zero skewness and keeps GAUSSIAN TAILS on both sides, so
# it absorbs skew but cannot absorb a genuinely heavy tail. A k-hat that stays
# high against it is a real signal, not a proposal artefact.
#
# The d-dimensional proposal is the PRODUCT of d univariate skew-normals, one
# per coordinate. In the whitened coordinate the diagnostic works in, the
# target's Gaussian part is isotropic and its leading Edgeworth correction is
# per-axis cubic (the cross terms cancel from the third cumulant to leading
# order -- see .joint_pareto_mode_skew), so the independent product is the right
# leading-order family there. It also sidesteps the multivariate skew-normal's
# single-skewing-latent constraint (delta' Omega_bar^-1 delta < 1), which caps
# the joint skewness an Azzalini-Dalla Valle vector can carry across d axes.
#
# Sampling and density are separate from sn_cdf() / sn_quantile() above by
# necessity: those route through Owen's T by stats::integrate() per point, which
# is right for a handful of reported quantiles and unusable for the hundreds of
# proposal draws scored here.
# ============================================================================

# Safety margin on the skewness clamp. At |delta| = 1 the shape `alpha` is
# infinite and the density degenerates to a half-normal with a hard edge -- a
# poor importance proposal, since a target draw on the truncated side would get
# zero proposal mass. Clamping a little inside sn_match()'s ceiling keeps
# `alpha` finite and both tails alive.
.SN_PROP_CLAMP <- 0.95

# Per-coordinate direct parameters of the product proposal matching the supplied
# centred moments. `mu` / `sd` / `skew` are length-d numerics. Each coordinate
# goes through sn_match() -- the single source of truth for the cumulant
# inversion -- after clamping the skewness inside the representable ceiling, so
# an over-skewed axis yields the most-skewed proposal available rather than
# sn_match()'s NULL (the proposal only has to COVER the target, not match it).
# Returns list(xi, omega, alpha, delta), each length d, or NULL when any
# coordinate's scale is unusable.
.sn_prop_from_moments <- function(mu, sd, skew) {
  d <- length(mu)
  if (length(sd) != d || length(skew) != d) return(NULL)
  if (any(!is.finite(mu)) || any(!is.finite(sd)) || any(sd <= 0)) return(NULL)
  skew <- as.numeric(skew)
  skew[!is.finite(skew)] <- 0
  lim  <- .SN_PROP_CLAMP * .SN_GAMMA_MAX
  skew <- pmax(pmin(skew, lim), -lim)

  ps <- lapply(seq_len(d), function(j) sn_match(mu[j], sd[j], skew[j]))
  if (any(vapply(ps, is.null, logical(1)))) return(NULL)
  alpha <- vapply(ps, function(p) p$alpha, numeric(1))
  out <- list(xi    = vapply(ps, function(p) p$xi,    numeric(1)),
              omega = vapply(ps, function(p) p$omega, numeric(1)),
              alpha = alpha,
              delta = alpha / sqrt(1 + alpha^2))
  if (any(!is.finite(unlist(out, use.names = FALSE)))) return(NULL)
  out
}

# Draw `n` points from the product proposal, by the Azzalini stochastic
# representation Z = delta |U0| + sqrt(1 - delta^2) U1 with U0, U1 independent
# standard normals (so Z ~ SN(0, 1, alpha)), then X = xi + omega Z. Returns an
# [n x d] matrix and consumes 2 * n * d variates from the caller's RNG stream.
.sn_prop_rand <- function(n, dp) {
  d  <- length(dp$xi)
  U0 <- matrix(abs(stats::rnorm(n * d)), n, d)
  U1 <- matrix(stats::rnorm(n * d), n, d)
  Z  <- sweep(U0, 2L, dp$delta, `*`) +
        sweep(U1, 2L, sqrt(pmax(1 - dp$delta^2, 0)), `*`)
  sweep(sweep(Z, 2L, dp$omega, `*`), 2L, dp$xi, `+`)
}

# Log-density of the product proposal at the rows of `U` ([n x d]), summed over
# coordinates. `pnorm(log.p = TRUE)` keeps the truncated side finite instead of
# underflowing to -Inf, which would send an importance ratio there to +Inf and
# manufacture a spurious tail. Returns a length-n numeric.
.sn_prop_logpdf <- function(U, dp) {
  Z <- sweep(sweep(U, 2L, dp$xi, `-`), 2L, dp$omega, `/`)
  lp <- stats::dnorm(Z, log = TRUE) +
        stats::pnorm(sweep(Z, 2L, dp$alpha, `*`), log.p = TRUE)
  as.numeric(rowSums(lp) + sum(log(2) - log(dp$omega)))
}
