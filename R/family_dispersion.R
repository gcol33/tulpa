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
# Three per-observation derivatives are not by themselves enough for the
# ASSEMBLED gradient to be exact, and which extra ingredient a family needs is
# decided by one quantity: the mode-motion channel q_eta = (dW/deta) s, the only
# route by which dx_hat/dphi reaches dm/dphi. Two ways for it to be right:
#
#   * dW/deta is identically zero (gaussian, lognormal, gamma, t: the working
#     weight does not move with eta), so dx_hat/dphi multiplies zero and the
#     inverse it was solved on cannot matter;
#   * dW/deta is non-zero and the solve is on the TRUE curvature, which needs
#     the family's observed curvature registered in C++ so `Hinv_mode` is
#     H_true^-1 rather than H^-1 (neg_binomial_2, neg_binomial_1,
#     truncated_neg_binomial_2, beta, inverse_gaussian, beta_binomial, tweedie).
#
# A family failing both is off by the mode-motion term -- ~1e-4 relative where
# the registered ones reach ~1e-9, small enough to converge and look plausible.
# test-family-dispersion.R asserts the disjunction for every registered entry
# rather than leaving it as a comment to keep in sync by hand.
#
# Families absent from this list have no free dispersion at all (poisson,
# binomial, truncated_poisson); `.family_dphi()` returns NULL for them and the
# caller refuses to estimate rather than reporting a fixed phi as an estimate.
#
# Every entry takes the same five arguments, including the second dispersion
# channel `phi2` that only `t` (degrees of freedom) and `tweedie` (variance
# power) read. .FAMILY_OPS varies its arity per family and pays for it with a
# branch at each call site; this registry has one consumer
# (.laplace_phi_fields), so one signature is cheaper than the dispatch.
# ============================================================================

# gaussian and lognormal are one normal density read on two scales: the
# lognormal is N(log y; eta, phi) plus a -log(y) Jacobian that carries no phi.
# So both differentiate the same three expressions at their own residual, and
# the pair cannot drift.
.normal_scale_dphi <- function(resid) {
  force(resid)
  list(
    dloglik = function(eta, y, n_trials, phi, phi2 = NULL) {
      resid(eta, y)^2 / (2 * phi^2) - 1 / (2 * phi)
    },
    dscore = function(eta, y, n_trials, phi, phi2 = NULL) {
      -resid(eta, y) / phi^2
    },
    dweight = function(eta, y, n_trials, phi, phi2 = NULL) {
      rep(-1 / phi^2, length(eta))
    }
  )
}

.normal_scale_dphi2 <- function(resid) {
  force(resid)
  list(
    dloglik2 = function(eta, y, n_trials, phi, phi2 = NULL) {
      0.5 / phi^2 - resid(eta, y)^2 / phi^3
    },
    dscore2 = function(eta, y, n_trials, phi, phi2 = NULL) {
      2 * resid(eta, y) / phi^3
    },
    dweight2 = function(eta, y, n_trials, phi, phi2 = NULL) {
      rep(2 / phi^3, length(eta))
    },
    dweight_deta = function(eta, y, n_trials, phi, phi2 = NULL) {
      rep(0, length(eta))
    }
  )
}

# The Student-t degrees of freedom, matching how .FAMILY_OPS$t reads them. The
# gradient path carries an absent second dispersion as NA rather than NULL, so
# both spellings fall through to the default.
.t_df <- function(phi2) {
  if (is.null(phi2) || !is.finite(phi2)) .STUDENT_T_DF else as.numeric(phi2)
}


# The zero-truncated negative binomial's shape terms, in the parameterization
# every truncated derivative below is written in.
#
# The retained mass is p = 1 - e^-a with a = phi log1p(mu/phi), and the working
# weight Newton builds H from is a function of the SHAPE alone:
#
#   w = f(a, a_e) = a_e / p - q a_e^2 / p^2,   a_e = da/deta
#
# so every weight derivative is a chain rule over (f_a, f_ae, f_aa, f_a_ae,
# f_aeae) times the shape's own derivatives, and the partials of f are the ones
# curvature_deta2_for_family already carries for this family
# (src/laplace_family_curvature.h). Deriving dweight, its second phi-derivative
# and the mixed eta-phi term from one place is what keeps three expressions that
# must agree from being written three times.
#
# Shape derivatives, with S = phi + mu:
#   a_e   = da/deta   = phi mu / S            a_ee   = phi^2 mu / S^2
#   a_p   = da/dphi   = log1p(mu/phi) - mu/S  a_pp   = -mu^2 / (phi S^2)
#   a_e_p = d2a/(deta dphi) = mu^2 / S^2      a_ee_p = 2 mu^2 phi / S^3
#   a_e_pp = d2 a_e / dphi2 = -2 mu^2 / S^3
.tnb2_shape <- function(mu, phi) {
  S  <- phi + mu
  a  <- phi * log1p(mu / phi)
  q  <- exp(-a)
  p  <- -expm1(-a)
  ps <- pmax(p, 1e-300)
  p2 <- ps * ps; p3 <- p2 * ps; p4 <- p3 * ps
  a_e <- phi * mu / S
  f_a     <- -a_e * q / p2 + a_e^2 * q * (ps + 2 * q) / p3
  f_ae    <- 1 / ps - 2 * q * a_e / p2
  f_aeae  <- -2 * q / p2
  f_a_ae  <- -q / p2 + 2 * a_e * q * (ps + 2 * q) / p3
  R_a     <- -q * (ps + 2 * q) / p3 - 2 * q * q * (2 * ps + 3 * q) / p4
  f_aa    <- a_e * q * (ps + 2 * q) / p3 + a_e^2 * R_a
  list(S = S, a = a, q = q, p = ps,
       a_e = a_e, a_ee = phi^2 * mu / S^2,
       a_p = log1p(mu / phi) - mu / S, a_pp = -mu^2 / (phi * S^2),
       a_e_p = mu^2 / S^2, a_ee_p = 2 * mu^2 * phi / S^3,
       a_e_pp = -2 * mu^2 / S^3,
       f_a = f_a, f_ae = f_ae, f_aa = f_aa, f_a_ae = f_a_ae, f_aeae = f_aeae)
}

#' @keywords internal
.FAMILY_DPHI <- list(

  # phi = size. mu = exp(eta); Var(y) = mu + mu^2 / phi.
  neg_binomial_2 = list(
    dloglik = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_log(eta)
      digamma(y + phi) - digamma(phi) + log(phi) - log(phi + mu) +
        1 - (phi + y) / (phi + mu)
    },
    dscore = function(eta, y, n_trials, phi, phi2 = NULL) {
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
    dweight = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_log(eta)
      mu * (y * (mu - phi) + 2 * phi * mu) / (mu + phi)^3
    }
  ),

  # phi = dispersion, mean mu = exp(eta), Var(y) = mu (1 + phi). The shape
  # r = mu / phi moves with BOTH the mean and the dispersion, so every
  # derivative below carries dr/dphi = -r/phi and the digamma ladder comes along.
  # Writing T = psi(y + r) - psi(r) - log1p(phi), the registry's own score is
  # s = r T, and reusing it keeps the two expressions from drifting:
  #
  #   dl/dphi  = -s/phi - (y + r)/(1 + phi) + y/phi
  #   ds/dphi  = -s/phi + r^2 T1 / phi - r/(1 + phi),  T1 = psi'(r) - psi'(y + r)
  #
  # The weight differentiated is the quasi-likelihood one H is built from,
  # mu / (1 + phi). It is not the observed curvature for this family -- that is
  # `obs_weight`, and the gap is what obs_curvature_delta_for_family carries into
  # the mode-motion solve, so dx_hat/dphi is already on the right inverse here.
  neg_binomial_1 = list(
    dloglik = function(eta, y, n_trials, phi, phi2 = NULL) {
      r <- .mean_log(eta) / phi
      s <- r * (digamma(y + r) - digamma(r) - log1p(phi))
      -s / phi - (y + r) / (1 + phi) + y / phi
    },
    dscore = function(eta, y, n_trials, phi, phi2 = NULL) {
      r  <- .mean_log(eta) / phi
      s  <- r * (digamma(y + r) - digamma(r) - log1p(phi))
      T1 <- trigamma(r) - trigamma(y + r)
      -s / phi + r * r * T1 / phi - r / (1 + phi)
    },
    dweight = function(eta, y, n_trials, phi, phi2 = NULL) {
      -.mean_log(eta) / (1 + phi)^2
    }
  ),

  # phi = size, conditioned on y >= 1. The density is the neg_binomial_2 one
  # minus log P(Y > 0), so each derivative is the untruncated entry above plus
  # the phi-derivative of that retained-mass term. Writing
  #
  #   S = phi + mu,  a = phi log1p(mu/phi),  q = e^-a = P(Y = 0),  p = 1 - q
  #   a_phi = da/dphi = log1p(mu/phi) - mu/S,   so   dp/dphi = q a_phi
  #
  # gives -d log p / dphi = -q a_phi / p for the density, and the score and
  # weight follow from the registry's own closed forms rather than from the
  # untruncated ones plus a correction -- `score` is phi (y - mu/p) / S and
  # `weight` is w0 (1 - q w0) with w0 = phi mu / (p S), and differentiating
  # those directly is shorter than assembling the two pieces.
  #
  # The weight differentiated here is the EXPECTED one, Var(y | y > 0). That is
  # deliberate and is the same rule the neg_binomial_2 note above states from
  # the other side: differentiate whatever weight the Newton solve builds H
  # from, because log|H| is what the outer objective carries. For this family
  # that is the expected form (laplace_family_link.h:368), chosen there because
  # it is positive for every mu; the observed curvature lives in `obs_weight`
  # and drives the mode motion instead, through Hinv_mode.
  truncated_neg_binomial_2 = list(
    dloglik = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_log(eta)
      S  <- phi + mu
      a  <- phi * log1p(mu / phi)
      q  <- exp(-a)
      p  <- -expm1(-a)
      a_phi <- log1p(mu / phi) - mu / S
      digamma(y + phi) - digamma(phi) + log(phi) - log(S) + 1 - (phi + y) / S -
        q * a_phi / p
    },
    dscore = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_log(eta)
      S  <- phi + mu
      a  <- phi * log1p(mu / phi)
      q  <- exp(-a)
      p  <- -expm1(-a)
      a_phi <- log1p(mu / phi) - mu / S
      mu * (y - mu / p) / S^2 + phi * mu * q * a_phi / (S * p^2)
    },
    # dw/dphi = f_a a_p + f_ae a_e_p: the shape route rather than the w0 one, so
    # this and the two second-order terms below come from a single derivation.
    dweight = function(eta, y, n_trials, phi, phi2 = NULL) {
      s <- .tnb2_shape(.mean_log(eta), phi)
      s$f_a * s$a_p + s$f_ae * s$a_e_p
    }
  ),

  # phi = residual VARIANCE (the R-side convention; the compiled kernels take
  # the SD and are handed sqrt(phi) at the boundary).
  gaussian = .normal_scale_dphi(function(eta, y) y - eta),

  # phi = log-scale VARIANCE. Same normal density at the log-scale residual.
  lognormal = .normal_scale_dphi(function(eta, y) log(y) - eta),

  # phi = shape. The working weight is phi itself (constant in eta), so the
  # dW/deta path contributes nothing here and dW/dphi is 1.
  gamma = list(
    dloglik = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_log(eta)
      log(phi) + 1 - digamma(phi) + log(y) - log(mu) - y / mu
    },
    dscore = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_log(eta)
      (y - mu) / mu
    },
    dweight = function(eta, y, n_trials, phi, phi2 = NULL) rep(1, length(eta))
  ),

  # phi = SCALE (not a variance), phi2 = degrees of freedom nu. Collecting the
  # registered log-density over D = nu phi^2 + (y - eta)^2 removes the nested
  # log1p and leaves every derivative rational in D:
  #
  #   l  = C(nu) - 0.5 log(pi) + 0.5 nu log(nu phi^2) - 0.5 (nu + 1) log D
  #
  # so dl/dphi = nu/phi - nu(nu + 1) phi / D. The working weight is the constant
  # Fisher information (nu + 1) / ((nu + 3) phi^2): free of eta, which is what
  # makes the mode motion irrelevant here even though the observed curvature
  # (nu + 1)(nu phi^2 - d^2) / D^2 is a different function.
  t = list(
    dloglik = function(eta, y, n_trials, phi, phi2 = NULL) {
      nu <- .t_df(phi2); d <- y - eta
      D  <- nu * phi^2 + d^2
      nu / phi - nu * (nu + 1) * phi / D
    },
    dscore = function(eta, y, n_trials, phi, phi2 = NULL) {
      nu <- .t_df(phi2); d <- y - eta
      D  <- nu * phi^2 + d^2
      -2 * nu * (nu + 1) * phi * d / D^2
    },
    dweight = function(eta, y, n_trials, phi, phi2 = NULL) {
      nu <- .t_df(phi2)
      rep(-2 * (nu + 1) / ((nu + 3) * phi^3), length(eta))
    }
  ),

  # phi = precision, a = mu phi and b = (1 - mu) phi, so da/dphi = mu and
  # db/dphi = 1 - mu carry every derivative. The weight differentiated is the
  # Fisher form phi^2 (psi'(a) + psi'(b)) dmu^2 that H is built from; the
  # observed curvature is a different function, and it reaches the mode motion
  # through obs_curvature_delta_for_family rather than through this entry.
  beta = list(
    dloglik = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_beta(eta)
      a <- mu * phi
      b <- (1 - mu) * phi
      digamma(phi) - mu * digamma(a) - (1 - mu) * digamma(b) +
        mu * log(y) + (1 - mu) * log1p(-y)
    },
    dscore = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_beta(eta)
      a <- mu * phi
      b <- (1 - mu) * phi
      dmu <- mu * (1 - mu)
      dmu * ((log(y) - log1p(-y) - digamma(a) + digamma(b)) +
               phi * (-trigamma(a) * mu + trigamma(b) * (1 - mu)))
    },
    dweight = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_beta(eta)
      a <- mu * phi
      b <- (1 - mu) * phi
      dmu <- mu * (1 - mu)
      dmu^2 * (2 * phi * (trigamma(a) + trigamma(b)) +
                 phi^2 * (psigamma(a, 2L) * mu + psigamma(b, 2L) * (1 - mu)))
    }
  ),

  # phi = dispersion, Var(y) = phi mu^3. Both the score and the working weight
  # are proportional to 1/phi, so their phi-derivatives are just -1/phi times
  # themselves; only the log-density carries a second term.
  inverse_gaussian = list(
    dloglik = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_log(eta)
      -0.5 / phi + (y - mu)^2 / (2 * phi^2 * mu^2 * y)
    },
    dscore = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_log(eta)
      -(y - mu) / (phi^2 * mu^2)
    },
    dweight = function(eta, y, n_trials, phi, phi2 = NULL) {
      -1 / (phi^2 * .mean_log(eta))
    }
  ),

  # phi = precision, n_trials = n. a + b = phi is free of eta but NOT of phi, so
  # the -lgamma(n + a + b) + lgamma(a + b) pair contributes psi(phi) - psi(n +
  # phi) on its own. The weight is the moment form n mu(1-mu) / D, whose only
  # phi dependence is the overdispersion factor D = 1 + (n-1)/(phi+1).
  beta_binomial = list(
    dloglik = function(eta, y, n_trials, phi, phi2 = NULL) {
      n  <- if (!is.null(n_trials)) n_trials else rep(1, length(eta))
      mu <- .mean_beta(eta); a <- mu * phi; b <- (1 - mu) * phi
      mu * (digamma(y + a) - digamma(a)) +
        (1 - mu) * (digamma(n - y + b) - digamma(b)) +
        digamma(phi) - digamma(n + phi)
    },
    dscore = function(eta, y, n_trials, phi, phi2 = NULL) {
      n  <- if (!is.null(n_trials)) n_trials else rep(1, length(eta))
      mu <- .mean_beta(eta); a <- mu * phi; b <- (1 - mu) * phi
      dmu <- mu * (1 - mu)
      dmu * ((digamma(y + a) - digamma(a) - digamma(n - y + b) + digamma(b)) +
               phi * (mu * (trigamma(y + a) - trigamma(a)) -
                        (1 - mu) * (trigamma(n - y + b) - trigamma(b))))
    },
    dweight = function(eta, y, n_trials, phi, phi2 = NULL) {
      n  <- if (!is.null(n_trials)) n_trials else rep(1, length(eta))
      mu <- .mean_beta(eta)
      D  <- 1 + (n - 1) / (phi + 1)
      n * mu * (1 - mu) * (n - 1) / (D^2 * (phi + 1)^2)
    }
  ),

  # phi = dispersion, phi2 = the variance power p. The score and weight are the
  # exponential-dispersion closed forms, both proportional to 1/phi. The
  # log-density is not: its normalizer is the compound Poisson-gamma series, and
  # since every term carries phi only through lambda and the gamma rate b -- each
  # exactly 1/phi -- differentiating log sum_n W_n leaves the series' own MEAN
  # event count,
  #
  #   dl/dphi = [lambda + b y - E_W[n] / (p - 1)] / phi
  #
  # with E_W[n] read off the same enumeration that produces the density, so the
  # two cannot use different truncations of the series.
  tweedie = list(
    dloglik = function(eta, y, n_trials, phi, phi2 = NULL) {
      p <- .tweedie_power(phi2)
      .tweedie_dloglik_dphi(.mean_log(eta), y, phi, p)
    },
    dscore = function(eta, y, n_trials, phi, phi2 = NULL) {
      p  <- .tweedie_power(phi2)
      mu <- .mean_log(eta)
      -(y - mu) / (phi^2 * mu^(p - 1))
    },
    dweight = function(eta, y, n_trials, phi, phi2 = NULL) {
      p <- .tweedie_power(phi2)
      -.mean_log(eta)^(2 - p) / phi^2
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


# ============================================================================
# Second-order dispersion derivatives: the phi Hessian's family half.
#
# The dispersion column and diagonal of the exact outer Hessian
# (dev_notes/laplace_phi_hessian.md) differentiate the phi GRADIENT once more,
# which needs four scalars per observation beyond the three in .FAMILY_DPHI,
# each differentiating the SAME registered loglik / score / weight:
#
#     dloglik2     = d2 loglik / dphi2            (L2)
#     dscore2      = d2 score  / dphi2            (Sc2)
#     dweight2     = d2 W      / dphi2            (DW2)
#     dweight_deta = d2 W      / (deta dphi)      (DWde, the mixed term)
#
# The fifth quantity the assembly needs, d(dscore/dphi)/deta, is the identity
# -dweight (score's eta-derivative is -W_obs, so its phi-derivative is -dW/dphi),
# not a new function.
#
# The border differentiates quantities formed on the TRUE-curvature inverse, so
# where the working weight is not the observed curvature it needs dH_true/dpsi --
# that is, the phi-derivative of W_obs - w on top of the eta-derivative the
# random-effect block already carries. A family in that position supplies a fifth
# entry, `dobs_weight` = d(W_obs)/dphi; a family whose working weight IS the
# observed curvature, or whose weight is free of eta (so the channel is zero
# either way), needs only the four. `.family_dphi2_needs_obs()` states which,
# and the assembly declines rather than pairing two different inverses.
#
# A family absent here keeps the phi gradient but hands its phi Hessian to the
# differencing stencil rather than reporting an inexact closed form.
# ============================================================================

#' @keywords internal
.FAMILY_DPHI2 <- list(

  # phi = size. Differentiates .FAMILY_DPHI$neg_binomial_2 once more in phi;
  # mu = exp(eta), s = mu + phi throughout.
  neg_binomial_2 = list(
    dloglik2 = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_log(eta)
      trigamma(y + phi) - trigamma(phi) + 1 / phi - 1 / (phi + mu) +
        (y - mu) / (phi + mu)^2
    },
    dscore2 = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_log(eta)
      -2 * (y - mu) * mu / (mu + phi)^3
    },
    dweight2 = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_log(eta)
      2 * mu * (mu * (mu - 2 * phi) + y * (phi - 2 * mu)) / (mu + phi)^4
    },
    dweight_deta = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_log(eta)
      mu * (phi^2 * (4 * mu - y) + 2 * phi * mu * (2 * y - mu) - y * mu^2) /
        (mu + phi)^4
    }
  ),

  # phi = size, conditioned on y >= 1. The log-density and score halves
  # differentiate the truncated first-order entry once more directly; the three
  # weight terms are one chain rule over f(a, a_e) and the shape's own
  # derivatives (.tnb2_shape), which is also where the first-order dweight comes
  # from. The mixed eta-phi term uses the equality of the shape's mixed
  # partials: d(a_p)/deta and d(a_e)/dphi are both mu^2 / S^2.
  truncated_neg_binomial_2 = list(
    dloglik2 = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_log(eta); s <- .tnb2_shape(mu, phi)
      q <- s$q; p <- s$p; a_p <- s$a_p
      trigamma(y + phi) - trigamma(phi) + 1 / phi - 1 / s$S + (y - mu) / s$S^2 +
        q * a_p^2 / p - q * s$a_pp / p + q^2 * a_p^2 / p^2
    },
    dscore2 = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_log(eta); s <- .tnb2_shape(mu, phi)
      q <- s$q; p <- s$p; a_p <- s$a_p; S <- s$S
      # d/dphi of mu (y - mu/p) / S^2, with d(1/p)/dphi = -q a_p / p^2.
      t1 <- mu^2 * q * a_p / (p^2 * S^2) - 2 * mu * (y - mu / p) / S^3
      # d/dphi of phi mu q a_p / (S p^2), as a quotient.
      num  <- phi * mu * q * a_p
      dnum <- mu * q * (a_p + phi * (s$a_pp - a_p^2))
      den  <- S * p^2
      dden <- p^2 + 2 * S * p * q * a_p
      t1 + (dnum * den - num * dden) / den^2
    },
    dweight2 = function(eta, y, n_trials, phi, phi2 = NULL) {
      s <- .tnb2_shape(.mean_log(eta), phi)
      (s$f_aa * s$a_p + s$f_a_ae * s$a_e_p) * s$a_p + s$f_a * s$a_pp +
        (s$f_a_ae * s$a_p + s$f_aeae * s$a_e_p) * s$a_e_p + s$f_ae * s$a_e_pp
    },
    dweight_deta = function(eta, y, n_trials, phi, phi2 = NULL) {
      s <- .tnb2_shape(.mean_log(eta), phi)
      (s$f_aa * s$a_e + s$f_a_ae * s$a_ee) * s$a_p + s$f_a * s$a_e_p +
        (s$f_a_ae * s$a_e + s$f_aeae * s$a_ee) * s$a_e_p + s$f_ae * s$a_ee_p
    },
    # d(W_obs)/dphi. Required of any family whose working weight is NOT the
    # observed curvature: the phi border differentiates quantities formed on
    # H_true^-1, so it needs dH_true/dpsi, which carries the phi-derivative of
    # W_obs - w as well as its eta-derivative. W_obs is the untruncated NB2
    # observed curvature plus d2 log p / deta2, and both halves are written over
    # the same shape terms as the weight above.
    dobs_weight = function(eta, y, n_trials, phi, phi2 = NULL) {
      mu <- .mean_log(eta); s <- .tnb2_shape(mu, phi); S <- s$S
      q <- s$q; p <- s$p; a_p <- s$a_p
      # (y + phi) phi mu / S^2, differentiated in phi.
      d_nb2 <- mu * ((y + 2 * phi) * S - 2 * (y + phi) * phi) / S^3
      # d2 log p / deta2 = (d2p_e p - dp_e^2) / p^2 over the eta-derivatives of
      # p, then differentiated in phi through q, p and the shape.
      dp_e  <- q * s$a_e
      d2p_e <- q * (s$a_ee - s$a_e^2)
      d_dp_e  <- -q * a_p * s$a_e + q * s$a_e_p
      d_d2p_e <- -q * a_p * (s$a_ee - s$a_e^2) +
        q * (s$a_ee_p - 2 * s$a_e * s$a_e_p)
      num  <- d2p_e * p - dp_e^2
      dnum <- d_d2p_e * p + d2p_e * (q * a_p) - 2 * dp_e * d_dp_e
      den  <- p^2
      dden <- 2 * p * q * a_p
      d_nb2 + (dnum * den - num * dden) / den^2
    }
  ),

  # phi = residual VARIANCE / log-scale VARIANCE. W = 1/phi is constant in eta,
  # so both mixed and eta-free-second weight derivatives vanish.
  gaussian  = .normal_scale_dphi2(function(eta, y) y - eta),
  lognormal = .normal_scale_dphi2(function(eta, y) log(y) - eta),

  # phi = shape. W = phi carries no eta and is linear in phi, so the two weight
  # curvatures are exactly zero and only the log-density's trigamma survives.
  gamma = list(
    dloglik2 = function(eta, y, n_trials, phi, phi2 = NULL) {
      rep(1 / phi - trigamma(phi), length(eta))
    },
    dscore2      = function(eta, y, n_trials, phi, phi2 = NULL) rep(0, length(eta)),
    dweight2     = function(eta, y, n_trials, phi, phi2 = NULL) rep(0, length(eta)),
    dweight_deta = function(eta, y, n_trials, phi, phi2 = NULL) rep(0, length(eta)),
    # W_obs = phi y / mu, so its phi-derivative is y / mu. The Fisher weight phi
    # is the observed curvature only in expectation, which is why this is owed
    # even though the weight itself carries no eta.
    dobs_weight = function(eta, y, n_trials, phi, phi2 = NULL) {
      y / .mean_log(eta)
    }
  ),

  # phi = scale, phi2 = nu. Differentiates the t entry once more over
  # D = nu phi^2 + (y - eta)^2, whose own phi-derivative is 2 nu phi.
  t = list(
    dloglik2 = function(eta, y, n_trials, phi, phi2 = NULL) {
      nu <- .t_df(phi2); d <- y - eta
      D  <- nu * phi^2 + d^2
      -nu / phi^2 - nu * (nu + 1) / D + 2 * nu^2 * (nu + 1) * phi^2 / D^2
    },
    dscore2 = function(eta, y, n_trials, phi, phi2 = NULL) {
      nu <- .t_df(phi2); d <- y - eta
      D  <- nu * phi^2 + d^2
      -2 * nu * (nu + 1) * d / D^2 + 8 * nu^2 * (nu + 1) * phi^2 * d / D^3
    },
    dweight2 = function(eta, y, n_trials, phi, phi2 = NULL) {
      nu <- .t_df(phi2)
      rep(6 * (nu + 1) / ((nu + 3) * phi^4), length(eta))
    },
    dweight_deta = function(eta, y, n_trials, phi, phi2 = NULL) {
      rep(0, length(eta))
    },
    # W_obs = (nu+1)(nu phi^2 - d^2) / D^2 is the redescending observed
    # information, a different function from the constant Fisher weight Newton
    # uses, so the border owes its phi-derivative.
    dobs_weight = function(eta, y, n_trials, phi, phi2 = NULL) {
      nu <- .t_df(phi2); d <- y - eta
      D  <- nu * phi^2 + d^2
      2 * nu * phi * (nu + 1) * (3 * d^2 - nu * phi^2) / D^3
    }
  )
)


#' Second-order dispersion derivatives for a family, or `NULL`
#'
#' @param family Family name, as registered in `.FAMILY_OPS`.
#' @return A list with `dloglik2`, `dscore2`, `dweight2` and `dweight_deta`, or
#'   `NULL` when the family's phi Hessian has not been derived. `NULL` is a
#'   refusal, not a fixed-value fallback: it hands the phi Hessian to the
#'   differencing stencil rather than reporting an inexact closed form.
#' @keywords internal
.family_dphi2 <- function(family) {
  nm <- .family_base(family)
  if (startsWith(nm, ".")) return(NULL)
  d2 <- .FAMILY_DPHI2[[nm]]
  # An entry that owes a d(W_obs)/dphi and does not carry one would be paired
  # with the wrong inverse in the border, which is the failure this refuses.
  if (!is.null(d2) && .family_dphi2_needs_obs(family) &&
      !is.function(d2$dobs_weight)) {
    return(NULL)
  }
  d2
}


#' Whether a family's phi Hessian additionally needs `d(W_obs)/dphi`
#'
#' TRUE whenever the Newton working weight is not the observed curvature, which
#' is exactly when the border's `Hinv` and the mode motion's `Hinv_mode` are
#' different matrices. For a family whose weight is also free of eta the two
#' inverses multiply a zero channel and the answer would come out the same
#' either way, but the entry is required regardless rather than resting on that
#' second coincidence.
#' @keywords internal
.family_dphi2_needs_obs <- function(family) {
  nm <- .family_base(family)
  !isTRUE(tryCatch(cpp_family_working_weight_is_observed(nm),
                   error = function(e) TRUE))
}


#' Families whose dispersion can be estimated
#' @keywords internal
.dispersion_families <- function() {
  nms <- names(.FAMILY_DPHI)
  nms[!startsWith(nms, ".")]
}
