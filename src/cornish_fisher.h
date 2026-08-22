// cornish_fisher.h
//
// Skew-corrected marginal quantiles for the inner Laplace approximation: the
// quantile-side partner of the gamma_3 cubic term (inner_laplace_skew.h).
//
// gamma_3(i) is the leading-order Edgeworth estimate of the third standardized
// cumulant of pi(x_i | theta, y) relative to the Gaussian inner Laplace, read
// off the cubic coefficient of the log density in the standardized coordinate
// z = (x_i - mu_i) / sigma_i. The Cornish-Fisher expansion is the inverse of
// that same Edgeworth series carried to the same order: for a standardized
// variate with third cumulant g, the p-quantile is
//
//   w(z_p; g) = z_p + (g / 6) (z_p^2 - 1) + O(g^2)
//
// (Cornish & Fisher 1938; Johnson, Kotz & Balakrishnan 1994 Ch. 12), so the
// corrected marginal quantile is mu_i + sigma_i {m_i + w(z_p; gamma_3(i))}.
//
// THE CENTRE m_i. w is the quantile function of a variable
// with MEAN ZERO, unit variance and skewness g. The expansion it inverts is RMC
// eq. (22),
//
//   log pi_SLA(z) = const - z^2 / 2 + gamma_1 z + (gamma_3 / 6) z^3,
//
// whose mean is NOT zero: expanding pi_SLA = phi(z){1 + gamma_1 z +
// (gamma_3 / 6) z^3 + ...} and integrating gives E[z] = gamma_1 + gamma_3 / 2,
// var 1 and skewness gamma_3 to the order kept. So the centre carries TWO terms,
// and only one of them is the paper's location term:
//
//   m_i = gamma_1(i) + gamma_3(i) / 2.
//
// The second is induced by the cubic term itself. Dropping it -- which placing
// w's mean-zero variable at mu_i does -- asserts E[z] = 0, i.e. gamma_1 =
// -gamma_3 / 2, which is a claim about the location term rather than an absence
// of one. That is the measured defect: on its intercept-only
// fixture gamma_1 is identically 0 (every eta reads the one latent coordinate, so
// var(eta_j | x_i) = 0), the whole missing centre is gamma_3 / 2, and the
// reshaping applied about mu_i was a net loss on the whole marginal while a plain
// relocation was a gain.
//
// RMC instead fit a skew normal constrained to mean gamma^(1), which sets aside
// the gamma^(3) / 2 contribution their own eq. (22) carries ("gamma^(3)
// contributes only to the skewness whereas the adjustment in the mean comes from
// gamma^(1)"). This engine does not fit their skew normal (see below), and the
// centre used here is the one eq. (22) implies, measured against exact
// quadrature rather than adopted: over 12 coefficients of 2-coefficient
// binomial-logit fixtures the total standardized distance from the reported
// centre to the exact marginal mean is 3.6171 at mu_i, 1.0219 with gamma_3 / 2
// alone, 2.5799 with gamma_1 alone and 0.1410 with both
// (test-inner-skew.R section 10).
//
// WHY A SERIES AND NOT A FITTED FAMILY. Rue, Martino & Chopin (2009) Sec 3.2.3
// fit a skew normal to their eq. (22) under three constraints -- mean
// gamma^(1), variance 1, third log-density derivative at the mode gamma^(3).
// Two of those inputs exist here and one does not: gamma^(1), the location
// shift, comes from their denominator expansion (eq. 20), which is diagonal
// only in their augmented x_j == eta_j representation and is not computed by
// this engine (see the SCOPE note in inner_laplace_skew.h). A skew normal fitted
// on gamma_3 alone is a different construction from theirs, and its attainable
// skewness saturates at |skewness| ~= 0.995 with the shape parameter diverging
// as that bound is approached -- inside the band this correction is gated to
// apply on. The Cornish-Fisher coefficient is linear in g, so it degrades
// gracefully instead, and it is the expansion of the same order gamma_3 itself
// is a leading term of.
//
// WHAT IS CORRECTED AND WHAT IS NOT. The centre and the skewness, to leading
// order; the scale is left at sigma_i, which is what eq. (22) gives it. At a
// symmetric level the reshaping term takes the same value at both ends
// (z_p^2 = z_{1-p}^2), so the SHAPE part of the correction relocates the
// interval without changing its width and only the centre and the asymmetric
// levels distinguish it from a shift.
//
// gamma_1 IS REQUIRED, NOT OPTIONAL. A coordinate whose location term did not
// compute (a widened unit, a field past the eta-variance budget) declines the
// whole correction and reports the Gaussian quantiles. Reading an absent
// gamma_1 as 0 would be the silently-wrong zero the inner diagnostics exist to
// avoid, and on a fit where the location term is large it is the dominant error.
//
// MONOTONICITY. w is a quantile function only where it increases in z:
// dw/dz = 1 + (g / 3) z, so w is monotone on [z_lo, z_hi] exactly when that
// stays positive at both ends (it is linear in z). Outside that range the
// expansion has left the regime where it defines an ordered pair of bounds and
// the caller falls back to the Gaussian quantiles rather than reporting a
// crossed interval.

#ifndef TULPA_CORNISH_FISHER_H
#define TULPA_CORNISH_FISHER_H

#include <cmath>

namespace tulpa {

// Cornish-Fisher standardized quantile at Gaussian quantile z and skewness g.
// Mean zero by construction: E[z + (g/6)(z^2 - 1)] = 0 under z ~ N(0, 1).
inline double cornish_fisher_z(double z, double g) {
  return z + (g / 6.0) * (z * z - 1.0);
}

// The standardized centre eq. (22) implies, from the location term and the mean
// the cubic term induces. NaN in either input propagates, so a coordinate with
// no gamma_1 has no centre and (via cornish_fisher_eligible) no correction.
inline double cornish_fisher_center(double gamma1, double gamma3) {
  return gamma1 + 0.5 * gamma3;
}

// Derivative of the above in z. Positive == locally order-preserving.
inline double cornish_fisher_dz(double z, double g) {
  return 1.0 + (g / 3.0) * z;
}

// Is w(.; g) increasing across the whole closed range [z_lo, z_hi]? The
// derivative is affine in z, so the endpoints decide.
inline bool cornish_fisher_monotone(double g, double z_lo, double z_hi) {
  return cornish_fisher_dz(z_lo, g) > 0.0 && cornish_fisher_dz(z_hi, g) > 0.0;
}

// Are the expansion's two terms inside their bands? One predicate owns the
// whole band decision, so no caller can admit a coordinate on one term alone
// and none re-derives the centre.
//
//   * a non-finite gamma_3 or gamma_1 says "not computable" and never means
//     "no skew" / "no shift" -- a non-finite gamma_1 reaches this through the
//     centre, which propagates it;
//   * |gamma_3| at or past `max_abs_gamma3` is the SHAPE band, past which the
//     leading-order expansion is being extrapolated out of its regime;
//   * |m| at or past `max_abs_centre` is the CENTRE band.
//     The reported quantile is mu_i + sigma_i {m_i + w(z_p; gamma_3)}, so the
//     correction relocates the marginal by m_i standard errors and the shape
//     band alone bounds only half of what it does. The shipped cutoff is `Inf`,
//     i.e. the band does not fire: measured over seven fixtures with an exact
//     reference, a large |m| carrying a small |gamma_3| is uniformly WEAK
//     correlation rather than a strong direction being extrapolated, so every
//     finite cutoff declines the coefficients the correction helps most
//     (the ladder and the derivation are in R/settings.R).
//     The predicate keeps the parameter, so a finite value restores the band
//     on every path at once.
inline bool cornish_fisher_in_band(double gamma1, double gamma3,
                                   double max_abs_gamma3,
                                   double max_abs_centre) {
  if (!std::isfinite(gamma3)) return false;
  if (!(std::fabs(gamma3) < max_abs_gamma3)) return false;
  const double m = cornish_fisher_center(gamma1, gamma3);
  if (!std::isfinite(m)) return false;
  if (!(std::fabs(m) < max_abs_centre)) return false;
  return true;
}

// May index i take the correction at all? The bands, plus a scale to correct
// in: a non-positive or non-finite sigma has no standardized coordinate.
inline bool cornish_fisher_eligible(double sigma, double gamma1, double gamma3,
                                    double max_abs_gamma3,
                                    double max_abs_centre) {
  if (!std::isfinite(sigma) || sigma <= 0.0) return false;
  return cornish_fisher_in_band(gamma1, gamma3, max_abs_gamma3,
                                max_abs_centre);
}

} // namespace tulpa

#endif // TULPA_CORNISH_FISHER_H
