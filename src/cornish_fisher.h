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
// corrected marginal quantile is mu_i + sigma_i w(z_p; gamma_3(i)).
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
// WHAT IS CORRECTED AND WHAT IS NOT. Only the skewness. The centre mu_i is
// left where the Laplace put it, matching the RMC parameterization with the
// absent location term at its absent value (their gamma^(1) = 0 gives a
// standardized correction of mean zero). At a symmetric level the two
// endpoints move by the same amount, so this reshapes WHERE the interval sits
// rather than how wide it is.
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
inline double cornish_fisher_z(double z, double g) {
  return z + (g / 6.0) * (z * z - 1.0);
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

// May index i take the correction at all? A non-finite or absent gamma_3 says
// "not computable" and never means "no skew"; |gamma_3| at or past
// `max_abs_gamma3` is the band cutoff past which the leading-order expansion
// is being extrapolated out of its regime; a non-positive scale has no
// standardized coordinate to correct in.
inline bool cornish_fisher_eligible(double sigma, double g,
                                    double max_abs_gamma3) {
  if (!std::isfinite(sigma) || sigma <= 0.0) return false;
  if (!std::isfinite(g)) return false;
  if (!(std::fabs(g) < max_abs_gamma3)) return false;
  return true;
}

} // namespace tulpa

#endif // TULPA_CORNISH_FISHER_H
