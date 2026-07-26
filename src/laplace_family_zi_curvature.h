// laplace_family_zi_curvature.h
//
// Eta- and z-derivatives of the two-process curvature block that
// zi::mixture_eta_weights_double() returns.
//
// The exact outer gradient needs d log|H| / d theta. With a zero-inflation
// process H is assembled from a 2 x 2 per-observation block over the count
// predictor eta and the zero predictor z,
//
//     W_i = [ W_ee  W_ez ]
//           [ W_ez  W_zz ]
//
// and BOTH predictors move with the mode as theta changes, so the chain rule
// picks up all six partials rather than the single dw/deta the one-process path
// uses. They are the exact derivatives of whatever mixture_eta_weights_double
// returns -- the same contract laplace_family_curvature.h states, for the same
// reason: H is built from that weight, so differentiating the true log density
// instead would be the gradient of an objective nobody optimizes.
//
// has_zi_curvature_derivative() below is the gate. It is NARROWER than
// zi::compiled_zi_supported(): the y = 0 branch of an untruncated mixture
// differentiates through P(Y = 0) a third time, which needs d/deta of the
// OBSERVED curvature. Only the families whose working weight already IS the
// observed curvature have that registered (curvature_deta_for_family reports
// the working one), which excludes neg_binomial_1. The truncated pair keep the
// gate because a zero-truncated base has p0 == 0: their y = 0 branch is the
// hurdle one, where the count predictor drops out and no third derivative is
// needed.

#ifndef TULPA_LAPLACE_FAMILY_ZI_CURVATURE_H
#define TULPA_LAPLACE_FAMILY_ZI_CURVATURE_H

#include <cmath>
#include <string>

#include "builtin_family_zi.h"
#include "laplace_family_curvature.h"
#include "laplace_family_link.h"

namespace tulpa {
namespace zi {

struct MixtureCurvatureDeriv {
    double dWee_deta;
    double dWee_dz;
    double dWez_deta;
    double dWez_dz;
    double dWzz_deta;
    double dWzz_dz;
};

// Whether mixture_curvature_deriv() is exact for this family. See the header
// note: narrower than compiled_zi_supported(), because the untruncated y = 0
// branch needs the eta-derivative of the observed curvature and only the
// families where working == observed have it.
inline bool has_zi_curvature_derivative(const std::string& family) {
    if (!compiled_zi_supported(family)) return false;
    if (is_zero_truncated(family)) return true;   // hurdle: y = 0 branch is flat
    return family == "poisson" || family == "binomial" ||
           family == "neg_binomial_2";
}

inline MixtureCurvatureDeriv mixture_curvature_deriv(
    double y, int n_trials, double eta_count, double logit_zi,
    const std::string& family, double phi, double phi2
) {
    const double pi_z = (logit_zi >= 0.0)
        ? 1.0 / (1.0 + std::exp(-logit_zi))
        : std::exp(logit_zi) / (1.0 + std::exp(logit_zi));
    const double q  = 1.0 - pi_z;
    const double pq = pi_z * q;

    MixtureCurvatureDeriv d{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // dW_zz/dz is the same in every branch: W_zz = pi (1 - pi) throughout, and
    // d(pi(1-pi))/dz = pi(1-pi)(1-2pi).
    const double dWzz_dz = pq * (1.0 - 2.0 * pi_z);

    if (y != 0.0) {
        // Additively separable: the count block is the base family's own weight
        // at eta, the cross term is zero, and W_zz sees z alone.
        d.dWee_deta = curvature_deta_for_family(y, n_trials, eta_count,
                                                family, phi, phi2);
        d.dWzz_dz   = dWzz_dz;
        return d;
    }

    if (is_zero_truncated(family)) {
        // Hurdle at y = 0: p0 = 0 leaves log(pi), so the count predictor
        // carries no curvature here and nothing of it varies with eta.
        d.dWzz_dz = dWzz_dz;
        return d;
    }

    // --- y = 0, untruncated: differentiate log D, D = pi + (1 - pi) p0, once
    //     more in each predictor. -----------------------------------------
    const double P0 = std::exp(log_lik_for_family(0.0, n_trials, eta_count,
                                                  family, phi, phi2));
    const GradHess gh0 = obs_grad_hess_for_family(0.0, n_trials, eta_count,
                                                  family, phi, phi2);
    const double s0 = gh0.grad;       // d log P0 / d eta
    const double w0 = gh0.neg_hess;   // -d2 log P0 / d eta2
    // d w0 / d eta. Gated to the families where the working weight IS the
    // observed curvature, so this reports the derivative of w0 and not of a
    // different weight (see has_zi_curvature_derivative).
    const double w0p = curvature_deta_for_family(0.0, n_trials, eta_count,
                                                 family, phi, phi2);

    // P0 = exp(L): P0' = P0 L', P0'' = P0(L'^2 + L''), P0''' = P0(L'^3 +
    // 3 L' L'' + L'''), with L' = s0, L'' = -w0, L''' = -w0p.
    const double P0_1 = P0 * s0;
    const double P0_2 = P0 * (s0 * s0 - w0);
    const double P0_3 = P0 * (s0 * s0 * s0 - 3.0 * s0 * w0 - w0p);

    const double D     = pi_z + q * P0;
    if (!(D > 0.0)) return d;
    const double D_e   = q * P0_1;
    const double D_z   = pq * (1.0 - P0);
    const double D_ee  = q * P0_2;
    const double D_ez  = -pq * P0_1;
    const double D_zz  = pq * (1.0 - 2.0 * pi_z) * (1.0 - P0);
    const double D_eee = q * P0_3;
    const double D_eez = -pq * P0_2;
    const double D_ezz = -pq * (1.0 - 2.0 * pi_z) * P0_1;
    const double D_zzz = pq * (1.0 - 6.0 * pi_z + 6.0 * pi_z * pi_z)
                            * (1.0 - P0);

    const double A1 = D_e / D;          // d log D / d eta
    const double A2 = D_z / D;          // d log D / d z
    // B_ab = d2 log D / da db, which is exactly -W_ab.
    const double B_ee = D_ee / D - A1 * A1;
    const double B_ez = D_ez / D - A1 * A2;
    const double B_zz = D_zz / D - A2 * A2;

    // W_ab = -B_ab, so each partial below is the negated derivative of B_ab.
    d.dWee_deta = -(D_eee / D - (D_ee / D) * A1 - 2.0 * A1 * B_ee);
    d.dWee_dz   = -(D_eez / D - (D_ee / D) * A2 - 2.0 * A1 * B_ez);
    d.dWez_deta = -(D_eez / D - (D_ez / D) * A1 - B_ee * A2 - A1 * B_ez);
    d.dWez_dz   = -(D_ezz / D - (D_ez / D) * A2 - B_ez * A2 - A1 * B_zz);
    d.dWzz_deta = -(D_ezz / D - (D_zz / D) * A1 - 2.0 * A2 * B_ez);
    d.dWzz_dz   = -(D_zzz / D - (D_zz / D) * A2 - 2.0 * A2 * B_zz);
    return d;
}

} // namespace zi
} // namespace tulpa

#endif // TULPA_LAPLACE_FAMILY_ZI_CURVATURE_H
