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
// OBSERVED curvature -- not the working weight curvature_deta_for_family
// reports. Where the two differ that is the registered derivative plus
// obs_curvature_delta_deta_for_family, so the gate asks for both halves; what it
// excludes is the families with no observed form at all (beta_binomial, t,
// tweedie). The truncated pair keep the gate for a different reason: a
// zero-truncated base has p0 == 0, so their y = 0 branch is the hurdle one,
// where the count predictor drops out and no third derivative is needed.

#ifndef TULPA_LAPLACE_FAMILY_ZI_CURVATURE_H
#define TULPA_LAPLACE_FAMILY_ZI_CURVATURE_H

#include <cmath>
#include <limits>
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
    // False when the observation could not be evaluated. Zero is a valid
    // derivative value, so an all-zero struct is indistinguishable from a
    // computed one and a failed observation would be summed in as a zero
    // contribution instead of taking the diagnostic down.
    bool ok = true;
};

inline MixtureCurvatureDeriv mixture_curvature_deriv_declined() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return MixtureCurvatureDeriv{nan, nan, nan, nan, nan, nan, false};
}

// Whether mixture_curvature_deriv() is exact for this family. See the header
// note: narrower than compiled_zi_supported(), because the untruncated y = 0
// branch needs the eta-derivative of the observed curvature and only the
// families where working == observed have it.
inline bool has_zi_curvature_derivative(const std::string& family) {
    if (!compiled_zi_supported(family)) return false;
    if (is_zero_truncated(family)) return true;   // hurdle: y = 0 branch is flat
    // Untruncated: the y = 0 branch differentiates log P(Y = 0) a THIRD time,
    // so it needs the true d(W_obs)/deta rather than the working weight's. That
    // is the registered derivative plus the observed-minus-working correction's,
    // so the gate is exactly "both halves are available" -- which admits
    // neg_binomial_1 now that its correction carries a tetragamma.
    return has_curvature_derivative(family) &&
           has_obs_curvature_delta_derivative(family);
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
    // d w0 / d eta -- the OBSERVED curvature's derivative, since this branch
    // differentiates the density rather than the weight H is built from. For
    // every family whose Newton weight already is the observed curvature the
    // correction term is identically zero and this is the registered derivative.
    const double w0p = obs_curvature_deta_for_family(0.0, n_trials, eta_count,
                                                     family, phi, phi2);

    // P0 = exp(L): P0' = P0 L', P0'' = P0(L'^2 + L''), P0''' = P0(L'^3 +
    // 3 L' L'' + L'''), with L' = s0, L'' = -w0, L''' = -w0p.
    const double P0_1 = P0 * s0;
    const double P0_2 = P0 * (s0 * s0 - w0);
    const double P0_3 = P0 * (s0 * s0 * s0 - 3.0 * s0 * w0 - w0p);

    // D is the mixture density at y = 0, pi + (1 - pi) P0. It is non-positive
    // only where the arithmetic has already failed -- P0 underflowed and pi_z
    // underflowed with it, or one of them is NaN -- so the answer is that this
    // observation could not be evaluated, not that its derivatives are zero.
    const double D     = pi_z + q * P0;
    if (!(D > 0.0)) return mixture_curvature_deriv_declined();
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

// Second derivatives of the same 2 x 2 block, for the closed-form outer
// Hessian.
//
// The block is W = -Hess(log density) in both branches -- separable at y != 0
// (and at y = 0 under a hurdle, where p0 = 0 leaves log(pi)), coupled through
// log D at y = 0 otherwise. So every second derivative of W is a FOURTH
// derivative of one scalar, and there are five distinct values rather than the
// nine an unconstrained 2 x 2 block would need, indexed by how many of the four
// derivatives are taken in eta. The equalities that collapse nine to five are
// the same ones already visible at third order, where mixture_curvature_deriv()
// computes dWee_dz and dWez_deta by two different expressions that agree.
//
// A hurdle is the special case where the three mixed fields vanish: W_ee then
// depends on eta alone and W_zz on z alone.
struct MixtureCurvatureDeriv2 {
    double d4_e4;     // d2 W_ee / deta2
    double d4_e3z;    // d2 W_ee / deta dz = d2 W_ez / deta2
    double d4_e2z2;   // d2 W_ee / dz2     = d2 W_ez / deta dz = d2 W_zz / deta2
    double d4_ez3;    // d2 W_ez / dz2     = d2 W_zz / deta dz
    double d4_z4;     // d2 W_zz / dz2
    bool ok = true;   // see MixtureCurvatureDeriv::ok
};

inline MixtureCurvatureDeriv2 mixture_curvature_deriv2_declined() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return MixtureCurvatureDeriv2{nan, nan, nan, nan, nan, false};
}

// The coupled y = 0 branch differentiates P(Y = 0) a FOURTH time, so on top of
// what the gradient's gate already requires it needs the second eta-derivative
// of the observed curvature -- which is what has_curvature_2nd_derivative()
// reports, given that the gradient's gate has already restricted this to the
// families whose working weight IS the observed curvature.
inline bool has_zi_curvature_2nd_derivative(const std::string& family) {
    if (!has_zi_curvature_derivative(family)) return false;
    if (is_zero_truncated(family)) return has_curvature_2nd_derivative(family);
    return has_curvature_2nd_derivative(family) &&
           has_obs_curvature_delta_2nd_derivative(family);
}

inline MixtureCurvatureDeriv2 mixture_curvature_deriv2(
    double y, int n_trials, double eta_count, double logit_zi,
    const std::string& family, double phi, double phi2
) {
    // Asked up front rather than at the call site: without it a family whose
    // second-order observed-curvature delta is unregistered gets a finite
    // number built from a wrong fourth-derivative input, and the closed outer
    // Hessian reports it as exact.
    if (!has_zi_curvature_2nd_derivative(family)) {
        return mixture_curvature_deriv2_declined();
    }

    const double pi_z = (logit_zi >= 0.0)
        ? 1.0 / (1.0 + std::exp(-logit_zi))
        : std::exp(logit_zi) / (1.0 + std::exp(logit_zi));
    const double q  = 1.0 - pi_z;
    const double pq = pi_z * q;
    const double m1 = 1.0 - 2.0 * pi_z;

    MixtureCurvatureDeriv2 d{0.0, 0.0, 0.0, 0.0, 0.0};

    // The logistic block's own fourth-order term, d2 W_zz / dz2. W_zz is
    // pi (1 - pi) in every separable branch and at y != 0 of a coupled one.
    const double dz4_sep = pq * (m1 * m1 - 2.0 * pq);

    // Separable branches: log density = f(eta) + g(z), so only the two pure
    // fields survive. y != 0 of any mixture, and y = 0 of a hurdle (where the
    // count predictor carries nothing).
    if (y != 0.0 || is_zero_truncated(family)) {
        d.d4_z4 = dz4_sep;
        if (y != 0.0) {
            d.d4_e4 = curvature_deta2_for_family(y, n_trials, eta_count,
                                                 family, phi, phi2);
        }
        return d;
    }

    // --- y = 0, untruncated: D = pi + (1 - pi) p0 couples both predictors ----
    const double P0 = std::exp(log_lik_for_family(0.0, n_trials, eta_count,
                                                  family, phi, phi2));
    const GradHess gh0 = obs_grad_hess_for_family(0.0, n_trials, eta_count,
                                                  family, phi, phi2);
    const double s0   = gh0.grad;       // d log P0 / d eta
    const double w0   = gh0.neg_hess;   // -d2 log P0 / d eta2
    const double w0p  = obs_curvature_deta_for_family(0.0, n_trials, eta_count,
                                                      family, phi, phi2);
    const double w0pp = obs_curvature_deta2_for_family(0.0, n_trials, eta_count,
                                                       family, phi, phi2);

    // P0 = exp(L) with L' = s0, L'' = -w0, L''' = -w0p, L'''' = -w0pp; the
    // coefficients are the complete Bell polynomials.
    const double P0_1 = P0 * s0;
    const double P0_2 = P0 * (s0 * s0 - w0);
    const double P0_3 = P0 * (s0 * s0 * s0 - 3.0 * s0 * w0 - w0p);
    const double P0_4 = P0 * (s0 * s0 * s0 * s0 - 6.0 * s0 * s0 * w0
                              + 3.0 * w0 * w0 - 4.0 * s0 * w0p - w0pp);

    const double Dv = pi_z + q * P0;
    if (!(Dv > 0.0)) return mixture_curvature_deriv2_declined();

    // pi is logistic in z alone; p0 depends on eta alone. Writing
    // D = p0 + pi (1 - p0), a pure eta-derivative is (1 - pi) p0^(m), a pure
    // z-derivative is pi^(n) (1 - p0), and a mixed one is -pi^(n) p0^(m).
    const double pi1 = pq;
    const double pi2 = pq * m1;
    const double pi3 = pq * (m1 * m1 - 2.0 * pq);
    const double pi4 = pq * m1 * (1.0 - 12.0 * pq);
    const double one_m_P0 = 1.0 - P0;

    double Dd[5][5];
    for (int a = 0; a < 5; ++a) for (int b = 0; b < 5; ++b) Dd[a][b] = 0.0;
    Dd[0][0] = Dv;
    Dd[1][0] = q * P0_1; Dd[2][0] = q * P0_2;
    Dd[3][0] = q * P0_3; Dd[4][0] = q * P0_4;
    Dd[0][1] = pi1 * one_m_P0; Dd[0][2] = pi2 * one_m_P0;
    Dd[0][3] = pi3 * one_m_P0; Dd[0][4] = pi4 * one_m_P0;
    Dd[1][1] = -pi1 * P0_1; Dd[2][1] = -pi1 * P0_2; Dd[3][1] = -pi1 * P0_3;
    Dd[1][2] = -pi2 * P0_1; Dd[2][2] = -pi2 * P0_2;
    Dd[1][3] = -pi3 * P0_1;

    // Derivatives of log D, built by index rather than expanded by hand: the
    // fourth-order forms have enough terms that a transcription slip would be
    // invisible against a plausible-looking number. Index 0 is eta, 1 is z.
    auto Rr = [&](const int* ix, int k) {
        int m = 0, n = 0;
        for (int t = 0; t < k; ++t) { if (ix[t] == 0) ++m; else ++n; }
        return Dd[m][n] / Dv;
    };
    auto Av = [&](int i) { const int a[1] = {i}; return Rr(a, 1); };
    auto L2 = [&](int i, int j) {
        const int a[2] = {i, j};
        return Rr(a, 2) - Av(i) * Av(j);
    };
    // L_abcd, from differentiating
    //   L_abc = D_abc/D - (D_ab A_c + D_ac A_b + D_bc A_a)/D + 2 A_a A_b A_c
    // once more, using d(X/D)/dd = X_d/D - (X/D) A_d and dA_a/dd = L_ad.
    auto L4 = [&](int i, int j, int k, int l) {
        const int a[4]    = {i, j, k, l};
        const int aijk[3] = {i, j, k}, aijl[3] = {i, j, l};
        const int aikl[3] = {i, k, l}, ajkl[3] = {j, k, l};
        const int aij[2]  = {i, j}, aik[2] = {i, k}, ajk[2] = {j, k};
        const double head = Rr(a, 4) - Rr(aijk, 3) * Av(l);
        const double mid =
            (Rr(aijl, 3) - Rr(aij, 2) * Av(l)) * Av(k) + Rr(aij, 2) * L2(k, l) +
            (Rr(aikl, 3) - Rr(aik, 2) * Av(l)) * Av(j) + Rr(aik, 2) * L2(j, l) +
            (Rr(ajkl, 3) - Rr(ajk, 2) * Av(l)) * Av(i) + Rr(ajk, 2) * L2(i, l);
        const double tail = 2.0 * (L2(i, l) * Av(j) * Av(k) +
                                   Av(i) * L2(j, l) * Av(k) +
                                   Av(i) * Av(j) * L2(k, l));
        return head - mid + tail;
    };

    // W_ab = -L_ab throughout, so each second derivative is minus an L4.
    d.d4_e4   = -L4(0, 0, 0, 0);
    d.d4_e3z  = -L4(0, 0, 0, 1);
    d.d4_e2z2 = -L4(0, 0, 1, 1);
    d.d4_ez3  = -L4(0, 1, 1, 1);
    d.d4_z4   = -L4(1, 1, 1, 1);
    return d;
}

} // namespace zi
} // namespace tulpa

#endif // TULPA_LAPLACE_FAMILY_ZI_CURVATURE_H
