// laplace_family_zi_phi.h
//
// Dispersion derivatives of the two-process mixture, for the exact outer
// gradient and the closed outer Hessian when `estimate_phi` runs alongside a
// zero-inflation process.
//
// WHY THIS IS A SEPARATE ENGINE. Under a HURDLE the base family's phi registry
// (R/family_dispersion.R) already IS the mixture's dispersion derivative: a
// zero-truncated base has P(Y = 0) = 0, so the y = 0 branch is log(pi) alone --
// phi-free -- and the y != 0 branch is log(1 - pi) plus the base density. Under
// GENUINE zero inflation the y = 0 branch is log(pi + (1 - pi) P(Y = 0, phi)),
// which carries phi through P(Y = 0) as well, and couples it to BOTH linear
// predictors. That branch is what this header supplies; everything at y != 0 is
// still the base registry's, masked, and the two are summed on the R side over
// disjoint rows.
//
// The mixture density has one shape in every branch,
//
//     D = q(z) B(eta, phi) + C(z),     q = 1 - pi,  B = exp(G),
//
// with C = pi at y = 0 and C = 0 at y != 0, and G the base log density at y (at
// y = 0 that is log P(Y = 0)). Because pi is logistic in z alone and B moves
// with eta and phi alone, every mixed partial of D factorizes, and every mixed
// partial of log D follows from them by ONE recursion -- the multivariate
// cumulant-from-moment identity
//
//     L(a) = R(a) - sum over proper subsets B of a containing a_1 of
//            L(B) R(a \ B),           R(a) = D_a / D
//
// which is exact at every order and for any mix of the three directions. That is
// why the sixteen fields below are sixteen CALLS rather than sixteen formulas:
// at fourth order the expanded forms have enough terms that a transcription slip
// would read as a plausible number rather than an obvious one.
//
// The curvature block is W = -Hess(log D) throughout, so each W field is minus
// the matching L, and the equalities among the mixed fields (dWee/dz = dWez/deta
// and so on) are properties of the recursion rather than assertions made here.

#ifndef TULPA_LAPLACE_FAMILY_ZI_PHI_H
#define TULPA_LAPLACE_FAMILY_ZI_PHI_H

#include <cmath>
#include <limits>
#include <string>

#include "builtin_family_zi.h"
#include "laplace_family_curvature.h"
#include "laplace_family_link.h"

namespace tulpa {
namespace zi {

// Bounds the index-list scratch in the log-derivative recursion below. Every
// field here is at most a fourth derivative; sized past that so a longer index
// list would be a visible change rather than a silent overrun. At namespace
// scope because the recursion lives in a local class, which may not carry
// static data members.
static const int ZI_PHI_MAXLEN = 8;

// Lam[a][b] = d^(a+b) log P(Y = 0 | eta, phi) / deta^a dphi^b.
//
// The pure-eta column is read off the machinery the one-process path already
// registers, so it cannot drift from it:
//
//     Lam[0][0] = log P0,  Lam[1][0] = score,  Lam[2][0] = -W_obs,
//     Lam[3][0] = -dW_obs/deta
//
// (the gate below restricts this to families whose Newton weight IS the observed
// curvature, which is what lets curvature_deta_for_family stand in for
// -d3 log P0 / deta3). Only the phi columns are family-specific, and only
// neg_binomial_2 among the admitted families has any phi at all.
//
// Orders: a + b <= 4 is what the fourth-order fields need, and within that
// b <= 2. Lam[4][0] is unused here (no field takes four eta derivatives) and is
// left at zero.
inline bool zero_prob_log_derivs(
    const std::string& family, double eta, int n_trials, double phi,
    double phi2, double Lam[5][3]
) {
    for (int a = 0; a < 5; ++a) for (int b = 0; b < 3; ++b) Lam[a][b] = 0.0;

    Lam[0][0] = log_lik_for_family(0.0, n_trials, eta, family, phi, phi2);
    const GradHess gh0 = obs_grad_hess_for_family(0.0, n_trials, eta, family,
                                                  phi, phi2);
    Lam[1][0] = gh0.grad;
    Lam[2][0] = -gh0.neg_hess;
    Lam[3][0] = -curvature_deta_for_family(0.0, n_trials, eta, family, phi,
                                           phi2);

    // Families with no dispersion parameter: the phi columns are exactly zero,
    // and every field this header returns collapses to zero with them.
    if (family == "poisson" || family == "binomial") return true;

    if (family == "neg_binomial_2") {
        // L = log P(Y = 0) = -phi log1p(mu / phi), mu = e^eta, S = phi + mu.
        // Every entry below is elementary in (mu, phi, S); each was checked
        // against the other differentiation order (d/dphi of the eta column
        // against d/deta of the phi column) before being written down.
        const double mu = tulpa_linalg::safe_exp(eta);
        const double S  = phi + mu;
        const double S2 = S * S, S3 = S2 * S, S4 = S3 * S;
        const double L1 = std::log1p(mu / phi);
        Lam[0][1] = mu / S - L1;
        Lam[1][1] = -mu * mu / S2;
        Lam[2][1] = -2.0 * phi * mu * mu / S3;
        Lam[3][1] = -2.0 * phi * mu * mu * (2.0 * phi - mu) / S4;
        Lam[0][2] = mu * mu / (phi * S2);
        Lam[1][2] = 2.0 * mu * mu / S3;
        Lam[2][2] = 2.0 * mu * mu * (2.0 * phi - mu) / S4;
        return true;
    }

    // A zero-truncated base never reaches the coupled branch (p0 = 0 makes the
    // mixture a hurdle, whose y = 0 branch is phi-free), so no phi column is
    // needed for it; the caller returns zeros there.
    if (is_zero_truncated(family)) return true;

    return false;
}

// Whether mixture_phi_deriv() is exact for this family. It needs the gradient's
// gate (which is what makes the pure-eta column above the true log-P0 ladder)
// plus a registered phi column.
inline bool has_zi_phi_deriv(const std::string& family) {
    if (!has_zi_curvature_derivative(family)) return false;
    double Lam[5][3];
    return zero_prob_log_derivs(family, 0.0, 1, 1.0,
                                std::numeric_limits<double>::quiet_NaN(), Lam);
}

// Every field carries exactly one or two phi derivatives; the pure (eta, z)
// fields are mixture_curvature_deriv/deriv2's, not repeated here.
//
// The three names marked "= ..." are the same fourth derivative reached two
// ways; only one of each pair is returned, and the R assembly reads it under
// whichever name the term it multiplies calls for.
struct MixturePhiDeriv {
    double dl_dp;        // d l / dphi
    double dsc_e_dp;     // d2 l / (deta dphi)
    double dsc_z_dp;     // d2 l / (dz dphi)
    double dWee_dp;      // d W_ee / dphi
    double dWez_dp;      // d W_ez / dphi
    double dWzz_dp;      // d W_zz / dphi
    double dWee_dp_de;   // d2 W_ee / (deta dphi)
    double dWee_dp_dz;   // d2 W_ee / (dz dphi)   = d2 W_ez / (deta dphi)
    double dWez_dp_dz;   // d2 W_ez / (dz dphi)   = d2 W_zz / (deta dphi)
    double dWzz_dp_dz;   // d2 W_zz / (dz dphi)
    double dl_dp2;       // d2 l / dphi2
    double dsc_e_dp2;    // d3 l / (deta dphi2)
    double dsc_z_dp2;    // d3 l / (dz dphi2)
    double dWee_dp2;     // d2 W_ee / dphi2
    double dWez_dp2;     // d2 W_ez / dphi2
    double dWzz_dp2;     // d2 W_zz / dphi2
    // False when the observation could not be evaluated; see
    // MixtureCurvatureDeriv::ok. Sixteen zeros are a valid answer here (y != 0
    // and a hurdle's y = 0 are both additively separable), so the failure has
    // to be carried separately from the values.
    bool ok = true;
};

inline MixturePhiDeriv mixture_phi_deriv_declined() {
    const double n = std::numeric_limits<double>::quiet_NaN();
    return MixturePhiDeriv{n, n, n, n, n, n, n, n,
                           n, n, n, n, n, n, n, n, false};
}

// The y = 0 branch of a GENUINELY zero-inflated mixture. Returns all-zero for
// y != 0 and for a hurdle's y = 0, because both are additively separable and
// their dispersion derivatives are the base registry's, applied on the R side.
inline MixturePhiDeriv mixture_phi_deriv(
    double y, int n_trials, double eta_count, double logit_zi,
    const std::string& family, double phi, double phi2
) {
    MixturePhiDeriv d{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (y != 0.0 || is_zero_truncated(family)) return d;

    // The two checks below are has_zi_phi_deriv(family) evaluated at this
    // observation rather than at a probe point: the gradient's gate, which is
    // what makes the pure-eta column the true log-P0 ladder, and a registered
    // phi column.
    if (!has_zi_curvature_derivative(family)) {
        return mixture_phi_deriv_declined();
    }

    // zero_prob_log_derivs returns false for a family with no registered phi
    // column, which is a static property of the family rather than an
    // arithmetic edge: converting it to zeros makes every observation of that
    // family contribute nothing to the dispersion column.
    double Lam[5][3];
    if (!zero_prob_log_derivs(family, eta_count, n_trials, phi, phi2, Lam)) {
        return mixture_phi_deriv_declined();
    }

    const double pi_z = pi_from_logit(logit_zi);
    const double q  = 1.0 - pi_z;
    const double pq = pi_z * q;
    const double m1 = 1.0 - 2.0 * pi_z;

    // B[m][k] = d^(m+k) P0 / deta^m dphi^k, P0 = exp(Lam). Raised one derivative
    // at a time by Leibniz on dP0 = P0 dLam, which is the multivariate Bell
    // recursion written as a convolution rather than as partition sums.
    double B[5][3];
    for (int m = 0; m < 5; ++m) for (int k = 0; k < 3; ++k) B[m][k] = 0.0;
    const int C2[5][5] = {{1,0,0,0,0}, {1,1,0,0,0}, {1,2,1,0,0},
                          {1,3,3,1,0}, {1,4,6,4,1}};
    B[0][0] = std::exp(Lam[0][0]);
    for (int k = 0; k + 1 < 3; ++k) {
        double acc = 0.0;
        for (int j = 0; j <= k; ++j) acc += C2[k][j] * B[0][k - j] * Lam[0][j + 1];
        B[0][k + 1] = acc;
    }
    for (int m = 0; m + 1 < 5; ++m) {
        for (int k = 0; k < 3; ++k) {
            if (m + 1 + k > 4) continue;
            double acc = 0.0;
            for (int i = 0; i <= m; ++i)
                for (int j = 0; j <= k; ++j)
                    acc += C2[m][i] * C2[k][j] * B[m - i][k - j] * Lam[i + 1][j];
            B[m + 1][k] = acc;
        }
    }

    // pi^(n), the logistic block's own derivatives in z.
    double pid[5];
    logit_pi_derivs(pi_z, pid);

    // Dd[m][n][k] = d^(m+n+k) D / deta^m dz^n dphi^k, D = q B + pi. A pure
    // z-derivative differentiates q and pi together; anything touching eta or
    // phi leaves only q B, whose z-derivatives are -pi^(n) for n >= 1.
    double Dd[4][4][3];
    for (int m = 0; m < 4; ++m) for (int n = 0; n < 4; ++n) for (int k = 0; k < 3; ++k) {
        Dd[m][n][k] = 0.0;
        if (m + n + k > 4) continue;
        const double qn = (n == 0) ? q : -pid[n];
        Dd[m][n][k] = (m == 0 && k == 0) ? (qn * B[0][0] + pid[n])
                                         : (qn * B[m][k]);
    }

    const double Dv = Dd[0][0][0];
    if (!(Dv > 0.0)) return mixture_phi_deriv_declined();

    // Alphabet: 0 = eta, 1 = z, 2 = phi. R(a) = D_a / D; L is the cumulant
    // recursion off the first listed index. Depth is at most four.
    struct Eng {
        const double (*Dd)[4][3];
        double Dv;
        double Rr(const int* ix, int len) const {
            int c[3] = {0, 0, 0};
            for (int t = 0; t < len; ++t) ++c[ix[t]];
            return Dd[c[0]][c[1]][c[2]] / Dv;
        }
        double L(const int* ix, int len) const {
            double out = Rr(ix, len);
            const int full = (1 << len) - 1;
            for (int mask = 1; mask < full; ++mask) {
                if (!(mask & 1)) continue;          // subsets containing ix[0]
                int sub[ZI_PHI_MAXLEN], rest[ZI_PHI_MAXLEN], ns = 0, nr = 0;
                for (int t = 0; t < len; ++t) {
                    if (mask & (1 << t)) sub[ns++] = ix[t]; else rest[nr++] = ix[t];
                }
                out -= L(sub, ns) * Rr(rest, nr);
            }
            return out;
        }
    };
    const Eng eng{Dd, Dv};
    auto L = [&](int a, int b = -1, int c = -1, int e = -1) {
        int ix[4]; int len = 0;
        ix[len++] = a;
        if (b >= 0) ix[len++] = b;
        if (c >= 0) ix[len++] = c;
        if (e >= 0) ix[len++] = e;
        return eng.L(ix, len);
    };

    d.dl_dp      =  L(2);
    d.dsc_e_dp   =  L(0, 2);
    d.dsc_z_dp   =  L(1, 2);
    d.dWee_dp    = -L(0, 0, 2);
    d.dWez_dp    = -L(0, 1, 2);
    d.dWzz_dp    = -L(1, 1, 2);
    d.dWee_dp_de = -L(0, 0, 0, 2);
    d.dWee_dp_dz = -L(0, 0, 1, 2);
    d.dWez_dp_dz = -L(0, 1, 1, 2);
    d.dWzz_dp_dz = -L(1, 1, 1, 2);
    d.dl_dp2     =  L(2, 2);
    d.dsc_e_dp2  =  L(0, 2, 2);
    d.dsc_z_dp2  =  L(1, 2, 2);
    d.dWee_dp2   = -L(0, 0, 2, 2);
    d.dWez_dp2   = -L(0, 1, 2, 2);
    d.dWzz_dp2   = -L(1, 1, 2, 2);
    return d;
}

} // namespace zi
} // namespace tulpa

#endif // TULPA_LAPLACE_FAMILY_ZI_PHI_H
