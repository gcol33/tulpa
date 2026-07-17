// multinomial_logit.h
// Baseline-category multinomial logit kernel. A genuinely
// NOMINAL (unordered) K-class response: class K is the baseline, the K-1 linear
// predictors eta_1..eta_{K-1} are coupled through the softmax denominator. This is
// a multi-process likelihood unit (n_processes = K-1), distinct from the
// single-process built-in families -- the per-observation Hessian is the full
// (K-1)x(K-1) coupled multinomial information.
//
//   denom   = 1 + sum_j exp(eta_j)
//   p_j     = exp(eta_j) / denom            (j = 1..K-1),   p_K = 1 / denom
//   ll      = (c < K ? eta_{c} : 0) - log(denom)           (observed class c)
//   grad_j  = [c == j] - p_j
//   -H_{jl} = p_j ([j == l] - p_l)          (multinomial Fisher = observed info,
//                                            data-independent, always PSD)
//
// The negative Hessian is the multinomial covariance diag(p) - p p' (over the
// non-baseline classes), which is positive semidefinite for any eta, so the inner
// Newton needs no Fisher fallback. eta_j is clamped for the exp to avoid overflow;
// the softmax is formed in a shifted, overflow-safe way.

#ifndef TULPA_MULTINOMIAL_LOGIT_H
#define TULPA_MULTINOMIAL_LOGIT_H

#include <cmath>
#include <vector>

namespace tulpa {

// Softmax probabilities over the K-1 non-baseline classes plus the baseline.
// Writes p[0..K-2] (non-baseline) and returns p_K (baseline). Overflow-safe:
// shifts by max(0, max eta) so the largest exponent is <= 0.
inline double multinomial_softmax(const double* eta, int Km1, double* p) {
    double m = 0.0;                              // baseline exponent is 0
    for (int j = 0; j < Km1; j++) if (eta[j] > m) m = eta[j];
    double denom = std::exp(0.0 - m);            // baseline term exp(-m)
    for (int j = 0; j < Km1; j++) {
        p[j] = std::exp(eta[j] - m);
        denom += p[j];
    }
    const double inv = 1.0 / denom;
    for (int j = 0; j < Km1; j++) p[j] *= inv;
    return inv * std::exp(0.0 - m);              // p_K = exp(-m)/denom
}

// Per-observation log-likelihood for observed class `cls` (1-based, 1..K).
inline double multinomial_logit_ll(const double* eta, int Km1, int cls) {
    std::vector<double> p(Km1);
    const double pK = multinomial_softmax(eta, Km1, p.data());
    return (cls <= Km1) ? std::log(p[cls - 1]) : std::log(pK);
}

// Per-observation score (grad_eta[Km1]) and negative Hessian (neg_hess row-major
// [Km1 x Km1]) at `eta` for observed class `cls`.
inline void multinomial_logit_grad_hess(const double* eta, int Km1, int cls,
                                        double* grad_eta, double* neg_hess) {
    std::vector<double> p(Km1);
    multinomial_softmax(eta, Km1, p.data());
    for (int j = 0; j < Km1; j++) {
        grad_eta[j] = ((cls - 1) == j ? 1.0 : 0.0) - p[j];
        for (int l = 0; l < Km1; l++)
            neg_hess[j * Km1 + l] = p[j] * ((j == l ? 1.0 : 0.0) - p[l]);
    }
}

} // namespace tulpa

#endif // TULPA_MULTINOMIAL_LOGIT_H
