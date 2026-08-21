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
#include <limits>
#include <vector>

namespace tulpa {

// Softmax probabilities over the K-1 non-baseline classes plus the baseline.
// Writes p[0..K-2] (non-baseline) and returns p_K (baseline). Overflow-safe:
// shifts by max(0, max eta) so the largest exponent is <= 0.
//
// `p` may be null, for a caller that wants only the shift and the log
// denominator. Those two are what the log-likelihood is read off: a
// probability is formed by exponentiating, so a class sitting ~745 below the
// shift flushes to exactly zero and its log is -Inf, where the value the
// algebra gives is finite. `shift_out` receives m and `log_denom_out` receives
// log(denom) after the shift, which is >= 0 because the largest term is 1.
inline double multinomial_softmax(const double* eta, int Km1, double* p,
                                  double* shift_out = nullptr,
                                  double* log_denom_out = nullptr) {
    double m = 0.0;                              // baseline exponent is 0
    for (int j = 0; j < Km1; j++) if (eta[j] > m) m = eta[j];
    double denom = std::exp(0.0 - m);            // baseline term exp(-m)
    for (int j = 0; j < Km1; j++) {
        const double e = std::exp(eta[j] - m);
        denom += e;
        if (p) p[j] = e;
    }
    const double inv = 1.0 / denom;
    if (p) for (int j = 0; j < Km1; j++) p[j] *= inv;
    if (shift_out) *shift_out = m;
    if (log_denom_out) *log_denom_out = std::log(denom);
    return inv * std::exp(0.0 - m);              // p_K = exp(-m)/denom
}

// `cls` is the observed class, 1-based in 1..K with K = Km1 + 1. Outside that
// range there is no observed class to score: cls <= 0 indexes before the start
// of the probability buffer, and cls > K reads as the baseline having been
// observed.
inline bool multinomial_class_valid(int Km1, int cls) {
    return Km1 >= 1 && cls >= 1 && cls <= Km1 + 1;
}

// Per-observation log-likelihood for observed class `cls` (1-based, 1..K).
// NaN for a `cls` outside that range. Formed from the shift and the log
// denominator rather than from log(p), so a separated class -- one whose eta
// sits far below the others, which is what separation in a categorical arm
// produces -- reports its finite value instead of -Inf. One -Inf observation
// makes the whole data log-likelihood -Inf, and the Newton line search never
// accepts a non-finite trial, so the solve backtracks off a point the model is
// well defined at.
inline double multinomial_logit_ll(const double* eta, int Km1, int cls) {
    if (!multinomial_class_valid(Km1, cls)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double m = 0.0, log_denom = 0.0;
    multinomial_softmax(eta, Km1, nullptr, &m, &log_denom);
    return (cls <= Km1 ? eta[cls - 1] : 0.0) - m - log_denom;
}

// Per-observation score (grad_eta[Km1]) and negative Hessian (neg_hess row-major
// [Km1 x Km1]) at `eta` for observed class `cls`.
//
// The negative Hessian is the multinomial covariance diag(p) - p p' and does
// not read `cls` at all, so it is returned for any `cls`. The score does: an
// out-of-range one simply never fires the indicator, which returns -p, a
// valid-looking gradient for no observed class, so it is NaN instead.
inline void multinomial_logit_grad_hess(const double* eta, int Km1, int cls,
                                        double* grad_eta, double* neg_hess) {
    if (Km1 < 1) return;
    std::vector<double> p(Km1);
    multinomial_softmax(eta, Km1, p.data());
    const bool valid = multinomial_class_valid(Km1, cls);
    for (int j = 0; j < Km1; j++) {
        grad_eta[j] = valid ? (((cls - 1) == j ? 1.0 : 0.0) - p[j])
                            : std::numeric_limits<double>::quiet_NaN();
        for (int l = 0; l < Km1; l++)
            neg_hess[j * Km1 + l] = p[j] * ((j == l ? 1.0 : 0.0) - p[l]);
    }
}

} // namespace tulpa

#endif // TULPA_MULTINOMIAL_LOGIT_H
