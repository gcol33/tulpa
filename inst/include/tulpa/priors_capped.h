// priors_capped.h
// Templated capped-SD log-priors shared across tulpa and downstream
// packages (tulpaGlmm, etc.). Prior to centralization, each package
// re-implemented `capped_normal_log_prior` in its own family headers.
// This file is the single source of truth.
//
// The capped Normal prior is what stabilized HMC NB on the bad seed
// 1821108667 (610s slowdown without it) per
// dev_notes/salvaged_logs/build_tulpa_capped_sd.log.

#ifndef TULPA_PRIORS_CAPPED_H
#define TULPA_PRIORS_CAPPED_H

#include <cmath>
#include "autodiff_arena.h"
#include "autodiff_fwd.h"

namespace tulpa {
namespace priors {

namespace detail {

// Branch-free "value getter" — the templated prior needs to inspect
// |x - mean| > cap via a plain double, regardless of whether T is a
// double, an arena::Var, or a fwd::Dual. Specialize per supported type.

inline double prior_val(double x) { return x; }
inline double prior_val(const arena::Var& x) { return x.val(); }
inline double prior_val(const fwd::Dual& x) { return x.val; }

}  // namespace detail

// Capped Normal log-prior on x.
//
//   z   = (x - mean) / sd
//   lp  = -0.5 * z^2
//   if |x - mean| > cap: lp -= cap_weight * (|x - mean| - cap)^4
//
// Pass cap <= 0 (or cap_weight <= 0) to disable the cap. The quartic
// is C^3 in (x - mean), so the autodiff gradient is well-defined
// everywhere except x == mean (where the implicit |·| has a kink that
// the supported region [|x - mean| <= cap] excludes anyway).
template<typename T>
inline T log_prior_capped_normal(const T& x, double mean, double sd,
                                 double cap, double cap_weight) {
    T centered = x - T(mean);
    T z = centered / T(sd);
    T lp = T(-0.5) * z * z;
    if (cap > 0.0 && cap_weight > 0.0) {
        double centered_val = detail::prior_val(centered);
        if (std::fabs(centered_val) > cap) {
            // abs_centered = |centered|; pick the autodiff branch by sign
            T abs_centered = (centered_val >= 0.0)
                ? centered
                : T(0.0) - centered;
            T over = abs_centered - T(cap);
            T over_sq = over * over;
            lp = lp - T(cap_weight) * over_sq * over_sq;
        }
    }
    return lp;
}

// Plain-double overload (avoids forcing callers to instantiate the
// template explicitly when they only need scalar evaluation).
inline double log_prior_capped_normal(double x, double mean, double sd,
                                      double cap, double cap_weight) {
    return log_prior_capped_normal<double>(x, mean, sd, cap, cap_weight);
}

}  // namespace priors
}  // namespace tulpa

#endif  // TULPA_PRIORS_CAPPED_H
