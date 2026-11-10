#ifndef TULPA_SPATIAL_RANGE_PRIOR_H
#define TULPA_SPATIAL_RANGE_PRIOR_H

#include <tulpa/range_prior.h>
#include "autodiff_utils.h"
#include "pc_prior.h"

namespace tulpa {
namespace priors {

template<typename T>
T spatial_range_log_prior(const T& log_range, const RangePrior& prior,
                           double U, double alpha) {
    using namespace math;
    if (prior.type == RangePriorType::UNIFORM) {
        const double value = get_value(safe_exp(log_range));
        if (value < prior.lower || value > prior.upper) return T(-INFINITY);
        return log_range - T(std::log(prior.upper - prior.lower));
    }
    if (prior.type == RangePriorType::LOGNORMAL) {
        const T z = (log_range - T(prior.meanlog)) / T(prior.sdlog);
        return -T(0.5) * z * z - T(std::log(prior.sdlog));
    }
    return log_prior_range_pc_at_log(log_range, U, alpha) + log_range;
}

inline void validate_spatial_range_prior(const RangePrior& prior, double U,
                                          double alpha, const char* name) {
    if (prior.type == RangePriorType::PC) {
        if (!pc_anchors_valid(U, alpha))
            Rcpp::stop("%s range PC prior needs valid U and alpha anchors", name);
    } else if (prior.type == RangePriorType::UNIFORM) {
        if (!(prior.lower >= 0.0 && prior.upper > prior.lower && std::isfinite(prior.upper)))
            Rcpp::stop("%s uniform range prior needs 0 <= lower < upper < Inf", name);
    } else if (!(std::isfinite(prior.meanlog) && prior.sdlog > 0.0 && std::isfinite(prior.sdlog))) {
        Rcpp::stop("%s lognormal range prior needs finite meanlog and positive sdlog", name);
    }
}

inline double spatial_range_prior_log_normalizer(const RangePrior& prior) {
    return prior.type == RangePriorType::LOGNORMAL ? -0.5 * std::log(2.0 * M_PI) : 0.0;
}

} // namespace priors
} // namespace tulpa
#endif
