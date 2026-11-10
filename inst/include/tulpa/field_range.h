#ifndef TULPA_FIELD_RANGE_H
#define TULPA_FIELD_RANGE_H

#include <tulpa/model_data.h>
#include <tulpa/ad_scalar_math.h>

namespace tulpa {

template<class T>
T field_range_from_coordinate(const T& coordinate, double lower, double upper, bool bounded) {
    return bounded ? T(lower) + T(upper - lower) * math::inv_logit(coordinate) :
        math::safe_exp(coordinate);
}

template<class T>
T uniform_range_coordinate_log_density(const T& coordinate, bool bounded) {
    if (!bounded) return coordinate;
    const T absolute = math::get_value(coordinate) >= 0 ? coordinate : -coordinate;
    return -absolute - T(2.0) * math::log1p_fn(math::safe_exp(-absolute));
}

inline double uniform_range_coordinate_score(double coordinate, bool bounded) {
    return bounded ? 1.0 - 2.0 * math::inv_logit(coordinate) : 1.0;
}

inline double field_range_log_slope(double coordinate, double lower, double upper, bool bounded) {
    if (!bounded) return 1.0;
    const double probability = math::inv_logit(coordinate);
    return (upper - lower) * probability * (1.0 - probability) /
        field_range_from_coordinate(coordinate, lower, upper, true);
}

template<class T>
T st_gp_range_from_coordinate(const T& coordinate, const ModelData& data, bool space) {
    const auto& st = data.spatiotemporal_data;
    return field_range_from_coordinate(coordinate,
        space ? st.phi_space_prior_lower : st.phi_time_prior_lower,
        space ? st.phi_space_prior_upper : st.phi_time_prior_upper, data.st_uniform_ranges);
}

template<class T>
T svc_range_from_coordinate(const T& coordinate, const ModelData& data) {
    return field_range_from_coordinate(coordinate, data.svc_range_lower,
        data.svc_range_upper, data.svc_range_prior == 1);
}

}
#endif
