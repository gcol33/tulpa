#ifndef TULPA_SVC_RANGE_H
#define TULPA_SVC_RANGE_H

#include <tulpa/model_data.h>
#include <tulpa/param_layout.h>
#include <tulpa/field_range.h>
#include "autodiff_utils.h"
#include "pc_prior.h"

namespace tulpa {

template<class T>
T svc_range_log_prior(const T& coordinate, const ModelData& data) {
    if (data.svc_range_prior == 1) {
        return uniform_range_coordinate_log_density(coordinate, true);
    }
    if (data.svc_range_prior == 2) return T(-0.5) * coordinate * coordinate;
    return log_prior_range_pc_at_log(coordinate,
        data.svc_phi_prior_U, data.svc_phi_prior_alpha) + coordinate;
}

template<class T>
std::vector<T> svc_nc_native_coordinates(const std::vector<T>& params,
    const ModelData& data, const ParamLayout& layout) {
    std::vector<T> native = params;
    if (data.svc_range_prior == 1)
        for (int j = 0; j < data.svc_data.n_svc; ++j) {
            const int index = layout.log_phi_svc_start + j;
            native[index] = math::safe_log(svc_range_from_coordinate(params[index], data));
        }
    return native;
}

}
#endif
