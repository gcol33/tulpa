#ifndef TULPA_FIELD_RANGE_H
#define TULPA_FIELD_RANGE_H

#include <tulpa/model_data.h>
#include <tulpa/ad_scalar_math.h>

namespace tulpa {

template<class T>
T svc_range_from_coordinate(const T& coordinate, const ModelData& data) {
    if (data.svc_range_prior == 1)
        return T(data.svc_range_lower) + T(data.svc_range_upper - data.svc_range_lower) *
            math::inv_logit(coordinate);
    return math::safe_exp(coordinate);
}

}
#endif
