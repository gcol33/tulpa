#ifndef TULPA_MODEL_POINT_API_H
#define TULPA_MODEL_POINT_API_H

#include <tulpa/nuts_api.h>

namespace tulpa {

struct ModelPointRequest {
    // Position is in the sampling chart fixed by coordinate_anchor, or native coordinates.
    const double* position = nullptr;
    int n_params = 0;
    const double* coordinate_anchor = nullptr;
    // Inverse image of the native anchor, or the native position in the identity chart.
    double* sampling_position = nullptr;
    double* value = nullptr;
    double* gradient = nullptr;
    double* stored = nullptr;
    // Structural prior and coordinate Jacobian; callback terms contribute to value.
    double* prior = nullptr;
};

using ModelPointFn = void (*)(const ModelData*, const ParamLayout*, const ModelPointRequest*);

inline ModelPointFn get_model_point_fn() {
    static ModelPointFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = reinterpret_cast<ModelPointFn>(R_GetCCallable("tulpa", "tulpa_model_point"));
    }
    return fn;
}

}
#endif
