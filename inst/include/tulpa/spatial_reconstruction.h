#ifndef TULPA_SPATIAL_RECONSTRUCTION_H
#define TULPA_SPATIAL_RECONSTRUCTION_H

#include "nuts_api.h"

namespace tulpa {

// Outputs are row-major n_sample by n_spatial_units. draw selects Gaussian
// conditional field draws; false returns conditional modes.
using SpatialReconstructionFn = void(*)(const ModelData* data, const ParamLayout* layout,
    const double* samples, int n_sample, unsigned int seed, bool draw,
    double* structured, double* unstructured);

inline SpatialReconstructionFn get_spatial_reconstruction_fn() {
    static SpatialReconstructionFn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = reinterpret_cast<SpatialReconstructionFn>(R_GetCCallable("tulpa", "tulpa_reconstruct_collapsed_spatial"));
    }
    return fn;
}

}
#endif
