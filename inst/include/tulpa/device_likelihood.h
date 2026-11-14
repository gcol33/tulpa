#ifndef TULPA_DEVICE_LIKELIHOOD_H
#define TULPA_DEVICE_LIKELIHOOD_H

#include <string>
#include <vector>

namespace tulpa {
struct ModelData;
struct ParamLayout;

// Column-major response channels remain resident during a fit. Constants may
// change between evaluations, including a Laplace grid cell's dispersion.
// The CUDA source defines tulpa_likelihood_row(response, n, i, eta, constants,
// value, score, curvature), with score[np] and row-major curvature[np*np].
struct DeviceLikelihood {
    std::string source;
    int variant = -1;
    // Additional parameter coordinates appended to eta, with score channels
    // for their direct likelihood derivatives.
    std::vector<int> parameter_indices;
    std::vector<double> response;
    std::vector<double> constants;
};

using DeviceLikelihoodFn = bool(*)(const ModelData&, const ParamLayout&,
    const void* response, bool initialize, DeviceLikelihood& kernel);
}
#endif
