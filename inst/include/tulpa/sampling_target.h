#ifndef TULPA_SAMPLING_TARGET_H
#define TULPA_SAMPLING_TARGET_H

namespace tulpa {

enum class SamplingTarget {
    JOINT_POSTERIOR,
    GAUSSIAN_SPATIAL_MARGINAL,
    LAPLACE_SPATIAL_MARGINAL
};

inline const char* sampling_target_name(SamplingTarget target) {
    switch (target) {
    case SamplingTarget::GAUSSIAN_SPATIAL_MARGINAL:
        return "Analytic Gaussian spatial marginal with exact conditional field draws";
    case SamplingTarget::LAPLACE_SPATIAL_MARGINAL:
        return "Laplace spatial marginal with Gaussian conditional field draws";
    default:
        return "Joint posterior";
    }
}

inline int sampling_target_tier(SamplingTarget target) {
    return target == SamplingTarget::LAPLACE_SPATIAL_MARGINAL ? 2 : 1;
}

}
#endif
