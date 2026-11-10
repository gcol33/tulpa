#ifndef TULPA_NATIVE_ST_GP_KERNEL_H
#define TULPA_NATIVE_ST_GP_KERNEL_H
#include "hmc_svc_autodiff.h"
#include <tulpa/cov_kernel.h>
namespace tulpa {
template<class T>
inline T st_cov_value(double d, const T& sigma2, const T& phi, CovType type) {
    const T range = type == CovType::GAUSSIAN ? T(std::sqrt(2.0)) * phi : phi;
    return tulpa_svc_ad::compute_cov(d, sigma2, range, type);
}
inline double st_cov_dphi(double d, double sigma2, double phi, double value, CovType type) {
    const double scale = type == CovType::GAUSSIAN ? std::sqrt(2.0) : 1.0;
    return scale * cov_dphi(d, scale * phi, value, sigma2, type);
}
}
#include <tulpa/st_gp_kernel.h>
#endif
