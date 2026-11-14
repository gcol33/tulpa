#ifndef TULPA_HSGP_FIELD_H
#define TULPA_HSGP_FIELD_H

#include "hsgp_spectral.h"
#include "dense_basis_apply.h"
#include <tulpa/hsgp_data.h>

namespace tulpa { namespace priors {

template<typename T>
void evaluate_hsgp_field(const T* beta, const T& sigma2, const T& lengthscale,
                         const HSGPData& basis, std::vector<T>& out) {
    std::vector<T> scaled_beta(basis.m_total);
    const HsgpKernel kernel = static_cast<HsgpKernel>(basis.kernel);
    for (int j = 0; j < basis.m_total; ++j) {
        const T spectrum = hsgp_spectral_density_2d(kernel, sigma2, lengthscale, basis.eigenvalues[j]);
        scaled_beta[j] = math::safe_sqrt(spectrum) * beta[j];
    }
    if (!device::apply_basis(basis.phi_flat.data(), basis.n_obs, basis.m_total, scaled_beta, out))
        out.assign(basis.n_obs, T(std::numeric_limits<double>::quiet_NaN()));
}

} }
#endif
