#ifndef TULPA_DENSE_BASIS_APPLY_H
#define TULPA_DENSE_BASIS_APPLY_H

#include "coordinate_linalg.h"
#include "device_dense.h"
#include "autodiff_utils.h"
#include <limits>

namespace tulpa { namespace device {

template<class T>
bool apply_basis(const double* basis, int rows, int columns,
                 const std::vector<T>& beta, std::vector<T>& out) {
    std::vector<double> values(columns), result(rows);
    for (int j = 0; j < columns; ++j) values[j] = tulpa::math::get_value(beta[j]);
    if (!dense_basis(basis).apply(basis, rows, columns, true, values.data(), result.data())) return false;
    if constexpr (std::is_same<T, double>::value) out = std::move(result);
    else if constexpr (tulpa::math::is_arena_var<T>::value) {
        const auto backward = [basis, rows, columns](const double*, int, const double*, int,
                                                     const double* adjoint, double* input) {
            if (!dense_basis(basis).apply(basis, rows, columns, true, adjoint, input, true))
                std::fill(input, input + columns, std::numeric_limits<double>::quiet_NaN());
        };
        const tulpa_coordinate_linalg::Matrix matrix = Eigen::Map<const Eigen::VectorXd>(result.data(), rows);
        out = tulpa_coordinate_linalg::custom(beta, matrix, backward);
    } else {
        for (int j = 0; j < columns; ++j) values[j] = beta[j].grad;
        std::vector<double> derivative(rows);
        if (!dense_basis(basis).apply(basis, rows, columns, true, values.data(), derivative.data())) return false;
        out.resize(rows);
        for (int i = 0; i < rows; ++i) out[i] = T(result[i], derivative[i]);
    }
    return true;
}

} }
#endif
