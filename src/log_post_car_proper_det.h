// log_post_car_proper_det.h
// Templated CAR-proper log-determinant via dense Cholesky.
// Self-contained: opens namespace tulpa with the math:: helpers in scope.
#ifndef TULPA_LOG_POST_CAR_PROPER_DET_H
#define TULPA_LOG_POST_CAR_PROPER_DET_H

#include <cmath>
#include <cstddef>
#include <vector>

#include "autodiff_utils.h"  // tulpa::math::safe_sqrt, safe_log, get_value

namespace tulpa {

using math::safe_sqrt;
using math::safe_log;
using math::get_value;

template<typename T>
static inline T car_proper_log_det_t(
    int n,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx,
    const std::vector<int>& n_neighbors,
    T rho
) {
    std::vector<T> Q(static_cast<size_t>(n) * n, T(0.0));
    for (int i = 0; i < n; i++) {
        Q[static_cast<size_t>(i) * n + i] = T(n_neighbors[i]);
        for (int k = adj_row_ptr[i]; k < adj_row_ptr[i + 1]; k++) {
            int j = adj_col_idx[k];
            Q[static_cast<size_t>(i) * n + j] = -rho;
        }
    }

    std::vector<T> L(static_cast<size_t>(n) * n, T(0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            T sum = Q[static_cast<size_t>(i) * n + j];
            for (int k = 0; k < j; k++) {
                sum = sum - L[static_cast<size_t>(i) * n + k] *
                            L[static_cast<size_t>(j) * n + k];
            }
            if (i == j) {
                if (get_value(sum) <= 0.0) {
                    return T(-INFINITY);
                }
                L[static_cast<size_t>(i) * n + j] = safe_sqrt(sum);
            } else {
                L[static_cast<size_t>(i) * n + j] =
                    sum / L[static_cast<size_t>(j) * n + j];
            }
        }
    }

    T log_det = T(0.0);
    for (int i = 0; i < n; i++) {
        log_det = log_det + T(2.0) * safe_log(L[static_cast<size_t>(i) * n + i]);
    }
    return log_det;
}

}  // namespace tulpa

#endif  // TULPA_LOG_POST_CAR_PROPER_DET_H
