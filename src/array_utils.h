// array_utils.h
// Utility functions for array layout transformations between R and C++

#ifndef TULPA_ARRAY_UTILS_H
#define TULPA_ARRAY_UTILS_H

#include <Rcpp.h>
#include <vector>

namespace tulpa {

// Flatten a 3D array from R column-major to C++ row-major order
// R stores: arr[i, j1, j2] at index (j2-1)*d1*d2 + (j1-1)*d1 + (i-1)  (0-indexed)
// C++ expects: i * d2 * d3 + j1 * d3 + j2 (i slowest, j2 fastest)
//
// For NNGP neighbor distances: dims = (N, nn, nn)
// C++ accesses as: i * nn * nn + j1 * nn + j2
inline std::vector<double> flatten_3d_rowmajor(
    const Rcpp::NumericVector& arr,
    int d1, int d2, int d3
) {
    std::vector<double> result(d1 * d2 * d3);

    for (int i = 0; i < d1; i++) {
        for (int j1 = 0; j1 < d2; j1++) {
            for (int j2 = 0; j2 < d3; j2++) {
                // R column-major index (0-based)
                int r_idx = j2 * d1 * d2 + j1 * d1 + i;
                // C++ row-major index
                int cpp_idx = i * d2 * d3 + j1 * d3 + j2;
                result[cpp_idx] = arr[r_idx];
            }
        }
    }

    return result;
}

}  // namespace tulpa

#endif  // TULPA_ARRAY_UTILS_H
