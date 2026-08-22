// array_utils.h
// Utility functions for array layout transformations between R and C++

#ifndef TULPA_ARRAY_UTILS_H
#define TULPA_ARRAY_UTILS_H

#include <Rcpp.h>
#include <cstddef>
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
    if (d1 < 0 || d2 < 0 || d3 < 0) {
        Rcpp::stop("flatten_3d_rowmajor: dimensions must be non-negative.");
    }
    // The index arithmetic below runs in std::size_t, and the extent is
    // checked against the input length, so a dimension triple that does not
    // describe `arr` is an error rather than a read past its end.
    const std::size_t n = static_cast<std::size_t>(d1)
                        * static_cast<std::size_t>(d2)
                        * static_cast<std::size_t>(d3);
    if (n != static_cast<std::size_t>(arr.size())) {
        Rcpp::stop("flatten_3d_rowmajor: d1 * d2 * d3 does not match length(arr).");
    }

    std::vector<double> result(n);

    for (std::size_t i = 0; i < static_cast<std::size_t>(d1); i++) {
        for (std::size_t j1 = 0; j1 < static_cast<std::size_t>(d2); j1++) {
            for (std::size_t j2 = 0; j2 < static_cast<std::size_t>(d3); j2++) {
                // R column-major index (0-based)
                std::size_t r_idx = j2 * static_cast<std::size_t>(d1)
                                       * static_cast<std::size_t>(d2)
                                  + j1 * static_cast<std::size_t>(d1) + i;
                // C++ row-major index
                std::size_t cpp_idx = i * static_cast<std::size_t>(d2)
                                        * static_cast<std::size_t>(d3)
                                    + j1 * static_cast<std::size_t>(d3) + j2;
                result[cpp_idx] = arr[r_idx];
            }
        }
    }

    return result;
}

}  // namespace tulpa

#endif  // TULPA_ARRAY_UTILS_H
