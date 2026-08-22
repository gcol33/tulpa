// laplace_types.h
// Shared lightweight types for the Laplace approximation engine.

#ifndef TULPA_LAPLACE_TYPES_H
#define TULPA_LAPLACE_TYPES_H

#include "laplace_core.h"
#include <algorithm>
#include <cstddef>
#include <vector>

namespace tulpa {

using DenseVec = std::vector<double>;

// Row-major flat-backed dense matrix, offering `H[i][j]` syntax over a single
// contiguous `std::vector<double>(n*n)`. One malloc per matrix, and rows laid
// end to end, so the scatter hot path accumulates row-stride-contiguously and
// a column traversal of the Hessian is a strided walk over one allocation
// rather than a pointer chase across n of them.
//
// Currently only the (n, n) square shapes used by the engine (Newton
// Hessians) are needed. `init` in the std::vector-compatible ctor /
// assign overloads is only read for its first element; callers always
// pass a uniform `DenseVec(n, fill_value)`.
class DenseMat {
public:
    DenseMat() : n_(0) {}

    DenseMat(std::size_t n, const DenseVec& init)
        : n_(n),
          data_(n * n, (n > 0 && !init.empty()) ? init[0] : 0.0) {}

    void assign(std::size_t n, const DenseVec& init) {
        n_ = n;
        data_.assign(n * n, (n > 0 && !init.empty()) ? init[0] : 0.0);
    }

    std::size_t size() const { return n_; }

    double* operator[](std::size_t i) {
        return data_.data() + i * n_;
    }
    const double* operator[](std::size_t i) const {
        return data_.data() + i * n_;
    }

    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }

    // Zero the matrix in place. Sizes unchanged. Faster than the
    // row-by-row range-for fill used with the old vector-of-vectors
    // layout because the backing is contiguous.
    void zero() {
        std::fill(data_.begin(), data_.end(), 0.0);
    }

private:
    std::size_t n_;
    std::vector<double> data_;
};

} // namespace tulpa

#endif // TULPA_LAPLACE_TYPES_H
