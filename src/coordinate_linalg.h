#ifndef TULPA_COORDINATE_LINALG_H
#define TULPA_COORDINATE_LINALG_H

#include <RcppEigen.h>
#include <tulpa/autodiff_arena.h>
#include <algorithm>
#include <vector>
#include "device_dense.h"

namespace tulpa_coordinate_linalg {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Var = tulpa::arena::Var;

inline double value(double x) { return x; }
inline double value(const Var& x) { return x.val(); }

template<class T> Matrix matrix(const std::vector<T>& x, int rows, int columns) {
    Matrix out(rows, columns);
    for (size_t i = 0; i < x.size(); ++i) out.data()[i] = value(x[i]);
    return out;
}

inline std::vector<double> flatten(const Matrix& x) {
    return {x.data(), x.data() + x.size()};
}

template<class Backward>
std::vector<Var> custom(const std::vector<Var>& inputs, const Matrix& output, Backward backward) {
    auto* arena = inputs.front().arena_;
    std::vector<int32_t> indices;
    indices.reserve(inputs.size());
    for (const auto& x : inputs) indices.push_back(x.idx_);
    std::vector<int32_t> result_indices;
    arena->add_custom_backward(indices, flatten(output), backward, result_indices);
    std::vector<Var> out(result_indices.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i].arena_ = arena;
        out[i].idx_ = result_indices[i];
    }
    return out;
}

struct Product {
    Matrix a, b, result;
    bool transpose;

    template<class T> Product(const std::vector<T>& left, const std::vector<T>& right,
        int rows, int inner, int columns, bool transpose_left) : transpose(transpose_left) {
        a = matrix(left, transpose ? inner : rows, transpose ? rows : inner);
        if (transpose) a = a.transpose().eval();
        b = matrix(right, inner, columns);
        result.noalias() = a * b;
    }

    void operator()(const double*, int, const double*, int, const double* adjoint, double* out) const {
        const Eigen::Map<const Matrix> g(adjoint, result.rows(), result.cols());
        Matrix da = g * b.transpose();
        if (transpose) da = da.transpose().eval();
        const Matrix db = a.transpose() * g;
        std::copy(da.data(), da.data() + da.size(), out);
        std::copy(db.data(), db.data() + db.size(), out + da.size());
    }
};

inline std::vector<double> product(const std::vector<double>& a, const std::vector<double>& b,
    int rows, int inner, int columns, bool transpose_left = false) {
    return flatten(Product(a, b, rows, inner, columns, transpose_left).result);
}

inline std::vector<Var> product(const std::vector<Var>& a, const std::vector<Var>& b,
    int rows, int inner, int columns, bool transpose_left = false) {
    Product operation(a, b, rows, inner, columns, transpose_left);
    std::vector<Var> inputs(a);
    inputs.insert(inputs.end(), b.begin(), b.end());
    return custom(inputs, operation.result, operation);
}

struct Cholesky {
    Matrix lower;
    bool ok;

    Cholesky() : ok(false) {}

    template<class T> Cholesky(const std::vector<T>& x, int n) {
        const Matrix input = matrix(x, n, n);
        auto& factor = tulpa::device::dense_factor("dense_gp");
        ok = factor.factorize(input.data(), n, true, "dense_gp");
        if (ok) {
            lower.resize(n, n);
            ok = factor.lower(lower.data(), true);
        }
    }

    void operator()(const double*, int, const double*, int, const double* adjoint, double* out) const {
        const int n = lower.rows();
        if (!tulpa::device::cholesky_adjoint(lower.data(), adjoint, out, n))
            std::fill(out, out + static_cast<std::size_t>(n) * n,
                      std::numeric_limits<double>::quiet_NaN());
    }
};

inline bool cholesky(const std::vector<double>& x, int n, std::vector<double>& lower) {
    const Cholesky operation(x, n);
    if (!operation.ok) return false;
    lower = flatten(operation.lower);
    return true;
}

inline bool cholesky(const std::vector<Var>& x, int n, std::vector<Var>& lower) {
    const Cholesky operation(x, n);
    if (!operation.ok) return false;
    lower = custom(x, operation.lower, operation);
    return true;
}

} // namespace tulpa_coordinate_linalg
#endif
