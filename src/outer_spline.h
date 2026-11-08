#ifndef TULPA_OUTER_SPLINE_H
#define TULPA_OUTER_SPLINE_H

#include <RcppEigen.h>
#include <algorithm>
#include <array>
#include <vector>
#include <utility>

namespace tulpa_outer_spline {

struct Axis {
    std::vector<double> x, knots;
    int degree;

    Axis(Rcpp::NumericVector nodes, bool cubic) : x(nodes.begin(), nodes.end()),
        degree(nodes.size() == 1 ? 0 : (cubic && nodes.size() >= 4 ? 3 : 1)) {
        if (x.empty() || !std::is_sorted(x.begin(), x.end()) ||
            std::adjacent_find(x.begin(), x.end()) != x.end())
            Rcpp::stop("Spline axes require distinct increasing nodes.");
        for (double value : x) if (!std::isfinite(value))
            Rcpp::stop("Spline axes must be finite.");
        knots.assign(degree + 1, x.front());
        if (degree == 3) {
            for (std::size_t j = 2; j + 2 < x.size(); ++j) knots.push_back(x[j]);
        } else {
            for (std::size_t j = 1; j + 1 < x.size(); ++j) knots.push_back(x[j]);
        }
        knots.insert(knots.end(), degree + 1, x.back());
    }

    Axis(std::vector<double> values, int polynomial_degree) : knots(std::move(values)),
        degree(polynomial_degree) {
        x.resize(knots.size() - degree - 1);
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] = 0;
            for (int j = 1; j <= degree; ++j) x[i] += knots[i + j] / degree;
        }
        if (degree) { x.front() = knots.front(); x.back() = knots.back(); }
    }

    std::array<double, 4> weights_at(double value, int span, int polynomial_degree) const {
        std::array<double, 4> left{}, right{};
        std::array<double, 4> weights{};
        weights[0] = 1;
        for (int j = 1; j <= polynomial_degree; ++j) {
            left[j] = value - knots[span + 1 - j];
            right[j] = knots[span + j] - value;
            double saved = 0;
            for (int r = 0; r < j; ++r) {
                const double t = weights[r] / (right[r + 1] + left[j - r]);
                weights[r] = saved + right[r + 1] * t;
                saved = left[j - r] * t;
            }
            weights[j] = saved;
        }
        return weights;
    }

    int basis(double value, std::array<double, 4>& weights) const {
        if (degree == 0) {
            weights.fill(0);
            weights[0] = 1;
            return 0;
        }
        value = std::max(x.front(), std::min(x.back(), value));
        int span = static_cast<int>(std::upper_bound(knots.begin(), knots.end(), value) -
                                    knots.begin()) - 1;
        span = std::max(degree, std::min(static_cast<int>(x.size()) - 1, span));
        weights = weights_at(value, span, degree);
        return span - degree;
    }
};

struct TensorCorners {
    std::vector<std::size_t> stride, offset;
    std::vector<unsigned char> digit;

    explicit TensorCorners(const std::vector<Axis>& axes) {
        std::size_t total = 1, corners = 1;
        for (const auto& axis : axes) {
            stride.push_back(total);
            total *= axis.x.size();
            corners *= axis.degree + 1;
        }
        offset.assign(corners, 0);
        digit.resize(corners * axes.size());
        for (std::size_t c = 0; c < corners; ++c) {
            std::size_t digits = c;
            for (std::size_t j = 0; j < axes.size(); ++j) {
                const int radix = axes[j].degree + 1, at = digits % radix;
                digits /= radix;
                digit[c * axes.size() + j] = static_cast<unsigned char>(at);
                offset[c] += at * stride[j];
            }
        }
    }
};

inline Eigen::MatrixXd collocation(const Axis& axis, const std::vector<double>& points) {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(points.size(), axis.x.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        std::array<double, 4> weights;
        const int first = axis.basis(points[i], weights);
        for (int j = 0; j <= axis.degree; ++j) matrix(i, first + j) = weights[j];
    }
    return matrix;
}

template<typename Operation>
inline Eigen::MatrixXd transform_axis(const Eigen::MatrixXd& values,
    const std::vector<std::size_t>& sizes, int axis, Operation operation) {
    const std::size_t total = values.rows(), n = sizes[axis], lines = total / n;
    const int fields = values.cols();
    std::size_t step = 1;
    for (int j = 0; j < axis; ++j) step *= sizes[j];
    const std::size_t block = step * n;
    Eigen::MatrixXd rhs(n, lines * fields);
    for (int f = 0; f < fields; ++f) {
        std::size_t line = 0;
        for (std::size_t outer = 0; outer < total; outer += block)
            for (std::size_t offset = 0; offset < step; ++offset, ++line)
                for (std::size_t r = 0; r < n; ++r)
                    rhs(r, line + lines * f) = values(outer + offset + r * step, f);
    }
    Eigen::MatrixXd fitted = operation(rhs);
    const std::size_t out_n = fitted.rows(), out_block = step * out_n;
    Eigen::MatrixXd output(lines * out_n, fields);
    for (int f = 0; f < fields; ++f) {
        std::size_t line = 0, out_outer = 0;
        for (std::size_t outer = 0; outer < total; outer += block, out_outer += out_block)
            for (std::size_t offset = 0; offset < step; ++offset, ++line)
                for (std::size_t r = 0; r < out_n; ++r)
                    output(out_outer + offset + r * step, f) = fitted(r, line + lines * f);
    }
    return output;
}

inline Eigen::MatrixXd coefficients(const std::vector<Axis>& axes, Eigen::MatrixXd output) {
    if (!output.allFinite()) Rcpp::stop("Spline values must be finite.");
    std::vector<std::size_t> sizes;
    for (const auto& axis : axes) sizes.push_back(axis.x.size());
    for (std::size_t j = 0; j < axes.size(); ++j) {
        if (axes[j].degree <= 1) continue;
        const Eigen::MatrixXd matrix = collocation(axes[j], axes[j].x);
        output = transform_axis(output, sizes, j, [&](const Eigen::MatrixXd& rhs) {
            return matrix.partialPivLu().solve(rhs).eval();
        });
    }
    return output;
}

inline Eigen::MatrixXd coefficients(const std::vector<Axis>& axes,
                                      const Rcpp::NumericMatrix& values) {
    return coefficients(axes, Eigen::MatrixXd(Eigen::Map<const Eigen::MatrixXd>(
        values.begin(), values.nrow(), values.ncol())));
}

inline void accumulate(const std::vector<Axis>& axes, const Eigen::MatrixXd& coefficients,
    const Rcpp::NumericMatrix& query, Rcpp::NumericMatrix& output) {
    const int d = axes.size(), fields = coefficients.cols();
    const std::size_t total = coefficients.rows();
    const TensorCorners layout(axes);
    std::vector<int> first(d);
    std::vector<std::array<double, 4>> weights(d);
    for (int i = 0; i < query.nrow(); ++i) {
        if (i % 4096 == 0) Rcpp::checkUserInterrupt();
        for (int j = 0; j < d; ++j) {
            if (!std::isfinite(query(i, j))) Rcpp::stop("Spline queries must be finite.");
            first[j] = axes[j].basis(query(i, j), weights[j]);
        }
        std::size_t base = 0;
        for (int j = 0; j < d; ++j) base += first[j] * layout.stride[j];
        for (std::size_t corner = 0; corner < layout.offset.size(); ++corner) {
            double weight = 1;
            for (int j = 0; j < d; ++j) weight *= weights[j][layout.digit[corner * d + j]];
            const std::size_t row = base + layout.offset[corner];
            if (weight != 0) for (int f = 0; f < fields; ++f)
                output[i + static_cast<std::size_t>(query.nrow()) * f] +=
                    weight * coefficients.data()[row + total * f];
        }
    }
}

}
#endif
