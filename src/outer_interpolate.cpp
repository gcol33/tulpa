#include <RcppEigen.h>
#include <algorithm>
#include <array>
#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include "sobol.h"

namespace tulpa_outer_spline {

struct Axis {
    std::vector<double> x, knots;
    int degree;

    Axis(Rcpp::NumericVector nodes, bool cubic) : x(nodes.begin(), nodes.end()),
        degree(cubic && nodes.size() >= 4 ? 3 : 1) {
        if (x.size() < 2 || !std::is_sorted(x.begin(), x.end()) ||
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

    int basis(double value, std::array<double, 4>& weights) const {
        value = std::max(x.front(), std::min(x.back(), value));
        int span = static_cast<int>(std::upper_bound(knots.begin(), knots.end(), value) -
                                    knots.begin()) - 1;
        span = std::max(degree, std::min(static_cast<int>(x.size()) - 1, span));
        std::array<double, 4> left{}, right{};
        weights.fill(0);
        weights[0] = 1;
        for (int j = 1; j <= degree; ++j) {
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
        return span - degree;
    }
};

} // namespace tulpa_outer_spline

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_outer_read_design(int n, int d, int shift_seed) {
    if (n < 1 || d < 1 || d > tulpa::SOBOL_MAX_DIM)
        Rcpp::stop("Outer read-design dimensions are outside the Sobol table.");
    Rcpp::NumericMatrix out(n, d);
    tulpa::sobol_points(n, d, out.begin());
    std::mt19937 rng(static_cast<unsigned int>(shift_seed));
    for (int j = 0; j < d; ++j) {
        const double shift = (static_cast<double>(rng()) + 0.5) / 4294967296.0;
        for (int i = 0; i < n; ++i) {
            const double value = out(i, j) + shift;
            out(i, j) = value - std::floor(value);
        }
    }
    return out;
}

// [[Rcpp::export]]
Eigen::SparseMatrix<double> cpp_outer_tensor_partition(Rcpp::List nodes,
    Rcpp::NumericMatrix query, Rcpp::NumericVector mass) {
    using tulpa_outer_spline::Axis;
    const int d = nodes.size();
    if (query.ncol() != d || mass.size() != query.nrow() || d < 1)
        Rcpp::stop("Outer partition dimensions are inconsistent.");
    std::vector<Axis> axes;
    std::vector<int> stride;
    int total = 1;
    for (int j = 0; j < d; ++j) {
        stride.push_back(total);
        axes.emplace_back(Rcpp::as<Rcpp::NumericVector>(nodes[j]), false);
        if (axes.back().x.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / total))
            Rcpp::stop("Outer partition exceeds sparse matrix index capacity.");
        total *= axes.back().x.size();
    }
    std::vector<Eigen::Triplet<double>> entries;
    const int corners = 1 << d;
    entries.reserve(static_cast<std::size_t>(query.nrow()) * corners);
    std::vector<int> first(d);
    std::vector<std::array<double, 4>> weights(d);
    for (int i = 0; i < query.nrow(); ++i) {
        if (!std::isfinite(mass[i]) || mass[i] < 0)
            Rcpp::stop("Outer partition masses must be finite and nonnegative.");
        for (int j = 0; j < d; ++j) {
            if (!std::isfinite(query(i, j))) Rcpp::stop("Outer partition queries must be finite.");
            first[j] = axes[j].basis(query(i, j), weights[j]);
        }
        for (int corner = 0; corner < corners; ++corner) {
            double weight = mass[i];
            int column = 0;
            for (int j = 0; j < d; ++j) {
                const int at = (corner >> j) & 1;
                column += (first[j] + at) * stride[j];
                weight *= weights[j][at];
            }
            if (weight > 0) entries.emplace_back(i, column, weight);
        }
    }
    Eigen::SparseMatrix<double> out(query.nrow(), total);
    out.setFromTriplets(entries.begin(), entries.end());
    out.makeCompressed();
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_outer_tensor_interpolate(Rcpp::List nodes,
    Rcpp::NumericMatrix values, Rcpp::NumericMatrix query, bool cubic = true) {
    using tulpa_outer_spline::Axis;
    const int d = nodes.size(), fields = values.ncol();
    if (query.ncol() != d || d == 0) Rcpp::stop("Spline query dimensions differ from its axes.");
    std::vector<Axis> axes;
    std::vector<std::size_t> stride;
    std::size_t total = 1, corners = 1;
    for (int j = 0; j < d; ++j) {
        stride.push_back(total);
        axes.emplace_back(Rcpp::as<Rcpp::NumericVector>(nodes[j]), cubic);
        if (axes.back().x.size() > static_cast<std::size_t>(values.nrow()) / total)
            Rcpp::stop("Spline values must cover the full tensor in axis order.");
        total *= axes.back().x.size();
        corners *= axes.back().degree + 1;
    }
    if (total != static_cast<std::size_t>(values.nrow()))
        Rcpp::stop("Spline values must cover the full tensor in axis order.");
    Eigen::MatrixXd coefficients = Eigen::Map<const Eigen::MatrixXd>(
        values.begin(), values.nrow(), fields);
    if (!coefficients.allFinite()) Rcpp::stop("Spline values must be finite.");
    for (int j = 0; j < d; ++j) {
        if (axes[j].degree == 1) continue;
        const int n = axes[j].x.size();
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n, n);
        for (int i = 0; i < n; ++i) {
            std::array<double, 4> weights;
            const int first = axes[j].basis(axes[j].x[i], weights);
            for (int r = 0; r <= axes[j].degree; ++r) B(i, first + r) = weights[r];
        }
        const std::size_t step = stride[j], block = step * n;
        const std::size_t lines = total / n;
        Eigen::MatrixXd rhs(n, lines * fields);
        for (int f = 0; f < fields; ++f) {
            std::size_t line = 0;
            for (std::size_t outer = 0; outer < total; outer += block)
                for (std::size_t offset = 0; offset < step; ++offset, ++line)
                    for (int r = 0; r < n; ++r)
                        rhs(r, line + lines * f) = coefficients(outer + offset + r * step, f);
        }
        Eigen::MatrixXd fitted = B.partialPivLu().solve(rhs);
        for (int f = 0; f < fields; ++f) {
            std::size_t line = 0;
            for (std::size_t outer = 0; outer < total; outer += block)
                for (std::size_t offset = 0; offset < step; ++offset, ++line)
                    for (int r = 0; r < n; ++r)
                        coefficients(outer + offset + r * step, f) = fitted(r, line + lines * f);
        }
    }
    Rcpp::NumericMatrix out(query.nrow(), fields);
    std::vector<int> first(d);
    std::vector<std::array<double, 4>> weights(d);
    for (int i = 0; i < query.nrow(); ++i) {
        if (i % 4096 == 0) Rcpp::checkUserInterrupt();
        for (int j = 0; j < d; ++j) {
            if (!std::isfinite(query(i, j))) Rcpp::stop("Spline queries must be finite.");
            first[j] = axes[j].basis(query(i, j), weights[j]);
        }
        for (std::size_t corner = 0; corner < corners; ++corner) {
            std::size_t digits = corner, row = 0;
            double weight = 1;
            for (int j = 0; j < d; ++j) {
                const int radix = axes[j].degree + 1, at = digits % radix;
                digits /= radix;
                row += (first[j] + at) * stride[j];
                weight *= weights[j][at];
            }
            if (weight != 0) for (int f = 0; f < fields; ++f)
                out(i, f) += weight * coefficients(row, f);
        }
    }
    return out;
}
