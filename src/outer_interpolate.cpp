#include <RcppEigen.h>
#include <algorithm>
#include <array>
#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <functional>
#include "sobol.h"

#include "outer_spline.h"

namespace tulpa_outer_mixture {

struct Value { double cdf, density; };

template<bool need_density>
static inline Value evaluate(const Rcpp::NumericMatrix& mu,
    const Rcpp::NumericMatrix& sd, const Rcpp::NumericVector& mass, int column,
    double x) {
    constexpr double inv_sqrt_2 = 0.70710678118654752440;
    constexpr double inv_sqrt_2pi = 0.39894228040143267794;
    double cdf = 0, density = 0, cdf_error = 0, density_error = 0;
    for (int i = 0; i < mu.nrow(); ++i) {
        const double s = sd(i, column);
        const double z = s > 0 ? (x - mu(i, column)) / s : 0;
        const double probability = s > 0 ? 0.5 * std::erfc(-z * inv_sqrt_2) :
            (x >= mu(i, column) ? 1.0 : 0.0);
        const double c = mass[i] * probability - cdf_error;
        const double next_cdf = cdf + c;
        cdf_error = (next_cdf - cdf) - c;
        cdf = next_cdf;
        if constexpr (need_density) {
            if (s > 0) {
                const double f = mass[i] * inv_sqrt_2pi * std::exp(-0.5 * z * z) / s - density_error;
                const double next_density = density + f;
                density_error = (next_density - density) - f;
                density = next_density;
            }
        }
    }
    return {cdf, density};
}

} // namespace tulpa_outer_mixture

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_nl_mixture_quantiles(Rcpp::NumericMatrix mu,
    Rcpp::NumericMatrix sd, Rcpp::NumericVector mass, Rcpp::NumericVector probability,
    Rcpp::NumericVector center, Rcpp::NumericVector scale, int max_iter = 80) {
    const int n = mu.nrow(), p = mu.ncol();
    if (n < 1 || p < 1 || sd.nrow() != n || sd.ncol() != p || mass.size() != n ||
        center.size() != p || scale.size() != p || max_iter < 1)
        Rcpp::stop("Gaussian mixture dimensions are inconsistent.");
    Rcpp::NumericMatrix out(p, probability.size());
    for (double weight : mass) if (!std::isfinite(weight) || weight < 0)
        Rcpp::stop("Gaussian mixture masses must be finite and nonnegative.");
    const double epsilon = 8 * std::numeric_limits<double>::epsilon();
    for (int j = 0; j < p; ++j) {
        double lower = std::numeric_limits<double>::infinity(), upper = -lower;
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(mu(i, j)) || !std::isfinite(sd(i, j)) || sd(i, j) < 0)
                Rcpp::stop("Gaussian mixture locations and scales must be finite, with nonnegative scales.");
            lower = std::min(lower, mu(i, j) - 10 * sd(i, j));
            upper = std::max(upper, mu(i, j) + 10 * sd(i, j));
        }
        for (int k = 0; k < probability.size(); ++k) {
            const double target = probability[k];
            if (std::isnan(target)) { out(j, k) = NA_REAL; continue; }
            double lo = lower, hi = upper;
            double x = center[j] + scale[j] * R::qnorm5(target, 0, 1, true, false);
            if (!std::isfinite(x) || x <= lo || x >= hi) x = lo / 2 + hi / 2;
            for (int iteration = 0; ; ++iteration) {
                if (iteration % 32 == 0) Rcpp::checkUserInterrupt();
                const bool newton = iteration < max_iter;
                const auto value = newton ? tulpa_outer_mixture::evaluate<true>(mu, sd, mass, j, x) :
                    tulpa_outer_mixture::evaluate<false>(mu, sd, mass, j, x);
                const bool below = value.cdf < target;
                if (below) lo = x; else hi = x;
                const double tolerance = std::max(std::numeric_limits<double>::denorm_min(),
                    epsilon * std::max(scale[j], std::abs(x)));
                if (hi - lo <= tolerance) break;
                double proposed = lo / 2 + hi / 2;
                if (newton) {
                    const double step = (value.cdf - target) / value.density;
                    if (std::isfinite(step) && std::abs(step) <= tolerance) {
                        const double adjacent = x + (below ? tolerance : -tolerance);
                        const double cdf = tulpa_outer_mixture::evaluate<false>(mu, sd, mass, j, adjacent).cdf;
                        if (below ? cdf >= target : cdf < target) break;
                    } else {
                        const double candidate = x - step;
                        if (std::isfinite(candidate) && candidate > lo && candidate < hi)
                            proposed = candidate;
                    }
                }
                x = proposed;
            }
            out(j, k) = x;
        }
    }
    return out;
}

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
    using tulpa_outer_spline::TensorCorners;
    const int d = nodes.size();
    if (query.ncol() != d || mass.size() != query.nrow() || d < 1)
        Rcpp::stop("Outer partition dimensions are inconsistent.");
    std::vector<Axis> axes;
    int total = 1;
    for (int j = 0; j < d; ++j) {
        axes.emplace_back(Rcpp::as<Rcpp::NumericVector>(nodes[j]), false);
        if (axes.back().x.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / total))
            Rcpp::stop("Outer partition exceeds sparse matrix index capacity.");
        total *= axes.back().x.size();
    }
    std::vector<Eigen::Triplet<double>> entries;
    const TensorCorners layout(axes);
    const std::size_t corners = layout.offset.size();
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
        std::size_t base = 0;
        for (int j = 0; j < d; ++j) base += first[j] * layout.stride[j];
        for (std::size_t corner = 0; corner < corners; ++corner) {
            double weight = mass[i];
            for (int j = 0; j < d; ++j) {
                const int at = layout.digit[corner * d + j];
                weight *= weights[j][at];
            }
            if (weight > 0) entries.emplace_back(i, base + layout.offset[corner], weight);
        }
    }
    Eigen::SparseMatrix<double> out(query.nrow(), total);
    out.setFromTriplets(entries.begin(), entries.end());
    out.makeCompressed();
    return out;
}

// [[Rcpp::export]]
Eigen::SparseMatrix<double> cpp_outer_nearest_partition(Rcpp::NumericMatrix points,
    Rcpp::NumericMatrix query, Rcpp::NumericVector mass) {
    const int n = points.nrow(), d = points.ncol();
    if (n < 1 || d < 1 || query.ncol() != d || mass.size() != query.nrow())
        Rcpp::stop("Outer partition dimensions are inconsistent.");
    for (double value : points) if (!std::isfinite(value))
        Rcpp::stop("Outer partition points must be finite.");
    for (double value : query) if (!std::isfinite(value))
        Rcpp::stop("Outer partition queries must be finite.");
    struct Node { int point, axis, left, right; };
    std::vector<Node> tree;
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::function<int(int, int, int)> build = [&](int lo, int hi, int depth) -> int {
        if (lo >= hi) return -1;
        const int axis = depth % d, middle = (lo + hi) / 2;
        std::nth_element(order.begin() + lo, order.begin() + middle, order.begin() + hi,
            [&](int a, int b) { return points(a, axis) < points(b, axis); });
        const int at = tree.size();
        tree.push_back({order[middle], axis, -1, -1});
        const int left = build(lo, middle, depth + 1);
        const int right = build(middle + 1, hi, depth + 1);
        tree[at].left = left;
        tree[at].right = right;
        return at;
    };
    build(0, n, 0);
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(query.nrow());
    for (int i = 0; i < query.nrow(); ++i) {
        if (!std::isfinite(mass[i]) || mass[i] < 0)
            Rcpp::stop("Outer partition masses must be finite and nonnegative.");
        int best = 0;
        double distance = std::numeric_limits<double>::infinity();
        std::function<void(int)> visit = [&](int at) {
            if (at < 0) return;
            const auto& node = tree[at];
            double square = 0;
            for (int j = 0; j < d; ++j) {
                const double difference = query(i, j) - points(node.point, j);
                square += difference * difference;
            }
            if (square < distance) { distance = square; best = node.point; }
            const double split = query(i, node.axis) - points(node.point, node.axis);
            visit(split < 0 ? node.left : node.right);
            if (split * split < distance) visit(split < 0 ? node.right : node.left);
        };
        visit(0);
        if (mass[i] > 0) entries.emplace_back(i, best, mass[i]);
    }
    Eigen::SparseMatrix<double> out(query.nrow(), n);
    out.setFromTriplets(entries.begin(), entries.end());
    out.makeCompressed();
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_outer_tensor_interpolate(Rcpp::List nodes,
    Rcpp::NumericMatrix values, Rcpp::NumericMatrix query, bool cubic = true) {
    using tulpa_outer_spline::Axis;
    using tulpa_outer_spline::TensorCorners;
    const int d = nodes.size(), fields = values.ncol();
    if (query.ncol() != d || d == 0) Rcpp::stop("Spline query dimensions differ from its axes.");
    std::vector<Axis> axes;
    std::size_t total = 1;
    for (int j = 0; j < d; ++j) {
        axes.emplace_back(Rcpp::as<Rcpp::NumericVector>(nodes[j]), cubic);
        if (axes.back().x.size() > static_cast<std::size_t>(values.nrow()) / total)
            Rcpp::stop("Spline values must cover the full tensor in axis order.");
        total *= axes.back().x.size();
    }
    if (total != static_cast<std::size_t>(values.nrow()))
        Rcpp::stop("Spline values must cover the full tensor in axis order.");
    const Eigen::MatrixXd coefficients = tulpa_outer_spline::coefficients(axes, values);
    Rcpp::NumericMatrix out(query.nrow(), fields);
    tulpa_outer_spline::accumulate(axes, coefficients, query, out);
    return out;
}
