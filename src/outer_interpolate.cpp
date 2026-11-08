#include <RcppEigen.h>
#include <algorithm>
#include <array>
#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <cstdint>
#include <cstring>
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
Rcpp::IntegerVector cpp_nl_axis_row_id(Rcpp::NumericMatrix points, int axis) {
    const int d = points.ncol();
    if (axis < 1 || axis > d) Rcpp::stop("The conditional axis must name a matrix column.");
    struct Hash {
        std::size_t operator()(const std::vector<std::uint64_t>& key) const {
            std::size_t value = 0;
            for (auto bits : key) value ^= std::hash<std::uint64_t>{}(bits) +
                static_cast<std::size_t>(0x9e3779b9) + (value << 6) + (value >> 2);
            return value;
        }
    };
    std::unordered_map<std::vector<std::uint64_t>, int, Hash> groups;
    groups.reserve(points.nrow());
    std::vector<std::uint64_t> key;
    key.reserve(d - 1);
    Rcpp::IntegerVector out(points.nrow());
    for (int i = 0; i < points.nrow(); ++i) {
        key.clear();
        for (int j = 0; j < d; ++j) if (j != axis - 1) {
            double value = points(i, j);
            if (R_IsNA(value)) value = NA_REAL;
            else if (std::isnan(value)) value = R_NaN;
            std::uint64_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            key.push_back(bits);
        }
        const auto inserted = groups.emplace(key, static_cast<int>(groups.size()) + 1);
        out[i] = inserted.first->second;
    }
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_nl_group_logsumexp(Rcpp::NumericVector values,
    Rcpp::IntegerVector groups, int n_groups) {
    if (values.size() != groups.size() || n_groups < 0)
        Rcpp::stop("Log-sum-exp groups have inconsistent dimensions.");
    const double negative_infinity = -std::numeric_limits<double>::infinity();
    Rcpp::NumericVector maximum(n_groups, negative_infinity);
    std::vector<bool> invalid(n_groups, false);
    std::vector<long double> sum(n_groups, 0);
    for (R_xlen_t i = 0; i < values.size(); ++i) {
        const int label = groups[i];
        if (label == NA_INTEGER || label < 1 || label > n_groups)
            Rcpp::stop("Log-sum-exp group indices must lie within their declared range.");
        const int group = label - 1;
        const double value = values[i];
        if (std::isnan(value) || value == std::numeric_limits<double>::infinity())
            invalid[group] = true;
        else maximum[group] = std::max(maximum[group], value);
    }
    for (R_xlen_t i = 0; i < values.size(); ++i) {
        const int group = groups[i] - 1;
        if (!invalid[group] && std::isfinite(maximum[group]))
            sum[group] += std::exp(values[i] - maximum[group]);
    }
    for (int group = 0; group < n_groups; ++group) {
        if (invalid[group] || !std::isfinite(maximum[group]))
            maximum[group] = negative_infinity;
        else maximum[group] += std::log(static_cast<double>(sum[group]));
    }
    return maximum;
}

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
    struct Node { int point, left, right; };
    std::vector<Node> tree;
    tree.reserve(n);
    std::vector<double> lower(static_cast<std::size_t>(n) * d);
    std::vector<double> upper(lower.size());
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::function<int(int, int)> build = [&](int lo, int hi) -> int {
        if (lo >= hi) return -1;
        const int at = tree.size(), middle = (lo + hi) / 2;
        tree.push_back({0, -1, -1});
        int axis = 0;
        double spread = -1;
        for (int j = 0; j < d; ++j) {
            const std::size_t offset = static_cast<std::size_t>(at) * d + j;
            lower[offset] = upper[offset] = points(order[lo], j);
            for (int k = lo + 1; k < hi; ++k) {
                lower[offset] = std::min(lower[offset], points(order[k], j));
                upper[offset] = std::max(upper[offset], points(order[k], j));
            }
            const double width = upper[offset] - lower[offset];
            if (width > spread) { spread = width; axis = j; }
        }
        std::nth_element(order.begin() + lo, order.begin() + middle, order.begin() + hi,
            [&](int a, int b) { return points(a, axis) < points(b, axis) ||
                (points(a, axis) == points(b, axis) && a < b); });
        tree[at].point = order[middle];
        const int left = build(lo, middle);
        const int right = build(middle + 1, hi);
        tree[at].left = left;
        tree[at].right = right;
        return at;
    };
    build(0, n);
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(query.nrow());
    for (int i = 0; i < query.nrow(); ++i) {
        if (!std::isfinite(mass[i]) || mass[i] < 0)
            Rcpp::stop("Outer partition masses must be finite and nonnegative.");
        int best = 0;
        double distance = std::numeric_limits<double>::infinity();
        auto box_distance = [&](int at) {
            if (at < 0) return std::numeric_limits<double>::infinity();
            double square = 0;
            for (int j = 0; j < d; ++j) {
                const std::size_t offset = static_cast<std::size_t>(at) * d + j;
                const double delta = std::max({lower[offset] - query(i, j),
                    query(i, j) - upper[offset], 0.0});
                square += delta * delta;
            }
            return square;
        };
        auto visit = [&](auto&& self, int at) -> void {
            if (at < 0) return;
            const auto& node = tree[at];
            double square = 0;
            for (int j = 0; j < d; ++j) {
                const double difference = query(i, j) - points(node.point, j);
                square += difference * difference;
            }
            if (square < distance || (square == distance && node.point < best)) {
                distance = square; best = node.point;
            }
            int first = node.left, second = node.right;
            double first_bound = box_distance(first), second_bound = box_distance(second);
            if (first_bound > second_bound) {
                std::swap(first, second); std::swap(first_bound, second_bound);
            }
            if (first >= 0 && first_bound <= distance) self(self, first);
            if (second >= 0 && second_bound <= distance) self(self, second);
        };
        visit(visit, 0);
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
