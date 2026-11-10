#ifndef TULPA_FIELD_COORDINATES_H
#define TULPA_FIELD_COORDINATES_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace tulpa {

// x_i = z_i sqrt((a + b) / (a exp(s theta) + b)). The coefficients are
// fixed information from prior curvature, likelihood curvature and the mean
// squared likelihood score at a zero field. In effect coordinates the prior
// information multiplies precision. In covariance-whitened coordinates the
// likelihood information multiplies variance. This retains whitening for
// weak fields and centers the amplitude of resolved fields.
struct FieldCoordinate {
    int begin, end;
    int scale_index;
    double scale_sign;
    double scaled_information;
    double fixed_information;
    int group_width = 0;
    bool preserve_constant = false;
    bool preserve_linear = false;

    struct NullComponent {
        double mean = 0, slope = 0, midpoint = 0, norm = 0;
        double at(int t) const { return mean + slope * (t - midpoint); }
    };
    int width() const { return group_width > 0 ? group_width : end - begin; }
    int rank() const {
        return end - begin - (end - begin) / width() *
            (static_cast<int>(preserve_constant) + static_cast<int>(preserve_linear && width() > 1));
    }
    NullComponent null_component(const std::vector<double>& v, int start) const {
        NullComponent out;
        if (!preserve_constant && !preserve_linear) return out;
        out.midpoint = 0.5 * (width() - 1);
        for (int t = 0; t < width(); ++t) {
            if (preserve_constant) out.mean += v[start + t] / width();
            if (preserve_linear) {
                const double time = t - out.midpoint;
                out.slope += time * v[start + t];
                out.norm += time * time;
            }
        }
        if (out.norm > 0) out.slope /= out.norm;
        return out;
    }
    void apply(const std::vector<double>& input, std::vector<double>& output, double a) const {
        if (!preserve_constant && !preserve_linear) {
            for (int i = begin; i < end; ++i) output[i] = a * input[i];
            return;
        }
        for (int start = begin; start < end; start += width()) {
            const auto null = null_component(input, start);
            for (int t = 0; t < width(); ++t) {
                const int i = start + t;
                output[i] = null.at(t) + a * (input[i] - null.at(t));
            }
        }
    }

    double log_scale(const std::vector<double>& q) const {
        const double lp = std::log(scaled_information) + scale_sign * q[scale_index];
        const double lc = fixed_information > 0 ? std::log(fixed_information) : -INFINITY;
        const double hi = std::max(lp, lc);
        return 0.5 * std::log(scaled_information + fixed_information) -
            0.5 * (hi + std::log(std::exp(lp - hi) + std::exp(lc - hi)));
    }
    double derivative(const std::vector<double>& q) const {
        const double log_fraction = std::log(scaled_information) +
            scale_sign * q[scale_index] + 2 * log_scale(q) -
            std::log(scaled_information + fixed_information);
        return -0.5 * scale_sign * std::exp(log_fraction);
    }
};

// phi = lower + (upper - lower) logistic(z), stored as log(phi).
struct LogRangeCoordinate {
    int index;
    double lower, upper;

    struct Geometry {
        double value, log_jacobian, derivative, jacobian_score;
    };

    Geometry geometry(double z) const {
        const double soft = std::log1p(std::exp(-std::abs(z)));
        const double log_s = -std::max(-z, 0.0) - soft;
        const double log_t = -std::max(z, 0.0) - soft;
        const double log_width = std::log(upper - lower);
        const double log_part = log_width + log_s;
        double value = log_part;
        if (lower > 0.0) {
            const double log_lower = std::log(lower);
            value = std::max(log_lower, log_part) +
                std::log1p(std::exp(-std::abs(log_lower - log_part)));
        }
        const double log_jacobian = log_width + log_s + log_t - value;
        const double derivative = std::exp(log_jacobian);
        return {value, log_jacobian, derivative,
                1.0 - 2.0 * std::exp(log_s) - derivative};
    }

    double inverse(double value) const {
        const double log_upper = std::log(upper);
        if (!(value < log_upper) || (lower > 0.0 && !(value > std::log(lower))))
            throw std::domain_error("Initial GP range must lie strictly inside its uniform prior bounds");
        const double numerator = lower == 0.0 ? value :
            value + std::log1p(-std::exp(std::log(lower) - value));
        const double denominator = log_upper + std::log1p(-std::exp(value - log_upper));
        return numerator - denominator;
    }

    double interior(double value) const {
        return std::isfinite(value) && value < std::log(upper) &&
            (lower == 0.0 || value > std::log(lower)) ? value : geometry(0.0).value;
    }
};

struct FieldCoordinateOperator {
    std::function<double(const std::vector<double>&, std::vector<double>&)> forward;
    std::function<void(std::vector<double>&)> inverse;
    std::function<void(const std::vector<double>&, std::vector<double>&)> pullback;
};

template<typename Gradient>
struct FieldCoordinates {
    std::vector<FieldCoordinate> entries;
    std::vector<LogRangeCoordinate> ranges;
    std::vector<FieldCoordinateOperator> operators;
    Gradient gradient = nullptr;
    Gradient prior_gradient = nullptr;

    bool empty() const { return entries.empty() && ranges.empty() && operators.empty(); }

    double forward(const std::vector<double>& z, std::vector<double>& x) const {
        x = z;
        double jacobian = 0;
        for (const auto& r : ranges) {
            const auto geometry = r.geometry(z[r.index]);
            x[r.index] = geometry.value;
            jacobian += geometry.log_jacobian;
        }
        for (const auto& op : operators) jacobian += op.forward(x, x);
        for (const auto& e : entries) {
            const double log_a = e.log_scale(x);
            const double a = std::exp(log_a);
            e.apply(z, x, a);
            jacobian += e.rank() * log_a;
        }
        return jacobian;
    }
    void inverse(std::vector<double>& x) const {
        for (const auto& e : entries) {
            const double a = std::exp(-e.log_scale(x));
            e.apply(x, x, a);
        }
        for (const auto& op : operators) op.inverse(x);
        for (const auto& r : ranges) x[r.index] = r.inverse(x[r.index]);
    }
    void pullback(const std::vector<double>& z, const std::vector<double>& x,
                  std::vector<double>& gradient) const {
        for (const auto& e : entries) {
            const double a = std::exp(e.log_scale(x));
            double scale_gradient = e.rank();
            for (int start = e.begin; start < e.end; start += e.width()) {
                const auto null = e.null_component(x, start);
                for (int t = 0; t < e.width(); ++t) {
                    const int i = start + t;
                    scale_gradient += (x[i] - null.at(t)) * gradient[i];
                }
            }
            e.apply(gradient, gradient, a);
            gradient[e.scale_index] += e.derivative(x) * scale_gradient;
        }
        if (!operators.empty()) {
            auto position = z;
            for (const auto& r : ranges) position[r.index] = x[r.index];
            for (const auto& op : operators) op.pullback(position, gradient);
        }
        for (const auto& r : ranges) {
            const auto geometry = r.geometry(z[r.index]);
            gradient[r.index] = geometry.derivative * gradient[r.index] + geometry.jacobian_score;
        }
    }
};

} // namespace tulpa
#endif
