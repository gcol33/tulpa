#ifndef TULPA_FIELD_COORDINATES_H
#define TULPA_FIELD_COORDINATES_H

#include <algorithm>
#include <cmath>
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

template<typename Gradient>
struct FieldCoordinates {
    std::vector<FieldCoordinate> entries;
    Gradient gradient = nullptr;
    Gradient prior_gradient = nullptr;

    double forward(const std::vector<double>& z, std::vector<double>& x) const {
        x = z;
        double jacobian = 0;
        for (const auto& e : entries) {
            const double log_a = e.log_scale(z);
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
    }
    void pullback(const std::vector<double>& z, const std::vector<double>& x,
                  std::vector<double>& gradient) const {
        for (const auto& e : entries) {
            const double a = std::exp(e.log_scale(z));
            double scale_gradient = e.rank();
            for (int start = e.begin; start < e.end; start += e.width()) {
                const auto null = e.null_component(x, start);
                for (int t = 0; t < e.width(); ++t) {
                    const int i = start + t;
                    scale_gradient += (x[i] - null.at(t)) * gradient[i];
                }
            }
            e.apply(gradient, gradient, a);
            gradient[e.scale_index] += e.derivative(z) * scale_gradient;
        }
    }
};

} // namespace tulpa
#endif
