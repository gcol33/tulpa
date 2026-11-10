#ifndef TULPA_RESTRICTED_SLOPES_H
#define TULPA_RESTRICTED_SLOPES_H
#include <vector>
#include <cmath>

namespace tulpa {

// Unit linear modes annihilated by an observation-space restriction carry
// independent normalized auxiliary coordinates. Their integrals equal one.
struct RestrictedSlopes {
    int n_times = 0;
    int n_groups = 0;
    int nc_index = 0;
    double nc_scale = 1.0;
    std::vector<double> linear;
    std::vector<double> directions;
    std::vector<int> unobserved;

    bool covers_all() const {
        return n_groups > 0 && directions.size() / n_groups + unobserved.size() ==
            static_cast<size_t>(n_groups);
    }

    template<class T>
    std::vector<T> coefficients(const T* field, bool noncentered) const {
        std::vector<T> slope(n_groups, T(0.0));
        for (int g = 0; g < n_groups; ++g) {
            if (noncentered) slope[g] = field[g * n_times + nc_index] * T(nc_scale);
            else for (int t = 0; t < n_times; ++t)
                slope[g] = slope[g] + field[g * n_times + t] * T(linear[t]);
        }
        return slope;
    }

    template<class T>
    T log_density(const T* field, bool noncentered) const {
        if (directions.empty() && unobserved.empty()) return T(0.0);
        const auto slope = coefficients(field, noncentered);
        T out(0.0);
        const double log_normalizer = 0.5 * std::log(2.0 * std::acos(-1.0));
        for (size_t j = 0; j < directions.size(); j += n_groups) {
            T value(0.0);
            for (int g = 0; g < n_groups; ++g) value = value + slope[g] * T(directions[j + g]);
            out = out - T(0.5) * value * value - T(log_normalizer);
        }
        for (int g : unobserved) out = out - T(0.5) * slope[g] * slope[g] - T(log_normalizer);
        if (noncentered) out = out + T((directions.size() / n_groups + unobserved.size()) * std::log(nc_scale));
        return out;
    }

    void remove(double* field) const {
        if (directions.empty() && unobserved.empty()) return;
        const auto slope = coefficients(field, false);
        std::vector<double> removed(n_groups, 0.0);
        for (size_t j = 0; j < directions.size(); j += n_groups) {
            double value = 0.0;
            for (int g = 0; g < n_groups; ++g) value += slope[g] * directions[j + g];
            for (int g = 0; g < n_groups; ++g) removed[g] += value * directions[j + g];
        }
        for (int g : unobserved) removed[g] = slope[g];
        for (int g = 0; g < n_groups; ++g) for (int t = 0; t < n_times; ++t)
            field[g * n_times + t] -= removed[g] * linear[t];
    }
};
}
#endif
