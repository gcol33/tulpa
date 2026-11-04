#ifndef TULPA_SLICE_UNIVARIATE_H
#define TULPA_SLICE_UNIVARIATE_H

#include <cmath>

namespace tulpa {

// Neal (2003), Slice sampling, Figures 3 and 5: randomly allocated finite
// stepping-out budget and shrinkage about the starting point.
template<class LogDensity, class Uniform>
double slice_univariate(double x, const LogDensity& log_density,
                        const Uniform& uniform, double width = 1.0,
                        int max_steps = 32) {
    const double lp = log_density(x);
    if (!std::isfinite(lp)) return x;
    const double level = lp + std::log(uniform());
    double left = x - width * uniform();
    double right = left + width;
    int j = static_cast<int>(max_steps * uniform());
    int k = max_steps - 1 - j;
    while (j-- > 0 && log_density(left) > level) left -= width;
    while (k-- > 0 && log_density(right) > level) right += width;
    for (;;) {
        const double proposal = left + uniform() * (right - left);
        if (log_density(proposal) >= level) return proposal;
        if (proposal < x) left = proposal;
        else right = proposal;
        if (proposal == x || std::nextafter(left, right) >= right) return x;
    }
}

} // namespace tulpa
#endif
