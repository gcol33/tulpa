// portable_math.h
// Thread-safe C++ math functions that replace R::lgammafn, R::digamma, etc.
// No R dependency in hot paths - avoids R's C stack checking and is OpenMP-safe.

#ifndef TULPA_PORTABLE_MATH_H
#define TULPA_PORTABLE_MATH_H

#include <cmath>
#include <limits>
#include <utility>   // std::pair, returned by portable_digamma_lgamma

namespace tulpa {
namespace math {

// Portable digamma using asymptotic expansion + recurrence
// For x > 0 only (sufficient for all our use cases).
// Unrolled recurrence (branchless for x >= 1) eliminates while-loop overhead.
inline double portable_digamma(double x) {
    if (x <= 0.0) return -1e10;  // guard (should not happen in practice)

    // Unrolled recurrence: always shift x to x+N where x+N >= 7.
    // For our use cases x >= 1 (counts + dispersion), so N=6 always suffices.
    // Each recurrence step: digamma(x) = digamma(x+1) - 1/x
    // Accumulate result = -1/x - 1/(x+1) - ... - 1/(x+N-1), then asymptotic at x+N.
    //
    // Conditional version for x < 7: at most 6 steps, unrolled.
    double result = 0.0;
    if (x < 7.0) {
        // Unrolled: up to 6 recurrence steps (handles x >= 1)
        if (x < 1.0) { result -= 1.0 / x; x += 1.0; }
        if (x < 2.0) { result -= 1.0 / x; x += 1.0; }
        if (x < 3.0) { result -= 1.0 / x; x += 1.0; }
        if (x < 4.0) { result -= 1.0 / x; x += 1.0; }
        if (x < 5.0) { result -= 1.0 / x; x += 1.0; }
        if (x < 6.0) { result -= 1.0 / x; x += 1.0; }
        if (x < 7.0) { result -= 1.0 / x; x += 1.0; }
    }
    // Asymptotic expansion: digamma(x) ~ log(x) - 1/(2x) - sum B_{2k}/(2k*x^{2k})
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    // Bernoulli coefficients: B_2/(2*x^2), B_4/(4*x^4), ...
    result += std::log(x) - 0.5 * inv_x
              - inv_x2 * (1.0/12.0
              - inv_x2 * (1.0/120.0
              - inv_x2 * (1.0/252.0
              - inv_x2 * (1.0/240.0
              - inv_x2 * (5.0/660.0
              - inv_x2 * (691.0/32760.0
              - inv_x2 * (1.0/12.0)))))));
    return result;
}

// Fused digamma + lgamma: compute both from the same recurrence shift.
// Saves log() calls by accumulating product in recurrence (single log vs up to 7).
// Returns {digamma(x), lgamma(x)}.
inline std::pair<double, double> portable_digamma_lgamma(double x) {
    if (x <= 0.0) return {-1e10, 1e10};

    double dig_result = 0.0;

    // Recurrence: digamma(x) = digamma(x+1) - 1/x
    //             lgamma(x)  = lgamma(x+1) - log(x)
    // Product trick: accumulate x*(x+1)*...*(x+k-1), take single log at end.
    // Saves up to 6 log() calls (each ~20-30 cycles).
    double product = 1.0;
    if (x < 7.0) {
        if (x < 1.0) { dig_result -= 1.0 / x; product *= x; x += 1.0; }
        if (x < 2.0) { dig_result -= 1.0 / x; product *= x; x += 1.0; }
        if (x < 3.0) { dig_result -= 1.0 / x; product *= x; x += 1.0; }
        if (x < 4.0) { dig_result -= 1.0 / x; product *= x; x += 1.0; }
        if (x < 5.0) { dig_result -= 1.0 / x; product *= x; x += 1.0; }
        if (x < 6.0) { dig_result -= 1.0 / x; product *= x; x += 1.0; }
        if (x < 7.0) { dig_result -= 1.0 / x; product *= x; x += 1.0; }
    }
    double lg_result = -std::log(product);  // Single log replaces up to 7

    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    double log_x = std::log(x);

    // Digamma asymptotic
    dig_result += log_x - 0.5 * inv_x
                  - inv_x2 * (1.0/12.0
                  - inv_x2 * (1.0/120.0
                  - inv_x2 * (1.0/252.0
                  - inv_x2 * (1.0/240.0
                  - inv_x2 * (5.0/660.0
                  - inv_x2 * (691.0/32760.0
                  - inv_x2 * (1.0/12.0)))))));

    // Lgamma asymptotic (Stirling): lgamma(x) = (x-0.5)*log(x) - x + 0.5*log(2*pi) + sum
    lg_result += (x - 0.5) * log_x - x + 0.9189385332046727  // 0.5*log(2*pi)
                 + inv_x * (1.0/12.0
                 - inv_x2 * (1.0/360.0
                 - inv_x2 * (1.0/1260.0
                 - inv_x2 * (1.0/1680.0
                 - inv_x2 * (1.0/1188.0)))));

    return {dig_result, lg_result};
}

// Portable trigamma using asymptotic expansion + recurrence
inline double portable_trigamma(double x) {
    if (x <= 0.0) return 1e10;  // guard
    double result = 0.0;
    // Use recurrence to shift x >= 7
    while (x < 7.0) {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    // Asymptotic expansion: trigamma(x) ~ 1/x + 1/(2x^2) + sum B_{2k}/(x^{2k+1})
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    result += inv_x + 0.5 * inv_x2
              + inv_x2 * inv_x * (1.0/6.0
              - inv_x2 * (1.0/30.0
              - inv_x2 * (1.0/42.0
              - inv_x2 * (1.0/30.0
              - inv_x2 * (5.0/66.0)))));
    return result;
}

// Portable tetragamma (psi'', the order-2 polygamma), same shape as the pair
// above: recurrence up to x >= 7, then the asymptotic series.
//
//   psi''(x) = psi''(x+1) - 2/x^3
//   psi''(x) ~ -[ 1/x^2 + 1/x^3 + sum_k B_2k (2k+1) / x^(2k+2) ]
//
// with B_2 = 1/6, B_4 = -1/30, B_6 = 1/42, B_8 = -1/30, B_10 = 5/66, giving the
// coefficients 1/2, -1/6, 1/6, -3/10, 5/6.
inline double portable_tetragamma(double x) {
    if (x <= 0.0) return -1e10;  // guard, matching portable_trigamma's sign
    double result = 0.0;
    while (x < 7.0) {
        result -= 2.0 / (x * x * x);
        x += 1.0;
    }
    const double inv_x  = 1.0 / x;
    const double inv_x2 = inv_x * inv_x;
    const double inv_x3 = inv_x2 * inv_x;
    result -= inv_x2 + inv_x3 + 0.5 * inv_x2 * inv_x2
              - inv_x3 * inv_x3 * (1.0/6.0
              - inv_x2 * (1.0/6.0
              - inv_x2 * (3.0/10.0
              - inv_x2 * (5.0/6.0))));
    return result;
}

// Portable pentagamma (psi''', the order-3 polygamma), completing the ladder.
//
//   psi'''(x) = psi'''(x+1) + 6/x^4
//   psi'''(x) ~ 2/x^3 + 3/x^4 + sum_k B_2k (2k+1)(2k+2) / x^(2k+3)
//
// with B_2 = 1/6, B_4 = -1/30, B_6 = 1/42, B_8 = -1/30, B_10 = 5/66, giving the
// coefficients 2, -1, 4/3, -3, 10.
inline double portable_pentagamma(double x) {
    if (x <= 0.0) return 1e10;  // guard
    double result = 0.0;
    while (x < 7.0) {
        result += 6.0 / (x * x * x * x);
        x += 1.0;
    }
    const double inv_x  = 1.0 / x;
    const double inv_x2 = inv_x * inv_x;
    const double inv_x3 = inv_x2 * inv_x;
    result += 2.0 * inv_x3 + 3.0 * inv_x3 * inv_x
              + inv_x3 * inv_x2 * (2.0
              - inv_x2 * (1.0
              - inv_x2 * (4.0/3.0
              - inv_x2 * (3.0
              - inv_x2 * (10.0)))));
    return result;
}

// Portable lchoose: log(C(n,k)) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
inline double portable_lchoose(int n, int k) {
    if (k < 0 || k > n) return -std::numeric_limits<double>::infinity();
    return std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
}

}  // namespace math
}  // namespace tulpa

#endif  // TULPA_PORTABLE_MATH_H
