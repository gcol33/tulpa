// temporal_gp_kernel.h
// Covariance kernels for the temporal Gaussian process on irregularly-spaced
// times, templated over the scalar type so the sampled (sigma2, phi) can be
// autodiff variables.
//
// The exponential kernel is an Ornstein-Uhlenbeck process, so its joint density
// factorizes into a first-order Markov chain and the prior evaluates in O(T)
// with no matrix at all. That recursion lives in tulpa_priors_temporal.h and is
// the fast path. The remaining kernels have no finite-dimensional state-space
// form, so they are evaluated densely: build the T x T covariance here, factor
// it, and read the multivariate-normal density off the factor.
//
// Only smoothnesses with a closed form are offered. Matern is defined for
// nu in {0.5, 1.5, 2.5} -- general nu needs a Bessel function of the second
// kind, which has no autodiff-friendly form here. nu = 0.5 IS the exponential
// kernel, so it routes to the state-space path rather than the dense one.
// R rejects every other nu at construction; this header is the definition those
// checks are written against.

#ifndef TULPA_TEMPORAL_GP_KERNEL_H
#define TULPA_TEMPORAL_GP_KERNEL_H

#include <cmath>
#include <vector>

#include "autodiff_utils.h"
#include "nngp_cond.h"        // templated chol_decomp / solve_lower
#include "tulpa/portable_math.h"
#include "tulpa/types.h"

namespace tulpa_temporal_gp {

using tulpa::TemporalCovType;
using namespace tulpa::math;

// Whether a kernel is evaluated by the O(T) Markov recursion (exponential /
// Matern nu = 1/2) or by a dense factorization.
inline bool cov_is_markov(TemporalCovType cov_type, double nu) {
    if (cov_type == TemporalCovType::EXPONENTIAL) return true;
    return cov_type == TemporalCovType::MATERN && std::abs(nu - 0.5) < 1e-12;
}

// k(d) for a time lag d >= 0. `sigma2` is the marginal variance, `phi` the
// lengthscale, `nu` the Matern smoothness and `period` the periodic kernel's
// period; the last two are read only by the kernel that uses them.
template <typename T>
inline T temporal_cov(double d, const T& sigma2, const T& phi,
                      TemporalCovType cov_type, double nu, double period) {
    const T r = T(d) / phi;
    switch (cov_type) {
        case TemporalCovType::MATERN: {
            if (std::abs(nu - 1.5) < 1e-12) {
                const T u = T(std::sqrt(3.0)) * r;
                return sigma2 * (T(1.0) + u) * safe_exp(-u);
            }
            if (std::abs(nu - 2.5) < 1e-12) {
                const T u = T(std::sqrt(5.0)) * r;
                return sigma2 * (T(1.0) + u + u * u / T(3.0)) * safe_exp(-u);
            }
            // nu = 1/2 collapses to the exponential kernel; every other nu is
            // rejected in R, so this is the only remaining case.
            return sigma2 * safe_exp(-r);
        }
        case TemporalCovType::GAUSSIAN:
            return sigma2 * safe_exp(-r * r);
        case TemporalCovType::PERIODIC: {
            const double s = std::sin(M_PI * d / period);
            return sigma2 * safe_exp(T(-2.0 * s * s) / (phi * phi));
        }
        case TemporalCovType::EXPONENTIAL:
        default:
            return sigma2 * safe_exp(-r);
    }
}

// Lower Cholesky of the T x T covariance over `time_values`. Returns false when
// the matrix is not numerically PD, which a caller turns into -Inf rather than
// a NaN gradient. The jitter matches the NNGP conditional kernels.
template <typename T>
inline bool temporal_cov_chol(const std::vector<double>& time_values,
                              int n_times, const T& sigma2, const T& phi,
                              TemporalCovType cov_type, double nu,
                              double period, std::vector<T>& L) {
    std::vector<T> K(static_cast<std::size_t>(n_times) * n_times);
    for (int i = 0; i < n_times; i++) {
        for (int j = 0; j <= i; j++) {
            const double d = std::abs(time_values[i] - time_values[j]);
            const T k = temporal_cov(d, sigma2, phi, cov_type, nu, period);
            K[static_cast<std::size_t>(i) * n_times + j] = k;
            K[static_cast<std::size_t>(j) * n_times + i] = k;
        }
    }
    return tulpa_nngp::chol_decomp(K, n_times, L, 1e-8);
}

// log N(f; 0, K) for one group, given the factor L of K.
//   -0.5 * (T log(2 pi) + 2 sum log L_ii + ||L^-1 f||^2)
template <typename T>
inline T dense_gp_log_density(const std::vector<T>& L, int n_times,
                              const std::vector<T>& f, int offset) {
    std::vector<T> f_g(n_times);
    for (int t = 0; t < n_times; t++) f_g[t] = f[offset + t];

    std::vector<T> z;
    tulpa_nngp::solve_lower(L, n_times, f_g, z);

    T quad = T(0.0);
    for (int t = 0; t < n_times; t++) quad = quad + z[t] * z[t];

    T log_det = T(0.0);
    for (int t = 0; t < n_times; t++) {
        log_det = log_det + safe_log(L[static_cast<std::size_t>(t) * n_times + t]);
    }

    return -T(0.5) * (T(n_times * std::log(2.0 * M_PI)) + T(2.0) * log_det + quad);
}

// Non-centered forward transform f = L z for one group.
template <typename T>
inline void dense_gp_forward(const std::vector<T>& L, int n_times,
                             const std::vector<T>& z, int offset,
                             std::vector<T>& f_out) {
    for (int i = 0; i < n_times; i++) {
        T acc = T(0.0);
        for (int j = 0; j <= i; j++) {
            acc = acc + L[static_cast<std::size_t>(i) * n_times + j] * z[offset + j];
        }
        f_out[offset + i] = acc;
    }
}

}  // namespace tulpa_temporal_gp

#endif  // TULPA_TEMPORAL_GP_KERNEL_H
