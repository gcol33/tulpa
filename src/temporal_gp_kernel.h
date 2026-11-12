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
#include <algorithm>

#include "autodiff_utils.h"
#include "hmc_temporal.h"     // ar1_one_minus_rho2, the shared correlation floor
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
                              double period, std::vector<T>& L, double relative_nugget = 0.0) {
    std::vector<T> K(static_cast<std::size_t>(n_times) * n_times);
    for (int i = 0; i < n_times; i++) {
        for (int j = 0; j <= i; j++) {
            const double d = std::abs(time_values[i] - time_values[j]);
            const T k = temporal_cov(d, sigma2, phi, cov_type, nu, period);
            K[static_cast<std::size_t>(i) * n_times + j] = k;
            K[static_cast<std::size_t>(j) * n_times + i] = k;
        }
    }
    if (relative_nugget > 0)
        for (int i = 0; i < n_times; ++i)
            K[static_cast<std::size_t>(i) * n_times + i] =
                K[static_cast<std::size_t>(i) * n_times + i] + sigma2 * T(relative_nugget);
    return tulpa_nngp::chol_decomp(K, n_times, L, relative_nugget > 0 ? 0.0 : 1e-8);
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

// ---------------------------------------------------------------------------
// The Ornstein-Uhlenbeck (exponential / Matern nu = 1/2) chain
// ---------------------------------------------------------------------------
//
// One transition per consecutive pair of `time_values`: the correlation
// rho_t = exp(-dt / phi) and the conditional-variance factor 1 - rho_t^2. Both
// parameterizations read this ONE pair, so the non-centered transform's scale
// a_t = sigma sqrt(1 - rho_t^2) and the centered branch's conditional variance
// sigma^2 (1 - rho_t^2) are the same number, and the floor -- the shared AR1
// one, on the correlation factor rather than on a precision -- binds at the
// same place on both. Flooring different quantities leaves the two orders of
// magnitude apart wherever a long lengthscale meets a fine time grid, which is
// the identity `test-temporal-gp-parameterization.R` pins.
//
// The chain depends only on (time_values, phi), so a field with several
// independent blocks over the SAME instants builds it once and reads it per
// block. `rho` and `omr2` come back with n_times - 1 entries (empty at T <= 1).
template <typename T>
inline void ou_chain(const std::vector<double>& time_values, int n_times,
                     const T& phi, std::vector<T>& rho, std::vector<T>& omr2) {
    const int n_step = n_times > 1 ? n_times - 1 : 0;
    rho.resize(n_step);
    omr2.resize(n_step);
    for (int t = 1; t < n_times; t++) {
        const double dt = time_values[t] - time_values[t - 1];
        rho[t - 1]  = safe_exp(T(-dt) / phi);
        omr2[t - 1] = tulpa_temporal::ar1_one_minus_rho2(rho[t - 1]);
    }
}

// log N(f; 0, K) for one block of an OU chain, centered parameterization:
// the marginal N(0, sigma2) at the first instant and one conditional
// N(rho_t f_{t-1}, sigma2 (1 - rho_t^2)) per transition.
template <typename T>
inline T ou_log_density(const T* f, int n_times, const T& sigma2,
                        const std::vector<T>& rho,
                        const std::vector<T>& omr2) {
    T lp = -T(0.5) * safe_log(T(2.0 * M_PI) * sigma2)
           - T(0.5) * f[0] * f[0] / sigma2;
    for (int t = 1; t < n_times; t++) {
        const T cond_var = sigma2 * omr2[t - 1];
        const T resid    = f[t] - rho[t - 1] * f[t - 1];
        lp = lp - T(0.5) * safe_log(T(2.0 * M_PI) * cond_var)
                - T(0.5) * resid * resid / cond_var;
    }
    return lp;
}

// Non-centered forward transform z -> f for one block of an OU chain:
// f[0] = sigma z[0], f[t] = rho_t f[t-1] + a_t z[t] with a_t the scale
// `ou_forward_scale` builds from the SAME `omr2` the density reads.
template <typename T>
inline void ou_forward_scale(const T& sigma, const std::vector<T>& omr2,
                             std::vector<T>& a) {
    a.resize(omr2.size());
    for (std::size_t t = 0; t < omr2.size(); t++) a[t] = sigma * safe_sqrt(omr2[t]);
}

template <typename T>
inline void ou_forward(const T* z, int n_times, const T& sigma,
                       const std::vector<T>& rho, const std::vector<T>& a,
                       T* f_out) {
    f_out[0] = sigma * z[0];
    for (int t = 1; t < n_times; t++) {
        f_out[t] = rho[t - 1] * f_out[t - 1] + a[t - 1] * z[t];
    }
}

// Non-centered forward transform f = L z for one group.
template <typename T>
inline void dense_gp_forward(const std::vector<T>& L, int n_times,
                             const T* z, T* f_out) {
    for (int i = 0; i < n_times; i++) {
        T acc = T(0.0);
        for (int j = 0; j <= i; j++) {
            acc = acc + L[static_cast<std::size_t>(i) * n_times + j] * z[j];
        }
        f_out[i] = acc;
    }
}

template <typename T>
inline void dense_gp_forward(const std::vector<T>& L, int n_times,
                             const std::vector<T>& z, int offset,
                             std::vector<T>& f_out) {
    dense_gp_forward(L, n_times, z.data() + offset, f_out.data() + offset);
}

// A covariance factor shared by independent fields on one time axis.
template <typename T>
class FieldKernel {
    int n_;
    bool markov_, valid_, separate_level_;
    T sigma2_, sigma_;
    std::vector<T> rho_, omr2_, a_, L_, level_axis_;
    T level_norm2_;

public:
    FieldKernel(const std::vector<double>& times, int n_times,
                const T& sigma2, const T& phi, TemporalCovType cov,
                double nu, double period, double relative_nugget = 0.0,
                bool separate_level = false)
        : n_(n_times), markov_(cov_is_markov(cov, nu)), valid_(true),
          separate_level_(separate_level),
          sigma2_(sigma2), sigma_(safe_sqrt(sigma2)) {
        if (markov_) {
            ou_chain(times, n_, phi, rho_, omr2_);
            ou_forward_scale(sigma_, omr2_, a_);
        } else {
            valid_ = temporal_cov_chol(times, n_, sigma2, phi, cov, nu,
                                       period, L_, relative_nugget);
        }
        if (!valid_ || !separate_level_) return;
        level_axis_.resize(n_);
        if (markov_) {
            level_axis_[0] = T(1.0);
            for (int t = 1; t < n_; ++t)
                level_axis_[t] = (T(1.0) - rho_[t - 1]) / safe_sqrt(omr2_[t - 1]);
        } else {
            std::vector<T> constant(n_, T(1.0));
            tulpa_nngp::solve_lower(L_, n_, constant, level_axis_);
            const T first = level_axis_[0];
            for (auto& v : level_axis_) v = v / first;
        }
        T norm2 = T(0.0);
        for (const auto& v : level_axis_) norm2 = norm2 + v * v;
        level_axis_[0] = level_axis_[0] + safe_sqrt(norm2);
        level_norm2_ = T(0.0);
        for (const auto& v : level_axis_) level_norm2_ = level_norm2_ + v * v;
    }

    bool valid() const { return valid_; }

    // H maps the first innovation onto L^-1 1, so L H z carries the
    // constant field direction in one independent Gaussian coordinate.
    void forward(const T* innovation, T* field) const {
        if (!separate_level_) {
            if (markov_) ou_forward(innovation, n_, sigma_, rho_, a_, field);
            else dense_gp_forward(L_, n_, innovation, field);
            return;
        }
        T projection = T(0.0);
        for (int t = 0; t < n_; ++t) projection = projection + level_axis_[t] * innovation[t];
        projection = T(2.0) * projection / level_norm2_;
        std::vector<T> rotated(n_);
        for (int t = 0; t < n_; ++t)
            rotated[t] = innovation[t] - projection * level_axis_[t];
        if (markov_) ou_forward(rotated.data(), n_, sigma_, rho_, a_, field);
        else dense_gp_forward(L_, n_, rotated.data(), field);
    }

    // Non-centred coordinates have a unit Gaussian prior. The returned field
    // is f = L H z; its density carries no transformation Jacobian.
    void accumulate_log_density(std::vector<T>& field, int offset,
                                bool noncentered, T& lp) const {
        if (!noncentered) {
            lp = lp + (markov_ ? ou_log_density(field.data() + offset, n_, sigma2_, rho_, omr2_)
                              : dense_gp_log_density(L_, n_, field, offset));
            return;
        }
        for (int t = 0; t < n_; ++t)
            lp = lp - T(0.5) * field[offset + t] * field[offset + t];
        std::vector<T> reconstructed(n_);
        forward(field.data() + offset, reconstructed.data());
        std::copy(reconstructed.begin(), reconstructed.end(), field.begin() + offset);
    }
};

}  // namespace tulpa_temporal_gp

#endif  // TULPA_TEMPORAL_GP_KERNEL_H
