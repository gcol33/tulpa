#ifndef TULPA_ST_HSGP_KERNEL_H
#define TULPA_ST_HSGP_KERNEL_H
#include <vector>
#include <tulpa/ad_scalar_math.h>
#include <tulpa/hsgp_spectral.h>
#include <tulpa/soft_sum_to_zero.h>
#include <tulpa/sum_to_zero.h>
#include <tulpa/st_margins.h>
#include <tulpa/temporal_nc.h>
namespace tulpa_st_hsgp {
using namespace tulpa::math;
template<class T>
T initial_precision(const T& rho) {
    const T value = T(1.0) - rho * rho;
    return get_value(value) < 1e-10 ? T(1e-10) : value;
}
template<class T>
void time_apply(const T* x, int nt, int type, bool cyclic, const T& rho, T* result) {
    for (int t = 0; t < nt; ++t) result[t] = T(0.0);
    if (type == 1 || type == 2) {
        const int count = cyclic ? nt : nt - (type == 2 ? 2 : 1);
        for (int t = 0; t < count; ++t) {
            const int a = t, b = (t + 1) % nt, c = (t + 2) % nt;
            if (type == 1) {
                const T d = x[b] - x[a];
                result[a] = result[a] - d; result[b] = result[b] + d;
            } else {
                const T d = x[c] - T(2.0) * x[b] + x[a];
                result[a] = result[a] + d; result[b] = result[b] - T(2.0) * d;
                result[c] = result[c] + d;
            }
        }
    } else {
        result[0] = initial_precision(rho) * x[0];
        for (int t = 1; t < nt; ++t) {
            const T d = x[t] - rho * x[t - 1];
            result[t] = result[t] + d; result[t - 1] = result[t - 1] - rho * d;
        }
    }
}
template<class T>
T coefficient_spectrum(double frequency, const T& sigma2, const T& lengthscale, int dimension, int kernel = 0) {
    const T value = tulpa_hsgp_spectral::spectral_density(frequency, sigma2, lengthscale, dimension, kernel);
    return get_value(value) < 1e-10 ? T(1e-10) : value;
}
template<class T>
T ar1_covariance_sums(const T& rho, int nt, std::vector<T>& sums) {
    sums.resize(nt);
    sums[nt - 1] = T(1.0);
    for (int t = nt - 2; t >= 0; --t) sums[t] = T(1.0) + rho * sums[t + 1];
    sums[0] = sums[0] / initial_precision(rho);
    T total = sums[0];
    for (int t = 1; t < nt; ++t) { sums[t] = sums[t] + rho * sums[t - 1]; total = total + sums[t]; }
    return total;
}
template<class T, class Data, class Basis>
void nc_forward(const T* raw, const T& tau, const T& rho, const T& sigma2,
    const T& lengthscale, const Data& st, const Basis& basis, T* field, int kernel = 0) {
    const int nt = st.n_times, type = static_cast<int>(st.temporal_type);
    const double lambda = tulpa::s2z_precision(nt);
    std::vector<T> covariance_sums;
    const T total = type == 3 ? ar1_covariance_sums(rho, nt, covariance_sums) : T(0.0);
    for (int j = 0; j < basis.m_total; ++j) {
        const T variance = coefficient_spectrum(basis.eigenvalues[j], sigma2, lengthscale, basis.n_dim, kernel) / tau;
        const T scale = safe_sqrt(variance);
        const std::vector<T> z(raw + j * nt, raw + (j + 1) * nt);
        T* x = field + j * nt;
        if (type == 3) {
            x[0] = scale * z[0] / safe_sqrt(initial_precision(rho));
            T sum = x[0];
            for (int t = 1; t < nt; ++t) { x[t] = rho * x[t - 1] + scale * z[t]; sum = sum + x[t]; }
            const T r = T(lambda) * variance * total;
            const T root = safe_sqrt(T(1.0) + r);
            const T contraction = r / (root * (root + T(1.0)));
            for (int t = 0; t < nt; ++t) x[t] = x[t] - contraction * covariance_sums[t] / total * sum;
        } else {
            tulpa_temporal_nc::rw_nc_forward(z.data(), nt, type, st.temporal_cyclic, scale, x);
            T sum(0.0), mean_coordinate(0.0);
            for (int t = 0; t < nt; ++t) sum = sum + x[t];
            if (st.temporal_cyclic) {
                for (int t = 0; t < nt; ++t) mean_coordinate = mean_coordinate + z[t];
                mean_coordinate = mean_coordinate / T(nt * std::sqrt(lambda * nt));
            } else mean_coordinate = z[nt - (type == 2 ? 2 : 1)] / T(nt * std::sqrt(lambda));
            for (int t = 0; t < nt; ++t) x[t] = x[t] - sum / T(nt) + mean_coordinate;
        }
    }
}
template<class Data, class Basis>
void nc_inverse(const double* field, double tau, double rho, double sigma2,
    double lengthscale, const Data& st, const Basis& basis, double* raw, int kernel = 0) {
    const int nt = st.n_times, type = static_cast<int>(st.temporal_type);
    const double lambda = tulpa::s2z_precision(nt);
    std::vector<double> covariance_sums;
    const double total = type == 3 ? ar1_covariance_sums(rho, nt, covariance_sums) : 0.0;
    for (int j = 0; j < basis.m_total; ++j) {
        const double variance = coefficient_spectrum(basis.eigenvalues[j], sigma2, lengthscale, basis.n_dim, kernel) / tau;
        const double scale = std::sqrt(variance);
        std::vector<double> x(field + j * nt, field + (j + 1) * nt);
        double sum = 0.0;
        for (double value : x) sum += value;
        double* z = raw + j * nt;
        if (type == 3) {
            const double r = lambda * variance * total;
            const double expansion = r / (std::sqrt(1.0 + r) + 1.0);
            for (int t = 0; t < nt; ++t) x[t] += expansion * covariance_sums[t] / total * sum;
            z[0] = x[0] * std::sqrt(initial_precision(rho)) / scale;
            for (int t = 1; t < nt; ++t) z[t] = (x[t] - rho * x[t - 1]) / scale;
        } else if (st.temporal_cyclic) {
            const double mean_coordinate = (sum / nt) * std::sqrt(lambda * nt);
            for (int t = 0; t < nt; ++t) {
                const int previous = (t + nt - 1) % nt;
                double difference = x[t] - x[previous];
                if (type == 2) difference = x[t] - 2.0 * x[previous] + x[(t + nt - 2) % nt];
                z[t] = difference / scale + mean_coordinate;
            }
        } else {
            for (int t = 0; t < nt - type; ++t)
                z[t] = (type == 1 ? x[t + 1] - x[t] : x[t + 2] - 2.0 * x[t + 1] + x[t]) / scale;
            z[nt - type] = sum * std::sqrt(lambda);
            if (type == 2) {
                double product = 0.0, norm = 0.0;
                for (int t = 0; t < nt; ++t) {
                    const double linear = tulpa_temporal_nc::lin_basis(t, nt);
                    product += x[t] * linear; norm += linear * linear;
                }
                z[nt - 1] = product / norm;
            }
        }
    }
}
template<class T, class Data, class Basis>
T log_density(const T* field, const T& tau, const T& rho,
    const T& sigma2, const T& lengthscale, const Data& st, const Basis& basis, int kernel = 0) {
    const int nt = st.n_times, type = static_cast<int>(st.temporal_type);
    const int rank = type == 1 ? tulpa::rw1_rank(nt, st.temporal_cyclic) :
        type == 2 ? tulpa::rw2_rank(nt, st.temporal_cyclic) : nt;
    T lp(0.0);
    std::vector<T> applied(nt), covariance_sums;
    const T covariance_sum = type == 3 ? ar1_covariance_sums(rho, nt, covariance_sums) : T(0.0);
    for (int j = 0; j < basis.m_total; ++j) {
        const T spectrum = coefficient_spectrum(basis.eigenvalues[j], sigma2, lengthscale, basis.n_dim, kernel);
        const T precision = tau / spectrum;
        const T* x = field + j * nt;
        T quad(0.0), sum(0.0);
        time_apply(x, nt, type, st.temporal_cyclic, rho, applied.data());
        for (int t = 0; t < nt; ++t) { sum = sum + x[t]; quad = quad + x[t] * applied[t]; }
        if (type == 3) {
            const T initial = initial_precision(rho);
            lp = lp + T(0.5) * safe_log(initial) + T(0.5) * safe_log(
                T(1.0) + T(tulpa::s2z_precision(nt)) * covariance_sum / precision);
        }
        lp = lp + T(0.5 * rank) * safe_log(precision) - T(0.5) * precision * quad
            - T(0.5 * tulpa::s2z_precision(nt)) * sum * sum;
    }
    return lp;
}
}
#endif
