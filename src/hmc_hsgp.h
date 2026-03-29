// hmc_hsgp.h
// Hilbert Space Gaussian Process (HSGP) approximation
// Based on Riutort-Mayol et al. (2023) and Stan's implementation
//
// HSGP approximates GP as: f(x) = sum_j phi_j(x) * sqrt(S(lambda_j)) * beta_j
// where phi_j are Laplacian eigenfunctions and S is the spectral density

#ifndef TULPA_HMC_HSGP_H
#define TULPA_HMC_HSGP_H

#include <vector>
#include <cmath>
#include <RcppEigen.h>
#include "tulpa/hsgp_data.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa_hsgp {

using tulpa::HSGPData;

// Spectral density for squared exponential kernel
// S(omega) = sigma^2 * sqrt(2*pi) * ell * exp(-0.5 * ell^2 * omega^2)
inline double spectral_density_se(double omega_sq, double sigma2, double lengthscale) {
    double ell = lengthscale;
    double ell2 = ell * ell;
    return sigma2 * std::sqrt(2.0 * M_PI) * ell * std::exp(-0.5 * ell2 * omega_sq);
}

// Derivative of spectral density w.r.t. sigma2
// dS/d(sigma2) = sqrt(2*pi) * ell * exp(-0.5 * ell^2 * omega^2) = S / sigma2
inline double dS_dsigma2(double omega_sq, double sigma2, double lengthscale) {
    return spectral_density_se(omega_sq, sigma2, lengthscale) / sigma2;
}

// Derivative of spectral density w.r.t. lengthscale
// S = sigma2 * sqrt(2*pi) * ell * exp(-0.5 * ell^2 * omega^2)
// dS/d(ell) = sigma2 * sqrt(2*pi) * [exp(...) + ell * (-ell * omega^2) * exp(...)]
//           = S * (1/ell - ell * omega^2)
inline double dS_dlengthscale(double omega_sq, double sigma2, double lengthscale) {
    double S = spectral_density_se(omega_sq, sigma2, lengthscale);
    double ell = lengthscale;
    return S * (1.0 / ell - ell * omega_sq);
}

// 1D Laplacian eigenfunction: phi_j(x) = 1/sqrt(L) * sin(pi*j*(x+L)/(2L))
// For x in [-L, L], j = 1, 2, ...
inline double phi_1d(double x, int j, double L) {
    double norm = 1.0 / std::sqrt(L);
    return norm * std::sin(M_PI * j * (x + L) / (2.0 * L));
}

// 1D eigenvalue: lambda_j = (pi*j / (2*L))^2
inline double lambda_1d(int j, double L) {
    double tmp = M_PI * j / (2.0 * L);
    return tmp * tmp;
}

// Setup HSGP for 2D coordinates
// coords: flattened [x1, y1, x2, y2, ...] (length 2*n_obs)
// m: basis functions per dimension
// c: boundary factor (L = c * max_range)
inline void setup_hsgp_2d(
    const std::vector<double>& coords,
    int n_obs,
    int m,
    double c,
    bool shared,
    HSGPData& data
) {
    data.n_obs = n_obs;
    data.n_dim = 2;
    data.m_per_dim = m;
    data.m_total = m * m;
    data.shared = shared;

    // Find coordinate ranges
    double x_min = coords[0], x_max = coords[0];
    double y_min = coords[1], y_max = coords[1];
    for (int i = 1; i < n_obs; i++) {
        double x = coords[2*i];
        double y = coords[2*i + 1];
        if (x < x_min) x_min = x;
        if (x > x_max) x_max = x;
        if (y < y_min) y_min = y;
        if (y > y_max) y_max = y;
    }

    double x_range = x_max - x_min;
    double y_range = y_max - y_min;
    double x_center = (x_max + x_min) / 2.0;
    double y_center = (y_max + y_min) / 2.0;

    // Boundary factors
    data.L1 = c * x_range / 2.0;
    data.L2 = c * y_range / 2.0;

    // Ensure minimum boundary
    if (data.L1 < 0.1) data.L1 = 0.1;
    if (data.L2 < 0.1) data.L2 = 0.1;

    // Scale coordinates to [-L, L]
    data.coords_scaled.resize(2 * n_obs);
    for (int i = 0; i < n_obs; i++) {
        data.coords_scaled[2*i] = coords[2*i] - x_center;
        data.coords_scaled[2*i + 1] = coords[2*i + 1] - y_center;
    }

    // Compute eigenvalues for 2D: lambda_{j1,j2} = lambda_j1 + lambda_j2
    data.eigenvalues.resize(data.m_total);
    for (int j1 = 1; j1 <= m; j1++) {
        for (int j2 = 1; j2 <= m; j2++) {
            int idx = (j1 - 1) * m + (j2 - 1);
            data.eigenvalues[idx] = lambda_1d(j1, data.L1) + lambda_1d(j2, data.L2);
        }
    }

    // Compute basis matrix: phi[i, j] = phi_{j1}(x_i) * phi_{j2}(y_i)
    data.phi_flat.resize(n_obs * data.m_total);
    for (int i = 0; i < n_obs; i++) {
        double x = data.coords_scaled[2*i];
        double y = data.coords_scaled[2*i + 1];

        for (int j1 = 1; j1 <= m; j1++) {
            double phi_x = phi_1d(x, j1, data.L1);
            for (int j2 = 1; j2 <= m; j2++) {
                double phi_y = phi_1d(y, j2, data.L2);
                int j_idx = (j1 - 1) * m + (j2 - 1);
                data.phi_flat[i * data.m_total + j_idx] = phi_x * phi_y;
            }
        }
    }
}

// Evaluate HSGP spatial effects: f = Phi * (sqrt(S) ⊙ beta)
// Vectorized with Eigen matrix-vector product (BLAS/SIMD)
inline void hsgp_evaluate(
    const std::vector<double>& beta,
    double sigma2,
    double lengthscale,
    const HSGPData& data,
    std::vector<double>& f
) {
    const int N = data.n_obs;
    const int M = data.m_total;
    f.resize(N);

    // scaled_beta[j] = sqrt(S[j]) * beta[j]
    Eigen::VectorXd scaled_beta(M);
    for (int j = 0; j < M; j++) {
        double S = spectral_density_se(data.eigenvalues[j], sigma2, lengthscale);
        scaled_beta(j) = std::sqrt(S) * beta[j];
    }

    // f = Phi * scaled_beta  (single BLAS matvec)
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Phi(data.phi_flat.data(), N, M);
    Eigen::Map<Eigen::VectorXd> f_vec(f.data(), N);
    f_vec.noalias() = Phi * scaled_beta;
}

// Log prior on beta: N(0, I)
inline double hsgp_log_prior_beta(const std::vector<double>& beta) {
    double log_prior = 0.0;
    for (size_t j = 0; j < beta.size(); j++) {
        log_prior += -0.5 * beta[j] * beta[j];
    }
    return log_prior;
}

// Gradient structure for HSGP
struct HSGPGradients {
    std::vector<double> grad_beta;  // Gradient w.r.t. beta
    double grad_log_sigma2;         // Gradient w.r.t. log(sigma2)
    double grad_log_lengthscale;    // Gradient w.r.t. log(lengthscale)
};

// Pre-allocated workspace for HSGP gradient computation.
// Eliminates 9+ heap allocations per gradient call (~11KB for m=6, N=500).
// Use as thread_local in the gradient function.
struct HSGPWorkspace {
    int N = 0;  // observations
    int M = 0;  // basis functions

    // For hsgp_evaluate:
    std::vector<double> hsgp_beta;     // M
    Eigen::VectorXd scaled_beta;       // M

    // For gradient computation:
    std::vector<double> hsgp_f;        // N
    std::vector<double> grad_f;        // N
    Eigen::VectorXd PhiT_gf;           // M
    Eigen::VectorXd sqrt_S;            // M
    Eigen::VectorXd dsqrtS_dsigma2;    // M
    Eigen::VectorXd dsqrtS_dlengthscale; // M

    // Output gradients (avoids HSGPGradients allocation)
    std::vector<double> grad_beta_out; // M

    void init(int n_obs, int m_total) {
        if (n_obs == N && m_total == M) return;
        N = n_obs;
        M = m_total;
        hsgp_beta.resize(M);
        scaled_beta.resize(M);
        hsgp_f.resize(N);
        grad_f.resize(N);
        PhiT_gf.resize(M);
        sqrt_S.resize(M);
        dsqrtS_dsigma2.resize(M);
        dsqrtS_dlengthscale.resize(M);
        grad_beta_out.resize(M);
    }
};

// Evaluate HSGP spatial effects using workspace buffers
inline void hsgp_evaluate_ws(
    const double* beta,
    double sigma2,
    double lengthscale,
    const HSGPData& data,
    HSGPWorkspace& ws
) {
    const int N = data.n_obs;
    const int M = data.m_total;

    // Compute sqrt(S[j]) once and cache for reuse in gradient step
    for (int j = 0; j < M; j++) {
        double S = spectral_density_se(data.eigenvalues[j], sigma2, lengthscale);
        ws.sqrt_S(j) = std::sqrt(S);
        ws.scaled_beta(j) = ws.sqrt_S(j) * beta[j];
    }

    // f = Phi * scaled_beta  (single BLAS matvec)
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Phi(data.phi_flat.data(), N, M);
    Eigen::Map<Eigen::VectorXd> f_vec(ws.hsgp_f.data(), N);
    f_vec.noalias() = Phi * ws.scaled_beta;
}

// Compute HSGP gradients using workspace buffers (zero allocation)
inline void hsgp_compute_gradients_ws(
    const double* beta,
    double sigma2,
    double lengthscale,
    const HSGPData& data,
    HSGPWorkspace& ws,
    double& grad_log_sigma2,
    double& grad_log_lengthscale
) {
    const int N = data.n_obs;
    const int M = data.m_total;
    grad_log_sigma2 = 0.0;
    grad_log_lengthscale = 0.0;

    // Map Phi as Eigen matrix (N × M, row-major to match phi_flat layout)
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Phi(data.phi_flat.data(), N, M);
    Eigen::Map<const Eigen::VectorXd> gf(ws.grad_f.data(), N);

    // Phi^T * grad_f  (single BLAS matvec, reused for all 3 gradient components)
    ws.PhiT_gf.noalias() = Phi.transpose() * gf;

    // Reuse sqrt_S cached by hsgp_evaluate_ws() — no exp() or sqrt() needed
    for (int j = 0; j < M; j++) {
        double omega_sq = data.eigenvalues[j];
        const double eps = 1e-10;
        double sqrt_S_safe = std::max(ws.sqrt_S(j), eps);
        double S = sqrt_S_safe * sqrt_S_safe;

        ws.dsqrtS_dsigma2(j) = 0.5 * sqrt_S_safe / sigma2;

        // dS/d(ell) = S * (1/ell - ell * omega_sq) — inline, no spectral_density_se() call
        double dS_dell = S * (1.0 / lengthscale - lengthscale * omega_sq);
        ws.dsqrtS_dlengthscale(j) = 0.5 * dS_dell / sqrt_S_safe;
    }

    // grad_beta[j] = sqrt_S[j] * (Phi^T * grad_f)[j]
    Eigen::Map<Eigen::VectorXd> gb(ws.grad_beta_out.data(), M);
    gb = ws.sqrt_S.cwiseProduct(ws.PhiT_gf);

    // grad_log_sigma2 = sigma2 * (dsqrtS_dsigma2 ⊙ beta) · (Phi^T * grad_f)
    Eigen::Map<const Eigen::VectorXd> beta_vec(beta, M);
    grad_log_sigma2 = sigma2 * ws.dsqrtS_dsigma2.cwiseProduct(beta_vec).dot(ws.PhiT_gf);

    // grad_log_lengthscale = lengthscale * (dsqrtS_dlengthscale ⊙ beta) · (Phi^T * grad_f)
    grad_log_lengthscale = lengthscale * ws.dsqrtS_dlengthscale.cwiseProduct(beta_vec).dot(ws.PhiT_gf);
}

// Compute HSGP gradients analytically (vectorized with Eigen)
// Original interface preserved for backward compatibility
// grad_f: gradient of log-likelihood w.r.t. f (computed from likelihood)
inline void hsgp_compute_gradients(
    const std::vector<double>& beta,
    double sigma2,
    double lengthscale,
    const HSGPData& data,
    const std::vector<double>& grad_f,  // d(log_lik)/d(f_i)
    HSGPGradients& grads
) {
    const int N = data.n_obs;
    const int M = data.m_total;
    grads.grad_beta.resize(M);
    grads.grad_log_sigma2 = 0.0;
    grads.grad_log_lengthscale = 0.0;

    // Map Phi as Eigen matrix (N × M, row-major to match phi_flat layout)
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Phi(data.phi_flat.data(), N, M);
    Eigen::Map<const Eigen::VectorXd> gf(grad_f.data(), N);

    // Phi^T * grad_f  (single BLAS matvec, reused for all 3 gradient components)
    Eigen::VectorXd PhiT_gf = Phi.transpose() * gf;

    // Precompute sqrt(S) and derivatives
    Eigen::VectorXd sqrt_S(M);
    Eigen::VectorXd dsqrtS_dsigma2(M);
    Eigen::VectorXd dsqrtS_dlengthscale(M);

    for (int j = 0; j < M; j++) {
        double omega_sq = data.eigenvalues[j];
        double S = spectral_density_se(omega_sq, sigma2, lengthscale);
        sqrt_S(j) = std::sqrt(S);

        const double eps = 1e-10;
        double sqrt_S_safe = std::max(sqrt_S(j), eps);

        dsqrtS_dsigma2(j) = 0.5 * sqrt_S_safe / sigma2;

        double dS_dell = dS_dlengthscale(omega_sq, sigma2, lengthscale);
        dsqrtS_dlengthscale(j) = 0.5 * dS_dell / sqrt_S_safe;
    }

    // grad_beta[j] = sqrt_S[j] * (Phi^T * grad_f)[j]
    Eigen::Map<Eigen::VectorXd> gb(grads.grad_beta.data(), M);
    gb = sqrt_S.cwiseProduct(PhiT_gf);

    // grad_log_sigma2 = sigma2 * (dsqrtS_dsigma2 ⊙ beta) · (Phi^T * grad_f)
    Eigen::Map<const Eigen::VectorXd> beta_vec(beta.data(), M);
    grads.grad_log_sigma2 = sigma2 * dsqrtS_dsigma2.cwiseProduct(beta_vec).dot(PhiT_gf);

    // grad_log_lengthscale = lengthscale * (dsqrtS_dlengthscale ⊙ beta) · (Phi^T * grad_f)
    grads.grad_log_lengthscale = lengthscale * dsqrtS_dlengthscale.cwiseProduct(beta_vec).dot(PhiT_gf);
}

} // namespace tulpa_hsgp

#endif // TULPA_HMC_HSGP_H
