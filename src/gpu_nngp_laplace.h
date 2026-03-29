// gpu_nngp_laplace.h
// GPU-accelerated NNGP for Laplace approximation.
// Batches the m×m neighbor covariance Cholesky factorizations on GPU.
// Falls back to CPU transparently if CUDA unavailable.

#ifndef TULPA_GPU_NNGP_LAPLACE_H
#define TULPA_GPU_NNGP_LAPLACE_H

#include "gpu_cuda.h"
#include <Rcpp.h>
#include <vector>
#include <cmath>

namespace tulpa {

// Compute covariance value
inline double nngp_cov_gpu(double d, double sigma2, double phi, int cov_type) {
    if (d < 1e-10) return sigma2;
    if (cov_type == 0) return sigma2 * std::exp(-d / phi);
    if (cov_type == 1) {
        double x = std::sqrt(3.0) * d / phi;
        return sigma2 * (1.0 + x) * std::exp(-x);
    }
    double x = std::sqrt(5.0) * d / phi;
    return sigma2 * (1.0 + x + x * x / 3.0) * std::exp(-x);
}

// Batch-compute all NNGP conditional means and variances.
// Uses GPU for the Cholesky step if available, CPU otherwise.
// w: spatial effects (n_spatial). Results: cond_mean[i], cond_var[i] for i in original order.
inline void batch_nngp_scatter(
    const std::vector<double>& w,
    int n_spatial, int nn,
    double sigma2, double phi_gp, int cov_type,
    const Rcpp::NumericMatrix& coords,
    const Rcpp::IntegerMatrix& nn_idx,
    const Rcpp::NumericMatrix& nn_dist,
    const Rcpp::IntegerVector& nn_order,
    std::vector<double>& cond_mean_out,
    std::vector<double>& cond_var_out,
    bool& gpu_used
) {
    cond_mean_out.assign(n_spatial, 0.0);
    cond_var_out.assign(n_spatial, sigma2);
    gpu_used = false;

    // Phase 1: Build all C matrices and c vectors on CPU
    struct LocData {
        int nngp_idx;     // index in NNGP ordering
        int obs_idx;      // original location index
        int n_nb;         // actual neighbor count
    };
    std::vector<LocData> locs;
    std::vector<std::vector<double>> C_mats;  // flattened nn×nn
    std::vector<std::vector<double>> c_vecs;  // length nn

    for (int i = 0; i < n_spatial; i++) {
        int obs_idx = nn_order[i];
        int n_nb = 0;
        for (int j = 0; j < nn; j++) {
            if (nn_idx(i, j) > 0) n_nb++;
        }
        if (n_nb == 0) continue;

        std::vector<double> C(nn * nn, 0.0);
        std::vector<double> c(nn, 0.0);

        for (int j = 0; j < n_nb; j++) {
            c[j] = nngp_cov_gpu(nn_dist(i, j), sigma2, phi_gp, cov_type);
        }
        for (int j1 = 0; j1 < n_nb; j1++) {
            int o1 = nn_order[nn_idx(i, j1) - 1];
            C[j1 * nn + j1] = sigma2;
            for (int j2 = j1 + 1; j2 < n_nb; j2++) {
                int o2 = nn_order[nn_idx(i, j2) - 1];
                double d12 = std::sqrt(
                    std::pow(coords(o1, 0) - coords(o2, 0), 2) +
                    std::pow(coords(o1, 1) - coords(o2, 1), 2));
                double cv = nngp_cov_gpu(d12, sigma2, phi_gp, cov_type);
                C[j1 * nn + j2] = cv;
                C[j2 * nn + j1] = cv;
            }
        }
        // Pad unused diagonal for stability
        for (int j = n_nb; j < nn; j++) C[j * nn + j] = 1.0;

        locs.push_back({i, obs_idx, n_nb});
        C_mats.push_back(std::move(C));
        c_vecs.push_back(std::move(c));
    }

    int batch_size = static_cast<int>(locs.size());
    if (batch_size == 0) return;

    // Phase 2: Cholesky factorize — GPU or CPU
    // After this, C_mats[b] contains L (lower triangular)
    bool chol_ok = false;
    if (batch_size >= 50) {  // GPU overhead threshold
        chol_ok = tulpa_gpu::cuda_batched_cholesky(C_mats, nn);
        if (chol_ok) gpu_used = true;
    }

    if (!chol_ok) {
        // CPU Cholesky per matrix
        for (int b = 0; b < batch_size; b++) {
            auto& L = C_mats[b];
            int n_nb = locs[b].n_nb;
            // In-place Cholesky (only n_nb×n_nb block)
            for (int j = 0; j < n_nb; j++) {
                for (int k = 0; k <= j; k++) {
                    double sum = L[j * nn + k];
                    for (int m = 0; m < k; m++) sum -= L[j * nn + m] * L[k * nn + m];
                    if (j == k) {
                        L[j * nn + j] = std::sqrt(std::max(1e-10, sum));
                    } else {
                        L[j * nn + k] = sum / L[k * nn + k];
                        L[k * nn + j] = 0.0;  // zero upper triangle
                    }
                }
            }
        }
    }

    // Phase 3: Solve and extract conditional mean/variance (CPU, O(m²) per loc)
    for (int b = 0; b < batch_size; b++) {
        auto& L = C_mats[b];
        auto& c = c_vecs[b];
        int n_nb = locs[b].n_nb;
        int obs_idx = locs[b].obs_idx;
        int nngp_idx = locs[b].nngp_idx;

        // Forward solve: L y = c
        std::vector<double> y(n_nb);
        for (int j = 0; j < n_nb; j++) {
            double sum = c[j];
            for (int k = 0; k < j; k++) sum -= L[j * nn + k] * y[k];
            y[j] = sum / L[j * nn + j];
        }

        // Back solve: L' alpha = y
        std::vector<double> alpha(n_nb);
        for (int j = n_nb - 1; j >= 0; j--) {
            double sum = y[j];
            for (int k = j + 1; k < n_nb; k++) sum -= L[k * nn + j] * alpha[k];
            alpha[j] = sum / L[j * nn + j];
        }

        // Conditional mean: alpha' * w_neighbors
        double cm = 0.0;
        for (int j = 0; j < n_nb; j++) {
            int nn_val = nn_idx(nngp_idx, j);
            if (nn_val <= 0 || nn_val > n_spatial) continue;
            int nn_orig = nn_order[nn_val - 1];
            if (nn_orig < 0 || nn_orig >= n_spatial) continue;
            cm += alpha[j] * w[nn_orig];
        }
        cond_mean_out[obs_idx] = cm;

        // Conditional variance: sigma2 - c' alpha
        double c_alpha = 0.0;
        for (int j = 0; j < n_nb; j++) c_alpha += c[j] * alpha[j];
        cond_var_out[obs_idx] = std::max(1e-10, sigma2 - c_alpha);
    }
}

} // namespace tulpa

#endif // TULPA_GPU_NNGP_LAPLACE_H
