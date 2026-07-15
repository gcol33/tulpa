// laplace_core_gp.cpp
// GP (NNGP) Laplace mode finder + its R export.
// Split from laplace_core.cpp on 2026-05-02.

#include "laplace_core.h"
#include "laplace_cholesky.h"
#include "laplace_newton.h"
#include "laplace_re_priors.h"
#include "laplace_scatter.h"
#include "laplace_temporal_priors.h"
#include "linalg_fast.h"
#include "gpu_nngp_laplace.h"
#include "laplace_spec_fit.h"   // as_offset_vec (offset marshalling)
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace tulpa {

// --- 4. GP (NNGP) ---
// Uses fully sparse Newton when n_spatial >= SPARSE_THRESHOLD (200).
// The H sparsity pattern comes from: beta×beta (dense p×p), spatial diagonal
// (likelihood), and NNGP neighbor pairs (prior). Total nnz ≈ p² + n_spatial × nn.

LaplaceResult laplace_mode_gp(
    const NumericVector& y, const IntegerVector& n,
    const NumericMatrix& X, const NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const NumericMatrix& coords,
    const IntegerMatrix& nn_idx, const NumericMatrix& nn_dist,
    const IntegerVector& nn_order,
    int n_spatial, int nn,
    double sigma2_gp, double phi_gp, int cov_type,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    const double* offset = nullptr,
    // Per-observation 1-based location index (length N). nullptr = identity
    // (obs i -> location i), the old behavior; a non-null map is required when
    // coordinates repeat (n_spatial unique locations < N), so an observation is
    // attached to its actual field node and later observations are not dropped.
    const int* obs_to_loc = nullptr
) {
    int N = y.size();
    int p = X.ncol();
    int n_x = p + n_re_groups + n_spatial;
    double tau_re = 1.0 / (sigma_re * sigma_re + 1e-10);
    int gp_start = p + n_re_groups;

    auto compute_eta = [&](const NumericVector& x, NumericVector& eta) {
        for (int i = 0; i < N; i++) {
            eta[i] = offset ? offset[i] : 0.0;
            for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
            if (n_re_groups > 0) {
                int g = (int)re_idx[i] - 1;
                if (g >= 0 && g < n_re_groups) eta[i] += x[p + g];
            }
            int loc = obs_to_loc ? (obs_to_loc[i] - 1) : i;
            if (loc >= 0 && loc < n_spatial) eta[i] += x[gp_start + loc];
        }
    };

    auto log_prior = [&](const NumericVector& x, const NumericVector&) {
        double lp = compute_log_prior_re(x, p, n_re_groups, tau_re);
        std::vector<double> w(n_spatial);
        for (int s = 0; s < n_spatial; s++) w[s] = x[gp_start + s];
        std::vector<double> cm, cv;
        bool gpu;
        batch_nngp_scatter(w, n_spatial, nn, sigma2_gp, phi_gp, cov_type,
                           coords, nn_idx, nn_dist, nn_order, cm, cv, gpu);
        for (int s = 0; s < n_spatial; s++) {
            double resid = w[s] - cm[s];
            lp += -0.5 * std::log(2.0 * M_PI * cv[s]) -
                  0.5 * resid * resid / cv[s];
        }
        return lp;
    };

    auto center = [](NumericVector&) {};

    // --- Sparse Newton path for large n_spatial ---
    if (n_x >= SPARSE_THRESHOLD) {
        // Build sparsity pattern: beta block + RE diagonal + GP diagonal + NNGP neighbors
        std::vector<std::pair<int,int>> pattern;

        // Beta-beta dense block
        for (int j1 = 0; j1 < p; j1++)
            for (int j2 = j1; j2 < p; j2++)
                pattern.push_back({j2, j1});

        // RE diagonal
        for (int g = 0; g < n_re_groups; g++)
            pattern.push_back({p + g, p + g});

        // Beta-GP cross terms (each obs contributes to its GP node × beta)
        for (int i = 0; i < std::min(N, n_spatial); i++) {
            int gp_idx = gp_start + i;
            for (int j = 0; j < p; j++)
                pattern.push_back({gp_idx, j});
        }

        // GP diagonal (likelihood + prior)
        for (int s = 0; s < n_spatial; s++)
            pattern.push_back({gp_start + s, gp_start + s});

        // NNGP neighbor + neighbor-of-neighbor pairs: the precision matrix
        // Λ = (I-A)' D⁻¹ (I-A) has off-diagonal entries at (focal, neighbor_k)
        // and at (neighbor_k, neighbor_kp) for every conditioning-set pair.
        // Without these, the Newton solve collapses to the diagonal-on-w
        // approximation and pointwise field recovery degrades.
        make_nngp_prior_sparsity_pattern(pattern, nn_idx, nn_order,
                                          n_spatial, nn, gp_start);

        SparseHessianBuilder H_builder;
        H_builder.init(n_x, pattern);

        auto scatter_sparse = [&](const NumericVector& x, const NumericVector& eta,
                                   DenseVec& grad, SparseHessianBuilder& H) {
            // Fixed effects
            for (int i = 0; i < N; i++) {
                auto gh = grad_hess_for_family(y[i], n[i], eta[i], family, phi);
                for (int j = 0; j < p; j++) {
                    grad[j] += gh.grad * X(i, j);
                    for (int k = 0; k <= j; k++) {
                        H.add(j, k, gh.neg_hess * X(i, j) * X(i, k));
                    }
                }
                // RE
                if (n_re_groups > 0) {
                    int g = (int)re_idx[i] - 1;
                    if (g >= 0 && g < n_re_groups) {
                        grad[p + g] += gh.grad;
                        H.add(p + g, p + g, gh.neg_hess);
                    }
                }
                // GP diagonal
                int loc = obs_to_loc ? (obs_to_loc[i] - 1) : i;
                if (loc >= 0 && loc < n_spatial) {
                    int gp_idx = gp_start + loc;
                    grad[gp_idx] += gh.grad;
                    H.add(gp_idx, gp_idx, gh.neg_hess);
                    // Cross with beta
                    for (int j = 0; j < p; j++) {
                        H.add(gp_idx, j, gh.neg_hess * X(i, j));
                    }
                }
            }

            // NNGP prior — full precision Λ = (I-A)' D⁻¹ (I-A).
            std::vector<double> w(n_spatial);
            for (int s = 0; s < n_spatial; s++) w[s] = x[gp_start + s];
            std::vector<double> cond_means, cond_vars, nngp_alpha;
            bool gpu_used;
            batch_nngp_scatter(w, n_spatial, nn, sigma2_gp, phi_gp, cov_type,
                               coords, nn_idx, nn_dist, nn_order,
                               cond_means, cond_vars, gpu_used, &nngp_alpha);
            apply_nngp_full_prior_sparse(grad, H, w, nngp_alpha, cond_vars,
                                           nn_idx, nn_order,
                                           n_spatial, nn, gp_start);

            // Beta + RE regularization
            double tau_beta = 1e-4;
            for (int j = 0; j < p; j++) { grad[j] -= tau_beta * x[j]; H.add(j, j, tau_beta); }
            for (int g = 0; g < n_re_groups; g++) {
                grad[p + g] -= tau_re * x[p + g]; H.add(p + g, p + g, tau_re);
            }
        };

        return laplace_newton_solve_sparse(
            y, n, family, phi, N, n_x,
            max_iter, tol, n_threads,
            compute_eta, scatter_sparse, center, log_prior,
            H_builder);
    }

    // --- Dense Newton path for small n_spatial ---
    auto scatter = [&](const NumericVector& x, const NumericVector& eta,
                       DenseVec& grad, DenseMat& H) {
        scatter_obs_grad_hess_base(y, n, X, re_idx, N, p, n_re_groups,
                                    eta, family, phi, grad, H, n_threads);
        for (int i = 0; i < N; i++) {
            int loc = obs_to_loc ? (obs_to_loc[i] - 1) : i;
            if (loc < 0 || loc >= n_spatial) continue;
            auto gh = grad_hess_for_family(y[i], n[i], eta[i], family, phi);
            int gp_idx = gp_start + loc;
            grad[gp_idx] += gh.grad;
            H[gp_idx][gp_idx] += gh.neg_hess;
        }
        std::vector<double> w(n_spatial);
        for (int s = 0; s < n_spatial; s++) w[s] = x[gp_start + s];
        std::vector<double> cond_means, cond_vars, nngp_alpha;
        bool gpu_used;
        batch_nngp_scatter(w, n_spatial, nn, sigma2_gp, phi_gp, cov_type,
                           coords, nn_idx, nn_dist, nn_order,
                           cond_means, cond_vars, gpu_used, &nngp_alpha);
        apply_nngp_full_prior_dense(grad, H, w, nngp_alpha, cond_vars,
                                      nn_idx, nn_order,
                                      n_spatial, nn, gp_start);
        add_re_beta_priors(grad, H, x, p, n_re_groups, tau_re);
    };

    return laplace_newton_solve(y, n, family, phi, N, n_x,
                                 max_iter, tol, n_threads,
                                 compute_eta, scatter, center, log_prior);
}


} // namespace tulpa

// =====================================================================
// R exports (call into tulpa:: functions defined above)
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_gp(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::NumericVector re_idx,
    int n_re_groups, double sigma_re,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order,
    int n_spatial, int nn,
    double sigma2_gp, double phi_gp, int cov_type,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> obs_to_loc_nullable = R_NilValue
) {
    std::vector<double> offset = tulpa::as_offset_vec(offset_nullable, y.size());
    std::vector<int> obs_to_loc;
    if (obs_to_loc_nullable.isNotNull()) {
        Rcpp::IntegerVector otl(obs_to_loc_nullable);
        obs_to_loc.assign(otl.begin(), otl.end());
    }
    tulpa::LaplaceResult result = tulpa::laplace_mode_gp(
        y, n, X, re_idx, n_re_groups, sigma_re,
        coords, nn_idx, nn_dist, nn_order, n_spatial, nn,
        sigma2_gp, phi_gp, cov_type,
        family, phi, max_iter, tol, n_threads,
        offset.empty() ? nullptr : offset.data(),
        obs_to_loc.empty() ? nullptr : obs_to_loc.data()
    );
    return tulpa::laplace_result_to_list(result);
}
