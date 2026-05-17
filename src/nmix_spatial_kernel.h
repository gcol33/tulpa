// nmix_spatial_kernel.h
// Spatial-extended kernel sweep and Hessian assembly for the Royle (2004)
// N-mixture model with a latent abundance-arm offset.
//
// State vector layout (n_x = p_lam + p_p + n_spatial):
//
//   x[0           : p_lam]                   = beta_lambda
//   x[p_lam       : p_lam + p_p]             = beta_p
//   x[p_lam + p_p : p_lam + p_p + n_spatial] = z   (spatial offset on lambda)
//
// Per-site linear predictors:
//
//   eta_lambda[s] = X_lambda[s,] . beta_lambda + z[ map_site_to_unit(s) ]
//   eta_p[ij]     = X_p[ij,]    . beta_p
//
// Score / Fisher / observed-information derivations -- see comments at top
// of nmix_laplace.cpp (non-spatial). The spatial extension simply adds one
// extra coordinate to e_lambda for each site (the indicator of its spatial
// unit), so the rank-1 marginal correction u_s u_s^T still describes the
// cross-arm coupling, with u_s now carrying a -1 at the z-coordinate of
// site s's spatial unit. The complete-data Fisher block is block-diagonal
// across (beta_lambda, beta_p) and exposes the lambda-z cross only.
//
// The ICAR / BYM2 / CAR-proper prior contributions are added by the per-
// grid-point driver via the existing laplace_spatial_priors.h helpers.

#ifndef TULPA_NMIX_SPATIAL_KERNEL_H
#define TULPA_NMIX_SPATIAL_KERNEL_H

#include "nmix_kernel.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace tulpa {

// One per-site kernel pass at the current (beta_lambda, beta_p, z).
// Returns total log-lik; fills per-site / per-visit grad and info vectors.
inline double nmix_kernel_sweep_spatial(
    const std::vector<std::vector<int>>& obs_by_site,
    const Rcpp::IntegerVector& y_R,
    const Eigen::VectorXd& eta_lambda,
    const Eigen::VectorXd& eta_p_long,
    int K_max,
    Eigen::VectorXd& grad_eta_lam,    // out [n_sites]
    Eigen::VectorXd& info_eta_lam,    // out [n_sites]
    Eigen::VectorXd& grad_eta_p,      // out [n_obs]
    Eigen::VectorXd& info_eta_p,      // out [n_obs]
    Eigen::VectorXd& mean_N,          // out [n_sites]
    Eigen::VectorXd& var_N,           // out [n_sites]
    Eigen::VectorXd& boundary_weight  // out [n_sites]
) {
    const int n_sites = static_cast<int>(obs_by_site.size());
    double log_lik = 0.0;
    for (int s = 0; s < n_sites; ++s) {
        const auto& idx = obs_by_site[s];
        const int J = static_cast<int>(idx.size());
        if (J == 0) {
            grad_eta_lam(s) = 0.0;
            info_eta_lam(s) = 0.0;
            mean_N(s) = std::exp(eta_lambda(s));
            var_N(s)  = std::exp(eta_lambda(s));
            boundary_weight(s) = 0.0;
            continue;
        }
        std::vector<int>    y_site(J);
        std::vector<double> eta_p_site(J);
        for (int j = 0; j < J; ++j) {
            y_site[j]     = y_R[idx[j]];
            eta_p_site[j] = eta_p_long(idx[j]);
        }
        NMixSiteResult r = compute_nmix_site(
            y_site.data(), eta_p_site.data(), J,
            eta_lambda(s), K_max
        );
        log_lik += r.log_lik;
        grad_eta_lam(s)    = r.grad_eta_lambda;
        info_eta_lam(s)    = r.info_eta_lambda;
        mean_N(s)          = r.mean_N;
        var_N(s)           = r.var_N;
        boundary_weight(s) = r.boundary_weight;
        for (int j = 0; j < J; ++j) {
            grad_eta_p(idx[j]) = r.grad_eta_p[j];
            info_eta_p(idx[j]) = r.info_eta_p[j];
        }
    }
    return log_lik;
}

// Cheap log-lik-only sweep at trial points (line search).
inline double nmix_kernel_log_lik_only_spatial(
    const std::vector<std::vector<int>>& obs_by_site,
    const Rcpp::IntegerVector& y_R,
    const Eigen::VectorXd& eta_lambda,
    const Eigen::VectorXd& eta_p_long,
    int K_max
) {
    double log_lik = 0.0;
    const int n_sites = static_cast<int>(obs_by_site.size());
    for (int s = 0; s < n_sites; ++s) {
        const auto& idx = obs_by_site[s];
        const int J = static_cast<int>(idx.size());
        if (J == 0) continue;
        std::vector<int>    y_site(J);
        std::vector<double> eta_p_site(J);
        for (int j = 0; j < J; ++j) {
            y_site[j]     = y_R[idx[j]];
            eta_p_site[j] = eta_p_long(idx[j]);
        }
        NMixSiteResult r = compute_nmix_site(
            y_site.data(), eta_p_site.data(), J,
            eta_lambda(s), K_max
        );
        if (!R_finite(r.log_lik)) return r.log_lik;
        log_lik += r.log_lik;
    }
    return log_lik;
}

// Compute eta_lambda[s] = X_lambda[s,] . beta_lambda + z[ map[s] ].
// `map_site_to_unit` is 0-based: site s's spatial unit is `map[s]`. Pass
// a length-n_sites vector with values in [0, n_spatial).
inline void compute_eta_lambda_spatial(
    const Eigen::Map<Eigen::MatrixXd>& X_lambda,
    const Eigen::VectorXd& beta_lambda,
    const Eigen::VectorXd& z,
    const std::vector<int>& map_site_to_unit,
    Eigen::VectorXd& eta_lambda /* out, length n_sites */
) {
    eta_lambda.noalias() = X_lambda * beta_lambda;
    const int n_sites = static_cast<int>(map_site_to_unit.size());
    for (int s = 0; s < n_sites; ++s) {
        eta_lambda(s) += z(map_site_to_unit[s]);
    }
}

// Assemble the marginal observed-information Hessian over
//   theta = (beta_lambda [p_lam], beta_p [p_p], z [n_spatial])
//
// For each site s the per-site contribution is
//   H_s = D_s  -  Var[N|y_s] u_s u_s^T
// with D_s the block-diagonal complete-data Fisher (no cross to z except via
// lambda) and u_s the (-eta_lambda gradient, +sum_j p_ij eta_p gradient)
// vector projected through the spatial mapping (a -1 at z-position map[s]).
//
// ICAR / BYM2 / CAR prior contributions on z are added separately by the
// caller, so this function only handles the likelihood-side scatter.
inline void nmix_assemble_obs_info_spatial(
    int p_lam, int p_p, int n_spatial,
    const Eigen::Map<Eigen::MatrixXd>& X_lambda,
    const Eigen::Map<Eigen::MatrixXd>& X_p,
    const Eigen::VectorXd& eta_p_long,
    const std::vector<std::vector<int>>& obs_by_site,
    const std::vector<int>& map_site_to_unit,
    const Eigen::VectorXd& info_eta_lam,
    const Eigen::VectorXd& info_eta_p,
    const Eigen::VectorXd& var_N,
    Eigen::MatrixXd& H_obs /* in/out: zero-initialized [n_x x n_x] */
) {
    const int n_sites = static_cast<int>(obs_by_site.size());
    const int z_start = p_lam + p_p;

    for (int s = 0; s < n_sites; ++s) {
        const auto& idx = obs_by_site[s];
        const int J = static_cast<int>(idx.size());
        if (J == 0) continue;
        const int u = map_site_to_unit[s];

        // ----- Complete-data Fisher D_s -----
        double w_lam = info_eta_lam(s);
        if (w_lam > 0.0) {
            // beta_lambda x beta_lambda block
            H_obs.block(0, 0, p_lam, p_lam)
                .selfadjointView<Eigen::Lower>()
                .rankUpdate(X_lambda.row(s).transpose(), w_lam);
            // z[u] x z[u] (diagonal entry)
            H_obs(z_start + u, z_start + u) += w_lam;
            // beta_lambda x z[u] cross-term: w_lam * X_lambda[s,]
            for (int k = 0; k < p_lam; ++k) {
                double v = w_lam * X_lambda(s, k);
                H_obs(k, z_start + u) += v;
                H_obs(z_start + u, k) += v;
            }
        }
        for (int j = 0; j < J; ++j) {
            double w_p = info_eta_p(idx[j]);
            if (w_p > 0.0) {
                H_obs.block(p_lam, p_lam, p_p, p_p)
                    .selfadjointView<Eigen::Lower>()
                    .rankUpdate(X_p.row(idx[j]).transpose(), w_p);
            }
        }

        // ----- Var[N|y_s] rank-1 marginal correction -----
        if (var_N(s) > 0.0) {
            const int n_x_local = p_lam + p_p + n_spatial;
            Eigen::VectorXd u_s = Eigen::VectorXd::Zero(n_x_local);
            // beta_lambda coord: -X_lambda[s,]
            u_s.segment(0, p_lam) = -X_lambda.row(s).transpose();
            // beta_p coord: sum_j p_ij X_p[ij,]
            Eigen::VectorXd ssum = Eigen::VectorXd::Zero(p_p);
            for (int j = 0; j < J; ++j) {
                double e = eta_p_long(idx[j]);
                double p_ij;
                if (e > 0.0) {
                    p_ij = 1.0 / (1.0 + std::exp(-e));
                } else {
                    double ee = std::exp(e);
                    p_ij = ee / (1.0 + ee);
                }
                ssum += p_ij * X_p.row(idx[j]).transpose();
            }
            u_s.segment(p_lam, p_p) = ssum;
            // z coord: -1 at u
            u_s(z_start + u) = -1.0;

            H_obs.selfadjointView<Eigen::Lower>().rankUpdate(u_s, -var_N(s));
        }
    }
    // Symmetrise from lower triangle. The block(...)+rankUpdate fills only
    // the lower; the explicit cross-fills above already hit both sides.
    H_obs = H_obs.selfadjointView<Eigen::Lower>();
}

// Complete-data Fisher Hessian (no var_N correction). Always PSD; used as
// a Levenberg-Marquardt fallback when the observed-info matrix is not PSD.
inline void nmix_assemble_complete_fisher_spatial(
    int p_lam, int p_p, int n_spatial,
    const Eigen::Map<Eigen::MatrixXd>& X_lambda,
    const Eigen::Map<Eigen::MatrixXd>& X_p,
    const std::vector<std::vector<int>>& obs_by_site,
    const std::vector<int>& map_site_to_unit,
    const Eigen::VectorXd& info_eta_lam,
    const Eigen::VectorXd& info_eta_p,
    Eigen::MatrixXd& H_f /* in/out: zero-initialized [n_x x n_x] */
) {
    const int n_sites = static_cast<int>(obs_by_site.size());
    const int z_start = p_lam + p_p;

    for (int s = 0; s < n_sites; ++s) {
        double w_lam = info_eta_lam(s);
        if (w_lam > 0.0) {
            const int u = map_site_to_unit[s];
            H_f.block(0, 0, p_lam, p_lam)
                .selfadjointView<Eigen::Lower>()
                .rankUpdate(X_lambda.row(s).transpose(), w_lam);
            H_f(z_start + u, z_start + u) += w_lam;
            for (int k = 0; k < p_lam; ++k) {
                double v = w_lam * X_lambda(s, k);
                H_f(k, z_start + u) += v;
                H_f(z_start + u, k) += v;
            }
        }
    }
    for (int o = 0; o < static_cast<int>(X_p.rows()); ++o) {
        double w_p = info_eta_p(o);
        if (w_p > 0.0) {
            H_f.block(p_lam, p_lam, p_p, p_p)
                .selfadjointView<Eigen::Lower>()
                .rankUpdate(X_p.row(o).transpose(), w_p);
        }
    }
    H_f = H_f.selfadjointView<Eigen::Lower>();
}

// Add the CAR(rho) contribution tau * Q(rho) to the z-block of H, and tau * Q z
// to the gradient slot for z. ICAR is the rho = 1 case; proper CAR uses rho in
// the eigenvalue interval (1/lambda_min, 1/lambda_max) of D^{-1}W.
//
// Q(rho) = D - rho * W,   Q_ii = n_neighbors[i],   Q_ij = -rho for i ~ j.
//
// Sign convention matches laplace_spatial_priors:
//   H_zz += tau * Q(rho),    grad_z -= tau * Q(rho) z
// (grad here is the *score*; the prior contributes -tau Q z to the score and
//  +tau Q to the negative-Hessian.)
inline void nmix_add_car_to_spatial_block(
    int p_lam, int p_p, int n_spatial,
    double tau, double rho,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors,
    const Eigen::VectorXd& z,
    Eigen::VectorXd& grad /* in/out [n_x] */,
    Eigen::MatrixXd& H    /* in/out [n_x x n_x] */
) {
    const int z_start = p_lam + p_p;
    for (int s = 0; s < n_spatial; ++s) {
        const int idx_zs = z_start + s;
        const double q_diag = static_cast<double>(n_neighbors[s]);
        // Diagonal entry of Q at s
        H(idx_zs, idx_zs) += tau * q_diag;
        // Gradient: -tau * (Q z)[s] = -tau (n_s z_s - rho * sum_{s' ~ s} z_{s'})
        double neighbor_sum = 0.0;
        for (int kk = adj_row_ptr[s]; kk < adj_row_ptr[s + 1]; ++kk) {
            int t = adj_col_idx[kk];
            neighbor_sum += z(t);
            if (t > s) {
                H(idx_zs, z_start + t) -= tau * rho;
                H(z_start + t, idx_zs) -= tau * rho;
            }
        }
        grad(idx_zs) -= tau * (q_diag * z(s) - rho * neighbor_sum);
    }
}

// Hessian-only variant. Adds tau * Q(rho) to the z-block of H without touching
// grad. Used at the converged mode when we just need log|H|.
inline void nmix_add_car_to_H_only(
    int p_lam, int p_p, int n_spatial,
    double tau, double rho,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors,
    Eigen::MatrixXd& H /* in/out [n_x x n_x] */
) {
    const int z_start = p_lam + p_p;
    for (int s = 0; s < n_spatial; ++s) {
        const int idx_zs = z_start + s;
        H(idx_zs, idx_zs) += tau * static_cast<double>(n_neighbors[s]);
        for (int kk = adj_row_ptr[s]; kk < adj_row_ptr[s + 1]; ++kk) {
            int t = adj_col_idx[kk];
            if (t > s) {
                H(idx_zs, z_start + t) -= tau * rho;
                H(z_start + t, idx_zs) -= tau * rho;
            }
        }
    }
}

// Apply a tiny ridge to the full diagonal. The intercept of beta_lambda plus
// a constant on z form a structural null direction (a shift in both leaves
// eta_lambda unchanged); we centre z after every Newton step to pin that
// direction, but the in-step Cholesky still needs the matrix to be PD. A
// ridge of ~1e-10 * mean(diag) is invisible at the science level and keeps
// the Cholesky safe.
inline void nmix_add_diagonal_ridge(Eigen::MatrixXd& H, double rel_ridge = 1e-10) {
    const int n = static_cast<int>(H.rows());
    if (n == 0) return;
    double mean_diag = 0.0;
    for (int i = 0; i < n; ++i) mean_diag += H(i, i);
    mean_diag /= n;
    double ridge = std::max(rel_ridge * mean_diag, 1e-12);
    for (int i = 0; i < n; ++i) H(i, i) += ridge;
}

// z' Q(rho) z quadratic form. Symmetry of adjacency lets us halve the work
// by only counting edges with t > s.
inline double nmix_car_quadratic_form(
    int n_spatial, double rho,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors,
    const Eigen::VectorXd& z
) {
    double quad = 0.0;
    for (int s = 0; s < n_spatial; ++s) {
        quad += n_neighbors[s] * z(s) * z(s);
        for (int kk = adj_row_ptr[s]; kk < adj_row_ptr[s + 1]; ++kk) {
            int t = adj_col_idx[kk];
            if (t > s) quad -= 2.0 * rho * z(s) * z(t);
        }
    }
    return quad;
}

// log p(z | tau) under ICAR: -0.5 * tau * z' Q z + 0.5 * (n - 1) * log(tau)
// (rank-deficient by 1; the (2 pi)^{n/2} normalising constant is absorbed by
// the caller into the Cartesian factor that is tau-independent).
inline double nmix_icar_log_prior(
    int n_spatial, double tau,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors,
    const Eigen::VectorXd& z
) {
    double quad = nmix_car_quadratic_form(
        n_spatial, /*rho=*/1.0, adj_row_ptr, adj_col_idx, n_neighbors, z);
    return -0.5 * tau * quad + 0.5 * (n_spatial - 1) * std::log(tau);
}

// log p(z | tau, rho) under proper CAR (full rank):
//   log p = 0.5 * log_det_Q_rho + 0.5 * n * log(tau) - 0.5 * tau * z' Q(rho) z
// log_det_Q_rho = log|Q(rho)| is independent of tau and z, so the caller
// precomputes it once per rho grid point via a dense Cholesky.
inline double nmix_car_proper_log_prior(
    int n_spatial, double tau, double rho, double log_det_Q_rho,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors,
    const Eigen::VectorXd& z
) {
    double quad = nmix_car_quadratic_form(
        n_spatial, rho, adj_row_ptr, adj_col_idx, n_neighbors, z);
    return 0.5 * log_det_Q_rho
         + 0.5 * n_spatial * std::log(tau)
         - 0.5 * tau * quad;
}

// Centering: subtract mean(z) from z, enforcing the sum-to-zero constraint
// that anchors ICAR identifiability against the global intercept.
inline void nmix_center_z(
    int p_lam, int p_p, int n_spatial,
    Eigen::VectorXd& x
) {
    if (n_spatial <= 0) return;
    const int z_start = p_lam + p_p;
    double mean = 0.0;
    for (int s = 0; s < n_spatial; ++s) mean += x(z_start + s);
    mean /= n_spatial;
    for (int s = 0; s < n_spatial; ++s) x(z_start + s) -= mean;
}

}  // namespace tulpa

#endif  // TULPA_NMIX_SPATIAL_KERNEL_H
