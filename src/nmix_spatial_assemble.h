// nmix_spatial_assemble.h
//
// Shared, spatial-structure-agnostic assembly of the N-mixture marginal
// observed-information and complete-data Fisher Hessians.
//
// The ICAR / proper-CAR and BYM2 variants differ ONLY in how the latent
// abundance predictor eta_lambda[s] loads onto the spatial coordinates of
// theta = (beta_lambda, beta_p, <spatial>):
//
//   ICAR / CAR : eta_lambda[s] += 1 * z[u(s)]                -> one  (col, coef)
//   BYM2       : eta_lambda[s] += a * v[u(s)] + b * w[u(s)]  -> two  (col, coef)
//
// Everything else -- the beta_lambda / beta_p complete-data Fisher blocks, the
// detection scatter ssum = sum_j p_ij x_p[ij,], the Var[N|y_s] rank-1 marginal
// correction (Louis 1982), and the lower-triangle symmetrise -- is identical
// across the two structures. We capture the spatial loading as a small functor
// (int s) -> SpatialLoading and assemble once, so adding a spatial structure is
// a new loading functor rather than a copied 90-line assembler (CLAUDE.md #5).

#ifndef TULPA_NMIX_SPATIAL_ASSEMBLE_H
#define TULPA_NMIX_SPATIAL_ASSEMBLE_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace tulpa {

// Loading of eta_lambda[s] onto the spatial coordinates of theta:
//   eta_lambda[s] += sum_{k < n} coef[k] * theta[col[k]].
// At most two entries (BYM2's structured + unstructured fields).
struct SpatialLoading {
    int n = 0;
    int col[2] = {0, 0};
    double coef[2] = {0.0, 0.0};
};

// Marginal observed-information Hessian over theta = (beta_lambda, beta_p,
// <spatial>). `loading(s)` returns the spatial loading of site s; the spatial
// coordinates it names must lie in [p_lam + p_p, n_x).
template <typename LoadingFn>
inline void nmix_assemble_obs_info(
    int p_lam, int p_p, int n_x,
    const Eigen::Map<Eigen::MatrixXd>& X_lambda,
    const Eigen::Map<Eigen::MatrixXd>& X_p,
    const Eigen::VectorXd& eta_p_long,
    const std::vector<std::vector<int>>& obs_by_site,
    const Eigen::VectorXd& info_eta_lam,
    const Eigen::VectorXd& info_eta_p,
    const Eigen::VectorXd& var_N,
    const Eigen::VectorXd& score_wt_lambda,   // N-coeff of s_lambda (1 Poisson, 1-q NB)
    LoadingFn loading,
    Eigen::MatrixXd& H_obs /* in/out: zero-initialized [n_x x n_x] */
) {
    const int n_sites = static_cast<int>(obs_by_site.size());

    for (int s = 0; s < n_sites; ++s) {
        const auto& idx = obs_by_site[s];
        const int J = static_cast<int>(idx.size());
        if (J == 0) continue;
        const SpatialLoading L = loading(s);

        // ----- Complete-data Fisher D_s -----
        const double w_lam = info_eta_lam(s);
        if (w_lam > 0.0) {
            // beta_lambda x beta_lambda block
            H_obs.block(0, 0, p_lam, p_lam)
                .selfadjointView<Eigen::Lower>()
                .rankUpdate(X_lambda.row(s).transpose(), w_lam);
            for (int i = 0; i < L.n; ++i) {
                const int ci = L.col[i];
                const double ai = L.coef[i];
                // spatial[i] diagonal
                H_obs(ci, ci) += w_lam * ai * ai;
                // spatial[i] x spatial[j] cross (BYM2 v[u] x w[u])
                for (int j = i + 1; j < L.n; ++j) {
                    const double vv = w_lam * ai * L.coef[j];
                    H_obs(ci, L.col[j]) += vv;
                    H_obs(L.col[j], ci) += vv;
                }
                // beta_lambda x spatial[i] cross
                for (int k = 0; k < p_lam; ++k) {
                    const double c = w_lam * ai * X_lambda(s, k);
                    H_obs(k, ci) += c;
                    H_obs(ci, k) += c;
                }
            }
        }
        for (int j = 0; j < J; ++j) {
            const double w_p = info_eta_p(idx[j]);
            if (w_p > 0.0) {
                H_obs.block(p_lam, p_lam, p_p, p_p)
                    .selfadjointView<Eigen::Lower>()
                    .rankUpdate(X_p.row(idx[j]).transpose(), w_p);
            }
        }

        // ----- Var[N|y_s] rank-1 marginal correction -----
        // The eta_lambda score has N-coefficient score_wt_lambda(s); it scales
        // every coordinate eta_lambda depends on -- beta_lambda (via X_lambda)
        // and the spatial coords (via their loading coefficients).
        if (var_N(s) > 0.0) {
            const double swl = score_wt_lambda(s);
            Eigen::VectorXd u_s = Eigen::VectorXd::Zero(n_x);
            u_s.segment(0, p_lam) = -swl * X_lambda.row(s).transpose();
            Eigen::VectorXd ssum = Eigen::VectorXd::Zero(p_p);
            for (int j = 0; j < J; ++j) {
                const double e = eta_p_long(idx[j]);
                double p_ij;
                if (e > 0.0) {
                    p_ij = 1.0 / (1.0 + std::exp(-e));
                } else {
                    const double ee = std::exp(e);
                    p_ij = ee / (1.0 + ee);
                }
                ssum += p_ij * X_p.row(idx[j]).transpose();
            }
            u_s.segment(p_lam, p_p) = ssum;
            for (int i = 0; i < L.n; ++i) u_s(L.col[i]) = -swl * L.coef[i];

            H_obs.selfadjointView<Eigen::Lower>().rankUpdate(u_s, -var_N(s));
        }
    }
    // Symmetrise from the lower triangle. block(...)+rankUpdate fills only the
    // lower; the explicit spatial cross-fills above already hit both sides.
    H_obs = H_obs.selfadjointView<Eigen::Lower>();
}

// Complete-data Fisher Hessian (no var_N correction). Always PSD; used as the
// Levenberg-Marquardt fallback when the observed-info matrix is not PSD.
template <typename LoadingFn>
inline void nmix_assemble_complete_fisher(
    int p_lam, int p_p, int n_x,
    const Eigen::Map<Eigen::MatrixXd>& X_lambda,
    const Eigen::Map<Eigen::MatrixXd>& X_p,
    const std::vector<std::vector<int>>& obs_by_site,
    const Eigen::VectorXd& info_eta_lam,
    const Eigen::VectorXd& info_eta_p,
    LoadingFn loading,
    Eigen::MatrixXd& H_f /* in/out: zero-initialized [n_x x n_x] */
) {
    const int n_sites = static_cast<int>(obs_by_site.size());

    for (int s = 0; s < n_sites; ++s) {
        const double w_lam = info_eta_lam(s);
        if (w_lam <= 0.0) continue;
        const SpatialLoading L = loading(s);
        H_f.block(0, 0, p_lam, p_lam)
            .selfadjointView<Eigen::Lower>()
            .rankUpdate(X_lambda.row(s).transpose(), w_lam);
        for (int i = 0; i < L.n; ++i) {
            const int ci = L.col[i];
            const double ai = L.coef[i];
            H_f(ci, ci) += w_lam * ai * ai;
            for (int j = i + 1; j < L.n; ++j) {
                const double vv = w_lam * ai * L.coef[j];
                H_f(ci, L.col[j]) += vv;
                H_f(L.col[j], ci) += vv;
            }
            for (int k = 0; k < p_lam; ++k) {
                const double c = w_lam * ai * X_lambda(s, k);
                H_f(k, ci) += c;
                H_f(ci, k) += c;
            }
        }
    }
    for (int o = 0; o < static_cast<int>(X_p.rows()); ++o) {
        const double w_p = info_eta_p(o);
        if (w_p > 0.0) {
            H_f.block(p_lam, p_lam, p_p, p_p)
                .selfadjointView<Eigen::Lower>()
                .rankUpdate(X_p.row(o).transpose(), w_p);
        }
    }
    H_f = H_f.selfadjointView<Eigen::Lower>();
}

}  // namespace tulpa

#endif  // TULPA_NMIX_SPATIAL_ASSEMBLE_H
