// nmix_spatial_kernel_bym2.h
// BYM2-specific kernel helpers for the Royle (2004) N-mixture model.
//
// Riebler et al. (2016) reparametrise the BYM model so that the joint
// spatial offset is
//   phi[u] = sigma * ( sqrt(rho / s) * v[u] + sqrt(1 - rho) * w[u] )
// with v ~ ICAR (unscaled, sum-to-zero), w ~ N(0, I) iid, and s the
// geometric mean of the non-zero eigenvalues of Q (the ICAR precision).
// Under this parametrisation sigma is the joint marginal standard
// deviation of phi and rho is the spatial fraction of variance.
//
// State vector layout (n_x = p_lam + p_p + 2 * n_spatial):
//
//   x[0           : p_lam]                            = beta_lambda
//   x[p_lam       : p_lam + p_p]                      = beta_p
//   x[p_lam + p_p : p_lam + p_p + n_spatial]          = v  (ICAR)
//   x[p_lam+p_p+n : p_lam + p_p + 2 * n_spatial]      = w  (iid)
//
// Linear predictor at site s mapped to unit u(s):
//
//   eta_lambda[s] = X_lambda[s,] . beta_lambda
//                 + a * v[u(s)] + b * w[u(s)]
//
// with a = sigma * sqrt(rho / scale_factor) and b = sigma * sqrt(1 - rho).
//
// Because (a, b) only multiply v and w in the linear predictor (not in the
// priors), the priors on v and w are *constant* in (sigma, rho) and drop
// out of the integration up to a tau-independent additive constant; the
// grid integration is over (sigma, rho) only.
//
// The (1, b/a)-style identifiability ridge between (lambda intercept) and
// (constant v, constant w) is pinned by sum-to-zero centering of v after
// each Newton step. The iid component w is identified by its N(0, I)
// prior and is not centred.

#ifndef TULPA_NMIX_SPATIAL_KERNEL_BYM2_H
#define TULPA_NMIX_SPATIAL_KERNEL_BYM2_H

#include "nmix_kernel.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace tulpa {

// eta_lambda[s] = X_lambda[s,] . beta_lambda + a * v[u(s)] + b * w[u(s)]
inline void compute_eta_lambda_bym2(
    const Eigen::Map<Eigen::MatrixXd>& X_lambda,
    const Eigen::VectorXd& beta_lambda,
    const Eigen::VectorXd& v,
    const Eigen::VectorXd& w,
    double a, double b,
    const std::vector<int>& map_site_to_unit,
    Eigen::VectorXd& eta_lambda /* out, length n_sites */
) {
    eta_lambda.noalias() = X_lambda * beta_lambda;
    const int n_sites = static_cast<int>(map_site_to_unit.size());
    for (int s = 0; s < n_sites; ++s) {
        const int u = map_site_to_unit[s];
        eta_lambda(s) += a * v(u) + b * w(u);
    }
}

// Assemble marginal observed-info Hessian over
//   theta = (beta_lambda, beta_p, v, w)
// for BYM2. Per-site complete-data Fisher block:
//   ds_beta_lambda = w_lam * x_lam[s,] x_lam[s,]^T
//   ds_v[u]        = w_lam * a^2          (diagonal)
//   ds_w[u]        = w_lam * b^2          (diagonal)
//   ds_v[u]w[u]    = w_lam * a * b        (cross)
//   ds_beta_lambda,v[u] = w_lam * a * x_lam[s,]
//   ds_beta_lambda,w[u] = w_lam * b * x_lam[s,]
// Plus the per-site Var[N|y_s] rank-1 correction with u_s carrying
//   -x_lam[s,] in the beta_lambda slot,
//   sum_j p_ij x_p[ij,] in the beta_p slot,
//   -a at v[u(s)], -b at w[u(s)].
inline void nmix_assemble_obs_info_bym2(
    int p_lam, int p_p, int n_spatial,
    double a, double b,
    const Eigen::Map<Eigen::MatrixXd>& X_lambda,
    const Eigen::Map<Eigen::MatrixXd>& X_p,
    const Eigen::VectorXd& eta_p_long,
    const std::vector<std::vector<int>>& obs_by_site,
    const std::vector<int>& map_site_to_unit,
    const Eigen::VectorXd& info_eta_lam,
    const Eigen::VectorXd& info_eta_p,
    const Eigen::VectorXd& var_N,
    const Eigen::VectorXd& score_wt_lambda,   // N-coeff of s_lambda (1 for Poisson)
    Eigen::MatrixXd& H_obs /* in/out: zero-initialized [n_x x n_x] */
) {
    const int n_sites = static_cast<int>(obs_by_site.size());
    const int v_start = p_lam + p_p;
    const int w_start = v_start + n_spatial;

    for (int s = 0; s < n_sites; ++s) {
        const auto& idx = obs_by_site[s];
        const int J = static_cast<int>(idx.size());
        if (J == 0) continue;
        const int u = map_site_to_unit[s];

        // ----- Complete-data Fisher D_s -----
        double w_lam = info_eta_lam(s);
        if (w_lam > 0.0) {
            // beta_lambda x beta_lambda
            H_obs.block(0, 0, p_lam, p_lam)
                .selfadjointView<Eigen::Lower>()
                .rankUpdate(X_lambda.row(s).transpose(), w_lam);
            // v[u] x v[u]
            H_obs(v_start + u, v_start + u) += w_lam * a * a;
            // w[u] x w[u]
            H_obs(w_start + u, w_start + u) += w_lam * b * b;
            // v[u] x w[u]
            const double w_vw = w_lam * a * b;
            H_obs(v_start + u, w_start + u) += w_vw;
            H_obs(w_start + u, v_start + u) += w_vw;
            // beta_lambda x v[u] and beta_lambda x w[u]
            for (int k = 0; k < p_lam; ++k) {
                const double cv = w_lam * a * X_lambda(s, k);
                H_obs(k, v_start + u) += cv;
                H_obs(v_start + u, k) += cv;
                const double cw = w_lam * b * X_lambda(s, k);
                H_obs(k, w_start + u) += cw;
                H_obs(w_start + u, k) += cw;
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
        // score_wt_lambda(s) is the N-coefficient of the eta_lambda score
        // (1 for Poisson, 1-q for NB); it scales beta_lambda (via X_lambda) and
        // the v / w coords (via a, b) that enter eta_lambda.
        if (var_N(s) > 0.0) {
            const double swl = score_wt_lambda(s);
            const int n_x_local = p_lam + p_p + 2 * n_spatial;
            Eigen::VectorXd u_s = Eigen::VectorXd::Zero(n_x_local);
            u_s.segment(0, p_lam) = -swl * X_lambda.row(s).transpose();
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
            u_s(v_start + u) = -swl * a;
            u_s(w_start + u) = -swl * b;

            H_obs.selfadjointView<Eigen::Lower>().rankUpdate(u_s, -var_N(s));
        }
    }
    H_obs = H_obs.selfadjointView<Eigen::Lower>();
}

// Complete-data Fisher Hessian for BYM2 (no var_N correction); always PSD.
inline void nmix_assemble_complete_fisher_bym2(
    int p_lam, int p_p, int n_spatial,
    double a, double b,
    const Eigen::Map<Eigen::MatrixXd>& X_lambda,
    const Eigen::Map<Eigen::MatrixXd>& X_p,
    const std::vector<std::vector<int>>& obs_by_site,
    const std::vector<int>& map_site_to_unit,
    const Eigen::VectorXd& info_eta_lam,
    const Eigen::VectorXd& info_eta_p,
    Eigen::MatrixXd& H_f /* in/out: zero-initialized [n_x x n_x] */
) {
    const int n_sites = static_cast<int>(obs_by_site.size());
    const int v_start = p_lam + p_p;
    const int w_start = v_start + n_spatial;

    for (int s = 0; s < n_sites; ++s) {
        double w_lam = info_eta_lam(s);
        if (w_lam > 0.0) {
            const int u = map_site_to_unit[s];
            H_f.block(0, 0, p_lam, p_lam)
                .selfadjointView<Eigen::Lower>()
                .rankUpdate(X_lambda.row(s).transpose(), w_lam);
            H_f(v_start + u, v_start + u) += w_lam * a * a;
            H_f(w_start + u, w_start + u) += w_lam * b * b;
            const double w_vw = w_lam * a * b;
            H_f(v_start + u, w_start + u) += w_vw;
            H_f(w_start + u, v_start + u) += w_vw;
            for (int k = 0; k < p_lam; ++k) {
                const double cv = w_lam * a * X_lambda(s, k);
                H_f(k, v_start + u) += cv;
                H_f(v_start + u, k) += cv;
                const double cw = w_lam * b * X_lambda(s, k);
                H_f(k, w_start + u) += cw;
                H_f(w_start + u, k) += cw;
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

// Add BYM2 prior contribution to gradient and Hessian:
//   log p(v) = -0.5 * v' Q v       (ICAR; rank-deficient by 1, constant in sigma/rho)
//   log p(w) = -0.5 * w' w         (iid N(0, I))
// Score: grad_v -= Q v,   grad_w -= w
// Hessian: H_{v,v} += Q,   H_{w,w} += I
inline void nmix_add_bym2_prior_to_grad_and_H(
    int p_lam, int p_p, int n_spatial,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors,
    const Eigen::VectorXd& v,
    const Eigen::VectorXd& w,
    Eigen::VectorXd& grad /* in/out [n_x] */,
    Eigen::MatrixXd& H    /* in/out [n_x x n_x] */
) {
    const int v_start = p_lam + p_p;
    const int w_start = v_start + n_spatial;

    // v block: ICAR precision Q (rho = 1.0, tau = 1.0).
    for (int s = 0; s < n_spatial; ++s) {
        const int idx_vs = v_start + s;
        const double q_diag = static_cast<double>(n_neighbors[s]);
        H(idx_vs, idx_vs) += q_diag;
        double neighbor_sum = 0.0;
        for (int kk = adj_row_ptr[s]; kk < adj_row_ptr[s + 1]; ++kk) {
            int t = adj_col_idx[kk];
            neighbor_sum += v(t);
            if (t > s) {
                H(idx_vs, v_start + t) -= 1.0;
                H(v_start + t, idx_vs) -= 1.0;
            }
        }
        grad(idx_vs) -= (q_diag * v(s) - neighbor_sum);
    }

    // w block: identity (iid N(0, I)).
    for (int s = 0; s < n_spatial; ++s) {
        const int idx_ws = w_start + s;
        H(idx_ws, idx_ws) += 1.0;
        grad(idx_ws) -= w(s);
    }
}

// Hessian-only variant for the final log|H| assembly.
inline void nmix_add_bym2_prior_to_H_only(
    int p_lam, int p_p, int n_spatial,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors,
    Eigen::MatrixXd& H /* in/out [n_x x n_x] */
) {
    const int v_start = p_lam + p_p;
    const int w_start = v_start + n_spatial;

    for (int s = 0; s < n_spatial; ++s) {
        const int idx_vs = v_start + s;
        H(idx_vs, idx_vs) += static_cast<double>(n_neighbors[s]);
        for (int kk = adj_row_ptr[s]; kk < adj_row_ptr[s + 1]; ++kk) {
            int t = adj_col_idx[kk];
            if (t > s) {
                H(idx_vs, v_start + t) -= 1.0;
                H(v_start + t, idx_vs) -= 1.0;
            }
        }
    }
    for (int s = 0; s < n_spatial; ++s) {
        H(w_start + s, w_start + s) += 1.0;
    }
}

// log p(v) + log p(w) for BYM2, dropping (sigma, rho)-independent additive
// constants. Used inside the line-search objective only -- the absolute
// value cancels in the Laplace marginal because it does not depend on
// (sigma, rho).
inline double nmix_bym2_log_prior(
    int n_spatial,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors,
    const Eigen::VectorXd& v,
    const Eigen::VectorXd& w
) {
    double quad_v = 0.0;
    for (int s = 0; s < n_spatial; ++s) {
        quad_v += n_neighbors[s] * v(s) * v(s);
        for (int kk = adj_row_ptr[s]; kk < adj_row_ptr[s + 1]; ++kk) {
            int t = adj_col_idx[kk];
            if (t > s) quad_v -= 2.0 * v(s) * v(t);
        }
    }
    double quad_w = w.squaredNorm();
    return -0.5 * quad_v - 0.5 * quad_w;
}

// Sum-to-zero centering of v (the ICAR component). The iid component w is
// identified by its prior and is not centred.
inline void nmix_center_v_bym2(
    int p_lam, int p_p, int n_spatial,
    Eigen::VectorXd& x
) {
    if (n_spatial <= 0) return;
    const int v_start = p_lam + p_p;
    double mean = 0.0;
    for (int s = 0; s < n_spatial; ++s) mean += x(v_start + s);
    mean /= n_spatial;
    for (int s = 0; s < n_spatial; ++s) x(v_start + s) -= mean;
}

}  // namespace tulpa

#endif  // TULPA_NMIX_SPATIAL_KERNEL_BYM2_H
