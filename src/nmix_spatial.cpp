// nmix_spatial.cpp
// Nested Laplace for the spatial Royle (2004) N-mixture model with an ICAR
// (intrinsic conditional autoregressive) prior on the abundance-arm spatial
// offset.
//
// Outer grid: 1D over tau_spatial (precision of the ICAR prior).
// Inner Newton at fixed tau_k finds the joint mode of
//   x = (beta_lambda [p_lam], beta_p [p_p], z [n_spatial])
// using complete-data Fisher curvature (PSD, EM-rate convergence) with a
// fallback when the marginal observed-information matrix is non-PSD. At the
// converged mode we form the marginal observed-information Hessian (with the
// Var[N|y] rank-1 correction) and the ICAR prior contribution to compute the
// Laplace log-marginal at that grid point.
//
// Per-grid log marginal (up to a tau-independent additive constant):
//   log p(y | tau_k) ~= log L(mode_k)
//                       + log p(z_mode_k | tau_k)
//                       - 0.5 log|H(mode_k; tau_k)|
// where the prior contribution log p(z | tau) carries the (n_spatial - 1)/2 *
// log(tau) factor reflecting the ICAR rank deficiency.
//
// The Cartesian (2π)^{n_x/2} constant is omitted because it is the same at
// every grid point and only inflates absolute log_marginal values; it has no
// effect on the integration weights or hyperparameter posterior moments.
//
// Reuses the per-site marginal kernel from nmix_kernel.h and the spatial
// scatter/Hessian helpers from nmix_spatial_kernel.h.

#include "nmix_kernel.h"
#include "nmix_spatial_kernel.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// [[Rcpp::depends(RcppEigen)]]

namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using tulpa::nmix_kernel_sweep_spatial;
using tulpa::nmix_kernel_log_lik_only_spatial;
using tulpa::compute_eta_lambda_spatial;
using tulpa::nmix_assemble_obs_info_spatial;
using tulpa::nmix_assemble_complete_fisher_spatial;
using tulpa::nmix_add_car_to_spatial_block;
using tulpa::nmix_add_car_to_H_only;
using tulpa::nmix_add_diagonal_ridge;
using tulpa::nmix_icar_log_prior;
using tulpa::nmix_car_proper_log_prior;
using tulpa::nmix_center_z;

enum class CarPriorKind { ICAR, CAR_PROPER };

inline double nmix_car_log_prior_dispatch(
    CarPriorKind kind,
    int n_spatial, double tau, double rho, double log_det_Q_rho,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors,
    const VectorXd& z
) {
    if (kind == CarPriorKind::ICAR) {
        return nmix_icar_log_prior(
            n_spatial, tau, adj_row_ptr, adj_col_idx, n_neighbors, z);
    }
    return nmix_car_proper_log_prior(
        n_spatial, tau, rho, log_det_Q_rho,
        adj_row_ptr, adj_col_idx, n_neighbors, z);
}

// Per-grid-point inner solve result.
struct SpatialInnerResult {
    VectorXd beta_lambda;
    VectorXd beta_p;
    VectorXd z;
    double log_lik;
    double log_marginal;
    double grad_norm;
    int n_iter;
    bool converged;
    double boundary_max;
};

// Shared inner Newton for ICAR (rho = 1.0, sum-to-zero centering, rank-deficient
// log-prior) and CAR_proper (rho < 1.0, no centering, full-rank log-prior with
// precomputed log_det_Q_rho).
SpatialInnerResult inner_newton_spatial_car(
    CarPriorKind kind,
    int p_lam, int p_p, int n_sites, int n_spatial, int n_obs,
    const Map<MatrixXd>& Xl,
    const Map<MatrixXd>& Xp,
    const Rcpp::IntegerVector& y_R,
    const std::vector<std::vector<int>>& obs_by_site,
    const std::vector<int>& map_site_to_unit,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors,
    double tau, double rho, double log_det_Q_rho,
    int K_max,
    int max_iter,
    double tol,
    const VectorXd& beta_lam_init,
    const VectorXd& beta_p_init,
    const VectorXd& z_init,
    bool verbose
) {
    SpatialInnerResult res;
    res.beta_lambda = beta_lam_init;
    res.beta_p      = beta_p_init;
    res.z           = z_init;
    res.converged   = false;
    res.n_iter      = 0;

    const int n_x = p_lam + p_p + n_spatial;

    VectorXd grad_eta_lam(n_sites), info_eta_lam(n_sites);
    VectorXd mean_N(n_sites), var_N(n_sites), boundary_weight(n_sites);
    VectorXd grad_eta_p(n_obs), info_eta_p(n_obs);
    VectorXd eta_lam(n_sites);
    VectorXd eta_p_long(n_obs);

    double log_lik = R_NegInf;
    double grad_norm = R_PosInf;

    for (int iter = 0; iter < max_iter; ++iter) {
        compute_eta_lambda_spatial(Xl, res.beta_lambda, res.z,
                                   map_site_to_unit, eta_lam);
        eta_p_long.noalias() = Xp * res.beta_p;

        log_lik = nmix_kernel_sweep_spatial(
            obs_by_site, y_R, eta_lam, eta_p_long, K_max,
            grad_eta_lam, info_eta_lam, grad_eta_p, info_eta_p,
            mean_N, var_N, boundary_weight
        );

        VectorXd grad = VectorXd::Zero(n_x);
        grad.segment(0, p_lam)         = Xl.transpose() * grad_eta_lam;
        grad.segment(p_lam, p_p)       = Xp.transpose() * grad_eta_p;
        // grad_z[u] = sum_{s: map[s]=u} grad_eta_lam[s]
        for (int s = 0; s < n_sites; ++s) {
            grad(p_lam + p_p + map_site_to_unit[s]) += grad_eta_lam(s);
        }

        // ICAR prior on z: subtract tau * Q z from gradient (assembled inside
        // nmix_add_icar_to_spatial_block below alongside the Hessian add).

        MatrixXd H = MatrixXd::Zero(n_x, n_x);
        nmix_assemble_obs_info_spatial(
            p_lam, p_p, n_spatial,
            Xl, Xp, eta_p_long, obs_by_site, map_site_to_unit,
            info_eta_lam, info_eta_p, var_N, H
        );
        nmix_add_car_to_spatial_block(
            p_lam, p_p, n_spatial, tau, rho,
            adj_row_ptr, adj_col_idx, n_neighbors,
            res.z, grad, H
        );

        grad_norm = grad.norm();
        if (verbose) {
            Rcpp::Rcout << "  iter " << iter
                        << "  log_lik " << log_lik
                        << "  grad_norm " << grad_norm
                        << "  boundary_max " << boundary_weight.maxCoeff() << "\n";
        }
        if (grad_norm < tol) {
            res.converged = true;
            res.n_iter = iter + 1;
            break;
        }

        // Cholesky solve. If H_obs (with var_N correction + ICAR) is not PSD,
        // fall back to complete-data Fisher + ICAR (always PSD when tau > 0).
        // A tiny ridge accommodates the structural (intercept, constant z)
        // null direction that centering pins after each step.
        nmix_add_diagonal_ridge(H);
        VectorXd delta;
        Eigen::LLT<MatrixXd> chol(H);
        if (chol.info() == Eigen::Success) {
            delta = chol.solve(grad);
        } else {
            MatrixXd H_f = MatrixXd::Zero(n_x, n_x);
            nmix_assemble_complete_fisher_spatial(
                p_lam, p_p, n_spatial,
                Xl, Xp, obs_by_site, map_site_to_unit,
                info_eta_lam, info_eta_p, H_f
            );
            nmix_add_car_to_H_only(
                p_lam, p_p, n_spatial, tau, rho,
                adj_row_ptr, adj_col_idx, n_neighbors, H_f
            );
            nmix_add_diagonal_ridge(H_f);
            Eigen::LLT<MatrixXd> chol_f(H_f);
            if (chol_f.info() != Eigen::Success) {
                Rcpp::warning("Cholesky failure (complete-data fallback) at iter %d, tau %.4f, rho %.4f.",
                              iter, tau, rho);
                break;
            }
            delta = chol_f.solve(grad);
            if (verbose) Rcpp::Rcout << "    (Fisher fallback)\n";
        }

        VectorXd delta_lam = delta.segment(0, p_lam);
        VectorXd delta_p   = delta.segment(p_lam, p_p);
        VectorXd delta_z   = delta.segment(p_lam + p_p, n_spatial);

        // Step halving on the joint log-posterior objective.
        double step = 1.0;
        bool stepped = false;
        VectorXd beta_lam_try, beta_p_try, z_try;
        VectorXd eta_lam_try(n_sites), eta_p_try(n_obs);
        for (int h = 0; h < 12; ++h) {
            beta_lam_try = res.beta_lambda + step * delta_lam;
            beta_p_try   = res.beta_p      + step * delta_p;
            z_try        = res.z           + step * delta_z;

            compute_eta_lambda_spatial(Xl, beta_lam_try, z_try,
                                       map_site_to_unit, eta_lam_try);
            eta_p_try.noalias() = Xp * beta_p_try;
            double ll_try = nmix_kernel_log_lik_only_spatial(
                obs_by_site, y_R, eta_lam_try, eta_p_try, K_max
            );
            double lp_try = nmix_car_log_prior_dispatch(
                kind, n_spatial, tau, rho, log_det_Q_rho,
                adj_row_ptr, adj_col_idx, n_neighbors, z_try
            );
            double obj_try = ll_try + lp_try;
            double obj_cur = log_lik + nmix_car_log_prior_dispatch(
                kind, n_spatial, tau, rho, log_det_Q_rho,
                adj_row_ptr, adj_col_idx, n_neighbors, res.z
            );
            if (R_finite(obj_try) && obj_try >= obj_cur - 1e-10) {
                res.beta_lambda = beta_lam_try;
                res.beta_p      = beta_p_try;
                res.z           = z_try;
                // Sum-to-zero centering is needed only for the rank-deficient
                // ICAR prior; CAR_proper is full-rank and the (intercept, z)
                // direction is identified by Q(rho) itself.
                if (kind == CarPriorKind::ICAR) {
                    VectorXd x_holder(n_x);
                    x_holder.segment(0, p_lam) = res.beta_lambda;
                    x_holder.segment(p_lam, p_p) = res.beta_p;
                    x_holder.segment(p_lam + p_p, n_spatial) = res.z;
                    nmix_center_z(p_lam, p_p, n_spatial, x_holder);
                    res.z = x_holder.segment(p_lam + p_p, n_spatial);
                }
                stepped = true;
                break;
            }
            step *= 0.5;
        }
        if (!stepped) {
            if (verbose) Rcpp::Rcout << "    (step halving exhausted)\n";
            break;
        }
        res.n_iter = iter + 1;
    }

    // ----- log marginal at the converged mode -----
    compute_eta_lambda_spatial(Xl, res.beta_lambda, res.z,
                               map_site_to_unit, eta_lam);
    eta_p_long.noalias() = Xp * res.beta_p;
    double log_lik_final = nmix_kernel_sweep_spatial(
        obs_by_site, y_R, eta_lam, eta_p_long, K_max,
        grad_eta_lam, info_eta_lam, grad_eta_p, info_eta_p,
        mean_N, var_N, boundary_weight
    );
    double log_prior_z_final = nmix_car_log_prior_dispatch(
        kind, n_spatial, tau, rho, log_det_Q_rho,
        adj_row_ptr, adj_col_idx, n_neighbors, res.z
    );

    MatrixXd H_final = MatrixXd::Zero(n_x, n_x);
    nmix_assemble_obs_info_spatial(
        p_lam, p_p, n_spatial,
        Xl, Xp, eta_p_long, obs_by_site, map_site_to_unit,
        info_eta_lam, info_eta_p, var_N, H_final
    );
    // For the log|H| term, use observed info + CAR(rho). If non-PSD, fall
    // back to complete-data Fisher + CAR(rho) (will overstate curvature
    // slightly but keeps the Laplace finite).
    nmix_add_car_to_H_only(
        p_lam, p_p, n_spatial, tau, rho,
        adj_row_ptr, adj_col_idx, n_neighbors, H_final
    );
    nmix_add_diagonal_ridge(H_final);
    Eigen::LLT<MatrixXd> chol(H_final);
    double log_det_H;
    if (chol.info() == Eigen::Success) {
        log_det_H = 2.0 * chol.matrixL().toDenseMatrix().diagonal()
                              .array().log().sum();
    } else {
        MatrixXd H_f = MatrixXd::Zero(n_x, n_x);
        nmix_assemble_complete_fisher_spatial(
            p_lam, p_p, n_spatial,
            Xl, Xp, obs_by_site, map_site_to_unit,
            info_eta_lam, info_eta_p, H_f
        );
        nmix_add_car_to_H_only(
            p_lam, p_p, n_spatial, tau, rho,
            adj_row_ptr, adj_col_idx, n_neighbors, H_f
        );
        nmix_add_diagonal_ridge(H_f);
        Eigen::LLT<MatrixXd> chol_f(H_f);
        if (chol_f.info() != Eigen::Success) {
            res.log_marginal = R_NegInf;
            res.log_lik = log_lik_final;
            res.grad_norm = grad_norm;
            res.boundary_max = boundary_weight.maxCoeff();
            return res;
        }
        log_det_H = 2.0 * chol_f.matrixL().toDenseMatrix().diagonal()
                                .array().log().sum();
    }

    res.log_lik = log_lik_final;
    res.log_marginal = log_lik_final + log_prior_z_final - 0.5 * log_det_H;
    res.grad_norm = grad_norm;
    res.boundary_max = boundary_weight.maxCoeff();
    return res;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_nmix_icar(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector site_idx,
    Rcpp::IntegerVector map_site_to_unit_R,   // 1-based site -> unit map
    Rcpp::NumericMatrix X_lambda_R,
    Rcpp::NumericMatrix X_p_R,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    int n_spatial,
    Rcpp::NumericVector tau_grid,
    Rcpp::NumericVector beta_lambda_init,
    Rcpp::NumericVector beta_p_init,
    Rcpp::Nullable<Rcpp::NumericVector> z_init = R_NilValue,
    int K_max = -1,
    int max_iter = 100,
    double tol = 1e-6,
    bool verbose = false
) {
    const int n_sites = X_lambda_R.nrow();
    const int p_lam   = X_lambda_R.ncol();
    const int n_obs   = X_p_R.nrow();
    const int p_p     = X_p_R.ncol();
    if ((int)y.size() != n_obs) Rcpp::stop("length(y) must equal nrow(X_p).");
    if ((int)site_idx.size() != n_obs) Rcpp::stop("length(site_idx) must equal nrow(X_p).");
    if ((int)map_site_to_unit_R.size() != n_sites) {
        Rcpp::stop("length(map_site_to_unit) must equal nrow(X_lambda).");
    }
    if ((int)beta_lambda_init.size() != p_lam) Rcpp::stop("beta_lambda_init length mismatch.");
    if ((int)beta_p_init.size() != p_p) Rcpp::stop("beta_p_init length mismatch.");
    if (K_max < 0) {
        int ymax = 0;
        for (int o = 0; o < n_obs; ++o) if (y[o] > ymax) ymax = y[o];
        K_max = ymax + 100;
    }

    std::vector<std::vector<int>> obs_by_site(n_sites);
    for (int o = 0; o < n_obs; ++o) {
        int s = site_idx[o] - 1;
        if (s < 0 || s >= n_sites) Rcpp::stop("site_idx out of range at obs %d.", o + 1);
        obs_by_site[s].push_back(o);
    }
    std::vector<int> map_site_to_unit(n_sites);
    for (int s = 0; s < n_sites; ++s) {
        int u = map_site_to_unit_R[s] - 1;
        if (u < 0 || u >= n_spatial) {
            Rcpp::stop("map_site_to_unit[%d] = %d out of range [1, %d].",
                       s + 1, map_site_to_unit_R[s], n_spatial);
        }
        map_site_to_unit[s] = u;
    }

    Map<MatrixXd> Xl(REAL(X_lambda_R), n_sites, p_lam);
    Map<MatrixXd> Xp(REAL(X_p_R), n_obs, p_p);

    VectorXd beta_lam_default = Map<VectorXd>(REAL(beta_lambda_init), p_lam);
    VectorXd beta_p_default   = Map<VectorXd>(REAL(beta_p_init), p_p);
    VectorXd z_default(n_spatial);
    if (z_init.isNotNull()) {
        Rcpp::NumericVector zi(z_init);
        if ((int)zi.size() != n_spatial) Rcpp::stop("length(z_init) must equal n_spatial.");
        for (int s = 0; s < n_spatial; ++s) z_default(s) = zi[s];
    } else {
        z_default.setZero();
    }
    VectorXd beta_lam = beta_lam_default;
    VectorXd beta_p   = beta_p_default;
    VectorXd z        = z_default;

    const int n_grid = tau_grid.size();
    const int n_x    = p_lam + p_p + n_spatial;

    Rcpp::NumericVector log_marginals(n_grid);
    Rcpp::IntegerVector n_iters(n_grid);
    Rcpp::LogicalVector convergeds(n_grid);
    Rcpp::NumericVector grad_norms(n_grid);
    Rcpp::NumericVector log_liks(n_grid);
    Rcpp::NumericVector boundary_maxes(n_grid);
    Rcpp::NumericMatrix modes(n_grid, n_x);

    for (int k = 0; k < n_grid; ++k) {
        SpatialInnerResult ir = inner_newton_spatial_car(
            CarPriorKind::ICAR,
            p_lam, p_p, n_sites, n_spatial, n_obs,
            Xl, Xp, y, obs_by_site, map_site_to_unit,
            adj_row_ptr, adj_col_idx, n_neighbors,
            tau_grid[k], /*rho=*/1.0, /*log_det_Q_rho=*/0.0,
            K_max, max_iter, tol,
            beta_lam, beta_p, z, verbose
        );
        log_marginals[k]  = ir.log_marginal;
        n_iters[k]        = ir.n_iter;
        convergeds[k]     = ir.converged;
        grad_norms[k]     = ir.grad_norm;
        log_liks[k]       = ir.log_lik;
        boundary_maxes[k] = ir.boundary_max;

        for (int j = 0; j < p_lam; ++j)        modes(k, j) = ir.beta_lambda(j);
        for (int j = 0; j < p_p; ++j)          modes(k, p_lam + j) = ir.beta_p(j);
        for (int j = 0; j < n_spatial; ++j)    modes(k, p_lam + p_p + j) = ir.z(j);

        // Reset to user-supplied defaults for the next grid point. Warm-
        // starting from the previous tau's mode is appealing but the
        // N-mixture identifiability ridge between (lambda intercept) and
        // (p intercept) shifts substantially across tau when z absorbs
        // different fractions of the spatial variance: at low tau the
        // optimum sits at high-lambda/low-p; at high tau, low-lambda/
        // high-p. Warm-starting across that ridge confounds Newton's
        // step-halving. Cold-starting each tau costs ~10-20 Newton
        // iterations per grid point and is dramatically more reliable.
        beta_lam = beta_lam_default;
        beta_p   = beta_p_default;
        z        = z_default;

        if (verbose) {
            Rcpp::Rcout << "[grid " << k + 1 << "/" << n_grid << "] tau=" << tau_grid[k]
                        << " log_marg=" << ir.log_marginal
                        << " n_iter=" << ir.n_iter
                        << " conv=" << ir.converged << "\n";
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("tau_grid")        = tau_grid,
        Rcpp::Named("log_marginal")    = log_marginals,
        Rcpp::Named("modes")           = modes,
        Rcpp::Named("n_iter")          = n_iters,
        Rcpp::Named("converged")       = convergeds,
        Rcpp::Named("grad_norm")       = grad_norms,
        Rcpp::Named("log_lik")         = log_liks,
        Rcpp::Named("boundary_max")    = boundary_maxes,
        Rcpp::Named("p_lambda")        = p_lam,
        Rcpp::Named("p_p")             = p_p,
        Rcpp::Named("n_spatial")       = n_spatial,
        Rcpp::Named("n_grid")          = n_grid,
        Rcpp::Named("K_max")           = K_max
    );
}

namespace {

// Compute log|Q(rho)| = log|D - rho * W| once per rho grid point. Returns
// -INFINITY if Q is not positive definite (callers should treat as a tail
// indicator and skip the grid point).
double log_det_Q_car_proper(
    int n_spatial, double rho,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors
) {
    MatrixXd Q = MatrixXd::Zero(n_spatial, n_spatial);
    for (int s = 0; s < n_spatial; ++s) {
        Q(s, s) = static_cast<double>(n_neighbors[s]);
        for (int kk = adj_row_ptr[s]; kk < adj_row_ptr[s + 1]; ++kk) {
            int t = adj_col_idx[kk];
            Q(s, t) = -rho;
        }
    }
    Eigen::LLT<MatrixXd> chol(Q);
    if (chol.info() != Eigen::Success) return R_NegInf;
    return 2.0 * chol.matrixL().toDenseMatrix().diagonal()
                    .array().log().sum();
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_nmix_car_proper(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector site_idx,
    Rcpp::IntegerVector map_site_to_unit_R,
    Rcpp::NumericMatrix X_lambda_R,
    Rcpp::NumericMatrix X_p_R,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    int n_spatial,
    Rcpp::NumericVector tau_grid,
    Rcpp::NumericVector rho_grid,
    Rcpp::NumericVector beta_lambda_init,
    Rcpp::NumericVector beta_p_init,
    Rcpp::Nullable<Rcpp::NumericVector> z_init = R_NilValue,
    int K_max = -1,
    int max_iter = 100,
    double tol = 1e-6,
    bool verbose = false
) {
    const int n_sites = X_lambda_R.nrow();
    const int p_lam   = X_lambda_R.ncol();
    const int n_obs   = X_p_R.nrow();
    const int p_p     = X_p_R.ncol();
    if ((int)y.size() != n_obs) Rcpp::stop("length(y) must equal nrow(X_p).");
    if ((int)site_idx.size() != n_obs) Rcpp::stop("length(site_idx) must equal nrow(X_p).");
    if ((int)map_site_to_unit_R.size() != n_sites) {
        Rcpp::stop("length(map_site_to_unit) must equal nrow(X_lambda).");
    }
    if ((int)beta_lambda_init.size() != p_lam) Rcpp::stop("beta_lambda_init length mismatch.");
    if ((int)beta_p_init.size() != p_p) Rcpp::stop("beta_p_init length mismatch.");
    if (K_max < 0) {
        int ymax = 0;
        for (int o = 0; o < n_obs; ++o) if (y[o] > ymax) ymax = y[o];
        K_max = ymax + 100;
    }

    std::vector<std::vector<int>> obs_by_site(n_sites);
    for (int o = 0; o < n_obs; ++o) {
        int s = site_idx[o] - 1;
        if (s < 0 || s >= n_sites) Rcpp::stop("site_idx out of range at obs %d.", o + 1);
        obs_by_site[s].push_back(o);
    }
    std::vector<int> map_site_to_unit(n_sites);
    for (int s = 0; s < n_sites; ++s) {
        int u = map_site_to_unit_R[s] - 1;
        if (u < 0 || u >= n_spatial) {
            Rcpp::stop("map_site_to_unit[%d] = %d out of range [1, %d].",
                       s + 1, map_site_to_unit_R[s], n_spatial);
        }
        map_site_to_unit[s] = u;
    }

    Map<MatrixXd> Xl(REAL(X_lambda_R), n_sites, p_lam);
    Map<MatrixXd> Xp(REAL(X_p_R), n_obs, p_p);

    VectorXd beta_lam_default = Map<VectorXd>(REAL(beta_lambda_init), p_lam);
    VectorXd beta_p_default   = Map<VectorXd>(REAL(beta_p_init), p_p);
    VectorXd z_default(n_spatial);
    if (z_init.isNotNull()) {
        Rcpp::NumericVector zi(z_init);
        if ((int)zi.size() != n_spatial) Rcpp::stop("length(z_init) must equal n_spatial.");
        for (int s = 0; s < n_spatial; ++s) z_default(s) = zi[s];
    } else {
        z_default.setZero();
    }

    const int n_tau = tau_grid.size();
    const int n_rho = rho_grid.size();
    const int n_grid = n_tau * n_rho;
    const int n_x    = p_lam + p_p + n_spatial;

    // Precompute log|Q(rho)| once per rho grid point.
    std::vector<double> log_det_Q_rho(n_rho);
    for (int r = 0; r < n_rho; ++r) {
        log_det_Q_rho[r] = log_det_Q_car_proper(
            n_spatial, rho_grid[r], adj_row_ptr, adj_col_idx, n_neighbors);
    }

    Rcpp::NumericMatrix tau_grid_out(n_grid, 2);  // (tau, rho) per grid point
    Rcpp::NumericVector log_marginals(n_grid);
    Rcpp::IntegerVector n_iters(n_grid);
    Rcpp::LogicalVector convergeds(n_grid);
    Rcpp::NumericVector grad_norms(n_grid);
    Rcpp::NumericVector log_liks(n_grid);
    Rcpp::NumericVector boundary_maxes(n_grid);
    Rcpp::NumericMatrix modes(n_grid, n_x);

    int k = 0;
    for (int r = 0; r < n_rho; ++r) {
        for (int t = 0; t < n_tau; ++t, ++k) {
            const double tau = tau_grid[t];
            const double rho = rho_grid[r];
            tau_grid_out(k, 0) = tau;
            tau_grid_out(k, 1) = rho;

            if (!R_finite(log_det_Q_rho[r])) {
                // Q(rho) not PD: skip but record placeholder.
                log_marginals[k]  = R_NegInf;
                n_iters[k]        = 0;
                convergeds[k]     = false;
                grad_norms[k]     = R_PosInf;
                log_liks[k]       = R_NegInf;
                boundary_maxes[k] = 0.0;
                continue;
            }

            // Cold-restart per grid point (same rationale as ICAR: the
            // identifiability ridge between (lambda intercept) and (p intercept)
            // shifts substantially across both tau and rho).
            VectorXd beta_lam = beta_lam_default;
            VectorXd beta_p   = beta_p_default;
            VectorXd z        = z_default;

            SpatialInnerResult ir = inner_newton_spatial_car(
                CarPriorKind::CAR_PROPER,
                p_lam, p_p, n_sites, n_spatial, n_obs,
                Xl, Xp, y, obs_by_site, map_site_to_unit,
                adj_row_ptr, adj_col_idx, n_neighbors,
                tau, rho, log_det_Q_rho[r],
                K_max, max_iter, tol,
                beta_lam, beta_p, z, verbose
            );
            log_marginals[k]  = ir.log_marginal;
            n_iters[k]        = ir.n_iter;
            convergeds[k]     = ir.converged;
            grad_norms[k]     = ir.grad_norm;
            log_liks[k]       = ir.log_lik;
            boundary_maxes[k] = ir.boundary_max;

            for (int j = 0; j < p_lam; ++j)        modes(k, j) = ir.beta_lambda(j);
            for (int j = 0; j < p_p; ++j)          modes(k, p_lam + j) = ir.beta_p(j);
            for (int j = 0; j < n_spatial; ++j)    modes(k, p_lam + p_p + j) = ir.z(j);

            if (verbose) {
                Rcpp::Rcout << "[grid " << k + 1 << "/" << n_grid
                            << "] tau=" << tau << " rho=" << rho
                            << " log_marg=" << ir.log_marginal
                            << " n_iter=" << ir.n_iter
                            << " conv=" << ir.converged << "\n";
            }
        }
    }

    Rcpp::colnames(tau_grid_out) = Rcpp::CharacterVector::create("tau", "rho");
    return Rcpp::List::create(
        Rcpp::Named("theta_grid")      = tau_grid_out,
        Rcpp::Named("tau_grid")        = tau_grid,
        Rcpp::Named("rho_grid")        = rho_grid,
        Rcpp::Named("log_det_Q_rho")   = Rcpp::wrap(log_det_Q_rho),
        Rcpp::Named("log_marginal")    = log_marginals,
        Rcpp::Named("modes")           = modes,
        Rcpp::Named("n_iter")          = n_iters,
        Rcpp::Named("converged")       = convergeds,
        Rcpp::Named("grad_norm")       = grad_norms,
        Rcpp::Named("log_lik")         = log_liks,
        Rcpp::Named("boundary_max")    = boundary_maxes,
        Rcpp::Named("p_lambda")        = p_lam,
        Rcpp::Named("p_p")             = p_p,
        Rcpp::Named("n_spatial")       = n_spatial,
        Rcpp::Named("n_grid")          = n_grid,
        Rcpp::Named("n_tau")           = n_tau,
        Rcpp::Named("n_rho")           = n_rho,
        Rcpp::Named("K_max")           = K_max
    );
}
