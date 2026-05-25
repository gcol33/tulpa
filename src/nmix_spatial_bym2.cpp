// nmix_spatial_bym2.cpp
// Nested Laplace for the spatial Royle (2004) N-mixture model with a BYM2
// (Riebler et al. 2016) prior on the abundance-arm spatial offset.
//
// Outer grid: 2D over (sigma, rho). sigma is the joint marginal standard
// deviation of the spatial offset; rho is the spatial fraction of
// variance. Inner Newton at fixed (sigma_k, rho_k) finds the joint mode of
//   x = (beta_lambda [p_lam], beta_p [p_p], v [n_spatial], w [n_spatial])
// using complete-data Fisher curvature with a fallback when the marginal
// observed-information matrix is non-PSD.
//
// Phi (the actual offset) is sigma * (sqrt(rho/scale)*v + sqrt(1-rho)*w);
// the priors on (v, w) are independent of (sigma, rho), so the prior
// contribution to log p(y | sigma, rho) is constant across grid points
// and absorbs into the same tau-independent additive constant as the
// (2 pi)^{n_x/2} Cartesian factor. Only the inner mode and log|H| change
// with (sigma, rho), and so does the data likelihood through the
// coefficients (a, b) of (v, w) in eta_lambda.

#include "nmix_kernel.h"
#include "nmix_spatial_kernel.h"      // nmix_kernel_sweep_spatial, log_lik_only_spatial
#include "nmix_spatial_kernel_bym2.h"
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
using tulpa::compute_eta_lambda_bym2;
using tulpa::nmix_assemble_obs_info_bym2;
using tulpa::nmix_assemble_complete_fisher_bym2;
using tulpa::nmix_add_bym2_prior_to_grad_and_H;
using tulpa::nmix_add_bym2_prior_to_H_only;
using tulpa::nmix_bym2_log_prior;
using tulpa::nmix_center_v_bym2;

inline void add_diagonal_ridge_bym2(MatrixXd& H, double rel_ridge = 1e-10) {
    const int n = static_cast<int>(H.rows());
    if (n == 0) return;
    double mean_diag = 0.0;
    for (int i = 0; i < n; ++i) mean_diag += H(i, i);
    mean_diag /= n;
    double ridge = std::max(rel_ridge * mean_diag, 1e-12);
    for (int i = 0; i < n; ++i) H(i, i) += ridge;
}

// Marginal (beta_lambda, beta_p) covariance from the factored joint Hessian:
// the top-left p_beta block of H^{-1} (the Gaussian/Laplace marginal of the
// coefficient subvector), solved as H X = [I; 0].
inline MatrixXd nmix_beta_cov_from_chol_bym2(const Eigen::LLT<MatrixXd>& chol,
                                             int n_x, int p_beta) {
    MatrixXd E = MatrixXd::Zero(n_x, p_beta);
    E.topLeftCorner(p_beta, p_beta).setIdentity();
    MatrixXd Hinv_cols = chol.solve(E);
    return Hinv_cols.topLeftCorner(p_beta, p_beta);
}

// Constrained coefficient covariance for BYM2. The structured component v is
// rank-deficient (sum-to-zero), so the (intercept, v-mean) direction is flat in
// the joint posterior and the unconstrained intercept variance is meaningless.
// Pin sum(v)=0 with a large quadratic penalty (the penalty-method form of the
// constraint); the iid component w is proper and is left alone. `H` is by value.
inline MatrixXd nmix_beta_cov_bym2(MatrixXd H, int n_x, int p_beta,
                                   int v_start, int n_spatial) {
    double md = H.diagonal().head(p_beta).cwiseAbs().mean();
    if (!(md > 0)) md = 1.0;
    const double kappa = 1e6 * md;
    for (int i = 0; i < n_spatial; ++i)
        for (int j = 0; j < n_spatial; ++j)
            H(v_start + i, v_start + j) += kappa;
    Eigen::LLT<MatrixXd> chol(H);
    if (chol.info() != Eigen::Success)
        return MatrixXd::Constant(p_beta, p_beta, R_NaN);
    return nmix_beta_cov_from_chol_bym2(chol, n_x, p_beta);
}

struct BYM2InnerResult {
    VectorXd beta_lambda;
    VectorXd beta_p;
    VectorXd v;
    VectorXd w;
    MatrixXd cov_beta;    // (p_lam+p_p) marginal coefficient covariance at mode
    double log_lik;
    double log_marginal;
    double grad_norm;
    int n_iter;
    bool converged;
    double boundary_max;
};

BYM2InnerResult inner_newton_bym2(
    int p_lam, int p_p, int n_sites, int n_spatial, int n_obs,
    double a, double b,
    const Map<MatrixXd>& Xl,
    const Map<MatrixXd>& Xp,
    const Rcpp::IntegerVector& y_R,
    const std::vector<std::vector<int>>& obs_by_site,
    const std::vector<int>& map_site_to_unit,
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors,
    int K_max,
    int max_iter,
    double tol,
    const VectorXd& beta_lam_init,
    const VectorXd& beta_p_init,
    const VectorXd& v_init,
    const VectorXd& w_init,
    bool verbose
) {
    BYM2InnerResult res;
    res.beta_lambda = beta_lam_init;
    res.beta_p      = beta_p_init;
    res.v           = v_init;
    res.w           = w_init;
    res.converged   = false;
    res.n_iter      = 0;

    const int n_x = p_lam + p_p + 2 * n_spatial;

    VectorXd grad_eta_lam(n_sites), info_eta_lam(n_sites);
    VectorXd mean_N(n_sites), var_N(n_sites), boundary_weight(n_sites);
    VectorXd grad_eta_p(n_obs), info_eta_p(n_obs);
    VectorXd eta_lam(n_sites);
    VectorXd eta_p_long(n_obs);

    double log_lik = R_NegInf;
    double grad_norm = R_PosInf;

    for (int iter = 0; iter < max_iter; ++iter) {
        compute_eta_lambda_bym2(Xl, res.beta_lambda, res.v, res.w, a, b,
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
        // grad_v[u] = a * sum_{s: map[s]=u} grad_eta_lam[s]
        // grad_w[u] = b * sum_{s: map[s]=u} grad_eta_lam[s]
        const int v_start = p_lam + p_p;
        const int w_start = v_start + n_spatial;
        for (int s = 0; s < n_sites; ++s) {
            const int u = map_site_to_unit[s];
            grad(v_start + u) += a * grad_eta_lam(s);
            grad(w_start + u) += b * grad_eta_lam(s);
        }

        MatrixXd H = MatrixXd::Zero(n_x, n_x);
        nmix_assemble_obs_info_bym2(
            p_lam, p_p, n_spatial, a, b,
            Xl, Xp, eta_p_long, obs_by_site, map_site_to_unit,
            info_eta_lam, info_eta_p, var_N, H
        );
        nmix_add_bym2_prior_to_grad_and_H(
            p_lam, p_p, n_spatial,
            adj_row_ptr, adj_col_idx, n_neighbors,
            res.v, res.w, grad, H
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

        // Cholesky solve. Fall back to complete-data Fisher if non-PSD.
        add_diagonal_ridge_bym2(H);
        VectorXd delta;
        Eigen::LLT<MatrixXd> chol(H);
        if (chol.info() == Eigen::Success) {
            delta = chol.solve(grad);
        } else {
            MatrixXd H_f = MatrixXd::Zero(n_x, n_x);
            nmix_assemble_complete_fisher_bym2(
                p_lam, p_p, n_spatial, a, b,
                Xl, Xp, obs_by_site, map_site_to_unit,
                info_eta_lam, info_eta_p, H_f
            );
            // Re-add the prior to the Fisher fallback (grad already updated
            // above). nmix_add_bym2_prior_to_H_only adds Q + I to the v, w
            // blocks; we only need the Hessian here.
            nmix_add_bym2_prior_to_H_only(
                p_lam, p_p, n_spatial,
                adj_row_ptr, adj_col_idx, n_neighbors, H_f
            );
            add_diagonal_ridge_bym2(H_f);
            Eigen::LLT<MatrixXd> chol_f(H_f);
            if (chol_f.info() != Eigen::Success) {
                Rcpp::warning("Cholesky failure (complete-data fallback) at iter %d, a %.4f, b %.4f.",
                              iter, a, b);
                break;
            }
            delta = chol_f.solve(grad);
            if (verbose) Rcpp::Rcout << "    (Fisher fallback)\n";
        }

        VectorXd delta_lam = delta.segment(0, p_lam);
        VectorXd delta_p   = delta.segment(p_lam, p_p);
        VectorXd delta_v   = delta.segment(v_start, n_spatial);
        VectorXd delta_w   = delta.segment(w_start, n_spatial);

        // Step halving on the joint log-posterior.
        double step = 1.0;
        bool stepped = false;
        VectorXd beta_lam_try, beta_p_try, v_try, w_try;
        VectorXd eta_lam_try(n_sites), eta_p_try(n_obs);
        for (int h = 0; h < 12; ++h) {
            beta_lam_try = res.beta_lambda + step * delta_lam;
            beta_p_try   = res.beta_p      + step * delta_p;
            v_try        = res.v           + step * delta_v;
            w_try        = res.w           + step * delta_w;

            compute_eta_lambda_bym2(Xl, beta_lam_try, v_try, w_try, a, b,
                                    map_site_to_unit, eta_lam_try);
            eta_p_try.noalias() = Xp * beta_p_try;
            double ll_try = nmix_kernel_log_lik_only_spatial(
                obs_by_site, y_R, eta_lam_try, eta_p_try, K_max
            );
            double lp_try = nmix_bym2_log_prior(
                n_spatial, adj_row_ptr, adj_col_idx, n_neighbors,
                v_try, w_try
            );
            double obj_try = ll_try + lp_try;
            double obj_cur = log_lik + nmix_bym2_log_prior(
                n_spatial, adj_row_ptr, adj_col_idx, n_neighbors,
                res.v, res.w
            );
            if (R_finite(obj_try) && obj_try >= obj_cur - 1e-10) {
                res.beta_lambda = beta_lam_try;
                res.beta_p      = beta_p_try;
                res.v           = v_try;
                res.w           = w_try;

                // Sum-to-zero on v (ICAR identifiability). w is identified
                // by its iid prior and is not centred.
                VectorXd x_holder(n_x);
                x_holder.segment(0, p_lam) = res.beta_lambda;
                x_holder.segment(p_lam, p_p) = res.beta_p;
                x_holder.segment(v_start, n_spatial) = res.v;
                x_holder.segment(w_start, n_spatial) = res.w;
                nmix_center_v_bym2(p_lam, p_p, n_spatial, x_holder);
                res.v = x_holder.segment(v_start, n_spatial);

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
    compute_eta_lambda_bym2(Xl, res.beta_lambda, res.v, res.w, a, b,
                            map_site_to_unit, eta_lam);
    eta_p_long.noalias() = Xp * res.beta_p;
    double log_lik_final = nmix_kernel_sweep_spatial(
        obs_by_site, y_R, eta_lam, eta_p_long, K_max,
        grad_eta_lam, info_eta_lam, grad_eta_p, info_eta_p,
        mean_N, var_N, boundary_weight
    );
    double log_prior_final = nmix_bym2_log_prior(
        n_spatial, adj_row_ptr, adj_col_idx, n_neighbors,
        res.v, res.w
    );

    MatrixXd H_final = MatrixXd::Zero(n_x, n_x);
    nmix_assemble_obs_info_bym2(
        p_lam, p_p, n_spatial, a, b,
        Xl, Xp, eta_p_long, obs_by_site, map_site_to_unit,
        info_eta_lam, info_eta_p, var_N, H_final
    );
    nmix_add_bym2_prior_to_H_only(
        p_lam, p_p, n_spatial,
        adj_row_ptr, adj_col_idx, n_neighbors, H_final
    );
    add_diagonal_ridge_bym2(H_final);
    const int p_beta = p_lam + p_p;
    Eigen::LLT<MatrixXd> chol(H_final);
    double log_det_H;
    const int v_start = p_lam + p_p;
    if (chol.info() == Eigen::Success) {
        log_det_H = 2.0 * chol.matrixL().toDenseMatrix().diagonal()
                              .array().log().sum();
        res.cov_beta = nmix_beta_cov_bym2(H_final, n_x, p_beta, v_start, n_spatial);
    } else {
        MatrixXd H_f = MatrixXd::Zero(n_x, n_x);
        nmix_assemble_complete_fisher_bym2(
            p_lam, p_p, n_spatial, a, b,
            Xl, Xp, obs_by_site, map_site_to_unit,
            info_eta_lam, info_eta_p, H_f
        );
        nmix_add_bym2_prior_to_H_only(
            p_lam, p_p, n_spatial,
            adj_row_ptr, adj_col_idx, n_neighbors, H_f
        );
        add_diagonal_ridge_bym2(H_f);
        Eigen::LLT<MatrixXd> chol_f(H_f);
        if (chol_f.info() != Eigen::Success) {
            res.cov_beta = MatrixXd::Constant(p_beta, p_beta, R_NaN);
            res.log_marginal = R_NegInf;
            res.log_lik = log_lik_final;
            res.grad_norm = grad_norm;
            res.boundary_max = boundary_weight.maxCoeff();
            return res;
        }
        log_det_H = 2.0 * chol_f.matrixL().toDenseMatrix().diagonal()
                                .array().log().sum();
        res.cov_beta = nmix_beta_cov_bym2(H_f, n_x, p_beta, v_start, n_spatial);
    }

    res.log_lik = log_lik_final;
    res.log_marginal = log_lik_final + log_prior_final - 0.5 * log_det_H;
    res.grad_norm = grad_norm;
    res.boundary_max = boundary_weight.maxCoeff();
    return res;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List cpp_nested_laplace_nmix_bym2(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector site_idx,
    Rcpp::IntegerVector map_site_to_unit_R,
    Rcpp::NumericMatrix X_lambda_R,
    Rcpp::NumericMatrix X_p_R,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    int n_spatial,
    Rcpp::NumericVector sigma_grid,
    Rcpp::NumericVector rho_grid,
    double scale_factor,
    Rcpp::NumericVector beta_lambda_init,
    Rcpp::NumericVector beta_p_init,
    Rcpp::Nullable<Rcpp::NumericVector> v_init = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> w_init = R_NilValue,
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
    if (scale_factor <= 0) Rcpp::stop("scale_factor must be positive.");
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
    VectorXd v_default(n_spatial);
    VectorXd w_default(n_spatial);
    if (v_init.isNotNull()) {
        Rcpp::NumericVector vi(v_init);
        if ((int)vi.size() != n_spatial) Rcpp::stop("length(v_init) must equal n_spatial.");
        for (int s = 0; s < n_spatial; ++s) v_default(s) = vi[s];
    } else {
        v_default.setZero();
    }
    if (w_init.isNotNull()) {
        Rcpp::NumericVector wi(w_init);
        if ((int)wi.size() != n_spatial) Rcpp::stop("length(w_init) must equal n_spatial.");
        for (int s = 0; s < n_spatial; ++s) w_default(s) = wi[s];
    } else {
        w_default.setZero();
    }

    const int n_sigma = sigma_grid.size();
    const int n_rho   = rho_grid.size();
    const int n_grid  = n_sigma * n_rho;
    const int n_x     = p_lam + p_p + 2 * n_spatial;

    Rcpp::NumericMatrix theta_grid_out(n_grid, 2);  // (sigma, rho)
    Rcpp::NumericVector log_marginals(n_grid);
    Rcpp::IntegerVector n_iters(n_grid);
    Rcpp::LogicalVector convergeds(n_grid);
    Rcpp::NumericVector grad_norms(n_grid);
    Rcpp::NumericVector log_liks(n_grid);
    Rcpp::NumericVector boundary_maxes(n_grid);
    Rcpp::NumericMatrix modes(n_grid, n_x);
    Rcpp::List cov_blocks(n_grid);   // per-grid marginal coef covariance

    int k = 0;
    for (int r = 0; r < n_rho; ++r) {
        for (int sg = 0; sg < n_sigma; ++sg, ++k) {
            const double sigma = sigma_grid[sg];
            const double rho   = rho_grid[r];
            theta_grid_out(k, 0) = sigma;
            theta_grid_out(k, 1) = rho;

            if (sigma <= 0 || rho < 0 || rho > 1) {
                log_marginals[k]  = R_NegInf;
                n_iters[k]        = 0;
                convergeds[k]     = false;
                grad_norms[k]     = R_PosInf;
                log_liks[k]       = R_NegInf;
                boundary_maxes[k] = 0.0;
                continue;
            }

            const double a = sigma * std::sqrt(rho / scale_factor);
            const double b = sigma * std::sqrt(1.0 - rho);

            // Cold-restart per grid point (the (lambda, p) identifiability
            // ridge shifts with the joint variance budget controlled by
            // (sigma, rho); warm-starting across the ridge confounds
            // step-halving).
            VectorXd beta_lam = beta_lam_default;
            VectorXd beta_p   = beta_p_default;
            VectorXd v        = v_default;
            VectorXd w        = w_default;

            BYM2InnerResult ir = inner_newton_bym2(
                p_lam, p_p, n_sites, n_spatial, n_obs,
                a, b,
                Xl, Xp, y, obs_by_site, map_site_to_unit,
                adj_row_ptr, adj_col_idx, n_neighbors,
                K_max, max_iter, tol,
                beta_lam, beta_p, v, w, verbose
            );
            log_marginals[k]  = ir.log_marginal;
            n_iters[k]        = ir.n_iter;
            convergeds[k]     = ir.converged;
            grad_norms[k]     = ir.grad_norm;
            log_liks[k]       = ir.log_lik;
            boundary_maxes[k] = ir.boundary_max;

            for (int j = 0; j < p_lam; ++j)        modes(k, j) = ir.beta_lambda(j);
            for (int j = 0; j < p_p; ++j)          modes(k, p_lam + j) = ir.beta_p(j);
            for (int j = 0; j < n_spatial; ++j)    modes(k, p_lam + p_p + j) = ir.v(j);
            for (int j = 0; j < n_spatial; ++j)    modes(k, p_lam + p_p + n_spatial + j) = ir.w(j);
            cov_blocks[k] = Rcpp::wrap(ir.cov_beta);

            if (verbose) {
                Rcpp::Rcout << "[grid " << k + 1 << "/" << n_grid
                            << "] sigma=" << sigma << " rho=" << rho
                            << " a=" << a << " b=" << b
                            << " log_marg=" << ir.log_marginal
                            << " n_iter=" << ir.n_iter
                            << " conv=" << ir.converged << "\n";
            }
        }
    }

    Rcpp::colnames(theta_grid_out) = Rcpp::CharacterVector::create("sigma", "rho");
    return Rcpp::List::create(
        Rcpp::Named("theta_grid")      = theta_grid_out,
        Rcpp::Named("sigma_grid")      = sigma_grid,
        Rcpp::Named("rho_grid")        = rho_grid,
        Rcpp::Named("scale_factor")    = scale_factor,
        Rcpp::Named("log_marginal")    = log_marginals,
        Rcpp::Named("modes")           = modes,
        Rcpp::Named("cov_blocks")      = cov_blocks,
        Rcpp::Named("n_iter")          = n_iters,
        Rcpp::Named("converged")       = convergeds,
        Rcpp::Named("grad_norm")       = grad_norms,
        Rcpp::Named("log_lik")         = log_liks,
        Rcpp::Named("boundary_max")    = boundary_maxes,
        Rcpp::Named("p_lambda")        = p_lam,
        Rcpp::Named("p_p")             = p_p,
        Rcpp::Named("n_spatial")       = n_spatial,
        Rcpp::Named("n_grid")          = n_grid,
        Rcpp::Named("n_sigma")         = n_sigma,
        Rcpp::Named("n_rho")           = n_rho,
        Rcpp::Named("K_max")           = K_max
    );
}
