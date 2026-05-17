// nmix_laplace.cpp
// Non-spatial fixed-effects Laplace fit for the Royle (2004) N-mixture model.
//
// Algorithm:
//   Inner Newton on (beta_lambda, beta_p) with the marginal observed Fisher
//   information matrix as curvature
//      H_obs = sum_i [J_i^T D_i J_i  -  Var[N|y_i] * u_i u_i^T]
//   where D_i = diag(lambda_i, E[N|y_i] p_{ij}(1-p_{ij})) is the per-site
//   complete-data Fisher diagonal across (eta_lambda, eta_p_ij) and
//   u_i = J_i^T v_i with v_i = (-1, p_i1, ..., p_iJ_i)^T encodes the
//   shared-latent coupling between arms. Each iteration is one (p_lam+p_p)
//   x (p_lam+p_p) Cholesky solve. Quadratic convergence near the mode.
//
//   If H_obs is not PSD at a given iterate (can happen far from the mode when
//   Var[N|y] dominates), fall back to the complete-data Fisher block (drop
//   the var_N rank-1 correction; always PSD; EM-rate convergence). This is a
//   Levenberg-Marquardt-style safeguard.
//
//   Final vcov = H_obs^{-1} at the converged mode.
//
// Returns: beta_lambda, beta_p, log_lik, vcov, n_iter, converged, grad_norm,
//          mean_N, var_N, boundary_weight (per-site diagnostics).

#include "nmix_kernel.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

// [[Rcpp::depends(RcppEigen)]]

namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;

// One per-site kernel pass over all sites. Aggregates per-site outputs into
// vectors aligned with the long-form layout. Returns total log-lik.
double kernel_sweep(
    const std::vector<std::vector<int>>& obs_by_site,
    const Rcpp::IntegerVector& y_R,
    const VectorXd& eta_lambda,
    const VectorXd& eta_p_long,
    int K_max,
    VectorXd& grad_eta_lam,    // out [n_sites]
    VectorXd& info_eta_lam,    // out [n_sites]
    VectorXd& grad_eta_p,      // out [n_obs]
    VectorXd& info_eta_p,      // out [n_obs]
    VectorXd& mean_N,          // out [n_sites]
    VectorXd& var_N,            // out [n_sites]
    VectorXd& boundary_weight   // out [n_sites]
) {
    const int n_sites = (int)obs_by_site.size();
    double log_lik = 0.0;
    for (int s = 0; s < n_sites; ++s) {
        const auto& idx = obs_by_site[s];
        const int J = (int)idx.size();
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
        tulpa::NMixSiteResult r = tulpa::compute_nmix_site(
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

// Cheap log-lik-only sweep for line search trial points.
double kernel_log_lik_only(
    const std::vector<std::vector<int>>& obs_by_site,
    const Rcpp::IntegerVector& y_R,
    const VectorXd& eta_lambda,
    const VectorXd& eta_p_long,
    int K_max
) {
    double log_lik = 0.0;
    const int n_sites = (int)obs_by_site.size();
    for (int s = 0; s < n_sites; ++s) {
        const auto& idx = obs_by_site[s];
        const int J = (int)idx.size();
        if (J == 0) continue;
        std::vector<int>    y_site(J);
        std::vector<double> eta_p_site(J);
        for (int j = 0; j < J; ++j) {
            y_site[j]     = y_R[idx[j]];
            eta_p_site[j] = eta_p_long(idx[j]);
        }
        tulpa::NMixSiteResult r = tulpa::compute_nmix_site(
            y_site.data(), eta_p_site.data(), J,
            eta_lambda(s), K_max
        );
        if (!R_finite(r.log_lik)) return r.log_lik;
        log_lik += r.log_lik;
    }
    return log_lik;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List cpp_nmix_laplace_fixed(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector site_idx,
    Rcpp::NumericMatrix X_lambda_R,
    Rcpp::NumericMatrix X_p_R,
    Rcpp::NumericVector beta_lambda_init,
    Rcpp::NumericVector beta_p_init,
    int K_max,
    int max_iter,
    double tol,
    bool verbose
) {
    const int n_sites = X_lambda_R.nrow();
    const int p_lam   = X_lambda_R.ncol();
    const int n_obs   = X_p_R.nrow();
    const int p_p     = X_p_R.ncol();
    if ((int)y.size() != n_obs) Rcpp::stop("y length must match nrow(X_p).");
    if ((int)site_idx.size() != n_obs) Rcpp::stop("site_idx length must match nrow(X_p).");
    if ((int)beta_lambda_init.size() != p_lam) Rcpp::stop("beta_lambda_init must have length ncol(X_lambda).");
    if ((int)beta_p_init.size() != p_p) Rcpp::stop("beta_p_init must have length ncol(X_p).");

    // Group observations by site (preserves input order within each site).
    std::vector<std::vector<int>> obs_by_site(n_sites);
    for (int o = 0; o < n_obs; ++o) {
        int s = site_idx[o] - 1;
        if (s < 0 || s >= n_sites) Rcpp::stop("site_idx out of range at obs %d.", o + 1);
        obs_by_site[s].push_back(o);
    }

    Map<MatrixXd> Xl(REAL(X_lambda_R), n_sites, p_lam);
    Map<MatrixXd> Xp(REAL(X_p_R), n_obs, p_p);
    VectorXd beta_lam = Map<VectorXd>(REAL(beta_lambda_init), p_lam);
    VectorXd beta_p   = Map<VectorXd>(REAL(beta_p_init), p_p);

    // Allocate per-iteration scratch.
    VectorXd grad_eta_lam(n_sites), info_eta_lam(n_sites);
    VectorXd mean_N(n_sites), var_N(n_sites), boundary_weight(n_sites);
    VectorXd grad_eta_p(n_obs), info_eta_p(n_obs);

    double log_lik = R_NegInf;
    double grad_norm = R_PosInf;
    bool converged = false;
    int iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {
        VectorXd eta_lam = Xl * beta_lam;
        VectorXd eta_p_long = Xp * beta_p;

        double new_log_lik = kernel_sweep(
            obs_by_site, y, eta_lam, eta_p_long, K_max,
            grad_eta_lam, info_eta_lam, grad_eta_p, info_eta_p,
            mean_N, var_N, boundary_weight
        );
        log_lik = new_log_lik;

        // Aggregate gradient and block-diagonal Fisher Hessians.
        VectorXd grad_beta_lam = Xl.transpose() * grad_eta_lam;
        VectorXd grad_beta_p   = Xp.transpose() * grad_eta_p;
        grad_norm = std::sqrt(grad_beta_lam.squaredNorm() + grad_beta_p.squaredNorm());

        if (verbose) {
            double bmax = boundary_weight.maxCoeff();
            Rcpp::Rcout << "iter " << iter
                        << "  log_lik " << log_lik
                        << "  grad_norm " << grad_norm
                        << "  boundary_max " << bmax << "\n";
        }
        if (grad_norm < tol) {
            converged = true;
            break;
        }

        // Assemble marginal observed Fisher information H_obs as the
        // (p_lam + p_p) x (p_lam + p_p) curvature matrix.
        const int p_total_iter = p_lam + p_p;
        MatrixXd H_iter = MatrixXd::Zero(p_total_iter, p_total_iter);
        for (int s = 0; s < n_sites; ++s) {
            const auto& idx = obs_by_site[s];
            const int J = (int)idx.size();
            if (J == 0) continue;
            // Block-diagonal Fisher contribution.
            double w_lam = info_eta_lam(s);
            if (w_lam > 0.0) {
                H_iter.block(0, 0, p_lam, p_lam)
                    .selfadjointView<Eigen::Lower>()
                    .rankUpdate(Xl.row(s).transpose(), w_lam);
            }
            for (int j = 0; j < J; ++j) {
                double w_p = info_eta_p(idx[j]);
                if (w_p > 0.0) {
                    H_iter.block(p_lam, p_lam, p_p, p_p)
                        .selfadjointView<Eigen::Lower>()
                        .rankUpdate(Xp.row(idx[j]).transpose(), w_p);
                }
            }
            // Var[N|y_i] rank-1 marginal correction.
            if (var_N(s) > 0.0) {
                VectorXd u(p_total_iter);
                u.segment(0, p_lam) = -Xl.row(s).transpose();
                VectorXd ssum = VectorXd::Zero(p_p);
                for (int j = 0; j < J; ++j) {
                    double e = eta_p_long(idx[j]);
                    double p_ij;
                    if (e > 0.0) p_ij = 1.0 / (1.0 + std::exp(-e));
                    else { double ee = std::exp(e); p_ij = ee / (1.0 + ee); }
                    ssum += p_ij * Xp.row(idx[j]).transpose();
                }
                u.segment(p_lam, p_p) = ssum;
                H_iter.selfadjointView<Eigen::Lower>().rankUpdate(u, -var_N(s));
            }
        }
        H_iter = H_iter.selfadjointView<Eigen::Lower>();

        VectorXd grad_total(p_total_iter);
        grad_total.segment(0, p_lam) = grad_beta_lam;
        grad_total.segment(p_lam, p_p) = grad_beta_p;

        VectorXd delta_total;
        Eigen::LLT<MatrixXd> chol(H_iter);
        if (chol.info() == Eigen::Success) {
            delta_total = chol.solve(grad_total);
        } else {
            // Fallback: complete-data Fisher (drop var_N correction, always PSD).
            MatrixXd H_fisher = MatrixXd::Zero(p_total_iter, p_total_iter);
            for (int s = 0; s < n_sites; ++s) {
                if (info_eta_lam(s) > 0.0) {
                    H_fisher.block(0, 0, p_lam, p_lam)
                        .selfadjointView<Eigen::Lower>()
                        .rankUpdate(Xl.row(s).transpose(), info_eta_lam(s));
                }
            }
            for (int o = 0; o < n_obs; ++o) {
                if (info_eta_p(o) > 0.0) {
                    H_fisher.block(p_lam, p_lam, p_p, p_p)
                        .selfadjointView<Eigen::Lower>()
                        .rankUpdate(Xp.row(o).transpose(), info_eta_p(o));
                }
            }
            H_fisher = H_fisher.selfadjointView<Eigen::Lower>();
            Eigen::LLT<MatrixXd> chol_f(H_fisher);
            if (chol_f.info() != Eigen::Success) {
                Rcpp::warning("Fallback Cholesky failure at iter %d -- aborting Newton.", iter);
                break;
            }
            delta_total = chol_f.solve(grad_total);
            if (verbose) Rcpp::Rcout << "  (Fisher fallback)\n";
        }
        VectorXd delta_lam = delta_total.segment(0, p_lam);
        VectorXd delta_p   = delta_total.segment(p_lam, p_p);

        // Step with halving. Accept the first step that does not regress the
        // log-lik. Newton with complete-data Fisher info is monotone in the
        // ideal case; halving guards against numerical or transient breaches.
        double step = 1.0;
        VectorXd beta_lam_try, beta_p_try;
        bool stepped = false;
        for (int h = 0; h < 12; ++h) {
            beta_lam_try = beta_lam + step * delta_lam;
            beta_p_try   = beta_p   + step * delta_p;
            VectorXd eta_lam_try = Xl * beta_lam_try;
            VectorXd eta_p_try   = Xp * beta_p_try;
            double ll_try = kernel_log_lik_only(obs_by_site, y, eta_lam_try, eta_p_try, K_max);
            if (R_finite(ll_try) && ll_try >= log_lik - 1e-10) {
                beta_lam = beta_lam_try;
                beta_p   = beta_p_try;
                stepped  = true;
                break;
            }
            step *= 0.5;
        }
        if (!stepped) {
            Rcpp::warning("Step halving exhausted at iter %d -- aborting Newton.", iter);
            break;
        }
    }

    // Recompute kernel state at converged beta for the final log-lik and the
    // marginal observed information matrix.
    VectorXd eta_lam = Xl * beta_lam;
    VectorXd eta_p_long = Xp * beta_p;
    double log_lik_final = kernel_sweep(
        obs_by_site, y, eta_lam, eta_p_long, K_max,
        grad_eta_lam, info_eta_lam, grad_eta_p, info_eta_p,
        mean_N, var_N, boundary_weight
    );

    // Marginal observed information at the mode.
    //   H_obs = block-diagonal Fisher (across arms) - sum_i Var[N|y_i] u_i u_i^T
    // where u_i = (-X_lam[i,], sum_j p_ij X_p[ij,]).
    const int p_total = p_lam + p_p;
    MatrixXd H_obs = MatrixXd::Zero(p_total, p_total);

    for (int s = 0; s < n_sites; ++s) {
        const auto& idx = obs_by_site[s];
        const int J = (int)idx.size();
        if (J == 0) continue;

        // Block-diagonal Fisher contributions.
        double w_lam = info_eta_lam(s);
        if (w_lam > 0.0) {
            H_obs.block(0, 0, p_lam, p_lam)
                .selfadjointView<Eigen::Lower>()
                .rankUpdate(Xl.row(s).transpose(), w_lam);
        }
        for (int j = 0; j < J; ++j) {
            double w_p = info_eta_p(idx[j]);
            if (w_p > 0.0) {
                H_obs.block(p_lam, p_lam, p_p, p_p)
                    .selfadjointView<Eigen::Lower>()
                    .rankUpdate(Xp.row(idx[j]).transpose(), w_p);
            }
        }

        // Var[N|y_i] rank-1 correction encoding the cross-arm and cross-visit
        // coupling that the marginal likelihood induces.
        if (var_N(s) > 0.0) {
            VectorXd u(p_total);
            u.segment(0, p_lam) = -Xl.row(s).transpose();
            VectorXd ssum = VectorXd::Zero(p_p);
            for (int j = 0; j < J; ++j) {
                double e = eta_p_long(idx[j]);
                double p_ij;
                if (e > 0.0) {
                    p_ij = 1.0 / (1.0 + std::exp(-e));
                } else {
                    double ee = std::exp(e);
                    p_ij = ee / (1.0 + ee);
                }
                ssum += p_ij * Xp.row(idx[j]).transpose();
            }
            u.segment(p_lam, p_p) = ssum;
            H_obs.selfadjointView<Eigen::Lower>().rankUpdate(u, -var_N(s));
        }
    }
    H_obs = H_obs.selfadjointView<Eigen::Lower>();

    // Invert for vcov.
    MatrixXd vcov;
    bool vcov_ok = true;
    {
        Eigen::LLT<MatrixXd> chol(H_obs);
        if (chol.info() == Eigen::Success) {
            vcov = chol.solve(MatrixXd::Identity(p_total, p_total));
        } else {
            vcov_ok = false;
            vcov = MatrixXd::Constant(p_total, p_total, NA_REAL);
        }
    }

    Rcpp::NumericVector beta_lam_out(beta_lam.data(), beta_lam.data() + p_lam);
    Rcpp::NumericVector beta_p_out(beta_p.data(), beta_p.data() + p_p);

    return Rcpp::List::create(
        Rcpp::Named("beta_lambda")     = beta_lam_out,
        Rcpp::Named("beta_p")          = beta_p_out,
        Rcpp::Named("log_lik")         = log_lik_final,
        Rcpp::Named("vcov")            = Rcpp::wrap(vcov),
        Rcpp::Named("vcov_ok")         = vcov_ok,
        Rcpp::Named("H_obs")           = Rcpp::wrap(H_obs),
        Rcpp::Named("n_iter")          = iter + (converged ? 1 : 0),
        Rcpp::Named("converged")       = converged,
        Rcpp::Named("grad_norm")       = grad_norm,
        Rcpp::Named("mean_N")          = Rcpp::wrap(mean_N),
        Rcpp::Named("var_N")           = Rcpp::wrap(var_N),
        Rcpp::Named("boundary_weight") = Rcpp::wrap(boundary_weight)
    );
}
