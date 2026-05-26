// nmix_laplace.cpp
// Non-spatial fixed-effects Laplace fit for the Royle (2004) N-mixture model
// with a Poisson OR negative-binomial abundance mixing distribution.
//
// Algorithm (abundance arm parameters):
//   - beta = (beta_lambda, beta_p): inner Newton on the marginal observed Fisher
//     information curvature
//        H_obs = sum_i [J_i^T D_i J_i  -  Var[N|y_i] * u_i u_i^T]
//     with D_i = diag(info_eta_lambda_i, info_eta_p_ij) the per-site complete-
//     data Fisher diagonal and u_i = J_i^T v_i, v_i = (-v_lam, p_i1,...) the
//     shared-latent coupling. v_lam = 1 (Poisson) or 1-q (NB), supplied by the
//     kernel as score_wt_lambda. Each iteration is one (p_lam+p_p)^2 Cholesky
//     solve; quadratic convergence near the mode. A complete-data Fisher block
//     (drop the var_N rank-1, always PSD) is the Levenberg-Marquardt fallback.
//   - theta = log r (NB dispersion): a single global scalar, profiled OUTSIDE
//     the beta-Newton by block coordinate ascent. Each outer step takes a
//     safeguarded 1-D Newton step on the profile score d log L / d theta
//     (analytic, from the kernel), holding beta fixed for the line search, then
//     re-solves the inner beta-Newton. Poisson is the nb = false case (no theta).
//
//   Final vcov = H_obs^{-1} at the converged mode, over (beta_lambda, beta_p)
//   and, under NB, theta -- a joint observed-information inverse that includes
//   SE(log r) and its correlation with beta (Louis 1982; see nmix_kernel.h).
//
// Returns: beta_lambda, beta_p, [log_r], log_lik, vcov, n_iter, converged,
//          grad_norm, mean_N, var_N, boundary_weight (per-site diagnostics).

#include "nmix_kernel.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <vector>

// [[Rcpp::depends(RcppEigen)]]

namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;

// Per-site / per-visit kernel outputs for one sweep over all sites. The NB
// dispersion fields are Poisson-neutral when r = +Inf (score_wt_lambda = 1,
// the rest 0), so the same struct serves both mixtures.
struct SweepState {
    VectorXd grad_eta_lam, info_eta_lam, mean_N, var_N, boundary_weight;
    VectorXd score_wt_lambda;                                  // N-coeff of s_lambda
    VectorXd grad_theta, info_theta, info_lam_theta, cov_N_stheta, var_stheta;
    VectorXd grad_eta_p, info_eta_p;                           // per-visit (n_obs)

    void resize(int n_sites, int n_obs) {
        grad_eta_lam.resize(n_sites);  info_eta_lam.resize(n_sites);
        mean_N.resize(n_sites);        var_N.resize(n_sites);
        boundary_weight.resize(n_sites); score_wt_lambda.resize(n_sites);
        grad_theta.resize(n_sites);    info_theta.resize(n_sites);
        info_lam_theta.resize(n_sites); cov_N_stheta.resize(n_sites);
        var_stheta.resize(n_sites);
        grad_eta_p.resize(n_obs);      info_eta_p.resize(n_obs);
    }
};

// One per-site kernel pass at the current (eta_lambda, eta_p, r). Fills `st`
// and returns the total log-lik.
double kernel_sweep(
    const std::vector<std::vector<int>>& obs_by_site,
    const Rcpp::IntegerVector& y_R,
    const VectorXd& eta_lambda,
    const VectorXd& eta_p_long,
    int K_max, double r,
    SweepState& st
) {
    const int n_sites = (int)obs_by_site.size();
    double log_lik = 0.0;
    for (int s = 0; s < n_sites; ++s) {
        const auto& idx = obs_by_site[s];
        const int J = (int)idx.size();
        if (J == 0) {
            st.grad_eta_lam(s) = 0.0;
            st.info_eta_lam(s) = 0.0;
            st.mean_N(s) = std::exp(eta_lambda(s));
            st.var_N(s)  = std::exp(eta_lambda(s));
            st.boundary_weight(s) = 0.0;
            st.score_wt_lambda(s) = 1.0;
            st.grad_theta(s) = 0.0;     st.info_theta(s) = 0.0;
            st.info_lam_theta(s) = 0.0; st.cov_N_stheta(s) = 0.0;
            st.var_stheta(s) = 0.0;
            continue;
        }
        std::vector<int>    y_site(J);
        std::vector<double> eta_p_site(J);
        for (int j = 0; j < J; ++j) {
            y_site[j]     = y_R[idx[j]];
            eta_p_site[j] = eta_p_long(idx[j]);
        }
        tulpa::NMixSiteResult res = tulpa::compute_nmix_site(
            y_site.data(), eta_p_site.data(), J, eta_lambda(s), K_max, r
        );
        log_lik += res.log_lik;
        st.grad_eta_lam(s)    = res.grad_eta_lambda;
        st.info_eta_lam(s)    = res.info_eta_lambda;
        st.mean_N(s)          = res.mean_N;
        st.var_N(s)           = res.var_N;
        st.boundary_weight(s) = res.boundary_weight;
        st.score_wt_lambda(s) = res.score_wt_lambda;
        st.grad_theta(s)      = res.grad_theta;
        st.info_theta(s)      = res.info_theta;
        st.info_lam_theta(s)  = res.info_lambda_theta;
        st.cov_N_stheta(s)    = res.cov_N_stheta;
        st.var_stheta(s)      = res.var_stheta;
        for (int j = 0; j < J; ++j) {
            st.grad_eta_p(idx[j]) = res.grad_eta_p[j];
            st.info_eta_p(idx[j]) = res.info_eta_p[j];
        }
    }
    return log_lik;
}

// Cheap log-lik-only sweep for line-search trial points.
double kernel_log_lik_only(
    const std::vector<std::vector<int>>& obs_by_site,
    const Rcpp::IntegerVector& y_R,
    const VectorXd& eta_lambda,
    const VectorXd& eta_p_long,
    int K_max, double r
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
        tulpa::NMixSiteResult res = tulpa::compute_nmix_site(
            y_site.data(), eta_p_site.data(), J, eta_lambda(s), K_max, r
        );
        if (!R_finite(res.log_lik)) return res.log_lik;
        log_lik += res.log_lik;
    }
    return log_lik;
}

// p_ij from a logit linear predictor, stable for either sign.
inline double inv_logit(double e) {
    if (e > 0.0) return 1.0 / (1.0 + std::exp(-e));
    double ee = std::exp(e);
    return ee / (1.0 + ee);
}

// Fill the leading (p_lam+p_p) block of H (lower triangle) with the marginal
// observed Fisher information for beta = (beta_lambda, beta_p), given a sweep
// state. score_wt_lambda scales the lambda part of the Var[N|y] rank-1 vector
// (1 for Poisson, 1-q for NB). The caller symmetrises H once at the end.
void assemble_beta_obs_info(
    int p_lam, int p_p,
    const Map<MatrixXd>& Xl, const Map<MatrixXd>& Xp,
    const VectorXd& eta_p_long,
    const std::vector<std::vector<int>>& obs_by_site,
    const SweepState& st,
    MatrixXd& H
) {
    const int n_sites = (int)obs_by_site.size();
    const int pb = p_lam + p_p;
    for (int s = 0; s < n_sites; ++s) {
        const auto& idx = obs_by_site[s];
        const int J = (int)idx.size();
        if (J == 0) continue;
        double w_lam = st.info_eta_lam(s);
        if (w_lam > 0.0) {
            H.block(0, 0, p_lam, p_lam)
                .selfadjointView<Eigen::Lower>()
                .rankUpdate(Xl.row(s).transpose(), w_lam);
        }
        for (int j = 0; j < J; ++j) {
            double w_p = st.info_eta_p(idx[j]);
            if (w_p > 0.0) {
                H.block(p_lam, p_lam, p_p, p_p)
                    .selfadjointView<Eigen::Lower>()
                    .rankUpdate(Xp.row(idx[j]).transpose(), w_p);
            }
        }
        if (st.var_N(s) > 0.0) {
            VectorXd u(pb);
            u.segment(0, p_lam) = -st.score_wt_lambda(s) * Xl.row(s).transpose();
            VectorXd ssum = VectorXd::Zero(p_p);
            for (int j = 0; j < J; ++j) {
                ssum += inv_logit(eta_p_long(idx[j])) * Xp.row(idx[j]).transpose();
            }
            u.segment(p_lam, p_p) = ssum;
            H.block(0, 0, pb, pb)
                .selfadjointView<Eigen::Lower>()
                .rankUpdate(u, -st.var_N(s));
        }
    }
}

// Fill the leading (p_lam+p_p) block of H (lower triangle) with the complete-
// data Fisher information (drop the var_N rank-1 correction). Always PSD; the
// Levenberg-Marquardt fallback when the observed-info matrix is not PSD.
void assemble_beta_fisher(
    int p_lam, int p_p,
    const Map<MatrixXd>& Xl, const Map<MatrixXd>& Xp,
    const std::vector<std::vector<int>>& obs_by_site,
    const SweepState& st,
    MatrixXd& H
) {
    const int n_sites = (int)obs_by_site.size();
    const int n_obs   = (int)Xp.rows();
    for (int s = 0; s < n_sites; ++s) {
        if (st.info_eta_lam(s) > 0.0) {
            H.block(0, 0, p_lam, p_lam)
                .selfadjointView<Eigen::Lower>()
                .rankUpdate(Xl.row(s).transpose(), st.info_eta_lam(s));
        }
    }
    for (int o = 0; o < n_obs; ++o) {
        if (st.info_eta_p(o) > 0.0) {
            H.block(p_lam, p_lam, p_p, p_p)
                .selfadjointView<Eigen::Lower>()
                .rankUpdate(Xp.row(o).transpose(), st.info_eta_p(o));
        }
    }
}

// Inner Newton on beta = (beta_lambda, beta_p) at fixed r. Mutates beta_lam /
// beta_p in place from their warm start; fills `st` at the converged beta and
// reports the log-lik, gradient norm, and convergence flag.
void inner_newton_beta(
    const std::vector<std::vector<int>>& obs_by_site,
    const Rcpp::IntegerVector& y,
    const Map<MatrixXd>& Xl, const Map<MatrixXd>& Xp,
    int p_lam, int p_p, int K_max, double r,
    int max_iter, double tol, bool verbose,
    VectorXd& beta_lam, VectorXd& beta_p,
    SweepState& st,
    double& log_lik, double& grad_norm, bool& converged, int& n_iter
) {
    const int pb = p_lam + p_p;
    log_lik = R_NegInf;
    grad_norm = R_PosInf;
    converged = false;
    int iter = 0;
    for (iter = 0; iter < max_iter; ++iter) {
        VectorXd eta_lam    = Xl * beta_lam;
        VectorXd eta_p_long = Xp * beta_p;
        log_lik = kernel_sweep(obs_by_site, y, eta_lam, eta_p_long, K_max, r, st);

        VectorXd grad_beta_lam = Xl.transpose() * st.grad_eta_lam;
        VectorXd grad_beta_p   = Xp.transpose() * st.grad_eta_p;
        grad_norm = std::sqrt(grad_beta_lam.squaredNorm() + grad_beta_p.squaredNorm());
        if (verbose) {
            Rcpp::Rcout << "    [beta] iter " << iter << "  log_lik " << log_lik
                        << "  grad_norm " << grad_norm << "\n";
        }
        if (grad_norm < tol) { converged = true; break; }

        MatrixXd H = MatrixXd::Zero(pb, pb);
        assemble_beta_obs_info(p_lam, p_p, Xl, Xp, eta_p_long, obs_by_site, st, H);
        H = H.selfadjointView<Eigen::Lower>();

        VectorXd grad_total(pb);
        grad_total.segment(0, p_lam)   = grad_beta_lam;
        grad_total.segment(p_lam, p_p) = grad_beta_p;

        VectorXd delta;
        Eigen::LLT<MatrixXd> chol(H);
        if (chol.info() == Eigen::Success) {
            delta = chol.solve(grad_total);
        } else {
            MatrixXd Hf = MatrixXd::Zero(pb, pb);
            assemble_beta_fisher(p_lam, p_p, Xl, Xp, obs_by_site, st, Hf);
            Hf = Hf.selfadjointView<Eigen::Lower>();
            Eigen::LLT<MatrixXd> chol_f(Hf);
            if (chol_f.info() != Eigen::Success) {
                Rcpp::warning("Fallback Cholesky failure at beta iter %d -- aborting.", iter);
                break;
            }
            delta = chol_f.solve(grad_total);
            if (verbose) Rcpp::Rcout << "      (Fisher fallback)\n";
        }
        VectorXd delta_lam = delta.segment(0, p_lam);
        VectorXd delta_p   = delta.segment(p_lam, p_p);

        double step = 1.0;
        bool stepped = false;
        for (int h = 0; h < 12; ++h) {
            VectorXd bl = beta_lam + step * delta_lam;
            VectorXd bp = beta_p   + step * delta_p;
            double ll_try = kernel_log_lik_only(obs_by_site, y, Xl * bl, Xp * bp, K_max, r);
            if (R_finite(ll_try) && ll_try >= log_lik - 1e-10) {
                beta_lam = bl; beta_p = bp; stepped = true; break;
            }
            step *= 0.5;
        }
        if (!stepped) {
            Rcpp::warning("beta step halving exhausted at iter %d -- aborting.", iter);
            break;
        }
    }
    // Refresh state at the final beta.
    log_lik = kernel_sweep(obs_by_site, y, Xl * beta_lam, Xp * beta_p, K_max, r, st);
    n_iter = iter;
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
    bool verbose,
    bool nb,
    double log_r_init,
    double theta_max
) {
    const int n_sites = X_lambda_R.nrow();
    const int p_lam   = X_lambda_R.ncol();
    const int n_obs   = X_p_R.nrow();
    const int p_p     = X_p_R.ncol();
    if ((int)y.size() != n_obs) Rcpp::stop("y length must match nrow(X_p).");
    if ((int)site_idx.size() != n_obs) Rcpp::stop("site_idx length must match nrow(X_p).");
    if ((int)beta_lambda_init.size() != p_lam) Rcpp::stop("beta_lambda_init must have length ncol(X_lambda).");
    if ((int)beta_p_init.size() != p_p) Rcpp::stop("beta_p_init must have length ncol(X_p).");

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

    const int pb = p_lam + p_p;
    const int p_total = pb + (nb ? 1 : 0);
    const double theta_min = std::log(1e-4);

    SweepState st; st.resize(n_sites, n_obs);

    double theta = nb ? log_r_init : R_PosInf;
    if (nb) theta = std::min(std::max(theta, theta_min), theta_max);
    double r = nb ? std::exp(theta) : std::numeric_limits<double>::infinity();

    double log_lik = R_NegInf, beta_grad_norm = R_PosInf;
    bool beta_conv = false;
    int beta_iter = 0;

    double grad_theta = 0.0, info_theta_tot = 0.0;
    bool theta_converged = !nb;          // Poisson: no theta to converge
    bool dispersion_boundary = false;
    int outer_iter = 0;
    const int outer_max = nb ? max_iter : 1;

    for (outer_iter = 0; outer_iter < outer_max; ++outer_iter) {
        inner_newton_beta(obs_by_site, y, Xl, Xp, p_lam, p_p, K_max, r,
                          max_iter, tol, verbose,
                          beta_lam, beta_p, st,
                          log_lik, beta_grad_norm, beta_conv, beta_iter);
        if (!nb) break;

        // Outer profile-score step on theta = log r (beta held fixed for the
        // line search). grad_theta and the theta-theta observed info aggregate
        // the per-site kernel outputs at the current mode.
        grad_theta = st.grad_theta.sum();
        info_theta_tot = (st.info_theta - st.var_stheta).sum();
        if (verbose) {
            Rcpp::Rcout << "  [theta] outer " << outer_iter << "  log_r " << theta
                        << "  grad_theta " << grad_theta
                        << "  info_theta " << info_theta_tot << "\n";
        }
        if (std::abs(grad_theta) < tol && beta_conv) { theta_converged = true; break; }

        double dth = (info_theta_tot > 1e-8)
                       ? grad_theta / info_theta_tot
                       : (grad_theta > 0 ? 1.0 : -1.0) * 0.5;
        if (dth >  1.5) dth =  1.5;
        if (dth < -1.5) dth = -1.5;

        VectorXd eta_lam    = Xl * beta_lam;
        VectorXd eta_p_long = Xp * beta_p;
        double ll_cur = log_lik;
        double step = 1.0;
        bool stepped = false;
        for (int h = 0; h < 25; ++h) {
            double th_try = std::min(std::max(theta + step * dth, theta_min), theta_max);
            if (th_try == theta) break;                     // clamped, no movement
            double r_try = std::exp(th_try);
            double ll_try = kernel_log_lik_only(obs_by_site, y, eta_lam, eta_p_long, K_max, r_try);
            if (R_finite(ll_try) && ll_try >= ll_cur - 1e-10) {
                theta = th_try; r = r_try; stepped = true; break;
            }
            step *= 0.5;
        }
        if (!stepped) {
            // Cannot improve theta: either pinned at a bound (data want
            // -> Poisson) or already optimal. A boundary pin is a converged
            // constrained optimum, not a failure -- the profile score stays
            // large there by design.
            dispersion_boundary = (theta >= theta_max - 1e-6);
            theta_converged = dispersion_boundary || (std::abs(grad_theta) < 1e-2);
            break;
        }
    }

    // Final state and joint observed-information Hessian at the mode.
    VectorXd eta_lam_f    = Xl * beta_lam;
    VectorXd eta_p_long_f = Xp * beta_p;
    double log_lik_final = kernel_sweep(obs_by_site, y, eta_lam_f, eta_p_long_f, K_max, r, st);
    grad_theta = nb ? st.grad_theta.sum() : 0.0;

    MatrixXd H_obs = MatrixXd::Zero(p_total, p_total);
    assemble_beta_obs_info(p_lam, p_p, Xl, Xp, eta_p_long_f, obs_by_site, st, H_obs);
    if (nb) {
        // theta row/col (last coordinate). Written into the lower triangle.
        //   H[th, beta_lam] += (E[I_c,lam,th] - C*(1-q)) X_lam[s]
        //   H[th, beta_p]   += C * sum_j p_ij X_p[ij]
        //   H[th, th]       += info_theta(s) - var_stheta(s)
        const int th = pb;
        for (int s = 0; s < n_sites; ++s) {
            const auto& idx = obs_by_site[s];
            const int J = (int)idx.size();
            if (J == 0) continue;
            double w_lam_th = st.info_lam_theta(s) - st.cov_N_stheta(s) * st.score_wt_lambda(s);
            for (int k = 0; k < p_lam; ++k) {
                H_obs(th, k) += w_lam_th * Xl(s, k);
            }
            double C = st.cov_N_stheta(s);
            for (int j = 0; j < J; ++j) {
                double p_ij = inv_logit(eta_p_long_f(idx[j]));
                for (int k = 0; k < p_p; ++k) {
                    H_obs(th, p_lam + k) += C * p_ij * Xp(idx[j], k);
                }
            }
            H_obs(th, th) += st.info_theta(s) - st.var_stheta(s);
        }
    }
    H_obs = H_obs.selfadjointView<Eigen::Lower>();

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

    bool converged = beta_conv && theta_converged;
    // At a dispersion boundary the theta score is a constrained-active gradient,
    // not a convergence residual -- report the beta gradient norm only.
    double grad_norm_report = (nb && !dispersion_boundary)
        ? std::sqrt(beta_grad_norm * beta_grad_norm + grad_theta * grad_theta)
        : beta_grad_norm;

    return Rcpp::List::create(
        Rcpp::Named("beta_lambda")         = beta_lam_out,
        Rcpp::Named("beta_p")              = beta_p_out,
        Rcpp::Named("log_r")               = nb ? theta : NA_REAL,
        Rcpp::Named("r")                   = nb ? r : NA_REAL,
        Rcpp::Named("grad_theta")          = nb ? grad_theta : NA_REAL,
        Rcpp::Named("dispersion_boundary") = nb ? dispersion_boundary : false,
        Rcpp::Named("log_lik")             = log_lik_final,
        Rcpp::Named("vcov")                = Rcpp::wrap(vcov),
        Rcpp::Named("vcov_ok")             = vcov_ok,
        Rcpp::Named("H_obs")               = Rcpp::wrap(H_obs),
        Rcpp::Named("n_iter")              = outer_iter + (converged ? 1 : 0),
        Rcpp::Named("converged")           = converged,
        Rcpp::Named("grad_norm")           = grad_norm_report,
        Rcpp::Named("mean_N")              = Rcpp::wrap(st.mean_N),
        Rcpp::Named("var_N")               = Rcpp::wrap(st.var_N),
        Rcpp::Named("boundary_weight")     = Rcpp::wrap(st.boundary_weight)
    );
}
