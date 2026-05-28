// nmix_community_em.cpp
// Community / multispecies N-mixture (spAbundance msNMix) by Laplace-EM -- the
// fast n_quad = 1 optimize path. This is the joint-Laplace stationary point the
// shared AGHQ engine targets (agq_plan.md Section 4.3: the EM Sigma M-step is the
// Sigma stationary point), reached by a block-coordinate Newton + closed-form EM
// covariance update instead of a finite-difference gradient over the joint
// (theta, log-Cholesky Sigma) parameter vector. The FD-gradient BFGS in
// tulpa_re_aghq re-runs every per-species mode-find inside each of hundreds of
// objective / gradient-perturbation evaluations; the EM converges in a handful of
// cheap iterations, so it restores the bespoke fitter's speed.
//
// Single marginal source: the per-species value / score / observed-information /
// complete-data Fisher all come from NMixCommunityOracle::eval_species (the same
// nmix_kernel.h per-site marginal the n_quad > 1 quadrature path uses), so there
// is no second copy of the N-mixture math.
//
//   E-step  joint mode of (mu, {b_s}) by block-coordinate Newton; the mode-find
//           curvature is the complete-data Fisher (PSD, EM-rate). Species are
//           conditionally independent given mu.
//   M-step  Sigma_k = mean_s [ b_k_s b_k_s' + Cov(b_k_s | y) ],
//           Cov(b_s | y) = (Fisher_s + Sigma^{-1})^{-1}.
//   SEs     marginal information for mu = Schur complement of the b-block in the
//           joint OBSERVED-information Hessian (the Var[N|y] rank-1 coupling
//           lives in the observed info; Louis 1982).
//
// Poisson only (the oracle's NB path is a follow-up).

#include "nmix_community_oracle.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include <limits>
#include <vector>

using namespace Rcpp;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

// Symmetric inverse of a small PD matrix via LLT, with a tiny ridge fallback.
MatrixXd safe_inverse(const MatrixXd& M) {
    const int d = (int)M.rows();
    Eigen::LLT<MatrixXd> llt(M);
    if (llt.info() == Eigen::Success)
        return llt.solve(MatrixXd::Identity(d, d));
    MatrixXd Mr = M;
    double md = M.diagonal().cwiseAbs().mean();
    if (!(md > 0)) md = 1.0;
    Mr.diagonal().array() += 1e-8 * md;
    return Mr.ldlt().solve(MatrixXd::Identity(d, d));
}

double logdet_spd(const MatrixXd& M) {
    Eigen::LLT<MatrixXd> llt(M);
    if (llt.info() != Eigen::Success) return std::numeric_limits<double>::quiet_NaN();
    double ld = 0.0;
    const MatrixXd& L = llt.matrixL();
    for (int i = 0; i < L.rows(); ++i) ld += std::log(L(i, i));
    return 2.0 * ld;
}

}  // namespace

// Community N-mixture Laplace-EM optimize driver (the n_quad = 1 default). It
// consumes the SAME native oracle (NMixCommunityOracle) the joint optimizer
// drives -- the per-species marginal / score / Fisher / observed-info all come
// from orc.eval_species, so there is one marginal source and the EM is purely an
// alternative outer solver. `oracle` is the external pointer from
// cpp_nmix_community_oracle(); `mu_init` is the stacked (mu_lambda, mu_p) warm
// start. Returns the contract tulpa_nmix_laplace_re packages.
// [[Rcpp::export]]
List cpp_nmix_community_em(SEXP oracle, NumericVector mu_init,
                          NumericMatrix Sigma_lambda_init, NumericMatrix Sigma_p_init,
                          int max_iter = 100, double tol = 1e-6,
                          int inner_max = 50, double inner_tol = 1e-8,
                          double sigma_beta = 100.0, bool verbose = false) {
    Rcpp::XPtr<tulpa::REGroupOracle> base(oracle);
    tulpa::NMixCommunityOracle* orcp =
        dynamic_cast<tulpa::NMixCommunityOracle*>(base.get());
    if (orcp == nullptr)
        Rcpp::stop("cpp_nmix_community_em: `oracle` is not an NMixCommunityOracle.");
    tulpa::NMixCommunityOracle& orc = *orcp;

    const int p_lam = orc.p_lam;
    const int p_p   = orc.p_p;
    const int d     = orc.d;
    const int S     = orc.n_groups;
    const double tau_beta = 1.0 / (sigma_beta * sigma_beta);

    VectorXd mu(d);
    for (int j = 0; j < d; ++j) mu[j] = mu_init[j];
    MatrixXd Sig_l = Eigen::Map<MatrixXd>(Sigma_lambda_init.begin(), p_lam, p_lam);
    MatrixXd Sig_p = Eigen::Map<MatrixXd>(Sigma_p_init.begin(), p_p, p_p);

    std::vector<VectorXd> b(S, VectorXd::Zero(d));

    auto block_prec = [&](const MatrixXd& Sl, const MatrixXd& Sp) {
        MatrixXd P = MatrixXd::Zero(d, d);
        P.topLeftCorner(p_lam, p_lam) = safe_inverse(Sl);
        P.bottomRightCorner(p_p, p_p) = safe_inverse(Sp);
        return P;
    };

    bool converged = false;
    int n_iter = 0;
    std::vector<MatrixXd> fisher_s(S, MatrixXd::Zero(d, d));   // last-inner per-species Fisher

    for (int it = 0; it < max_iter; ++it) {
        n_iter = it + 1;
        const MatrixXd P = block_prec(Sig_l, Sig_p);

        // ---- E-step: joint mode of (mu, {b_s}) by block-coordinate Newton ----
        for (int inner = 0; inner < inner_max; ++inner) {
            orc.rebind(mu.data());
            double max_step = 0.0;
            // (a) per-species b update (each species independent given mu)
            for (int s = 0; s < S; ++s) {
                const tulpa::NMixCommunityOracle::SpeciesEval e =
                    orc.eval_species(s, b[s].data(), /*want_negH=*/false, /*want_fisher=*/true);
                const VectorXd g    = e.grad - P * b[s];
                const MatrixXd Hb   = e.fisher + P;
                const VectorXd step = safe_inverse(Hb) * g;
                b[s] += step;
                max_step = std::max(max_step, step.cwiseAbs().maxCoeff());
            }
            // (b) mu update (pooled over species at current b)
            VectorXd g_mu = VectorXd::Zero(d);
            MatrixXd H_mu = MatrixXd::Zero(d, d);
            for (int s = 0; s < S; ++s) {
                const tulpa::NMixCommunityOracle::SpeciesEval e =
                    orc.eval_species(s, b[s].data(), /*want_negH=*/false, /*want_fisher=*/true);
                fisher_s[s] = e.fisher;
                g_mu += e.grad;
                H_mu += e.fisher;
            }
            g_mu -= tau_beta * mu;
            H_mu.diagonal().array() += tau_beta;
            const VectorXd step_mu = safe_inverse(H_mu) * g_mu;
            mu += step_mu;
            max_step = std::max(max_step, step_mu.cwiseAbs().maxCoeff());
            if (max_step < inner_tol) break;
        }

        // ---- M-step: EM update of the community covariances ----
        const MatrixXd Pm = block_prec(Sig_l, Sig_p);
        MatrixXd Sl_new = MatrixXd::Zero(p_lam, p_lam);
        MatrixXd Sp_new = MatrixXd::Zero(p_p, p_p);
        for (int s = 0; s < S; ++s) {
            const MatrixXd cov_s = safe_inverse(fisher_s[s] + Pm);   // Cov(b_s | y)
            const VectorXd bl = b[s].head(p_lam);
            const VectorXd bp = b[s].tail(p_p);
            Sl_new += bl * bl.transpose() + cov_s.topLeftCorner(p_lam, p_lam);
            Sp_new += bp * bp.transpose() + cov_s.bottomRightCorner(p_p, p_p);
        }
        Sl_new /= (double)S;
        Sp_new /= (double)S;

        const double dSig = std::max((Sl_new - Sig_l).cwiseAbs().maxCoeff(),
                                     (Sp_new - Sig_p).cwiseAbs().maxCoeff());
        Sig_l = Sl_new;
        Sig_p = Sp_new;
        if (verbose) Rcpp::Rcout << "iter " << n_iter << "  dSigma=" << dSig << "\n";
        if (dSig < tol) { converged = true; break; }
    }

    // ---- final pass: observed-info marginal Hessian for mu, Laplace logLik ----
    const MatrixXd P = block_prec(Sig_l, Sig_p);
    const double logdetP = logdet_spd(P);
    MatrixXd I_mu = MatrixXd::Zero(d, d);
    I_mu.diagonal().array() += tau_beta;
    double loglik_marg = 0.0;
    MatrixXd blup_l(S, p_lam), blup_p(S, p_p);
    orc.rebind(mu.data());
    for (int s = 0; s < S; ++s) {
        const tulpa::NMixCommunityOracle::SpeciesEval e =
            orc.eval_species(s, b[s].data(), /*want_negH=*/true, /*want_fisher=*/false);
        const MatrixXd& A = e.negH;            // observed-info coefficient curvature
        const MatrixXd Bb = A + P;
        const MatrixXd Bbinv = safe_inverse(Bb);
        // Schur complement A - A Bbinv A, via explicit intermediates (a fused
        // triple-product expression template instantiates pathologically under
        // -O2 on MinGW g++).
        const MatrixXd ABbinv = A * Bbinv;
        I_mu += A;
        I_mu.noalias() -= ABbinv * A;
        const double ldH = logdet_spd(Bb);
        loglik_marg += e.logL - 0.5 * b[s].dot(P * b[s])
                       + 0.5 * logdetP - 0.5 * ldH;
        blup_l.row(s) = b[s].head(p_lam).transpose();
        blup_p.row(s) = b[s].tail(p_p).transpose();
    }
    const MatrixXd vcov = safe_inverse(I_mu);

    NumericVector mu_lambda(p_lam), mu_p(p_p);
    for (int j = 0; j < p_lam; ++j) mu_lambda[j] = mu[j];
    for (int j = 0; j < p_p; ++j)   mu_p[j] = mu[p_lam + j];

    NumericMatrix vcov_out(d, d);
    for (int i = 0; i < d; ++i) for (int j = 0; j < d; ++j) vcov_out(i, j) = vcov(i, j);
    NumericMatrix Sl_out(p_lam, p_lam), Sp_out(p_p, p_p);
    for (int i = 0; i < p_lam; ++i) for (int j = 0; j < p_lam; ++j) Sl_out(i, j) = Sig_l(i, j);
    for (int i = 0; i < p_p; ++i) for (int j = 0; j < p_p; ++j) Sp_out(i, j) = Sig_p(i, j);
    NumericMatrix bl_out(S, p_lam), bp_out(S, p_p);
    for (int s = 0; s < S; ++s) {
        for (int j = 0; j < p_lam; ++j) bl_out(s, j) = blup_l(s, j);
        for (int j = 0; j < p_p; ++j)   bp_out(s, j) = blup_p(s, j);
    }

    return List::create(
        _["mu_lambda"]    = mu_lambda,
        _["mu_p"]         = mu_p,
        _["vcov"]         = vcov_out,
        _["Sigma_lambda"] = Sl_out,
        _["Sigma_p"]      = Sp_out,
        _["b_lambda"]     = bl_out,
        _["b_p"]          = bp_out,
        _["log_lik"]      = loglik_marg,
        _["converged"]    = converged,
        _["n_iter"]       = n_iter);
}
