// spde_fractional_marginal.cpp
// Numerically stable Laplace log-marginal for a fractional rSPDE at a fixed
// (range, sigma). The C++ counterpart of the former R implementation in
// .spde_nested_logmarginal_at / .spde_family_wll ("no modelling
// math in R").
//
// The precision-space Laplace marginal needs 0.5(log|Q| - log|H|), but the
// rational precision Q = Pl' C^-1 Pl squares cond(Pl) (cond(Q) ~ 1e13+), so the
// two determinants computed directly lose digits in a range-dependent way and
// range stops being identifiable. This forms the SAME marginal through the
// matrix-determinant lemma on the obs-space matrix
//   B = (A_eff Pl^-1) C (A_eff Pl^-1)' + X X' / tau_beta      (n_obs x n_obs),
// well-conditioned because the cross term is built through the operator factor
// Pl (cond = sqrt(cond(Q))), never an explicit Q^-1. Then:
//   log|H| - log|Q| = log|I + W^{1/2} B W^{1/2}|       (det-lemma, non-gaussian)
//   marginal        = loglik(eta_hat) - 0.5 x'Qx - 0.5 (log|H| - log|Q|),
//     x'Qx = ||C^{-1/2} Pl x||^2 + tau_beta ||beta||^2  (Pl matvec, no inverse).
// Gaussian is the exact conjugate marginal y ~ N(off, B + phi^2 I); phi is the
// residual SD (variance phi^2), matching the engine family convention and the
// integer path (laplace_family_link.h).
//
// The mode (beta_hat, x_hat) -- x is the auxiliary weight, field u = Pr x -- comes
// from cpp_laplace_fit_spde_precomputed; only the marginal is computed here.

#include <RcppEigen.h>
#include "laplace_likelihoods.h"
#include "linalg_fast.h"
#include "spde_zero_mass.h"
#include <string>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseMatrix;

// [[Rcpp::export]]
double cpp_spde_fractional_logmarginal(
    const Eigen::Map<Eigen::VectorXd>            y,
    const Eigen::Map<Eigen::MatrixXd>            X,
    const Eigen::Map<Eigen::SparseMatrix<double>> A_eff,   // n_obs x n_sub
    const Eigen::Map<Eigen::SparseMatrix<double>> Pl,      // n_sub x n_sub
    const Eigen::Map<Eigen::VectorXd>            C0sub,    // n_sub
    std::string                                  family,
    double                                       phi,
    const Eigen::Map<Eigen::VectorXd>            beta_hat, // p   (non-gaussian)
    const Eigen::Map<Eigen::VectorXd>            x_hat,    // n_sub (non-gaussian)
    Rcpp::IntegerVector                          n_trials,
    Rcpp::Nullable<Rcpp::NumericVector>          offset_nullable = R_NilValue,
    double                                       tau_beta = 1e-4,
    Rcpp::Nullable<Rcpp::NumericVector>          weights_nullable = R_NilValue
) {
    const int n     = static_cast<int>(y.size());
    const int n_sub = static_cast<int>(C0sub.size());
    const int p     = static_cast<int>(X.cols());

    if (static_cast<int>(X.rows()) != n) {
        Rcpp::stop("nrow(X) (%d) must equal the number of observations (%d).",
                   static_cast<int>(X.rows()), n);
    }
    if (static_cast<int>(A_eff.rows()) != n ||
        static_cast<int>(A_eff.cols()) != n_sub) {
        Rcpp::stop("A_eff must be %d x %d; got %d x %d.",
                   n, n_sub, static_cast<int>(A_eff.rows()),
                   static_cast<int>(A_eff.cols()));
    }
    if (static_cast<int>(Pl.rows()) != n_sub ||
        static_cast<int>(Pl.cols()) != n_sub) {
        Rcpp::stop("Pl must be %d x %d; got %d x %d.",
                   n_sub, n_sub, static_cast<int>(Pl.rows()),
                   static_cast<int>(Pl.cols()));
    }
    if (family == "poisson" || family == "binomial") {
        if (static_cast<int>(beta_hat.size()) != p) {
            Rcpp::stop("length(beta_hat) (%d) must equal ncol(X) (%d).",
                       static_cast<int>(beta_hat.size()), p);
        }
        if (static_cast<int>(x_hat.size()) != n_sub) {
            Rcpp::stop("length(x_hat) (%d) must equal length(C0sub) (%d).",
                       static_cast<int>(x_hat.size()), n_sub);
        }
    }

    VectorXd off = VectorXd::Zero(n);
    if (offset_nullable.isNotNull()) {
        Rcpp::NumericVector o(offset_nullable);
        if (static_cast<int>(o.size()) != n) {
            Rcpp::stop("offset length (%d) must equal the number of "
                       "observations (%d).", static_cast<int>(o.size()), n);
        }
        for (int i = 0; i < n; ++i) off[i] = o[i];
    }

    // Per-observation likelihood weight. Absent -> the unweighted branches
    // below run byte-identically to before the weight channel existed.
    const bool weighted = weights_nullable.isNotNull();
    VectorXd wt = VectorXd::Ones(n);
    if (weighted) {
        Rcpp::NumericVector wv(weights_nullable);
        if (static_cast<int>(wv.size()) != n) {
            Rcpp::stop("weights length (%d) must equal the number of "
                       "observations (%d).", static_cast<int>(wv.size()), n);
        }
        for (int i = 0; i < n; ++i) wt[i] = wv[i];
    }

    // Mt = Pl^{-1} A_eff'  (n_sub x n) via the operator factor's LU (general
    // sparse; Pl is the rational product factor, not triangular).
    SparseMatrix<double> PlS = Pl;
    PlS.makeCompressed();
    Eigen::SparseLU<SparseMatrix<double>> lu;
    lu.compute(PlS);
    if (lu.info() != Eigen::Success) return R_NegInf;
    MatrixXd Mt = lu.solve(MatrixXd(A_eff.transpose()));   // n_sub x n
    if (lu.info() != Eigen::Success) return R_NegInf;

    // B = Mt' diag(C0sub) Mt + X X' / tau_beta, symmetrized.
    MatrixXd B = Mt.transpose() * (C0sub.asDiagonal() * Mt);
    B.noalias() += (X * X.transpose()) / tau_beta;
    B = 0.5 * (B + B.transpose());

    if (family == "gaussian") {
        if (!weighted) {
            // Exact conjugate marginal: y - off ~ N(0, B + phi^2 I).
            MatrixXd V = B;
            V.diagonal().array() += phi * phi;
            Eigen::LLT<MatrixXd> llt(V);
            if (llt.info() != Eigen::Success) return R_NegInf;
            const double half_logdetV =
                llt.matrixLLT().diagonal().array().log().sum();   // 0.5 log|V|
            VectorXd r = y - off;
            VectorXd z = llt.matrixL().solve(r);                  // L z = r
            return -half_logdetV - 0.5 * z.squaredNorm();
        }
        // Weighted conjugate marginal, in the form that survives a zero weight.
        // With K = diag(w_i) / phi^2 the exponent of the weighted density is
        // -0.5 (y - eta)' K (y - eta), so
        //   int exp(...) N(eta; off, B) deta
        //     = |I + K B|^{-1/2} exp(-0.5 r' (I + K B)^{-1} K r),
        // which is finite at w_i = 0 where the equivalent scaled-variance form
        // (B + phi^2 W^{-1}) is not. The normalizer of the weighted density is
        // -0.5 sum_i w_i log(2 pi phi^2); the 2 pi part is dropped to keep the
        // unweighted branch's convention (it omits -0.5 n log(2 pi)), leaving
        // -(sum_i w_i) log(phi), which reduces to the branch above at w == 1.
        const double inv_phi2 = 1.0 / (phi * phi);
        VectorXd k = wt * inv_phi2;
        MatrixXd G = k.asDiagonal() * B;
        G.diagonal().array() += 1.0;
        Eigen::PartialPivLU<MatrixXd> lu_G(G);
        const double logdet_G =
            lu_G.matrixLU().diagonal().array().abs().log().sum();
        VectorXd r = y - off;
        VectorXd quad_v = lu_G.solve(k.asDiagonal() * r);
        return -0.5 * logdet_G - 0.5 * r.dot(quad_v) - wt.sum() * std::log(phi);
    }

    // Non-gaussian: GLM working weights w and loglik at the mode's eta. A
    // per-observation weight scales this row's log-density and its Fisher
    // information alike, matching builtin_family_ll_double /
    // builtin_family_eta_weights -- so the marginal describes the same weighted
    // model the mode was found under.
    VectorXd eta = X * beta_hat + A_eff * x_hat + off;
    VectorXd w(n);
    double loglik = 0.0;
    if (family == "poisson") {
        for (int i = 0; i < n; ++i) {
            const double lam = tulpa_linalg::safe_exp(eta[i]);
            w[i] = wt[i] * lam;
            loglik += wt[i] * (y[i] * eta[i] - lam - std::lgamma(y[i] + 1.0));
        }
    } else if (family == "binomial") {
        if (static_cast<int>(n_trials.size()) != n) {
            Rcpp::stop("n_trials length (%d) must equal the number of "
                       "observations (%d).",
                       static_cast<int>(n_trials.size()), n);
        }
        // The engine's kernel and working weight, so the marginal and the mode
        // (found by cpp_laplace_fit_spde_precomputed) are the same density. A
        // materialized probability rounds to 0 or 1 near |eta| ~ 40 and turns
        // the whole cell into -Inf on a representation choice.
        for (int i = 0; i < n; ++i) {
            const int nt_i = n_trials[i];
            const int y_i  = static_cast<int>(y[i]);
            w[i] = wt[i] * tulpa::neg_hess_log_lik_binomial(y_i, nt_i, eta[i]);
            loglik += wt[i] * (R::lchoose(static_cast<double>(nt_i), y[i])
                             + tulpa::log_lik_binomial_kernel(y_i, nt_i, eta[i]));
        }
    } else {
        Rcpp::stop("Fractional-nu nested SPDE integration supports family in "
                   "{gaussian, poisson, binomial}; got '%s'.", family);
    }

    // x'Qx via the Pl matvec: ||C^{-1/2} Pl x||^2 + tau_beta ||beta||^2.
    // A zero-mass mesh node contributes nothing, matching the c0_inv = 0 floor
    // every other SPDE assembly applies (spde_zero_mass.h). Dividing by it
    // outright makes the whole cell NaN, which the R tryCatch wrappers do not
    // catch.
    double quad = tau_beta * beta_hat.squaredNorm();
    VectorXd Plx = Pl * x_hat;                                // n_sub
    for (int j = 0; j < n_sub; ++j) {
        const double c0_inv = tulpa::spde_c0_inv(C0sub[j]);
        quad += Plx[j] * Plx[j] * c0_inv;
    }

    // log|H| - log|Q| = log|I + W^{1/2} B W^{1/2}| (det-lemma).
    VectorXd sw = w.array().sqrt();
    MatrixXd Gm = sw.asDiagonal() * B * sw.asDiagonal();
    Gm.diagonal().array() += 1.0;
    Eigen::LLT<MatrixXd> llt(Gm);
    if (llt.info() != Eigen::Success) return R_NegInf;
    const double logdet_term =
        2.0 * llt.matrixLLT().diagonal().array().log().sum();

    return loglik - 0.5 * quad - 0.5 * logdet_term;
}
