// spde_fractional_marginal.cpp
// Numerically stable Laplace log-marginal for a fractional rSPDE at a fixed
// (range, sigma). The C++ counterpart of the former R implementation in
// .spde_nested_logmarginal_at / .spde_family_wll (gcol33/tulpa, "no modelling
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
    double                                       tau_beta = 1e-4
) {
    const int n     = static_cast<int>(y.size());
    const int n_sub = static_cast<int>(C0sub.size());

    VectorXd off = VectorXd::Zero(n);
    if (offset_nullable.isNotNull()) {
        Rcpp::NumericVector o(offset_nullable);
        for (int i = 0; i < n; ++i) off[i] = o[i];
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

    // Non-gaussian: GLM working weights w and loglik at the mode's eta.
    VectorXd eta = X * beta_hat + A_eff * x_hat + off;
    VectorXd w(n);
    double loglik = 0.0;
    if (family == "poisson") {
        for (int i = 0; i < n; ++i) {
            const double lam = std::exp(eta[i]);
            w[i] = lam;
            loglik += y[i] * eta[i] - lam - std::lgamma(y[i] + 1.0);
        }
    } else if (family == "binomial") {
        for (int i = 0; i < n; ++i) {
            const double pi_ = 1.0 / (1.0 + std::exp(-eta[i]));
            const double nt  = static_cast<double>(n_trials[i]);
            w[i] = nt * pi_ * (1.0 - pi_);
            loglik += R::lchoose(nt, y[i]) + y[i] * std::log(pi_)
                    + (nt - y[i]) * std::log1p(-pi_);
        }
    } else {
        Rcpp::stop("Fractional-nu nested SPDE integration supports family in "
                   "{gaussian, poisson, binomial}; got '%s'.", family);
    }

    // x'Qx via the Pl matvec: ||C^{-1/2} Pl x||^2 + tau_beta ||beta||^2.
    double quad = tau_beta * beta_hat.squaredNorm();
    VectorXd Plx = Pl * x_hat;                                // n_sub
    for (int j = 0; j < n_sub; ++j) quad += Plx[j] * Plx[j] / C0sub[j];

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
