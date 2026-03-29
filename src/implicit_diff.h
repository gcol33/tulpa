// implicit_diff.h
// Implicit differentiation of Laplace-marginalized log-likelihood
// w.r.t. hyperparameters, for NUTS sampling over theta.
//
// Given:
//   log p(y|theta) ≈ log p(y|x*,theta) + log p(x*|theta) - 0.5 log|H| + const
//
// Gradient (via implicit function theorem):
//   d/d(theta) log p(y|theta) = [partial terms] - 0.5 tr(H^{-1} dH/d(theta))
//
// The trace term uses selected inversion (Takahashi) to get H^{-1} at
// the nonzero positions of dH/d(theta) = dQ/d(theta).
//
// Reference: Margossian, Vehtari et al. (2023-2024)

#ifndef TULPA_IMPLICIT_DIFF_H
#define TULPA_IMPLICIT_DIFF_H

#include "sparse_cholesky.h"
#include "spde_qbuilder.h"
#include "laplace_helpers.h"
#include <Rcpp.h>
#include <vector>
#include <cmath>

namespace tulpa {

// Gradient of SPDE Laplace log-marginal w.r.t. (log_range, log_sigma).
// Requires: mode x* (from inner Laplace), Q builder, and the H factorization.
//
// Returns: gradient vector of length 2 (d/d(log_range), d/d(log_sigma))
struct ImplicitDiffResult {
    double grad_log_range;
    double grad_log_sigma;
    double log_marginal;
};

inline ImplicitDiffResult spde_implicit_gradient(
    const Rcpp::NumericVector& mode,
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X,
    int N, int p, int n_mesh, int mesh_start,
    const ARows& a_rows,
    const SpdeQBuilder& qb,
    double range, double sigma_spde, double nu,
    const Rcpp::NumericVector& C0_diag,
    const Rcpp::NumericVector& G1_x,
    const Rcpp::IntegerVector& G1_i,
    const Rcpp::IntegerVector& G1_p,
    const std::string& family, double phi,
    SparseCholeskySolver& solver
) {
    ImplicitDiffResult res;
    double kappa = std::sqrt(8.0 * nu) / range;
    double tau = 1.0 / (std::sqrt(4.0 * M_PI) * kappa * sigma_spde);

    // --- Term 1: d/d(theta) of log p(x*|theta) = d/d(theta) of -0.5 x*'Q(theta)x* ---
    // Q depends on theta through kappa and tau.
    // d(kappa)/d(log_range) = -kappa (since kappa = c/range, d(log_range) = d(range)/range)
    // d(tau)/d(log_sigma) = -tau (since tau = c/(kappa*sigma), d(log_sigma) = d(sigma)/sigma)
    // d(tau)/d(log_range) = -tau (kappa appears in tau)

    // Compute x*'Qx* and x*'(dQ/d_theta)x*
    // dQ/d(log_range) = Q_kappa * d(kappa)/d(log_range) where Q_kappa = dQ/d(kappa)
    // dQ/d(log_sigma) = Q_tau * d(tau)/d(log_sigma) where Q_tau = dQ/d(tau)

    // For alpha=2: Q = tau^2 * (kappa^4*C + 2*kappa^2*G + GC^{-1}G)
    // dQ/d(log_kappa) = tau^2 * (4*kappa^4*C + 4*kappa^2*G) = 2*tau^2*(2*kappa^4*C + 2*kappa^2*G)
    // dQ/d(log_tau) = 2*Q (since Q is proportional to tau^2)

    // Simplify: d(log p)/d(log_range) involves d(kappa)/d(log_range) = -kappa
    // and d(log p)/d(log_sigma) involves d(tau)/d(log_sigma) = -tau

    // Build dQ/d(log_range) and dQ/d(log_sigma) using the Q builder's per-entry contributions
    double k2 = kappa * kappa;
    double k4 = k2 * k2;
    double tau2 = tau * tau;

    // x*'Q x* (quadratic form for current Q)
    double xQx = 0.0;
    // x*'(dQ/d_log_range) x*
    double xdQdr_x = 0.0;
    // x*'(dQ/d_log_sigma) x*
    double xdQds_x = 0.0;

    int nnz_q = qb.nnz();
    for (int col = 0; col < n_mesh; col++) {
        for (int qidx = qb.Q_p[col]; qidx < qb.Q_p[col + 1]; qidx++) {
            int row = qb.Q_i[qidx];
            double wi = mode[mesh_start + row];
            double wj = mode[mesh_start + col];
            double wiwj = wi * wj;

            // Current Q value
            double q = qb.Q_x[qidx];
            xQx += wiwj * q;

            // dQ/d(log_kappa): derivative of each term w.r.t. log(kappa)
            // d/d(log_kappa) [tau^2*(k4*c0 + 2*k2*g1 + gdg)]
            // = tau^2 * (4*k4*c0 + 4*k2*g1)
            double dq_dlogk = tau2 * (4.0 * k4 * qb.c0_contrib[qidx] +
                                       4.0 * k2 * qb.g1_contrib[qidx]);
            // d(log_kappa)/d(log_range) = -1
            // So dQ/d(log_range) = -dQ/d(log_kappa)
            xdQdr_x -= wiwj * dq_dlogk;

            // dQ/d(log_tau) = 2*Q (tau^2 → 2*tau^2)
            // d(log_tau)/d(log_sigma) involves the chain rule:
            // tau = 1/(sqrt(4pi)*kappa*sigma), so d(log_tau)/d(log_sigma) = -1
            // Also d(log_tau)/d(log_range) = -1 (via kappa)
            // So dQ/d(log_sigma) = -2*Q
            xdQds_x -= 2.0 * wiwj * q;

            // Range also affects tau: d(log_tau)/d(log_range) = -1
            // So dQ/d(log_range) += -2*Q (from tau channel)
            xdQdr_x -= 2.0 * wiwj * q;
        }
    }

    // log p(x*|theta) = -0.5 * xQx (ignoring normalization for now)
    double d_logprior_d_logrange = -0.5 * xdQdr_x;
    double d_logprior_d_logsigma = -0.5 * xdQds_x;

    // --- Term 2: -0.5 tr(H^{-1} dH/d(theta)) ---
    // dH/d(theta) = dQ/d(theta) (likelihood part doesn't depend on theta)
    // tr(H^{-1} dQ/d(theta)) = sum_{(i,j) in Q} H^{-1}_{ij} * dQ_{ij}/d(theta)

    // Get H^{-1} diagonal (and selected off-diagonal) via Takahashi
    std::vector<double> H_inv_diag = solver.selected_inversion_diagonal();

    // For the trace, we need H^{-1}_{ij} at Q's nonzero positions.
    // selected_inversion_diagonal only gives the diagonal.
    // For a full implicit diff, we'd need the full selected inversion (off-diagonal too).
    // For now, use the diagonal approximation:
    // tr(H^{-1} dQ/d(theta)) ≈ sum_i H^{-1}_{ii} * dQ_{ii}/d(theta)
    // This is exact when Q and dQ are diagonal, approximate otherwise.

    double trace_range = 0.0;
    double trace_sigma = 0.0;
    for (int col = 0; col < n_mesh; col++) {
        for (int qidx = qb.Q_p[col]; qidx < qb.Q_p[col + 1]; qidx++) {
            int row = qb.Q_i[qidx];
            if (row != col) continue;  // diagonal only for now

            int orig_row = row;  // already in original ordering from Takahashi
            if (orig_row < 0 || orig_row >= (int)H_inv_diag.size()) continue;

            double h_inv = H_inv_diag[mesh_start + orig_row];

            double dq_dlogk = tau2 * (4.0 * k4 * qb.c0_contrib[qidx] +
                                       4.0 * k2 * qb.g1_contrib[qidx]);
            // dQ/d(log_range) = -(dQ/d(log_kappa)) - 2Q
            trace_range += h_inv * (-dq_dlogk - 2.0 * qb.Q_x[qidx]);
            // dQ/d(log_sigma) = -2Q
            trace_sigma += h_inv * (-2.0 * qb.Q_x[qidx]);
        }
    }

    res.grad_log_range = d_logprior_d_logrange - 0.5 * trace_range;
    res.grad_log_sigma = d_logprior_d_logsigma - 0.5 * trace_sigma;

    return res;
}

} // namespace tulpa

#endif // TULPA_IMPLICIT_DIFF_H
