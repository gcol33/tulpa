// implicit_diff.h
// Implicit differentiation of Laplace-marginalized log-likelihood
// w.r.t. hyperparameters, for NUTS sampling over theta.
//
// Given:
//   log p(y|theta) ~ log p(y|x*,theta) + log p(x*|theta) - 0.5 log|H| + const
//
// Gradient (via implicit function theorem):
//   d/d(theta) log p(y|theta) = [partial terms] - 0.5 tr(H^{-1} dH/d(theta))
//
// The trace terms use selected inversion (Takahashi) to get H^{-1} and Q^{-1}
// at the nonzero positions of dQ/d(theta).
//
// Reference: Margossian, Vehtari et al. (2023-2024)

#ifndef TULPA_IMPLICIT_DIFF_H
#define TULPA_IMPLICIT_DIFF_H

#include "sparse_cholesky.h"
#include "spde_logdet.h"
#include "spde_qbuilder.h"
#include <Rcpp.h>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace tulpa {

// Gradient of SPDE Laplace log-marginal w.r.t. (log_range, log_sigma).
// Requires: mode x* (from inner Laplace), Q builder, and the log-determinant
// helper holding the factor of Q(theta) the value used.
//
// Returns: gradient vector of length 2 (d/d(log_range), d/d(log_sigma)).
// `ok` is false when the posterior Hessian at (range, sigma) could not be
// factorized; the gradients are then NaN.
struct ImplicitDiffResult {
    double grad_log_range;
    double grad_log_sigma;
    bool   ok;
};

// `mode` is `std::vector<double>` to match LaplaceResult::mode directly
// (laplace_core.h: that type was chosen so the inner solver can populate
// the mode inside OpenMP parallel regions, where Rf_allocVector is not
// thread-safe). Taking the same type here avoids a wrap copy at the call
// site and makes the function safe to invoke from those regions later.
inline ImplicitDiffResult spde_implicit_gradient(
    const std::vector<double>& mode,
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X,
    int N, int p, int n_mesh, int mesh_start,
    const ARows& a_rows,
    const SpdeQBuilder& qb,
    SpdeQLogDet& qld,
    double range, double sigma_spde, double nu,
    const std::string& family, double phi,
    SparseCholeskySolver& solver,
    const double* offset = nullptr,
    const Rcpp::NumericVector& re_idx = Rcpp::NumericVector(),
    int n_re_groups = 0, double sigma_re = 1.0
) {
    ImplicitDiffResult res;
    res.grad_log_range = std::numeric_limits<double>::quiet_NaN();
    res.grad_log_sigma = std::numeric_limits<double>::quiet_NaN();
    res.ok             = false;

    // The dQ/dtheta closed forms below differentiate the alpha = 2 assembly
    // term by term, and read d(log_tau)/d(log_kappa) = -nu at nu = 1.
    if (std::abs(nu - 1.0) > 1e-10) {
        Rcpp::stop("spde_implicit_gradient implements the alpha = 2 (nu = 1) "
                   "operator; got nu = %g.", nu);
    }

    double kappa, tau;
    std::tie(kappa, tau) = spde_range_sigma_to_kappa_tau(range, sigma_spde, nu);

    // --- Term 1: d/d(theta) of log p(x*|theta) ---
    //   log p(x*|theta) = 0.5 log|Q(theta)| - 0.5 x*' Q(theta) x*
    // Q depends on theta through kappa and tau:
    //   kappa = sqrt(8 nu) / range         => d(log_kappa)/d(log_range) = -1
    //   tau   = 1/(sqrt(4 pi nu) kappa^nu sigma)
    //                                      => d(log_tau)/d(log_sigma)   = -1
    //                                         d(log_tau)/d(log_range)   = +nu
    //
    // For alpha = 2: Q = tau^2 (kappa^4 C + 2 kappa^2 G + G C^-1 G) + R, with
    // R the theta-independent orphan ridge (spde_zero_mass.h). Only the
    // assembled part carries the tau^2, so
    //   dQ/d(log_kappa) = tau^2 (4 kappa^4 C + 4 kappa^2 G)
    //   dQ/d(log_tau)   = 2 (Q - R)
    const double k2 = kappa * kappa;
    const double k4 = k2 * k2;
    const double tau2 = tau * tau;

    // x*'(dQ/d_log_range) x* and x*'(dQ/d_log_sigma) x*
    double xdQdr_x = 0.0;
    double xdQds_x = 0.0;

    for (int col = 0; col < n_mesh; col++) {
        for (int qidx = qb.Q_p[col]; qidx < qb.Q_p[col + 1]; qidx++) {
            int row = qb.Q_i[qidx];
            double wiwj = mode[mesh_start + row] * mode[mesh_start + col];

            // Assembled (theta-dependent) part of this entry.
            double q_theta = qb.Q_x[qidx] - qb.orphan_contrib[qidx];

            double dq_dlogk = tau2 * (4.0 * k4 * qb.c0_contrib()[qidx] +
                                       4.0 * k2 * qb.g1_contrib()[qidx]);

            // Range moves kappa (coefficient -1) and, through kappa, tau
            // (coefficient +nu = +1).
            xdQdr_x += wiwj * (-dq_dlogk + 2.0 * q_theta);
            // Sigma moves tau alone (coefficient -1).
            xdQds_x += wiwj * (-2.0 * q_theta);
        }
    }

    double d_logprior_d_logrange = -0.5 * xdQdr_x;
    double d_logprior_d_logsigma = -0.5 * xdQds_x;

    // --- Term 2: -0.5 tr(H^{-1} dH/d(theta)), and the prior normalizer's
    //     +0.5 tr(Q^{-1} dQ/d(theta)) ---
    // dH/d(theta) = dQ/d(theta) (the likelihood part does not depend on theta).
    // Both traces run over the full (symmetric) Q pattern: the off-diagonal is
    // where the SPDE smoothing lives, so it carries real trace weight.
    //
    // The prior normalizer 0.5 log|Q(theta)| is NOT absorbed by -0.5 log|H|
    // (H = Q + A'WA != Q) -- see spde_logdet.h. It is the Occam term that gives
    // the (range, sigma) marginal an interior maximum instead of railing sigma
    // to the prior boundary, so its derivative belongs here whenever the value
    // carries it.
    //
    // Rebuild the posterior Hessian H = (X'WX | X'WA ; A'WX | A'WA + Q) at the
    // mode through the same scatter the inner Newton loop uses, and factor it
    // here. That loop uses the dense Cholesky path below the sparse threshold,
    // so its `solver` is not guaranteed to hold a factor; building and
    // factoring H here makes the selected inversion available for any mesh
    // size. The Takahashi selected inverse then gives H^{-1}_{ij} on the
    // fill-in pattern of the Cholesky factor, a superset of Q's nonzeros, so
    // every (i,j) on Q's pattern is available.
    int n_x = mesh_start + n_mesh;
    std::vector<double> eta(N, 0.0);
    spde_compute_eta(X, N, p, mesh_start, a_rows, offset,
                     re_idx, n_re_groups, mode, eta);

    const double tau_re = (n_re_groups > 0)
                          ? 1.0 / (sigma_re * sigma_re + 1e-10) : 0.0;
    DenseVec grad_scratch(n_x, 0.0);
    DenseMat H(n_x, DenseVec(1, 0.0));
    spde_scatter(y, n_trials, X, N, p, n_mesh, mesh_start, a_rows, qb,
                 family, phi, re_idx, n_re_groups, tau_re, DEFAULT_TAU_BETA,
                 mode, eta, grad_scratch, H);

    cholmod_sparse* H_sp = dense_to_cholmod_sparse_drop(H, n_x, 1e-14, &solver.common());
    solver.reset();
    solver.analyze(H_sp);
    const bool h_factored = solver.factorize(H_sp);
    M_cholmod_free_sparse(&H_sp, &solver.common());

    // An indefinite H at this (range, sigma) has no selected inverse. Report
    // non-finite gradients, which the sampler treats as a rejection, rather
    // than reading an empty one.
    if (!h_factored) return res;

    SparseCholeskySolver::SelectedInverse H_inv = solver.selected_inversion_full();
    // Q's own selected inverse, off the factor the log-determinant already
    // built for this cell.
    SparseCholeskySolver::SelectedInverse Q_inv =
        qld.solver.selected_inversion_full();
    if (H_inv.n == 0 || Q_inv.n == 0) return res;

    double trace_range = 0.0;   // tr(H^-1 dQ/dlog_range)
    double trace_sigma = 0.0;   // tr(H^-1 dQ/dlog_sigma)
    double q_trace_range = 0.0; // tr(Q^-1 dQ/dlog_range)
    double q_trace_sigma = 0.0; // tr(Q^-1 dQ/dlog_sigma)
    for (int col = 0; col < n_mesh; col++) {
        for (int qidx = qb.Q_p[col]; qidx < qb.Q_p[col + 1]; qidx++) {
            int row = qb.Q_i[qidx];

            double h_inv = H_inv.at(mesh_start + row, mesh_start + col);
            double q_inv = Q_inv.at(row, col);

            double q_theta = qb.Q_x[qidx] - qb.orphan_contrib[qidx];
            double dq_dlogk = tau2 * (4.0 * k4 * qb.c0_contrib()[qidx] +
                                       4.0 * k2 * qb.g1_contrib()[qidx]);
            double dq_range = -dq_dlogk + 2.0 * q_theta;
            double dq_sigma = -2.0 * q_theta;

            trace_range   += h_inv * dq_range;
            trace_sigma   += h_inv * dq_sigma;
            q_trace_range += q_inv * dq_range;
            q_trace_sigma += q_inv * dq_sigma;
        }
    }

    res.grad_log_range = d_logprior_d_logrange + 0.5 * q_trace_range
                         - 0.5 * trace_range;
    res.grad_log_sigma = d_logprior_d_logsigma + 0.5 * q_trace_sigma
                         - 0.5 * trace_sigma;
    res.ok = true;

    return res;
}

} // namespace tulpa

#endif // TULPA_IMPLICIT_DIFF_H
