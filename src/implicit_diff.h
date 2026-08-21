// implicit_diff.h
// Implicit differentiation of Laplace-marginalized log-likelihood
// w.r.t. hyperparameters, for NUTS sampling over theta.
//
// Given:
//   log p(y|theta) ~ log p(y|x*,theta) + log p(x*|theta) - 0.5 log|H| + const
//
// Gradient (via implicit function theorem). The TOTAL derivative has three
// parts. Stationarity of the score cancels dx*/d(theta) out of the first two
// terms, but not out of the determinant: H = A' diag(w(eta)) A + Q(theta) is
// built from the per-observation curvature w, and eta reads the mode, so
// log|H| depends on theta through x*(theta) as well as directly.
//
//   d/d(theta) log p(y|theta)
//     =  d/d(theta) log p(x*|theta)                     [Q's own dependence]
//        - 0.5 tr(H^{-1} dQ/d(theta))                   [H's direct dependence]
//        - 0.5 sum_k tr(H^{-1} dH/dx_k) (dx*/d(theta))_k    [H via the mode]
//
// Without the third part the result is not the derivative of the log-marginal
// the caller reports; it is the partial derivative at a frozen mode.
//
// The first two use selected inversion (Takahashi) to get H^{-1} and Q^{-1} at
// the nonzero positions of dQ/d(theta); the third reads the same factor of H
// through solves.
//
// Reference: Margossian, Vehtari et al. (2023-2024)

#ifndef TULPA_IMPLICIT_DIFF_H
#define TULPA_IMPLICIT_DIFF_H

#include "laplace_family_curvature.h"
#include "sparse_cholesky.h"
#include "spde_logdet.h"
#include "spde_qbuilder.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace tulpa {

// Gradient of SPDE Laplace log-marginal w.r.t. (log_range, log_sigma).
// Requires: mode x* (from inner Laplace), Q builder, and the log-determinant
// helper holding the factor of Q(theta) the value used.
//
// `mode` must be the point the log-marginal was evaluated at, i.e. the
// Newton iterate itself. A sum-to-zero presentation fold moves the mesh field
// without moving the arm intercept, so a folded mode is a different point:
// the quadratic form x*' Q x* below and the curvature H is built from both
// change, and the expansion no longer belongs to the reported value.
//
// Returns: gradient vector of length 2 (d/d(log_range), d/d(log_sigma)).
// `ok` is false when the posterior Hessian at (range, sigma) could not be
// factorized; the gradients are then NaN.
struct ImplicitDiffResult {
    double grad_log_range;
    double grad_log_sigma;
    bool   ok;
};

// dQ/d(log_range) and dQ/d(log_sigma) at one entry of the alpha = 2 pattern.
//
// Q = tau^2 (kappa^4 C + 2 kappa^2 G + G C^-1 G) + R, with R the
// theta-independent orphan ridge (spde_zero_mass.h). Only the assembled part
// carries the tau^2, so dQ/d(log_tau) = 2 (Q - R) and
// dQ/d(log_kappa) = tau^2 (4 kappa^4 C + 4 kappa^2 G). Range moves kappa
// (coefficient -1) and, through kappa, tau (coefficient +nu = +1); sigma moves
// tau alone (coefficient -1).
struct SpdeDQEntry {
    double d_range;
    double d_sigma;
};

inline SpdeDQEntry spde_dq_entry(const SpdeQBuilder& qb, int qidx,
                                 double k2, double k4, double tau2) {
    const double q_theta  = qb.Q_x[qidx] - qb.orphan_contrib[qidx];
    const double dq_dlogk = tau2 * (4.0 * k4 * qb.c0_contrib()[qidx] +
                                    4.0 * k2 * qb.g1_contrib()[qidx]);
    SpdeDQEntry dq;
    dq.d_range = -dq_dlogk + 2.0 * q_theta;
    dq.d_sigma = -2.0 * q_theta;
    return dq;
}

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

    // The mode term below differentiates the weight H is actually built from
    // and solves the mode Jacobian against H itself. Both are exact only where
    // the Newton working weight IS the observed curvature -- otherwise H is
    // not the second derivative the Jacobian is governed by, and the exact
    // gradient would need a second factorization of A' W_obs A + Q.
    if (!working_weight_is_observed(family) ||
        !has_curvature_derivative(family)) {
        Rcpp::stop("spde_implicit_gradient needs a family whose Newton working "
                   "weight is the observed curvature and carries a closed-form "
                   "eta-derivative; '%s' does not.", family.c_str());
    }

    double kappa, tau;
    std::tie(kappa, tau) = spde_range_sigma_to_kappa_tau(range, sigma_spde, nu);

    const int n_x = mesh_start + n_mesh;

    // --- Term 1: d/d(theta) of log p(x*|theta) ---
    //   log p(x*|theta) = 0.5 log|Q(theta)| - 0.5 x*' Q(theta) x*
    // Q depends on theta through kappa and tau:
    //   kappa = sqrt(8 nu) / range         => d(log_kappa)/d(log_range) = -1
    //   tau   = 1/(sqrt(4 pi nu) kappa^nu sigma)
    //                                      => d(log_tau)/d(log_sigma)   = -1
    //                                         d(log_tau)/d(log_range)   = +nu
    // spde_dq_entry() carries the per-entry closed forms.
    const double k2 = kappa * kappa;
    const double k4 = k2 * k2;
    const double tau2 = tau * tau;

    // x*'(dQ/d_log_range) x* and x*'(dQ/d_log_sigma) x*, and alongside them the
    // matvecs (dQ/d(theta)) x* on the mesh slice, which the mode Jacobian
    // solves against below. Both read the same pass over Q's pattern.
    double xdQdr_x = 0.0;
    double xdQds_x = 0.0;
    std::vector<double> dQx_range(n_x, 0.0);
    std::vector<double> dQx_sigma(n_x, 0.0);

    for (int col = 0; col < n_mesh; col++) {
        const double x_col = mode[mesh_start + col];
        for (int qidx = qb.Q_p[col]; qidx < qb.Q_p[col + 1]; qidx++) {
            const int row = qb.Q_i[qidx];
            const SpdeDQEntry dq = spde_dq_entry(qb, qidx, k2, k4, tau2);
            const double wiwj = mode[mesh_start + row] * x_col;

            xdQdr_x += wiwj * dq.d_range;
            xdQds_x += wiwj * dq.d_sigma;
            dQx_range[mesh_start + row] += dq.d_range * x_col;
            dQx_sigma[mesh_start + row] += dq.d_sigma * x_col;
        }
    }

    double d_logprior_d_logrange = -0.5 * xdQdr_x;
    double d_logprior_d_logsigma = -0.5 * xdQds_x;

    // --- Term 2: -0.5 tr(H^{-1} dQ/d(theta)), and the prior normalizer's
    //     +0.5 tr(Q^{-1} dQ/d(theta)) ---
    // H's DIRECT theta dependence is Q's alone: at a frozen mode the likelihood
    // block A' diag(w) A does not move. What it contributes through the mode is
    // Term 3 below.
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

    // --- Term 3: -0.5 tr(H^{-1} dH/dx) dx*/d(theta), the mode's movement ---
    // Differentiating the score A' l'(eta) - P x* = 0 gives the mode Jacobian
    //   dx*/d(theta) = -H^{-1} (dQ/d(theta)) x*,
    // exact here because the working weight H carries is the observed
    // curvature (gated above). With a_i the observation's row of the full
    // design [X | RE | A] and w_i = w(eta_i) the curvature H is built from,
    //   dH/dx_k = sum_i (dw_i/deta_i) a_ik a_i a_i',
    // so the whole term collapses to a sum over observations,
    //   -0.5 sum_i (dw_i/deta_i) (a_i' H^{-1} a_i) (a_i' dx*/d(theta)),
    // in which a_i' H^{-1} a_i is the Gaussian-approximation variance of eta_i
    // and a_i' dx*/d(theta) the movement of eta_i. Both read the factor built
    // above, so nothing is refactorized.
    for (int j = 0; j < n_x; j++) {
        dQx_range[j] = -dQx_range[j];
        dQx_sigma[j] = -dQx_sigma[j];
    }
    std::vector<double> dx_range(n_x, 0.0), dx_sigma(n_x, 0.0);
    solver.solve(dQx_range.data(), dx_range.data(), n_x);
    solver.solve(dQx_sigma.data(), dx_sigma.data(), n_x);

    // The offset is constant in theta, so the eta tangent carries none.
    std::vector<double> deta_range(N, 0.0), deta_sigma(N, 0.0);
    spde_compute_eta(X, N, p, mesh_start, a_rows, /*offset=*/nullptr,
                     re_idx, n_re_groups, dx_range, deta_range);
    spde_compute_eta(X, N, p, mesh_start, a_rows, /*offset=*/nullptr,
                     re_idx, n_re_groups, dx_sigma, deta_sigma);

    double mode_trace_range = 0.0;
    double mode_trace_sigma = 0.0;
    {
        std::vector<double> a_row(n_x, 0.0), h_inv_a(n_x, 0.0);
        for (int i = 0; i < N; i++) {
            std::fill(a_row.begin(), a_row.end(), 0.0);
            for (int j = 0; j < p; j++) a_row[j] = X(i, j);
            const int g = spde_re_group(re_idx, i, n_re_groups);
            if (g >= 0) a_row[p + g] += 1.0;
            for (const auto& ae : a_rows[i]) {
                a_row[mesh_start + ae.mesh_idx] += ae.weight;
            }
            solver.solve(a_row.data(), h_inv_a.data(), n_x);
            double var_eta = 0.0;
            for (int j = 0; j < n_x; j++) var_eta += a_row[j] * h_inv_a[j];

            const double dw_deta = curvature_deta_for_family(
                y[i], n_trials[i], eta[i], family, phi);
            mode_trace_range += dw_deta * var_eta * deta_range[i];
            mode_trace_sigma += dw_deta * var_eta * deta_sigma[i];
        }
    }

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
            const int row = qb.Q_i[qidx];

            const double h_inv = H_inv.at(mesh_start + row, mesh_start + col);
            const double q_inv = Q_inv.at(row, col);
            const SpdeDQEntry dq = spde_dq_entry(qb, qidx, k2, k4, tau2);

            trace_range   += h_inv * dq.d_range;
            trace_sigma   += h_inv * dq.d_sigma;
            q_trace_range += q_inv * dq.d_range;
            q_trace_sigma += q_inv * dq.d_sigma;
        }
    }

    res.grad_log_range = d_logprior_d_logrange + 0.5 * q_trace_range
                         - 0.5 * trace_range - 0.5 * mode_trace_range;
    res.grad_log_sigma = d_logprior_d_logsigma + 0.5 * q_trace_sigma
                         - 0.5 * trace_sigma - 0.5 * mode_trace_sigma;
    res.ok = true;

    return res;
}

} // namespace tulpa

#endif // TULPA_IMPLICIT_DIFF_H
