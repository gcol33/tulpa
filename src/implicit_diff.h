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

            // kappa channel of dQ/d(log_range). With
            //   Q = tau^2 (kappa^4 c0 + 2 kappa^2 g1 + gdg),
            //   dQ/d(log_kappa) = tau^2 (4 kappa^4 c0 + 4 kappa^2 g1),
            // and kappa = sqrt(8 nu)/range gives d(log_kappa)/d(log_range) = -1,
            // so the kappa channel of dQ/d(log_range) is -dQ/d(log_kappa).
            double dq_dlogk = tau2 * (4.0 * k4 * qb.c0_contrib[qidx] +
                                       4.0 * k2 * qb.g1_contrib[qidx]);
            xdQdr_x -= wiwj * dq_dlogk;

            // tau channel. Q is proportional to tau^2, so dQ/d(log_tau) = 2 Q.
            // tau = 1/(sqrt(4 pi) kappa sigma) gives d(log_tau)/d(log_sigma) = -1,
            // hence dQ/d(log_sigma) = -2 Q.
            xdQds_x -= 2.0 * wiwj * q;

            // Range also enters tau through kappa: d(log_tau)/d(log_range) =
            // -d(log_kappa)/d(log_range) = +1, so the tau channel of
            // dQ/d(log_range) is +2 Q.
            xdQdr_x += 2.0 * wiwj * q;
        }
    }

    // log p(x*|theta) quadratic part: -0.5 x*'Q x*. The 0.5 log|Q|
    // normalization is omitted here to match the inner solver's log_marginal,
    // which carries -0.5 x*'Q x* without the 0.5 log|Q| term.
    double d_logprior_d_logrange = -0.5 * xdQdr_x;
    double d_logprior_d_logsigma = -0.5 * xdQds_x;

    // --- Term 2: -0.5 tr(H^{-1} dH/d(theta)) ---
    // dH/d(theta) = dQ/d(theta) (likelihood part doesn't depend on theta).
    // tr(H^{-1} dQ/d(theta)) = sum_{(i,j) in Q} H^{-1}_{ij} * dQ_{ij}/d(theta),
    // over the full (symmetric) Q pattern — the off-diagonal is where the SPDE
    // smoothing lives, so it carries real trace weight.
    //
    // Rebuild the posterior Hessian H = (X'WX | X'WA ; A'WX | A'WA + Q) at the
    // mode, matching run_spde_laplace's scatter exactly, and factor it here.
    // The inner Newton solve uses the dense Cholesky path below the sparse
    // threshold, so its `solver` is not guaranteed to hold a factor; building
    // and factoring H here makes the selected inversion available for any mesh
    // size. The Takahashi selected inverse then gives H^{-1}_{ij} on the
    // fill-in pattern of the Cholesky factor, a superset of Q's nonzeros, so
    // every (i,j) on Q's pattern is available.
    int n_x = mesh_start + n_mesh;
    DenseMat H(n_x, DenseVec(1, 0.0));
    for (int i = 0; i < N; i++) {
        // eta at the mode for observation i.
        double eta_i = 0.0;
        for (int j = 0; j < p; j++) eta_i += X(i, j) * mode[j];
        for (const auto& ae : a_rows[i]) eta_i += ae.weight * mode[mesh_start + ae.mesh_idx];
        auto gh = grad_hess_for_family(y[i], n_trials[i], eta_i, family, phi);
        double w = gh.neg_hess;
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < p; k++) H[j][k] += w * X(i, j) * X(i, k);
        }
        const auto& row = a_rows[i];
        for (size_t s1 = 0; s1 < row.size(); s1++) {
            int idx1 = mesh_start + row[s1].mesh_idx;
            double a1 = row[s1].weight;
            H[idx1][idx1] += w * a1 * a1;
            for (int j = 0; j < p; j++) {
                H[j][idx1] += w * X(i, j) * a1;
                H[idx1][j] += w * X(i, j) * a1;
            }
            for (size_t s2 = s1 + 1; s2 < row.size(); s2++) {
                int idx2 = mesh_start + row[s2].mesh_idx;
                double cross = w * a1 * row[s2].weight;
                H[idx1][idx2] += cross;
                H[idx2][idx1] += cross;
            }
        }
    }
    for (int j = 0; j < n_mesh; j++) {
        for (int qidx = qb.Q_p[j]; qidx < qb.Q_p[j + 1]; qidx++) {
            int qi = qb.Q_i[qidx];
            H[mesh_start + qi][mesh_start + j] += qb.Q_x[qidx];
        }
    }
    const double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) H[j][j] += tau_beta;

    cholmod_sparse* H_sp = dense_to_cholmod_sparse_drop(H, n_x, 1e-14, &solver.common());
    solver.reset();
    solver.analyze(H_sp);
    solver.factorize(H_sp);
    M_cholmod_free_sparse(&H_sp, &solver.common());

    SparseCholeskySolver::SelectedInverse H_inv = solver.selected_inversion_full();

    double trace_range = 0.0;
    double trace_sigma = 0.0;
    for (int col = 0; col < n_mesh; col++) {
        for (int qidx = qb.Q_p[col]; qidx < qb.Q_p[col + 1]; qidx++) {
            int row = qb.Q_i[qidx];

            double h_inv = H_inv.at(mesh_start + row, mesh_start + col);

            double dq_dlogk = tau2 * (4.0 * k4 * qb.c0_contrib[qidx] +
                                       4.0 * k2 * qb.g1_contrib[qidx]);
            // dQ/d(log_range) = -(dQ/d(log_kappa)) + 2Q (kappa channel minus,
            // tau channel plus — d(log_tau)/d(log_range) = +1).
            trace_range += h_inv * (-dq_dlogk + 2.0 * qb.Q_x[qidx]);
            // dQ/d(log_sigma) = -2Q (d(log_tau)/d(log_sigma) = -1).
            trace_sigma += h_inv * (-2.0 * qb.Q_x[qidx]);
        }
    }

    res.grad_log_range = d_logprior_d_logrange - 0.5 * trace_range;
    res.grad_log_sigma = d_logprior_d_logsigma - 0.5 * trace_sigma;

    return res;
}

} // namespace tulpa

#endif // TULPA_IMPLICIT_DIFF_H
