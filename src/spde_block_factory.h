// spde_block_factory.h
// Build a LatentBlock for an SPDE Matern field on a shared mesh, plugged
// into the joint multi-arm nested-Laplace driver via the LatentBlock
// interface.
//
// SPDE uses INDEXED_MULTI semantics: each obs sees ~3 mesh nodes via
// barycentric weights from the FEM projector A. obs_indices fills the per-
// obs (mesh_node, weight) list; the sparse scatter consumes it. The prior
// is the FEM precision Q(kappa, tau) built by SpdeQBuilder; the per-cell
// `prep` callback rebuilds Q values for the cell's (range, sigma) pair.
//
// Mesh data (C0_diag, G1) and the FEM projector A are SHARED across arms
// in spec form; per-arm A_rows are materialized at factory time so per-obs
// lookups in the eta/scatter loops are O(nnz_per_row).
//
// Not allowed as the copy block (first ship restriction inherited from
// nested_laplace_joint_multi.cpp file header: copy semantics for spatial
// fields are gated to icar/bym2/car_proper).
//
// Two axis schemas, dispatched at factory time on `use_rational`:
//   * use_rational = false (integer alpha): axes are (range, sigma).
//   * use_rational = true  (fractional nu): axes are still (range, sigma);
//                                            rational coefs are constants
//                                            captured at factory time.
//
// SPDE forces the sparse Newton path (the dense scatter does not support
// INDEXED_MULTI). add_prior is left empty — calling the dense path on a
// blocks-list containing SPDE will skip the SPDE prior contribution, which
// would silently corrupt the fit. The dispatch in solve_at_theta_impl
// (1.4a) MUST route to the sparse path whenever any block has
// contrib_kind != INDEXED_SINGLE.

#ifndef TULPA_SPDE_BLOCK_FACTORY_H
#define TULPA_SPDE_BLOCK_FACTORY_H

#include "latent_block.h"
#include "laplace_re_priors.h"           // center_effects
#include "sparse_hessian.h"
#include "spde_qbuilder.h"               // SpdeQBuilder, ARows, build_A_rows
#include <Rcpp.h>
#include <memory>
#include <utility>
#include <vector>

namespace tulpa {

inline LatentBlock make_spde_block(
    int                            start,
    int                            n_mesh,
    const Rcpp::List&              A_x_per_arm,
    const Rcpp::List&              A_i_per_arm,
    const Rcpp::List&              A_p_per_arm,
    const Rcpp::IntegerVector&     n_obs_per_arm,
    int                            n_arms,
    int                            block_index,
    const Rcpp::NumericVector&     C0_diag,
    const Rcpp::NumericVector&     G1_x,
    const Rcpp::IntegerVector&     G1_i,
    const Rcpp::IntegerVector&     G1_p,
    double                         nu,
    int                            axis_range,
    int                            axis_sigma,
    const Rcpp::NumericMatrix&     theta_grid,
    bool                           use_rational,
    const std::vector<double>&     rational_poles,
    const std::vector<double>&     rational_weights
) {
    if (static_cast<int>(A_x_per_arm.size()) != n_arms ||
        static_cast<int>(A_i_per_arm.size()) != n_arms ||
        static_cast<int>(A_p_per_arm.size()) != n_arms ||
        n_obs_per_arm.size() != n_arms) {
        Rcpp::stop("Block %d (type 'spde'): A_x/A_i/A_p/n_obs_per_arm must "
                   "each have length n_arms (%d).",
                   block_index + 1, n_arms);
    }

    // Build per-arm ARows once at factory time.
    auto a_rows_per_arm =
        std::make_shared<std::vector<ARows>>(n_arms);
    for (int k = 0; k < n_arms; k++) {
        Rcpp::NumericVector A_x = A_x_per_arm[k];
        Rcpp::IntegerVector A_i = A_i_per_arm[k];
        Rcpp::IntegerVector A_p = A_p_per_arm[k];
        if (A_p.size() != n_mesh + 1) {
            Rcpp::stop("Block %d (type 'spde'): A_p[[%d]] must have length "
                       "n_mesh + 1 (%d), got %d.",
                       block_index + 1, k + 1, n_mesh + 1,
                       static_cast<int>(A_p.size()));
        }
        (*a_rows_per_arm)[k] = build_A_rows(n_obs_per_arm[k], n_mesh,
                                              A_x, A_i, A_p);
    }

    // Shared QBuilder — pattern fixed at init, values rebuilt per outer cell.
    auto qb = std::make_shared<SpdeQBuilder>();
    qb->init(n_mesh, C0_diag, G1_x, G1_i, G1_p);

    // alpha = nu + d/2 with d = 2.
    const int alpha = static_cast<int>(std::round(nu)) + 1;

    LatentBlock block;
    block.start = start;
    block.size  = n_mesh;
    block.contrib_kind = BlockContribKind::INDEXED_MULTI;
    block.prior_kind   = PriorFillKind::SPDE_Q;
    block.d_fac        = [](int) -> double { return 1.0; };
    // arm_scale left empty — copy not supported.

    // Per-obs (mesh_node_1based, weight) list. Pattern builder + sparse
    // scatter consume this for INDEXED_MULTI fill.
    block.obs_indices = [a_rows_per_arm](
        int i, int k_arm,
        std::vector<std::pair<int,double>>& out
    ) {
        out.clear();
        const ARows& rows = (*a_rows_per_arm)[k_arm];
        if (i < 0 || i >= static_cast<int>(rows.size())) return;
        const auto& row = rows[i];
        out.reserve(row.size());
        for (const auto& ae : row) {
            // 1-based block-local index; pattern builder / scatter add
            // `start` and subtract 1.
            out.emplace_back(ae.mesh_idx + 1, ae.weight);
        }
    };

    // idx left empty — INDEXED_MULTI uses obs_indices, not idx.

    // Per outer-grid cell: rebuild Q values for the cell's (range, sigma).
    block.prep = [qb, axis_range, axis_sigma, theta_grid,
                   nu, alpha, use_rational,
                   rational_poles, rational_weights](int k_grid) -> bool {
        double range = theta_grid(k_grid, axis_range);
        double sigma = theta_grid(k_grid, axis_sigma);
        if (!(range > 0.0) || !(sigma > 0.0)) return false;
        double kappa = std::sqrt(8.0 * nu) / range;
        double tau   = 1.0 / (std::sqrt(4.0 * M_PI) * kappa * sigma);
        if (use_rational) {
            qb->rebuild_rational(kappa, tau, rational_poles, rational_weights);
        } else {
            qb->rebuild(kappa, tau, alpha);
        }
        return true;
    };

    // Pattern: every Q nonzero contributes a lower-triangle entry in the
    // joint H pattern.
    block.add_prior_pattern = [start, n_mesh, qb](
        std::vector<std::pair<int,int>>& out
    ) {
        for (int col = 0; col < n_mesh; col++) {
            for (int idx = qb->Q_p[col]; idx < qb->Q_p[col + 1]; idx++) {
                int row = qb->Q_i[idx];
                if (row >= col) {
                    out.emplace_back(start + row, start + col);
                }
            }
        }
    };

    // Sparse Q scatter. Mirrors the inline prior-Q block in
    // cpp_nested_laplace_spde (spde_laplace.cpp lines 402-410): gradient
    // uses the FULL Q (both triangles); H uses LOWER triangle only.
    block.add_prior_sparse = [start, n_mesh, qb](
        SparseHessianBuilder& H, DenseVec& grad,
        const Rcpp::NumericVector& x, int /*k_grid*/
    ) {
        for (int col = 0; col < n_mesh; col++) {
            for (int idx = qb->Q_p[col]; idx < qb->Q_p[col + 1]; idx++) {
                int row = qb->Q_i[idx];
                double q = qb->Q_x[idx];
                grad[start + row] -= q * x[start + col];
                if (row >= col) {
                    H.add(start + row, start + col, q);
                }
            }
        }
    };

    // add_prior left empty: SPDE forces the sparse path. See file header.

    block.log_prior = [start, n_mesh, qb](
        const Rcpp::NumericVector& x, int /*k_grid*/
    ) -> double {
        double qf = 0.0;
        for (int col = 0; col < n_mesh; col++) {
            double x_col = x[start + col];
            for (int idx = qb->Q_p[col]; idx < qb->Q_p[col + 1]; idx++) {
                qf += x[start + qb->Q_i[idx]] * qb->Q_x[idx] * x_col;
            }
        }
        // Drop the log|Q|/2 normalizer — it is folded into the Laplace
        // log-marginal via the Hessian's log-determinant. Matches the
        // existing standalone driver's log_prior in spde_laplace.cpp.
        return -0.5 * qf;
    };

    block.center = [start, n_mesh](Rcpp::NumericVector& x) -> double {
        return center_effects(x, start, n_mesh);
    };

    return block;
}

} // namespace tulpa

#endif // TULPA_SPDE_BLOCK_FACTORY_H
