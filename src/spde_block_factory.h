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
// Not allowed as the copy block (restriction inherited from
// nested_laplace_joint_multi.cpp file header: copy semantics for spatial
// fields are gated to icar/bym2/car_proper).
//
// Operator assembly is selected at factory time by `use_rational`:
//   * use_rational = false: integer alpha, the exact FEM construction.
//   * use_rational = true:  fractional nu, rational coefficients captured
//                           at factory time.
// Independently, `direct_kappa_tau` selects what the two theta_grid axes
// hold -- Matern (range, sigma) converted per cell, or the operator
// (kappa, tau) used as-is.
//
// make_spde_block_precomputed is the third entry: the precision arrives already
// assembled (the rational rSPDE construction assembles Q = Pl' C^-1 Pl and the
// obs map A_eff = A Pr in R, .spde_rational_assemble), so there is no operator
// to rebuild and the block's latent is the auxiliary weights x rather than the
// field u = Pr x. Everything downstream of the assembly -- obs_indices, the
// pattern, both prior scatters, log_prior -- reads whatever CSC the builder
// holds and is shared with the FEM entries through spde_assemble_block.
//
// In the JOINT driver SPDE forces the sparse Newton path: that driver's dense
// scatter resolves INDEXED_SINGLE blocks through `idx` only, so it cannot see
// an INDEXED_MULTI block's per-obs mesh weights. Its dispatch routes to the
// sparse path whenever any block has contrib_kind != INDEXED_SINGLE. The dense
// `add_prior` below is for the non-joint multi-block driver, whose scatter does
// handle INDEXED_MULTI; it mirrors add_prior_sparse value-for-value.

#ifndef TULPA_SPDE_BLOCK_FACTORY_H
#define TULPA_SPDE_BLOCK_FACTORY_H

#include "latent_block.h"
#include "laplace_re_priors.h"           // center_effects
#include "nl_cell_cache.h"
#include "sparse_hessian.h"
#include "spde_qbuilder.h"               // SpdeQBuilder, ARows, build_A_rows
#include "spde_logdet.h"                 // SpdeQLogDet (0.5 log|Q| normalizer)
#include <Rcpp.h>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace tulpa {

// spde_range_sigma_to_kappa_tau (the Matern axis conversion the nested
// integrators apply per grid cell) lives in spde_qbuilder.h, next to the
// assembly it parameterizes.

// Per-cell state (nl_cell_cache.h): Q values and the prior normalizer
// 0.5 log|Q(theta)| are written per outer cell by prep(), and the joint driver
// runs outer cells concurrently — a shared builder/scalar would let one cell's
// prep clobber another cell's in-flight Newton reads. Each slot copies the
// pattern-initialized template on first use; the CHOLMOD log-det state
// (symbolic factor) is per-slot too.
struct SpdeCellState {
    SpdeQBuilder qb;
    SpdeQLogDet  qld;
    double       half_ldQ = 0.0;
};

// Per-arm ARows, materialized once at factory time so per-obs lookups in the
// eta/scatter loops are O(nnz_per_row).
inline std::shared_ptr<std::vector<ARows>> spde_build_a_rows_per_arm(
    const Rcpp::List&          A_x_per_arm,
    const Rcpp::List&          A_i_per_arm,
    const Rcpp::List&          A_p_per_arm,
    const Rcpp::IntegerVector& n_obs_per_arm,
    int                        n_arms,
    int                        n_mesh,
    int                        block_index
) {
    if (static_cast<int>(A_x_per_arm.size()) != n_arms ||
        static_cast<int>(A_i_per_arm.size()) != n_arms ||
        static_cast<int>(A_p_per_arm.size()) != n_arms ||
        n_obs_per_arm.size() != n_arms) {
        Rcpp::stop("Block %d (type 'spde'): A_x/A_i/A_p/n_obs_per_arm must "
                   "each have length n_arms (%d).",
                   block_index + 1, n_arms);
    }
    auto a_rows_per_arm = std::make_shared<std::vector<ARows>>(n_arms);
    for (int k = 0; k < n_arms; k++) {
        Rcpp::NumericVector A_x = A_x_per_arm[k];
        Rcpp::IntegerVector A_i = A_i_per_arm[k];
        Rcpp::IntegerVector A_p = A_p_per_arm[k];
        char arm_label[64];
        std::snprintf(arm_label, sizeof(arm_label),
                      "Block %d (type 'spde'): A[[%d]]", block_index + 1, k + 1);
        spde_validate_projector(n_mesh, n_obs_per_arm[k], A_x, A_i, A_p,
                                arm_label);
        (*a_rows_per_arm)[k] = build_A_rows(n_obs_per_arm[k], n_mesh,
                                            A_x, A_i, A_p);
    }
    return a_rows_per_arm;
}

// Assemble the LatentBlock around a pattern-seeded template builder. Every
// field below reads whatever CSC the per-cell builder holds, so the FEM entries
// (which rebuild Q from the cell's operator parameters) and the precomputed
// entry (whose Q is fixed) share all of it; only `refresh` and `center_field`
// differ between them.
//
// `cell_in_domain` is an optional pre-claim parameter-domain gate. It runs
// BEFORE the cache slot is claimed, so a cell outside the operator's domain
// does not retract the calling thread's currently published state.
// `refresh` then fills the claimed slot and reports PD feasibility; returning
// false marks the cell infeasible (log_marginal = -inf), matching the
// proper-CAR PD gate.
inline LatentBlock spde_assemble_block(
    int                                          start,
    int                                          n_mesh,
    std::shared_ptr<std::vector<ARows>>          a_rows_per_arm,
    std::shared_ptr<SpdeQBuilder>                qb,
    std::function<bool(int)>                     cell_in_domain,
    std::function<bool(int, SpdeCellState&)>     refresh,
    bool                                         center_field
) {
    auto cell_cache = std::make_shared<NlCellCache<SpdeCellState>>(
        [qb](SpdeCellState& st) { st.qb = *qb; });

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

    block.prep = [cell_cache, cell_in_domain, refresh](int k_grid) -> bool {
        if (cell_in_domain && !cell_in_domain(k_grid)) return false;
        auto& st = cell_cache->claim();
        if (!refresh(k_grid, st)) return false;
        cell_cache->publish(k_grid);
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
    block.add_prior_sparse = [start, n_mesh, cell_cache](
        SparseHessianBuilder& H, DenseVec& grad,
        const Rcpp::NumericVector& x, int k_grid
    ) {
        const SpdeQBuilder& q_cell = cell_cache->find(k_grid).qb;
        for (int col = 0; col < n_mesh; col++) {
            for (int idx = q_cell.Q_p[col]; idx < q_cell.Q_p[col + 1]; idx++) {
                int row = q_cell.Q_i[idx];
                double q = q_cell.Q_x[idx];
                grad[start + row] -= q * x[start + col];
                if (row >= col) {
                    H.add(start + row, start + col, q);
                }
            }
        }
    };

    // Dense add_prior: the same FEM precision Q scattered into a DenseMat for
    // the dense Newton path (n_x < SPARSE_THRESHOLD), used by the non-joint
    // multi-block driver. Mirrors add_prior_sparse exactly -- gradient walks
    // the FULL Q (both triangles via the CSC), Hessian fills both triangles so
    // the caller's lower->upper symmetrise does not clobber it. Reads the same
    // qb->Q_x rebuilt by prep(), so there is one source of FEM assembly.
    block.add_prior = [start, n_mesh, cell_cache](
        DenseVec& grad, DenseMat& H,
        const Rcpp::NumericVector& x, int k_grid
    ) {
        const SpdeQBuilder& q_cell = cell_cache->find(k_grid).qb;
        for (int col = 0; col < n_mesh; col++) {
            for (int idx = q_cell.Q_p[col]; idx < q_cell.Q_p[col + 1]; idx++) {
                int row = q_cell.Q_i[idx];
                double q = q_cell.Q_x[idx];
                grad[start + row] -= q * x[start + col];
                H[start + row][start + col] += q;
            }
        }
    };

    block.log_prior = [start, n_mesh, cell_cache](
        const Rcpp::NumericVector& x, int k_grid
    ) -> double {
        const auto& st = cell_cache->find(k_grid);
        const SpdeQBuilder& q_cell = st.qb;
        double qf = 0.0;
        for (int col = 0; col < n_mesh; col++) {
            double x_col = x[start + col];
            for (int idx = q_cell.Q_p[col]; idx < q_cell.Q_p[col + 1]; idx++) {
                qf += x[start + q_cell.Q_i[idx]] * q_cell.Q_x[idx] * x_col;
            }
        }
        // log p(x|theta) = 0.5 log|Q(theta)| - 0.5 x'Qx (the theta-independent
        // -(n/2) log(2 pi) is dropped). 0.5 log|Q| is the prior normalizer set
        // by prep() each cell; it is the Occam term that makes the (range,sigma)
        // marginal interior-peaked rather than monotone in sigma. It is NOT
        // absorbed by the Laplace Hessian log-determinant (H = Q + A'WA != Q).
        return st.half_ldQ - 0.5 * qf;
    };

    // Sum-to-zero centring for intercept identifiability. Left empty for the
    // precomputed rational latent, whose auxiliary weights x are not the field
    // (u = Pr x) and whose proper prior (kappa^2 > 0) already identifies the
    // constant mode.
    if (center_field) {
        block.center = [start, n_mesh](Rcpp::NumericVector& x) {
            return tulpa::center_intercept(x, start, n_mesh);
        };
    }

    return block;
}

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
    const std::vector<double>&     rational_weights,
    // Meaning of the two theta_grid columns selected by axis_range / axis_sigma.
    // false (default): Matern (range, sigma), converted per cell by
    // spde_range_sigma_to_kappa_tau -- what the nested integrators grid over.
    // true: the columns ARE the operator parameters (kappa, tau) and are used
    // as-is. The fixed-hyperparameter single fit takes (kappa, tau) from its
    // caller, so a round-trip through (range, sigma) would be the conversion
    // composed with its inverse; taking them directly keeps that fit exact.
    bool                           direct_kappa_tau = false,
    // Optional: receives the FEM precision's nonzero count. The pattern is
    // fixed across cells (prep only reweights it), so it is a property of the
    // mesh and is read off the template builder below -- a caller that reports
    // Q_nnz does not enumerate the pattern a second time to get it.
    int*                           q_nnz_out = nullptr
) {
    auto a_rows_per_arm = spde_build_a_rows_per_arm(
        A_x_per_arm, A_i_per_arm, A_p_per_arm, n_obs_per_arm,
        n_arms, n_mesh, block_index);

    spde_validate_fem(n_mesh, C0_diag, G1_x, G1_i, G1_p);

    // alpha = nu + d/2 with d = 2.
    const int alpha = static_cast<int>(std::round(nu)) + 1;

    // Template QBuilder — pattern built once at factory time. It seeds the
    // per-cell slot copies and serves the (immutable) pattern reads in
    // add_prior_pattern. The chain is built for the requested operator order,
    // so the pattern widens with alpha; the rational assembly shifts the
    // alpha = 2 stencil regardless of the (fractional) nu it approximates.
    auto qb = std::make_shared<SpdeQBuilder>();
    qb->init(n_mesh, C0_diag, G1_x, G1_i, G1_p, use_rational ? 2 : alpha);
    if (q_nnz_out) *q_nnz_out = qb->nnz();

    auto in_domain = [axis_range, axis_sigma, theta_grid](int k_grid) -> bool {
        return theta_grid(k_grid, axis_range) > 0.0 &&
               theta_grid(k_grid, axis_sigma) > 0.0;
    };

    // Per outer-grid cell: rebuild Q values for the cell's (range, sigma) and
    // recompute the prior normalizer 0.5 log|Q(theta)| consumed by log_prior.
    auto refresh = [axis_range, axis_sigma, theta_grid, nu, use_rational,
                    direct_kappa_tau, rational_poles, rational_weights](
        int k_grid, SpdeCellState& st
    ) -> bool {
        const double a0 = theta_grid(k_grid, axis_range);
        const double a1 = theta_grid(k_grid, axis_sigma);
        double kappa, tau;
        if (direct_kappa_tau) {
            kappa = a0;
            tau   = a1;
        } else {
            std::tie(kappa, tau) = spde_range_sigma_to_kappa_tau(a0, a1, nu);
        }
        if (use_rational) {
            st.qb.rebuild_rational(kappa, tau, rational_poles, rational_weights);
        } else {
            st.qb.rebuild(kappa, tau);
        }
        return st.qld.half_logdet(st.qb, st.half_ldQ);
    };

    return spde_assemble_block(start, n_mesh, a_rows_per_arm, qb,
                               in_domain, refresh, /*center_field=*/true);
}

// SPDE block over a PRECOMPUTED precision and observation map.
//
// The rational rSPDE construction (.spde_rational_assemble, R/brasil.R) makes
// the latent the auxiliary weights x ~ N(0, Q^-1) with field u = Pr x, so the
// obs map is A_eff = A Pr and the precision is Q = Pl' C^-1 Pl. Both arrive
// assembled as CSC; nothing here depends on the outer cell, so `refresh` only
// computes the prior normalizer and there is no operator rebuild.
//
// `Q_*` is the FULL symmetric precision in CSC (both triangles), the layout
// SpdeQBuilder holds and every scatter/log_prior below reads.
inline LatentBlock make_spde_block_precomputed(
    int                            start,
    int                            n_mesh,
    const Rcpp::List&              A_x_per_arm,
    const Rcpp::List&              A_i_per_arm,
    const Rcpp::List&              A_p_per_arm,
    const Rcpp::IntegerVector&     n_obs_per_arm,
    int                            n_arms,
    int                            block_index,
    const Rcpp::IntegerVector&     Q_p,
    const Rcpp::IntegerVector&     Q_i,
    const Rcpp::NumericVector&     Q_x,
    int*                           q_nnz_out = nullptr
) {
    auto a_rows_per_arm = spde_build_a_rows_per_arm(
        A_x_per_arm, A_i_per_arm, A_p_per_arm, n_obs_per_arm,
        n_arms, n_mesh, block_index);

    if (Q_p.size() != n_mesh + 1) {
        Rcpp::stop("Block %d (type 'spde', precomputed): length(Q_p) must be "
                   "n_mesh + 1 (%d), got %d.",
                   block_index + 1, n_mesh + 1, static_cast<int>(Q_p.size()));
    }
    if (Q_i.size() != Q_x.size()) {
        Rcpp::stop("Block %d (type 'spde', precomputed): Q_i (%d) and Q_x (%d) "
                   "must have the same length.",
                   block_index + 1, static_cast<int>(Q_i.size()),
                   static_cast<int>(Q_x.size()));
    }
    // Both prior scatters walk this CSC and write grad[start + row] /
    // H[start + row][start + col], so a row index outside the block or a
    // column pointer past the value count writes into another block's latent
    // slice or past the end of the Hessian.
    {
        char q_label[64];
        std::snprintf(q_label, sizeof(q_label),
                      "Block %d (type 'spde', precomputed): Q",
                      block_index + 1);
        spde_validate_csc(n_mesh, Q_p, Q_i, static_cast<int>(Q_x.size()),
                          n_mesh, q_label);
    }

    auto qb = std::make_shared<SpdeQBuilder>();
    qb->n_mesh = n_mesh;
    qb->Q_p.assign(Q_p.begin(), Q_p.end());
    qb->Q_i.assign(Q_i.begin(), Q_i.end());
    qb->Q_x.assign(Q_x.begin(), Q_x.end());
    if (q_nnz_out) *q_nnz_out = qb->nnz();

    // Q does not depend on the cell, so the slot copy made at first claim is
    // already the finished precision; only 0.5 log|Q| has to be computed, once
    // per slot. A precision the Cholesky cannot factor leaves the normalizer at
    // 0 and keeps the cell feasible: the rational Q spans ~1e13 in condition
    // number by construction, and its callers either need only the mode or
    // re-derive the marginal through the well-conditioned operator factor Pl
    // (.spde_nested_logmarginal_at), so dropping the cell would discard a valid
    // mode over a normalizer they do not read.
    auto refresh = [](int, SpdeCellState& st) -> bool {
        if (!st.qld.half_logdet(st.qb, st.half_ldQ)) st.half_ldQ = 0.0;
        return true;
    };

    return spde_assemble_block(start, n_mesh, a_rows_per_arm, qb,
                               /*cell_in_domain=*/nullptr, refresh,
                               /*center_field=*/false);
}

} // namespace tulpa

#endif // TULPA_SPDE_BLOCK_FACTORY_H
