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

#include "block_projector.h"             // projector rows + obs_indices
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
LatentBlock spde_assemble_block(
    int                                          start,
    int                                          n_mesh,
    std::shared_ptr<std::vector<ARows>>          a_rows_per_arm,
    std::shared_ptr<SpdeQBuilder>                qb,
    std::function<bool(int)>                     cell_in_domain,
    std::function<bool(int, SpdeCellState&)>     refresh,
    bool                                         center_field
);

LatentBlock make_spde_block(
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
);

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
    auto a_rows_per_arm = build_projector_rows_per_arm(
        A_x_per_arm, A_i_per_arm, A_p_per_arm, n_obs_per_arm,
        n_arms, n_mesh, block_index, "spde");

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
