// nested_laplace_multi.h
// The single-response LatentBlock driver: every single-block nested kernel
// (nested_laplace.cpp, via length-1 LatentBlock vectors) and the multi-block
// dispatch (nested_laplace_multi.cpp) solve their outer grid as a one-arm fit
// on the joint driver (nested_laplace_joint_multi.h).

#ifndef TULPA_NESTED_LAPLACE_MULTI_H
#define TULPA_NESTED_LAPLACE_MULTI_H

#include "hard_constraints.h"             // collect_hard_constraints
#include "tulpa_stop.h"
#include "laplace_newton.h"               // SPARSE_THRESHOLD
#include "latent_block.h"
#include "nested_laplace_grid.h"
#include "nested_laplace_joint_core.h"     // make_single_arm
#include "nested_laplace_joint_multi.h"    // the joint driver
#include <Rcpp.h>
#include <algorithm>
#include <vector>

namespace tulpa {

// Whether every row's loading vector is the same at every cell up to the
// per-block scalars d_fac_b(k). A DENSE_BASIS block's weights are its basis
// row at the cell (HSGP folds sqrt(S_j(theta_k)) into weight j), which scales
// each coefficient by its own factor, so rows are classed per cell there.
inline bool nl_loadings_cell_invariant(const std::vector<LatentBlock>& blocks) {
    for (const auto& b : blocks) {
        if (b.contrib_kind == BlockContribKind::DENSE_BASIS) return false;
    }
    return true;
}

// Cell-dependent bases change the coordinates carried by a neighbour's warm
// start. Their screening density uses the converged inner mode, so its ranking
// is independent of that coordinate map.
inline int nl_screen_iterations(const std::vector<LatentBlock>& blocks,
                                 int requested, int max_iter) {
    return nl_loadings_cell_invariant(blocks)
        ? requested : std::max(requested, max_iter);
}

// A single-response block fit as a one-arm fit on the joint driver. The latent
// blocks own their observation maps, so the arm carries only the response, the
// fixed-effect design and the RE block. `force_sparse` selects the joint sparse
// Newton outright; otherwise the joint driver applies its own rule (the latent
// dimension against SPARSE_THRESHOLD, and blocks_require_sparse). The sparse
// Newton assembles the latent blocks on their structural pattern, folds an
// intrinsic field's sum-to-zero pin at solve time and reaches a block's prior
// through add_prior_sparse. A model-supplied likelihood rides the arm's spec;
// the per-row predictor and its variance come back as `fitted_eta` /
// `fitted_eta_var`, one row per cell.
inline Rcpp::List run_single_arm_block_joint(
    int n_grid,
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const std::vector<LatentBlock>& blocks,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    bool store_modes, const Rcpp::NumericVector& x_init, bool store_Q,
    bool force_sparse,
    int n_threads_outer, double prune_tol,
    const LikelihoodSpec* ext_spec, void* ext_response,
    tulpa_progress::GridProgress* progress, GridCheckpoint* ckpt,
    bool compute_skew, const std::vector<int>* skew_probe_idx,
    const SubspaceDebiasOptions* debias, const CilaOptions* cila,
    int screen_iters, bool compute_fitted_var,
    Rcpp::Nullable<Rcpp::NumericVector> offset,
    const ScreenOffset& screen_log_offset,
    const BetaPrior* beta_prior = nullptr
) {
    std::vector<ParsedArm> parsed;
    std::vector<JointArm> arms;
    // With no RE block the caller may pass an empty re_idx; the arm reads one
    // entry per row, so it gets zeros.
    const Rcpp::NumericVector arm_re_idx =
        (n_re_groups == 0 && static_cast<int>(re_idx.size()) != N)
            ? Rcpp::NumericVector(N, 0.0) : re_idx;
    make_single_arm(parsed, arms, X, arm_re_idx, Rcpp::IntegerVector(N, 0),
                    p, n_re_groups, sigma_re, y, n_trials, family, phi, N,
                    offset);
    if (ext_spec) {
        arms[0].spec          = ext_spec;
        arms[0].response_data = ext_response;
    }
    if (beta_prior) apply_beta_prior(parsed[0], *beta_prior);

    Rcpp::List out = run_multi_block_nested_laplace_joint(
        n_grid, arms, parsed, blocks, /*n_x_after_re=*/p + n_re_groups,
        max_iter, tol, n_threads, store_modes, x_init, store_Q,
        /*prep_at_grid=*/nullptr, n_threads_outer,
        /*tile_ids=*/std::vector<int>(),
        /*tile_pilot_cells=*/std::vector<int>(), prune_tol,
        force_sparse,
        /*cell_coupling_spec=*/nullptr,
        JointPDMode::LM, StepCurvature::Observed,
        /*hessian_refresh=*/1, progress, ckpt,
        /*x_init_per_cell=*/std::vector<double>(),
        compute_skew, skew_probe_idx,
        /*fixed_block=*/nullptr, debias, cila,
        /*inner_sparse_override=*/0, screen_iters,
        store_modes && compute_fitted_var, screen_log_offset);
    if (store_modes) nl_attach_fitted_eta_single_arm(out, arms, parsed, blocks);
    Rcpp::List cc = write_constraint_rows(collect_hard_constraints(blocks));
    if (cc.size() > 0) out["constraint_cols"] = cc;
    return out;
}

// The outer grid over a vector of LatentBlocks for one response.
inline Rcpp::List run_multi_block_nested_laplace(
    int n_grid,
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const std::vector<LatentBlock>& blocks,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    bool store_modes,
    const Rcpp::NumericVector& x_init,
    bool store_Q = false,
    int n_threads_outer = 1,
    double prune_tol = 0.0,
    const LikelihoodSpec* ext_spec = nullptr,
    void* ext_response = nullptr,
    tulpa_progress::GridProgress* progress = nullptr,
    GridCheckpoint* ckpt = nullptr,
    // Inner-Laplace skewness diagnostic (inner_laplace_skew.h), opt-in like
    // store_Q. Applied only to the FULL per-cell solve, never the cheap-pass
    // warm-start screen; the caller is expected to pass a length-1 theta grid
    // (n_grid == 1) built at a single target theta (typically the outer
    // grid's MAP cell) when requesting this, mirroring the outer Pareto-k
    // diagnostic's re-dispatch-at-a-point convention -- see
    // .nl_inner_skew_at_theta() in R/laplace_diagnostics.R.
    bool compute_skew = false,
    const std::vector<int>* skew_probe_idx = nullptr,
    // Subspace debias (subspace_debias.h). Unlike the
    // diagnostics above this runs on EVERY integrated cell, because the
    // correction it produces enters the reported marginal as a mixture over the
    // outer grid -- one node's corrected shape is not the grid's. The cheap
    // warm-start screen never runs it. nullptr or an empty index set never
    // reaches the sampler, so the grid is unchanged and consumes no random
    // number.
    const SubspaceDebiasOptions* debias = nullptr,
    // Corrected integrated Laplace (inner_cila.h). Runs on
    // every fully-solved cell for the same reason the debias does -- the
    // corrected marginal is a reweighting of each cell's own particle set, so
    // all of them travel out -- and never on the cheap warm-start screen. Its
    // auxiliary points come from an engine-owned stream keyed by the cell index,
    // so the outer grid stays parallel-integrable with the correction on.
    const CilaOptions* cila = nullptr,
    // Inner Newton steps per cell in the cheap screening sweep, forwarded to
    // run_nested_laplace_grid.
    int screen_iters = CHEAP_SCREEN_ITERS,
    // Whether to fill `fitted_eta_var`. The per-row predictive variance costs
    // one back-solve per DISTINCT loading vector per cell (see nl_build_row_classes
    // above), which on a design with few repeated rows is the dominant cost of
    // a cell. A caller that reads only `fitted_eta` -- or only the marginal
    // summaries -- passes false and the pass is skipped outright; the returned
    // list then carries no `fitted_eta_var` element.
    bool compute_fitted_var = true,
    // Per-observation offset on the linear predictor (empty: none).
    const std::vector<double>& offset = std::vector<double>(),
    // Per-cell log hyperprior + log cell measure the cheap screen ranks with
    // (run_nested_laplace_grid); empty ranks on the log-marginal alone.
    const ScreenOffset& screen_log_offset = ScreenOffset(),
    // The fixed-effect prior; nullptr is the kernel default (DEFAULT_SIGMA_BETA).
    const BetaPrior* beta_prior = nullptr
) {
    int n_x = p + n_re_groups;
    for (const auto& b : blocks) {
        n_x = std::max(n_x, b.start + b.size);
    }
    // n_re_groups > 0 with an re_idx of the wrong length is a caller error, not
    // a configuration: silently dropping the block would fit a model with no
    // random effect and still return a log-marginal for it.
    if (n_re_groups > 0 && (int) re_idx.size() != N) {
        tulpa::stop("length(re_idx) (%d) must equal n_obs (%d) when "
                   "n_re_groups (%d) is positive.",
                   {(int) re_idx.size(), N, n_re_groups});
    }
    if (!offset.empty() && (int) offset.size() != N) {
        tulpa::stop("offset length (%d) must equal the number of "
                   "observations (%d).", {(int) offset.size(), N});
    }
    Rcpp::Nullable<Rcpp::NumericVector> arm_offset = R_NilValue;
    if (!offset.empty()) arm_offset = Rcpp::wrap(offset);
    // The joint driver's sparse Newton past SPARSE_THRESHOLD latents or with a
    // block whose prior scatters only sparsely; its dense Newton otherwise.
    return run_single_arm_block_joint(
        n_grid, y, n_trials, X, re_idx, N, p, n_re_groups, sigma_re,
        blocks, family, phi, max_iter, tol, n_threads, store_modes,
        x_init, store_Q,
        n_x >= SPARSE_THRESHOLD || blocks_have_sparse_only_prior(blocks),
        n_threads_outer, prune_tol, ext_spec, ext_response, progress, ckpt,
        compute_skew, skew_probe_idx, debias, cila, screen_iters,
        compute_fitted_var, arm_offset, screen_log_offset, beta_prior);
}

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_MULTI_H
