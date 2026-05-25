// laplace_spec_solve.h
// Spec-driven Laplace inner solve for any number of processes (np >= 1),
// exposed for reuse by the nested-Laplace outer-grid driver
// (nested_laplace_multi.h).
//
// `spec_inner_solve` is the one place the LikelihoodSpec helpers
// (compute_eta_spec / scatter_spec / total_log_lik_spec / log_prior_latent) are
// wrapped as the closures of the shared Newton loop (laplace_newton_solve_ll).
// It is np-agnostic: the eta buffer is sized N * np and the helpers handle the
// multi-process coupling internally, so the single-arm standalone caller, the
// outer-grid driver, and the multi-process (np >= 2) spec entry all run the same
// loop body -- a single source of truth (dev_notes/plans/clean_migration.md, Phase L). The
// implementation lives in laplace_spec.cpp where the anonymous-namespace spec
// helpers are visible.
//
// The caller owns the Newton scratch and the sparse solver, so the driver can
// pool one per outer-grid thread and, after the call returns, read the live
// Cholesky factor (left resident at the converged mode) for the per-row
// predictive variance without refactorising. base_params carries the pinned
// hyperparameter slots (e.g. log_sigma_re) and a latent warm start; the
// returned mode is in compacted [beta | RE | blocks] coordinates.

#ifndef TULPA_LAPLACE_SPEC_SOLVE_H
#define TULPA_LAPLACE_SPEC_SOLVE_H

#include "tulpa/likelihood.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include "laplace_cholesky.h"     // LaplaceResult
#include "laplace_newton.h"       // NewtonScratch
#include "latent_block.h"
#include "sparse_cholesky.h"
#include <utility>
#include <vector>

namespace tulpa {

LaplaceResult spec_inner_solve(
    const ModelData& data,
    const ParamLayout& layout,
    const std::vector<LatentBlock>* blocks,
    int k_grid,
    const LikelihoodSpec& spec,
    const void* response_data,
    const std::vector<int>& re_group_1based,
    int max_iter, double tol, int n_threads,
    const std::vector<double>& base_params,
    NewtonScratch& scratch,
    SparseCholeskySolver* solver,
    bool store_Q,
    const std::vector<std::pair<int, int>>* inv_block_layout
);

} // namespace tulpa

#endif // TULPA_LAPLACE_SPEC_SOLVE_H
