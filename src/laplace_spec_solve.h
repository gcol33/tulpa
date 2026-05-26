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

// Full Gaussian fixed-effect prior (per-coef mean + precision). Pointer-only
// in this declaration, so the forward decl suffices; full definition in
// laplace_re_priors.h (included by laplace_spec.cpp). A null beta_prior keeps
// the legacy scalar N(0, data.sigma_beta^2 I) ridge.
struct BetaPrior;

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
    const std::vector<std::pair<int, int>>* inv_block_layout,
    const BetaPrior* beta_prior = nullptr
);

// Result-returning standalone spec Laplace (defined in laplace_spec.cpp).
// Builds the [beta | RE | blocks] layout, runs spec_inner_solve, and returns the
// full LaplaceResult: compacted mode, log_marginal, and -- when return_re_cov --
// the per-(term,group) marginal covariance blocks (LaplaceResult.re_cov_flat)
// the EM M-step consumes. beta_prior overrides the scalar data.sigma_beta ridge
// with a full per-coef Gaussian. This is the entry the standalone single-point
// Laplace R exports (cpp_laplace_fit{,_multi_re,_spatial,_bym2}) route through.
LaplaceResult laplace_mode_spec_dense_solve(
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& params_inout,
    const std::vector<int>& re_group_1based,
    int max_iter, double tol, int n_threads,
    const std::vector<LatentBlock>* blocks = nullptr,
    int k_grid = 0,
    const BetaPrior* beta_prior = nullptr,
    bool return_re_cov = false
);

} // namespace tulpa

#endif // TULPA_LAPLACE_SPEC_SOLVE_H
