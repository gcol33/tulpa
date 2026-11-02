// nl_entry_inputs.cpp
// Definition of nl_run_joint_sparse_entry, declared in nl_entry_inputs.h.

#include <RcppEigen.h>
#include "nl_entry_inputs.h"

namespace tulpa {

Rcpp::List nl_run_joint_sparse_entry(
    const NlEntryInputs& in,
    int n_grid,
    std::uint64_t struct_seed,
    const std::vector<Rcpp::NumericVector>& ckpt_axes,
    std::vector<JointArm>& arms,
    std::vector<ParsedArm>& parsed,
    const std::vector<LatentBlock>& blocks,
    int n_x,
    const NlAxisOut& out_axes
) {
    NlEntryRun run(in, struct_seed, ckpt_axes);
    for (auto& pa : parsed) apply_beta_prior(pa, in.beta_prior);
    Rcpp::List out = run_multi_block_nested_laplace_joint_sparse_impl(
        n_grid, arms, parsed, blocks, n_x,
        in.max_iter, in.tol, in.n_threads,
        /*store_modes=*/true, in.x_init, in.store_Q,
        /*prep_at_grid=*/nullptr,
        /*tile_ids=*/std::vector<int>(),
        /*tile_pilot_cells=*/std::vector<int>(),
        in.prune_tol,
        /*cell_coupling_spec=*/nullptr,
        /*coupled_arms=*/std::vector<int>(),
        /*cell_rows=*/std::vector<std::vector<std::vector<int>>>(),
        /*n_cells=*/0,
        JointPDMode::LM, StepCurvature::Observed,
        /*hessian_refresh=*/1, /*n_threads_outer=*/1,
        /*progress=*/nullptr, run.ckpt.get(),
        /*x_init_per_cell=*/std::vector<double>(),
        in.compute_skew, run.skew_idx_ptr,
        /*fixed_block=*/nullptr, run.debias_req.ptr, run.cila_req.ptr,
        in.screen_iters, in.compute_fitted_var, in.screen_offset()
    );
    nl_attach_fitted_eta_single_arm(out, arms, parsed, blocks);
    nl_attach_axes(out, out_axes);
    return out;
}

} // namespace tulpa
