// nl_entry_inputs.h
// The response / design / control arguments every nested-Laplace outer-grid
// entry carries, and one runner per outer-grid driver.
//
// A grid entry differs from its siblings in its field: the structural
// fingerprint, the latent block(s), the hyperparameter axes it lays the grid
// over, and the axes it reports back. The rest of every entry -- the
// checkpoint, the probe-index resolve, the debias / CILA unwraps, and the
// driver's own argument block -- is the same at all of them and lives here, so
// the argument order a driver expects is written once.

#ifndef TULPA_NL_ENTRY_INPUTS_H
#define TULPA_NL_ENTRY_INPUTS_H

#include "latent_block.h"
#include "laplace_spec_fit.h"          // unwrap_skew_idx, DebiasRequest, CilaRequest
#include "nested_laplace_checkpoint.h" // GridCheckpoint, make_nl_grid_checkpoint
#include "nested_laplace_joint_core.h" // ParsedArm, JointArm
#include "nested_laplace_joint_multi.h"
#include "nested_laplace_multi.h"
#include <Rcpp.h>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tulpa {

// Optional warm-start mode as an owning vector (empty when absent).
inline Rcpp::NumericVector nl_unwrap_x_init(
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable
) {
    if (x_init_nullable.isNotNull()) {
        return Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
    }
    return Rcpp::NumericVector();
}

// The arguments every cpp_nested_laplace_* grid entry declares, under the names
// it declares them under: the response and design, the inner Newton budget, the
// warm start, the checkpoint, the diagnostics and their probe indices, the
// debias / CILA requests, and the outer-grid cheap-pass screening knobs.
// The runners below read the drivers' inputs from here.
struct NlEntryInputs {
    Rcpp::NumericVector y;
    Rcpp::IntegerVector n_trials;
    Rcpp::NumericMatrix X;
    Rcpp::NumericVector re_idx;
    int                 n_re_groups  = 0;
    double              sigma_re     = 1.0;
    std::string         family;
    double              phi          = 1.0;
    int                 max_iter     = 0;
    double              tol          = 0.0;
    int                 n_threads    = 1;
    Rcpp::NumericVector x_init;
    bool                store_Q      = false;
    std::string         checkpoint_path;
    bool                compute_skew = false;
    Rcpp::Nullable<Rcpp::IntegerVector> skew_idx = R_NilValue;
    Rcpp::Nullable<Rcpp::List>          debias   = R_NilValue;
    Rcpp::Nullable<Rcpp::List>          cila     = R_NilValue;
    // Carried only by the entries that take one. It moves every cell's mode, so
    // it enters the checkpoint fingerprint.
    Rcpp::Nullable<Rcpp::NumericVector> offset   = R_NilValue;
    // Outer-grid cheap-pass screening. A positive tolerance makes the driver
    // rank the lattice with a short warm-started inner Newton of
    // `screen_iters` steps per cell and skip the full solve on cells whose
    // screened weight falls below it; zero runs every cell. The screen skips
    // the per-row fitted-variance pass, which `compute_fitted_var` also
    // controls for the full solve.
    double prune_tol          = 0.0;
    int    screen_iters       = CHEAP_SCREEN_ITERS;
    bool   compute_fitted_var = true;

    int N() const { return static_cast<int>(y.size()); }
    int p() const { return X.ncol(); }
};

// The axes a fit reports back, attached in the order given.
using NlAxisOut = std::vector<std::pair<std::string, Rcpp::NumericVector>>;

inline void nl_attach_axes(Rcpp::List& out, const NlAxisOut& axes) {
    for (const auto& a : axes) out[a.first] = a.second;
}

// Checkpoint, probe indices and the two options unwraps, held together for the
// duration of one driver call: the driver takes borrowed pointers into all of
// them.
struct NlEntryRun {
    std::unique_ptr<GridCheckpoint> ckpt;
    std::vector<int>                skew_idx_vec;
    const std::vector<int>*         skew_idx_ptr = nullptr;
    DebiasRequest                   debias_req;
    CilaRequest                     cila_req;

    NlEntryRun(const NlEntryInputs& in, std::uint64_t struct_seed,
               const std::vector<Rcpp::NumericVector>& ckpt_axes)
        : ckpt(make_nl_grid_checkpoint(
              in.checkpoint_path, struct_seed, in.max_iter, in.tol,
              in.y, in.n_trials, in.X, in.re_idx, in.n_re_groups, in.sigma_re,
              in.family, in.phi, ckpt_axes, in.offset)),
          skew_idx_ptr(unwrap_skew_idx(in.compute_skew, in.skew_idx,
                                       skew_idx_vec)),
          debias_req(in.debias),
          cila_req(in.cila)
    {}
};

// LatentBlock outer-grid driver (icar / bym2 / car_proper / temporal).
inline Rcpp::List nl_run_multi_block_entry(
    const NlEntryInputs& in,
    int n_grid,
    std::uint64_t struct_seed,
    const std::vector<Rcpp::NumericVector>& ckpt_axes,
    const std::vector<LatentBlock>& blocks,
    const NlAxisOut& out_axes
) {
    NlEntryRun run(in, struct_seed, ckpt_axes);
    Rcpp::List out = run_multi_block_nested_laplace(
        n_grid, in.y, in.n_trials, in.X, in.re_idx,
        in.N(), in.p(), in.n_re_groups, in.sigma_re,
        blocks,
        in.family, in.phi, in.max_iter, in.tol, in.n_threads,
        /*store_modes=*/true, in.x_init,
        in.store_Q, /*n_threads_outer=*/1, in.prune_tol,
        /*ext_spec=*/nullptr, /*ext_response=*/nullptr,
        /*progress=*/nullptr, run.ckpt.get(),
        in.compute_skew, run.skew_idx_ptr,
        run.debias_req.ptr, run.cila_req.ptr,
        in.screen_iters, in.compute_fitted_var
    );
    nl_attach_axes(out, out_axes);
    return out;
}

// Joint-sparse outer-grid driver at one arm (nngp / hsgp): neither field's
// eta / scatter shape fits the INDEXED_SINGLE LatentBlock path, so each builds
// its own arm and reaches the sparse impl directly.
inline Rcpp::List nl_run_joint_sparse_entry(
    const NlEntryInputs& in,
    int n_grid,
    std::uint64_t struct_seed,
    const std::vector<Rcpp::NumericVector>& ckpt_axes,
    std::vector<JointArm>& arms,
    const std::vector<ParsedArm>& parsed,
    const std::vector<LatentBlock>& blocks,
    int n_x,
    const NlAxisOut& out_axes
) {
    NlEntryRun run(in, struct_seed, ckpt_axes);
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
        JointPDMode::LM, CurvatureMode::Observed,
        /*hessian_refresh=*/1, /*n_threads_outer=*/1,
        /*progress=*/nullptr, run.ckpt.get(),
        /*x_init_per_cell=*/std::vector<double>(),
        in.compute_skew, run.skew_idx_ptr,
        /*fixed_block=*/nullptr, run.debias_req.ptr, run.cila_req.ptr,
        in.screen_iters
    );
    nl_attach_axes(out, out_axes);
    return out;
}

} // namespace tulpa

// Collect the shared entry arguments into a tulpa::NlEntryInputs by the names
// every grid entry declares them under. Member-by-member assignment in one
// token sequence: an entry binds its own `max_iter` to `max_iter` and has no
// way to bind it to `n_threads`, and a member added to NlEntryInputs is filled
// at every entry at once. `offset` is assigned by the entries that carry one.
#define TULPA_NL_ENTRY_INPUTS                                              \
    ([&]() {                                                               \
        tulpa::NlEntryInputs nl_in_;                                       \
        nl_in_.y               = y;                                        \
        nl_in_.n_trials        = n;                                        \
        nl_in_.X               = X;                                        \
        nl_in_.re_idx          = re_idx;                                   \
        nl_in_.n_re_groups     = n_re_groups;                              \
        nl_in_.sigma_re        = sigma_re;                                 \
        nl_in_.family          = family;                                   \
        nl_in_.phi             = phi;                                      \
        nl_in_.max_iter        = max_iter;                                 \
        nl_in_.tol             = tol;                                      \
        nl_in_.n_threads       = n_threads;                                \
        nl_in_.x_init          = tulpa::nl_unwrap_x_init(x_init_nullable); \
        nl_in_.store_Q         = store_Q;                                  \
        nl_in_.checkpoint_path = checkpoint_path;                          \
        nl_in_.compute_skew    = compute_skew;                             \
        nl_in_.skew_idx        = skew_idx;                                 \
        nl_in_.debias          = debias;                                   \
        nl_in_.cila            = cila;                                     \
        nl_in_.prune_tol          = prune_tol;                             \
        nl_in_.screen_iters       = screen_iters;                          \
        nl_in_.compute_fitted_var = compute_fitted_var;                    \
        return nl_in_;                                                     \
    }())

#endif // TULPA_NL_ENTRY_INPUTS_H
