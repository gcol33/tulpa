// nested_laplace_checkpoint.h
// Grid-cell checkpoint/resume for the nested-Laplace outer grid (gcol33/tulpa#50).
//
// The outer grid is the natural checkpoint boundary: each cell over the
// hyperparameter coordinate is independent and produces a self-contained
// LaplaceResult. A killed or interrupted fit can resume from the last completed
// cell instead of restarting, which matters for EVA-scale fits that run for
// hours. The on-disk format, load/append/torn-tail logic, fingerprinting, and
// per-cell key construction are generic and live in checkpoint_io.h; this file
// is the LaplaceResult specialization (GridCheckpoint) plus the single-arm
// kernel checkpoint factory.
//
// Keying is by the cell's hyperparameter coordinate (the theta-grid row plus any
// per-arm phi value), not the integer index, so an adaptive-grid refinement pass
// that appends newly spawned cells stores them under their own keys and a resume
// hits every previously completed cell regardless of visit order. "Fresh vs
// resume" is decided by the R front door (it removes the file before the first
// kernel call when resume = FALSE); at this layer a present, matching file is
// always loaded, so the several kernel calls within one fit share it.

#ifndef TULPA_NESTED_LAPLACE_CHECKPOINT_H
#define TULPA_NESTED_LAPLACE_CHECKPOINT_H

#include "checkpoint_io.h"
#include "laplace_core.h"
#include <memory>
#include <string>
#include <vector>

namespace tulpa {

// LaplaceResult payload (de)serialization -- the customization points the
// generic CheckpointLog finds by ADL. Every field a resumed cell must reproduce
// bit-for-bit is written, so a loaded cell is indistinguishable from a freshly
// solved one.
inline std::string ckpt_serialize(const LaplaceResult& r) {
    std::string buf;
    ckpt_put(buf, r.log_marginal);
    ckpt_put(buf, r.log_det_Q);
    ckpt_put<std::int32_t>(buf, r.n_iter);
    ckpt_put<std::uint8_t>(buf, r.converged ? 1u : 0u);
    ckpt_put_span(buf, r.mode);
    ckpt_put<std::int32_t>(buf, r.Q_csc_n);
    ckpt_put_span(buf, r.Q_csc_p);
    ckpt_put_span(buf, r.Q_csc_i);
    ckpt_put_span(buf, r.Q_csc_x);
    ckpt_put_span(buf, r.re_cov_flat);
    ckpt_put_span(buf, r.re_cov_block_sizes);
    return buf;
}

inline bool ckpt_deserialize(CkptReader& rd, LaplaceResult& r) {
    r.log_marginal       = rd.get<double>();
    r.log_det_Q          = rd.get<double>();
    r.n_iter             = rd.get<std::int32_t>();
    r.converged          = (rd.get<std::uint8_t>() != 0);
    r.mode               = rd.get_span<double>();
    r.Q_csc_n            = rd.get<std::int32_t>();
    r.Q_csc_p            = rd.get_span<int>();
    r.Q_csc_i            = rd.get_span<int>();
    r.Q_csc_x            = rd.get_span<double>();
    r.re_cov_flat        = rd.get_span<double>();
    r.re_cov_block_sizes = rd.get_span<int>();
    return rd.ok;
}

using GridCheckpoint = CheckpointLog<LaplaceResult>;

// Build a checkpoint for a single-arm nested-Laplace kernel (the
// run_multi_block_nested_laplace / sparse_impl callers: icar, bym2, car_proper,
// temporal, the ST variants, nngp, hsgp). `struct_seed` is a per-kernel
// fingerprint of everything NOT visible here -- the kernel tag plus its latent
// structure (adjacency, coords / neighbour graph, temporal layout) -- folded by
// the caller; this helper then folds the shared observation inputs and the grid
// axes on top. Keys are the per-cell coordinate over `grid_axes` (each a
// length-n_grid column; multi-axis grids are pre-expanded to n_grid R-side).
// Returns nullptr when `path` is empty, so a caller wires it unconditionally.
inline std::unique_ptr<GridCheckpoint> make_nl_grid_checkpoint(
    const std::string& path,
    std::uint64_t struct_seed,
    int max_iter, double tol,
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X,
    int n_re_groups, double sigma_re,
    const std::string& family, double phi,
    const std::vector<Rcpp::NumericVector>& grid_axes)
{
    if (path.empty()) return nullptr;
    Fingerprint fp;
    fp.h = struct_seed;
    fp.fold_pod(max_iter);
    fp.fold_pod(tol);
    fp.fold_pod(static_cast<int>(y.size()));
    fp.fold_pod(static_cast<int>(X.ncol()));
    fp.fold_pod(n_re_groups);
    fp.fold_pod(sigma_re);
    fp.fold_str(family);
    fp.fold_pod(phi);
    if (y.size())        fp.fold(y.begin(), (std::size_t)y.size() * sizeof(double));
    if (n_trials.size()) fp.fold(n_trials.begin(),
                                 (std::size_t)n_trials.size() * sizeof(int));
    if (X.size())        fp.fold(X.begin(), (std::size_t)X.size() * sizeof(double));
    int n_grid = grid_axes.empty() ? 0 : static_cast<int>(grid_axes[0].size());
    CellKeyBuilder kb(n_grid);
    for (const auto& ax : grid_axes) {
        if (ax.size()) fp.fold(ax.begin(), (std::size_t)ax.size() * sizeof(double));
        kb.add_axis(ax.begin());
    }
    return std::unique_ptr<GridCheckpoint>(
        new GridCheckpoint(path, fp.value(), kb.take()));
}

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_CHECKPOINT_H
