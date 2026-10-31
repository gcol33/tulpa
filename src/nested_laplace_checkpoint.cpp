// nested_laplace_checkpoint.cpp
// Definitions of the outer-grid checkpoint declared in nested_laplace_checkpoint.h.

#include <RcppEigen.h>
#include "nested_laplace_checkpoint.h"

namespace tulpa {

std::unique_ptr<GridCheckpoint> make_nl_grid_checkpoint(
    const std::string& path,
    std::uint64_t struct_seed,
    int max_iter, double tol,
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X,
    const Rcpp::NumericVector& re_idx,
    int n_re_groups, double sigma_re,
    const std::string& family, double phi,
    const std::vector<Rcpp::NumericVector>& grid_axes,

    Rcpp::Nullable<Rcpp::NumericVector> offset,
    Rcpp::Nullable<Rcpp::NumericVector> weights) {
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
    fp.fold_rvec(y);
    fp.fold_rvec(n_trials);
    fp.fold_rvec(X);
    // The per-observation RE group assignment changes every cell's mode and
    // log_marginal, so it fingerprints alongside n_re_groups (the count).
    fp.fold_rvec(re_idx);
    fp.fold_rvec_nullable(offset);
    fp.fold_rvec_nullable(weights);
    // The grid's axis VALUES are not folded: a cell's key is its own exact
    // coordinate, so a cell solved on one grid is the same solve wherever else
    // that coordinate appears, and only fully solved cells are recorded (a
    // screened one never is). Folding them gave every grid of one fit its own
    // fingerprint, so the placement pass's recentred refit -- the second kernel
    // call on the same path -- refused the file its first call had just written,
    // and a front-door fit whose default axis recentred could not checkpoint at
    // all. The axis COUNT still separates layouts (a key's length is one double
    // per non-empty axis), as the multi-block entry's `axis_offsets` does.
    int n_grid = grid_axes.empty() ? 0 : static_cast<int>(grid_axes[0].size());
    CellKeyBuilder kb(n_grid);
    int n_axes = 0;
    for (const auto& ax : grid_axes) {
        // Skip a degenerate (empty) axis: rw1 / rw2 pass an empty rho_grid, and
        // reading n_grid doubles from ax.begin() would run off the buffer.
        if (ax.size()) {
            kb.add_axis(ax.begin());
            n_axes++;
        }
    }
    fp.fold_pod(n_axes);
    return std::unique_ptr<GridCheckpoint>(
        new GridCheckpoint(path, fp.value(), kb.take()));
}

} // namespace tulpa
