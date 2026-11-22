// laplace_block_fit.h
// Single-point Laplace fit over latent blocks the caller has built: fixed
// effects, an optional iid RE term held at `sigma_re`, and the blocks at fixed
// hyperparameters, solved by the dense spec solver. The cpp_laplace_fit_<field>
// R exports differ only in how they validate their inputs and build their
// blocks; the layout, the warm start, the solve and the result list are here.

#ifndef TULPA_LAPLACE_BLOCK_FIT_H
#define TULPA_LAPLACE_BLOCK_FIT_H

#include "laplace_spec_fit.h"      // SpecFamilyInputs, build_spec_family_inputs, unwrap_skew_idx
#include "laplace_spec_solve.h"    // laplace_mode_spec_dense_solve
#include "latent_block.h"
#include "nested_laplace_grid.h"   // nl_check_positive
#include <Rcpp.h>
#include <cmath>
#include <string>
#include <vector>

namespace tulpa {

// `blocks` occupy the n_block_latent latent slots after [beta | RE]; their
// callbacks may borrow caller locals, which outlive this call. A non-null
// `x_init` warm-starts the leading [beta | RE | blocks] latent slots, which sit
// contiguously before log_sigma_re. `offset` / `weights` are empty for "none".
inline Rcpp::List laplace_fit_blocks(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n,
    const Rcpp::NumericMatrix& X,
    const std::vector<int>& re_group, int n_re_groups, double sigma_re,
    const std::string& family, double phi,
    const std::vector<double>& offset, const std::vector<double>& weights,
    const std::vector<LatentBlock>& blocks, int n_block_latent,
    const Rcpp::Nullable<Rcpp::NumericVector>& x_init_nullable,
    int max_iter, double tol, int n_threads, int sparse_override,
    bool compute_skew, const Rcpp::Nullable<Rcpp::IntegerVector>& skew_idx
) {
    const bool has_re = n_re_groups > 0;
    const int block_start = X.ncol() + (has_re ? n_re_groups : 0);

    SpecFamilyInputs in;
    build_spec_family_inputs(
        in, y, n, X, re_group, n_re_groups, sigma_re, family, phi,
        /*sigma_beta=*/DEFAULT_SIGMA_BETA, n_block_latent,
        weights.empty() ? nullptr : weights.data(),
        offset.empty() ? nullptr : offset.data());

    std::vector<double> params(in.layout.total_params, 0.0);
    if (has_re) {
        nl_check_positive("sigma_re", sigma_re);
        params[in.layout.log_sigma_re_idx] = std::log(sigma_re);
    }
    if (x_init_nullable.isNotNull()) {
        Rcpp::NumericVector xi = Rcpp::as<Rcpp::NumericVector>(x_init_nullable);
        const int n_lat = block_start + n_block_latent;
        for (int j = 0; j < n_lat && j < (int)xi.size(); j++) params[j] = xi[j];
    }

    std::vector<int> skew_idx_vec;
    const std::vector<int>* skew_idx_ptr =
        unwrap_skew_idx(compute_skew, skew_idx, skew_idx_vec);

    LaplaceResult res = laplace_mode_spec_dense_solve(
        in.data, in.layout, params, in.re_group, max_iter, tol, n_threads,
        &blocks, /*k_grid=*/0, /*beta_prior=*/nullptr,
        /*return_re_cov=*/false, sparse_override,
        /*store_Q=*/false, compute_skew, skew_idx_ptr);
    return laplace_result_to_list(res);
}

} // namespace tulpa

#endif // TULPA_LAPLACE_BLOCK_FIT_H
