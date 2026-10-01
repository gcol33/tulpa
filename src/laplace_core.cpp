// laplace_core.cpp
// Core Laplace approximation engine for tulpa
// Implements Laplace approximation for latent Gaussian models.
// All model-specific Laplace functions run through the laplace_newton_solve()
// template in laplace_newton.h.

#include "laplace_core.h"
#include "laplace_newton.h"
#include "laplace_re_priors.h"
#include "laplace_spec_fit.h"     // spec-solver marshalling for the single-point fits
#include "laplace_multi_re_problem.h"
#include "re_structure.h"         // shared multi-term RE ModelData marshalling
#include "laplace_spatial_priors.h"
#include "laplace_temporal_priors.h"
#include "linalg_fast.h"
#include "gpu_nngp_laplace.h"
#include "sparse_hessian.h"
#include "hessian_pattern_guard.h"
#include "nested_laplace_grid.h"   // nl_check_positive
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// =====================================================================
// R exports
// =====================================================================

// `debias` is the subspace-debias request: a list carrying
// `idx` (the 1-based latent index set to correct by Metropolis along the
// Gaussian-conditional-mean surface) and the optional sweep budget `n_iter` /
// `warmup` / `thin`. An absent or empty index set leaves the solve bit-for-bit
// as it was and consumes no random number.
// [[Rcpp::export]]
Rcpp::List cpp_laplace_fit_multi_re(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::List re_idx_list,
    Rcpp::IntegerVector re_ngroups,
    Rcpp::List re_sigma_list,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads = 1,
    Rcpp::Nullable<Rcpp::List> re_Z_list = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> re_ncoefs = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> weights = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> offset = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> x_init = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> beta_prior_mean = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> beta_prior_sd = R_NilValue,
    bool return_re_cov = false,
    double phi2 = NA_REAL,
    Rcpp::Nullable<Rcpp::NumericMatrix> X_zi = R_NilValue,
    double zi_prior_sd = 2.5,
    bool return_joint_hessian = false,
    bool compute_skew = false,
    Rcpp::Nullable<Rcpp::IntegerVector> skew_idx = R_NilValue,
    Rcpp::Nullable<Rcpp::List> debias = R_NilValue
) {
    // Multi-term RE (intercept / slopes / correlated) + built-in family through
    // the unified spec solver. The marshalling into the multi-term ModelData /
    // ParamLayout -- weights, offset, the full per-coef beta prior, zero
    // inflation -- is MultiReProblem's; this entry adds the one covariance, the
    // warm start, and the extras a single fit can ask for.
    tulpa::MultiReProblem prob(
        y, n, X, re_idx_list, re_ngroups, re_sigma_list, family, phi,
        re_Z_list, re_ncoefs, weights, offset, beta_prior_mean, beta_prior_sd,
        phi2, X_zi, zi_prior_sd);
    std::vector<double> params = prob.params_at(re_sigma_list, x_init);

    std::vector<int> re_group_empty;   // groups come from re_group_multi_flat
    std::vector<int> skew_idx_vec;
    const std::vector<int>* skew_idx_ptr =
        tulpa::unwrap_skew_idx(compute_skew, skew_idx, skew_idx_vec);
    tulpa::SubspaceDebiasOptions db_opts;
    const tulpa::SubspaceDebiasOptions* db_ptr =
        tulpa::unwrap_debias(debias, db_opts);
    tulpa::LaplaceResult res = tulpa::laplace_mode_spec_dense_solve(
        prob.data(), prob.layout(), params, re_group_empty, max_iter, tol,
        n_threads, /*blocks=*/nullptr, /*k_grid=*/0, prob.beta_prior(),
        return_re_cov, /*sparse_override=*/0, return_joint_hessian,
        compute_skew, skew_idx_ptr, db_ptr);
    return tulpa::laplace_result_to_list(res);
}

// The Laplace log-marginal of ONE multi-term random-effect model at S
// covariances, `re_sigma_batch[[s]]` being the s-th point's `re_sigma_list`.
// This is what an importance batch over the covariance hyperparameters pays:
// S inner solves of a model whose data, designs and priors never change, so
// the problem is marshalled once and the solves share it.
//
// Every point starts its Newton iteration from the same `x_init` (the mode at
// the proposal's centre, for an importance batch) rather than from its
// neighbour's mode, so a point's value does not depend on which thread solved
// it or in what order: the batch returns the same vector at any width. Each
// solve converges to the same mode its cold start would, to the solver's own
// tolerance.
//
// A point is -Inf where its covariance cannot be converted (a non-finite or
// non-positive SD from an overflowing draw), where its solve throws, or where
// the log-marginal it reaches is not finite: the zero-weight convention every
// importance target in the engine follows, rather than an error that would
// discard the other S - 1 values. The points run across `n_threads_outer`
// threads, each solve single-threaded, with one Newton scratch and one sparse
// solver per thread.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_laplace_log_marginal_multi_re_batch(
    Rcpp::NumericVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::List re_idx_list,
    Rcpp::IntegerVector re_ngroups,
    Rcpp::List re_sigma_batch,
    std::string family, double phi = 1.0,
    int max_iter = 100, double tol = 1e-6, int n_threads_outer = 1,
    Rcpp::Nullable<Rcpp::List> re_Z_list = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> re_ncoefs = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> weights = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> offset = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> x_init = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> beta_prior_mean = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> beta_prior_sd = R_NilValue,
    double phi2 = NA_REAL,
    Rcpp::Nullable<Rcpp::NumericMatrix> X_zi = R_NilValue,
    double zi_prior_sd = 2.5
) {
    const int S = re_sigma_batch.size();
    const double neg_inf = -std::numeric_limits<double>::infinity();
    Rcpp::NumericVector out(S, neg_inf);
    if (S == 0) return out;

    tulpa::MultiReProblem prob(
        y, n, X, re_idx_list, re_ngroups,
        Rcpp::as<Rcpp::List>(re_sigma_batch[0]), family, phi,
        re_Z_list, re_ncoefs, weights, offset, beta_prior_mean, beta_prior_sd,
        phi2, X_zi, zi_prior_sd);
    const int n_x = tulpa::laplace_spec_dense_check(prob.data(), prob.layout());

    // The per-point parameter vectors are built here, on the main thread: the
    // conversion reports a bad covariance through R, which is a zero-weight
    // point for this batch and an R error nowhere.
    std::vector<std::vector<double>> params(S);
    std::vector<char> usable(S, 0);
    for (int s = 0; s < S; s++) {
        try {
            params[s] = prob.params_at(Rcpp::as<Rcpp::List>(re_sigma_batch[s]),
                                       x_init);
            usable[s] = 1;
        } catch (std::exception&) {
            usable[s] = 0;
        }
    }

    const int n_outer = std::max(1, std::min(n_threads_outer, S));
    const int n_eta = prob.data().N * prob.data().n_processes;
    std::vector<tulpa::NewtonScratch> scratch_pool(n_outer);
    for (auto& sc : scratch_pool) sc.allocate(n_x, n_eta);
    std::vector<tulpa::SparseCholeskySolver> solver_pool(n_outer);
    std::vector<double> lm(S, neg_inf);
    const std::vector<int> re_group_empty;   // groups come from re_group_multi_flat

    const tulpa::HessianPatternGuard pattern_guard;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(n_outer) \
        if (n_outer > 1)
    #endif
    for (int s = 0; s < S; s++) {
        if (!usable[s]) continue;
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_in_parallel() ? omp_get_thread_num() : 0;
        #endif
        try {
            tulpa::LaplaceResult r = tulpa::spec_inner_solve(
                prob.data(), prob.layout(), /*blocks=*/nullptr, /*k_grid=*/0,
                prob.spec(), prob.response(), re_group_empty, max_iter, tol,
                /*n_threads=*/1, params[s], scratch_pool[tid],
                &solver_pool[tid], /*store_Q=*/false,
                /*inv_block_layout=*/nullptr, prob.beta_prior());
            if (std::isfinite(r.log_marginal)) lm[s] = r.log_marginal;
        } catch (...) {
            lm[s] = neg_inf;
        }
    }
    pattern_guard.check("the multi-term random-effect log-marginal batch");

    for (int s = 0; s < S; s++) out[s] = lm[s];
    return out;
}

// Spatial / BYM2 / RSR mode finders and their R exports live in
// laplace_core_spatial.cpp, the GP / multiscale GP / multiscale temporal ones in
// laplace_core_gp.cpp, and nested Laplace / SPDE in nested_laplace.cpp and
// spde_laplace.cpp.
