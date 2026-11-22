// nested_laplace_entries.h
// The nested-Laplace grid entries, declared once.
//
// Each is an Rcpp::export function defined at global scope in
// nested_laplace.cpp, so it has no header of its own, and a caller inside the
// package that is not the generated RcppExports.cpp has to be given one. The
// defaults stay on the definitions, where Rcpp's attribute parser reads them to
// build the R wrappers; a caller reaching an entry through this header passes
// every argument, so the engine's front-door defaults are never inherited
// silently across the C-callable boundary.

#ifndef TULPA_NESTED_LAPLACE_ENTRIES_H
#define TULPA_NESTED_LAPLACE_ENTRIES_H

#include <Rcpp.h>
#include <string>

Rcpp::List cpp_nested_laplace_spde(
    Rcpp::NumericVector y,
    Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    Rcpp::NumericVector re_idx,
    int n_re_groups,
    double sigma_re,
    Rcpp::NumericVector A_x,
    Rcpp::IntegerVector A_i,
    Rcpp::IntegerVector A_p,
    int n_obs,
    int n_mesh,
    Rcpp::NumericVector C0_diag,
    Rcpp::NumericVector G1_x,
    Rcpp::IntegerVector G1_i,
    Rcpp::IntegerVector G1_p,
    Rcpp::NumericVector range_grid,
    Rcpp::NumericVector sigma_grid,
    double nu,
    std::string family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable,
    Rcpp::Nullable<Rcpp::NumericVector> rational_poles_nullable,
    Rcpp::Nullable<Rcpp::NumericVector> rational_weights_nullable,
    bool store_Q,
    std::string checkpoint_path,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable,
    bool compute_skew,
    Rcpp::Nullable<Rcpp::IntegerVector> skew_idx,
    Rcpp::Nullable<Rcpp::List> debias,
    Rcpp::Nullable<Rcpp::List> cila,
    double prune_tol,
    int screen_iters,
    bool compute_fitted_var,
    Rcpp::Nullable<Rcpp::NumericVector> screen_log_offset
,
    Rcpp::Nullable<Rcpp::List> beta_prior,
    bool progress,
    int progress_every,
    double progress_throttle,
    std::string progress_file, int n_threads_outer);

#endif // TULPA_NESTED_LAPLACE_ENTRIES_H
