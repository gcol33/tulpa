// areal_input_check.h
// One boundary check for the areal (ICAR / BYM2 / proper-CAR) entry points.
//
// Every one of them takes the CSR adjacency, the per-site neighbour count and
// the per-observation site index straight from R and hands them to the scatter,
// which uses all four as array offsets. One of those offsets indexes a WRITE:
// `H[sp_idx][spatial_start + adj_col_idx[k]] -= tau * rho` in
// laplace_spatial_priors.cpp, and DenseMat::operator[] is raw pointer
// arithmetic with no bound. So an out-of-range adjacency from a consumer that
// builds its own CSR, or a spatial_idx subset without a matching
// n_spatial_units, is heap corruption rather than an R-level error -- not
// diagnosable from R and not reproducible run to run.
//
// Index bases, both taken from how the kernels read them:
//   adj_col_idx  0-based, in [0, n_spatial_units)
//   spatial_idx  1-based, in [1, n_spatial_units]  (latent_block.h reads
//                x[start + idx(i, k) - 1])

#ifndef TULPA_AREAL_INPUT_CHECK_H
#define TULPA_AREAL_INPUT_CHECK_H

#include <Rcpp.h>

namespace tulpa {

// Validate the CSR adjacency and the per-site neighbour count. `who` names the
// calling entry point so the R-level message says which call was wrong.
inline void check_areal_adjacency(
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors,
    int n_spatial_units,
    const char* who
) {
    if (n_spatial_units < 1) {
        Rcpp::stop("%s: n_spatial_units (%d) must be at least 1.",
                   who, n_spatial_units);
    }
    if (static_cast<int>(adj_row_ptr.size()) != n_spatial_units + 1) {
        Rcpp::stop("%s: length(adj_row_ptr) (%d) must be n_spatial_units + 1 "
                   "(%d).", who, static_cast<int>(adj_row_ptr.size()),
                   n_spatial_units + 1);
    }
    if (static_cast<int>(n_neighbors.size()) != n_spatial_units) {
        Rcpp::stop("%s: length(n_neighbors) (%d) must equal n_spatial_units "
                   "(%d).", who, static_cast<int>(n_neighbors.size()),
                   n_spatial_units);
    }
    if (adj_row_ptr[0] != 0) {
        Rcpp::stop("%s: adj_row_ptr[1] (%d) must be 0 (CSR row pointers are "
                   "0-based).", who, adj_row_ptr[0]);
    }
    for (int s = 0; s < n_spatial_units; s++) {
        if (adj_row_ptr[s + 1] < adj_row_ptr[s]) {
            Rcpp::stop("%s: adj_row_ptr is not non-decreasing at site %d "
                       "(%d then %d).", who, s + 1, adj_row_ptr[s],
                       adj_row_ptr[s + 1]);
        }
    }
    if (adj_row_ptr[n_spatial_units] != static_cast<int>(adj_col_idx.size())) {
        Rcpp::stop("%s: adj_row_ptr[n_spatial_units + 1] (%d) must equal "
                   "length(adj_col_idx) (%d).", who,
                   adj_row_ptr[n_spatial_units],
                   static_cast<int>(adj_col_idx.size()));
    }
    for (int k = 0; k < static_cast<int>(adj_col_idx.size()); k++) {
        if (adj_col_idx[k] < 0 || adj_col_idx[k] >= n_spatial_units) {
            Rcpp::stop("%s: adj_col_idx[%d] (%d) is outside [0, %d). Neighbour "
                       "columns are 0-based.", who, k + 1, adj_col_idx[k],
                       n_spatial_units);
        }
    }
}

// Validate the per-observation site index. Separate from the adjacency check so
// an entry that carries no spatial_idx (or carries several) calls it per index
// vector.
inline void check_areal_site_index(
    const Rcpp::IntegerVector& spatial_idx,
    int n_obs,
    int n_spatial_units,
    const char* who
) {
    if (static_cast<int>(spatial_idx.size()) != n_obs) {
        Rcpp::stop("%s: length(spatial_idx) (%d) must equal the number of "
                   "observations (%d).", who,
                   static_cast<int>(spatial_idx.size()), n_obs);
    }
    for (int i = 0; i < n_obs; i++) {
        if (spatial_idx[i] < 1 || spatial_idx[i] > n_spatial_units) {
            Rcpp::stop("%s: spatial_idx[%d] (%d) is outside [1, %d]. Site "
                       "indices are 1-based.", who, i + 1, spatial_idx[i],
                       n_spatial_units);
        }
    }
}

// Both halves, for the common entry that takes one adjacency and one site index.
inline void check_areal_inputs(
    const Rcpp::IntegerVector& adj_row_ptr,
    const Rcpp::IntegerVector& adj_col_idx,
    const Rcpp::IntegerVector& n_neighbors,
    const Rcpp::IntegerVector& spatial_idx,
    int n_obs,
    int n_spatial_units,
    const char* who
) {
    check_areal_adjacency(adj_row_ptr, adj_col_idx, n_neighbors,
                          n_spatial_units, who);
    check_areal_site_index(spatial_idx, n_obs, n_spatial_units, who);
}

}  // namespace tulpa

#endif  // TULPA_AREAL_INPUT_CHECK_H
