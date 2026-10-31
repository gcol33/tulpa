// areal_input_check.cpp
// Definition of check_areal_adjacency, declared in areal_input_check.h.

#include <RcppEigen.h>
#include "areal_input_check.h"

namespace tulpa {

void check_areal_adjacency(
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

} // namespace tulpa
