// array_utils.cpp
// Utility functions for array layout transformations between R and C++

#include <Rcpp.h>
#include "array_utils.h"

// [[Rcpp::export]]
Rcpp::NumericVector cpp_flatten_3d_rowmajor(
    Rcpp::NumericVector arr,
    int d1, int d2, int d3
) {
    std::vector<double> result = tulpa::flatten_3d_rowmajor(arr, d1, d2, d3);
    return Rcpp::wrap(result);
}
