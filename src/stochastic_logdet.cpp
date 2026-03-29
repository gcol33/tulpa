// stochastic_logdet.cpp
// Rcpp export for stochastic log-determinant estimation

#include "stochastic_logdet.h"
#include <Rcpp.h>

// [[Rcpp::export]]
double cpp_stochastic_log_determinant(
    Rcpp::NumericVector Q_x, Rcpp::IntegerVector Q_i, Rcpp::IntegerVector Q_p,
    int n, int n_probes = 30, int n_lanczos = 50, int seed = 42
) {
    std::vector<int> col_ptr(Q_p.begin(), Q_p.end());
    std::vector<int> row_idx(Q_i.begin(), Q_i.end());
    std::vector<double> values(Q_x.begin(), Q_x.end());

    return tulpa::stochastic_log_determinant(
        col_ptr, row_idx, values, n,
        n_probes, n_lanczos, static_cast<unsigned int>(seed)
    );
}
