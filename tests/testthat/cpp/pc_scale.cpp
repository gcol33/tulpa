#include <RcppEigen.h>
#include "pg_shared.h"
#include "pg_spatial.h"

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_pc_scale(int n, double quad, int rank,
                                     double U, double alpha) {
    double sigma = 1.0;
    Rcpp::NumericVector out(n);
    for (int i = 0; i < n; ++i) {
        tulpa::pg_update_scale_pc(quad, rank, U, alpha, sigma);
        out[i] = std::log(sigma);
    }
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_bym2_rho(int n, Rcpp::NumericVector field,
    Rcpp::NumericVector iid, Rcpp::NumericVector omega,
    Rcpp::NumericVector residual, double a, double b) {
    double rho = 0.5;
    Rcpp::NumericVector out(n);
    for (int i = 0; i < n; ++i) {
        rho = tulpa::update_rho_bym2(field, iid, 1.0, 1.0,
                                    omega, residual, rho, a, b);
        out[i] = rho;
    }
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_bym2_sigma(int n, Rcpp::NumericVector field,
    Rcpp::NumericVector iid, Rcpp::NumericVector omega,
    Rcpp::NumericVector residual, double U, double alpha) {
    Rcpp::NumericVector out(n);
    for (int i = 0; i < n; ++i) {
        out[i] = tulpa::update_sigma_spatial_bym2(field, iid, 0.4, 1.0,
                                                omega, residual, U, alpha);
    }
    return out;
}
