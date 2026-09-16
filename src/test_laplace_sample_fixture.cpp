// test_laplace_sample_fixture.cpp
// Draws x = mode + L^{-T} z from a Gaussian precision H (Cholesky of
// H + LAPLACE_UNIFORM_RIDGE * I), the same uniform regularization every
// Laplace solve uses. No production door samples a bare (mode, H) pair --
// tulpa_posterior_draws() draws from a retained fixed-effect COVARIANCE
// instead (.nl_draw_chol()), never straight from a precision matrix -- so
// this is a numerical-correctness fixture for the generic sample-from-
// precision primitive, kept under the cpp_test_* naming used for the other
// fixtures rather than in the production kernel file.

#include "laplace_cholesky.h"
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_laplace_sample(
    Rcpp::NumericVector mode, Rcpp::NumericMatrix H, int n_samples
) {
    int n_x = mode.size();
    Rcpp::NumericMatrix samples(n_samples, n_x);

    // The ridge goes on a clone. Rcpp binds a REALSXP argument without
    // duplicating it, so ridging `H` in place writes the ridge into the R
    // matrix the caller still holds, and a second call on the same matrix
    // samples from a precision carrying the ridge twice.
    Rcpp::NumericMatrix Hr = Rcpp::clone(H);
    for (int j = 0; j < n_x; j++) Hr(j, j) += tulpa::LAPLACE_UNIFORM_RIDGE;
    Rcpp::NumericMatrix L(n_x, n_x);
    double log_det;
    tulpa::dense_cholesky_factorize(Hr, n_x, L, log_det);

    // Sample: z ~ N(0, I), x = mode + L^{-T} z
    for (int s = 0; s < n_samples; s++) {
        Rcpp::NumericVector z(n_x);
        for (int j = 0; j < n_x; j++) z[j] = R::rnorm(0.0, 1.0);

        // Solve L' x_centered = z (back substitution)
        Rcpp::NumericVector x_centered(n_x);
        for (int j = n_x - 1; j >= 0; j--) {
            double sum = z[j];
            for (int k = j + 1; k < n_x; k++) sum -= L(k, j) * x_centered[k];
            x_centered[j] = sum / L(j, j);
        }

        for (int j = 0; j < n_x; j++) samples(s, j) = mode[j] + x_centered[j];
    }
    return samples;
}
