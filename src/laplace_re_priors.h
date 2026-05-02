// laplace_re_priors.h
// Random-effect, fixed-effect, and centering helpers for Laplace solvers.

#ifndef TULPA_LAPLACE_RE_PRIORS_H
#define TULPA_LAPLACE_RE_PRIORS_H

#include "laplace_types.h"
#include <Rcpp.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {

inline void add_re_beta_priors(
    DenseVec& grad, DenseMat& H,
    const Rcpp::NumericVector& x,
    int p, int n_re_groups, double tau_re
) {
    for (int g = 0; g < n_re_groups; g++) {
        grad[p + g] -= tau_re * x[p + g];
        H[p + g][p + g] += tau_re;
    }
    double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) {
        grad[j] -= tau_beta * x[j];
        H[j][j] += tau_beta;
    }
}

inline void add_re_beta_priors(
    Rcpp::NumericVector& grad, Rcpp::NumericMatrix& H,
    const Rcpp::NumericVector& x,
    int p, int n_re_groups, double tau_re
) {
    for (int g = 0; g < n_re_groups; g++) {
        grad[p + g] -= tau_re * x[p + g];
        H(p + g, p + g) += tau_re;
    }
    double tau_beta = 1e-4;
    for (int j = 0; j < p; j++) {
        grad[j] -= tau_beta * x[j];
        H(j, j) += tau_beta;
    }
}

inline double compute_log_prior_re(
    const Rcpp::NumericVector& x, int p, int n_re_groups, double tau_re
) {
    double log_prior = 0.0;
    for (int g = 0; g < n_re_groups; g++) {
        log_prior += -0.5 * tau_re * x[p + g] * x[p + g];
    }
    if (n_re_groups > 0) {
        log_prior += 0.5 * n_re_groups * std::log(tau_re / (2.0 * M_PI));
    }
    return log_prior;
}

inline void center_effects(Rcpp::NumericVector& x, int start, int length) {
    if (length <= 0) return;
    double mean = 0.0;
    for (int i = 0; i < length; i++) mean += x[start + i];
    mean /= length;
    for (int i = 0; i < length; i++) x[start + i] -= mean;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_RE_PRIORS_H
