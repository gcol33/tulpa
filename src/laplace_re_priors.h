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

// The weak default fixed-effect prior, beta_j ~ N(0, 100^2), in the two
// parameterizations the kernels read it in: the nested-Laplace prior helpers
// take a precision, the spec path takes an SD on ModelData::sigma_beta. Both
// are defined here so a change to the default cannot move only one of them.
inline constexpr double DEFAULT_TAU_BETA   = 1e-4;
inline constexpr double DEFAULT_SIGMA_BETA = 100.0;
static_assert(DEFAULT_SIGMA_BETA * DEFAULT_SIGMA_BETA * DEFAULT_TAU_BETA
                  - 1.0 < 1e-12 &&
              1.0 - DEFAULT_SIGMA_BETA * DEFAULT_SIGMA_BETA * DEFAULT_TAU_BETA
                  < 1e-12,
              "DEFAULT_SIGMA_BETA and DEFAULT_TAU_BETA must describe the same "
              "prior: tau = 1 / sigma^2.");

// Floor added to sigma_re^2 before inverting it into a precision, so a zero or
// near-zero random-effect SD yields a large but finite tau_re instead of a
// division by zero.
inline constexpr double RE_VARIANCE_FLOOR = 1e-10;

// Gaussian prior on the fixed effects beta. With both vectors empty it
// reproduces the historical weak prior beta_j ~ N(0, 1e4) applied
// uniformly. A non-empty `tau` overrides the precision (1/sd^2) per
// coefficient; a non-empty `mean` shifts the prior mean per coefficient.
// When set, each vector must have length p -- the R layer recycles scalar
// mean/sd to full length before constructing this.
struct BetaPrior {
    std::vector<double> mean;  // length p, or empty -> 0
    std::vector<double> tau;   // length p precision, or empty -> DEFAULT_TAU_BETA

    double tau_at(int j) const { return tau.empty() ? DEFAULT_TAU_BETA : tau[j]; }
    double mean_at(int j) const { return mean.empty() ? 0.0 : mean[j]; }
};

// Sink accessors, so the penalty below is written once for both the DenseMat /
// DenseVec pair the sparse solvers carry and the Rcpp pair the dense ones do.
// Adding a third sink needs one more pair of overloads here.
inline void prior_sub_grad(DenseVec& g, int j, double v)            { g[j] -= v; }
inline void prior_sub_grad(Rcpp::NumericVector& g, int j, double v) { g[j] -= v; }
inline void prior_add_diag(DenseMat& H, int j, double v)            { H[j][j] += v; }
inline void prior_add_diag(Rcpp::NumericMatrix& H, int j, double v) { H(j, j) += v; }

// grad -= tau * (beta - mean); H += tau on the diagonal. Single source of
// truth for the fixed-effect Gaussian penalty across every Laplace solver.
template <class VecLike, class MatLike>
inline void add_beta_prior(
    VecLike& grad, MatLike& H,
    const Rcpp::NumericVector& x, int p, const BetaPrior& bp
) {
    for (int j = 0; j < p; j++) {
        double tau = bp.tau_at(j);
        prior_sub_grad(grad, j, tau * (x[j] - bp.mean_at(j)));
        prior_add_diag(H, j, tau);
    }
}

// Quadratic log-prior contribution -0.5 * sum tau_j * (beta_j - mean_j)^2.
// Normalizing constant omitted (it is fixed when tau is fixed, so it
// cancels in any model comparison at fixed structure) -- matching the
// historical inline behaviour.
inline double log_prior_beta(
    const Rcpp::NumericVector& x, int p, const BetaPrior& bp
) {
    double lp = 0.0;
    for (int j = 0; j < p; j++) {
        double d = x[j] - bp.mean_at(j);
        lp -= 0.5 * bp.tau_at(j) * d * d;
    }
    return lp;
}

// The RE-group diagonal beside the fixed-effect penalty, same two sinks.
template <class VecLike, class MatLike>
inline void add_re_beta_priors(
    VecLike& grad, MatLike& H,
    const Rcpp::NumericVector& x,
    int p, int n_re_groups, double tau_re
) {
    for (int g = 0; g < n_re_groups; g++) {
        prior_sub_grad(grad, p + g, tau_re * x[p + g]);
        prior_add_diag(H, p + g, tau_re);
    }
    add_beta_prior(grad, H, x, p, BetaPrior());
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

// Subtract the mean from x[start, start+length), returning the mean that
// was applied. Single-arm callers ignore the return value; joint drivers
// use it to shift per-arm intercepts so eta is preserved when a rank-
// deficient block is re-centered after a Newton step.
inline double center_effects(Rcpp::NumericVector& x, int start, int length) {
    if (length <= 0) return 0.0;
    double mean = 0.0;
    for (int i = 0; i < length; i++) mean += x[start + i];
    mean /= length;
    for (int i = 0; i < length; i++) x[start + i] -= mean;
    return mean;
}

} // namespace tulpa

#endif // TULPA_LAPLACE_RE_PRIORS_H
