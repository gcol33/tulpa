// nested_laplace_joint_core.h
// Shared bookkeeping for joint multi-likelihood nested Laplace backends.
//
// The per-arm latent-vector layout (parsed[k].beta_start, .re_start, .p,
// .n_re_groups, sigma_re, tau_re) and the per-arm β/RE priors are factored
// here so every joint backend can reuse them. The inner Newton itself
// lives in nested_laplace_joint_multi.h, which consumes the
// ParsedArm/JointArm vectors built by parse_joint_arms() below alongside a
// std::vector<LatentBlock> describing the spatial prior(s).

#ifndef TULPA_NESTED_LAPLACE_JOINT_CORE_H
#define TULPA_NESTED_LAPLACE_JOINT_CORE_H

#include "laplace_core.h"
#include "laplace_newton_joint.h"
#include "laplace_re_priors.h"
#include <Rcpp.h>
#include <vector>

namespace tulpa {

// One arm parsed out of the R-side `arms_list`. Owns its Rcpp objects so
// the per-grid-point lambdas can capture references safely.
struct ParsedArm {
    Rcpp::NumericMatrix X;            // [N_arm x p_arm], column-major
    Rcpp::NumericVector re_idx;       // [N_arm], 1-based; 0/NA -> no RE
    Rcpp::IntegerVector spatial_idx;  // [N_arm], 1-based
    int                 p;
    int                 n_re_groups;
    double              sigma_re;
    int                 beta_start;   // offset into joint x where this arm's beta begins
    int                 re_start;     // offset for this arm's RE block
    double              tau_re;
};

// Parse the R-side arms_list into ParsedArm + JointArm vectors and assign
// (beta_start, re_start) offsets. Returns the running latent dimension
// after all per-arm blocks; the caller appends the spatial block(s).
inline int parse_joint_arms(
    const Rcpp::List& arms_list,
    std::vector<ParsedArm>& parsed_out,
    std::vector<JointArm>&  arms_out
) {
    int n_arms = arms_list.size();
    if (n_arms < 1) Rcpp::stop("arms_list must contain at least one arm.");

    parsed_out.assign(n_arms, ParsedArm{});
    arms_out.assign(n_arms,   JointArm{});

    int n_x_running = 0;
    for (int k = 0; k < n_arms; k++) {
        Rcpp::List a = arms_list[k];
        ParsedArm& pa = parsed_out[k];
        pa.X           = Rcpp::as<Rcpp::NumericMatrix>(a["X"]);
        pa.re_idx      = Rcpp::as<Rcpp::NumericVector>(a["re_idx"]);
        pa.spatial_idx = Rcpp::as<Rcpp::IntegerVector>(a["spatial_idx"]);
        pa.p           = pa.X.ncol();
        pa.n_re_groups = Rcpp::as<int>(a["n_re_groups"]);
        pa.sigma_re    = Rcpp::as<double>(a["sigma_re"]);
        pa.tau_re      = (pa.n_re_groups > 0)
                         ? 1.0 / (pa.sigma_re * pa.sigma_re + 1e-10)
                         : 0.0;
        pa.beta_start  = n_x_running;
        n_x_running   += pa.p;

        arms_out[k].y        = Rcpp::as<Rcpp::NumericVector>(a["y"]);
        arms_out[k].n_trials = Rcpp::as<Rcpp::IntegerVector>(a["n_trials"]);
        arms_out[k].family   = Rcpp::as<std::string>(a["family"]);
        arms_out[k].phi      = Rcpp::as<double>(a["phi"]);
        arms_out[k].N        = (int)arms_out[k].y.size();

        int N_k = arms_out[k].N;
        if ((int)pa.X.nrow() != N_k) {
            Rcpp::stop("Arm %d: nrow(X) (%d) != length(y) (%d).",
                       k + 1, (int)pa.X.nrow(), N_k);
        }
        if ((int)arms_out[k].n_trials.size() != N_k) {
            Rcpp::stop("Arm %d: length(n_trials) (%d) != length(y) (%d).",
                       k + 1, (int)arms_out[k].n_trials.size(), N_k);
        }
        if ((int)pa.re_idx.size() != N_k) {
            Rcpp::stop("Arm %d: length(re_idx) (%d) != length(y) (%d).",
                       k + 1, (int)pa.re_idx.size(), N_k);
        }
        if ((int)pa.spatial_idx.size() != N_k) {
            Rcpp::stop("Arm %d: length(spatial_idx) (%d) != length(y) (%d).",
                       k + 1, (int)pa.spatial_idx.size(), N_k);
        }
    }
    // Second pass: REs come after all betas.
    for (int k = 0; k < n_arms; k++) {
        parsed_out[k].re_start = n_x_running;
        n_x_running += parsed_out[k].n_re_groups;
    }
    return n_x_running;
}

// Per-arm Gaussian beta + RE priors. Identical across all backends.
// `tau_beta` is a weak prior precision (default 1e-4 ≡ sd ~ 100).
inline void add_per_arm_beta_re_priors(
    DenseVec& grad, DenseMat& H,
    const Rcpp::NumericVector& x,
    const std::vector<ParsedArm>& parsed,
    double tau_beta = 1e-4
) {
    for (const ParsedArm& pa : parsed) {
        for (int j = 0; j < pa.p; j++) {
            grad[pa.beta_start + j] -= tau_beta * x[pa.beta_start + j];
            H[pa.beta_start + j][pa.beta_start + j] += tau_beta;
        }
        for (int g = 0; g < pa.n_re_groups; g++) {
            grad[pa.re_start + g] -= pa.tau_re * x[pa.re_start + g];
            H[pa.re_start + g][pa.re_start + g] += pa.tau_re;
        }
    }
}

// Per-arm RE log-prior contribution. Matches single-arm convention: weak
// beta prior is dropped from log_prior so the joint log-marginal stays
// comparable to two single-arm fits.
inline double log_prior_per_arm_re(const Rcpp::NumericVector& x,
                                    const std::vector<ParsedArm>& parsed) {
    constexpr double LOG_2PI = 1.8378770664093454835606594728112;
    double lp = 0.0;
    for (const ParsedArm& pa : parsed) {
        for (int g = 0; g < pa.n_re_groups; g++) {
            double v = x[pa.re_start + g];
            lp -= 0.5 * pa.tau_re * v * v;
        }
        if (pa.n_re_groups > 0) {
            lp += 0.5 * pa.n_re_groups * (std::log(pa.tau_re) - LOG_2PI);
        }
    }
    return lp;
}

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_JOINT_CORE_H
