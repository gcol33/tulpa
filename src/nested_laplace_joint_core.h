// nested_laplace_joint_core.h
// Shared bookkeeping for joint multi-likelihood nested Laplace backends.
//
// Each backend (BYM2, ICAR, CAR_proper, ...) owns its own outer grid and
// spatial prior, but the per-arm beta/RE scatter and the cross-block
// Hessian assembly are identical. Those pieces live here so the backend
// kernels in nested_laplace_joint.cpp stay focused on what differs:
// latent layout (1 vs. 2 spatial sub-blocks), per-grid d-factors, and
// the spatial-prior add/log_prior callbacks.

#ifndef TULPA_NESTED_LAPLACE_JOINT_CORE_H
#define TULPA_NESTED_LAPLACE_JOINT_CORE_H

#include "laplace_core.h"
#include "laplace_newton_joint.h"
#include "laplace_re_priors.h"
#include "laplace_family_link.h"
#include <Rcpp.h>
#include <array>
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

// Per-arm linear-predictor evaluation. Same loop body for all backends; the
// only thing that changes across backends is `n_sub` (1 or 2) and the
// effective d-factors per sub-block (already multiplied by arm_scale).
inline void compute_arm_eta_joint_generic(
    const Rcpp::NumericVector& x,
    Rcpp::NumericVector&        eta_out,
    const ParsedArm&            pa,
    int                         N_k,
    int                         n_spatial_units,
    int                         n_sub,
    const std::array<int, 2>&   sub_starts,
    const std::array<double,2>& d_eff,
    int                         n_threads
) {
    int p_k = pa.p;
    int n_re_k = pa.n_re_groups;
    int bstart = pa.beta_start;
    int rstart = pa.re_start;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) \
        num_threads(n_threads > 0 ? n_threads : 1)
    #endif
    for (int i = 0; i < N_k; i++) {
        double e = 0.0;
        for (int j = 0; j < p_k; j++) e += pa.X(i, j) * x[bstart + j];
        if (n_re_k > 0) {
            int g = (int)pa.re_idx[i] - 1;
            if (g >= 0 && g < n_re_k) e += x[rstart + g];
        }
        if (n_spatial_units > 0) {
            int s = pa.spatial_idx[i] - 1;
            if (s >= 0 && s < n_spatial_units) {
                for (int b = 0; b < n_sub; b++) {
                    e += d_eff[b] * x[sub_starts[b] + s];
                }
            }
        }
        eta_out[i] = e;
    }
}

// Per-arm gradient/Hessian scatter for one arm at one grid point. Generic
// over the number of spatial sub-blocks (n_sub == 1 for ICAR/CAR_proper,
// n_sub == 2 for BYM2). The caller supplies effective d-factors d_eff[b]
// already multiplied by arm_scale (1.0 for the base arm, alpha for the
// copy arm).
inline void scatter_arm_obs_joint_generic(
    const Rcpp::NumericVector&  x,
    const Rcpp::NumericVector&  eta,
    const ParsedArm&            pa,
    const JointArm&             arm,
    int                         n_spatial_units,
    int                         n_sub,
    const std::array<int, 2>&   sub_starts,
    const std::array<double,2>& d_eff,
    DenseVec& grad, DenseMat& H
) {
    int p_k = pa.p;
    int n_re_k = pa.n_re_groups;
    int bstart = pa.beta_start;
    int rstart = pa.re_start;
    const std::string& family = arm.family;
    double phi_disp = arm.phi;

    for (int i = 0; i < arm.N; i++) {
        auto gh = grad_hess_for_family(
            arm.y[i], arm.n_trials[i], eta[i], family, phi_disp);

        int g_re = -1;
        if (n_re_k > 0) {
            int gi = (int)pa.re_idx[i] - 1;
            if (gi >= 0 && gi < n_re_k) g_re = rstart + gi;
        }
        std::array<int, 2> sub_idx = {-1, -1};
        if (n_spatial_units > 0) {
            int s = pa.spatial_idx[i] - 1;
            if (s >= 0 && s < n_spatial_units) {
                for (int b = 0; b < n_sub; b++) {
                    sub_idx[b] = sub_starts[b] + s;
                }
            }
        }

        // Beta block: gradient + diagonal-block Hessian
        for (int j = 0; j < p_k; j++) {
            double Xij = pa.X(i, j);
            grad[bstart + j] += gh.grad * Xij;
            for (int l = 0; l < p_k; l++) {
                H[bstart + j][bstart + l] += gh.neg_hess * Xij * pa.X(i, l);
            }
            if (g_re >= 0) {
                H[bstart + j][g_re] += gh.neg_hess * Xij;
                H[g_re][bstart + j] += gh.neg_hess * Xij;
            }
            for (int b = 0; b < n_sub; b++) {
                if (sub_idx[b] < 0) continue;
                H[bstart + j][sub_idx[b]] += gh.neg_hess * Xij * d_eff[b];
                H[sub_idx[b]][bstart + j] += gh.neg_hess * Xij * d_eff[b];
            }
        }

        // RE block: diagonal + cross with spatial sub-blocks
        if (g_re >= 0) {
            grad[g_re] += gh.grad;
            H[g_re][g_re] += gh.neg_hess;
            for (int b = 0; b < n_sub; b++) {
                if (sub_idx[b] < 0) continue;
                H[g_re][sub_idx[b]] += gh.neg_hess * d_eff[b];
                H[sub_idx[b]][g_re] += gh.neg_hess * d_eff[b];
            }
        }

        // Spatial block (within and cross sub-blocks, all at index s)
        for (int b1 = 0; b1 < n_sub; b1++) {
            if (sub_idx[b1] < 0) continue;
            grad[sub_idx[b1]] += gh.grad * d_eff[b1];
            for (int b2 = 0; b2 < n_sub; b2++) {
                if (sub_idx[b2] < 0) continue;
                H[sub_idx[b1]][sub_idx[b2]] +=
                    gh.neg_hess * d_eff[b1] * d_eff[b2];
            }
        }
    }
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
