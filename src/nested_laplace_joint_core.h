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
#include "sparse_hessian.h"
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
    // Optional per-coefficient Gaussian beta prior (length p each). When set,
    // overrides the scalar `tau_beta` default in the prior helpers below: the
    // consumer supplies an informative intercept prior to break the occupancy
    // psi-p identifiability ridge on the coupled arm. Empty -> fall back to the
    // weak scalar default (back-compatible).
    Rcpp::NumericVector beta_prior_mean;
    Rcpp::NumericVector beta_prior_prec;
    // Optional per-observation fixed offset on this arm's linear predictor
    // (eta = offset + X beta + RE + blocks). Length N_arm when set; empty ->
    // no offset (treated as zero). Read by every joint compute_eta site.
    Rcpp::NumericVector offset;
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

        // Optional per-coefficient beta prior (mean + precision). Length must
        // match pa.p when supplied; absence leaves the empty vectors that the
        // prior helpers read as "use the scalar tau_beta default".
        if (a.containsElementNamed("beta_prior_mean")) {
            pa.beta_prior_mean =
                Rcpp::as<Rcpp::NumericVector>(a["beta_prior_mean"]);
            if ((int)pa.beta_prior_mean.size() != pa.p) {
                Rcpp::stop("Arm %d: length(beta_prior_mean) (%d) != p (%d).",
                           k + 1, (int)pa.beta_prior_mean.size(), pa.p);
            }
        }
        if (a.containsElementNamed("beta_prior_prec")) {
            pa.beta_prior_prec =
                Rcpp::as<Rcpp::NumericVector>(a["beta_prior_prec"]);
            if ((int)pa.beta_prior_prec.size() != pa.p) {
                Rcpp::stop("Arm %d: length(beta_prior_prec) (%d) != p (%d).",
                           k + 1, (int)pa.beta_prior_prec.size(), pa.p);
            }
        }
        // Optional per-observation fixed offset on the linear predictor. Length
        // must match this arm's N (validated after arms_out[k].N is set below).
        if (a.containsElementNamed("offset") &&
            !Rf_isNull(a["offset"])) {
            pa.offset = Rcpp::as<Rcpp::NumericVector>(a["offset"]);
        }

        arms_out[k].y        = Rcpp::as<Rcpp::NumericVector>(a["y"]);
        arms_out[k].n_trials = Rcpp::as<Rcpp::IntegerVector>(a["n_trials"]);
        arms_out[k].family   = Rcpp::as<std::string>(a["family"]);
        arms_out[k].phi      = Rcpp::as<double>(a["phi"]);
        arms_out[k].N        = (int)arms_out[k].y.size();
        // Optional grouped beta sufficient statistics. When
        // supplied they must match this arm's N and carry n_trials as the group
        // count; the built-in beta spec then reads (n_trials, slog_y, slog_1my).
        if (a.containsElementNamed("slog_y") && !Rf_isNull(a["slog_y"])) {
            arms_out[k].slog_y   = Rcpp::as<Rcpp::NumericVector>(a["slog_y"]);
            arms_out[k].slog_1my = Rcpp::as<Rcpp::NumericVector>(a["slog_1my"]);
            if ((int)arms_out[k].slog_y.size()   != arms_out[k].N ||
                (int)arms_out[k].slog_1my.size() != arms_out[k].N) {
                Rcpp::stop("Arm %d: length(slog_y)/length(slog_1my) must equal "
                           "length(y) (%d).", k + 1, arms_out[k].N);
            }
        }
        // Optional interval-censored Gaussian bounds (ordinal cover). When
        // supplied they must match this arm's N; the built-in interval_gaussian
        // spec then reads (lower, upper) instead of the point response y.
        if (a.containsElementNamed("lower") && !Rf_isNull(a["lower"])) {
            arms_out[k].lower = Rcpp::as<Rcpp::NumericVector>(a["lower"]);
            arms_out[k].upper = Rcpp::as<Rcpp::NumericVector>(a["upper"]);
            if ((int)arms_out[k].lower.size() != arms_out[k].N ||
                (int)arms_out[k].upper.size() != arms_out[k].N) {
                Rcpp::stop("Arm %d: length(lower)/length(upper) must equal "
                           "length(y) (%d).", k + 1, arms_out[k].N);
            }
        }
        // Optional upper-truncated Gaussian ceiling (truncated lognormal cover,
        //). When supplied it must match this arm's N; the built-in
        // truncated_gaussian spec reads (y, trunc_upper). +Inf entries => no
        // truncation on that row.
        if (a.containsElementNamed("trunc_upper") && !Rf_isNull(a["trunc_upper"])) {
            arms_out[k].trunc_upper = Rcpp::as<Rcpp::NumericVector>(a["trunc_upper"]);
            if ((int)arms_out[k].trunc_upper.size() != arms_out[k].N) {
                Rcpp::stop("Arm %d: length(trunc_upper) must equal "
                           "length(y) (%d).", k + 1, arms_out[k].N);
            }
        }
        if (pa.offset.size() != 0 && (int)pa.offset.size() != arms_out[k].N) {
            Rcpp::stop("Arm %d: length(offset) (%d) != N (%d).",
                       k + 1, (int)pa.offset.size(), arms_out[k].N);
        }
        // Per-arm field coefficient (default 1.0). R side parses the user-
        // facing `field_coef` into the resolved scalar `field_coef_const`
        // before calling. See R/nested_laplace_joint_helpers.R's
        // `.normalise_arm_field_coef`.
        arms_out[k].field_coef = a.containsElementNamed("field_coef_const")
                                  ? Rcpp::as<double>(a["field_coef_const"])
                                  : 1.0;
        // Per-arm cell coupling (Change 2b). When `coupled`
        // is true, the inner Newton dispatches this arm's per-cell
        // contribution to the registered CellCouplingSpec instead of summing
        // per-obs through the family. `cell_obs_map` is 1-based; length is
        // validated against arms[k].N below.
        arms_out[k].coupled = a.containsElementNamed("coupled")
                              ? Rcpp::as<bool>(a["coupled"])
                              : false;
        if (arms_out[k].coupled) {
            if (!a.containsElementNamed("cell_obs_map")) {
                Rcpp::stop("Arm %d: coupled = TRUE requires `cell_obs_map`.",
                           k + 1);
            }
            arms_out[k].cell_obs_map =
                Rcpp::as<Rcpp::IntegerVector>(a["cell_obs_map"]);
        }

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
        if (arms_out[k].coupled &&
            (int)arms_out[k].cell_obs_map.size() != N_k) {
            Rcpp::stop("Arm %d: length(cell_obs_map) (%d) != length(y) (%d).",
                       k + 1, (int)arms_out[k].cell_obs_map.size(), N_k);
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
        const bool has_prior = ((int)pa.beta_prior_prec.size() == pa.p);
        for (int j = 0; j < pa.p; j++) {
            const double prec = has_prior ? pa.beta_prior_prec[j] : tau_beta;
            const double mean = ((int)pa.beta_prior_mean.size() == pa.p)
                                ? pa.beta_prior_mean[j] : 0.0;
            grad[pa.beta_start + j] -= prec * (x[pa.beta_start + j] - mean);
            H[pa.beta_start + j][pa.beta_start + j] += prec;
        }
        for (int g = 0; g < pa.n_re_groups; g++) {
            grad[pa.re_start + g] -= pa.tau_re * x[pa.re_start + g];
            H[pa.re_start + g][pa.re_start + g] += pa.tau_re;
        }
    }
}

// Sparse twin of add_per_arm_beta_re_priors. Writes diagonal entries via
// SparseHessianBuilder::add.
inline void add_per_arm_beta_re_priors_sparse(
    DenseVec& grad, SparseHessianBuilder& H,
    const Rcpp::NumericVector& x,
    const std::vector<ParsedArm>& parsed,
    double tau_beta = 1e-4
) {
    for (const ParsedArm& pa : parsed) {
        const bool has_prior = ((int)pa.beta_prior_prec.size() == pa.p);
        for (int j = 0; j < pa.p; j++) {
            const int idx = pa.beta_start + j;
            const double prec = has_prior ? pa.beta_prior_prec[j] : tau_beta;
            const double mean = ((int)pa.beta_prior_mean.size() == pa.p)
                                ? pa.beta_prior_mean[j] : 0.0;
            grad[idx] -= prec * (x[idx] - mean);
            H.add(idx, idx, prec);
        }
        for (int g = 0; g < pa.n_re_groups; g++) {
            int idx = pa.re_start + g;
            grad[idx] -= pa.tau_re * x[idx];
            H.add(idx, idx, pa.tau_re);
        }
    }
}

// Per-arm RE + beta log-prior contribution. The weak default beta prior
// (tau_beta) is dropped so the joint log-marginal stays comparable to two
// single-arm fits. An INFORMATIVE per-arm beta prior (beta_prior_prec set by
// the consumer) IS included: it must enter the penalized objective so the
// inner Newton's line search accepts the prior-driven steps that
// add_per_arm_beta_re_priors writes into the gradient. Without this the
// gradient pulls beta toward the prior mean while the objective (likelihood
// only) rejects the step, pinning beta at the unpenalized MLE -- which at the
// occupancy psi-p boundary runs to +Inf (occu_cover coupling).
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
        if ((int)pa.beta_prior_prec.size() == pa.p) {
            for (int j = 0; j < pa.p; j++) {
                const double prec = pa.beta_prior_prec[j];
                const double mean = ((int)pa.beta_prior_mean.size() == pa.p)
                                    ? pa.beta_prior_mean[j] : 0.0;
                const double d = x[pa.beta_start + j] - mean;
                lp -= 0.5 * prec * d * d;
                lp += 0.5 * (std::log(prec) - LOG_2PI);
            }
        }
    }
    return lp;
}

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_JOINT_CORE_H
