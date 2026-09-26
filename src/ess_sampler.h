// ess_sampler.h
// Elliptical Slice Sampling for models with Gaussian priors
//
// ESS is efficient for sampling parameters with Gaussian priors because it:
// 1. Uses the prior as the proposal distribution
// 2. Automatically adapts to the posterior geometry
// 3. Has no tuning parameters (unlike HMC step size)
//
// Reference: Murray, Adams, MacKay (2010) "Elliptical Slice Sampling"
// https://proceedings.mlr.press/v9/murray10a.html
//
// All randomness goes through R's RNG (R::norm_rand / R::unif_rand, and the
// accept test in rwmh.h), so set.seed in R reproduces a run and the sampler
// must not be called from inside a parallel region. The caller is responsible
// for the surrounding GetRNGstate / PutRNGstate pair (Rcpp::RNGScope, which the
// Rcpp attribute wrappers emit and the C shim establishes explicitly).

#ifndef TULPA_ESS_SAMPLER_H
#define TULPA_ESS_SAMPLER_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <limits>
#include <algorithm>
#include "hmc_sampler.h"
#include "rwmh.h"
#include "tulpa_priors_re.h"     // re_term_view: the RE block's layout
#include "re_cov_gibbs_sweep.h"  // translate_aliased_block
#include "tulpa/re_term_prior.h" // re_term_scales

namespace tulpa_ess {

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

// ============================================================================
// ESS Configuration
// ============================================================================

struct ESSConfig {
    int n_iter;                 // Total iterations
    int n_warmup;               // Warmup iterations (for adaptation)
    int n_thin;                 // Thinning interval (values below 1 read as 1)

    bool verbose;               // Print progress
    int print_every;            // Print every N iterations

    // ESS-specific options
    bool adapt_during_warmup;   // Adapt the RWMH proposal SDs, the scale-move
                                // widths and the block ellipses during warmup
    int adapt_interval;         // Sweeps between RWMH proposal-SD updates

    // Slice updates of each random-effect log-SD beyond its RWMH step: in the
    // block's own parameterization, in the other one (the effects held fixed
    // under a non-centered block, z under a centered one), and, for a scalar
    // term once warmup has read its groups, along the partially non-centered
    // path between them (ess_re_scale_move, ess_re_pncp_move). Together they
    // cross the funnel between a scale and its effects that any one alone
    // crawls along. joint_sigma_proposal_sd is their initial stepping-out
    // width, tracked during warmup when adapt_during_warmup is set.
    bool joint_sigma_re;
    double joint_sigma_proposal_sd;

    ESSConfig()
        : n_iter(2000), n_warmup(1000), n_thin(1), verbose(true),
          print_every(100),
          adapt_during_warmup(true), adapt_interval(100),
          joint_sigma_re(false), joint_sigma_proposal_sd(0.1) {}
};

// ============================================================================
// ESS Result
// ============================================================================

struct ESSResult {
    Eigen::MatrixXd samples;       // (n_save x n_params) posterior samples
    std::vector<double> log_lik;   // Log-likelihood at each sample
    int n_slice_evals;             // Total likelihood evaluations
    double avg_slice_evals;        // Average evaluations per ESS step
    int n_slice_exhausted;         // Block updates that returned unchanged
    int n_degenerate_scale;        // Block updates skipped on a zero scale
    bool success;
    std::string error_msg;
};

// ============================================================================
// Gaussian Prior specification
// ============================================================================

// Specifies which parameters share one zero-mean isotropic ellipse for ESS.
// Every block the builder below emits is N(0, scale^2 I): the structured
// blocks that would need a covariance (spatial / temporal / GP) error there
// rather than being carried here.
struct GaussianPrior {
    std::vector<int> param_indices;  // Indices of parameters in this block
    double scale;                    // Ellipse is N(0, scale^2 * I)
    int scale_param_idx;             // If >= 0, scale = exp(params[scale_param_idx])
                                     // (a sampled log-SD, e.g. the RE log_sigma);
                                     // refreshed each sweep so the ellipse tracks it.

    // A fitted ellipse N(center, diag(sd^2)) in place of N(0, scale^2 I), set
    // from the block's own warmup draws (EllipseFit). Empty until fitted.
    Eigen::VectorXd center;
    Eigen::VectorXd sd;

    // A non-centered block whose coordinates are effects divided by sampled
    // scales, z_i = b_i / exp(params[effect_scale_idx[i]]): its ellipse is
    // fitted to the effects (center_b, sd_b) and carried to z at the current
    // scales each sweep, so it follows them instead of being fixed at the
    // ones warmup ended on. Empty otherwise.
    std::vector<int> effect_scale_idx;
    Eigen::VectorXd center_b;
    Eigen::VectorXd sd_b;

    // exp(params[effect_scale_idx[i]]) per coordinate, or all ones.
    Eigen::VectorXd effect_units(const std::vector<double>& params) const {
        Eigen::VectorXd u = Eigen::VectorXd::Ones(param_indices.size());
        for (size_t i = 0; i < effect_scale_idx.size(); i++)
            u(i) = std::exp(params[effect_scale_idx[i]]);
        return u;
    }

    GaussianPrior() : scale(1.0), scale_param_idx(-1) {}

    bool fitted() const { return center.size() > 0; }

    Eigen::VectorXd ellipse_center(int n) const {
        return fitted() ? center : Eigen::VectorXd::Zero(n);
    }

    // nu ~ N(0, ellipse covariance).
    Eigen::VectorXd draw_nu(int n) const {
        Eigen::VectorXd e(n);
        for (int i = 0; i < n; i++) e(i) = R::norm_rand();
        if (fitted()) return sd.cwiseProduct(e);
        return scale * e;
    }

    // log N(f; ellipse) up to a constant in f: the factor the ellipse already
    // carries, which the slice target removes.
    double log_ellipse(const Eigen::VectorXd& f) const {
        if (!fitted()) return -0.5 * f.squaredNorm() / (scale * scale);
        const Eigen::VectorXd r = f - center;
        return -0.5 * r.cwiseQuotient(sd).squaredNorm();
    }
};

// ============================================================================
// Core ESS step for a single block of parameters
// ============================================================================

// Performs one ESS step for one block against its ellipse, N(0, scale^2 I) or a
// fitted N(center, diag(sd^2)); for a fitted one read f - center for f below
// (Nishihara, Murray & Adams 2014). Returns new parameter values for the block.
//
// Algorithm:
// 1. Draw nu ~ N(0, scale^2 I)
// 2. Set threshold: log_y = log_lik(f) + log(u), u ~ Uniform(0,1)
// 3. Draw initial angle: theta ~ Uniform(0, 2*pi)
// 4. Set bracket: [theta_min, theta_max] = [theta - 2*pi, theta]
// 5. Loop:
//    a. f' = f * cos(theta) + nu * sin(theta)
//    b. If log_lik(f') > log_y: accept and return f'
//    c. Else shrink bracket toward 0 and draw new theta
//
// `exhausted` is set when the block is returned unchanged: either the level was
// not orderable (below) or the bracket collapsed without an acceptance.
template<typename LogLikFn>
Eigen::VectorXd ess_step(
    const Eigen::VectorXd& f,           // Current parameter values (block)
    const GaussianPrior& prior,         // Ellipse for this block
    LogLikFn log_lik_fn,                // Function: f -> log_likelihood
    double current_log_lik,             // Current log-likelihood
    int& n_evals,                       // Output: number of likelihood evaluations
    bool& exhausted                     // Output: block returned unchanged
) {
    n_evals = 0;
    exhausted = false;

    // A level of NaN or +Inf orders against no proposal, so every comparison in
    // the shrink loop below is false and the block would come back unchanged
    // after max_shrinks full target evaluations. Cost it one evaluation instead.
    // A level of -Inf is usable: any finite proposal clears it.
    if (std::isnan(current_log_lik) ||
        current_log_lik == std::numeric_limits<double>::infinity()) {
        exhausted = true;
        return f;
    }

    // Step 1: Draw nu from the ellipse (centered at 0) and read its center
    const int n = f.size();
    const Eigen::VectorXd nu = prior.draw_nu(n);
    const Eigen::VectorXd c = prior.ellipse_center(n);

    // Step 2: Set threshold
    const double log_y = current_log_lik + std::log(R::unif_rand());

    // Step 3: Initial angle
    double theta = R::unif_rand() * 2.0 * M_PI;
    double theta_min = theta - 2.0 * M_PI;
    double theta_max = theta;

    const int max_shrinks = 1000;  // Prevent infinite loop

    for (int shrink = 0; shrink < max_shrinks; shrink++) {
        // Step 5a: Proposal on ellipse
        const double cos_theta = std::cos(theta);
        const double sin_theta = std::sin(theta);
        Eigen::VectorXd f_prime = c + (f - c) * cos_theta + nu * sin_theta;

        // Step 5b: Evaluate log-likelihood
        const double log_lik_prime = log_lik_fn(f_prime);
        n_evals++;

        // A NaN target orders against nothing, so it is rejected explicitly
        // rather than through the comparison. The bracket still shrinks, which
        // is the only move that leaves the loop terminating.
        if (!std::isnan(log_lik_prime) && log_lik_prime > log_y) {
            // Accept
            return f_prime;
        }

        // Step 5c: Shrink bracket
        if (theta < 0) {
            theta_min = theta;
        } else {
            theta_max = theta;
        }

        // Draw new theta from shrunk bracket
        theta = R::unif_rand() * (theta_max - theta_min) + theta_min;
    }

    exhausted = true;
    return f;
}

// ============================================================================
// Build Gaussian priors from model structure
// ============================================================================

// Extract which parameters have Gaussian priors from the model
inline std::vector<GaussianPrior> build_gaussian_priors(
    const ModelData& data,
    const ParamLayout& layout,
    int n_params
) {
    std::vector<GaussianPrior> priors;

    // Fixed effects: beta ~ N(0, sigma_beta^2). Walk every process's
    // β block; the slice for process k is
    // [process_beta_start[k], process_beta_start[k] + process_beta_count[k]).
    // All β's share one Gaussian-prior block (single data.sigma_beta).
    {
        GaussianPrior prior;
        for (int k = 0; k < data.n_processes; k++) {
            const int start = layout.process_beta_start[k];
            const int count = layout.process_beta_count[k];
            for (int j = 0; j < count; j++) {
                prior.param_indices.push_back(start + j);
            }
        }

        if (!prior.param_indices.empty()) {
            prior.scale = data.sigma_beta;
            priors.push_back(prior);
        }
    }

    // Random effects: re ~ N(0, sigma_re^2)
    // Note: sigma_re is estimated, so we handle this specially
    if (layout.has_re && layout.re_end > layout.re_start) {
        // Multi-term RE carries a distinct sigma per term, but the ESS prior
        // block binds a SINGLE scale (log_sigma_re_idx) over the whole
        // [re_start, re_end) span -- terms beyond the first would be frozen at
        // the first term's sigma (or 1.0), silently dropping their shrinkage.
        // Error rather than mis-sample, matching the structured-spatial block
        // below.
        if (layout.log_sigma_re_multi.size() > 1) {
            Rcpp::stop("ESS prior builder: multiple random-effect terms with "
                       "distinct sigma are not supported -- the ESS Gaussian "
                       "prior block binds a single RE scale, so only the first "
                       "term's sigma would be applied. Use a per-term sampler "
                       "(NUTS / Gibbs) or a single RE term.");
        }
        GaussianPrior prior;
        for (int j = layout.re_start; j < layout.re_end; j++) {
            prior.param_indices.push_back(j);
        }

        // The ellipse is the latent block's own prior. Under the non-centered
        // parameterization the sampler builds (re = sigma z, and every
        // correlated term) that is z ~ N(0, I), so the scale stays 1; binding
        // it to sigma_re there sat the ellipse off the prior by the factor
        // sigma_re. Only a centered block, re ~ N(0, sigma_re^2 I), tracks the
        // sampled log-SD, refreshed each sweep.
        // A non-centered uncorrelated term's coordinates are its effects over
        // their coefficient's SD, which the fitted ellipse follows.
        const tulpa::priors::ReTermView v0 =
            tulpa::priors::re_term_view(data, layout, 0);
        if (data.re_parameterization != 1 && !layout.has_re_correlated_slopes &&
            layout.log_sigma_re_idx >= 0) {
            prior.scale_param_idx = layout.log_sigma_re_idx;
        } else if (!tulpa::priors::re_term_correlated(v0.chol_start,
                                                      v0.n_coefs) &&
                   v0.re_start == layout.re_start &&
                   v0.n_groups * v0.n_coefs == layout.re_end - layout.re_start) {
            bool ok = true;
            for (int idx : v0.log_sigma_idx) ok = ok && idx >= 0;
            for (int j = 0; ok && j < layout.re_end - layout.re_start; j++)
                prior.effect_scale_idx.push_back(
                    v0.log_sigma_idx[j % v0.n_coefs]);
        }
        priors.push_back(prior);
    }

    // Spatial effects (ICAR / BYM2): phi ~ N(0, tau^-1 * Q^-), where Q is the
    // graph Laplacian with a sum-to-zero constraint. The ESS prior block carries
    // an isotropic N(0, scale^2 I) covariance only; it cannot represent the
    // neighbour-coupling precision Q, so sampling a structured spatial field
    // here would silently drop all spatial smoothing. Error rather than fall
    // back to N(0, I).
    if (layout.has_spatial && layout.spatial_end > layout.spatial_start) {
        Rcpp::stop("ESS prior builder: structured spatial (ICAR/BYM2) block at "
                   "indices [%d, %d) requires the graph-Laplacian precision "
                   "tau^-1 Q^-, which the ESS Gaussian-prior block cannot carry. "
                   "Sample the spatial field with the NUTS/Laplace spatial path, "
                   "not the legacy ESS sampler.",
                   layout.spatial_start, layout.spatial_end);
    }

    // Temporal effects: phi_temporal ~ N(0, tau_temporal^-1 * Q) with Q the
    // AR1 / RW graph precision. Same limitation as the spatial block: the ESS
    // Gaussian-prior block is isotropic and cannot carry Q. Error rather than
    // silently drop the temporal correlation.
    if (layout.has_temporal && layout.temporal_end > layout.temporal_start) {
        Rcpp::stop("ESS prior builder: structured temporal (AR1/RW) block at "
                   "indices [%d, %d) requires the graph precision "
                   "tau^-1 Q^-, which the ESS Gaussian-prior block cannot carry. "
                   "Sample the temporal field with the NUTS/Laplace temporal path, "
                   "not the legacy ESS sampler.",
                   layout.temporal_start, layout.temporal_end);
    }

    // GP spatial effects: w ~ N(0, K(sigma2, phi)) with K the (NN)GP kernel
    // covariance. The ESS Gaussian-prior block is isotropic N(0, scale^2 I) and
    // cannot represent K -- an isotropic N(0, 1) ellipse combined with the full
    // log-posterior target (which already carries the GP prior) samples the
    // wrong distribution (extra N(0,1) factor, no kernel correlation). Error
    // rather than sample silently wrong, matching the spatial / temporal blocks.
    if (layout.is_gp && layout.gp_w_end > layout.gp_w_start) {
        Rcpp::stop("ESS prior builder: GP field block at indices [%d, %d) "
                   "requires the kernel covariance K(sigma2, phi), which the "
                   "isotropic ESS Gaussian-prior block cannot carry. Sample the "
                   "GP field with the NUTS/Laplace GP path, not the legacy ESS "
                   "sampler.",
                   layout.gp_w_start, layout.gp_w_end);
    }

    // ZI coefficients: beta_zi ~ N(0, zi_prior_sd^2)
    if (layout.has_zi && layout.beta_zi_end > layout.beta_zi_start) {
        GaussianPrior prior;
        for (int j = layout.beta_zi_start; j < layout.beta_zi_end; j++) {
            prior.param_indices.push_back(j);
        }

        prior.scale = data.zi_prior_sd;
        priors.push_back(prior);
    }

    // Latent factors. The templated latent prior is N(0, 1) on each factor
    // score (tulpa_priors_latent.h -- the per-factor sigma multiplies into eta,
    // not into the score), so scale = 1 is the ellipse that matches it. Unlike
    // the structured blocks above, an isotropic ellipse is not an approximation
    // here: the slice target subtracts exactly this ellipse's own quadratic, so
    // the block conditional it samples is the model's whatever the scale is
    // (see the target built in run_ess_sampler). The scale only decides how
    // close the ellipse sits to the block's actual prior, and hence how often a
    // proposal clears the slice.
    if (layout.has_latent && layout.latent_factor_end > layout.latent_factor_start) {
        GaussianPrior prior;
        for (int j = layout.latent_factor_start; j < layout.latent_factor_end; j++) {
            prior.param_indices.push_back(j);
        }
        priors.push_back(prior);
    }

    return priors;
}

// ============================================================================
// Identify non-Gaussian parameters (sampled with MH or other methods)
// ============================================================================

inline std::vector<int> get_non_gaussian_params(
    const ParamLayout& layout,
    int n_params
) {
    std::vector<int> non_gaussian;

    // Variance parameters (log scale) - use random walk MH. A slope term
    // carries one log-SD per coefficient plus, when correlated, its Cholesky
    // block; log_sigma_re_idx names the FIRST coefficient's only, so reading
    // it alone froze every other scale of the term at its initial value --
    // the slope SD of (1 + x | g) sat at exactly 1 for every draw
    // (gcol33/tulpa#877).
    if (layout.has_re && layout.has_re_slopes) {
        for (size_t t = 0; t < layout.log_sigma_re_slopes.size(); t++) {
            for (int idx : layout.log_sigma_re_slopes[t]) {
                if (idx >= 0) non_gaussian.push_back(idx);
            }
            if (t < layout.chol_re_start_multi.size() &&
                layout.chol_re_start_multi[t] >= 0) {
                for (int k = layout.chol_re_start_multi[t];
                     k < layout.chol_re_end_multi[t]; k++) {
                    non_gaussian.push_back(k);
                }
            }
        }
    } else if (layout.has_re && layout.log_sigma_re_idx >= 0) {
        non_gaussian.push_back(layout.log_sigma_re_idx);
    }

    // Overdispersion-style scalars: generic LikelihoodSpec models pack
    // model-specific extras (e.g. log_phi for a GLMM family, log_sigma
    // for Gaussian) into a contiguous block at layout.extra_offset.
    // The inner-block identity is owned by the LikelihoodSpec author,
    // so this loop is parameterization-agnostic.
    if (layout.extra_offset >= 0 && layout.n_extra_params > 0) {
        for (int j = 0; j < layout.n_extra_params; j++) {
            non_gaussian.push_back(layout.extra_offset + j);
        }
    }

    if (layout.has_spatial && layout.log_tau_spatial_idx >= 0) {
        non_gaussian.push_back(layout.log_tau_spatial_idx);
    }

    if (layout.is_bym2) {
        if (layout.log_sigma_bym2_idx >= 0) {
            non_gaussian.push_back(layout.log_sigma_bym2_idx);
        }
        if (layout.logit_rho_bym2_idx >= 0) {
            non_gaussian.push_back(layout.logit_rho_bym2_idx);
        }
    }

    if (layout.has_temporal && layout.log_tau_temporal_idx >= 0) {
        non_gaussian.push_back(layout.log_tau_temporal_idx);
    }

    if (layout.is_ar1 && layout.logit_rho_ar1_idx >= 0) {
        non_gaussian.push_back(layout.logit_rho_ar1_idx);
    }

    if (layout.is_gp) {
        if (layout.log_sigma2_gp_idx >= 0) {
            non_gaussian.push_back(layout.log_sigma2_gp_idx);
        }
        if (layout.log_phi_gp_idx >= 0) {
            non_gaussian.push_back(layout.log_phi_gp_idx);
        }
    }

    return non_gaussian;
}

// Every coordinate the sweep does not reach is a coordinate the chain never
// moves: its "draws" are the initial value repeated, and every quantity it
// feeds is conditioned on that value with nothing in the output to say so
// (gcol33/tulpa#877, and #201 before it across RE terms). The two lists above
// are built from named layout fields, so a field they do not name falls
// through silently. Close the partition here instead: any index in neither a
// Gaussian block nor the RWMH list joins the RWMH list, which is valid for any
// coordinate (a Metropolis step on the full log-posterior) and only slower
// than a dedicated move.
inline void complete_ess_partition(
    const std::vector<GaussianPrior>& gaussian_priors,
    std::vector<int>& non_gaussian,
    int n_params
) {
    std::vector<bool> covered(n_params, false);
    for (const auto& prior : gaussian_priors)
        for (int j : prior.param_indices)
            if (j >= 0 && j < n_params) covered[j] = true;
    for (int j : non_gaussian)
        if (j >= 0 && j < n_params) covered[j] = true;
    for (int j = 0; j < n_params; j++)
        if (!covered[j]) non_gaussian.push_back(j);
}

// ============================================================================
// Random walk Metropolis-Hastings step for non-Gaussian parameters
// ============================================================================

template<typename LogPostFn>
double rwmh_step(
    std::vector<double>& params,
    int idx,
    double proposal_sd,
    LogPostFn log_post_fn,
    double current_log_post,
    bool& accepted
) {
    double old_val = params[idx];
    double proposal = old_val + proposal_sd * R::norm_rand();

    params[idx] = proposal;
    double proposed_log_post = log_post_fn(params);

    double log_alpha = proposed_log_post - current_log_post;

    if (tulpa::rw_accept(log_alpha)) {
        accepted = true;
        return proposed_log_post;
    } else {
        params[idx] = old_val;
        accepted = false;
        return current_log_post;
    }
}

// ============================================================================
// Adaptive proposal SD for RWMH
// ============================================================================

struct AdaptiveProposal {
    std::vector<double> proposal_sds;
    std::vector<int> n_accepted;
    std::vector<int> n_total;
    double target_rate;
    int adapt_interval;
    int rounds = 0;

    AdaptiveProposal(int n_params, int adapt_every, double init_sd = 0.1,
                     double target = 0.234)
        : proposal_sds(n_params, init_sd),
          n_accepted(n_params, 0),
          n_total(n_params, 0),
          target_rate(target),
          adapt_interval(std::max(1, adapt_every)) {}

    void record(int idx, bool accepted) {
        n_total[idx]++;
        if (accepted) n_accepted[idx]++;
    }

    void adapt(int iter) {
        if (iter == 0 || iter % adapt_interval != 0) return;
        // Robbins-Monro in the number of adaptation rounds, not of sweeps: a
        // gain of 1 / sqrt(sweep) moved the log step by at most ~0.1 a round,
        // so a warmup of 1000 could not carry a step of 0.1 to the ~0.3 a
        // log-SD with 30 groups wants (gcol33/tulpa#877).
        rounds++;
        const double gamma = 1.0 / std::sqrt(static_cast<double>(rounds));

        for (size_t i = 0; i < proposal_sds.size(); i++) {
            if (n_total[i] > 0) {
                double rate = static_cast<double>(n_accepted[i]) / n_total[i];
                double log_sd = std::log(proposal_sds[i]);
                log_sd += gamma * (rate - target_rate);
                proposal_sds[i] = std::exp(log_sd);
                // Reset counters
                n_accepted[i] = 0;
                n_total[i] = 0;
            }
        }
    }
};

// ============================================================================
// Random-effect moves the block updates cannot make (gcol33/tulpa#877)
// ============================================================================
//
// The Gaussian blocks are sliced one at a time against their own ellipses and
// the log-SDs are random-walked with the latent block held fixed. Three
// directions of a hierarchical posterior are then crossed only at the rate of
// the narrowest conditional step; on y ~ x + (1 | g) the intercept read
// split-Rhat ~1.3-1.9 and bulk ESS 1-16 of 1000 at the default length:
//
//   * the level a fixed effect shares with a random-effect coefficient whose
//     design column repeats it (the intercept against (1 | g)): eta is
//     unchanged by beta_j + d, b_{g,c} - d, so only the priors resolve d, and
//     each block moves only as far as its conditional given the other allows.
//     ess_translate_alias draws it exactly (the translation the re_cov_gibbs
//     sweep makes, translate_aliased_block);
//   * a covariate that varies between groups against the group effects that
//     absorb its between-group part: ess_fixed_shift_move slices them jointly;
//   * a term's scale against its effects: with the non-centered z held fixed
//     a change of sigma rescales every effect, which the data forbid whenever
//     the groups are informative. ess_re_scale_move slices each log-SD in both
//     parameterizations, and ess_re_pncp_move along the partially non-centered
//     path between them.
//
// Each is an exact update of the full log-posterior (a Metropolis-corrected
// Gibbs draw, or a one-dimensional slice along a line or a scale path with its
// Jacobian), so none changes what is sampled, only how fast.

// One random-effect term as the moves read it: its layout, whether its latent
// block holds z (re = sigma L z) or the effects themselves, and per coefficient
// the process-0 fixed-effect column whose design repeats the coefficient's
// (-1 where none does).
struct EssReTerm {
    int t = 0;                        // the term's index in the layout
    tulpa::priors::ReTermView v;
    bool noncentered = true;
    std::vector<int> alias;
};

// Group of observation i under term t (1-based, 0 = none), read the way the
// linear predictor reads it (generic_re_effect, log_post_generic_impl.h).
inline int ess_obs_group(const ModelData& data, const ParamLayout& layout,
                         int i, int t) {
    const int n_terms = tulpa::priors::re_n_terms(data);
    if (!data.re_group_multi_flat.empty() &&
        (layout.has_re_slopes || n_terms > 1)) {
        const size_t k = (size_t)i * n_terms + t;
        return k < data.re_group_multi_flat.size() ? data.re_group_multi_flat[k]
                                                   : 0;
    }
    return (t == 0 && !data.re_group.empty()) ? data.re_group[i] : 0;
}

// Coefficient c's random-effect design column under term t into z (1 for the
// group intercept, the slope's covariate otherwise, 0 where the observation
// has no group), and each observation's group into grp (1-based, 0 = none).
// False when the column cannot be read.
inline bool ess_re_design_column(const ModelData& data,
                                 const ParamLayout& layout, int t, int c,
                                 int n_coefs, std::vector<double>& z,
                                 std::vector<int>& grp) {
    const int N = data.N;
    const bool has_int = tulpa::re_term_has_intercept(data, t);
    const int n_slopes = n_coefs - (has_int ? 1 : 0);
    const std::vector<double>* slopes =
        (t < (int)data.re_slope_matrices.size()) ? &data.re_slope_matrices[t]
                                                 : nullptr;
    const bool is_int = has_int && c == 0;
    const int s = c - (has_int ? 1 : 0);
    if (!is_int && !(slopes && (int)slopes->size() >= N * n_slopes))
        return false;
    z.assign(N, 0.0);
    grp.assign(N, 0);
    bool nonzero = false;
    for (int i = 0; i < N; i++) {
        grp[i] = ess_obs_group(data, layout, i, t);
        if (grp[i] <= 0) continue;
        z[i] = is_int ? 1.0 : (*slopes)[(size_t)i * n_slopes + s];
        if (z[i] != 0.0) nonzero = true;
    }
    return nonzero;
}

// The single linear predictor's design, or null: with several processes an
// effect can be shared into more than one, and one fixed effect cannot absorb
// a move of it.
inline const tulpa::ProcessData* ess_single_process(const ModelData& data) {
    if (data.n_processes != 1 || data.processes.empty()) return nullptr;
    if (!data.sharing.re.empty() && !data.sharing.re[0]) return nullptr;
    const auto& proc = data.processes[0];
    if (proc.p <= 0 || (int)proc.X_flat.size() < data.N * proc.p) return nullptr;
    return &proc;
}

// The fixed-effect column, if any, equal on every observation to coefficient
// c's random-effect design column. The move that reads this is
// Metropolis-corrected, so a column read as aliased that is not costs
// rejections, never correctness.
inline std::vector<int> ess_re_fixed_alias(const ModelData& data,
                                           const ParamLayout& layout, int t,
                                           int n_coefs) {
    std::vector<int> alias(n_coefs, -1);
    const tulpa::ProcessData* proc = ess_single_process(data);
    if (!proc) return alias;
    const int p = proc->p, N = data.N;
    std::vector<double> z;
    std::vector<int> grp;
    for (int c = 0; c < n_coefs; c++) {
        if (!ess_re_design_column(data, layout, t, c, n_coefs, z, grp)) continue;
        for (int j = 0; j < p && alias[c] < 0; j++) {
            bool same = true;
            for (int i = 0; i < N && same; i++) {
                const double xij = proc->X_flat[(size_t)i * p + j];
                same = std::abs(xij - z[i]) <= 1e-10 * (1.0 + std::abs(z[i]));
            }
            if (same) alias[c] = j;
        }
    }
    return alias;
}

inline std::vector<EssReTerm> ess_re_terms(const ModelData& data,
                                           const ParamLayout& layout) {
    std::vector<EssReTerm> terms;
    if (!layout.has_re) return terms;
    const int n_terms = tulpa::priors::re_n_terms(data);
    for (int t = 0; t < n_terms; t++) {
        EssReTerm term;
        term.t = t;
        term.v = tulpa::priors::re_term_view(data, layout, t);
        if (term.v.re_start < 0 || term.v.n_groups <= 0) continue;
        bool ok = true;
        for (int idx : term.v.log_sigma_idx) ok = ok && idx >= 0;
        if (!ok) continue;
        term.noncentered = data.re_parameterization == 1 ||
            tulpa::priors::re_term_correlated(term.v.chol_start, term.v.n_coefs);
        term.alias = ess_re_fixed_alias(data, layout, t, term.v.n_coefs);
        terms.push_back(std::move(term));
    }
    return terms;
}

// The term's prior factor A = diag(sigma) L (L = I for an uncorrelated term),
// so an effect vector is b_g = A z_g ~ N(0, A A'), read through the same
// re_term_scales() the prior density reads.
inline Eigen::MatrixXd ess_re_scale_factor(const std::vector<double>& params,
                                           const EssReTerm& term) {
    const int nc = term.v.n_coefs;
    const bool correlated =
        tulpa::priors::re_term_correlated(term.v.chol_start, nc);
    std::vector<double> sig(nc);
    std::vector<double> L_flat(correlated ? nc * nc : 0, 0.0);
    tulpa::priors::re_term_scales(params.data(), term.v.log_sigma_idx.data(),
                                  nc, term.v.chol_start, sig.data(),
                                  L_flat.data());
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(nc, nc);
    if (correlated) {
        for (int r = 0; r < nc; r++)
            for (int c = 0; c < nc; c++) A(r, c) = L_flat[r * nc + c];
    }
    for (int r = 0; r < nc; r++) A.row(r) *= sig[r];
    return A;
}

// The term's effects as a (groups x coefficients) matrix: B = Z A' for a
// non-centered block, the block itself for a centered one. And back.
inline Eigen::MatrixXd ess_re_effects(const std::vector<double>& params,
                                      const EssReTerm& term,
                                      const Eigen::MatrixXd& A) {
    const int G = term.v.n_groups, nc = term.v.n_coefs;
    Eigen::MatrixXd Z(G, nc);
    for (int g = 0; g < G; g++)
        for (int c = 0; c < nc; c++)
            Z(g, c) = params[term.v.re_start + g * nc + c];
    return term.noncentered ? Eigen::MatrixXd(Z * A.transpose()) : Z;
}

inline void ess_re_store_effects(std::vector<double>& params,
                                 const EssReTerm& term,
                                 const Eigen::MatrixXd& A,
                                 const Eigen::MatrixXd& B) {
    const int G = term.v.n_groups, nc = term.v.n_coefs;
    // Z = B A'^-1, i.e. Z' = A^-1 B'; A is lower triangular.
    const Eigen::MatrixXd Z = term.noncentered
        ? Eigen::MatrixXd(A.triangularView<Eigen::Lower>()
                              .solve(B.transpose()).transpose())
        : B;
    for (int g = 0; g < G; g++)
        for (int c = 0; c < nc; c++)
            params[term.v.re_start + g * nc + c] = Z(g, c);
}

// Draw the level the term's aliased coefficients share with their fixed
// effects (translate_aliased_block). Along that line the likelihood is
// constant, so the conditional is the Gaussian the two priors make and the draw
// is its exact Gibbs step: a group move with unit Jacobian (Liu & Sabatti
// 2000). It is proposed from that Gaussian and Metropolis-corrected against the
// full log-posterior, which accepts it with probability one when the alias
// holds and keeps the chain exact when it does not. The Gaussian does not
// depend on where on the line the chain sits, so it is an independence
// proposal along the line. Returns the new log-posterior.
template<typename LogPostFn>
double ess_translate_alias(std::vector<double>& params, const EssReTerm& term,
                           const ModelData& data, const ParamLayout& layout,
                           LogPostFn log_post_fn, double current_log_post,
                           bool& accepted) {
    accepted = false;
    bool any = false;
    for (int j : term.alias) any = any || j >= 0;
    const int p = layout.process_beta_count.empty() ? 0
                  : layout.process_beta_count[0];
    if (!any || p <= 0) return current_log_post;
    const int beta0 = layout.process_beta_start[0];
    const Eigen::MatrixXd A = ess_re_scale_factor(params, term);
    Eigen::LLT<Eigen::MatrixXd> llt_cov(A * A.transpose());
    if (llt_cov.info() != Eigen::Success) return current_log_post;
    const Eigen::MatrixXd Q =
        llt_cov.solve(Eigen::MatrixXd::Identity(A.rows(), A.cols()));
    if (!Q.allFinite()) return current_log_post;

    Eigen::MatrixXd B = ess_re_effects(params, term, A);
    Eigen::VectorXd beta(p);
    for (int j = 0; j < p; j++) beta(j) = params[beta0 + j];
    const Eigen::VectorXd mean0 = Eigen::VectorXd::Zero(p);
    const Eigen::VectorXd sd = Eigen::VectorXd::Constant(p, data.sigma_beta);

    // The Gaussian the draw is proposed from, up to a constant.
    const double prec_beta = 1.0 / (data.sigma_beta * data.sigma_beta);
    auto log_q = [&](const Eigen::MatrixXd& Bm, const Eigen::VectorXd& bt) {
        double lq = -0.5 * (Bm * Q).cwiseProduct(Bm).sum();
        for (int j : term.alias)
            if (j >= 0) lq -= 0.5 * prec_beta * bt(j) * bt(j);
        return lq;
    };
    const double lq_old = log_q(B, beta);

    const std::vector<double> saved = params;
    if (!tulpa::translate_aliased_block(B, beta, Q, term.alias, mean0, sd))
        return current_log_post;
    for (int j = 0; j < p; j++) params[beta0 + j] = beta(j);
    ess_re_store_effects(params, term, A, B);

    const double lp_new = log_post_fn(params);
    const double log_alpha =
        (lp_new - current_log_post) - (log_q(B, beta) - lq_old);
    if (tulpa::rw_accept(log_alpha) && std::isfinite(lp_new)) {
        accepted = true;
        return lp_new;
    }
    params = saved;
    return current_log_post;
}

// One univariate slice-sampling update of x on the log density h (Neal 2003):
// step out by w, at most max_steps widths in all, then shrink. h_x is h at x
// on entry and at the returned point on exit. A NaN density reads as outside
// the slice, and a non-finite level leaves x where it is.
template<typename H>
double slice_step_1d(double x, double& h_x, H h, double w, int max_steps = 32) {
    if (!std::isfinite(h_x) || !(w > 0.0)) return x;
    const double log_y = h_x + std::log(R::unif_rand());
    double lo = x - w * R::unif_rand();
    double hi = lo + w;
    int j = static_cast<int>(std::floor(max_steps * R::unif_rand()));
    int k = max_steps - 1 - j;
    while (j-- > 0 && h(lo) > log_y) lo -= w;
    while (k-- > 0 && h(hi) > log_y) hi += w;
    for (int it = 0; it < 200; it++) {
        const double x1 = lo + R::unif_rand() * (hi - lo);
        const double h1 = h(x1);
        if (!std::isnan(h1) && h1 > log_y) {
            h_x = h1;
            return x1;
        }
        if (x1 < x) lo = x1; else hi = x1;
    }
    return x;
}

// Slice-sample the log-SD of coefficient c, either in the block's own
// parameterization (hold_latent: the latent block is left where it is) or in
// the other one: a
// non-centered block keeps the effects b_g = A z_g fixed (z_g' = A'^-1 A z_g,
// det = e^-delta per group, so log Jacobian -G delta); a centered one keeps
// z = b / sigma fixed (b' = b e^delta, +G delta). In the coordinates the move
// holds fixed this is the log-SD's full conditional, so the slice update is
// exact whatever the likelihood; `width` is its stepping-out width. Returns
// the new log-posterior; `delta` is the step taken.
template<typename LogPostFn>
double ess_re_scale_move(std::vector<double>& params, const EssReTerm& term,
                         int c, bool hold_latent, double width,
                         LogPostFn log_post_fn, double current_log_post,
                         double& delta) {
    const int G = term.v.n_groups, nc = term.v.n_coefs;
    const int li = term.v.log_sigma_idx[c];
    const std::vector<double> base = params;
    const Eigen::MatrixXd B = (term.noncentered && !hold_latent)
        ? ess_re_effects(base, term, ess_re_scale_factor(base, term))
        : Eigen::MatrixXd();

    // Place the state at step d along the move; returns its log Jacobian.
    auto place = [&](double d) {
        params = base;
        params[li] += d;
        if (hold_latent) return 0.0;
        if (term.noncentered) {
            ess_re_store_effects(params, term, ess_re_scale_factor(params, term),
                                 B);
            return -double(G) * d;
        }
        const double scl = std::exp(d);
        for (int g = 0; g < G; g++) params[term.v.re_start + g * nc + c] *= scl;
        return double(G) * d;
    };
    auto h = [&](double d) {
        const double log_jac = place(d);
        return log_post_fn(params) + log_jac;
    };

    double h_d = current_log_post;
    delta = slice_step_1d(0.0, h_d, h, width);
    const double log_jac = place(delta);
    return delta == 0.0 ? current_log_post : h_d - log_jac;
}

// The scale of an uncorrelated term's coefficient against its effects,
// partially non-centered (Papaspiliopoulos, Roberts & Skold 2003). The two
// scale moves above hold z (every effect scales with sigma) or b (none does);
// the posterior sits in between, group by group. With each group's likelihood
// read as a Gaussian N(b_g; c_g, v_g), b_g | sigma has mean
// m_g = c_g sigma^2 / (sigma^2 + v_g) and SD s_g = sqrt(v_g sigma^2 /
// (sigma^2 + v_g)), and the move carries each effect's standardized position
// r_g = (b_g - m_g) / s_g to the new scale: b_g' = m_g(sigma') + r_g
// s_g(sigma'). A group the data pin (v_g -> 0) keeps its effect; one they
// barely see (v_g -> Inf) scales it with sigma. That joint shrinkage of every
// effect with sigma is the slow mode the two moves alone leave: on the #877
// poisson fixture the effects' own spread read bulk ESS 5 to 99 of 1000, and
// sigma, which follows it, 31 to 465. It is also the way out of the neck of
// the funnel: a chain that reaches sigma ~ 0 with the effects shrunk to
// nothing has no single-parameterization move back, since growing sigma alone
// rescales effects the data do not support, and growing the effects alone
// meets their prior.
//
// c_g and v_g are read off the likelihood itself, not off the draws (draws
// taken inside the neck describe the neck): three evaluations of the
// log-posterior along b_g with the effect's own prior removed give its slope
// and curvature, and one Newton step from the current effect gives c_g. A
// group with no curvature is read as v_g = Inf. They only shape the move,
// which is sliced on the exact target with its Jacobian, so a poor reading
// costs mixing, not correctness; they are re-read a few times during warmup
// and fixed from its end.
struct EssPncp {
    bool ready = false;
    Eigen::VectorXd c, v;             // v(g) = Inf: fully non-centered

    // m_g and log s_g at scale sigma.
    void moments(int g, double sigma, double& m, double& log_s) const {
        const double s2 = sigma * sigma;
        if (!std::isfinite(v(g))) {
            m = 0.0;
            log_s = std::log(sigma);
            return;
        }
        const double k = s2 / (s2 + v(g));
        m = c(g) * k;
        log_s = 0.5 * std::log(v(g) * k);
    }
};

// The effects of coefficient c of an uncorrelated term, b_{g,c}.
inline Eigen::VectorXd ess_re_coef_effects(const std::vector<double>& params,
                                           const EssReTerm& term, int c) {
    const int G = term.v.n_groups, nc = term.v.n_coefs;
    const double sigma = std::exp(params[term.v.log_sigma_idx[c]]);
    Eigen::VectorXd b(G);
    for (int g = 0; g < G; g++) {
        const double lat = params[term.v.re_start + g * nc + c];
        b(g) = term.noncentered ? sigma * lat : lat;
    }
    return b;
}

// Read c_g and v_g for coefficient c off the likelihood at the current state
// (see EssPncp); params is left as it was.
template<typename LogPostFn>
void ess_pncp_read(std::vector<double>& params, const EssReTerm& term, int c,
                   LogPostFn log_post_fn, EssPncp& pn) {
    const int G = term.v.n_groups, nc = term.v.n_coefs;
    const double sigma = std::exp(params[term.v.log_sigma_idx[c]]);
    if (!(sigma > 0.0) || !std::isfinite(sigma)) return;
    const Eigen::VectorXd b0 = ess_re_coef_effects(params, term, c);
    const double h = 0.1;
    pn.c.resize(G);
    pn.v.resize(G);
    // log-likelihood of group g's effect at b: the log-posterior with the
    // effect's own N(0, sigma^2) prior removed (in the latent coordinates the
    // prior carries, so the removal matches it under either parameterization).
    auto lik = [&](int g, double b) {
        const int k = term.v.re_start + g * nc + c;
        const double saved = params[k];
        params[k] = term.noncentered ? b / sigma : b;
        const double lat = params[k];
        const double prior = term.noncentered ? -0.5 * lat * lat
                                              : -0.5 * lat * lat / (sigma * sigma);
        const double out = log_post_fn(params) - prior;
        params[k] = saved;
        return out;
    };
    bool ok = true;
    for (int g = 0; g < G && ok; g++) {
        const double f0 = lik(g, b0(g));
        const double fp = lik(g, b0(g) + h);
        const double fm = lik(g, b0(g) - h);
        ok = std::isfinite(f0) && std::isfinite(fp) && std::isfinite(fm);
        const double d2 = (fp + fm - 2.0 * f0) / (h * h);
        if (ok && d2 < -1e-8) {
            pn.v(g) = -1.0 / d2;
            pn.c(g) = b0(g) + (fp - fm) / (2.0 * h) * pn.v(g);
        } else {
            pn.v(g) = std::numeric_limits<double>::infinity();
            pn.c(g) = 0.0;
        }
    }
    pn.ready = ok;
}

// Slice-sample the log-SD of coefficient c of an uncorrelated term along the
// partially non-centered path. Returns the new log-posterior.
template<typename LogPostFn>
double ess_re_pncp_move(std::vector<double>& params, const EssReTerm& term,
                        int c, const EssPncp& pn, double width,
                        LogPostFn log_post_fn, double current_log_post) {
    const int G = term.v.n_groups, nc = term.v.n_coefs;
    const int li = term.v.log_sigma_idx[c];
    const std::vector<double> base = params;
    const double sigma0 = std::exp(base[li]);
    Eigen::VectorXd r(G), log_s0(G);
    for (int g = 0; g < G; g++) {
        const double lat = base[term.v.re_start + g * nc + c];
        const double b = term.noncentered ? sigma0 * lat : lat;
        double m;
        pn.moments(g, sigma0, m, log_s0(g));
        r(g) = (b - m) / std::exp(log_s0(g));
    }

    // Place the state at step d along the path; returns its log Jacobian in
    // the latent coordinates.
    auto place = [&](double d) {
        params = base;
        params[li] += d;
        const double sigma = std::exp(params[li]);
        double log_jac = term.noncentered ? -double(G) * d : 0.0;
        for (int g = 0; g < G; g++) {
            double m, log_s;
            pn.moments(g, sigma, m, log_s);
            const double b = m + r(g) * std::exp(log_s);
            params[term.v.re_start + g * nc + c] =
                term.noncentered ? b / sigma : b;
            log_jac += log_s - log_s0(g);
        }
        return log_jac;
    };
    auto h = [&](double d) {
        const double log_jac = place(d);
        return log_post_fn(params) + log_jac;
    };

    double h_d = current_log_post;
    const double d = slice_step_1d(0.0, h_d, h, width);
    const double log_jac = place(d);
    return d == 0.0 ? current_log_post : h_d - log_jac;
}

// A fixed effect whose column varies between groups, and the part of that
// variation the group effects carry: per term, the (groups x coefficients)
// least-squares projection of X_j onto each coefficient's design within each
// group -- the group mean of x for a group intercept. A move of beta_j by d
// with every effect b_{g,c} moved by -d a_{g,c} leaves eta changed only by the
// within-group part of X_j, the part the data resolve sharply, so it crosses
// the correlation between a coefficient and the group effects that absorb its
// between-group part. Sliced one block at a time that correlation held x's
// lag-1 autocorrelation at 0.65 on the #877 poisson fixture (bulk ESS ~100 of
// 1000, against ~600 with no random effect). A column aliased to a coefficient
// (a_{g,c} = 1 in every group) is the exact translation above and is left to
// it.
struct EssFixedShift {
    int j = -1;
    std::vector<Eigen::MatrixXd> a;   // per term in re_terms
};

inline std::vector<EssFixedShift> ess_fixed_shifts(
    const ModelData& data, const ParamLayout& layout,
    const std::vector<EssReTerm>& terms) {
    std::vector<EssFixedShift> shifts;
    const tulpa::ProcessData* proc = ess_single_process(data);
    if (!proc || terms.empty()) return shifts;
    const int p = proc->p, N = data.N;
    std::vector<bool> aliased(p, false);
    for (const auto& term : terms)
        for (int j : term.alias)
            if (j >= 0) aliased[j] = true;

    // Per term and coefficient, the design column and groups (read once).
    std::vector<std::vector<std::vector<double>>> Z(terms.size());
    std::vector<std::vector<std::vector<int>>> GR(terms.size());
    std::vector<std::vector<bool>> ok(terms.size());
    for (size_t t = 0; t < terms.size(); t++) {
        const int nc = terms[t].v.n_coefs;
        Z[t].resize(nc); GR[t].resize(nc); ok[t].assign(nc, false);
        // The term's index in the layout is its position among the terms the
        // layout declares; ess_re_terms keeps them in that order.
        for (int c = 0; c < nc; c++)
            ok[t][c] = ess_re_design_column(data, layout, terms[t].t, c, nc,
                                            Z[t][c], GR[t][c]);
    }

    for (int j = 0; j < p; j++) {
        if (aliased[j]) continue;
        EssFixedShift sh;
        sh.j = j;
        bool any = false;
        for (size_t t = 0; t < terms.size(); t++) {
            const int G = terms[t].v.n_groups, nc = terms[t].v.n_coefs;
            Eigen::MatrixXd a = Eigen::MatrixXd::Zero(G, nc);
            for (int c = 0; c < nc; c++) {
                if (!ok[t][c]) continue;
                Eigen::VectorXd num = Eigen::VectorXd::Zero(G);
                Eigen::VectorXd den = Eigen::VectorXd::Zero(G);
                for (int i = 0; i < N; i++) {
                    const int g = GR[t][c][i] - 1;
                    if (g < 0 || g >= G) continue;
                    const double zi = Z[t][c][i];
                    num(g) += proc->X_flat[(size_t)i * p + j] * zi;
                    den(g) += zi * zi;
                }
                for (int g = 0; g < G; g++)
                    if (den(g) > 0.0) a(g, c) = num(g) / den(g);
            }
            if (a.cwiseAbs().maxCoeff() > 1e-12) any = true;
            sh.a.push_back(a);
        }
        if (any) shifts.push_back(std::move(sh));
    }
    return shifts;
}

// Slice-sample beta_j + d, b_{g,c} - d a_{g,c} along that line (the latent
// block moving by the matching z direction, A^-1 a_g under a non-centered
// term). The direction is fixed while the move runs, so this is a slice update
// of one coordinate in a linear reparameterization with unit Jacobian.
// Returns the new log-posterior.
template<typename LogPostFn>
double ess_fixed_shift_move(std::vector<double>& params,
                            const EssFixedShift& shift,
                            const std::vector<EssReTerm>& terms,
                            const ParamLayout& layout, double width,
                            LogPostFn log_post_fn, double current_log_post) {
    const std::vector<double> base = params;
    std::vector<double> u(params.size(), 0.0);
    u[layout.process_beta_start[0] + shift.j] = 1.0;
    for (size_t t = 0; t < terms.size(); t++) {
        if (shift.a[t].cwiseAbs().maxCoeff() <= 1e-12) continue;
        ess_re_store_effects(u, terms[t], ess_re_scale_factor(base, terms[t]),
                             -shift.a[t]);
    }
    auto h = [&](double d) {
        for (size_t k = 0; k < params.size(); k++) params[k] = base[k] + d * u[k];
        return log_post_fn(params);
    };
    double h_d = current_log_post;
    const double d = slice_step_1d(0.0, h_d, h, width);
    for (size_t k = 0; k < params.size(); k++) params[k] = base[k] + d * u[k];
    return d == 0.0 ? current_log_post : h_d;
}

// ============================================================================
// Warmup-fitted ellipses
// ============================================================================
//
// A block's prior is a poor ellipse whenever the data dominate it: the fixed
// effects under a N(0, sigma_beta^2) prior sit ~0.1 wide inside an ellipse 2.5
// or 10 wide, so each slice lands a step of one conditional width in a random
// direction. Any ellipse is valid, since the slice target removes whichever one
// is used (log_ellipse), so the warmup fits one to each block's own draws: mean
// and SD per coordinate over the second half of warmup, refitted over its last
// quarter, the SD widened by INFLATE. It is fixed from the end of warmup, so
// the stored chain runs one kernel. A block whose ellipse tracks a sampled
// log-SD (a centered RE block) keeps it.
//
// Measured on the #877 fixtures, and why the fit stops at a diagonal: a
// dense ellipse on the fixed effects and group effects pooled into one block
// (the draws' covariance, shrunk toward its diagonal) took x's bulk ESS from
// ~150 to 3-65 of 1000 -- 250 autocorrelated warmup draws underestimate the
// small eigenvalues of a 32-dimensional covariance, and an ellipse narrower
// than the target in any direction is where ESS mixes worst -- and pooling
// alone, diagonal, still cost x half its ESS. INFLATE 1.5 cost more than 1.15
// did, for the same reason in 30 dimensions: the proposals start on a shell
// the target rarely reaches, and shrink back to short steps.
struct EllipseFit {
    static constexpr double INFLATE = 1.15;
    std::vector<Eigen::VectorXd> sum, sumsq;
    int n = 0;
    const int start, mid, end;

    EllipseFit(const std::vector<GaussianPrior>& priors, int n_warmup)
        : start(n_warmup / 2), mid(n_warmup - n_warmup / 4), end(n_warmup) {
        for (const auto& pr : priors) {
            sum.push_back(Eigen::VectorXd::Zero(pr.param_indices.size()));
            sumsq.push_back(Eigen::VectorXd::Zero(pr.param_indices.size()));
        }
    }

    void observe(int iter, const std::vector<double>& params,
                 std::vector<GaussianPrior>& priors) {
        if (iter < start) return;
        for (size_t b = 0; b < priors.size(); b++) {
            const Eigen::VectorXd unit = priors[b].effect_units(params);
            for (size_t i = 0; i < priors[b].param_indices.size(); i++) {
                const double v = unit(i) * params[priors[b].param_indices[i]];
                sum[b](i) += v;
                sumsq[b](i) += v * v;
            }
        }
        n++;
        // Fit at the three-quarter mark and at the end of warmup, restarting
        // the window after the first so the final ellipse reads the chain the
        // first one already improved.
        if (iter == mid - 1 || iter == end - 1) {
            fit(priors);
            for (size_t b = 0; b < priors.size(); b++) {
                sum[b].setZero();
                sumsq[b].setZero();
            }
            n = 0;
        }
    }

private:
    void fit(std::vector<GaussianPrior>& priors) const {
        if (n < 20) return;
        for (size_t b = 0; b < priors.size(); b++) {
            if (priors[b].scale_param_idx >= 0) continue;
            const Eigen::VectorXd m = sum[b] / double(n);
            Eigen::VectorXd var = sumsq[b] / double(n) - m.cwiseProduct(m);
            bool ok = m.allFinite() && var.allFinite();
            for (int i = 0; ok && i < var.size(); i++) ok = var(i) > 0.0;
            if (!ok) continue;
            if (!priors[b].effect_scale_idx.empty()) {
                priors[b].center_b = m;
                priors[b].sd_b = INFLATE * var.cwiseSqrt();
            } else {
                priors[b].center = m;
                priors[b].sd = INFLATE * var.cwiseSqrt();
            }
        }
    }
};

// ============================================================================
// Main ESS sampler
// ============================================================================

// Forward declaration of log_post computation
double compute_log_post_double(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
);

inline ESSResult run_ess_sampler(
    const std::vector<double>& init_params,
    const ModelData& data,
    const ParamLayout& layout,
    const ESSConfig& config
) {
    ESSResult result;
    result.success = true;

    int n_params = init_params.size();
    // Number of stored draws: post-warmup iters store when
    // (iter - n_warmup) % thin == 0, which fires ceil(post / thin) times.
    const int thin = std::max(1, config.n_thin);
    const int n_post = config.n_iter - config.n_warmup;
    const int n_save = n_post > 0 ? (n_post + thin - 1) / thin : 0;

    result.samples.resize(n_save, n_params);
    result.log_lik.resize(n_save);
    result.n_slice_evals = 0;
    result.n_slice_exhausted = 0;
    result.n_degenerate_scale = 0;

    // Build Gaussian priors for ESS blocks
    std::vector<GaussianPrior> gaussian_priors = build_gaussian_priors(data, layout, n_params);

    // Get non-Gaussian parameters for RWMH
    std::vector<int> non_gaussian = get_non_gaussian_params(layout, n_params);
    complete_ess_partition(gaussian_priors, non_gaussian, n_params);

    // Random-effect terms, and the one-dimensional slice moves the sweep
    // adds for them: each log-SD in the other parameterization (when
    // joint_sigma_re), and each fixed effect with its between-group part.
    const std::vector<EssReTerm> re_terms = ess_re_terms(data, layout);
    const std::vector<EssFixedShift> shifts =
        ess_fixed_shifts(data, layout, re_terms);
    struct SliceMove {
        int kind;          // 0 / 2: log-SD (term, coef) in the other / the
                           // block's own parameterization; 3: a scalar
                           // term's log-SD partially non-centered;
                           // 1: shifts[term]
        int term, coef;
        int tracked;       // the coordinate whose spread sets the width
        double width, sum = 0.0, sumsq = 0.0;
        EssPncp pn;        // kind 3: the path, read off the likelihood
    };
    std::vector<SliceMove> slice_moves;
    if (config.joint_sigma_re) {
        for (int t = 0; t < (int)re_terms.size(); t++)
            for (int c = 0; c < re_terms[t].v.n_coefs; c++)
                for (int kind : {2, 0, 3}) {
                    if (kind == 3 && tulpa::priors::re_term_correlated(
                            re_terms[t].v.chol_start, re_terms[t].v.n_coefs))
                        continue;
                    slice_moves.push_back({kind, t, c,
                                           re_terms[t].v.log_sigma_idx[c],
                                           config.joint_sigma_proposal_sd});
                }
    }
    // Sweeps at which warmup re-reads the partially non-centered paths.
    auto pncp_read_now = [&](int iter) {
        const int w = config.n_warmup;
        for (int at : {w / 10, w / 4, w / 2, w - w / 4, w - 1})
            if (iter == at) return true;
        return false;
    };
    for (int k = 0; k < (int)shifts.size(); k++)
        slice_moves.push_back({1, k, 0,
                               layout.process_beta_start[0] + shifts[k].j,
                               config.joint_sigma_proposal_sd});
    int slice_n = 0;

    // Adaptive proposals for the RWMH parameters. The slice widths are set
    // during warmup to three SDs of the tracked coordinate's own draws so far;
    // tracking the steps taken instead feeds back -- a narrow width takes
    // short steps, which narrow it further, until stepping out no longer
    // reaches the edge of the slice.
    const int n_rwmh = (int)non_gaussian.size();
    AdaptiveProposal adaptive(n_rwmh, config.adapt_interval);

    EllipseFit ellipse_fit(gaussian_priors, config.n_warmup);

    // Current state
    std::vector<double> params = init_params;

    // Compute initial log-posterior
    double current_log_post = compute_log_post_double(params, data, layout);

    if (!std::isfinite(current_log_post)) {
        result.success = false;
        result.error_msg = "Initial log-posterior is not finite";
        return result;
    }

    // Progress
    if (config.verbose) {
        Rcpp::Rcout << "Running ESS sampler...\n";
        Rcpp::Rcout << "  Parameters: " << n_params << "\n";
        Rcpp::Rcout << "  Gaussian blocks: " << gaussian_priors.size() << "\n";
        Rcpp::Rcout << "  Non-Gaussian params: " << non_gaussian.size() << "\n";
        Rcpp::Rcout << "  Iterations: " << config.n_iter << " (warmup: " << config.n_warmup << ")\n";
    }

    int save_idx = 0;
    int total_ess_evals = 0;
    int total_ess_steps = 0;
    int n_slice_exhausted = 0;
    int n_degenerate_scale = 0;

    for (int iter = 0; iter < config.n_iter; iter++) {
        // Check for user interrupt
        if (iter % 100 == 0) {
            Rcpp::checkUserInterrupt();
        }

        // ----------------------------------------------------------------
        // ESS updates for Gaussian-prior blocks
        // ----------------------------------------------------------------
        for (auto& prior : gaussian_priors) {
            // Refresh the ellipse scale from the sampled log-SD when bound to a
            // parameter (the RE block tracks sigma_re = exp(log_sigma_re)).
            if (prior.scale_param_idx >= 0) {
                prior.scale = std::exp(params[prior.scale_param_idx]);
            }
            if (!prior.effect_scale_idx.empty() && prior.center_b.size() > 0) {
                const Eigen::VectorXd unit = prior.effect_units(params);
                if (unit.allFinite() && unit.minCoeff() > 0.0) {
                    prior.center = prior.center_b.cwiseQuotient(unit);
                    prior.sd = prior.sd_b.cwiseQuotient(unit);
                }
            }
            const double blk_prec = 1.0 / (prior.scale * prior.scale);

            // A sufficiently negative sampled log-SD underflows the scale to
            // exactly zero, making blk_prec infinite and the slice target below
            // -Inf + Inf = NaN at every angle. Leave the block where it is for
            // this sweep and count it; the hyperparameter moves that follow can
            // still carry the scale back into range.
            if (!prior.fitted() && (!(prior.scale > 0.0) ||
                                    !std::isfinite(prior.scale) ||
                                    !std::isfinite(blk_prec))) {
                n_degenerate_scale++;
                continue;
            }

            // Extract current values for this block
            int block_size = prior.param_indices.size();
            Eigen::VectorXd f(block_size);
            for (int i = 0; i < block_size; i++) {
                f(i) = params[prior.param_indices[i]];
            }

            // ESS slice target: the ellipse (nu ~ N(0, scale^2 I), or the
            // fitted N(center, diag(sd^2))) already carries a Gaussian factor on
            // this block, so the slice threshold must be the full log-posterior
            // with exactly that factor removed (log_ellipse). ESS then targets
            // N(f; ellipse) * exp(lp - log_ellipse(f)) = exp(lp), the block
            // conditional itself, for ANY ellipse: the correction's contract is
            // with the ellipse (both read `prior`), not with whatever the
            // templated priors contribute for this block. The ellipse decides
            // how closely it tracks the block's conditional, i.e. how often a
            // proposal clears the slice, not what is sampled. Passing the full
            // posterior uncorrected is what double-counts the block prior and
            // over-shrinks; the normalizer is constant in f and cancels in the
            // slice.
            auto log_lik_fn = [&](const Eigen::VectorXd& f_new) -> double {
                std::vector<double> params_temp = params;
                for (int i = 0; i < block_size; i++) {
                    params_temp[prior.param_indices[i]] = f_new(i);
                }
                double lp = compute_log_post_double(params_temp, data, layout);
                return lp - prior.log_ellipse(f_new);  // remove the ellipse
            };

            // Perform ESS step. The slice level must be this block's target at
            // the current f, evaluated with the SAME log_lik_fn (block prior
            // removed) and the current params (which already include earlier
            // blocks' updates this sweep), so compute it directly.
            int n_evals = 0;
            bool exhausted = false;
            double cur_block_loglik = log_lik_fn(f);
            Eigen::VectorXd f_new = ess_step(f, prior, log_lik_fn,
                                             cur_block_loglik, n_evals, exhausted);
            if (exhausted) n_slice_exhausted++;

            // Update params
            for (int i = 0; i < block_size; i++) {
                params[prior.param_indices[i]] = f_new(i);
            }

            total_ess_evals += n_evals;
            total_ess_steps++;
        }

        // Update log-posterior after ESS updates
        current_log_post = compute_log_post_double(params, data, layout);

        // ----------------------------------------------------------------
        // RWMH updates for non-Gaussian parameters
        // ----------------------------------------------------------------
        for (size_t i = 0; i < non_gaussian.size(); i++) {
            int idx = non_gaussian[i];
            bool accepted = false;
            current_log_post = rwmh_step(
                params, idx, adaptive.proposal_sds[i],
                [&](const std::vector<double>& p) { return compute_log_post_double(p, data, layout); },
                current_log_post, accepted
            );
            adaptive.record(i, accepted);
        }

        // ----------------------------------------------------------------
        // Random-effect moves (see above): the slice moves, then the exact
        // fixed-effect / group-level translation.
        // ----------------------------------------------------------------
        auto log_post_of = [&](const std::vector<double>& p) {
            return compute_log_post_double(p, data, layout);
        };
        for (auto& mv : slice_moves) {
            if (mv.kind == 3) {
                if (mv.pn.ready)
                    current_log_post = ess_re_pncp_move(
                        params, re_terms[mv.term], mv.coef, mv.pn, mv.width,
                        log_post_of, current_log_post);
            } else if (mv.kind != 1) {
                double delta = 0.0;
                current_log_post = ess_re_scale_move(
                    params, re_terms[mv.term], mv.coef, mv.kind == 2, mv.width,
                    log_post_of, current_log_post, delta);
            } else {
                current_log_post = ess_fixed_shift_move(
                    params, shifts[mv.term], re_terms, layout, mv.width,
                    log_post_of, current_log_post);
            }
        }
        for (const auto& term : re_terms) {
            bool accepted = false;
            current_log_post = ess_translate_alias(
                params, term, data, layout, log_post_of, current_log_post,
                accepted);
        }
        if (iter < config.n_warmup && config.adapt_during_warmup &&
            pncp_read_now(iter)) {
            for (auto& mv : slice_moves)
                if (mv.kind == 3)
                    ess_pncp_read(params, re_terms[mv.term], mv.coef,
                                  log_post_of, mv.pn);
        }
        if (iter < config.n_warmup && config.adapt_during_warmup &&
            !slice_moves.empty()) {
            slice_n++;
            for (auto& mv : slice_moves) {
                const double v = params[mv.tracked];
                mv.sum += v;
                mv.sumsq += v * v;
                if (slice_n >= 20 && iter % config.adapt_interval == 0) {
                    const double m = mv.sum / slice_n;
                    const double var = mv.sumsq / slice_n - m * m;
                    if (var > 0.0 && std::isfinite(var))
                        mv.width = std::max(3.0 * std::sqrt(var), 1e-2);
                }
            }
        }

        // Adapt RWMH proposals during warmup, and fit the block ellipses.
        if (iter < config.n_warmup && config.adapt_during_warmup) {
            adaptive.adapt(iter);
            ellipse_fit.observe(iter, params, gaussian_priors);
        }

        // ----------------------------------------------------------------
        // Store sample
        // ----------------------------------------------------------------
        if (iter >= config.n_warmup && (iter - config.n_warmup) % thin == 0) {
            for (int j = 0; j < n_params; j++) {
                result.samples(save_idx, j) = params[j];
            }
            result.log_lik[save_idx] = current_log_post;
            save_idx++;
        }

        // Progress
        if (config.verbose && (iter + 1) % config.print_every == 0) {
            Rcpp::Rcout << "Iter " << (iter + 1) << "/" << config.n_iter;
            if (iter < config.n_warmup) {
                Rcpp::Rcout << " (warmup)";
            }
            Rcpp::Rcout << " log_post = " << current_log_post << "\n";
        }
    }

    result.n_slice_evals = total_ess_evals;
    result.avg_slice_evals = total_ess_steps > 0 ?
        static_cast<double>(total_ess_evals) / total_ess_steps : 0.0;
    result.n_slice_exhausted = n_slice_exhausted;
    result.n_degenerate_scale = n_degenerate_scale;

    if (config.verbose) {
        Rcpp::Rcout << "ESS complete. Avg slice evals per step: "
                    << result.avg_slice_evals << "\n";
    }

    // Reported once for the whole run: a per-sweep warning from inside the
    // sampler is both noise and, under options(warn = 2), a longjmp through the
    // sampler loop's own C++ frames.
    if (n_slice_exhausted > 0 || n_degenerate_scale > 0) {
        char msg[320];
        std::snprintf(msg, sizeof(msg),
            "ESS: %d of %d block updates returned unchanged (the slice level "
            "was not orderable, or the bracket collapsed without an "
            "acceptance), and %d were skipped on a prior scale that underflowed "
            "to zero. Those blocks did not move.",
            n_slice_exhausted, total_ess_steps, n_degenerate_scale);
        Rcpp::warning(std::string(msg));
    }

    return result;
}

} // namespace tulpa_ess

#endif // TULPA_ESS_SAMPLER_H
