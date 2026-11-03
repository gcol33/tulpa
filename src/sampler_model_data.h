// sampler_model_data.h
// Builds the full ModelData + ParamLayout the model-agnostic sampler kernels
// (NUTS / ESS / SGHMC / SGLD / MCLMC / SMC / VI) consume, threading the latent
// structure -- random effects (intercept / slopes / correlated / multi-term),
// areal spatial fields (ICAR / BYM2), and temporal fields (RW1 / RW2 / AR1 /
// GP over irregular times) --
// through the same built-in-family spec scaffold the fixed-effect path uses
//. The kernels need no change: they already sample the full
// parameter vector compute_param_layout() lays out and score it through the
// generic log-posterior (priors + hyperpriors all included), so populating the
// ModelData and deriving the layout is the entire job.
//
// The hyperparameters (log_sigma_re per term/coef, log_tau_spatial / BYM2
// sigma+rho, log_tau_temporal / AR1 rho) are sampled JOINTLY with the latent
// effects and fixed effects -- this is the exact-MCMC tier doing full Bayes over
// the variance components, not conditioning on them like the Laplace / logpost
// backends. The half-Cauchy / Gamma / Uniform hyperpriors live in the templated
// priors (tulpa_priors_re.h / _icar.h / _temporal.h) and enter the same density
// the kernels score, so jointly sampling them is proper.

#ifndef TULPA_SAMPLER_MODEL_DATA_H
#define TULPA_SAMPLER_MODEL_DATA_H

#include <cmath>
#include <limits>
#include "tulpa/likelihood.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include "tulpa/types.h"
#include "tulpa/autodiff_arena.h"
#include "tulpa/autodiff_fwd.h"
#include "laplace_builtin_family_spec.h"  // builtin_family_spec / BuiltinFamilyResponse
#include "builtin_family_ll_ad.h"         // builtin_family_ll_ad / builtin_family_has_ad
#include "re_structure.h"                 // populate_re_structure
#include "icar_kernel.h"                   // count_graph_components
#include "areal_input_check.h"             // check_areal_inputs
#include "linalg_fast.h"                    // require_coords_2col
#include "hmc_hsgp.h"                      // tulpa_hsgp::setup_hsgp_2d (HSGP basis)
#include "hmc_sampler.h"                  // tulpa_hmc::compute_param_layout
#include "pc_prior.h"                     // tulpa::pc_anchors_valid, pc_rate
#include "tulpa_priors_re.h"               // re_term_view
#include "tulpa_priors_mstemporal.h"        // multiscale_*_noncentered
#include <Rcpp.h>
#include <string>
#include <vector>

namespace tulpa {

// Read a PC-prior anchor pair off a spec list, leaving the defaults in place
// when the spec does not carry them, and reject a pair the calibration cannot
// represent.
//
// lambda = -log(alpha)/U is defined for U > 0 and alpha in (0, 1) only. Outside
// that the density is -Inf or NaN at every value of the scale, inside a
// gradient, with nothing naming the parameter that caused it. The anchors are
// fixed for the whole fit, so this is the one place the check can both throw
// and say which argument was wrong -- pc_prior.h's own guard runs in AD code
// and OpenMP regions, where it can only fall back to a flat prior.
inline void read_pc_anchors(const Rcpp::List& spec,
                            const char* key_U, const char* key_alpha,
                            double& U, double& alpha, const char* what) {
    if (spec.containsElementNamed(key_U))
        U = Rcpp::as<double>(spec[key_U]);
    if (spec.containsElementNamed(key_alpha))
        alpha = Rcpp::as<double>(spec[key_alpha]);
    if (!pc_anchors_valid(U, alpha)) {
        Rcpp::stop("build_sampler_model_inputs: %s PC prior anchors must "
                   "satisfy `%s` > 0 and 0 < `%s` < 1 (got %g and %g). The "
                   "pair calibrates P(sigma > U) = alpha.",
                   what, key_U, key_alpha, U, alpha);
    }
}

// read_pc_anchors() for a pair with no default: the spec must carry both.
inline void require_pc_anchors(const Rcpp::List& spec,
                               const char* key_U, const char* key_alpha,
                               double& U, double& alpha, const char* what) {
    if (!spec.containsElementNamed(key_U) ||
        !spec.containsElementNamed(key_alpha)) {
        Rcpp::stop("build_sampler_model_inputs: the %s prior needs `%s` and "
                   "`%s` on the spec.", what, key_U, key_alpha);
    }
    read_pc_anchors(spec, key_U, key_alpha, U, alpha, what);
}

// The per-observation field-node map for a sampler-side GP spec is 0-based:
// the R layer passes `obs_to_loc - 1`, and the log-posterior reads
// `w[obs_to_loc[i]]` with no bound of its own.
inline void check_obs_to_loc_0based(
    const Rcpp::IntegerVector& otl, int n_obs, int n_loc, const char* who
) {
    if (static_cast<int>(otl.size()) != n_obs) {
        Rcpp::stop("%s: length(obs_to_loc) (%d) must equal the number of "
                   "observations (%d).", who,
                   static_cast<int>(otl.size()), n_obs);
    }
    for (int i = 0; i < n_obs; i++) {
        if (otl[i] == NA_INTEGER || otl[i] < 0 || otl[i] >= n_loc) {
            Rcpp::stop("%s: obs_to_loc[%d] (%d) is outside [0, %d). The "
                       "sampler-side node map is 0-based.", who, i + 1,
                       otl[i], n_loc);
        }
    }
}

// A neighbour table arrives from R as `n_loc x nn` and is copied into a flat
// buffer indexed by (i, j) against the `nn` the SAME spec carries in a separate
// field. Rcpp::Matrix::operator()(i, j) resolves to i + nrows * j with no bounds
// test, so a table whose shape disagrees with that `nn` reads past the SEXP data
// block instead of erroring. Checked once here, where the field that is wrong
// can still be named.
inline void check_nn_tables(
    const Rcpp::IntegerMatrix& idx, const Rcpp::NumericMatrix& dist,
    int n_loc, int nn, const char* who,
    const char* idx_field = "nn_idx", const char* dist_field = "nn_dist"
) {
    if (nn < 0) {
        Rcpp::stop("%s: nn (%d) must be non-negative.", who, nn);
    }
    if (idx.nrow() != n_loc || idx.ncol() != nn) {
        Rcpp::stop("%s: %s is %d x %d; must be %d x %d.", who, idx_field,
                   static_cast<int>(idx.nrow()), static_cast<int>(idx.ncol()),
                   n_loc, nn);
    }
    if (dist.nrow() != n_loc || dist.ncol() != nn) {
        Rcpp::stop("%s: %s is %d x %d; must be %d x %d.", who, dist_field,
                   static_cast<int>(dist.nrow()), static_cast<int>(dist.ncol()),
                   n_loc, nn);
    }
}

// A vector copied wholesale and then indexed downstream against a dimension the
// spec carries elsewhere. The kernels that read these either guard by falling
// back to -Inf or do not guard at all, so the length disagreement is worth an
// error at the boundary rather than a fit that silently reports nothing.
inline void check_vec_len(R_xlen_t got, R_xlen_t want,
                          const char* who, const char* field) {
    if (got != want) {
        Rcpp::stop("%s: length(%s) is %d; must be %d.", who, field,
                   static_cast<int>(got), static_cast<int>(want));
    }
}

// The permutation and its inverse are read as location offsets, so both their
// length and their values are part of the contract.
inline void check_nn_order(const Rcpp::IntegerVector& ord, int n_loc,
                           const char* who, const char* field) {
    check_vec_len(ord.size(), n_loc, who, field);
    for (int i = 0; i < n_loc; i++) {
        if (ord[i] == NA_INTEGER || ord[i] < 0 || ord[i] >= n_loc) {
            Rcpp::stop("%s: %s[%d] (%d) is outside [0, %d). The sampler-side "
                       "ordering is 0-based.", who, field, i + 1, ord[i], n_loc);
        }
    }
}

// The kernel a continuous-time temporal GP is built from, read off a spec list
// under `cov` / `nu` / `period`. One reader, asked by `temporal_gp()` and by a
// GP-evolving temporally-varying coefficient, so the two cannot come to accept
// different sets of kernels or different smoothnesses. Matern is closed-form
// only at nu in {1/2, 3/2, 5/2} (see temporal_gp_kernel.h); R rejects the rest
// at construction and this is the backstop for a consumer building the spec by
// hand. `who` names the door in the message.
struct TemporalGPKernel {
    TemporalCovType cov_type = TemporalCovType::EXPONENTIAL;
    double nu = 0.5;
    double period = 1.0;
};

inline TemporalGPKernel read_temporal_gp_kernel(const Rcpp::List& spec,
                                                const char* who) {
    static const tulpa::EnumEntry<tulpa::TemporalCovType> cov_table[] = {
        {"exponential", tulpa::TemporalCovType::EXPONENTIAL},
        {"matern",      tulpa::TemporalCovType::MATERN},
        {"gaussian",    tulpa::TemporalCovType::GAUSSIAN},
        {"periodic",    tulpa::TemporalCovType::PERIODIC}};
    TemporalGPKernel k;
    k.cov_type = tulpa::parse_enum(
        Rcpp::as<std::string>(spec["cov"]), cov_table,
        tulpa::TemporalCovType::EXPONENTIAL);
    if (spec.containsElementNamed("nu") && !Rf_isNull(spec["nu"]))
        k.nu = Rcpp::as<double>(spec["nu"]);
    if (spec.containsElementNamed("period") && !Rf_isNull(spec["period"]))
        k.period = Rcpp::as<double>(spec["period"]);
    if (k.cov_type == tulpa::TemporalCovType::MATERN &&
        std::abs(k.nu - 0.5) > 1e-12 &&
        std::abs(k.nu - 1.5) > 1e-12 &&
        std::abs(k.nu - 2.5) > 1e-12) {
        Rcpp::stop("%s: Matern smoothness nu = %g has no closed form here; use "
                   "0.5, 1.5 or 2.5.", who, k.nu);
    }
    if (k.cov_type == tulpa::TemporalCovType::PERIODIC && !(k.period > 0.0)) {
        Rcpp::stop("%s: the periodic kernel's period must be positive (got %g).",
                   who, k.period);
    }
    return k;
}

// A GP lengthscale's bounded support, read as `phi_prior_lower` /
// `phi_prior_upper` and refused unless lower < upper. The bounds hold the
// interval the logit map runs over, so an inverted pair is a support the
// sampler cannot be in rather than a wide one.
inline void read_temporal_gp_phi_bounds(const Rcpp::List& spec, const char* who,
                                        double& lower, double& upper) {
    if (spec.containsElementNamed("phi_prior_lower"))
        lower = Rcpp::as<double>(spec["phi_prior_lower"]);
    if (spec.containsElementNamed("phi_prior_upper"))
        upper = Rcpp::as<double>(spec["phi_prior_upper"]);
    if (!(lower < upper)) {
        Rcpp::stop("%s: lengthscale bounds must satisfy lower < upper "
                   "(got %g, %g).", who, lower, upper);
    }
}

// Spec-solver-style inputs kept alive together: data borrows spec & resp (and
// resp borrows y / n_trials), so the whole struct must outlive the kernel run.
// `y` aliases the caller's NumericVector (no copy) -- it must outlive too.
struct SamplerModelInputs {
    LikelihoodSpec        spec;
    BuiltinFamilyResponse resp;
    std::vector<int>      n_trials;   // stable storage behind resp.n_trials
    ModelData             data;
    ParamLayout           layout;
};

// Build the [beta | RE | spatial | temporal | hyperparameters] ModelData +
// ParamLayout for a built-in-family GLM with optional latent structure. The RE /
// spatial / temporal specs are R lists (null when absent); see build_sampler...
// in tulpa_sample_glmm.cpp for the field contract.
void build_sampler_model_inputs(
    SamplerModelInputs& in,
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X,
    const std::string& family, double phi, double phi2, double sigma_beta,
    const std::vector<double>& offset,        // empty => no offset
    double sigma_re_scale,
    const Rcpp::Nullable<Rcpp::List>& re_spec,
    const Rcpp::Nullable<Rcpp::List>& spatial_spec,
    const Rcpp::Nullable<Rcpp::List>& temporal_spec,
    const Rcpp::Nullable<Rcpp::List>& svc_spec = R_NilValue,
    const Rcpp::Nullable<Rcpp::List>& tvc_spec = R_NilValue,
    const Rcpp::Nullable<Rcpp::List>& zi_spec = R_NilValue
);

// Starting positions for parameters whose support is a declared interval,
// where the coordinate's own origin is not a sensible place to begin.
//
// Start a multi-scale range at the geometric mean of its declared bounds
// rather than at the origin's phi = 1.
//
// This began as a correctness fix: the block used to reject a range outside
// (range_*_lower, range_*_upper) with a hard -INFINITY, so for bounds
// excluding 1 the chain started outside its own support, never moved, and
// returned an all-zero field at a 100% divergence rate. That wall is gone --
// each scale now carries a PC range prior proper on (0, inf) -- so phi = 1 is
// always a valid start and this is no longer load-bearing. It stays because
// the bounds are the user's own statement of where that scale lives, and
// starting a regional range of a few units at 1 wastes warmup walking there. A
// caller-supplied init is left alone: an explicit starting position is the
// caller's choice, including for a resumed chain.
// A continuous-time GP's lengthscale starts at a FRACTION of the spread of its
// own time instants rather than at the midpoint of the declared support. The
// default support is (0.01, 10) and both GP doors standardize their time
// values, so the midpoint is a lengthscale five times the data's own spread:
// every pair of instants is then correlated to within rounding, the dense
// T x T covariance is numerically rank-one, and its Cholesky jitter binds at
// the starting point.
//
// Measured on a 24-instant irregular fixture, deviation of the runtime
// gradient check on the lengthscale against the starting ratio
// lengthscale / sd(time): Matern 5/2 clean at 0.1 / 0.25 / 0.5 / 1.0, then
// 9.3e-04 at 2.0 and 2.4e-03 at 5.0; Gaussian clean at 0.1 / 0.25, then
// 1.2e-03 at 0.5, 3.2e-03 at 1.0, 6.5e-03 at 2.0 and 8.3e-03 at 5.0 -- the
// smoothness ladder of a floor binding, not a wrong derivative. The fraction
// is read against the SD, which is what those ratios are stated in, and 0.2
// clears the smoothest kernel's threshold of 0.25 with margin.
//
// Only the CENTERED parameterization evaluates that log-determinant in the
// target, so only it fell back to numerical gradients; the non-centered
// default reaches the Cholesky through `f = L z` alone. Both start here, since
// a lengthscale five times the spread is not a place either wants to begin
// (gcol33/tulpa#847, gcol33/tulpa#851).
constexpr double kGpPhiInitFraction = 0.2;

// The logit of `fraction * sd(time_values)` on the support (lower, upper),
// or NA when the instants carry no spread or the bounds are not an interval.
inline double gp_phi_logit_init(const std::vector<double>& time_values,
                                double b_lo, double b_hi) {
    if (time_values.size() < 2 || !(b_lo < b_hi)) return NA_REAL;
    double mean = 0.0;
    for (double v : time_values) mean += v;
    mean /= static_cast<double>(time_values.size());
    double ss = 0.0;
    for (double v : time_values) ss += (v - mean) * (v - mean);
    const double spread =
        std::sqrt(ss / static_cast<double>(time_values.size() - 1));
    if (!(spread > 0.0)) return NA_REAL;

    // Held off both ends of the support: the logit of an endpoint is infinite.
    const double pad = 1e-3 * (b_hi - b_lo);
    double phi0 = kGpPhiInitFraction * spread;
    phi0 = std::min(std::max(phi0, b_lo + pad), b_hi - pad);
    const double u = (phi0 - b_lo) / (b_hi - b_lo);
    return std::log(u / (1.0 - u));
}

inline void init_gp_lengthscales(std::vector<double>& q,
                                 const ModelData& data,
                                 const ParamLayout& layout) {
    if (layout.logit_phi_tvc_gp_start >= 0) {
        const double l0 = gp_phi_logit_init(data.tvc_data.time_values,
                                            data.tvc_gp_phi_prior_lower,
                                            data.tvc_gp_phi_prior_upper);
        if (R_finite(l0)) {
            for (int j = layout.logit_phi_tvc_gp_start;
                 j < layout.logit_phi_tvc_gp_end; j++) {
                q[j] = l0;
            }
        }
    }
    if (layout.is_temporal_gp && layout.logit_phi_temporal_gp_idx >= 0) {
        const double l0 =
            gp_phi_logit_init(data.temporal_gp_data.time_values,
                              data.temporal_gp_phi_prior_lower,
                              data.temporal_gp_phi_prior_upper);
        if (R_finite(l0)) q[layout.logit_phi_temporal_gp_idx] = l0;
    }
}

// The log of a scale prior's median SD: Exponential(rate) on the SD (the PC
// prior, rate -log(alpha) / U, and the latent factors' prior) has median
// log(2) / rate, Half-Cauchy(0, s) has median s. NaN outside the prior's domain.
inline double exponential_median_log_sd(double rate) {
    if (!(rate > 0.0) || !std::isfinite(rate)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::log(std::log(2.0) / rate);
}
inline double pc_median_log_sd(double U, double alpha) {
    if (!pc_anchors_valid(U, alpha)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return exponential_median_log_sd(pc_rate(U, alpha));
}
inline double half_cauchy_median_log_sd(double scale) {
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::log(scale);
}

// Every non-centred field -- one whose values are its scale times sampled
// coordinates of unit prior scale -- starts with its scale at the median of
// that scale's prior rather than at the coordinate 0 (sigma = 1). The
// multi-chain start disperses each coordinate by Uniform(-2, 2), so a chain
// starts with its field moved by the scale times that offset: at a scale far
// above the posterior's, the likelihood pays for a field of that size, and
// the energy of a trajectory started there is spent along the scale, whose
// prior falls off only linearly in its log as the scale shrinks. A chain
// carried far down that direction in its first iteration stays there through
// warmup. On a gaussian RW2 + iid multiscale fit (six data sets, three seeds)
// with the short-term scale started at sigma = 1, 6 of 18 fits had a chain
// whose log-variance reached -21 to -217 (Rhat up to 1.6); at the PC median,
// 0 of 18. A log-variance coordinate starts at 2 log(median), a log-SD
// coordinate at log(median).
inline void init_noncentered_scales(std::vector<double>& q,
                                    const ModelData& data,
                                    const ParamLayout& layout) {
    const int D = static_cast<int>(q.size());
    auto set = [&q, D](int idx, double log_sd, double per_log_sd) {
        if (idx >= 0 && idx < D && std::isfinite(log_sd)) {
            q[idx] = per_log_sd * log_sd;
        }
    };
    auto set_span = [&set](int lo, int hi, double log_sd, double per_log_sd) {
        if (lo < 0) return;
        for (int j = lo; j < hi; j++) set(j, log_sd, per_log_sd);
    };

    if (layout.has_re) {
        const double m = half_cauchy_median_log_sd(data.sigma_re_scale);
        for (int t = 0; t < priors::re_n_terms(data); t++) {
            const priors::ReTermView v = priors::re_term_view(data, layout, t);
            if (data.re_parameterization != 1 &&
                !priors::re_term_correlated(v.chol_start, v.n_coefs)) {
                continue;
            }
            for (int c = 0; c < v.n_coefs; c++) set(v.log_sigma_idx[c], m, 1.0);
        }
    }
    if (layout.is_bym2 && !layout.is_bym2_collapsed) {
        set(layout.log_sigma_bym2_idx,
            half_cauchy_median_log_sd(data.sigma_re_scale), 1.0);
    }
    if (layout.is_gp && !layout.is_gp_collapsed && data.gp_parameterization == 1) {
        set(layout.log_sigma2_gp_idx,
            pc_median_log_sd(data.gp_sigma2_prior_U, data.gp_sigma2_prior_alpha),
            2.0);
    }
    if (layout.is_hsgp) {
        set(layout.log_sigma2_hsgp_idx,
            pc_median_log_sd(data.hsgp_sigma2_prior_U,
                             data.hsgp_sigma2_prior_alpha), 2.0);
    }
    if (layout.is_multiscale_gp &&
        (data.msgp_parameterization == 1 || data.msgp_is_hsgp)) {
        set(layout.log_sigma2_gp_local_idx,
            pc_median_log_sd(data.ms_sigma2_local_prior_U,
                             data.ms_sigma2_local_prior_alpha), 2.0);
        set(layout.log_sigma2_gp_regional_idx,
            pc_median_log_sd(data.ms_sigma2_regional_prior_U,
                             data.ms_sigma2_regional_prior_alpha), 2.0);
    }
    if (layout.has_svc && (data.svc_parameterization == 1 || data.svc_is_hsgp)) {
        set_span(layout.log_sigma2_svc_start, layout.log_sigma2_svc_end,
                 half_cauchy_median_log_sd(data.svc_sigma2_prior_scale), 2.0);
    }
    if (layout.is_temporal_gp && data.temporal_gp_parameterization == 1) {
        set(layout.log_sigma2_temporal_gp_idx,
            pc_median_log_sd(data.temporal_gp_sigma2_prior_U,
                             data.temporal_gp_sigma2_prior_alpha), 2.0);
    }
    if (layout.has_multiscale_temporal) {
        if (priors::multiscale_trend_noncentered(data, layout)) {
            set(layout.log_sigma2_trend_idx,
                pc_median_log_sd(data.ms_sigma2_trend_prior_U,
                                 data.ms_sigma2_trend_prior_alpha), 2.0);
        }
        if (priors::multiscale_seasonal_noncentered(data, layout)) {
            set(layout.log_sigma2_seasonal_idx,
                pc_median_log_sd(data.ms_sigma2_seasonal_prior_U,
                                 data.ms_sigma2_seasonal_prior_alpha), 2.0);
        }
        if (priors::multiscale_short_noncentered(data, layout)) {
            set(layout.log_sigma2_short_idx,
                pc_median_log_sd(data.ms_sigma2_short_prior_U,
                                 data.ms_sigma2_short_prior_alpha), 2.0);
        }
    }
    if (layout.has_latent && data.latent_n_factors > 0) {
        set_span(layout.log_sigma_latent_start, layout.log_sigma_latent_end,
                 exponential_median_log_sd(data.latent_sigma_prior_rate), 1.0);
    }
}

inline void init_bounded_support_params(std::vector<double>& q,
                                        const ModelData& data,
                                        const ParamLayout& layout) {
    init_gp_lengthscales(q, data, layout);
    init_noncentered_scales(q, data, layout);

    if (!layout.is_multiscale_gp || !data.has_multiscale_gp || data.msgp_is_hsgp) {
        return;
    }
    const MultiscaleGPData& ms = data.multiscale_gp_data;
    auto log_geometric_mean = [](double lo, double hi) {
        return 0.5 * (std::log(lo) + std::log(hi));
    };
    if (layout.log_phi_gp_local_idx >= 0 &&
        ms.range_local_lower > 0.0 &&
        ms.range_local_upper > ms.range_local_lower) {
        q[layout.log_phi_gp_local_idx] =
            log_geometric_mean(ms.range_local_lower, ms.range_local_upper);
    }
    if (layout.log_phi_gp_regional_idx >= 0 &&
        ms.range_regional_lower > 0.0 &&
        ms.range_regional_upper > ms.range_regional_lower) {
        q[layout.log_phi_gp_regional_idx] =
            log_geometric_mean(ms.range_regional_lower, ms.range_regional_upper);
    }
}

// Column names for the full sampler parameter vector, walking the ParamLayout's
// own index fields (the single source of truth for which column is which) so the
// names line up with the draws regardless of how compute_param_layout ordered
// the blocks. Hyperparameters keep their sampled scale in the name (log_sigma_re
// is on the log scale, logit_rho_* on the logit scale); back-transform downstream.
Rcpp::CharacterVector sampler_param_names(
    const ModelData& data, const ParamLayout& layout,
    const Rcpp::Nullable<Rcpp::CharacterVector>& fixed_names
);

} // namespace tulpa

#endif // TULPA_SAMPLER_MODEL_DATA_H
