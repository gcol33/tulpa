// sampler_model_data.h
// Builds the full ModelData + ParamLayout the model-agnostic sampler kernels
// (NUTS / ESS / SGHMC / SGLD / MCLMC / SMC / VI) consume, threading the latent
// structure -- random effects (intercept / slopes / correlated / multi-term),
// areal spatial fields (ICAR / BYM2), and temporal fields (RW1 / RW2 / AR1) --
// through the same built-in-family spec scaffold the fixed-effect path uses
// (gcol33/tulpa#75). The kernels need no change: they already sample the full
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

#include "tulpa/likelihood.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include "tulpa/types.h"
#include "tulpa/autodiff_arena.h"
#include "tulpa/autodiff_fwd.h"
#include "laplace_builtin_family_spec.h"  // builtin_family_spec / BuiltinFamilyResponse
#include "builtin_family_ll_ad.h"         // builtin_family_ll_ad / builtin_family_has_ad
#include "re_structure.h"                 // populate_re_structure
#include "hmc_sampler.h"                  // tulpa_hmc::compute_param_layout
#include <Rcpp.h>
#include <string>
#include <vector>

namespace tulpa {

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
inline void build_sampler_model_inputs(
    SamplerModelInputs& in,
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X,
    const std::string& family, double phi, double sigma_beta,
    const std::vector<double>& offset,        // empty => no offset
    double sigma_re_scale,
    const Rcpp::Nullable<Rcpp::List>& re_spec,
    const Rcpp::Nullable<Rcpp::List>& spatial_spec,
    const Rcpp::Nullable<Rcpp::List>& temporal_spec
) {
    const int N = y.size();
    const int p = X.ncol();

    // --- Process design (fixed effects) + optional offset. ---
    ProcessData proc;
    proc.p = p;
    proc.X_flat.resize((size_t)N * p);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < p; j++)
            proc.X_flat[(size_t)i * p + j] = X(i, j);
    if (!offset.empty()) proc.offset.assign(offset.begin(), offset.end());

    // --- Built-in family spec; add the AD likelihood for analytic gradients
    //     where the family/link is one of the AD-covered closed forms. ---
    in.spec = builtin_family_spec(family);
    if (builtin_family_has_ad(family)) {
        in.spec.ll_arena = &builtin_family_ll_ad<tulpa::arena::Var>;
        in.spec.ll_fwd   = &builtin_family_ll_ad<::fwd::Dual>;
    }

    in.n_trials.assign(n_trials.begin(), n_trials.end());
    in.resp.y        = y.begin();
    in.resp.n_trials = in.n_trials.data();
    in.resp.N        = N;
    in.resp.family   = family;
    in.resp.phi      = phi;
    in.resp.weights  = nullptr;

    in.data = ModelData();
    in.data.n_processes         = 1;
    in.data.processes.push_back(proc);
    in.data.N                   = N;
    in.data.sigma_beta          = sigma_beta;
    in.data.sigma_re_scale      = sigma_re_scale;
    in.data.re_parameterization = 1;          // non-centered, as the layout/prior expect
    in.data.likelihood_spec     = &in.spec;
    in.data.model_response_data = &in.resp;
    in.data.sharing.init(1);
    in.data.zi_type = ZIType::NONE;

    // --- Random effects (multi-term, intercept / slopes / correlated). ---
    if (re_spec.isNotNull()) {
        Rcpp::List re = Rcpp::as<Rcpp::List>(re_spec);
        Rcpp::List      idx_list = Rcpp::as<Rcpp::List>(re["idx"]);
        Rcpp::IntegerVector ng   = Rcpp::as<Rcpp::IntegerVector>(re["ngroups"]);
        Rcpp::IntegerVector nc   = Rcpp::as<Rcpp::IntegerVector>(re["ncoefs"]);
        Rcpp::LogicalVector cor  = Rcpp::as<Rcpp::LogicalVector>(re["correlated"]);
        const int K = ng.size();
        std::vector<int>  ngroups(ng.begin(), ng.end());
        std::vector<int>  ncoefs(nc.begin(), nc.end());
        std::vector<bool> correlated(K, false);
        for (int t = 0; t < K; t++) correlated[t] = (cor[t] == TRUE);
        Rcpp::Nullable<Rcpp::List> Zl = R_NilValue;
        if (re.containsElementNamed("Z") && !Rf_isNull(re["Z"])) {
            Zl = Rcpp::Nullable<Rcpp::List>(Rcpp::as<Rcpp::List>(re["Z"]));
        }
        populate_re_structure(in.data, N, idx_list, ngroups, ncoefs, Zl, correlated);
        // Single intercept-only term: compute_param_layout's single-RE branch
        // sizes the RE block from the legacy data.n_re_groups, and the generic
        // eta assembler reads the legacy per-obs data.re_group for that branch
        // (the multi / slopes branches read re_group_multi_flat). Set both.
        if (K == 1 && !in.data.has_re_slopes) {
            in.data.n_re_groups = in.data.re_n_groups_multi[0];
            Rcpp::IntegerVector gi0 = Rcpp::as<Rcpp::IntegerVector>(idx_list[0]);
            in.data.re_group.assign(gi0.begin(), gi0.end());
        }
    }

    // --- Areal spatial field (ICAR / BYM2). ---
    if (spatial_spec.isNotNull()) {
        Rcpp::List sp = Rcpp::as<Rcpp::List>(spatial_spec);
        std::string stype = Rcpp::as<std::string>(sp["type"]);
        Rcpp::IntegerVector sidx = Rcpp::as<Rcpp::IntegerVector>(sp["spatial_idx"]);
        Rcpp::IntegerVector rp   = Rcpp::as<Rcpp::IntegerVector>(sp["adj_row_ptr"]);
        Rcpp::IntegerVector ci   = Rcpp::as<Rcpp::IntegerVector>(sp["adj_col_idx"]);
        Rcpp::IntegerVector nn   = Rcpp::as<Rcpp::IntegerVector>(sp["n_neighbors"]);
        in.data.n_spatial_units = Rcpp::as<int>(sp["n_spatial_units"]);
        in.data.spatial_group.assign(sidx.begin(), sidx.end());
        in.data.adj_row_ptr.assign(rp.begin(), rp.end());
        in.data.adj_col_idx.assign(ci.begin(), ci.end());
        in.data.n_neighbors.assign(nn.begin(), nn.end());
        if (stype == "icar") {
            in.data.spatial_type = SpatialType::ICAR;
        } else if (stype == "bym2") {
            in.data.spatial_type = SpatialType::BYM2;
            in.data.bym2_scale_factor =
                sp.containsElementNamed("scale_factor")
                    ? Rcpp::as<double>(sp["scale_factor"]) : 1.0;
        } else {
            Rcpp::stop("build_sampler_model_inputs: spatial type '%s' is not "
                       "supported on the sampler path (use 'icar' or 'bym2').",
                       stype.c_str());
        }
    }

    // --- Temporal field (RW1 / RW2 / AR1). ---
    if (temporal_spec.isNotNull()) {
        Rcpp::List tp = Rcpp::as<Rcpp::List>(temporal_spec);
        std::string ttype = Rcpp::as<std::string>(tp["type"]);
        Rcpp::IntegerVector tidx = Rcpp::as<Rcpp::IntegerVector>(tp["time_idx"]);
        in.data.temporal_time_idx.assign(tidx.begin(), tidx.end());
        in.data.n_times = Rcpp::as<int>(tp["n_times"]);
        in.data.n_temporal_groups = Rcpp::as<int>(tp["n_groups"]);
        if (tp.containsElementNamed("group_idx") && !Rf_isNull(tp["group_idx"])) {
            Rcpp::IntegerVector gidx = Rcpp::as<Rcpp::IntegerVector>(tp["group_idx"]);
            in.data.temporal_group_idx.assign(gidx.begin(), gidx.end());
        }
        in.data.temporal_cyclic =
            tp.containsElementNamed("cyclic") && Rcpp::as<bool>(tp["cyclic"]);
        in.data.n_temporal_params = in.data.n_times * in.data.n_temporal_groups;
        if (ttype == "rw1")      in.data.temporal_type = TemporalType::RW1;
        else if (ttype == "rw2") in.data.temporal_type = TemporalType::RW2;
        else if (ttype == "ar1") in.data.temporal_type = TemporalType::AR1;
        else Rcpp::stop("build_sampler_model_inputs: temporal type '%s' is not "
                        "supported on the sampler path (use 'rw1'/'rw2'/'ar1').",
                        ttype.c_str());
    }

    in.layout = tulpa_hmc::compute_param_layout(in.data);
}

// Column names for the full sampler parameter vector, walking the ParamLayout's
// own index fields (the single source of truth for which column is which) so the
// names line up with the draws regardless of how compute_param_layout ordered
// the blocks. Hyperparameters keep their sampled scale in the name (log_sigma_re
// is on the log scale, logit_rho_* on the logit scale); back-transform downstream.
inline Rcpp::CharacterVector sampler_param_names(
    const ModelData& data, const ParamLayout& layout,
    const Rcpp::Nullable<Rcpp::CharacterVector>& fixed_names
) {
    const int D = layout.total_params;
    Rcpp::CharacterVector nm(D);
    for (int k = 0; k < D; k++) nm[k] = "param[" + std::to_string(k + 1) + "]";

    auto set = [&](int idx, const std::string& s) {
        if (idx >= 0 && idx < D) nm[idx] = s;
    };

    // Fixed effects.
    Rcpp::CharacterVector fx;
    bool have_fx = fixed_names.isNotNull();
    if (have_fx) fx = Rcpp::as<Rcpp::CharacterVector>(fixed_names);
    for (size_t kp = 0; kp < layout.process_beta_start.size(); kp++) {
        const int start = layout.process_beta_start[kp];
        const int count = layout.process_beta_count[kp];
        for (int j = 0; j < count; j++) {
            if (have_fx && (start + j) < fx.size())
                set(start + j, Rcpp::as<std::string>(fx[start + j]));
            else
                set(start + j, "beta[" + std::to_string(start + j + 1) + "]");
        }
    }

    // Random effects.
    if (layout.has_re) {
        const int n_terms = (int)layout.re_start_multi.size();
        if (layout.has_re_slopes && n_terms > 0) {
            for (int t = 0; t < n_terms; t++) {
                const int q = layout.re_n_coefs_multi[t];
                for (int c = 0; c < (int)layout.log_sigma_re_slopes[t].size(); c++)
                    set(layout.log_sigma_re_slopes[t][c],
                        "log_sigma_re[t" + std::to_string(t + 1) + ".c" + std::to_string(c + 1) + "]");
                if (layout.chol_re_start_multi[t] >= 0)
                    for (int k = layout.chol_re_start_multi[t]; k < layout.chol_re_end_multi[t]; k++)
                        set(k, "L_re[t" + std::to_string(t + 1) + "." +
                               std::to_string(k - layout.chol_re_start_multi[t] + 1) + "]");
                int e = 0;
                for (int j = layout.re_start_multi[t]; j < layout.re_end_multi[t]; j++) {
                    const int g = e / (q > 0 ? q : 1), c = e % (q > 0 ? q : 1);
                    set(j, "re[t" + std::to_string(t + 1) + ".g" + std::to_string(g + 1) +
                           ".c" + std::to_string(c + 1) + "]");
                    e++;
                }
            }
        } else if (n_terms > 1) {
            for (int t = 0; t < n_terms; t++) {
                set(layout.log_sigma_re_multi[t], "log_sigma_re[t" + std::to_string(t + 1) + "]");
                int g = 0;
                for (int j = layout.re_start_multi[t]; j < layout.re_end_multi[t]; j++)
                    set(j, "re[t" + std::to_string(t + 1) + ".g" + std::to_string(++g) + "]");
            }
        } else {
            set(layout.log_sigma_re_idx, "log_sigma_re");
            int g = 0;
            for (int j = layout.re_start; j < layout.re_end; j++)
                set(j, "re[" + std::to_string(++g) + "]");
        }
    }

    // Spatial.
    if (layout.has_spatial) {
        if (layout.is_bym2) {
            set(layout.log_sigma_bym2_idx, "log_sigma_spatial");
            set(layout.logit_rho_bym2_idx, "logit_rho_bym2");
            int u = 0;
            for (int j = layout.spatial_start; j < layout.spatial_end; j++)
                set(j, "phi_spatial[" + std::to_string(++u) + "]");
            u = 0;
            for (int j = layout.theta_bym2_start; j < layout.theta_bym2_end; j++)
                set(j, "theta_spatial[" + std::to_string(++u) + "]");
        } else {
            set(layout.log_tau_spatial_idx, "log_tau_spatial");
            int u = 0;
            for (int j = layout.spatial_start; j < layout.spatial_end; j++)
                set(j, "phi_spatial[" + std::to_string(++u) + "]");
        }
    }

    // Temporal.
    if (layout.has_temporal) {
        set(layout.log_tau_temporal_idx, "log_tau_temporal");
        if (layout.is_ar1) set(layout.logit_rho_ar1_idx, "logit_rho_ar1");
        int u = 0;
        for (int j = layout.temporal_start; j < layout.temporal_end; j++)
            set(j, "phi_temporal[" + std::to_string(++u) + "]");
    }

    return nm;
}

} // namespace tulpa

#endif // TULPA_SAMPLER_MODEL_DATA_H
