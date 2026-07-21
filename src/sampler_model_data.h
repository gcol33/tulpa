// sampler_model_data.h
// Builds the full ModelData + ParamLayout the model-agnostic sampler kernels
// (NUTS / ESS / SGHMC / SGLD / MCLMC / SMC / VI) consume, threading the latent
// structure -- random effects (intercept / slopes / correlated / multi-term),
// areal spatial fields (ICAR / BYM2), and temporal fields (RW1 / RW2 / AR1) --
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
#include "hmc_hsgp.h"                      // tulpa_hsgp::setup_hsgp_2d (HSGP basis)
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
    const Rcpp::Nullable<Rcpp::List>& temporal_spec,
    const Rcpp::Nullable<Rcpp::List>& svc_spec = R_NilValue,
    const Rcpp::Nullable<Rcpp::List>& tvc_spec = R_NilValue,
    const Rcpp::Nullable<Rcpp::List>& zi_spec = R_NilValue
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

    // --- Zero inflation. Unlike the spec-driven Laplace path -- which has no
    //     `logit_zi` channel and so carries the ZI predictor as a second
    //     process -- the sampler paths pass it to the likelihood callback
    //     directly, from X_zi_flat and the beta_zi parameter block that
    //     hmc_param_layout sizes off zi_type. builtin_family_ll_ad wraps the
    //     base density in the mixture whenever zi_type != NONE. ---
    in.data.zi_type = ZIType::NONE;
    if (zi_spec.isNotNull()) {
        Rcpp::List zs = Rcpp::as<Rcpp::List>(zi_spec);
        Rcpp::NumericMatrix X_zi = Rcpp::as<Rcpp::NumericMatrix>(zs["X"]);
        if (X_zi.nrow() != N) {
            Rcpp::stop("zi_spec$X has %d rows but y has length %d.",
                       (int)X_zi.nrow(), N);
        }
        const int p_zi = X_zi.ncol();
        if (p_zi > 0) {
            in.data.p_zi = p_zi;
            in.data.X_zi_flat.resize((size_t)N * p_zi);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < p_zi; j++)
                    in.data.X_zi_flat[(size_t)i * p_zi + j] = X_zi(i, j);
            if (zs.containsElementNamed("prior_sd")) {
                in.data.zi_prior_sd = Rcpp::as<double>(zs["prior_sd"]);
            }
            // The engine's ZI_* enum values select which mechanism a model
            // package computes; the built-in mixture is the same for every
            // count family, so the family-specific tag only has to be non-NONE
            // for the layout to size beta_zi and the callback to see the logit.
            in.data.zi_type = ZIType::ZI_POISSON;
        }
    }

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

    // --- Spatial field: areal (ICAR / BYM2) or continuous NNGP (gp / nngp). ---
    if (spatial_spec.isNotNull()) {
        Rcpp::List sp = Rcpp::as<Rcpp::List>(spatial_spec);
        std::string stype = Rcpp::as<std::string>(sp["type"]);
        if (stype == "icar" || stype == "bym2") {
            Rcpp::IntegerVector sidx = Rcpp::as<Rcpp::IntegerVector>(sp["spatial_idx"]);
            Rcpp::IntegerVector rp   = Rcpp::as<Rcpp::IntegerVector>(sp["adj_row_ptr"]);
            Rcpp::IntegerVector ci   = Rcpp::as<Rcpp::IntegerVector>(sp["adj_col_idx"]);
            Rcpp::IntegerVector nn   = Rcpp::as<Rcpp::IntegerVector>(sp["n_neighbors"]);
            in.data.n_spatial_units = Rcpp::as<int>(sp["n_spatial_units"]);
            in.data.spatial_group.assign(sidx.begin(), sidx.end());
            in.data.adj_row_ptr.assign(rp.begin(), rp.end());
            in.data.adj_col_idx.assign(ci.begin(), ci.end());
            in.data.n_neighbors.assign(nn.begin(), nn.end());
            in.data.n_spatial_components = tulpa::count_graph_components(
                in.data.n_spatial_units, in.data.adj_row_ptr.data(),
                in.data.adj_col_idx.data());
            if (stype == "icar") {
                in.data.spatial_type = SpatialType::ICAR;
            } else {
                in.data.spatial_type = SpatialType::BYM2;
                in.data.bym2_scale_factor =
                    sp.containsElementNamed("scale_factor")
                        ? Rcpp::as<double>(sp["scale_factor"]) : 1.0;
            }
        } else if (stype == "gp" || stype == "nngp") {
            // Continuous single-scale NNGP field. compute_param_layout keys the
            // GP block on spatial_type == GP (not just has_gp), then allocates
            // log_sigma2_gp / log_phi_gp / the n_obs field. GPData conventions
            // match the hmc_gp kernels: coords row-major; nn_idx / nn_dist
            // row-major [n_loc x nn]; nn_neighbor_dist row-major [i, j1, j2] (the
            // R layer aperms the [n_loc,nn,nn] array); nn_order 0-based; centred
            // (gp_parameterization = 0) so the stored field draws stay valid.
            in.data.spatial_type = SpatialType::GP;
            auto& g = in.data.gp_data;
            Rcpp::NumericMatrix coords = Rcpp::as<Rcpp::NumericMatrix>(sp["coords"]);
            const int n_loc = coords.nrow();
            g.n_obs = n_loc;
            g.nn = Rcpp::as<int>(sp["nn"]);
            g.coords.resize(2 * (std::size_t)n_loc);
            for (int i = 0; i < n_loc; ++i) {
                g.coords[2 * (std::size_t)i]     = coords(i, 0);
                g.coords[2 * (std::size_t)i + 1] = coords(i, 1);
            }
            Rcpp::IntegerMatrix nnix = Rcpp::as<Rcpp::IntegerMatrix>(sp["nn_idx"]);
            Rcpp::NumericMatrix nnd  = Rcpp::as<Rcpp::NumericMatrix>(sp["nn_dist"]);
            g.nn_idx.resize((std::size_t)n_loc * g.nn);
            g.nn_dist.resize((std::size_t)n_loc * g.nn);
            for (int i = 0; i < n_loc; ++i)
                for (int j = 0; j < g.nn; ++j) {
                    g.nn_idx[(std::size_t)i * g.nn + j]  = nnix(i, j);
                    g.nn_dist[(std::size_t)i * g.nn + j] = nnd(i, j);
                }
            Rcpp::NumericVector nnnd =
                Rcpp::as<Rcpp::NumericVector>(sp["nn_neighbor_dist"]);
            g.nn_neighbor_dist.assign(nnnd.begin(), nnnd.end());
            Rcpp::IntegerVector nord = Rcpp::as<Rcpp::IntegerVector>(sp["nn_order"]);
            g.nn_order.assign(nord.begin(), nord.end());
            Rcpp::IntegerVector nordi =
                Rcpp::as<Rcpp::IntegerVector>(sp["nn_order_inv"]);
            g.nn_order_inv.assign(nordi.begin(), nordi.end());
            Rcpp::IntegerVector otl = Rcpp::as<Rcpp::IntegerVector>(sp["obs_to_loc"]);
            g.obs_to_loc.assign(otl.begin(), otl.end());
            g.cov_type = static_cast<CovType>(Rcpp::as<int>(sp["cov_type"]));
            g.nu = Rcpp::as<double>(sp["nu"]);
            in.data.has_gp = true;
            in.data.gp_parameterization = 0;
            in.data.gp_phi_prior_U     = Rcpp::as<double>(sp["phi_prior_U"]);
            in.data.gp_phi_prior_alpha = Rcpp::as<double>(sp["phi_prior_alpha"]);
            if (sp.containsElementNamed("sigma2_prior_U"))
                in.data.gp_sigma2_prior_U = Rcpp::as<double>(sp["sigma2_prior_U"]);
            if (sp.containsElementNamed("sigma2_prior_alpha"))
                in.data.gp_sigma2_prior_alpha =
                    Rcpp::as<double>(sp["sigma2_prior_alpha"]);
        } else if (stype == "car_proper") {
            // Proper CAR: Q(rho) = D - rho W, full-rank (PD) so the field mean
            // is identified (no ICAR sum-to-zero). compute_param_layout keys
            // the CAR block on spatial_type == CAR_PROPER, allocating log_tau,
            // logit_rho_car and the n_spatial_units field. The generic-NUTS
            // log-prior needs a differentiable log|Q(rho)|; the eigenvalues of
            // the symmetric normalized adjacency (mu_i, precomputed in R since
            // they are fixed data) turn it into sum_i log(1 - rho mu_i), which
            // autodiff handles in closed form -- no per-gradient Cholesky.
            Rcpp::IntegerVector sidx = Rcpp::as<Rcpp::IntegerVector>(sp["spatial_idx"]);
            Rcpp::IntegerVector rp   = Rcpp::as<Rcpp::IntegerVector>(sp["adj_row_ptr"]);
            Rcpp::IntegerVector ci   = Rcpp::as<Rcpp::IntegerVector>(sp["adj_col_idx"]);
            Rcpp::IntegerVector nn   = Rcpp::as<Rcpp::IntegerVector>(sp["n_neighbors"]);
            in.data.n_spatial_units = Rcpp::as<int>(sp["n_spatial_units"]);
            in.data.spatial_group.assign(sidx.begin(), sidx.end());
            in.data.adj_row_ptr.assign(rp.begin(), rp.end());
            in.data.adj_col_idx.assign(ci.begin(), ci.end());
            in.data.n_neighbors.assign(nn.begin(), nn.end());
            in.data.n_spatial_components = tulpa::count_graph_components(
                in.data.n_spatial_units, in.data.adj_row_ptr.data(),
                in.data.adj_col_idx.data());
            in.data.spatial_type = SpatialType::CAR_PROPER;
            Rcpp::NumericVector eig = Rcpp::as<Rcpp::NumericVector>(sp["adj_eigenvalues"]);
            in.data.car_adj_eigenvalues.assign(eig.begin(), eig.end());
            in.data.car_rho_lower = Rcpp::as<double>(sp["rho_lower"]);
            in.data.car_rho_upper = Rcpp::as<double>(sp["rho_upper"]);
        } else if (stype == "hsgp") {
            // Hilbert-space GP. compute_param_layout keys the HSGP block on
            // spatial_type == HSGP, allocating log_sigma2_hsgp /
            // log_lengthscale_hsgp / m_total basis coefficients; the field is
            // evaluated per observation (phi_flat[i, j] * scaled beta_j) so no
            // obs->location map is needed. The Laplacian basis is built here by
            // setup_hsgp_2d -- the single source of truth every HSGP path uses
            // -- so the basis math is never duplicated in R.
            in.data.spatial_type = SpatialType::HSGP;
            Rcpp::NumericMatrix coords = Rcpp::as<Rcpp::NumericMatrix>(sp["coords"]);
            const int n_obs = coords.nrow();
            std::vector<double> flat(2 * (std::size_t)n_obs);
            for (int i = 0; i < n_obs; ++i) {
                flat[2 * (std::size_t)i]     = coords(i, 0);
                flat[2 * (std::size_t)i + 1] = coords(i, 1);
            }
            const int m    = Rcpp::as<int>(sp["m"]);
            const double cc = Rcpp::as<double>(sp["c"]);
            tulpa_hsgp::setup_hsgp_2d(flat, n_obs, m, cc, /*shared=*/true,
                                      in.data.hsgp_data);
            in.data.has_hsgp = true;
        } else {
            Rcpp::stop("build_sampler_model_inputs: spatial type '%s' is not "
                       "supported on the sampler path (use 'icar'/'bym2'/"
                       "'car_proper'/'gp'/'nngp'/'hsgp').", stype.c_str());
        }
    }

    // --- Multiscale temporal (trend + seasonal + short-term). ---
    // A validated temporal_multiscale() spec carries its own field names
    // (time_index / group_index, and the three component strings), so it is
    // parsed before the single-component branch reads `time_idx`. The layout
    // (compute_param_layout) and prior (compute_multiscale_temporal_prior) key
    // off data.has_multiscale_temporal, which nothing else in tulpa sets.
    if (temporal_spec.isNotNull() &&
        Rcpp::as<std::string>(Rcpp::as<Rcpp::List>(temporal_spec)["type"]) ==
            "multiscale") {
        Rcpp::List tp = Rcpp::as<Rcpp::List>(temporal_spec);
        auto& ms = in.data.multiscale_temporal_data;

        Rcpp::IntegerVector tidx = Rcpp::as<Rcpp::IntegerVector>(tp["time_index"]);
        ms.time_index.assign(tidx.begin(), tidx.end());
        ms.n_obs   = static_cast<int>(tidx.size());
        ms.n_times = Rcpp::as<int>(tp["n_times"]);
        ms.n_groups = tp.containsElementNamed("n_groups") &&
                      !Rf_isNull(tp["n_groups"])
                          ? Rcpp::as<int>(tp["n_groups"]) : 1;
        if (tp.containsElementNamed("group_index") && !Rf_isNull(tp["group_index"])) {
            Rcpp::IntegerVector gidx = Rcpp::as<Rcpp::IntegerVector>(tp["group_index"]);
            ms.group_index.assign(gidx.begin(), gidx.end());
        } else {
            ms.group_index.assign(ms.n_obs, 1);
        }

        const std::string trend = Rcpp::as<std::string>(tp["trend"]);
        if      (trend == "rw1")  ms.trend_type = TemporalType::RW1;
        else if (trend == "rw2")  ms.trend_type = TemporalType::RW2;
        else if (trend == "none") ms.trend_type = TemporalType::NONE;
        else Rcpp::stop("build_sampler_model_inputs: multiscale trend '%s' is "
                        "not supported (use 'rw1'/'rw2'/'none').", trend.c_str());

        ms.seasonal_period =
            (tp.containsElementNamed("seasonal") && !Rf_isNull(tp["seasonal"]))
                ? Rcpp::as<int>(tp["seasonal"]) : 0;

        const std::string st = Rcpp::as<std::string>(tp["short_term"]);
        if      (st == "ar1")  ms.short_term_type = TemporalType::AR1;
        else if (st == "iid")  ms.short_term_type = TemporalType::IID;
        else if (st == "none") ms.short_term_type = TemporalType::NONE;
        else Rcpp::stop("build_sampler_model_inputs: multiscale short_term '%s' "
                        "is not supported (use 'ar1'/'iid'/'none').", st.c_str());

        if (ms.trend_type == TemporalType::NONE && ms.seasonal_period == 0 &&
            ms.short_term_type == TemporalType::NONE)
            Rcpp::stop("build_sampler_model_inputs: the multiscale temporal "
                       "spec has no active component.");

        ms.shared = !tp.containsElementNamed("shared") ||
                    Rf_isNull(tp["shared"]) || Rcpp::as<bool>(tp["shared"]);
        in.data.has_multiscale_temporal = true;
    }
    // --- Temporal field (RW1 / RW2 / AR1). ---
    else if (temporal_spec.isNotNull()) {
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
        // AR1 rho prior: Beta(a, b) on u = (rho + 1)/2 (default (1, 1) = uniform).
        if (tp.containsElementNamed("rho_prior_a"))
            in.data.ar1_rho_prior_a = Rcpp::as<double>(tp["rho_prior_a"]);
        if (tp.containsElementNamed("rho_prior_b"))
            in.data.ar1_rho_prior_b = Rcpp::as<double>(tp["rho_prior_b"]);
        in.data.n_temporal_params = in.data.n_times * in.data.n_temporal_groups;
        if (ttype == "rw1")      in.data.temporal_type = TemporalType::RW1;
        else if (ttype == "rw2") in.data.temporal_type = TemporalType::RW2;
        else if (ttype == "ar1") in.data.temporal_type = TemporalType::AR1;
        else Rcpp::stop("build_sampler_model_inputs: temporal type '%s' is not "
                        "supported on the sampler path (use 'rw1'/'rw2'/'ar1').",
                        ttype.c_str());
    }

    // --- Spatially-varying coefficients (NNGP). compute_param_layout keys the
    //     SVC block on data.has_svc, allocating log_sigma2_svc / log_phi_svc /
    //     the [n_svc x n_obs] fields; the generic log-post adds
    //     eta_i += sum_j X_svc[i,j] * w_j(s_i). NNGP conventions match the GP
    //     field (coords row-major; nn_idx 1-based; nn_order 0-based); the SVC
    //     kernel derives neighbour-pair distances from coords, so no
    //     nn_neighbor_dist is needed. ---
    if (svc_spec.isNotNull()) {
        Rcpp::List sv = Rcpp::as<Rcpp::List>(svc_spec);
        auto& s = in.data.svc_data;
        Rcpp::NumericMatrix coords = Rcpp::as<Rcpp::NumericMatrix>(sv["coords"]);
        const int n_obs = coords.nrow();
        const int n_svc = Rcpp::as<int>(sv["n_svc"]);
        s.n_obs = n_obs;
        s.n_svc = n_svc;
        s.nn = Rcpp::as<int>(sv["nn"]);
        s.coords.resize(2 * (std::size_t)n_obs);
        for (int i = 0; i < n_obs; ++i) {
            s.coords[2 * (std::size_t)i]     = coords(i, 0);
            s.coords[2 * (std::size_t)i + 1] = coords(i, 1);
        }
        Rcpp::IntegerMatrix nnix = Rcpp::as<Rcpp::IntegerMatrix>(sv["nn_idx"]);
        Rcpp::NumericMatrix nnd  = Rcpp::as<Rcpp::NumericMatrix>(sv["nn_dist"]);
        s.nn_idx.resize((std::size_t)n_obs * s.nn);
        s.nn_dist.resize((std::size_t)n_obs * s.nn);
        for (int i = 0; i < n_obs; ++i)
            for (int j = 0; j < s.nn; ++j) {
                s.nn_idx[(std::size_t)i * s.nn + j]  = nnix(i, j);
                s.nn_dist[(std::size_t)i * s.nn + j] = nnd(i, j);
            }
        Rcpp::IntegerVector nord = Rcpp::as<Rcpp::IntegerVector>(sv["nn_order"]);
        s.nn_order.assign(nord.begin(), nord.end());
        Rcpp::IntegerVector nordi = Rcpp::as<Rcpp::IntegerVector>(sv["nn_order_inv"]);
        s.nn_order_inv.assign(nordi.begin(), nordi.end());
        Rcpp::IntegerVector svci = Rcpp::as<Rcpp::IntegerVector>(sv["svc_indices"]);
        s.svc_indices.assign(svci.begin(), svci.end());
        Rcpp::NumericVector xsvc = Rcpp::as<Rcpp::NumericVector>(sv["X_svc"]);
        s.X_svc.assign(xsvc.begin(), xsvc.end());   // row-major [n_obs x n_svc]
        s.cov_type = static_cast<CovType>(Rcpp::as<int>(sv["cov_type"]));
        in.data.has_svc = true;
        in.data.svc_is_hsgp = false;
        in.data.svc_phi_prior_U     = Rcpp::as<double>(sv["phi_prior_U"]);
        in.data.svc_phi_prior_alpha = Rcpp::as<double>(sv["phi_prior_alpha"]);
        if (sv.containsElementNamed("sigma2_prior_scale"))
            in.data.svc_sigma2_prior_scale = Rcpp::as<double>(sv["sigma2_prior_scale"]);
    }

    // --- Temporally-varying coefficients (RW1 / RW2 / AR1). compute_param_layout
    //     keys the TVC block on data.has_tvc, allocating log_tau_tvc, optional
    //     logit_rho_tvc (AR1) and the [n_groups x n_tvc x n_times] fields; the
    //     generic log-post adds eta_i += sum_j X_tvc[i,j] * w_j(g_i, t_i). ---
    if (tvc_spec.isNotNull()) {
        Rcpp::List tv = Rcpp::as<Rcpp::List>(tvc_spec);
        auto& t = in.data.tvc_data;
        t.n_obs   = N;
        t.n_times = Rcpp::as<int>(tv["n_times"]);
        t.n_tvc   = Rcpp::as<int>(tv["n_tvc"]);
        t.n_groups = Rcpp::as<int>(tv["n_groups"]);
        Rcpp::IntegerVector ti = Rcpp::as<Rcpp::IntegerVector>(tv["time_index"]);
        t.time_index.assign(ti.begin(), ti.end());
        Rcpp::IntegerVector gi = Rcpp::as<Rcpp::IntegerVector>(tv["group_index"]);
        t.group_index.assign(gi.begin(), gi.end());
        Rcpp::IntegerVector tvci = Rcpp::as<Rcpp::IntegerVector>(tv["tvc_indices"]);
        t.tvc_indices.assign(tvci.begin(), tvci.end());
        Rcpp::NumericVector xtvc = Rcpp::as<Rcpp::NumericVector>(tv["X_tvc"]);
        t.X_tvc.assign(xtvc.begin(), xtvc.end());   // row-major [n_obs x n_tvc]
        std::string st = Rcpp::as<std::string>(tv["structure"]);
        if (st == "rw1")      t.structure = TemporalType::RW1;
        else if (st == "rw2") t.structure = TemporalType::RW2;
        else if (st == "ar1") t.structure = TemporalType::AR1;
        else Rcpp::stop("build_sampler_model_inputs: TVC structure '%s' is not "
                        "supported (use 'rw1'/'rw2'/'ar1').", st.c_str());
        t.cyclic = tv.containsElementNamed("cyclic") && Rcpp::as<bool>(tv["cyclic"]);
        in.data.has_tvc = true;
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

    // Zero-inflation coefficients. Named to match the R front door, which
    // prefixes the ziformula design columns with "zi_"; when those names are
    // supplied they are appended to fixed_names and picked up above, so this
    // only supplies the positional fallback.
    if (layout.has_zi && layout.beta_zi_start >= 0) {
        for (int k = layout.beta_zi_start; k < layout.beta_zi_end; k++) {
            if (nm[k] == "param[" + std::to_string(k + 1) + "]") {
                set(k, "beta_zi[" +
                       std::to_string(k - layout.beta_zi_start + 1) + "]");
            }
        }
    }

    // Multi-scale temporal: trend / seasonal / short-term arms plus their
    // scales, so a fitted multiscale block is addressable by name rather than
    // by position in the latent vector.
    if (layout.has_multiscale_temporal) {
        set(layout.log_sigma2_trend_idx,    "log_sigma2_trend");
        set(layout.log_sigma2_seasonal_idx, "log_sigma2_seasonal");
        set(layout.log_sigma2_short_idx,    "log_sigma2_short");
        set(layout.logit_rho_short_idx,     "logit_rho_short");
        for (int k = layout.trend_start; k >= 0 && k < layout.trend_end; k++)
            set(k, "trend[" + std::to_string(k - layout.trend_start + 1) + "]");
        for (int k = layout.seasonal_start; k >= 0 && k < layout.seasonal_end; k++)
            set(k, "seasonal[" +
                   std::to_string(k - layout.seasonal_start + 1) + "]");
        for (int k = layout.short_term_start; k >= 0 && k < layout.short_term_end; k++)
            set(k, "short_term[" +
                   std::to_string(k - layout.short_term_start + 1) + "]");
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
            if (layout.is_car_proper)
                set(layout.logit_rho_car_idx, "logit_rho_car");
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

    // Continuous GP / NNGP field (one column per unique location).
    if (layout.is_gp) {
        set(layout.log_sigma2_gp_idx, "log_sigma2_gp");
        set(layout.log_phi_gp_idx, "log_phi_gp");
        int u = 0;
        for (int j = layout.gp_w_start; j < layout.gp_w_end; j++)
            set(j, "gp_w[" + std::to_string(++u) + "]");
    }

    // Hilbert-space GP field (m_total basis coefficients).
    if (layout.is_hsgp) {
        set(layout.log_sigma2_hsgp_idx, "log_sigma2_hsgp");
        set(layout.log_lengthscale_hsgp_idx, "log_lengthscale_hsgp");
        int u = 0;
        for (int j = layout.hsgp_beta_start; j < layout.hsgp_beta_end; j++)
            set(j, "hsgp_beta[" + std::to_string(++u) + "]");
    }

    // Spatially-varying coefficients: per-term (sigma2, phi) hypers then the
    // per-term NNGP field, laid out w_flat[j * n_obs + i].
    if (layout.has_svc && data.svc_data.n_svc > 0) {
        const int n_svc = data.svc_data.n_svc;
        for (int j = 0; j < n_svc; j++) {
            set(layout.log_sigma2_svc_start + j,
                "log_sigma2_svc[" + std::to_string(j + 1) + "]");
            set(layout.log_phi_svc_start + j,
                "log_phi_svc[" + std::to_string(j + 1) + "]");
        }
        int u = 0;
        for (int q = layout.svc_w_start; q < layout.svc_w_end; q++)
            set(q, "svc_w[" + std::to_string(++u) + "]");
    }

    // Temporally-varying coefficients: per-term log_tau (+ logit_rho for AR1)
    // then the per-group/term/time field.
    if (layout.has_tvc && data.tvc_data.n_tvc > 0) {
        const int n_tvc = data.tvc_data.n_tvc;
        for (int j = 0; j < n_tvc; j++)
            set(layout.log_tau_tvc_start + j,
                "log_tau_tvc[" + std::to_string(j + 1) + "]");
        if (layout.logit_rho_tvc_start >= 0)
            for (int j = 0; j < n_tvc; j++)
                set(layout.logit_rho_tvc_start + j,
                    "logit_rho_tvc[" + std::to_string(j + 1) + "]");
        int u = 0;
        for (int q = layout.tvc_w_start; q < layout.tvc_w_end; q++)
            set(q, "tvc_w[" + std::to_string(++u) + "]");
    }

    return nm;
}

} // namespace tulpa

#endif // TULPA_SAMPLER_MODEL_DATA_H
