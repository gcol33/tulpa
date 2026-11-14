// tulpa_generic_sampler.cpp
// Proof-of-concept: generic model fitting via LikelihoodFn callback
// Demonstrates the tulpa multi-process interface end-to-end

#include <Rcpp.h>
#include "fixed_effect_prior.h"
#include <R_ext/Rdynload.h>
#include <vector>
#include <cmath>
#include "hmc_chain_stack.h"
#include "hmc_sampler.h"
#include "log_post_impl.h"
#include "linalg_fast.h"
#include "tulpa/likelihood.h"
#include "tulpa/autodiff_arena.h"
#include "tulpa/nuts_api.h"
#include "sampler_model_data.h"
#include <tulpa/field_setup_api.h>
#include "hmc_field_coordinates.h"
#include "hmc_spatial_marginal.h"
#include "sampler_field_store.h"
#include "sampler_resume.h"

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

// [[Rcpp::export]]
Rcpp::List cpp_test_temporal_coordinates(Rcpp::NumericVector z, int n_times,
    int n_groups, std::string structure, bool cyclic, double log_tau,
    double logit_rho = 0.3, bool group_centering = true) {
    ModelData data;
    data.n_processes = 1;
    data.processes.resize(1);
    data.sharing.init(1);
    data.n_times = n_times;
    data.n_temporal_groups = n_groups;
    data.n_temporal_params = n_times * n_groups;
    data.temporal_group_centering = group_centering;
    data.temporal_cyclic = cyclic;
    if (structure == "rw1") data.temporal_type = tulpa::TemporalType::RW1;
    else if (structure == "rw2") data.temporal_type = tulpa::TemporalType::RW2;
    else if (structure == "ar1") data.temporal_type = tulpa::TemporalType::AR1;
    else if (structure == "iid") data.temporal_type = tulpa::TemporalType::IID;
    else Rcpp::stop("Unknown temporal structure");
    tulpa::LikelihoodSpec spec;
    spec.n_processes = 1;
    data.likelihood_spec = &spec;
    const auto layout = tulpa_hmc::compute_param_layout(data);
    if (z.size() != data.n_temporal_params) Rcpp::stop("z length must match temporal dimensions");
    std::vector<double> q(layout.total_params, 0.0);
    q[layout.log_tau_temporal_idx] = log_tau;
    if (layout.is_ar1) q[layout.logit_rho_ar1_idx] = logit_rho;
    std::copy(z.begin(), z.end(), q.begin() + layout.temporal_start);
    std::vector<double> field;
    double tau = 0, rho = 0, sigma2 = 0, phi = 0;
    data.temporal_parameterization = 1;
    const double nc = tulpa::priors::compute_temporal_prior(q, data, layout,
        field, tau, rho, sigma2, phi);
    std::copy(field.begin(), field.end(), q.begin() + layout.temporal_start);
    data.temporal_parameterization = 0;
    std::vector<double> centered_field;
    const double centered = tulpa::priors::compute_temporal_prior(q, data, layout,
        centered_field, tau, rho, sigma2, phi);
    return Rcpp::List::create(Rcpp::Named("field") = field,
        Rcpp::Named("nc") = nc, Rcpp::Named("centered") = centered);
}

// [[Rcpp::export]]
Rcpp::List cpp_test_field_coordinates(
    Rcpp::NumericVector y, Rcpp::NumericMatrix X,
    Rcpp::Nullable<Rcpp::List> spatial_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> temporal_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> svc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> tvc_spec = R_NilValue,
    double shift = 0.3, int n_iter = 0,
    Rcpp::Nullable<Rcpp::NumericVector> point = R_NilValue,
    bool native_point = true, bool gaussian = false,
    Rcpp::Nullable<Rcpp::NumericVector> working_eta = R_NilValue,
    Rcpp::Nullable<Rcpp::List> latent_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> st_spec = R_NilValue) {
    tulpa::SamplerModelInputs in;
    tulpa::build_sampler_model_inputs(in, y, Rcpp::IntegerVector(y.size(), 1), X,
        "gaussian", 1.0, NA_REAL, 2.5, {}, 1.0, R_NilValue,
        spatial_spec, temporal_spec, svc_spec, tvc_spec, R_NilValue);
    auto& data = in.data;
    if (latent_spec.isNotNull()) {
        const Rcpp::List latent(latent_spec);
        data.has_latent = true;
        data.latent_bilinear = true;
        data.latent_loading_processes = 1;
        data.latent_n_factors = Rcpp::as<int>(latent["n_factors"]);
        if (latent.containsElementNamed("constraint"))
            data.latent_constraint = Rcpp::as<int>(latent["constraint"]);
    }
    if (st_spec.isNotNull()) {
        const Rcpp::List specification(st_spec);
        data.has_spatiotemporal = true;
        data.st_margin_centering = true;
        data.st_parameterization = Rcpp::as<int>(specification["parameterization"]);
        auto& st = data.spatiotemporal_data;
        st.type = static_cast<tulpa::STType>(Rcpp::as<int>(specification["type"]));
        st.temporal_type = static_cast<tulpa::TemporalType>(Rcpp::as<int>(specification["temporal_type"]));
        st.temporal_cyclic = Rcpp::as<bool>(specification["cyclic"]);
        st.n_spatial = Rcpp::as<int>(specification["n_spatial"]);
        st.n_times = Rcpp::as<int>(specification["n_times"]);
        st.n_params = st.n_spatial * st.n_times;
        st.st_flat = Rcpp::as<std::vector<int>>(specification["st_flat"]);
        st.adj_row_ptr = Rcpp::as<std::vector<int>>(specification["adj_row_ptr"]);
        st.adj_col_idx = Rcpp::as<std::vector<int>>(specification["adj_col_idx"]);
        if (st.adj_row_ptr.size() == static_cast<size_t>(st.n_spatial + 1)) {
            st.n_neighbors.resize(st.n_spatial);
            for (int s = 0; s < st.n_spatial; ++s)
                st.n_neighbors[s] = st.adj_row_ptr[s + 1] - st.adj_row_ptr[s];
        }
        if (st.type == tulpa::STType::SEPARABLE || st.type == tulpa::STType::NONSEP_GP) {
            data.st_margin_centering = false;
            data.st_uniform_ranges = true;
            st.nn = Rcpp::as<int>(specification["nn"]);
            st.coords = Rcpp::as<std::vector<double>>(specification["coords"]);
            st.time_values = Rcpp::as<std::vector<double>>(specification["time_values"]);
            st.nn_idx = Rcpp::as<std::vector<int>>(specification["nn_idx"]);
            st.nn_dist_space = Rcpp::as<std::vector<double>>(specification["nn_dist_space"]);
            st.nn_dist_time = Rcpp::as<std::vector<double>>(specification["nn_dist_time"]);
            st.nn_order = Rcpp::as<std::vector<int>>(specification["nn_order"]);
            st.nonsep_type = static_cast<tulpa::NonsepType>(Rcpp::as<int>(specification["nonsep"]));
            st.cov_space = static_cast<tulpa::CovType>(Rcpp::as<int>(specification["cov_space"]));
            st.cov_time = static_cast<tulpa::CovType>(Rcpp::as<int>(specification["cov_time"]));
        }
        if (specification.containsElementNamed("hsgp_coords")) {
            data.st_is_hsgp = true;
            data.st_margin_centering = false;
            data.st_hsgp_coords = Rcpp::as<std::vector<double>>(specification["hsgp_coords"]);
            data.st_hsgp_m_per_dim = Rcpp::as<int>(specification["m"]);
            st.t_idx = Rcpp::as<std::vector<int>>(specification["t_idx"]);
            tulpa::prepare_field(data, tulpa::FieldSetup::ST_HSGP);
            if (specification.containsElementNamed("kernel"))
                data.st_hsgp_data.kernel = Rcpp::as<int>(specification["kernel"]);
        }
    }
    if (data.st_margin_centering) tulpa::setup_st_spatial_partition(data.spatiotemporal_data);
    if (working_eta.isNotNull()) in.spec.working_eta = Rcpp::as<std::vector<double>>(working_eta);
    const auto layout = tulpa_hmc::compute_param_layout(data);
    const int p = layout.total_params;
    std::vector<double> q(p, 0.0);
    tulpa::init_bounded_support_params(q, data, layout);
    for (int i = 0; i < p; ++i) q[i] += shift * std::sin(i + 1.0);
    if (point.isNotNull()) {
        q = Rcpp::as<std::vector<double>>(point);
        if (static_cast<int>(q.size()) != p) Rcpp::stop("point length must match the parameter layout");
    }
    tulpa_hmc::GradientModeFitScope gradient_scope;
    tulpa_hmc::g_gradient_mode = tulpa_hmc::GradientMode::AUTODIFF_ARENA;
    auto map = tulpa_hmc::make_field_coordinates(q, data, layout);
    if (gaussian) {
        map.entries.clear();
        map.operators.clear();
        map.operators.push_back(tulpa_hmc::prepare_spatial_gaussian_coordinate(data, layout));
    }
    if (point.isNotNull() && native_point) map.inverse(q);
    std::vector<double> x;
    const double jacobian = map.forward(q, x);
    const double raw_lp = tulpa_hmc::compute_log_post(x, data, layout);
    std::vector<double> stored = x;
    tulpa::store_field_draw(stored.data(), x, data, layout);
    std::vector<double> st_inverse;
    double st_log_jacobian = NA_REAL;
    std::vector<double> st_normalizer_gradient, st_normalizer_numerical;
    if (data.st_is_hsgp && data.st_parameterization == 1) {
        st_inverse.assign(stored.begin() + layout.st_delta_start, stored.begin() + layout.st_delta_end);
        const double rho = layout.logit_rho_st_idx < 0 ? 0.0 :
            2.0 * tulpa::math::inv_logit(x[layout.logit_rho_st_idx]) - 1.0;
        tulpa_st_hsgp::nc_inverse(st_inverse.data(),
            tulpa::math::safe_exp(x[layout.log_tau_st_idx]), rho,
            tulpa::math::safe_exp(x[layout.log_sigma2_st_hsgp_idx]), tulpa::math::safe_exp(x[layout.log_lengthscale_st_hsgp_idx]),
            data.spatiotemporal_data, data.st_hsgp_data, st_inverse.data(), data.st_hsgp_data.kernel);
    }
    if (layout.is_st_gp && data.st_parameterization == 1) {
        st_inverse.assign(stored.begin() + layout.st_delta_start, stored.begin() + layout.st_delta_end);
        tulpa_st_gp::StGpCoordinates<double> coordinates;
        coordinates.build(1.0 / tulpa::math::safe_exp(x[layout.log_tau_st_idx]),
            tulpa::st_gp_range_from_coordinate(x[layout.log_phi_st_space_idx], data, true),
            tulpa::st_gp_range_from_coordinate(x[layout.log_phi_st_time_idx], data, false),
            data.spatiotemporal_data);
        coordinates.inverse(st_inverse.data(), st_inverse.data());
        st_log_jacobian = coordinates.log_jacobian;
    }
    if (layout.is_st_gp) {
        const double tau = tulpa::math::safe_exp(x[layout.log_tau_st_idx]);
        const double space = tulpa::st_gp_range_from_coordinate(x[layout.log_phi_st_space_idx], data, true);
        const double time = tulpa::st_gp_range_from_coordinate(x[layout.log_phi_st_time_idx], data, false);
        const auto normalizer = tulpa_st_gp::st_gp_margin_normalizer_grad(tau, space, time, data.spatiotemporal_data);
        const auto& st = data.spatiotemporal_data;
        st_normalizer_gradient = {normalizer.log_tau,
            normalizer.log_phi_space * tulpa::field_range_log_slope(x[layout.log_phi_st_space_idx],
                st.phi_space_prior_lower, st.phi_space_prior_upper, data.st_uniform_ranges),
            normalizer.log_phi_time * tulpa::field_range_log_slope(x[layout.log_phi_st_time_idx],
                st.phi_time_prior_lower, st.phi_time_prior_upper, data.st_uniform_ranges)};
        const auto value = [&](const std::vector<double>& point) {
            double out = 0.0;
            tulpa_st_gp::st_gp_margin_log_normalizer(1.0 / tulpa::math::safe_exp(point[layout.log_tau_st_idx]),
                tulpa::st_gp_range_from_coordinate(point[layout.log_phi_st_space_idx], data, true),
                tulpa::st_gp_range_from_coordinate(point[layout.log_phi_st_time_idx], data, false), st, &out);
            return out;
        };
        for (int index : {layout.log_tau_st_idx, layout.log_phi_st_space_idx, layout.log_phi_st_time_idx}) {
            auto plus = x, minus = x;
            const double step = 2e-5;
            plus[index] += step; minus[index] -= step;
            st_normalizer_numerical.push_back((value(plus) - value(minus)) / (2.0 * step));
        }
    }
    std::vector<double> raw_gradient, prior_gradient;
    tulpa_hmc::compute_gradient(x, data, layout, raw_gradient);
    std::vector<double> inverse = x;
    map.inverse(inverse);
    Rcpp::List blocks;
    for (const auto& e : map.entries) {
        blocks.push_back(Rcpp::List::create(
            Rcpp::Named("begin") = e.begin, Rcpp::Named("end") = e.end,
            Rcpp::Named("scale_index") = e.scale_index,
            Rcpp::Named("scale_sign") = e.scale_sign,
            Rcpp::Named("scaled_information") = e.scaled_information,
            Rcpp::Named("fixed_information") = e.fixed_information,
            Rcpp::Named("group_width") = e.width(),
            Rcpp::Named("preserve_constant") = e.preserve_constant,
            Rcpp::Named("preserve_linear") = e.preserve_linear));
    }
    double last_reported_lp = NA_REAL, last_native_lp = NA_REAL;
    Rcpp::LogicalVector auto_dense_equal = Rcpp::LogicalVector::create(NA_LOGICAL);
    Rcpp::List auto_dense_detail;
    if (n_iter > 0) {
        auto sample = [&](tulpa::MassMatrixType metric) {
            return tulpa_hmc::run_hmc_chain_cpp(x, data, layout,
                n_iter, n_iter / 2, 0, 1, 18, false, 8, metric, .9, nullptr, {});
        };
        const auto result = sample(tulpa::MassMatrixType::AUTO);
        last_reported_lp = result.log_prob.back();
        last_native_lp = tulpa_hmc::compute_log_post(result.final_position, data, layout);
        if (layout.is_bym2) {
            const auto dense = sample(tulpa::MassMatrixType::DENSE);
            auto_dense_equal[0] = result.samples_flat == dense.samples_flat &&
                result.n_leapfrog == dense.n_leapfrog && result.energy == dense.energy;
            auto max_diff = [](const std::vector<double>& a, const std::vector<double>& b) {
                if (a.size() != b.size()) return R_PosInf;
                double m = 0.0;
                for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
                return m;
            };
            int first_sample = -1;
            const size_t width = static_cast<size_t>(result.n_params_stored);
            for (size_t i = 0; width && i < std::min(result.samples_flat.size(), dense.samples_flat.size()); ++i)
                if (result.samples_flat[i] != dense.samples_flat[i]) { first_sample = static_cast<int>(i / width); break; }
            auto_dense_detail = Rcpp::List::create(
                Rcpp::Named("samples_max_diff") = max_diff(result.samples_flat, dense.samples_flat),
                Rcpp::Named("first_differing_draw") = first_sample,
                Rcpp::Named("energy_max_diff") = max_diff(result.energy, dense.energy),
                Rcpp::Named("leapfrog_equal") = result.n_leapfrog == dense.n_leapfrog,
                Rcpp::Named("epsilon") = Rcpp::NumericVector::create(result.epsilon, dense.epsilon));
        }
    }
    tulpa_hmc::FieldCoordinateScope coordinate_scope(&map);
    std::vector<double> gradient, numerical, full_prior, numerical_prior;
    double lp = 0;
    tulpa_hmc::compute_gradient(q, data, layout, gradient, &lp);
    tulpa_hmc::compute_gradient_generic_numerical(q, data, layout, numerical, nullptr);
    auto prior_fn = tulpa_hmc::resolve_prior_gradient_fn(
        tulpa_hmc::g_gradient_mode, data, layout);
    prior_fn(q, data, layout, full_prior, nullptr);
    tulpa_hmc::compute_gradient_prior_numerical(q, data, layout, numerical_prior, nullptr);
    const bool verified = tulpa_hmc::verify_gradient_runtime(q, data, layout, 1e-4);
    tulpa::GenericLogPostState<double> state;
    const double prior_lp = tulpa::initialize_generic_state(x, data, layout, state);
    tulpa::precompute_generic_fixed_eta(data, state);
    Rcpp::NumericMatrix eta(data.N, data.n_processes);
    for (int i = 0; i < data.N; ++i) {
        double row[tulpa::MAX_PROCESSES];
        tulpa::generic_eta_at(i, data, layout, state, row);
        for (int k = 0; k < data.n_processes; ++k) eta(i, k) = row[k];
    }
    return Rcpp::List::create(Rcpp::Named("q") = q, Rcpp::Named("x") = x,
        Rcpp::Named("eta") = eta, Rcpp::Named("prior_lp") = prior_lp,
        Rcpp::Named("stored") = stored,
        Rcpp::Named("st_inverse") = st_inverse,
        Rcpp::Named("st_log_jacobian") = st_log_jacobian,
        Rcpp::Named("st_normalizer_gradient") = st_normalizer_gradient,
        Rcpp::Named("st_normalizer_numerical") = st_normalizer_numerical,
        Rcpp::Named("last_reported_lp") = last_reported_lp,
        Rcpp::Named("last_native_lp") = last_native_lp,
        Rcpp::Named("auto_dense_equal") = auto_dense_equal,
        Rcpp::Named("auto_dense_detail") = auto_dense_detail,
        Rcpp::Named("inverse") = inverse, Rcpp::Named("jacobian") = jacobian,
        Rcpp::Named("raw_lp") = raw_lp, Rcpp::Named("lp") = lp,
        Rcpp::Named("spatial_log_normalizer") =
            tulpa::priors::spatial_icar_bym2_prior_log_normalizer(data, layout),
        Rcpp::Named("gradient") = gradient, Rcpp::Named("numerical") = numerical,
        Rcpp::Named("raw_gradient") = raw_gradient,
        Rcpp::Named("prior_gradient") = full_prior,
        Rcpp::Named("numerical_prior") = numerical_prior,
        Rcpp::Named("verified") = verified, Rcpp::Named("blocks") = blocks,
        Rcpp::Named("conditional") = !map.operators.empty(),
        Rcpp::Named("numerical_dispatch") =
            tulpa_hmc::resolve_gradient_fn(tulpa_hmc::GradientMode::NUMERICAL,
                data, layout) == &tulpa_hmc::compute_gradient_generic_numerical);
}

template<typename T>
T unit_gaussian_likelihood(
    int i, const T* eta, const T&, const T&, const std::vector<T>&,
    const ModelData&, const ParamLayout&, const void* response) {
    const auto* y = static_cast<const std::vector<double>*>(response);
    const T residual = T((*y)[i]) - eta[0];
    return T(-0.5) * residual * residual;
}

template<typename T>
void unit_gaussian_weights(int i, const T* eta, tulpa::EtaWeightScalar<T>, tulpa::EtaWeightScalar<T>, const std::vector<T>&,
    const ModelData&, const ParamLayout&, const void* response, T* score, T* weights) {
    const auto& y = *static_cast<const std::vector<double>*>(response);
    score[0] = T(y[i]) - eta[0];
    weights[0] = T(1);
}

// [[Rcpp::export]]
Rcpp::List cpp_test_collapsed_spatial(Rcpp::NumericVector y, Rcpp::NumericMatrix X,
    Rcpp::List spatial_spec, Rcpp::NumericVector point, int n_draw = 0, int seed = 19) {
    tulpa::SamplerModelInputs in;
    tulpa::build_sampler_model_inputs(in, y, Rcpp::IntegerVector(y.size(), 1), X,
        "gaussian", 1.0, NA_REAL, 2.5, {}, 1.0, R_NilValue,
        spatial_spec, R_NilValue, R_NilValue, R_NilValue, R_NilValue);
    std::vector<double> response(y.begin(), y.end()), q(point.begin(), point.end());
    in.spec.ll_double = unit_gaussian_likelihood<double>;
    in.spec.ll_arena = unit_gaussian_likelihood<tulpa::arena::Var>;
    in.spec.eta_weights_fn = unit_gaussian_weights<double>;
    in.spec.eta_weights_arena_fn = unit_gaussian_weights<tulpa::arena::Var>;
    in.spec.gaussian_spatial_marginal = true;
    in.data.model_response_data = &response;
    const auto layout = tulpa_hmc::compute_param_layout(in.data);
    if (!tulpa_hmc::has_collapsed_spatial(layout) || layout.total_params != static_cast<int>(q.size()))
        Rcpp::stop("point must match a collapsed Gaussian spatial layout");
    tulpa_hmc::GradientModeFitScope scope;
    tulpa_hmc::g_gradient_mode = tulpa_hmc::GradientMode::AUTODIFF_ARENA;
    std::vector<double> gradient, numerical, prior_gradient, numerical_prior;
    double lp = 0, prior = 0;
    tulpa_hmc::compute_gradient(q, in.data, layout, gradient, &lp);
    tulpa_hmc::compute_gradient_generic_numerical(q, in.data, layout, numerical, nullptr);
    tulpa_hmc::compute_prior_gradient_collapsed_spatial(q, in.data, layout, prior_gradient, &prior);
    tulpa_hmc::compute_gradient_prior_numerical(q, in.data, layout, numerical_prior, nullptr);
    const bool verified = tulpa_hmc::verify_gradient_runtime(q, in.data, layout, 1e-4);
    const int units = layout.is_gp ? in.data.gp_data.n_obs : in.data.n_spatial_units;
    Rcpp::NumericMatrix phi(n_draw, units), theta(n_draw, layout.is_bym2 ? units : 0);
    Rcpp::NumericVector mode_phi(units), mode_theta(layout.is_bym2 ? units : 0);
    tulpa_hmc::reconstruct_collapsed_spatial(&in.data, &layout, q.data(), 1, seed, false,
        mode_phi.begin(), layout.is_bym2 ? mode_theta.begin() : nullptr);
    if (n_draw > 0) {
        std::vector<double> rows, phi_flat(static_cast<size_t>(n_draw) * units), theta_flat(layout.is_bym2 ? phi_flat.size() : 0);
        for (int i = 0; i < n_draw; ++i) rows.insert(rows.end(), q.begin(), q.end());
        tulpa_hmc::reconstruct_collapsed_spatial(&in.data, &layout, rows.data(), n_draw, seed, true,
            phi_flat.data(), theta_flat.empty() ? nullptr : theta_flat.data());
        for (int i = 0; i < n_draw; ++i) for (int j = 0; j < units; ++j) {
            phi(i, j) = phi_flat[i * units + j];
            if (layout.is_bym2) theta(i, j) = theta_flat[i * units + j];
        }
    }
    return Rcpp::List::create(Rcpp::Named("lp") = lp, Rcpp::Named("prior") = prior,
        Rcpp::Named("target") = tulpa::sampling_target_name(tulpa_hmc::spatial_sampling_target(in.data, layout)),
        Rcpp::Named("inference_tier") = tulpa::sampling_target_tier(tulpa_hmc::spatial_sampling_target(in.data, layout)),
        Rcpp::Named("gradient") = gradient, Rcpp::Named("numerical") = numerical,
        Rcpp::Named("prior_gradient") = prior_gradient, Rcpp::Named("numerical_prior") = numerical_prior,
        Rcpp::Named("verified") = verified,
        Rcpp::Named("normalizer") = tulpa_hmc::compute_log_prior_normalizer(in.data, layout, nullptr),
        Rcpp::Named("mode_phi") = mode_phi,
        Rcpp::Named("mode_theta") = mode_theta, Rcpp::Named("phi") = phi, Rcpp::Named("theta") = theta);
}

// [[Rcpp::export]]
Rcpp::List cpp_test_sampler_controls(
    Rcpp::NumericVector y_r, Rcpp::NumericMatrix X_r,
    int L = 0, std::string metric = "diag", bool walnuts = false,
    int n_chains = 1, int seed = 18, int n_iter = 2000, int n_warmup = 750) {
    std::vector<double> y(y_r.begin(), y_r.end());
    tulpa::LikelihoodSpec spec;
    spec.n_processes = 1;
    spec.ll_double = unit_gaussian_likelihood<double>;
    spec.ll_arena = unit_gaussian_likelihood<tulpa::arena::Var>;
    ModelData data;
    data.N = y.size();
    data.n_processes = 1;
    data.sigma_beta = 2.5;
    data.processes.resize(1);
    auto& process = data.processes[0];
    process.p = X_r.ncol();
    for (int i = 0; i < X_r.nrow(); ++i)
        for (int j = 0; j < X_r.ncol(); ++j)
            process.X_flat.push_back(X_r(i, j));
    data.sharing.init(1);
    data.likelihood_spec = &spec;
    data.model_response_data = &y;
    const auto layout = tulpa_hmc::compute_param_layout(data);
    const int p = layout.total_params;
    tulpa::SamplerControl control;
    control.L = L;
    control.walnuts = walnuts;
    if (metric == "diag") control.metric = tulpa::MassMatrixType::DIAG;
    else if (metric == "dense") control.metric = tulpa::MassMatrixType::DENSE;
    else if (metric == "block_diag") control.metric = tulpa::MassMatrixType::BLOCK_DIAG;
    else if (metric == "auto") control.metric = tulpa::MassMatrixType::AUTO;
    else Rcpp::stop("Unknown metric");
    if (n_chains < 1) Rcpp::stop("n_chains must be positive");
    std::vector<double> init(static_cast<size_t>(n_chains) * p, 0.0);
    std::vector<tulpa::NUTSResult> results(n_chains);
    if (n_chains == 1) {
        tulpa::get_nuts_fn()(&data, &layout, init.data(), p, n_iter, n_warmup,
            8, 0.85, seed, 0, &control, nullptr, 0.0, results.data());
    } else {
        tulpa::get_nuts_chains_fn()(&data, &layout, init.data(), p, n_chains,
            n_iter, n_warmup, 8, 0.85, seed, 0, &control, nullptr, nullptr, results.data());
    }
    std::vector<tulpa_hmc::HMCResultCpp> direct;
    const auto* wc = walnuts ? &control.walnuts_config : nullptr;
    if (n_chains == 1) {
        direct.push_back(tulpa_hmc::run_hmc_chain_cpp(std::vector<double>(p, 0.0),
            data, layout, n_iter, n_warmup, L, 1, seed, false, 8,
            control.metric, 0.85, wc, {}));
    } else {
        direct = tulpa_hmc::run_hmc_parallel_chains_cpp(
            std::vector<std::vector<double>>(n_chains, std::vector<double>(p, 0.0)),
            {}, data, n_iter, n_warmup, L, n_chains, seed, false, 8,
            control.metric, 0.85, wc, "", &layout);
    }
    bool identical = true;
    Rcpp::List draws(n_chains), steps(n_chains), energy(n_chains);
    Rcpp::CharacterVector samplers(n_chains);
    for (int c = 0; c < n_chains; ++c) {
        auto& r = results[c];
        Rcpp::NumericMatrix m(r.n_sample, p);
        for (int i = 0; i < r.n_sample; ++i) {
            for (int j = 0; j < p; ++j) {
                m(i, j) = r.samples[static_cast<size_t>(i) * p + j];
                identical &= m(i, j) == direct[c].sample_row(i)[j];
            }
            identical &= r.n_leapfrog[i] == direct[c].n_leapfrog[i];
            identical &= r.energy[i] == direct[c].energy[i];
        }
        draws[c] = m;
        steps[c] = Rcpp::IntegerVector(r.n_leapfrog, r.n_leapfrog + r.n_sample);
        energy[c] = Rcpp::NumericVector(r.energy, r.energy + r.n_sample);
        samplers[c] = r.sampler;
        r.free_buffers();
    }
    return Rcpp::List::create(Rcpp::Named("draws") = draws,
        Rcpp::Named("identical") = identical, Rcpp::Named("n_leapfrog") = steps,
        Rcpp::Named("energy") = energy, Rcpp::Named("sampler") = samplers,
        Rcpp::Named("gradient_verified") = tulpa_hmc::verify_gradient_runtime(
            std::vector<double>(p, 0.2), data, layout, 1e-4));
}

// ============================================================================
// Gaussian response data (model-specific, opaque to tulpa engine)
// ============================================================================
struct GaussianData {
    std::vector<double> y;  // Response vector
};

// ============================================================================
// Gaussian log-likelihood (templated for autodiff compatibility)
// ============================================================================
template<typename T>
T gaussian_likelihood(
    int i,
    const T* eta,
    const T& logit_zi,
    const T& logit_oi,
    const std::vector<T>& params,
    const ModelData& data,
    const ParamLayout& layout,
    const void* model_data
) {
    const auto* gd = static_cast<const GaussianData*>(model_data);
    T residual = T(gd->y[i]) - eta[0];
    T log_sigma = params[layout.extra_offset];  // log(residual SD)
    // Normal log-density up to constants, in log_sigma:
    //   log p = -log_sigma - 0.5 * r^2 * exp(-2 * log_sigma)
    T neg_log_sigma = T(0.0) - log_sigma;
    T half_resid_sq = T(0.5) * residual * residual;
    T neg_two_ls = neg_log_sigma + neg_log_sigma;
    // `exp` resolves through ADL: std::exp for T = double, tulpa::arena::exp
    // for arena::Var.
    using std::exp;
    T precision = exp(neg_two_ls);
    return neg_log_sigma - half_resid_sq * precision;
}

// ============================================================================
// Shared Gaussian fixture: build the (GaussianData, LikelihoodSpec, ModelData,
// ParamLayout) bundle from y/X. The caller owns gd/spec/data/layout as locals
// so the pointers ModelData holds into gd/spec stay valid. Single source of
// truth for the single- and multi-chain generic entry points.
// ============================================================================
static void build_gaussian_model(
    const Rcpp::NumericVector& y_r,
    const Rcpp::NumericMatrix& X_r,
    double sigma_beta,
    GaussianData& gd,
    tulpa::LikelihoodSpec& spec,
    ModelData& data,
    ParamLayout& layout
) {
    const int N = y_r.size();
    const int p = X_r.ncol();

    gd.y.assign(y_r.begin(), y_r.end());

    spec.name = "gaussian";
    spec.n_processes = 1;
    spec.ll_double = gaussian_likelihood<double>;
    spec.ll_arena = gaussian_likelihood<tulpa::arena::Var>;
    spec.n_extra_params = 1;  // log_sigma

    data.N = N;
    data.n_processes = 1;
    data.sigma_beta = sigma_beta;

    tulpa::ProcessData proc;
    proc.p = p;
    proc.X_flat.resize(N * p);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < p; j++) {
            proc.X_flat[i * p + j] = X_r(i, j);
        }
    }
    data.processes.push_back(proc);
    data.model_response_data = &gd;
    data.likelihood_spec = &spec;
    data.sharing.init(1);

    data.zi_type = tulpa::ZIType::NONE;
    data.p_zi = 0;
    data.p_oi = 0;
    data.zi_prior_sd = 1.0;
    data.oi_prior_sd = 1.0;

    layout = tulpa_hmc::compute_param_layout(data);
}

// Column names for the Gaussian fixture: beta[1..p], log_sigma.
static Rcpp::CharacterVector gaussian_col_names(int p) {
    Rcpp::CharacterVector cn(p + 1);
    for (int j = 0; j < p; j++) {
        cn[j] = "beta[" + std::to_string(j + 1) + "]";
    }
    cn[p] = "log_sigma";
    return cn;
}

// ============================================================================
// Generic NUTS sampler for multi-process models
// Uses tulpa's full NUTS backend with dual averaging and mass matrix adaptation.
// Model packages provide a LikelihoodFn<double> via LikelihoodSpec.
// ============================================================================


// [[Rcpp::export]]
Rcpp::List cpp_tulpa_fit_generic(
    Rcpp::NumericVector y_r,           // Response vector
    Rcpp::NumericMatrix X_r,           // Design matrix (single process for now)
    Rcpp::Nullable<Rcpp::NumericVector> sigma_beta = R_NilValue,
    int n_iter = 2000,
    int n_warmup = 1000,
    int max_treedepth = 10,
    double adapt_delta = 0.8,
    int seed = 42,
    bool verbose = true,
    Rcpp::Nullable<Rcpp::NumericVector> init = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> inv_metric_init = R_NilValue,
    double epsilon_init = 0.0
) {
    const double sigma_beta_value =
        tulpa::read_sigma_beta(sigma_beta);
    const int p = X_r.ncol();

    GaussianData gd;
    tulpa::LikelihoodSpec spec;
    ModelData data;
    ParamLayout layout;
    build_gaussian_model(y_r, X_r, sigma_beta_value, gd, spec, data, layout);
    int n_params = layout.total_params;

    // Initial values — default origin, or caller-supplied for chain resume.
    std::vector<double> init_vec(n_params, 0.0);
    if (init.isNotNull()) {
        Rcpp::NumericVector iv(init);
        if ((int)iv.size() != n_params) {
            Rcpp::stop("init length %d != n_params %d", (int)iv.size(), n_params);
        }
        init_vec.assign(iv.begin(), iv.end());
    }

    // Optional warm-start inverse-mass diagonal (e.g. from a previous fit's
    // inv_metric output). Empty -> structural warm-start.
    const auto metrics = tulpa::read_inv_metric_init(inv_metric_init, 1, n_params);
    const std::vector<double> inv_metric_vec = metrics.empty() ? std::vector<double>() : metrics[0];

    // Run full NUTS (L=0 means NUTS)
    tulpa_hmc::HMCResultCpp result = tulpa_hmc::run_hmc_chain_cpp(
        init_vec, data, layout,
        n_iter, n_warmup,
        0,            // L=0 means NUTS
        1,            // chain_id
        static_cast<unsigned int>(seed),
        verbose,
        max_treedepth,
        tulpa::MassMatrixType::DIAG,
        adapt_delta,
        nullptr,      // walnuts off
        inv_metric_vec, epsilon_init
    );

    // Convert to R matrix
    int n_sample = result.n_sample;
    Rcpp::NumericMatrix draws(n_sample, n_params);
    for (int s = 0; s < n_sample; s++) {
        const double* row = result.sample_row(s);
        for (int j = 0; j < n_params; j++) {
            draws(s, j) = row[j];
        }
    }

    // Name columns
    Rcpp::CharacterVector col_names = gaussian_col_names(p);
    Rcpp::colnames(draws) = col_names;

    // Posterior means
    Rcpp::NumericVector means(n_params, 0.0);
    for (int s = 0; s < n_sample; s++) {
        for (int j = 0; j < n_params; j++) {
            means[j] += draws(s, j) / n_sample;
        }
    }
    means.names() = col_names;

    return Rcpp::List::create(
        Rcpp::Named("draws") = draws,
        Rcpp::Named("means") = means,
        Rcpp::Named("n_samples") = n_sample,
        Rcpp::Named("n_params") = n_params,
        Rcpp::Named("log_prob") = Rcpp::wrap(result.log_prob),
        Rcpp::Named("accept_prob") = Rcpp::wrap(result.accept_prob),
        Rcpp::Named("divergent") = Rcpp::wrap(result.divergent),
        Rcpp::Named("treedepth") = Rcpp::wrap(result.treedepth),
        Rcpp::Named("energy") = Rcpp::wrap(result.energy),
        // Post-warmup saturation count. Written and checkpointed but read by
        // nothing until gcol33/tulpa#703.
        Rcpp::Named("n_max_treedepth") = result.n_max_treedepth,
        Rcpp::Named("sampler") = result.sampler.empty() ? "nuts" : result.sampler,
        Rcpp::Named("epsilon") = result.epsilon,
        // Warm-start / resume outputs
        Rcpp::Named("inv_metric") = Rcpp::wrap(result.inv_metric_diag),
        Rcpp::Named("final_position") = Rcpp::wrap(result.final_position)
    );
}

// ============================================================================
// Multi-chain generic NUTS.
// Runs n_chains chains via tulpa's OpenMP across-chain core in one call and
// returns draws stacked chain-major with a chain_id vector — the layout
// tulpa::mcmc_diagnostics consumes — plus per-chain epsilon, adapted
// inverse-mass diagonal, and final position (for resume).
//
// `init` / `inv_metric_init`, when supplied, are [n_chains x n_params]
// matrices (row c = chain c). `init` NULL -> origin for every chain;
// `inv_metric_init` NULL -> structural default for every chain. Passing the
// previous fit's `final_position`, `inv_metric` and `epsilon` as
// `init`, `inv_metric_init` and `epsilon_init` with n_warmup = 0 continues
// the chains.
// ============================================================================
// [[Rcpp::export]]
Rcpp::List cpp_tulpa_fit_generic_chains(
    Rcpp::NumericVector y_r,
    Rcpp::NumericMatrix X_r,
    int n_chains = 4,
    Rcpp::Nullable<Rcpp::NumericVector> sigma_beta = R_NilValue,
    int n_iter = 2000,
    int n_warmup = 1000,
    int max_treedepth = 10,
    double adapt_delta = 0.8,
    int seed = 42,
    bool verbose = false,
    Rcpp::Nullable<Rcpp::NumericMatrix> init = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericMatrix> inv_metric_init = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> epsilon_init = R_NilValue,
    std::string checkpoint_path = ""
) {
    const double sigma_beta_value =
        tulpa::read_sigma_beta(sigma_beta);
    if (n_chains < 1) Rcpp::stop("n_chains must be >= 1");
    const int p = X_r.ncol();

    GaussianData gd;
    tulpa::LikelihoodSpec spec;
    ModelData data;
    ParamLayout layout;
    build_gaussian_model(y_r, X_r, sigma_beta_value, gd, spec, data, layout);
    const int n_params = layout.total_params;

    // Per-chain initial position (rows of `init`, else origin).
    std::vector<std::vector<double>> q_init_per_chain(
        n_chains, std::vector<double>(n_params, 0.0));
    if (init.isNotNull()) {
        Rcpp::NumericMatrix im(init);
        if (im.nrow() != n_chains || im.ncol() != n_params) {
            Rcpp::stop("init must be [n_chains x n_params] = [%d x %d]",
                       n_chains, n_params);
        }
        for (int c = 0; c < n_chains; c++) {
            for (int j = 0; j < n_params; j++) q_init_per_chain[c][j] = im(c, j);
        }
    }

    // Per-chain inverse-mass diagonal (rows of `inv_metric_init`, else default).
    const auto inv_metric_per_chain = tulpa::read_inv_metric_init(inv_metric_init, n_chains, n_params);

    std::vector<tulpa_hmc::HMCResultCpp> chains = tulpa_hmc::run_hmc_parallel_chains_cpp(
        q_init_per_chain, inv_metric_per_chain, data,
        n_iter, n_warmup,
        0,                // L=0 means NUTS
        n_chains, static_cast<unsigned int>(seed), verbose,
        max_treedepth, tulpa::MassMatrixType::DIAG, adapt_delta,
        nullptr,          // walnuts off
        checkpoint_path, nullptr, tulpa::read_epsilon_init(epsilon_init, n_chains)
    );

    tulpa_hmc::StackedChains st = tulpa_hmc::stack_hmc_chains(chains, n_chains, n_params);

    Rcpp::CharacterVector col_names = gaussian_col_names(p);
    Rcpp::colnames(st.draws) = col_names;
    Rcpp::colnames(st.inv_metric) = col_names;
    Rcpp::colnames(st.final_position) = col_names;

    return Rcpp::List::create(
        Rcpp::Named("draws") = st.draws,
        Rcpp::Named("chain_id") = st.chain_id,
        Rcpp::Named("n_chains") = n_chains,
        Rcpp::Named("n_samples") = st.n_sample_per_chain,
        Rcpp::Named("n_params") = n_params,
        Rcpp::Named("log_prob") = st.log_prob,
        Rcpp::Named("accept_prob") = st.accept_prob,
        Rcpp::Named("divergent") = st.divergent,
        Rcpp::Named("treedepth") = st.treedepth,
        Rcpp::Named("energy") = st.energy,
        Rcpp::Named("sampler") = chains[0].sampler.empty() ? "nuts" : chains[0].sampler,
        Rcpp::Named("epsilon") = st.epsilon,
        // Per-chain warm-start / resume outputs
        Rcpp::Named("inv_metric") = st.inv_metric,
        Rcpp::Named("final_position") = st.final_position
    );
}

// ============================================================================
// C-ABI round-trip verifier for the resume outputs.
//
// The Rcpp-wrapper paths (cpp_tulpa_fit_generic / *_chains) read inv_metric
// and final_position directly off HMCResultCpp. Downstream packages instead
// reach the engine through the registered C ABI: tulpa::get_nuts_fn()
// -> R_GetCCallable("tulpa", "tulpa_run_nuts_generic") -> NUTSResult.
// fill_nuts_result_from_cpp() copies the resume fields into the C-ABI struct;
// this helper exercises that exact path so a regression to that copy (or to
// the ABI version / registration) fails a unit test rather than only showing
// up in a downstream consumer.
//
// Returns the two C-ABI resume fields (`inv_metric_out`, `final_position`)
// and the matrix of draws, so R-level tests can assert: (a) the C ABI
// populates the resume fields with the same values the Rcpp wrapper exposes,
// and (b) a continued chain warm-started from them samples the same posterior.
// ============================================================================
// [[Rcpp::export]]
Rcpp::List cpp_test_c_abi_resume_roundtrip(
    Rcpp::NumericVector y_r,
    Rcpp::NumericMatrix X_r,
    int n_iter = 600,
    int n_warmup = 300,
    int max_treedepth = 7,
    double adapt_delta = 0.85,
    int seed = 9,
    Rcpp::Nullable<Rcpp::NumericVector> init = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> inv_metric_init = R_NilValue,
    double epsilon_init = 0.0
) {
    GaussianData gd;
    tulpa::LikelihoodSpec spec;
    ModelData data;
    ParamLayout layout;
    build_gaussian_model(y_r, X_r, tulpa::DEFAULT_SIGMA_BETA, gd, spec, data, layout);
    const int n_params = layout.total_params;

    std::vector<double> init_vec(n_params, 0.0);
    if (init.isNotNull()) {
        Rcpp::NumericVector iv(init);
        if ((int)iv.size() != n_params) {
            Rcpp::stop("init length %d != n_params %d", (int)iv.size(), n_params);
        }
        init_vec.assign(iv.begin(), iv.end());
    }

    const auto metrics = tulpa::read_inv_metric_init(inv_metric_init, 1, n_params);
    const double* inv_metric_ptr = metrics.empty() ? nullptr : metrics[0].data();

    // Reach the engine the way downstream packages do: through R_GetCCallable.
    // get_nuts_fn() caches the lookup and runs the ABI-version check.
    tulpa::NUTSFn run_nuts = tulpa::get_nuts_fn();
    if (run_nuts == nullptr) {
        Rcpp::stop("tulpa_run_nuts_generic is not registered");
    }

    tulpa::NUTSResult result = {};
    run_nuts(
        &data, &layout, init_vec.data(), n_params,
        n_iter, n_warmup, max_treedepth, adapt_delta,
        static_cast<unsigned int>(seed),
        /*verbose=*/0,
        nullptr,
        inv_metric_ptr,
        epsilon_init,
        &result
    );

    const int ns = result.n_sample;
    Rcpp::NumericMatrix draws(ns, n_params);
    for (int s = 0; s < ns; s++) {
        for (int j = 0; j < n_params; j++) {
            draws(s, j) = result.samples[s * n_params + j];
        }
    }
    Rcpp::CharacterVector col_names = gaussian_col_names(X_r.ncol());
    Rcpp::colnames(draws) = col_names;

    // The C-ABI resume fields — the thing this helper exists to verify.
    Rcpp::NumericVector inv_metric_out(n_params), final_position(n_params);
    for (int j = 0; j < n_params; j++) {
        inv_metric_out[j] = result.inv_metric_out[j];
        final_position[j] = result.final_position[j];
    }
    inv_metric_out.names() = col_names;
    final_position.names() = col_names;

    double epsilon = result.epsilon;
    result.free_buffers();

    return Rcpp::List::create(
        Rcpp::Named("draws") = draws,
        Rcpp::Named("n_samples") = ns,
        Rcpp::Named("n_params") = n_params,
        Rcpp::Named("epsilon") = epsilon,
        Rcpp::Named("inv_metric_out") = inv_metric_out,
        Rcpp::Named("final_position") = final_position
    );
}

// ============================================================================
// C-ABI round-trip verifier for the multi-chain runner.
//
// Mirrors cpp_test_c_abi_resume_roundtrip but for tulpa_run_nuts_chains:
// reaches the engine through tulpa::get_nuts_chains_fn() (R_GetCCallable),
// packs the chain-major [n_chains * n_params] init / inv_metric_diag buffers,
// allocates a NUTSResult[n_chains] for the output, and converts the per-chain
// results into R-side matrices.
//
// Returns draws stacked chain-major + a chain_id vector + n_chains — the
// exact (draws, chain_id, n_chains) layout tulpa::mcmc_diagnostics() reads
// — plus per-chain `inv_metric_out` / `final_position` for the resume
// contract verification.
// ============================================================================
// [[Rcpp::export]]
Rcpp::List cpp_test_c_abi_chains_roundtrip(
    Rcpp::NumericVector y_r,
    Rcpp::NumericMatrix X_r,
    int n_chains = 3,
    int n_iter = 600,
    int n_warmup = 300,
    int max_treedepth = 7,
    double adapt_delta = 0.85,
    int seed = 30,
    Rcpp::Nullable<Rcpp::NumericMatrix> init = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericMatrix> inv_metric_init = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> epsilon_init = R_NilValue
) {
    if (n_chains < 1) Rcpp::stop("n_chains must be >= 1");

    GaussianData gd;
    tulpa::LikelihoodSpec spec;
    ModelData data;
    ParamLayout layout;
    build_gaussian_model(y_r, X_r, tulpa::DEFAULT_SIGMA_BETA, gd, spec, data, layout);
    const int n_params = layout.total_params;
    const int p = X_r.ncol();

    // Per-chain init (chain-major rows), default origin.
    std::vector<double> init_flat((std::size_t)n_chains * n_params, 0.0);
    if (init.isNotNull()) {
        Rcpp::NumericMatrix im(init);
        if (im.nrow() != n_chains || im.ncol() != n_params) {
            Rcpp::stop("init must be [n_chains x n_params] = [%d x %d]",
                       n_chains, n_params);
        }
        for (int c = 0; c < n_chains; c++) {
            for (int j = 0; j < n_params; j++) {
                init_flat[(std::size_t)c * n_params + j] = im(c, j);
            }
        }
    }

    // Per-chain inverse-mass diagonal (chain-major rows), nullptr -> default.
    std::vector<double> inv_metric_flat;
    const auto metrics = tulpa::read_inv_metric_init(inv_metric_init, n_chains, n_params);
    if (!metrics.empty()) {
        inv_metric_flat.resize(static_cast<std::size_t>(n_chains) * n_params);
        for (int c = 0; c < n_chains; c++) {
            for (int j = 0; j < n_params; j++) {
                inv_metric_flat[(std::size_t)c * n_params + j] = metrics[c][j];
            }
        }
    }
    const double* inv_metric_ptr = inv_metric_flat.empty() ? nullptr : inv_metric_flat.data();

    // Reach the engine through R_GetCCallable. get_nuts_chains_fn() caches
    // the lookup and runs the ABI-version check on first use.
    tulpa::NUTSChainsFn run_chains = tulpa::get_nuts_chains_fn();
    if (run_chains == nullptr) {
        Rcpp::stop("tulpa_run_nuts_chains is not registered");
    }

    std::vector<tulpa::NUTSResult> results(n_chains, tulpa::NUTSResult{});
    const auto steps = tulpa::read_epsilon_init(epsilon_init, n_chains);
    run_chains(
        &data, &layout, init_flat.data(), n_params, n_chains,
        n_iter, n_warmup, max_treedepth, adapt_delta,
        static_cast<unsigned int>(seed),
        /*verbose=*/0,
        nullptr,
        inv_metric_ptr,
        steps.empty() ? nullptr : steps.data(),
        results.data()
    );

    // Row count is the sum of the per-chain counts: a resumed chain can carry a
    // different one, and n_chains * results[0].n_sample would then overrun.
    int n_total = 0;
    bool equal_counts = true;
    for (int c = 0; c < n_chains; c++) {
        n_total += results[c].n_sample;
        if (results[c].n_sample != results[0].n_sample) equal_counts = false;
    }
    const int n_sample = equal_counts ? results[0].n_sample : -1;

    Rcpp::NumericMatrix draws(n_total, n_params);
    Rcpp::IntegerVector chain_id(n_total);
    Rcpp::NumericVector epsilon(n_chains);
    Rcpp::NumericMatrix inv_metric_out(n_chains, n_params);
    Rcpp::NumericMatrix final_position(n_chains, n_params);

    int r = 0;
    for (int c = 0; c < n_chains; c++) {
        const tulpa::NUTSResult& res = results[c];
        for (int s = 0; s < res.n_sample; s++) {
            for (int j = 0; j < n_params; j++) {
                draws(r, j) = res.samples[s * n_params + j];
            }
            chain_id[r] = c + 1;
            r++;
        }
        epsilon[c] = res.epsilon;
        for (int j = 0; j < n_params; j++) {
            inv_metric_out(c, j) = res.inv_metric_out[j];
            final_position(c, j) = res.final_position[j];
        }
    }

    // Free the C-ABI-owned per-chain buffers. Mirrors what a downstream
    // consumer would do after copying results into its own R objects.
    for (int c = 0; c < n_chains; c++) results[c].free_buffers();

    Rcpp::CharacterVector col_names = gaussian_col_names(p);
    Rcpp::colnames(draws) = col_names;
    Rcpp::colnames(inv_metric_out) = col_names;
    Rcpp::colnames(final_position) = col_names;

    return Rcpp::List::create(
        Rcpp::Named("draws") = draws,
        Rcpp::Named("chain_id") = chain_id,
        Rcpp::Named("n_chains") = n_chains,
        Rcpp::Named("n_samples") = n_sample,
        Rcpp::Named("n_params") = n_params,
        Rcpp::Named("epsilon") = epsilon,
        Rcpp::Named("inv_metric_out") = inv_metric_out,
        Rcpp::Named("final_position") = final_position
    );
}
