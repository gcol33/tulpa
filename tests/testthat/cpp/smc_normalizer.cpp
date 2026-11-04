// The normalized log prior the SMC evidence adds back: compute_log_prior plus
// compute_log_prior_normalizer, at rows of a parameter matrix, for the
// built-in-family GLM with the latent blocks the sampler front door builds. A
// test compares it with the prior density written out independently in R. A
// theta with no rows returns the layout's parameter names alone.

#include <Rcpp.h>
#include <string>
#include <vector>

#include "laplace_spec_fit.h"      // as_offset_vec
#include "sampler_model_data.h"    // build_sampler_model_inputs / sampler_param_names
#include "smc_interweave.h"

// [[Rcpp::export]]
Rcpp::List cpp_test_log_prior_normalized(
    Rcpp::NumericMatrix theta, Rcpp::NumericVector y,
    Rcpp::IntegerVector n_trials, Rcpp::NumericMatrix X, std::string family,
    double phi, double sigma_beta, double sigma_re_scale,
    Rcpp::Nullable<Rcpp::List> re_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> spatial_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> temporal_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> svc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> tvc_spec = R_NilValue) {
    tulpa::SamplerModelInputs in;
    std::vector<double> offset = tulpa::as_offset_vec(R_NilValue, y.size());
    tulpa::build_sampler_model_inputs(
        in, y, n_trials, X, family, phi, NA_REAL, sigma_beta, offset,
        sigma_re_scale, re_spec, spatial_spec, temporal_spec, svc_spec, tvc_spec);
    const int D = in.layout.total_params;
    if (theta.nrow() > 0 && theta.ncol() != D) {
        Rcpp::stop("theta has %d columns but the layout has %d parameters",
                   theta.ncol(), D);
    }
    std::vector<std::string> declined;
    const double norm =
        tulpa_hmc::compute_log_prior_normalizer(in.data, in.layout, &declined);
    Rcpp::NumericVector lp(theta.nrow());
    std::vector<double> th(D);
    for (int r = 0; r < theta.nrow(); r++) {
        for (int j = 0; j < D; j++) th[j] = theta(r, j);
        lp[r] = tulpa_hmc::compute_log_prior(th, in.data, in.layout) + norm;
    }
    return Rcpp::List::create(
        Rcpp::Named("log_prior") = lp,
        Rcpp::Named("normalizer") = norm,
        Rcpp::Named("declined") = Rcpp::wrap(declined),
        Rcpp::Named("names") = tulpa::sampler_param_names(in.data, in.layout,
                                                          R_NilValue));
}

// The position every sampler backend starts from (init_bounded_support_params
// on a zero vector), named by the layout.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_test_sampler_init(
    Rcpp::NumericVector y, Rcpp::IntegerVector n_trials, Rcpp::NumericMatrix X,
    std::string family, double phi, double sigma_beta, double sigma_re_scale,
    Rcpp::Nullable<Rcpp::List> re_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> spatial_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> temporal_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> svc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> tvc_spec = R_NilValue) {
    tulpa::SamplerModelInputs in;
    std::vector<double> offset = tulpa::as_offset_vec(R_NilValue, y.size());
    tulpa::build_sampler_model_inputs(
        in, y, n_trials, X, family, phi, NA_REAL, sigma_beta, offset,
        sigma_re_scale, re_spec, spatial_spec, temporal_spec, svc_spec, tvc_spec);
    std::vector<double> init(in.layout.total_params, 0.0);
    tulpa::init_bounded_support_params(init, in.data, in.layout);
    Rcpp::NumericVector out(init.begin(), init.end());
    out.names() = tulpa::sampler_param_names(in.data, in.layout, R_NilValue);
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_nngp_interweave(int n, double power,
    Rcpp::List spatial_spec, bool interweave_level = false) {
    const int N = Rcpp::as<Rcpp::NumericMatrix>(spatial_spec["coords"]).nrow();
    tulpa::SamplerModelInputs in;
    Rcpp::NumericMatrix X(N, 1);
    if (interweave_level) std::fill(X.begin(), X.end(), 1.0);
    tulpa::build_sampler_model_inputs(
        in, Rcpp::NumericVector(N), Rcpp::IntegerVector(),
        X, "gaussian", 1.0, NA_REAL, 1.0,
        std::vector<double>(N, 0.0), 2.5, R_NilValue, spatial_spec,
        R_NilValue, R_NilValue, R_NilValue);
    std::vector<double> theta(in.layout.total_params, 0.0);
    for (int i = 0; i < N; ++i) theta[in.layout.gp_w_start + i] = 0.5 * (i + 1) - 0.7;
    auto target = [&](const std::vector<double>& x, double s) {
        double q = 0.0;
        for (double v : x) q -= 0.5 * (v - 0.2) * (v - 0.2);
        const double diff = x[in.layout.log_sigma2_gp_idx] - x[in.layout.gp_w_start];
        q -= 0.35 * diff * diff;
        return (1.0 - s) * q + s *
            tulpa_hmc::compute_log_prior(x, in.data, in.layout);
    };
    std::mt19937 rng(108);
    Rcpp::NumericMatrix out(n, N + 3);
    for (int i = 0; i < n; ++i) {
        tulpa::smc_interweave_nngp(theta, power, rng, in.data, in.layout, target);
        out(i, 0) = theta[in.layout.log_sigma2_gp_idx];
        out(i, 1) = theta[in.layout.log_phi_gp_idx];
        for (int j = 0; j < N; ++j) out(i, j + 2) = theta[in.layout.gp_w_start + j];
        out(i, N + 2) = theta[in.layout.process_beta_start[0]];
    }
    return out;
}

// [[Rcpp::export]]
Rcpp::List cpp_test_nngp_field_block(int n, double power, Rcpp::List spatial_spec) {
    const int N = Rcpp::as<Rcpp::NumericMatrix>(spatial_spec["coords"]).nrow();
    tulpa::SamplerModelInputs in;
    Rcpp::NumericMatrix X(N, 1);
    std::fill(X.begin(), X.end(), 1.0);
    Rcpp::NumericVector y = Rcpp::NumericVector::create(1.0, -0.4, 0.7);
    if (N != y.size()) Rcpp::stop("The field-block fixture needs three locations.");
    tulpa::build_sampler_model_inputs(in, y, Rcpp::IntegerVector(), X,
        "gaussian", 0.5, NA_REAL, 1.0, std::vector<double>(N, 0.0),
        2.5, R_NilValue, spatial_spec, R_NilValue, R_NilValue, R_NilValue);
    std::vector<double> theta(in.layout.total_params, 0.0);
    theta[in.layout.log_sigma2_gp_idx] = 0.4;
    theta[in.layout.log_phi_gp_idx] = -0.6;
    auto target = [&](const std::vector<double>& x, double s,
                       std::vector<double>& grad) {
        double post = 0.0;
        tulpa_hmc::compute_gradient(x, in.data, in.layout, grad, &post);
        double q = 0.0;
        for (size_t j = 0; j < x.size(); ++j) {
            q -= 0.5 * (x[j] - 0.2) * (x[j] - 0.2);
            grad[j] = s * grad[j] - (1.0 - s) * (x[j] - 0.2);
        }
        const int a = in.layout.log_sigma2_gp_idx, z = in.layout.gp_w_start;
        const double diff = x[a] - x[z];
        q -= 0.35 * diff * diff;
        grad[a] -= (1.0 - s) * 0.7 * diff;
        grad[z] += (1.0 - s) * 0.7 * diff;
        return (1.0 - s) * q + s * post;
    };
    tulpa::SmcNngpFieldHmc block(in.data, in.layout);
    tulpa::SmcNngpFieldMap map(theta, in.data, in.layout, block.beta);
    auto physical = map.physical();
    std::vector<double> grad;
    map.value_gradient(physical, power, target, grad);
    double error = 0.0;
    for (size_t j = 0; j < physical.size(); ++j) {
        auto plus = physical, minus = physical;
        plus[j] += 1e-6; minus[j] -= 1e-6;
        std::vector<double> unused;
        const double numeric = (map.value_gradient(plus, power, target, unused) -
            map.value_gradient(minus, power, target, unused)) / 2e-6;
        error = std::max(error, std::abs(numeric - grad[j]) / (1.0 + std::abs(numeric)));
    }
    std::mt19937 rng(351);
    std::normal_distribution<double> normal(0.0, 0.2);
    std::vector<std::vector<double>> population(256);
    for (auto& particle : population) {
        auto x = physical;
        for (double& value : x) value += normal(rng);
        map.standardize(x, particle);
    }
    block.adapt(population, in.data, in.layout);
    Rcpp::NumericMatrix draws(n, physical.size());
    for (int i = 0; i < n; ++i) {
        block.step(theta, power, rng, 0, in.data, in.layout, target);
        const auto x = tulpa::SmcNngpFieldMap(theta, in.data, in.layout, block.beta).physical();
        for (size_t j = 0; j < x.size(); ++j) draws(i, j) = x[j];
    }
    return Rcpp::List::create(Rcpp::Named("draws") = draws,
        Rcpp::Named("gradient_error") = error,
        Rcpp::Named("log_variance") = theta[in.layout.log_sigma2_gp_idx],
        Rcpp::Named("log_range") = theta[in.layout.log_phi_gp_idx]);
}
