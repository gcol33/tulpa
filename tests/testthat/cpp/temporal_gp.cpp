// temporal_gp.cpp
// Drives the templated temporal-GP density at both parameterizations.
//
// The two parameterizations target the same posterior, so their densities are
// related exactly by the forward transform's log-determinant. That identity is
// what pins the conditional variance the two branches use: the non-centered
// branch reaches it through the transform's scale a_t and the centered branch
// through cond_var, and if the two floored different quantities no single
// determinant reconciles them at a configuration where the floor binds.
//
// The density is the shipped compute_temporal_prior; only the ModelData and the
// layout are built here, the way a consumer package would fill them.

#include <Rcpp.h>

#include <vector>

#include "hmc_sampler.h"
#include "tulpa_priors_temporal.h"
#include "tulpa_priors_tvc.h"

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

namespace {

// One temporal-GP block over `times`, `n_groups` independent replicates of it,
// and nothing else. Field values (or their z) occupy [0, n_temporal), the two
// hyperparameters follow.
void tgp_fixture(const std::vector<double>& times, int n_groups,
                 double phi_lower, double phi_upper, int parameterization,
                 ModelData& data, ParamLayout& layout) {
  const int T_times = static_cast<int>(times.size());

  data.n_times = T_times;
  data.n_temporal_groups = n_groups;
  data.has_temporal_gp = true;
  data.temporal_gp_phi_prior_lower = phi_lower;
  data.temporal_gp_phi_prior_upper = phi_upper;
  data.temporal_gp_parameterization = parameterization;

  auto& g = data.temporal_gp_data;
  g.n_obs = T_times;
  g.n_groups = n_groups;
  g.time_values = times;
  g.cov_type = tulpa::TemporalCovType::EXPONENTIAL;

  const int n_temporal = T_times * n_groups;
  layout.has_temporal = true;
  layout.is_temporal_gp = true;
  layout.temporal_start = 0;
  layout.temporal_end = n_temporal;
  layout.log_sigma2_temporal_gp_idx = n_temporal;
  layout.logit_phi_temporal_gp_idx = n_temporal + 1;
  layout.total_params = n_temporal + 2;
}

}  // namespace

// Evaluate the temporal-GP density at one parameterization.
//
// `field` is the sampled coordinate: z under "noncentered", the field itself
// under "centered". Returns the log density and, for the non-centered branch,
// the reconstructed field the observation loop is handed -- which is the point
// the centered branch has to be evaluated at for the two to be comparable.
//
// [[Rcpp::export]]
Rcpp::List cpp_test_temporal_gp_density(
    Rcpp::NumericVector times, int n_groups,
    Rcpp::NumericVector field, double log_sigma2, double logit_phi,
    int parameterization, double phi_lower = 0.01, double phi_upper = 10.0,
    bool tvc = false, int covariance = 0, double nu = 1.5, double period = 2.0
) {
  const std::vector<double> tv(times.begin(), times.end());
  ModelData data;
  ParamLayout layout;
  tgp_fixture(tv, n_groups, phi_lower, phi_upper, parameterization,
              data, layout);
  data.temporal_gp_data.cov_type = static_cast<tulpa::TemporalCovType>(covariance);
  data.temporal_gp_data.nu = nu;
  data.temporal_gp_data.period = period;

  const int n_temporal = static_cast<int>(times.size()) * n_groups;
  if (field.size() != n_temporal) {
    Rcpp::stop("cpp_test_temporal_gp_density: length(field) (%d) != "
               "length(times) * n_groups (%d).",
               (int) field.size(), n_temporal);
  }

  std::vector<double> params(layout.total_params, 0.0);
  for (int k = 0; k < n_temporal; ++k) params[k] = field[k];
  params[layout.log_sigma2_temporal_gp_idx] = log_sigma2;
  params[layout.logit_phi_temporal_gp_idx] = logit_phi;

  std::vector<double> phi_temporal;
  double sigma2_out = 0.0, phi_out = 0.0;
  if (tvc) {
    auto& td = data.tvc_data;
    td.n_obs = td.n_times = static_cast<int>(times.size());
    td.n_groups = n_groups; td.n_tvc = 1;
    td.structure = tulpa::TemporalType::GP;
    td.time_values = tv;
    td.cov_type = data.temporal_gp_data.cov_type;
    td.nu = nu; td.period = period;
    td.X_tvc.assign(td.n_obs, 1.0);
    td.group_index.assign(td.n_obs, 1);
    for (int t = 0; t < td.n_times; ++t) td.time_index.push_back(t + 1);
    layout.has_temporal = layout.is_temporal_gp = false;
    layout.has_tvc = data.has_tvc = true;
    layout.tvc_w_start = 0; layout.tvc_w_end = n_temporal;
    layout.log_sigma2_tvc_gp_start = n_temporal;
    layout.logit_phi_tvc_gp_start = n_temporal + 1;
    data.tvc_gp_parameterization = parameterization;
    data.tvc_gp_phi_prior_lower = phi_lower;
    data.tvc_gp_phi_prior_upper = phi_upper;
  }
  auto evaluate = [&](const auto& q, auto& field) {
    using T = typename std::decay_t<decltype(q)>::value_type;
    std::vector<T> eta;
    if (tvc) return tulpa::priors::compute_tvc_prior(q, data, layout, eta, field);
    T tau = T(0), rho = T(0), sigma2 = T(0), phi = T(0);
    return tulpa::priors::compute_temporal_prior(q, data, layout, field, tau, rho, sigma2, phi);
  };
  const double lp = evaluate(params, phi_temporal);
  sigma2_out = std::exp(log_sigma2);
  phi_out = phi_lower + (phi_upper - phi_lower) / (1 + std::exp(-logit_phi));
  Rcpp::NumericVector gradient(params.size());
  tulpa::arena::ArenaScope scope;
  std::vector<tulpa::arena::Var> q, field_ad;
  for (double x : params) q.emplace_back(x);
  auto result = evaluate(q, field_ad);
  result.backward();
  for (size_t k = 0; k < params.size(); ++k) gradient[k] = q[k].adj();

  return Rcpp::List::create(
      Rcpp::_["log_post"] = lp,
      Rcpp::_["gradient"] = gradient,
      Rcpp::_["field"] = Rcpp::NumericVector(phi_temporal.begin(),
                                             phi_temporal.end()),
      Rcpp::_["sigma2"] = sigma2_out,
      Rcpp::_["phi"] = phi_out);
}
