// test_st_iv_fixture.cpp
// A Knorr-Held Type-IV spatiotemporal interaction model the ENGINE can fit.
//
// Nothing in tulpa sets ModelData::has_spatiotemporal: the interaction is a
// consumer-package configuration, spatiotemporal() errors at the R door, and
// the whole Type-IV sampler path is therefore unreachable from this package's
// own tests. That is what left the precision-informed mass override half-wired
// through four releases with no test able to see it.
//
// This fixture closes that: it fills a ModelData with a Type-IV interaction
// directly, exactly as a consumer package would, and drives tulpa's NUTS over
// it. Two families, chosen for what they say about the override rather than
// for coverage:
//
//   gaussian - the inner Gaussian approximation to the interaction block's
//              conditional posterior is EXACT, so diag(Q^-1) is the marginal
//              variance rather than an estimate of it. The override's best
//              case, and the arm that says whether the machinery is right.
//   poisson  - Knorr-Held's own setting (areal disease counts). The
//              approximation is a Laplace one and the working weights move
//              with the position. The arm that says whether it is useful.

#include <Rcpp.h>

#include <cmath>
#include <string>
#include <vector>

#include "hmc_mass_st_gmrf.h"
#include "hmc_sampler.h"
#include "log_post_impl.h"
#include "tulpa/autodiff_arena.h"
#include "tulpa/autodiff_fwd.h"
#include "tulpa/likelihood.h"

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

namespace {

// Response + family for the fixture. Gaussian carries one extra parameter
// (log_sigma at layout.extra_offset); poisson carries none.
struct StIvData {
  std::vector<double> y;
  bool gaussian = false;
};

template <typename T>
T st_iv_likelihood(
    int i,
    const T* eta,
    const T& /*logit_zi*/,
    const T& /*logit_oi*/,
    const std::vector<T>& params,
    const ModelData& /*data*/,
    const ParamLayout& layout,
    const void* model_data
) {
  const auto* d = static_cast<const StIvData*>(model_data);
  using std::exp;
  if (d->gaussian) {
    // log_sigma is the single extra parameter; the density is written in it so
    // the scale is sampled on an unconstrained coordinate.
    const T log_sigma = params[layout.extra_offset];
    const T resid = T(d->y[i]) - eta[0];
    const T neg_log_sigma = T(0.0) - log_sigma;
    return neg_log_sigma
         - T(0.5) * resid * resid * exp(neg_log_sigma + neg_log_sigma);
  }
  // Poisson, dropping the -log(y!) constant.
  return T(d->y[i]) * eta[0] - exp(eta[0]);
}

// Eta-space score and negative Hessian, the IRLS contract EtaWeightsFn states.
// This is the callback the mass override reads, and it is also what any
// consumer-package spec has to ship for the override to be available.
void st_iv_eta_weights(
    int i,
    const double* eta,
    double /*logit_zi*/,
    double /*logit_oi*/,
    const std::vector<double>& params,
    const ModelData& /*data*/,
    const ParamLayout& layout,
    const void* model_data,
    double* grad_eta,
    double* neg_hess_eta
) {
  const auto* d = static_cast<const StIvData*>(model_data);
  if (d->gaussian) {
    const double prec = std::exp(-2.0 * params[layout.extra_offset]);
    grad_eta[0] = (d->y[i] - eta[0]) * prec;
    neg_hess_eta[0] = prec;
    return;
  }
  const double mu = std::exp(eta[0]);
  grad_eta[0] = d->y[i] - mu;
  neg_hess_eta[0] = mu;
}

// Gaussian prior on log_sigma, so the extra parameter is proper.
double st_iv_extra_prior(
    const std::vector<double>& params,
    const ParamLayout& layout,
    const void* model_data
) {
  const auto* d = static_cast<const StIvData*>(model_data);
  if (!d->gaussian) return 0.0;
  const double ls = params[layout.extra_offset];
  return -0.5 * ls * ls;
}

tulpa::arena::Var st_iv_extra_prior_arena(
    const std::vector<tulpa::arena::Var>& params,
    const ParamLayout& layout,
    const void* model_data
) {
  const auto* d = static_cast<const StIvData*>(model_data);
  if (!d->gaussian) return tulpa::arena::Var(0.0);
  tulpa::arena::Var ls = params[layout.extra_offset];
  return tulpa::arena::Var(-0.5) * ls * ls;
}

// Fill the ModelData a consumer package would hand the engine for a Type-IV
// interaction. Caller owns `sd` / `spec` / `data` / `layout` as locals, since
// ModelData holds pointers into the first two.
void build_st_iv_model(
    const Rcpp::NumericVector& y_r,
    const Rcpp::NumericMatrix& X_r,
    const Rcpp::IntegerVector& s_idx_r,
    const Rcpp::IntegerVector& t_idx_r,
    const Rcpp::IntegerVector& adj_row_ptr_r,
    const Rcpp::IntegerVector& adj_col_idx_r,
    int S, int T,
    const std::string& family,
    const std::string& temporal,
    int st_parameterization,
    double sigma_beta,
    StIvData& sd,
    tulpa::LikelihoodSpec& spec,
    ModelData& data,
    ParamLayout& layout
) {
  const int N = y_r.size();
  const int p = X_r.ncol();

  if (X_r.nrow() != N) Rcpp::stop("nrow(X) must equal length(y)");
  if (s_idx_r.size() != N || t_idx_r.size() != N) {
    Rcpp::stop("s_idx / t_idx must have length(y) entries");
  }
  if ((int)adj_row_ptr_r.size() != S + 1) {
    Rcpp::stop("adj_row_ptr must have S + 1 = %d entries", S + 1);
  }

  sd.y.assign(y_r.begin(), y_r.end());
  sd.gaussian = (family == "gaussian");
  if (!sd.gaussian && family != "poisson") {
    Rcpp::stop("family must be \"poisson\" or \"gaussian\"; got \"%s\"",
               family.c_str());
  }

  spec.name = "st_iv_" + family;
  spec.n_processes = 1;
  spec.ll_double = st_iv_likelihood<double>;
  spec.ll_arena = st_iv_likelihood<tulpa::arena::Var>;
  spec.ll_fwd = st_iv_likelihood< ::fwd::Dual>;
  spec.eta_weights_fn = &st_iv_eta_weights;
  spec.n_extra_params = sd.gaussian ? 1 : 0;
  spec.extra_prior = sd.gaussian ? &st_iv_extra_prior : nullptr;
  spec.extra_prior_arena = sd.gaussian ? &st_iv_extra_prior_arena : nullptr;

  data.N = N;
  data.n_processes = 1;
  data.sigma_beta = sigma_beta;

  tulpa::ProcessData proc;
  proc.p = p;
  proc.X_flat.resize((std::size_t)N * p);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < p; j++) proc.X_flat[(std::size_t)i * p + j] = X_r(i, j);
  }
  data.processes.push_back(proc);
  data.model_response_data = &sd;
  data.likelihood_spec = &spec;
  data.sharing.init(1);

  data.zi_type = tulpa::ZIType::NONE;
  data.p_zi = 0;
  data.p_oi = 0;
  data.zi_prior_sd = 1.0;
  data.oi_prior_sd = 1.0;

  data.has_spatiotemporal = true;
  data.st_parameterization = st_parameterization;
  data.st_is_hsgp = false;
  data.st_sigma2_prior_U = 1.0;
  data.st_sigma2_prior_alpha = 0.01;

  auto& st = data.spatiotemporal_data;
  st.type = tulpa::STType::TYPE_IV;
  st.shared = true;
  st.n_spatial = S;
  st.n_times = T;
  st.n_params = S * T;
  st.temporal_type = (temporal == "rw2") ? tulpa::TemporalType::RW2
                                         : tulpa::TemporalType::RW1;
  st.temporal_cyclic = false;

  st.s_idx.assign(s_idx_r.begin(), s_idx_r.end());
  st.t_idx.assign(t_idx_r.begin(), t_idx_r.end());
  st.st_flat.resize(N);
  for (int i = 0; i < N; i++) {
    const int s = st.s_idx[i], t = st.t_idx[i];
    if (s < 1 || s > S || t < 1 || t > T) {
      Rcpp::stop("s_idx / t_idx entry %d is outside [1, S] x [1, T]", i + 1);
    }
    st.st_flat[i] = (s - 1) * T + t;
  }

  st.adj_row_ptr.assign(adj_row_ptr_r.begin(), adj_row_ptr_r.end());
  st.adj_col_idx.assign(adj_col_idx_r.begin(), adj_col_idx_r.end());
  st.n_neighbors.resize(S);
  for (int s = 0; s < S; s++) {
    st.n_neighbors[s] = st.adj_row_ptr[s + 1] - st.adj_row_ptr[s];
  }

  layout = tulpa_hmc::compute_param_layout(data);
}

}  // namespace

// Fit the Type-IV fixture with tulpa's NUTS under a named metric.
//
// Returns the per-iteration leapfrog counts and divergence flags alongside the
// draws, which is what a metric comparison is scored on: cost per effective
// sample is leapfrog steps, not wall clock, and wall clock on a shared machine
// is not reproducible.
// [[Rcpp::export]]
Rcpp::List cpp_test_st_iv_nuts(
    Rcpp::NumericVector y,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector s_idx,
    Rcpp::IntegerVector t_idx,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    int S,
    int T,
    std::string family = "poisson",
    std::string temporal = "rw1",
    int st_parameterization = 0,
    std::string mass_matrix = "diag",
    int n_iter = 1000,
    int n_warmup = 500,
    int max_treedepth = 10,
    double adapt_delta = 0.8,
    int seed = 1,
    double sigma_beta = 10.0,
    bool verbose = false
) {
  StIvData sd;
  tulpa::LikelihoodSpec spec;
  ModelData data;
  ParamLayout layout;
  build_st_iv_model(y, X, s_idx, t_idx, adj_row_ptr, adj_col_idx, S, T,
                    family, temporal, st_parameterization, sigma_beta,
                    sd, spec, data, layout);

  const int n_params = layout.total_params;
  const tulpa::MassMatrixType metric = tulpa_hmc::parse_metric_type(mass_matrix);

  std::vector<double> init(n_params, 0.0);
  tulpa_hmc::HMCResultCpp res = tulpa_hmc::run_hmc_chain_cpp(
      init, data, layout, n_iter, n_warmup,
      /*L=*/0, /*chain_id=*/0, (unsigned int)seed, verbose,
      max_treedepth, metric, adapt_delta, /*riemannian=*/0,
      std::vector<double>());

  const int n_sample = res.n_sample;
  Rcpp::NumericMatrix draws(n_sample, n_params);
  for (int s = 0; s < n_sample; s++) {
    const double* row = res.sample_row(s);
    for (int j = 0; j < n_params; j++) draws(s, j) = row[j];
  }

  return Rcpp::List::create(
      Rcpp::Named("draws") = draws,
      Rcpp::Named("n_params") = n_params,
      Rcpp::Named("st_delta_start") = layout.st_delta_start,
      Rcpp::Named("st_delta_end") = layout.st_delta_end,
      Rcpp::Named("log_tau_st_idx") = layout.log_tau_st_idx,
      Rcpp::Named("log_prob") = Rcpp::wrap(res.log_prob),
      Rcpp::Named("accept_prob") = Rcpp::wrap(res.accept_prob),
      Rcpp::Named("n_leapfrog") = Rcpp::wrap(res.n_leapfrog),
      Rcpp::Named("divergent") = Rcpp::wrap(res.divergent),
      Rcpp::Named("treedepth") = Rcpp::wrap(res.treedepth),
      Rcpp::Named("epsilon") = res.epsilon,
      Rcpp::Named("inv_metric") = Rcpp::wrap(res.inv_metric_diag),
      Rcpp::Named("st_gmrf_applied") = res.st_gmrf_applied,
      Rcpp::Named("st_gmrf_declined") = res.st_gmrf_declined);
}

// Where compute_param_layout puts the Type-IV block, with nothing evaluated.
// The positions the other two entry points take have to be laid out against
// this, and both of them refuse a wrong-length one, so the layout has to be
// readable without supplying a position at all.
// [[Rcpp::export]]
Rcpp::List cpp_test_st_iv_layout(
    Rcpp::NumericVector y,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector s_idx,
    Rcpp::IntegerVector t_idx,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    int S,
    int T,
    std::string family = "poisson",
    std::string temporal = "rw1",
    int st_parameterization = 0,
    double sigma_beta = 10.0
) {
  StIvData sd;
  tulpa::LikelihoodSpec spec;
  ModelData data;
  ParamLayout layout;
  build_st_iv_model(y, X, s_idx, t_idx, adj_row_ptr, adj_col_idx, S, T,
                    family, temporal, st_parameterization, sigma_beta,
                    sd, spec, data, layout);
  return Rcpp::List::create(
      Rcpp::Named("n_params") = layout.total_params,
      Rcpp::Named("st_delta_start") = layout.st_delta_start,
      Rcpp::Named("st_delta_end") = layout.st_delta_end,
      Rcpp::Named("log_tau_st_idx") = layout.log_tau_st_idx,
      Rcpp::Named("extra_offset") = layout.extra_offset);
}

// The override's own output at a caller-supplied position, with no sampler
// around it: diag(Q^-1) over the interaction block plus the status the chain
// would have recorded. This is what the equivalence tests score against a
// dense inverse, and what the quadratic-form test reads the assembly from.
// [[Rcpp::export]]
Rcpp::List cpp_test_st_iv_gmrf_mass(
    Rcpp::NumericVector y,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector s_idx,
    Rcpp::IntegerVector t_idx,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    int S,
    int T,
    Rcpp::NumericVector q,
    std::string family = "poisson",
    std::string temporal = "rw1",
    int st_parameterization = 0,
    double sigma_beta = 10.0,
    bool with_eta_weights = true
) {
  StIvData sd;
  tulpa::LikelihoodSpec spec;
  ModelData data;
  ParamLayout layout;
  build_st_iv_model(y, X, s_idx, t_idx, adj_row_ptr, adj_col_idx, S, T,
                    family, temporal, st_parameterization, sigma_beta,
                    sd, spec, data, layout);
  // A spec shipping no IRLS callback is the decline path the generic
  // interface has to handle, so the fixture can drop it on request.
  if (!with_eta_weights) spec.eta_weights_fn = nullptr;

  if ((int)q.size() != layout.total_params) {
    Rcpp::stop("q has %d entries; the layout has %d parameters",
               (int)q.size(), layout.total_params);
  }
  const std::vector<double> qv(q.begin(), q.end());

  tulpa_hmc::StGmrfMassResult r = tulpa_hmc::st_gmrf_inv_mass(qv, data, layout);

  return Rcpp::List::create(
      Rcpp::Named("ok") = r.ok,
      Rcpp::Named("reason") = std::string(r.reason),
      Rcpp::Named("inv_mass") = Rcpp::wrap(r.inv_mass),
      Rcpp::Named("n_block") = r.n_block,
      Rcpp::Named("n_curvature_clamped") = r.n_curvature_clamped,
      Rcpp::Named("ridge_applied") = r.ridge_applied,
      Rcpp::Named("log_det_Q") = r.log_det_Q,
      Rcpp::Named("st_delta_start") = layout.st_delta_start,
      Rcpp::Named("st_delta_end") = layout.st_delta_end,
      Rcpp::Named("log_tau_st_idx") = layout.log_tau_st_idx,
      Rcpp::Named("n_params") = layout.total_params);
}

// The engine's own log-posterior at q, for the Type-IV fixture. The tests
// difference it to reach the interaction block's exact Hessian, which is what
// diag(Q^-1) is scored against -- an arbiter outside the assembly rather than
// the assembly checked against itself.
// [[Rcpp::export]]
double cpp_test_st_iv_log_post(
    Rcpp::NumericVector y,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector s_idx,
    Rcpp::IntegerVector t_idx,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    int S,
    int T,
    Rcpp::NumericVector q,
    std::string family = "poisson",
    std::string temporal = "rw1",
    int st_parameterization = 0,
    double sigma_beta = 10.0
) {
  StIvData sd;
  tulpa::LikelihoodSpec spec;
  ModelData data;
  ParamLayout layout;
  build_st_iv_model(y, X, s_idx, t_idx, adj_row_ptr, adj_col_idx, S, T,
                    family, temporal, st_parameterization, sigma_beta,
                    sd, spec, data, layout);
  if ((int)q.size() != layout.total_params) {
    Rcpp::stop("q has %d entries; the layout has %d parameters",
               (int)q.size(), layout.total_params);
  }
  const std::vector<double> qv(q.begin(), q.end());
  return tulpa::compute_log_post_generic_spec_double(qv, data, layout);
}
