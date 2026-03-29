// vi_sampler.cpp
// Variational Inference backend for numdenom
// Main entry point with variant selection and Rcpp exports

#include "hmc_sampler.h"
#include "vi_types.h"
#include "vi_optimizer.h"
#include "vi_meanfield.h"
#include "vi_fullrank.h"
#include "vi_lowrank.h"
#include "autodiff.h"
#include "log_post_impl.h"
#include <Rcpp.h>

using namespace Rcpp;
using namespace tulpa_hmc;

namespace tulpa {
namespace vi {

// =====================================================================
// Bridge function: compute log joint and gradient
// =====================================================================

// This function bridges VI with the existing HMC gradient infrastructure
double compute_log_joint_grad(
    const Eigen::VectorXd& params_eigen,
    const ModelData& data,
    const ParamLayout& layout,
    Eigen::VectorXd& grad_eigen
) {
  int D = params_eigen.size();

  // Map Eigen to std::vector (zero-copy where possible)
  std::vector<double> params(params_eigen.data(), params_eigen.data() + D);

  // Compute gradient (H-mode returns log_post as byproduct via pointer)
  std::vector<double> grad(D);
  double log_post = 0.0;
  compute_gradient(params, data, layout, grad, &log_post);

  // Map back to Eigen
  grad_eigen = Eigen::Map<Eigen::VectorXd>(grad.data(), D);

  return log_post;
}

// =====================================================================
// VI Dispatcher
// =====================================================================

VIResult fit_vi(
    const ModelData& data,
    const ParamLayout& layout,
    int D,
    const VIConfig& config,
    const Eigen::VectorXd* init_mu = nullptr
) {
  // Select variant
  VIVariant variant = select_variant(D, config);

  if (config.verbose) {
    Rcpp::Rcout << "VI variant: " << variant_to_string(variant)
                << " (D=" << D << ")\n";
  }

  // Dispatch to appropriate implementation
  switch (variant) {
    case VIVariant::MEANFIELD:
      return fit_meanfield(data, layout, D, config, init_mu);

    case VIVariant::LOWRANK:
      return fit_lowrank(data, layout, D, config, init_mu);

    case VIVariant::FULLRANK:
      return fit_fullrank(data, layout, D, config, init_mu, nullptr);

    default:
      Rcpp::stop("Unknown VI variant");
  }

  return VIResult();  // Never reached
}

} // namespace vi
} // namespace tulpa

// =====================================================================
// Rcpp Exports
// =====================================================================

// [[Rcpp::export]]
Rcpp::List cpp_vi_fit(
    Rcpp::NumericVector q_init,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    std::string model_type_str,
    Rcpp::List re_params,
    Rcpp::List spatial_params,
    Rcpp::List temporal_params,
    Rcpp::List prior_params,
    Rcpp::List zi_params,
    Rcpp::List latent_params,
    Rcpp::List st_params,
    Rcpp::List vi_options,
    bool verbose = true
) {
  using namespace tulpa_hmc;
  using namespace tulpa::vi;

  // Parse VI config
  VIConfig config = parse_vi_config(vi_options);
  config.verbose = verbose;

  // Build ModelData (same as HMC)
  ModelData data;

  // Response data
  data.N = y_num.size();
  data.legacy.y_num.resize(data.N);
  data.legacy.y_denom.resize(data.N);
  data.legacy.y_denom_cont.resize(data.N);
  for (int i = 0; i < data.N; i++) {
    data.legacy.y_num[i] = y_num[i];
    data.legacy.y_denom[i] = y_denom[i];
    data.legacy.y_denom_cont[i] = y_denom_cont[i];
  }

  // Design matrices
  data.legacy.p_num = X_num.ncol();
  data.legacy.p_denom = X_denom.ncol();
  data.legacy.X_num_flat.resize(data.N * data.legacy.p_num);
  data.legacy.X_denom_flat.resize(data.N * data.legacy.p_denom);

  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.legacy.p_num; j++) {
      data.legacy.X_num_flat[i * data.legacy.p_num + j] = X_num(i, j);
    }
    for (int j = 0; j < data.legacy.p_denom; j++) {
      data.legacy.X_denom_flat[i * data.legacy.p_denom + j] = X_denom(i, j);
    }
  }

  // Model type
  if (model_type_str == "binomial") {
    data.legacy.model_type = ModelType::BINOMIAL;
  } else if (model_type_str == "negbin_negbin") {
    data.legacy.model_type = ModelType::NEGBIN_NEGBIN;
  } else if (model_type_str == "poisson_gamma") {
    data.legacy.model_type = ModelType::POISSON_GAMMA;
  } else {
    Rcpp::stop("Unknown model type: " + model_type_str);
  }

  // Random effects
  Rcpp::IntegerVector re_group = re_params["group"];
  data.n_re_groups = Rcpp::as<int>(re_params["n_groups"]);
  data.re_group.resize(data.N);
  for (int i = 0; i < data.N; i++) {
    data.re_group[i] = re_group[i];
  }

  // Multi-term RE support
  data.n_re_terms = Rcpp::as<int>(re_params["n_terms"]);
  data.total_re_groups = data.n_re_groups;
  data.has_re_slopes = false;
  data.has_re_correlated_slopes = false;
  data.total_re_params = data.n_re_groups;
  data.total_sigma_params = (data.n_re_groups > 0) ? 1 : 0;
  data.total_chol_params = 0;

  // Spatial structure
  std::string spatial_type = Rcpp::as<std::string>(spatial_params["type"]);
  if (spatial_type == "none") {
    data.spatial_type = SpatialType::NONE;
  } else if (spatial_type == "icar") {
    data.spatial_type = SpatialType::ICAR;
  } else if (spatial_type == "bym2") {
    data.spatial_type = SpatialType::BYM2;
  } else {
    data.spatial_type = SpatialType::NONE;
  }

  Rcpp::IntegerVector spatial_group = spatial_params["group"];
  data.n_spatial_units = Rcpp::as<int>(spatial_params["n_units"]);
  data.spatial_group.resize(data.N);
  for (int i = 0; i < data.N; i++) {
    data.spatial_group[i] = spatial_group[i];
  }

  if (data.spatial_type == SpatialType::ICAR || data.spatial_type == SpatialType::BYM2) {
    Rcpp::IntegerVector adj_row_ptr = spatial_params["adj_row_ptr"];
    Rcpp::IntegerVector adj_col_idx = spatial_params["adj_col_idx"];
    Rcpp::IntegerVector n_neighbors = spatial_params["n_neighbors"];

    data.adj_row_ptr.resize(adj_row_ptr.size());
    data.adj_col_idx.resize(adj_col_idx.size());
    data.n_neighbors.resize(n_neighbors.size());

    for (int i = 0; i < adj_row_ptr.size(); i++) data.adj_row_ptr[i] = adj_row_ptr[i];
    for (int i = 0; i < adj_col_idx.size(); i++) data.adj_col_idx[i] = adj_col_idx[i];
    for (int i = 0; i < n_neighbors.size(); i++) data.n_neighbors[i] = n_neighbors[i];

    data.bym2_scale_factor = Rcpp::as<double>(spatial_params["bym2_scale"]);
  }

  // Temporal structure
  std::string temporal_type_str = Rcpp::as<std::string>(temporal_params["type"]);
  if (temporal_type_str == "none") {
    data.temporal_type = TemporalType::NONE;
  } else if (temporal_type_str == "rw1") {
    data.temporal_type = TemporalType::RW1;
  } else if (temporal_type_str == "rw2") {
    data.temporal_type = TemporalType::RW2;
  } else if (temporal_type_str == "ar1") {
    data.temporal_type = TemporalType::AR1;
  } else {
    data.temporal_type = TemporalType::NONE;
  }

  Rcpp::IntegerVector temporal_time_idx = temporal_params["time_idx"];
  Rcpp::IntegerVector temporal_group_idx = temporal_params["group_idx"];
  data.n_times = Rcpp::as<int>(temporal_params["n_times"]);
  data.n_temporal_groups = Rcpp::as<int>(temporal_params["n_groups"]);
  data.n_temporal_params = Rcpp::as<int>(temporal_params["n_params"]);
  data.temporal_cyclic = Rcpp::as<bool>(temporal_params["cyclic"]);
  data.temporal_shared = Rcpp::as<bool>(temporal_params["shared"]);
  data.tau_temporal_shape = Rcpp::as<double>(temporal_params["tau_shape"]);
  data.tau_temporal_rate = Rcpp::as<double>(temporal_params["tau_rate"]);

  data.temporal_time_idx.resize(data.N);
  data.temporal_group_idx.resize(data.N);
  for (int i = 0; i < data.N; i++) {
    data.temporal_time_idx[i] = temporal_time_idx[i];
    data.temporal_group_idx[i] = temporal_group_idx[i];
  }

  // Zero-inflation
  std::string zi_type_str = Rcpp::as<std::string>(zi_params["type"]);
  if (zi_type_str == "none") {
    data.zi_type = tulpa_zi::ZIType::NONE;
  } else if (zi_type_str == "zi" || zi_type_str == "zi_poisson") {
    data.zi_type = tulpa_zi::ZIType::ZI_POISSON;
  } else if (zi_type_str == "zi_negbin") {
    data.zi_type = tulpa_zi::ZIType::ZI_NEGBIN;
  } else if (zi_type_str == "zi_binomial") {
    data.zi_type = tulpa_zi::ZIType::ZI_BINOMIAL;
  } else if (zi_type_str == "hurdle" || zi_type_str == "hurdle_poisson") {
    data.zi_type = tulpa_zi::ZIType::HURDLE_POISSON;
  } else if (zi_type_str == "hurdle_negbin") {
    data.zi_type = tulpa_zi::ZIType::HURDLE_NEGBIN;
  } else if (zi_type_str == "hurdle_binomial") {
    data.zi_type = tulpa_zi::ZIType::HURDLE_BINOMIAL;
  } else {
    data.zi_type = tulpa_zi::ZIType::NONE;
  }

  if (data.zi_type != tulpa_zi::ZIType::NONE) {
    Rcpp::NumericMatrix X_zi = zi_params["X"];
    data.p_zi = X_zi.ncol();
    data.X_zi_flat.resize(data.N * data.p_zi);
    for (int i = 0; i < data.N; i++) {
      for (int j = 0; j < data.p_zi; j++) {
        data.X_zi_flat[i * data.p_zi + j] = X_zi(i, j);
      }
    }
  } else {
    data.p_zi = 0;
  }
  data.zi_prior_sd = Rcpp::as<double>(zi_params["prior_sd"]);

  // Latent factors
  data.has_latent = Rcpp::as<bool>(latent_params["has_latent"]);
  data.latent_n_factors = Rcpp::as<int>(latent_params["n_factors"]);
  data.latent_shared = Rcpp::as<bool>(latent_params["shared"]);
  data.latent_scale = Rcpp::as<bool>(latent_params["scale"]);
  data.latent_constraint = Rcpp::as<int>(latent_params["constraint"]);
  data.latent_sigma_prior_rate = Rcpp::as<double>(latent_params["sigma_prior_rate"]);

  // Spatiotemporal
  data.has_spatiotemporal = Rcpp::as<bool>(st_params["has_spatiotemporal"]);

  // Prior parameters
  data.sigma_beta = Rcpp::as<double>(prior_params["sigma_beta"]);
  data.sigma_re_scale = Rcpp::as<double>(prior_params["sigma_re_scale"]);
  data.phi_prior_shape = Rcpp::as<double>(prior_params["phi_shape"]);
  data.phi_prior_rate = Rcpp::as<double>(prior_params["phi_rate"]);
  data.tau_spatial_shape = Rcpp::as<double>(prior_params["tau_spatial_shape"]);
  data.tau_spatial_rate = Rcpp::as<double>(prior_params["tau_spatial_rate"]);

  // Initialize flags for non-supported features
  data.has_svc = false;
  data.has_gp = false;
  data.has_multiscale_gp = false;
  data.has_hsgp = false;
  data.has_rsr = false;
  data.has_multiscale_temporal = false;
  data.n_threads = 1;

  // Initialize SVC data (empty)
  data.svc_data.n_svc = 0;
  data.svc_data.n_obs = data.N;
  data.svc_data.nn = 0;
  data.svc_sigma2_prior_scale = 1.0;
  data.svc_phi_prior_lower = 0.1;
  data.svc_phi_prior_upper = 10.0;

  // Compute parameter layout
  ParamLayout layout = compute_param_layout(data);
  int D = q_init.size();

  // Initial values
  Eigen::VectorXd init_mu(D);
  for (int i = 0; i < D; ++i) {
    init_mu(i) = q_init[i];
  }

  // Set gradient mode to forward autodiff (fastest general-purpose)
  set_gradient_mode(GradientMode::AUTO);

  // Fit VI
  VIResult result = fit_vi(data, layout, D, config, &init_mu);

  // Return results as R list
  return vi_result_to_list(result);
}

// [[Rcpp::export]]
int cpp_vi_get_n_params(
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    std::string model_type_str,
    Rcpp::List re_params,
    Rcpp::List spatial_params,
    Rcpp::List temporal_params,
    Rcpp::List zi_params,
    Rcpp::List latent_params
) {
  // This is a helper to compute parameter count without running VI
  // Mirrors the logic in compute_param_layout

  int n_params = 0;

  // Fixed effects
  n_params += X_num.ncol();
  n_params += X_denom.ncol();

  // Random effects
  int n_re_groups = Rcpp::as<int>(re_params["n_groups"]);
  if (n_re_groups > 0) {
    n_params += 1;  // log_sigma_re
    n_params += n_re_groups;  // RE values
  }

  // Overdispersion
  if (model_type_str == "negbin_negbin") {
    n_params += 2;  // log_phi_num, log_phi_denom
  } else if (model_type_str == "poisson_gamma") {
    n_params += 1;  // log_shape
  }

  // Spatial
  std::string spatial_type = Rcpp::as<std::string>(spatial_params["type"]);
  int n_spatial = Rcpp::as<int>(spatial_params["n_units"]);
  if (spatial_type == "icar") {
    n_params += 1;  // log_tau
    n_params += n_spatial;
  } else if (spatial_type == "bym2") {
    n_params += 2;  // log_sigma, logit_rho
    n_params += 2 * n_spatial;  // phi + theta
  }

  // Temporal
  std::string temporal_type = Rcpp::as<std::string>(temporal_params["type"]);
  int n_temporal = Rcpp::as<int>(temporal_params["n_params"]);
  if (temporal_type != "none" && n_temporal > 0) {
    n_params += 1;  // log_tau
    n_params += n_temporal;
    if (temporal_type == "ar1") {
      n_params += 1;  // logit_rho
    }
  }

  // Zero-inflation
  std::string zi_type = Rcpp::as<std::string>(zi_params["type"]);
  if (zi_type != "none") {
    Rcpp::NumericMatrix X_zi = zi_params["X"];
    n_params += X_zi.ncol();
  }

  // Latent factors
  bool has_latent = Rcpp::as<bool>(latent_params["has_latent"]);
  if (has_latent) {
    int n_factors = Rcpp::as<int>(latent_params["n_factors"]);
    int n_obs = Rcpp::as<int>(latent_params["n_obs"]);
    n_params += n_factors;           // log_sigma per factor
    n_params += n_obs * n_factors;   // factor scores
  }

  return n_params;
}
