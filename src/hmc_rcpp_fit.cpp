// hmc_rcpp_fit.cpp
// Rcpp-facing HMC fit entry points and ModelData construction.

#include "hmc_sampler.h"
#include <Rcpp.h>
#include <algorithm>
#include <atomic>

// =====================================================================
// R EXPORTS
// =====================================================================

// HMC sampler with bundled list arguments to avoid R's 65-arg limit for .Call
// Parameters are bundled into logical groups:
//   re_params: random effects (group, n_groups, n_terms, group_matrix, slopes, etc.)
//   spatial_params: spatial structure (type, group, adjacency, etc.)
//   temporal_params: temporal structure (type, time_idx, group_idx, etc.)
//   prior_params: prior hyperparameters
//   zi_params: zero-inflation (type, X_zi, prior_sd)
//   latent_params: latent factors
//   st_params: spatiotemporal interaction
// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit(
    Rcpp::NumericVector q_init,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_num_cont,
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
    Rcpp::List tvc_params,  // Time-varying coefficients
    Rcpp::List svc_params,  // Spatially-varying coefficients
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    int n_threads,
    bool verbose,
    std::string gradient_mode_str = "auto",
    int max_treedepth = 10,
    std::string metric_str = "auto",
    double adapt_delta = -1.0,
    int riemannian = -1,
    bool gradient_check_only = false
) {
  using namespace tulpa_hmc;

  // Set global gradient mode from R parameter
  GradientMode grad_mode = parse_gradient_mode(gradient_mode_str);
  set_gradient_mode(grad_mode);

  // Parse metric type
  MassMatrixType metric_type = parse_metric_type(metric_str);

  // =========================================================================
  // Extract bundled parameters from lists with defensive checks
  // =========================================================================

  // Random effects parameters
  // Use eager deep copies to prevent R GC from invalidating memory during HMC
  std::vector<int> re_group = Rcpp::as<std::vector<int>>(re_params["group"]);
  int n_re_groups = Rcpp::as<int>(re_params["n_groups"]);
  int n_re_terms = Rcpp::as<int>(re_params["n_terms"]);

  // Handle group_matrix which may be numeric or integer
  Rcpp::IntegerMatrix re_group_matrix;
  SEXP group_mat_sexp = re_params["group_matrix"];
  if (Rf_isMatrix(group_mat_sexp)) {
    if (TYPEOF(group_mat_sexp) == INTSXP) {
      re_group_matrix = Rcpp::as<Rcpp::IntegerMatrix>(group_mat_sexp);
    } else {
      // Convert numeric to integer
      Rcpp::NumericMatrix nm(group_mat_sexp);
      re_group_matrix = Rcpp::IntegerMatrix(nm.nrow(), nm.ncol());
      for (int i = 0; i < nm.nrow(); i++) {
        for (int j = 0; j < nm.ncol(); j++) {
          re_group_matrix(i, j) = static_cast<int>(nm(i, j));
        }
      }
    }
  } else {
    // Create dummy matrix
    re_group_matrix = Rcpp::IntegerMatrix(1, 1);
    re_group_matrix(0, 0) = 0;
  }

  std::vector<int> re_n_groups_vec = Rcpp::as<std::vector<int>>(re_params["n_groups_vec"]);
  bool has_re_slopes = Rcpp::as<bool>(re_params["has_slopes"]);
  bool has_re_correlated_slopes = Rcpp::as<bool>(re_params["has_correlated_slopes"]);
  std::vector<int> re_n_coefs_vec = Rcpp::as<std::vector<int>>(re_params["n_coefs_vec"]);
  std::vector<int> re_correlated_vec = Rcpp::as<std::vector<int>>(re_params["correlated_vec"]);
  std::vector<int> re_n_chol_vec = Rcpp::as<std::vector<int>>(re_params["n_chol_vec"]);
  Rcpp::List slope_matrices_list = re_params["slope_matrices"];

  // Spatial parameters (eager deep copies)
  std::string spatial_type_str = Rcpp::as<std::string>(spatial_params["type"]);
  std::vector<int> spatial_group = Rcpp::as<std::vector<int>>(spatial_params["group"]);
  int n_spatial_units = Rcpp::as<int>(spatial_params["n_units"]);
  std::vector<int> adj_row_ptr = Rcpp::as<std::vector<int>>(spatial_params["adj_row_ptr"]);
  std::vector<int> adj_col_idx = Rcpp::as<std::vector<int>>(spatial_params["adj_col_idx"]);
  std::vector<int> n_neighbors = Rcpp::as<std::vector<int>>(spatial_params["n_neighbors"]);
  double bym2_scale_factor = Rcpp::as<double>(spatial_params["bym2_scale"]);

  // Precision mass matrix data (Q_inv and L_Q for ICAR/BYM2)
  std::vector<double> spatial_Q_inv;
  std::vector<double> spatial_L_Q;
  if (spatial_params.containsElementNamed("Q_inv") &&
      spatial_params.containsElementNamed("L_Q")) {
    SEXP qi_sexp = spatial_params["Q_inv"];
    SEXP lq_sexp = spatial_params["L_Q"];
    if (!Rf_isNull(qi_sexp) && !Rf_isNull(lq_sexp)) {
      spatial_Q_inv = Rcpp::as<std::vector<double>>(qi_sexp);
      spatial_L_Q = Rcpp::as<std::vector<double>>(lq_sexp);
    }
  }

  // Temporal parameters (eager deep copies)
  std::string temporal_type_str = Rcpp::as<std::string>(temporal_params["type"]);
  std::vector<int> temporal_time_idx = Rcpp::as<std::vector<int>>(temporal_params["time_idx"]);
  std::vector<int> temporal_group_idx = Rcpp::as<std::vector<int>>(temporal_params["group_idx"]);
  int n_times = Rcpp::as<int>(temporal_params["n_times"]);
  int n_temporal_groups = Rcpp::as<int>(temporal_params["n_groups"]);
  int n_temporal_params = Rcpp::as<int>(temporal_params["n_params"]);
  bool temporal_cyclic = Rcpp::as<bool>(temporal_params["cyclic"]);
  bool temporal_shared = Rcpp::as<bool>(temporal_params["shared"]);
  double tau_temporal_shape = Rcpp::as<double>(temporal_params["tau_shape"]);
  double tau_temporal_rate = Rcpp::as<double>(temporal_params["tau_rate"]);

  // Prior parameters
  double sigma_beta = Rcpp::as<double>(prior_params["sigma_beta"]);
  double sigma_re_scale = Rcpp::as<double>(prior_params["sigma_re_scale"]);
  double phi_prior_shape = Rcpp::as<double>(prior_params["phi_shape"]);
  double phi_prior_rate = Rcpp::as<double>(prior_params["phi_rate"]);
  double tau_spatial_shape = Rcpp::as<double>(prior_params["tau_spatial_shape"]);
  double tau_spatial_rate = Rcpp::as<double>(prior_params["tau_spatial_rate"]);

  // Zero-inflation parameters
  std::string zi_type_str = Rcpp::as<std::string>(zi_params["type"]);

  // Handle X_zi which may be numeric matrix or NULL
  Rcpp::NumericMatrix X_zi;
  SEXP xzi_sexp = zi_params["X"];
  if (!Rf_isNull(xzi_sexp) && Rf_isMatrix(xzi_sexp)) {
    X_zi = Rcpp::as<Rcpp::NumericMatrix>(xzi_sexp);
  } else {
    // Create empty dummy matrix
    X_zi = Rcpp::NumericMatrix(1, 1);
    X_zi(0, 0) = 1.0;
  }
  double zi_prior_sd = Rcpp::as<double>(zi_params["prior_sd"]);

  // One-inflation parameters (for OI-binomial and ZOIB)
  Rcpp::NumericMatrix X_oi;
  SEXP xoi_sexp = zi_params["X_oi"];
  if (!Rf_isNull(xoi_sexp) && Rf_isMatrix(xoi_sexp)) {
    X_oi = Rcpp::as<Rcpp::NumericMatrix>(xoi_sexp);
  } else {
    // Create empty dummy matrix
    X_oi = Rcpp::NumericMatrix(1, 1);
    X_oi(0, 0) = 1.0;
  }
  int p_oi = 0;
  SEXP p_oi_sexp = zi_params["p_oi"];
  if (!Rf_isNull(p_oi_sexp)) {
    p_oi = Rcpp::as<int>(p_oi_sexp);
  }
  double oi_prior_sd = zi_prior_sd;  // Default to same as ZI
  SEXP oi_prior_sd_sexp = zi_params["oi_prior_sd"];
  if (!Rf_isNull(oi_prior_sd_sexp)) {
    oi_prior_sd = Rcpp::as<double>(oi_prior_sd_sexp);
  }

  // Latent factor parameters
  bool has_latent = Rcpp::as<bool>(latent_params["has_latent"]);
  int latent_n_factors = Rcpp::as<int>(latent_params["n_factors"]);
  bool latent_shared = Rcpp::as<bool>(latent_params["shared"]);
  bool latent_scale = Rcpp::as<bool>(latent_params["scale"]);
  int latent_constraint = Rcpp::as<int>(latent_params["constraint"]);
  double latent_sigma_prior_rate = Rcpp::as<double>(latent_params["sigma_prior_rate"]);

  // =========================================================================
  // Set up model data
  // =========================================================================
  ModelData data;

  // Copy response data
  data.legacy.y_num = std::vector<int>(y_num.begin(), y_num.end());
  data.legacy.y_denom = std::vector<int>(y_denom.begin(), y_denom.end());
  data.legacy.y_num_cont = std::vector<double>(y_num_cont.begin(), y_num_cont.end());
  data.legacy.y_denom_cont = std::vector<double>(y_denom_cont.begin(), y_denom_cont.end());

  // Flatten design matrices for cache efficiency
  data.legacy.p_num = X_num.ncol();
  data.legacy.p_denom = X_denom.ncol();
  data.N = y_num.size();

  data.legacy.X_num_flat.resize(data.N * data.legacy.p_num);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.legacy.p_num; j++) {
      data.legacy.X_num_flat[i * data.legacy.p_num + j] = X_num(i, j);
    }
  }

  data.legacy.X_denom_flat.resize(data.N * data.legacy.p_denom);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.legacy.p_denom; j++) {
      data.legacy.X_denom_flat[i * data.legacy.p_denom + j] = X_denom(i, j);
    }
  }

  // Random effects (already deep copied above)
  data.re_group = re_group;
  data.n_re_groups = n_re_groups;
  data.n_re_terms = n_re_terms;

  // Random slopes flags
  data.has_re_slopes = has_re_slopes;
  data.has_re_correlated_slopes = has_re_correlated_slopes;

  // RE parameterization: 0 = centered, 1 = non-centered
  // Non-centered uses z ~ N(0,1) prior, centered uses re ~ N(0, sigma^2)
  data.re_parameterization = Rcpp::as<int>(re_params["parameterization"]);

  if (n_re_terms > 0) {
    // Multi-term RE structure (with or without slopes)
    data.re_group_multi.resize(n_re_terms);
    data.re_n_groups_multi.resize(n_re_terms);
    data.re_offsets.resize(n_re_terms);
    data.re_n_coefs.resize(n_re_terms);
    data.re_correlated.resize(n_re_terms);
    data.re_n_chol.resize(n_re_terms);
    data.re_n_slopes.resize(n_re_terms);
    data.re_slope_matrices.resize(n_re_terms);

    int offset = 0;
    int total_re_params = 0;
    int total_sigma_params = 0;
    int total_chol_params = 0;

    for (int t = 0; t < n_re_terms; t++) {
      data.re_n_groups_multi[t] = re_n_groups_vec[t];
      data.re_offsets[t] = offset;

      // Slopes metadata
      int n_coefs = has_re_slopes ? re_n_coefs_vec[t] : 1;
      data.re_n_coefs[t] = n_coefs;
      data.re_correlated[t] = has_re_slopes ? (re_correlated_vec[t] != 0) : false;
      data.re_n_chol[t] = has_re_slopes ? re_n_chol_vec[t] : 0;
      data.re_n_slopes[t] = n_coefs - 1;  // Number of slopes (excluding intercept)

      // Process slope matrix for this term
      if (has_re_slopes && data.re_n_slopes[t] > 0 && slope_matrices_list.size() > t) {
        SEXP mat_sexp = slope_matrices_list[t];
        if (!Rf_isNull(mat_sexp)) {
          Rcpp::NumericMatrix slope_mat(mat_sexp);
          int n_rows = slope_mat.nrow();
          int n_cols = slope_mat.ncol();
          data.re_slope_matrices[t].resize(n_rows * n_cols);
          for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
              data.re_slope_matrices[t][i * n_cols + j] = slope_mat(i, j);
            }
          }
        }
      }

      offset += re_n_groups_vec[t];
      total_re_params += re_n_groups_vec[t] * n_coefs;
      total_sigma_params += n_coefs;
      total_chol_params += data.re_n_chol[t];

      // Extract column t from re_group_matrix
      data.re_group_multi[t].resize(data.N);
      for (int i = 0; i < data.N; i++) {
        data.re_group_multi[t][i] = re_group_matrix(i, t);
      }
    }
    data.total_re_groups = offset;

    // Build contiguous flat array: obs-major layout re_group_multi_flat[i * n_re_terms + t]
    // Obs-major is cache-friendly: inner loop over terms for each observation
    data.re_group_multi_flat.resize(n_re_terms * data.N);
    for (int i = 0; i < data.N; i++) {
      for (int t = 0; t < n_re_terms; t++) {
        data.re_group_multi_flat[i * n_re_terms + t] = data.re_group_multi[t][i];
      }
    }
    data.total_re_params = total_re_params;
    data.total_sigma_params = total_sigma_params;
    data.total_chol_params = total_chol_params;
  } else {
    // No RE terms
    data.total_re_groups = n_re_groups;
    data.total_re_params = n_re_groups;
    data.total_sigma_params = (n_re_groups > 0) ? 1 : 0;
    data.total_chol_params = 0;
  }

  // Model type
  if (model_type_str == "binomial") {
    data.legacy.model_type = ModelType::BINOMIAL;
  } else if (model_type_str == "negbin_negbin") {
    data.legacy.model_type = ModelType::NEGBIN_NEGBIN;
  } else if (model_type_str == "poisson_gamma") {
    data.legacy.model_type = ModelType::POISSON_GAMMA;
  } else if (model_type_str == "negbin_gamma") {
    data.legacy.model_type = ModelType::NEGBIN_GAMMA;
  } else if (model_type_str == "gamma_gamma") {
    data.legacy.model_type = ModelType::GAMMA_GAMMA;
  } else if (model_type_str == "lognormal") {
    data.legacy.model_type = ModelType::LOGNORMAL;
  } else if (model_type_str == "beta_binomial") {
    data.legacy.model_type = ModelType::BETA_BINOMIAL;
  } else {
    data.legacy.model_type = ModelType::POISSON_GAMMA;  // fallback
  }

  // Spatial structure
  if (spatial_type_str == "icar") {
    data.spatial_type = SpatialType::ICAR;
  } else if (spatial_type_str == "bym2") {
    data.spatial_type = SpatialType::BYM2;
  } else if (spatial_type_str == "car_proper") {
    data.spatial_type = SpatialType::CAR_PROPER;
  } else {
    data.spatial_type = SpatialType::NONE;
  }

  data.spatial_group = spatial_group;  // Already deep copied above
  data.n_spatial_units = n_spatial_units;
  data.adj_row_ptr = adj_row_ptr;
  data.adj_col_idx = adj_col_idx;
  data.n_neighbors = n_neighbors;
  data.bym2_scale_factor = bym2_scale_factor;
  data.spatial_Q_inv = std::move(spatial_Q_inv);
  data.spatial_L_Q = std::move(spatial_L_Q);

  // Proper-CAR rho bounds (eigenvalue-derived in R; defaults to (0, 1))
  if (data.spatial_type == SpatialType::CAR_PROPER &&
      spatial_params.containsElementNamed("rho_lower") &&
      spatial_params.containsElementNamed("rho_upper")) {
    data.car_rho_lower = Rcpp::as<double>(spatial_params["rho_lower"]);
    data.car_rho_upper = Rcpp::as<double>(spatial_params["rho_upper"]);
  } else {
    data.car_rho_lower = 0.0;
    data.car_rho_upper = 1.0;
  }

  // Collapsed ICAR/BYM2 parameterization
  data.icar_collapsed = false;
  data.bym2_collapsed = false;
  if (spatial_params.containsElementNamed("parameterization")) {
      std::string spatial_param_str = Rcpp::as<std::string>(spatial_params["parameterization"]);
      if (spatial_param_str == "collapsed") {
          if (data.spatial_type == SpatialType::ICAR) {
              data.icar_collapsed = true;
          } else if (data.spatial_type == SpatialType::BYM2) {
              data.bym2_collapsed = true;
          }
      }
  }

  // Temporal structure
  if (temporal_type_str == "rw1") {
    data.temporal_type = TemporalType::RW1;
  } else if (temporal_type_str == "rw2") {
    data.temporal_type = TemporalType::RW2;
  } else if (temporal_type_str == "ar1") {
    data.temporal_type = TemporalType::AR1;
  } else if (temporal_type_str == "gp") {
    data.temporal_type = TemporalType::GP;

    // GP-specific parameters
    data.has_temporal_gp = true;
    data.temporal_gp_data.n_obs = data.n_times;  // Use n_times, not N (total obs)
    data.temporal_gp_data.n_groups = n_temporal_groups;
    data.temporal_gp_data.time_values = Rcpp::as<std::vector<double>>(temporal_params["time_values"]);
    data.temporal_gp_data.group_index = temporal_group_idx;

    // Parse covariance type
    std::string cov_type_str = Rcpp::as<std::string>(temporal_params["cov_type"]);
    data.temporal_gp_data.cov_type = tulpa_temporal_gp::parse_temporal_cov_type(cov_type_str);
    data.temporal_gp_data.nu = Rcpp::as<double>(temporal_params["nu"]);
    data.temporal_gp_data.period = Rcpp::as<double>(temporal_params["period"]);
    data.temporal_gp_data.shared = temporal_shared;

    // GP priors
    data.temporal_gp_sigma2_prior_U = Rcpp::as<double>(temporal_params["gp_sigma2_prior_U"]);
    data.temporal_gp_sigma2_prior_alpha = Rcpp::as<double>(temporal_params["gp_sigma2_prior_alpha"]);
    data.temporal_gp_phi_prior_lower = Rcpp::as<double>(temporal_params["gp_phi_prior_lower"]);
    data.temporal_gp_phi_prior_upper = Rcpp::as<double>(temporal_params["gp_phi_prior_upper"]);

    // Parameterization: 0=centered, 1=non-centered (default NC)
    std::string gp_param_str = "noncentered";
    if (temporal_params.containsElementNamed("gp_parameterization")) {
        gp_param_str = Rcpp::as<std::string>(temporal_params["gp_parameterization"]);
    }
    data.temporal_gp_parameterization = (gp_param_str == "centered") ? 0 : 1;
  } else if (temporal_type_str == "multiscale") {
    data.temporal_type = TemporalType::NONE;  // MS_t uses its own data path
    data.has_temporal_gp = false;
    data.has_multiscale_temporal = true;

    // Extract MS_t data from temporal_params
    data.multiscale_temporal_data.n_times = n_times;
    data.multiscale_temporal_data.n_groups = n_temporal_groups;
    data.multiscale_temporal_data.n_obs = data.N;
    data.multiscale_temporal_data.time_index = temporal_time_idx;
    data.multiscale_temporal_data.group_index = temporal_group_idx;
    data.multiscale_temporal_data.shared = temporal_shared;

    int ms_seasonal_period = 0;
    if (temporal_params.containsElementNamed("seasonal_period"))
      ms_seasonal_period = Rcpp::as<int>(temporal_params["seasonal_period"]);
    data.multiscale_temporal_data.seasonal_period = ms_seasonal_period;

    std::string ms_trend_type = "rw1";
    if (temporal_params.containsElementNamed("trend_type"))
      ms_trend_type = Rcpp::as<std::string>(temporal_params["trend_type"]);
    std::string ms_short_type = "ar1";
    if (temporal_params.containsElementNamed("short_term_type"))
      ms_short_type = Rcpp::as<std::string>(temporal_params["short_term_type"]);
    data.multiscale_temporal_data.trend_type = tulpa_temporal::parse_temporal_type(ms_trend_type);
    data.multiscale_temporal_data.short_term_type = tulpa_temporal::parse_temporal_type(ms_short_type);

    // MS_t priors
    data.ms_sigma2_trend_prior_U = 1.0;
    data.ms_sigma2_trend_prior_alpha = 0.01;
    data.ms_sigma2_seasonal_prior_U = 1.0;
    data.ms_sigma2_seasonal_prior_alpha = 0.01;
    data.ms_sigma2_short_prior_U = 1.0;
    data.ms_sigma2_short_prior_alpha = 0.01;
    if (temporal_params.containsElementNamed("sigma2_trend_prior_U"))
      data.ms_sigma2_trend_prior_U = Rcpp::as<double>(temporal_params["sigma2_trend_prior_U"]);
    if (temporal_params.containsElementNamed("sigma2_trend_prior_alpha"))
      data.ms_sigma2_trend_prior_alpha = Rcpp::as<double>(temporal_params["sigma2_trend_prior_alpha"]);
    if (temporal_params.containsElementNamed("sigma2_short_prior_U"))
      data.ms_sigma2_short_prior_U = Rcpp::as<double>(temporal_params["sigma2_short_prior_U"]);
    if (temporal_params.containsElementNamed("sigma2_short_prior_alpha"))
      data.ms_sigma2_short_prior_alpha = Rcpp::as<double>(temporal_params["sigma2_short_prior_alpha"]);
  } else {
    data.temporal_type = TemporalType::NONE;
    data.has_temporal_gp = false;
  }

  data.temporal_time_idx = temporal_time_idx;  // Already deep copied above
  data.temporal_group_idx = temporal_group_idx;
  data.n_times = n_times;
  data.n_temporal_groups = n_temporal_groups;
  data.n_temporal_params = n_temporal_params;
  data.temporal_cyclic = temporal_cyclic;
  data.temporal_shared = temporal_shared;
  data.tau_temporal_shape = tau_temporal_shape;
  data.tau_temporal_rate = tau_temporal_rate;

  // Zero-inflation structure
  data.zi_type = tulpa_zi::parse_zi_type(zi_type_str);
  // Use explicit p_zi from R (not X_zi.ncol()) because OI-only models
  // pass a 1-column placeholder X_zi but p_zi=0
  {
    SEXP p_zi_sexp = zi_params["p_zi"];
    data.p_zi = (!Rf_isNull(p_zi_sexp)) ? Rcpp::as<int>(p_zi_sexp) : X_zi.ncol();
  }
  data.zi_prior_sd = zi_prior_sd;
  data.X_zi_flat.resize(data.N * data.p_zi);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_zi; j++) {
      data.X_zi_flat[i * data.p_zi + j] = X_zi(i, j);
    }
  }

  // One-inflation structure (for OI-binomial and ZOIB)
  data.p_oi = p_oi;
  data.oi_prior_sd = oi_prior_sd;
  if (p_oi > 0) {
    data.X_oi_flat.resize(data.N * data.p_oi);
    for (int i = 0; i < data.N; i++) {
      for (int j = 0; j < data.p_oi; j++) {
        data.X_oi_flat[i * data.p_oi + j] = X_oi(i, j);
      }
    }
  }

  // Priors
  data.sigma_beta = sigma_beta;
  data.sigma_re_scale = sigma_re_scale;
  data.phi_prior_shape = phi_prior_shape;
  data.phi_prior_rate = phi_prior_rate;
  data.tau_spatial_shape = tau_spatial_shape;
  data.tau_spatial_rate = tau_spatial_rate;

  // Parallelization
  data.n_threads = n_threads;

  // Initialize feature flags that are not used in cpp_hmc_fit (only in cpp_hmc_fit_gp)
  data.has_gp = false;
  data.has_multiscale_gp = false;
  data.has_multiscale_temporal = false;
  data.has_rsr = false;
  data.has_svc = false;
  data.has_hsgp = false;

  // Latent factors
  data.has_latent = has_latent;
  data.latent_n_factors = latent_n_factors;
  data.latent_shared = latent_shared;
  data.latent_scale = latent_scale;
  data.latent_constraint = latent_constraint;
  data.latent_sigma_prior_rate = latent_sigma_prior_rate;

  // Spatiotemporal interaction - extract from list
  bool has_spatiotemporal = st_params.size() > 0 && Rcpp::as<bool>(st_params["has_spatiotemporal"]);
  data.has_spatiotemporal = has_spatiotemporal;
  if (has_spatiotemporal) {
    // Extract parameters from list (eager deep copies to prevent R GC issues)
    std::string st_type_str = Rcpp::as<std::string>(st_params["type"]);
    bool st_shared = Rcpp::as<bool>(st_params["shared"]);
    int st_n_spatial = Rcpp::as<int>(st_params["n_spatial"]);
    int st_n_times = Rcpp::as<int>(st_params["n_times"]);
    int st_n_params = Rcpp::as<int>(st_params["n_params"]);
    std::vector<int> st_s_idx = Rcpp::as<std::vector<int>>(st_params["s_idx"]);
    std::vector<int> st_t_idx = Rcpp::as<std::vector<int>>(st_params["t_idx"]);
    std::vector<int> st_flat = Rcpp::as<std::vector<int>>(st_params["st_flat"]);
    std::string st_temporal_type_str = Rcpp::as<std::string>(st_params["temporal_type"]);
    bool st_temporal_cyclic = Rcpp::as<bool>(st_params["temporal_cyclic"]);
    std::vector<int> st_adj_row_ptr = Rcpp::as<std::vector<int>>(st_params["adj_row_ptr"]);
    std::vector<int> st_adj_col_idx = Rcpp::as<std::vector<int>>(st_params["adj_col_idx"]);
    double st_sigma2_prior_U = Rcpp::as<double>(st_params["sigma2_prior_U"]);
    double st_sigma2_prior_alpha = Rcpp::as<double>(st_params["sigma2_prior_alpha"]);

    // Parse ST type (accept both R-side "I"/"IV" and legacy "type_i"/"type_iv")
    if (st_type_str == "I" || st_type_str == "type_i") {
      data.spatiotemporal_data.type = STType::TYPE_I;
    } else if (st_type_str == "II" || st_type_str == "type_ii") {
      data.spatiotemporal_data.type = STType::TYPE_II;
    } else if (st_type_str == "III" || st_type_str == "type_iii") {
      data.spatiotemporal_data.type = STType::TYPE_III;
    } else if (st_type_str == "IV" || st_type_str == "type_iv") {
      data.spatiotemporal_data.type = STType::TYPE_IV;
    } else if (st_type_str == "separable") {
      data.spatiotemporal_data.type = STType::SEPARABLE;
    } else if (st_type_str == "nonsep_gp") {
      data.spatiotemporal_data.type = STType::NONSEP_GP;
    } else {
      Rcpp::stop("Unknown spatiotemporal type: '%s'. Expected one of: I, II, III, IV, separable, nonsep_gp",
                 st_type_str.c_str());
    }

    data.spatiotemporal_data.shared = st_shared;
    data.spatiotemporal_data.n_spatial = st_n_spatial;
    data.spatiotemporal_data.n_times = st_n_times;
    data.spatiotemporal_data.n_params = st_n_params;

    // Observation indexing (already deep copied above)
    data.spatiotemporal_data.s_idx = st_s_idx;
    data.spatiotemporal_data.t_idx = st_t_idx;
    data.spatiotemporal_data.st_flat = st_flat;

    // Temporal type for Type II/IV
    if (st_temporal_type_str == "rw1") {
      data.spatiotemporal_data.temporal_type = TemporalType::RW1;
    } else if (st_temporal_type_str == "rw2") {
      data.spatiotemporal_data.temporal_type = TemporalType::RW2;
    } else if (st_temporal_type_str == "ar1") {
      data.spatiotemporal_data.temporal_type = TemporalType::AR1;
    } else {
      data.spatiotemporal_data.temporal_type = TemporalType::RW1;  // Default
    }
    data.spatiotemporal_data.temporal_cyclic = st_temporal_cyclic;

    // Spatial adjacency for Type III/IV (already deep copied above)
    data.spatiotemporal_data.adj_row_ptr = st_adj_row_ptr;
    data.spatiotemporal_data.adj_col_idx = st_adj_col_idx;

    // Prior parameters
    data.st_sigma2_prior_U = st_sigma2_prior_U;
    data.st_sigma2_prior_alpha = st_sigma2_prior_alpha;

    // Parameterization: centered by default. NC requires spectral decomposition
    // (Kronecker eigenvectors of Q_s ⊗ Q_t) which is not yet implemented.
    // Simple scaling NC (z = delta * sqrt(tau)) preserves GMRF anisotropy
    // and makes performance worse (eps=0.003, td=11.5 vs eps=0.006, td=10).
    data.st_parameterization = 0;  // Always centered for now
    if (st_params.containsElementNamed("parameterization")) {
      std::string st_param_str = Rcpp::as<std::string>(st_params["parameterization"]);
      data.st_parameterization = (st_param_str == "centered") ? 0 : 1;
    }

    // Kronecker precision data for ST_IV (precomputed in R)
    if (st_params.containsElementNamed("Qs_inv") &&
        st_params.containsElementNamed("Qt_inv")) {
      SEXP qs_sexp = st_params["Qs_inv"];
      SEXP ls_sexp = st_params["Ls"];
      SEXP qt_sexp = st_params["Qt_inv"];
      SEXP lt_sexp = st_params["Lt"];
      if (!Rf_isNull(qs_sexp) && !Rf_isNull(qt_sexp) &&
          !Rf_isNull(ls_sexp) && !Rf_isNull(lt_sexp)) {
        data.st_Qs_inv = Rcpp::as<std::vector<double>>(qs_sexp);
        data.st_Ls = Rcpp::as<std::vector<double>>(ls_sexp);
        data.st_Qt_inv = Rcpp::as<std::vector<double>>(qt_sexp);
        data.st_Lt = Rcpp::as<std::vector<double>>(lt_sexp);
      }
    }

    // HSGP-ST: spectral basis spatiotemporal interaction
    data.st_is_hsgp = false;
    if (st_params.containsElementNamed("st_is_hsgp") &&
        Rcpp::as<bool>(st_params["st_is_hsgp"])) {
      data.st_is_hsgp = true;
      data.spatiotemporal_data.is_hsgp = true;
      int st_hsgp_m = Rcpp::as<int>(st_params["hsgp_m"]);
      double st_hsgp_c = Rcpp::as<double>(st_params["hsgp_c"]);
      std::vector<double> st_hsgp_coords = Rcpp::as<std::vector<double>>(st_params["hsgp_coords"]);
      bool st_hsgp_scale = true;
      if (st_params.containsElementNamed("hsgp_scale_coords"))
        st_hsgp_scale = Rcpp::as<bool>(st_params["hsgp_scale_coords"]);

      // Setup HSGP basis (Phi matrix + eigenvalues)
      tulpa_hsgp::setup_hsgp_2d(
        st_hsgp_coords, data.N,
        st_hsgp_m, st_hsgp_c, st_hsgp_scale,
        data.st_hsgp_data);
      data.spatiotemporal_data.hsgp_m_total = data.st_hsgp_data.m_total;
      // Override n_spatial and n_params for HSGP-ST
      data.spatiotemporal_data.n_spatial = data.st_hsgp_data.m_total;
      data.spatiotemporal_data.n_params = data.st_hsgp_data.m_total * st_n_times;
    }
  } else {
    data.spatiotemporal_data.type = STType::NONE;
  }

  // TVC (Temporally-Varying Coefficients) parameters
  bool has_tvc = tvc_params.size() > 0 && Rcpp::as<bool>(tvc_params["has_tvc"]);
  data.has_tvc = has_tvc;
  if (has_tvc) {
    // Extract TVC parameters (eager deep copies to prevent R GC issues)
    int tvc_n_tvc = Rcpp::as<int>(tvc_params["n_tvc"]);
    int tvc_n_times = Rcpp::as<int>(tvc_params["n_times"]);
    int tvc_n_groups = Rcpp::as<int>(tvc_params["n_groups"]);
    std::string tvc_structure_str = Rcpp::as<std::string>(tvc_params["structure"]);
    bool tvc_shared = Rcpp::as<bool>(tvc_params["shared"]);
    bool tvc_cyclic = Rcpp::as<bool>(tvc_params["cyclic"]);
    std::vector<int> tvc_indices = Rcpp::as<std::vector<int>>(tvc_params["tvc_indices"]);
    std::vector<int> tvc_time_index = Rcpp::as<std::vector<int>>(tvc_params["time_index"]);
    std::vector<int> tvc_group_index = Rcpp::as<std::vector<int>>(tvc_params["group_index"]);
    std::vector<double> tvc_X_tvc = Rcpp::as<std::vector<double>>(tvc_params["X_tvc"]);
    double tvc_tau_shape = Rcpp::as<double>(tvc_params["tau_shape"]);
    double tvc_tau_rate = Rcpp::as<double>(tvc_params["tau_rate"]);

    // Populate TVC data structure
    data.tvc_data.n_obs = data.N;
    data.tvc_data.n_tvc = tvc_n_tvc;
    data.tvc_data.n_times = tvc_n_times;
    data.tvc_data.n_groups = tvc_n_groups;
    data.tvc_data.shared = tvc_shared;
    data.tvc_data.cyclic = tvc_cyclic;
    data.tvc_data.tvc_indices = tvc_indices;
    data.tvc_data.time_index = tvc_time_index;
    data.tvc_data.group_index = tvc_group_index;
    data.tvc_data.X_tvc = tvc_X_tvc;

    // Parse TVC temporal structure
    if (tvc_structure_str == "rw1") {
      data.tvc_data.structure = tulpa_temporal::TemporalType::RW1;
    } else if (tvc_structure_str == "rw2") {
      data.tvc_data.structure = tulpa_temporal::TemporalType::RW2;
    } else if (tvc_structure_str == "ar1") {
      data.tvc_data.structure = tulpa_temporal::TemporalType::AR1;
    } else if (tvc_structure_str == "iid") {
      data.tvc_data.structure = tulpa_temporal::TemporalType::IID;
    } else {
      data.tvc_data.structure = tulpa_temporal::TemporalType::RW1;  // Default
    }

    // Prior parameters
    data.tvc_tau_shape = tvc_tau_shape;
    data.tvc_tau_rate = tvc_tau_rate;

    // Pre-allocate gradient workspace buffers (avoids per-call heap allocation)
    data.tvc_data.init_workspace();
  } else {
    data.tvc_data.n_tvc = 0;
    data.tvc_data.n_times = 0;
    data.tvc_data.n_groups = 1;
  }

  // SVC (Spatially-Varying Coefficients) parameters
  bool has_svc = svc_params.size() > 0 && Rcpp::as<bool>(svc_params["has_svc"]);
  data.has_svc = has_svc;
  if (has_svc) {
    // Extract SVC parameters (eager deep copies to prevent R GC issues)
    int svc_n_svc = Rcpp::as<int>(svc_params["n_svc"]);
    int svc_nn = Rcpp::as<int>(svc_params["nn"]);
    bool svc_shared = Rcpp::as<bool>(svc_params["shared"]);
    std::string svc_cov_type_str = Rcpp::as<std::string>(svc_params["cov_type"]);
    std::vector<double> svc_coords = Rcpp::as<std::vector<double>>(svc_params["coords"]);
    std::vector<int> svc_indices = Rcpp::as<std::vector<int>>(svc_params["svc_indices"]);
    std::vector<double> svc_X_svc = Rcpp::as<std::vector<double>>(svc_params["X_svc"]);
    double svc_sigma2_scale = Rcpp::as<double>(svc_params["sigma2_prior_scale"]);
    double svc_phi_lower = Rcpp::as<double>(svc_params["phi_prior_lower"]);
    double svc_phi_upper = Rcpp::as<double>(svc_params["phi_prior_upper"]);

    // Check if this is HSGP-based SVC
    std::string svc_approx = "nngp";
    if (svc_params.containsElementNamed("svc_approx")) {
      svc_approx = Rcpp::as<std::string>(svc_params["svc_approx"]);
    }
    data.svc_is_hsgp = (svc_approx == "hsgp");

    // Populate SVC data structure (shared fields)
    data.svc_data.n_obs = data.N;
    data.svc_data.n_svc = svc_n_svc;
    data.svc_data.shared = svc_shared;
    data.svc_data.coords = svc_coords;
    data.svc_data.svc_indices = svc_indices;
    data.svc_data.X_svc = svc_X_svc;

    if (data.svc_is_hsgp) {
      // HSGP-based SVC: set up basis functions
      int hsgp_m = Rcpp::as<int>(svc_params["hsgp_m"]);
      double hsgp_c = Rcpp::as<double>(svc_params["hsgp_c"]);
      data.svc_hsgp_m_per_dim = hsgp_m;
      data.svc_hsgp_boundary_factor = hsgp_c;

      // Set up HSGP basis (shared across all SVC terms)
      tulpa_hsgp::setup_hsgp_2d(svc_coords, data.N, hsgp_m, hsgp_c,
                                  svc_shared, data.svc_hsgp_data);

      // No NNGP data needed
      data.svc_data.nn = 0;
    } else {
      // NNGP-based SVC: set up neighbor structure
      data.svc_data.nn = svc_nn;
      data.svc_data.nn_idx = Rcpp::as<std::vector<int>>(svc_params["nn_idx"]);
      data.svc_data.nn_dist = Rcpp::as<std::vector<double>>(svc_params["nn_dist"]);
      data.svc_data.nn_order = Rcpp::as<std::vector<int>>(svc_params["nn_order"]);
      data.svc_data.nn_order_inv = Rcpp::as<std::vector<int>>(svc_params["nn_order_inv"]);

      // Parse SVC covariance type
      if (svc_cov_type_str == "exponential") {
        data.svc_data.cov_type = tulpa_svc::CovType::EXPONENTIAL;
      } else if (svc_cov_type_str == "matern") {
        data.svc_data.cov_type = tulpa_svc::CovType::MATERN;
      } else if (svc_cov_type_str == "gaussian") {
        data.svc_data.cov_type = tulpa_svc::CovType::GAUSSIAN;
      } else if (svc_cov_type_str == "spherical") {
        data.svc_data.cov_type = tulpa_svc::CovType::SPHERICAL;
      } else {
        data.svc_data.cov_type = tulpa_svc::CovType::EXPONENTIAL;
      }
    }

    // Prior parameters
    data.svc_sigma2_prior_scale = svc_sigma2_scale;
    data.svc_phi_prior_lower = svc_phi_lower;
    data.svc_phi_prior_upper = svc_phi_upper;

    // Pre-allocate SVC workspace buffers
    data.svc_data.init_workspace();
  } else {
    data.svc_data.n_svc = 0;
    data.svc_data.n_obs = data.N;
    data.svc_data.nn = 0;
    data.svc_is_hsgp = false;
  }

  // Initialize parameters
  std::vector<double> q0(q_init.begin(), q_init.end());

  // Memory barrier to ensure all copies complete before HMC execution
  // This prevents R GC from invalidating memory during sampling
  std::atomic_thread_fence(std::memory_order_seq_cst);

  // =========================================================================
  // Gradient check only mode: compare N, A, A_r, H without sampling
  // =========================================================================
  if (gradient_check_only) {
    ParamLayout layout = compute_param_layout(data);
    double tol = 1e-4;

    // Compute numerical gradient (reference)
    std::vector<double> grad_N;
    compute_gradient_numerical(q0, data, layout, grad_N);

    // Also compute numerical gradient for impl-based log_post (used by A/A_r)
    std::vector<double> grad_N_impl;
    compute_gradient_numerical_impl(q0, data, layout, grad_N_impl);

    // Helper: compute max relative difference
    auto max_rel_diff = [](const std::vector<double>& a, const std::vector<double>& b) -> double {
      double mx = 0.0;
      for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        double diff = std::abs(a[i] - b[i]);
        double scale = std::max(1.0, std::max(std::abs(a[i]), std::abs(b[i])));
        mx = std::max(mx, diff / scale);
      }
      return mx;
    };

    // Helper: check if a function is autodiff-based (differentiates log_post_impl<T>)
    auto is_autodiff_fn = [](GradientFn fn) -> bool {
      return fn == &compute_gradient_arena ||
             fn == &compute_gradient_forward ||
             fn == &compute_gradient_autodiff;
    };

    // Try H mode
    double h_vs_n = -1.0;  // -1 means not available
    GradientFn h_fn = resolve_gradient_fn(GradientMode::HANDCODED, data, layout);
    if (h_fn != &compute_gradient_numerical && h_fn != &compute_gradient_numerical_impl) {
      // If H resolved to an autodiff function (fallback), compare against N_impl
      // since autodiff differentiates log_post_impl, not compute_log_post
      const auto& ref = is_autodiff_fn(h_fn) ? grad_N_impl : grad_N;
      std::vector<double> grad_H;
      h_fn(q0, data, layout, grad_H, nullptr);
      h_vs_n = max_rel_diff(grad_H, ref);

      // Print top-5 worst parameters when H fails
      if (h_vs_n > 1e-4) {
        std::vector<std::pair<double,int>> diffs;
        for (size_t i = 0; i < grad_H.size() && i < ref.size(); i++) {
          double diff = std::abs(grad_H[i] - ref[i]);
          double scale = std::max(1.0, std::max(std::abs(grad_H[i]), std::abs(ref[i])));
          diffs.push_back({diff/scale, (int)i});
        }
        std::sort(diffs.begin(), diffs.end(), [](auto&a,auto&b){return a.first>b.first;});
        Rprintf("  H gradient mismatch (top 5 of %d params):\n", (int)grad_H.size());
        for (int k = 0; k < std::min(5,(int)diffs.size()); k++) {
          int idx = diffs[k].second;
          Rprintf("    param[%d]: H=%.6e N=%.6e  rel_diff=%.2e\n",
                  idx, grad_H[idx], ref[idx], diffs[k].first);
        }
        // Print layout info for worst param
        int worst = diffs[0].second;
        Rprintf("  Layout: beta_num[%d-%d] beta_denom[%d-%d] re[%d+]\n",
                layout.legacy.beta_num_start, layout.legacy.beta_num_start+data.legacy.p_num-1,
                layout.legacy.beta_denom_start, layout.legacy.beta_denom_start+data.legacy.p_denom-1,
                layout.re_start);
        if (layout.has_spatial) Rprintf("  spatial[%d-%d]\n", layout.spatial_start, layout.spatial_end-1);
        if (layout.has_temporal) Rprintf("  temporal[%d-%d]\n", layout.temporal_start, layout.temporal_end-1);
        if (layout.has_spatiotemporal) Rprintf("  ST delta[%d+] tau_st[%d]\n",
                                               layout.st_delta_start, layout.log_tau_st_idx);
        if (layout.has_tvc) Rprintf("  TVC w[%d+] tau[%d+]\n", layout.tvc_w_start, layout.log_tau_tvc_start);
        if (layout.has_svc) Rprintf("  SVC w[%d+] sigma2[%d+] phi[%d+]\n",
                                    layout.svc_w_start, layout.log_sigma2_svc_start, layout.log_phi_svc_start);
        if (layout.has_re_slopes) Rprintf("  has_re_slopes=true n_re_terms=%d\n", data.n_re_terms);
        if (layout.is_temporal_gp) Rprintf("  temporal_gp: sigma2[%d] phi[%d]\n",
                                           layout.log_sigma2_temporal_gp_idx, layout.logit_phi_temporal_gp_idx);
        if (layout.has_multiscale_temporal) Rprintf("  ms_temporal\n");
        if (layout.has_latent) Rprintf("  latent\n");
      }
    }

    // Try A_r mode (arena autodiff)
    double ar_vs_n = -1.0;
    GradientFn ar_fn = resolve_gradient_fn(GradientMode::AUTODIFF_ARENA, data, layout);
    if (ar_fn != &compute_gradient_numerical && ar_fn != &compute_gradient_numerical_impl) {
      std::vector<double> grad_Ar;
      ar_fn(q0, data, layout, grad_Ar, nullptr);
      ar_vs_n = max_rel_diff(grad_Ar, grad_N_impl);
    }

    // Try A mode (forward autodiff)
    double a_vs_n = -1.0;
    GradientFn a_fn = resolve_gradient_fn(GradientMode::AUTODIFF_FWD, data, layout);
    if (a_fn != &compute_gradient_numerical && a_fn != &compute_gradient_numerical_impl) {
      std::vector<double> grad_A;
      a_fn(q0, data, layout, grad_A, nullptr);
      a_vs_n = max_rel_diff(grad_A, grad_N_impl);
    }

    // H vs A_r cross-check
    double h_vs_ar = -1.0;
    if (h_vs_n >= 0 && ar_vs_n >= 0) {
      std::vector<double> grad_H2, grad_Ar2;
      h_fn(q0, data, layout, grad_H2, nullptr);
      ar_fn(q0, data, layout, grad_Ar2, nullptr);
      h_vs_ar = max_rel_diff(grad_H2, grad_Ar2);

      // Print top-3 worst parameters when cross-check fails
      if (h_vs_ar > 1e-4) {
        std::vector<std::pair<double,int>> diffs;
        for (size_t i = 0; i < grad_H2.size() && i < grad_Ar2.size(); i++) {
          double diff = std::abs(grad_H2[i] - grad_Ar2[i]);
          double scale = std::max(1.0, std::max(std::abs(grad_H2[i]), std::abs(grad_Ar2[i])));
          diffs.push_back({diff/scale, (int)i});
        }
        std::sort(diffs.begin(), diffs.end(), [](auto&a,auto&b){return a.first>b.first;});
        Rprintf("  H vs A_r cross-check DIVERGE (top 3 of %d params):\n", (int)grad_H2.size());
        for (int k = 0; k < std::min(3,(int)diffs.size()); k++) {
          int idx = diffs[k].second;
          Rprintf("    param[%d]: H=%.8e A_r=%.8e  rel_diff=%.2e\n",
                  idx, grad_H2[idx], grad_Ar2[idx], diffs[k].first);
        }
      }
    }

    return Rcpp::List::create(
      Rcpp::Named("h_vs_n") = h_vs_n,
      Rcpp::Named("ar_vs_n") = ar_vs_n,
      Rcpp::Named("a_vs_n") = a_vs_n,
      Rcpp::Named("h_vs_ar") = h_vs_ar,
      Rcpp::Named("tol") = tol,
      // h_ok: check h_vs_n first; if it fails but h_vs_ar passes, H is still
      // correct (compute_log_post vs log_post_impl diverge for some ZI models)
      Rcpp::Named("h_ok") = (h_vs_n >= 0)
        ? ((h_vs_n < tol) ? true : (h_vs_ar >= 0 && h_vs_ar < tol))
        : NA_LOGICAL,
      Rcpp::Named("ar_ok") = (ar_vs_n >= 0) ? (ar_vs_n < tol) : NA_LOGICAL,
      Rcpp::Named("a_ok") = (a_vs_n >= 0) ? (a_vs_n < tol) : NA_LOGICAL,
      Rcpp::Named("n_params") = (int)q0.size()
    );
  }

  // Run sampler
  if (n_chains == 1) {
    ParamLayout layout = compute_param_layout(data);
    HMCResult result = run_hmc_chain(
      q0, data, layout, n_iter, n_warmup, L, 0, seed, verbose, max_treedepth, metric_type, adapt_delta, riemannian
    );

    Rcpp::List ret = Rcpp::List::create(
      Rcpp::Named("samples") = result.samples,
      Rcpp::Named("log_prob") = result.log_prob,
      Rcpp::Named("accept_prob") = result.accept_prob,
      Rcpp::Named("n_leapfrog") = result.n_leapfrog,
      Rcpp::Named("treedepth") = result.treedepth,
      Rcpp::Named("divergent") = result.divergent,
      Rcpp::Named("epsilon") = result.epsilon,
      Rcpp::Named("n_warmup") = result.n_warmup,
      Rcpp::Named("n_sample") = result.n_sample,
      Rcpp::Named("n_chains") = 1,
      Rcpp::Named("sampler") = result.sampler.empty()
        ? ((L == 0) ? std::string("NUTS") : std::string("HMC"))
        : result.sampler
    );
    if (result.n_gp_collapsed > 0) {
      ret["gp_w_star"] = result.gp_w_star;
    }
    if (result.n_icar_collapsed > 0) {
      ret["icar_phi_star"] = result.icar_phi_star;
      if (result.bym2_theta_star.nrow() > 0) {
        ret["bym2_theta_star"] = result.bym2_theta_star;
      }
    }
    return ret;
  } else {
    // Multiple chains
    std::vector<HMCResult> results = run_hmc_parallel_chains(
      q0, data, n_iter, n_warmup, L, n_chains, seed, verbose, max_treedepth, metric_type, adapt_delta, riemannian
    );

    // Combine results
    int n_sample = results[0].n_sample;
    int n_params = results[0].samples.ncol();

    Rcpp::List samples_list(n_chains);
    Rcpp::List log_prob_list(n_chains);
    Rcpp::List accept_prob_list(n_chains);
    Rcpp::List n_leapfrog_list(n_chains);
    Rcpp::List treedepth_list(n_chains);
    Rcpp::List divergent_list(n_chains);
    Rcpp::NumericVector epsilon_vec(n_chains);

    // Determine sampler name: if any chain switched, report it
    std::string sampler_name = (L == 0) ? "NUTS" : "HMC";
    for (int c = 0; c < n_chains; c++) {
      samples_list[c] = results[c].samples;
      log_prob_list[c] = results[c].log_prob;
      accept_prob_list[c] = results[c].accept_prob;
      n_leapfrog_list[c] = results[c].n_leapfrog;
      treedepth_list[c] = results[c].treedepth;
      divergent_list[c] = results[c].divergent;
      epsilon_vec[c] = results[c].epsilon;
      if (!results[c].sampler.empty()) {
        sampler_name = results[c].sampler;
      }
    }

    return Rcpp::List::create(
      Rcpp::Named("samples") = samples_list,
      Rcpp::Named("log_prob") = log_prob_list,
      Rcpp::Named("accept_prob") = accept_prob_list,
      Rcpp::Named("n_leapfrog") = n_leapfrog_list,
      Rcpp::Named("treedepth") = treedepth_list,
      Rcpp::Named("divergent") = divergent_list,
      Rcpp::Named("epsilon") = epsilon_vec,
      Rcpp::Named("n_warmup") = n_warmup,
      Rcpp::Named("n_sample") = n_sample,
      Rcpp::Named("n_chains") = n_chains,
      Rcpp::Named("sampler") = sampler_name
    );
  }
}

// HMC sampler for GP-based spatial models
// Parameters are bundled into lists to avoid R's .Call argument limit:
//   gp_params: GP spatial parameters
//   ms_gp_params: multiscale GP parameters
//   ms_temporal_params: multiscale temporal parameters
//   rsr_params: RSR parameters
// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit_gp(
    Rcpp::NumericVector q_init,
    Rcpp::IntegerVector y_num,
    Rcpp::IntegerVector y_denom,
    Rcpp::NumericVector y_num_cont,
    Rcpp::NumericVector y_denom_cont,
    Rcpp::NumericMatrix X_num,
    Rcpp::NumericMatrix X_denom,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    std::string model_type_str,
    Rcpp::List gp_params,
    Rcpp::List ms_gp_params,
    Rcpp::List ms_temporal_params,
    Rcpp::List rsr_params,
    Rcpp::List temporal_params,  // Regular temporal (RW1/RW2/AR1/GP) — was missing, caused silent drop
    double sigma_beta,
    double sigma_re_scale,
    double phi_prior_shape,
    double phi_prior_rate,
    std::string zi_type_str,
    Rcpp::NumericMatrix X_zi,
    double zi_prior_sd,
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    int n_threads,
    bool verbose,
    int max_treedepth = 10,
    double adapt_delta = -1.0,
    std::string metric_str = "auto",
    std::string gradient_mode_str = "auto",
    Rcpp::List tvc_params = Rcpp::List(),
    bool gradient_check_only = false
) {
  using namespace tulpa_hmc;

  // Parse metric and gradient mode from string parameters
  GradientMode grad_mode = parse_gradient_mode(gradient_mode_str);
  set_gradient_mode(grad_mode);
  MassMatrixType metric_type = parse_metric_type(metric_str);
  // Force all Rcpp parameter extractions into eagerly-copied std::vectors FIRST
  // This prevents R garbage collection from invalidating lazy Rcpp views during C++ execution
  // The original debug output workaround worked because I/O forced R to sync; this achieves
  // the same effect through explicit eager copying without visible output.

  // Extract GP parameters - convert to native C++ types immediately
  std::string gp_type_str = Rcpp::as<std::string>(gp_params["gp_type"]);

  // Force eager copy into std::vectors (not Rcpp views that could be GC'd)
  std::vector<double> coords_vec = Rcpp::as<std::vector<double>>(gp_params["coords"]);
  std::vector<int> nn_idx_vec = Rcpp::as<std::vector<int>>(gp_params["nn_idx"]);
  std::vector<double> nn_dist_vec = Rcpp::as<std::vector<double>>(gp_params["nn_dist"]);
  std::vector<int> nn_order_vec = Rcpp::as<std::vector<int>>(gp_params["nn_order"]);
  std::vector<int> nn_order_inv_vec = Rcpp::as<std::vector<int>>(gp_params["nn_order_inv"]);
  std::vector<double> nn_neighbor_dist_vec = Rcpp::as<std::vector<double>>(gp_params["nn_neighbor_dist"]);  // Phase 1.3

  int nn = Rcpp::as<int>(gp_params["nn"]);
  std::string cov_type_str = Rcpp::as<std::string>(gp_params["cov_type"]);
  double nu = Rcpp::as<double>(gp_params["nu"]);
  bool gp_shared = Rcpp::as<bool>(gp_params["shared"]);
  double gp_sigma2_prior_U = Rcpp::as<double>(gp_params["sigma2_prior_U"]);
  double gp_sigma2_prior_alpha = Rcpp::as<double>(gp_params["sigma2_prior_alpha"]);
  double gp_phi_prior_lower = Rcpp::as<double>(gp_params["phi_prior_lower"]);
  double gp_phi_prior_upper = Rcpp::as<double>(gp_params["phi_prior_upper"]);

  // GP solver configuration
  std::string gp_solver_str = Rcpp::as<std::string>(gp_params["solver"]);
  double gp_cg_tol = Rcpp::as<double>(gp_params["cg_tol"]);
  int gp_cg_maxiter = Rcpp::as<int>(gp_params["cg_maxiter"]);

  // Observation-to-location mapping (1-based from R, convert to 0-based)
  std::vector<int> gp_obs_to_loc_r = Rcpp::as<std::vector<int>>(gp_params["gp_obs_to_loc"]);
  int gp_n_unique = Rcpp::as<int>(gp_params["n_unique"]);

  // Memory barrier to ensure all extractions complete before proceeding
  std::atomic_thread_fence(std::memory_order_seq_cst);

  // Extract multiscale GP parameters - eager copy to std::vectors
  std::vector<int> nn_idx_local_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_idx_local"]);
  std::vector<double> nn_dist_local_vec = Rcpp::as<std::vector<double>>(ms_gp_params["nn_dist_local"]);
  std::vector<int> nn_order_local_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_order_local"]);
  std::vector<int> nn_order_inv_local_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_order_inv_local"]);
  int nn_local = Rcpp::as<int>(ms_gp_params["nn_local"]);
  std::vector<int> nn_idx_regional_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_idx_regional"]);
  std::vector<double> nn_dist_regional_vec = Rcpp::as<std::vector<double>>(ms_gp_params["nn_dist_regional"]);
  std::vector<int> nn_order_regional_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_order_regional"]);
  std::vector<int> nn_order_inv_regional_vec = Rcpp::as<std::vector<int>>(ms_gp_params["nn_order_inv_regional"]);
  int nn_regional = Rcpp::as<int>(ms_gp_params["nn_regional"]);
  std::vector<double> nn_neighbor_dist_local_vec = Rcpp::as<std::vector<double>>(ms_gp_params["nn_neighbor_dist_local"]);  // Phase 1.3
  std::vector<double> nn_neighbor_dist_regional_vec = Rcpp::as<std::vector<double>>(ms_gp_params["nn_neighbor_dist_regional"]);  // Phase 1.3
  double range_local_lower = Rcpp::as<double>(ms_gp_params["range_local_lower"]);
  double range_local_upper = Rcpp::as<double>(ms_gp_params["range_local_upper"]);
  double range_regional_lower = Rcpp::as<double>(ms_gp_params["range_regional_lower"]);
  double range_regional_upper = Rcpp::as<double>(ms_gp_params["range_regional_upper"]);
  double ms_sigma2_local_prior_U = Rcpp::as<double>(ms_gp_params["sigma2_local_prior_U"]);
  double ms_sigma2_local_prior_alpha = Rcpp::as<double>(ms_gp_params["sigma2_local_prior_alpha"]);
  double ms_sigma2_regional_prior_U = Rcpp::as<double>(ms_gp_params["sigma2_regional_prior_U"]);
  double ms_sigma2_regional_prior_alpha = Rcpp::as<double>(ms_gp_params["sigma2_regional_prior_alpha"]);
  std::string msgp_sampler_str = Rcpp::as<std::string>(ms_gp_params["sampler"]);

  // Extract multiscale temporal parameters - eager copy
  std::string ms_temporal_type_str = Rcpp::as<std::string>(ms_temporal_params["type"]);
  std::vector<int> ms_time_index_vec = Rcpp::as<std::vector<int>>(ms_temporal_params["time_index"]);
  std::vector<int> ms_group_index_vec = Rcpp::as<std::vector<int>>(ms_temporal_params["group_index"]);
  int ms_n_times = Rcpp::as<int>(ms_temporal_params["n_times"]);
  int ms_n_groups = Rcpp::as<int>(ms_temporal_params["n_groups"]);
  std::string trend_type_str = Rcpp::as<std::string>(ms_temporal_params["trend_type"]);
  int seasonal_period = Rcpp::as<int>(ms_temporal_params["seasonal_period"]);
  std::string short_term_type_str = Rcpp::as<std::string>(ms_temporal_params["short_term_type"]);
  bool ms_temporal_shared = Rcpp::as<bool>(ms_temporal_params["shared"]);
  double ms_sigma2_trend_prior_U = Rcpp::as<double>(ms_temporal_params["sigma2_trend_prior_U"]);
  double ms_sigma2_trend_prior_alpha = Rcpp::as<double>(ms_temporal_params["sigma2_trend_prior_alpha"]);
  double ms_sigma2_seasonal_prior_U = Rcpp::as<double>(ms_temporal_params["sigma2_seasonal_prior_U"]);
  double ms_sigma2_seasonal_prior_alpha = Rcpp::as<double>(ms_temporal_params["sigma2_seasonal_prior_alpha"]);
  double ms_sigma2_short_prior_U = Rcpp::as<double>(ms_temporal_params["sigma2_short_prior_U"]);
  double ms_sigma2_short_prior_alpha = Rcpp::as<double>(ms_temporal_params["sigma2_short_prior_alpha"]);

  // Extract RSR parameters - eager copy
  bool has_rsr = Rcpp::as<bool>(rsr_params["has_rsr"]);
  std::vector<double> rsr_projection_vec = Rcpp::as<std::vector<double>>(rsr_params["projection"]);
  int rsr_n = Rcpp::as<int>(rsr_params["n"]);

  // Second memory barrier after all Rcpp extractions
  std::atomic_thread_fence(std::memory_order_seq_cst);
  using namespace tulpa_hmc;

  // Set up model data
  ModelData data;

  // Copy response data
  data.legacy.y_num = std::vector<int>(y_num.begin(), y_num.end());
  data.legacy.y_denom = std::vector<int>(y_denom.begin(), y_denom.end());
  data.legacy.y_num_cont = std::vector<double>(y_num_cont.begin(), y_num_cont.end());
  data.legacy.y_denom_cont = std::vector<double>(y_denom_cont.begin(), y_denom_cont.end());
  data.N = y_num.size();

  // Flatten design matrices
  data.legacy.p_num = X_num.ncol();
  data.legacy.p_denom = X_denom.ncol();

  data.legacy.X_num_flat.resize(data.N * data.legacy.p_num);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.legacy.p_num; j++) {
      data.legacy.X_num_flat[i * data.legacy.p_num + j] = X_num(i, j);
    }
  }

  data.legacy.X_denom_flat.resize(data.N * data.legacy.p_denom);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.legacy.p_denom; j++) {
      data.legacy.X_denom_flat[i * data.legacy.p_denom + j] = X_denom(i, j);
    }
  }

  // Random effects (single-term legacy path)
  data.re_group = std::vector<int>(re_group.begin(), re_group.end());
  data.n_re_groups = n_re_groups;

  // Initialize multi-term RE fields to indicate single-term mode
  data.n_re_terms = 0;  // 0 means use legacy single-term path
  data.total_re_groups = n_re_groups;
  data.has_re_slopes = false;  // GP interface doesn't support random slopes
  data.has_re_correlated_slopes = false;

  // Model type
  if (model_type_str == "binomial") {
    data.legacy.model_type = ModelType::BINOMIAL;
  } else if (model_type_str == "negbin_negbin") {
    data.legacy.model_type = ModelType::NEGBIN_NEGBIN;
  } else if (model_type_str == "poisson_gamma") {
    data.legacy.model_type = ModelType::POISSON_GAMMA;
  } else if (model_type_str == "negbin_gamma") {
    data.legacy.model_type = ModelType::NEGBIN_GAMMA;
  } else if (model_type_str == "gamma_gamma") {
    data.legacy.model_type = ModelType::GAMMA_GAMMA;
  } else if (model_type_str == "lognormal") {
    data.legacy.model_type = ModelType::LOGNORMAL;
  } else if (model_type_str == "beta_binomial") {
    data.legacy.model_type = ModelType::BETA_BINOMIAL;
  } else {
    data.legacy.model_type = ModelType::POISSON_GAMMA;  // fallback
  }

  // Covariance type
  tulpa_gp::CovType cov_type;
  if (cov_type_str == "exponential") {
    cov_type = tulpa_gp::CovType::EXPONENTIAL;
  } else if (cov_type_str == "matern") {
    cov_type = tulpa_gp::CovType::MATERN;
  } else if (cov_type_str == "gaussian") {
    cov_type = tulpa_gp::CovType::GAUSSIAN;
  } else {
    cov_type = tulpa_gp::CovType::SPHERICAL;
  }

  // GP spatial structure
  if (gp_type_str == "gp") {
    data.spatial_type = SpatialType::GP;
    data.has_gp = true;
    data.has_multiscale_gp = false;
    data.has_hsgp = false;

    data.gp_data.n_obs = gp_n_unique;  // Unique locations, not total observations
    data.gp_data.nn = nn;
    data.gp_data.coords = coords_vec;  // Already std::vector from eager copy
    data.gp_data.nn_idx = nn_idx_vec;
    data.gp_data.nn_dist = nn_dist_vec;
    data.gp_data.nn_neighbor_dist = nn_neighbor_dist_vec;  // Phase 1.3: cached pairwise distances
    // Convert obs_to_loc from R's 1-based to C++'s 0-based indexing
    data.gp_data.obs_to_loc.resize(gp_obs_to_loc_r.size());
    for (size_t i = 0; i < gp_obs_to_loc_r.size(); i++) {
      data.gp_data.obs_to_loc[i] = gp_obs_to_loc_r[i] - 1;
    }
    // Convert from R's 1-based to C++'s 0-based indexing
    data.gp_data.nn_order.resize(nn_order_vec.size());
    for (size_t i = 0; i < nn_order_vec.size(); i++) {
      data.gp_data.nn_order[i] = nn_order_vec[i] - 1;
    }
    data.gp_data.nn_order_inv.resize(nn_order_inv_vec.size());
    for (size_t i = 0; i < nn_order_inv_vec.size(); i++) {
      data.gp_data.nn_order_inv[i] = nn_order_inv_vec[i] - 1;
    }
    data.gp_data.cov_type = cov_type;
    data.gp_data.nu = nu;
    data.gp_data.shared = gp_shared;

    // Set solver configuration
    data.gp_data.solver_config.solver = tulpa_gp::parse_gp_solver(gp_solver_str);
    data.gp_data.solver_config.cg_tol = gp_cg_tol;
    data.gp_data.solver_config.cg_maxiter = gp_cg_maxiter;
    data.gp_data.solver_config.n_obs = gp_n_unique;  // Unique locations

    data.gp_sigma2_prior_U = gp_sigma2_prior_U;
    data.gp_sigma2_prior_alpha = gp_sigma2_prior_alpha;
    data.gp_phi_prior_lower = gp_phi_prior_lower;
    data.gp_phi_prior_upper = gp_phi_prior_upper;

    // GP parameterization: centered (default), noncentered, or collapsed
    if (gp_params.containsElementNamed("parameterization")) {
        std::string gp_param_str = Rcpp::as<std::string>(gp_params["parameterization"]);
        if (gp_param_str == "collapsed") {
            data.gp_parameterization = 0;  // Not relevant for collapsed
            data.gp_collapsed = true;
        } else {
            data.gp_parameterization = (gp_param_str == "centered") ? 0 : 1;
            data.gp_collapsed = false;
        }
    } else {
        data.gp_parameterization = 0;  // Default: centered
        data.gp_collapsed = false;
    }

  } else if (gp_type_str == "multiscale_gp") {
    data.spatial_type = SpatialType::MULTISCALE_GP;
    data.has_gp = false;
    data.has_multiscale_gp = true;
    data.has_hsgp = false;

    // Check if using HSGP approximation
    std::string msgp_approx = "nngp";
    if (gp_params.containsElementNamed("msgp_approx")) {
      msgp_approx = Rcpp::as<std::string>(gp_params["msgp_approx"]);
    }
    data.msgp_is_hsgp = (msgp_approx == "hsgp");

    if (data.msgp_is_hsgp) {
      // HSGP-MSGP: set up shared basis functions, no NNGP neighbor computation
      int hsgp_m = Rcpp::as<int>(gp_params["hsgp_m"]);
      double hsgp_c = Rcpp::as<double>(gp_params["hsgp_c"]);
      tulpa_hsgp::setup_hsgp_2d(coords_vec, data.N, hsgp_m, hsgp_c,
                                  gp_shared, data.msgp_hsgp_data);
      data.multiscale_gp_data.shared = gp_shared;
      // Set n_obs for consistency (used by param layout check)
      data.multiscale_gp_data.n_obs = data.N;

      // Lengthscale prior means from range bounds (geometric mean on log scale)
      data.ms_log_ls_local_mean = 0.5 * (std::log(range_local_lower) + std::log(range_local_upper));
      data.ms_log_ls_local_sd = 0.5;
      data.ms_log_ls_regional_mean = 0.5 * (std::log(range_regional_lower) + std::log(range_regional_upper));
      data.ms_log_ls_regional_sd = 0.5;

      if (verbose) {
        Rcpp::Rcout << "  HSGP-MSGP: m=" << hsgp_m << ", c=" << hsgp_c
                    << ", m_total=" << data.msgp_hsgp_data.m_total
                    << " (local+regional: " << 2 * data.msgp_hsgp_data.m_total << " basis coefficients)\n";
        Rcpp::Rcout << "  Lengthscale priors: local LogN(" << data.ms_log_ls_local_mean
                    << ", " << data.ms_log_ls_local_sd << "), regional LogN("
                    << data.ms_log_ls_regional_mean << ", " << data.ms_log_ls_regional_sd << ")\n";
      }
    } else {
      // NNGP-MSGP: standard neighbor-based computation
      data.multiscale_gp_data.n_obs = gp_n_unique;  // Unique locations, not total observations
      data.multiscale_gp_data.coords = coords_vec;  // Already std::vector from eager copy
      // Convert obs_to_loc from R's 1-based to C++'s 0-based indexing
      data.multiscale_gp_data.obs_to_loc.resize(gp_obs_to_loc_r.size());
      for (size_t i = 0; i < gp_obs_to_loc_r.size(); i++) {
        data.multiscale_gp_data.obs_to_loc[i] = gp_obs_to_loc_r[i] - 1;
      }

      // Local scale - use pre-copied std::vectors
      data.multiscale_gp_data.nn_local = nn_local;
      data.multiscale_gp_data.nn_idx_local = nn_idx_local_vec;
      data.multiscale_gp_data.nn_dist_local = nn_dist_local_vec;
      // Convert from R's 1-based to C++'s 0-based indexing
      data.multiscale_gp_data.nn_order_local.resize(nn_order_local_vec.size());
      for (size_t i = 0; i < nn_order_local_vec.size(); i++) {
        data.multiscale_gp_data.nn_order_local[i] = nn_order_local_vec[i] - 1;
      }
      data.multiscale_gp_data.nn_order_inv_local.resize(nn_order_inv_local_vec.size());
      for (size_t i = 0; i < nn_order_inv_local_vec.size(); i++) {
        data.multiscale_gp_data.nn_order_inv_local[i] = nn_order_inv_local_vec[i] - 1;
      }
      data.multiscale_gp_data.nn_neighbor_dist_local = nn_neighbor_dist_local_vec;  // Phase 1.3

      // Regional scale - use pre-copied std::vectors
      data.multiscale_gp_data.nn_regional = nn_regional;
      data.multiscale_gp_data.nn_idx_regional = nn_idx_regional_vec;
      data.multiscale_gp_data.nn_dist_regional = nn_dist_regional_vec;
      // Convert from R's 1-based to C++'s 0-based indexing
      data.multiscale_gp_data.nn_order_regional.resize(nn_order_regional_vec.size());
      for (size_t i = 0; i < nn_order_regional_vec.size(); i++) {
        data.multiscale_gp_data.nn_order_regional[i] = nn_order_regional_vec[i] - 1;
      }
      data.multiscale_gp_data.nn_order_inv_regional.resize(nn_order_inv_regional_vec.size());
      for (size_t i = 0; i < nn_order_inv_regional_vec.size(); i++) {
        data.multiscale_gp_data.nn_order_inv_regional[i] = nn_order_inv_regional_vec[i] - 1;
      }
      data.multiscale_gp_data.nn_neighbor_dist_regional = nn_neighbor_dist_regional_vec;  // Phase 1.3

      // Range constraints
      data.multiscale_gp_data.range_local_lower = range_local_lower;
      data.multiscale_gp_data.range_local_upper = range_local_upper;
      data.multiscale_gp_data.range_regional_lower = range_regional_lower;
      data.multiscale_gp_data.range_regional_upper = range_regional_upper;

      data.multiscale_gp_data.cov_type = cov_type;
      data.multiscale_gp_data.nu = nu;
      data.multiscale_gp_data.sampler = tulpa_gp::parse_msgp_sampler(msgp_sampler_str);
    }

    data.multiscale_gp_data.shared = gp_shared;
    data.ms_sigma2_local_prior_U = ms_sigma2_local_prior_U;
    data.ms_sigma2_local_prior_alpha = ms_sigma2_local_prior_alpha;
    data.ms_sigma2_regional_prior_U = ms_sigma2_regional_prior_U;
    data.ms_sigma2_regional_prior_alpha = ms_sigma2_regional_prior_alpha;

  } else if (gp_type_str == "hsgp") {
    data.spatial_type = SpatialType::HSGP;
    data.has_gp = false;
    data.has_multiscale_gp = false;
    data.has_hsgp = true;

    // HSGP parameters from gp_params
    int hsgp_m = Rcpp::as<int>(gp_params["hsgp_m"]);
    double hsgp_c = Rcpp::as<double>(gp_params["hsgp_c"]);
    bool hsgp_shared = gp_shared;

    // Setup HSGP data structure with precomputed basis functions
    tulpa_hsgp::setup_hsgp_2d(coords_vec, data.N, hsgp_m, hsgp_c,
                                hsgp_shared, data.hsgp_data);

    data.hsgp_m_per_dim = hsgp_m;
    data.hsgp_boundary_factor = hsgp_c;

  } else {
    data.spatial_type = SpatialType::NONE;
    data.has_gp = false;
    data.has_multiscale_gp = false;
    data.has_hsgp = false;
  }

  // Initialize adjacency for ICAR/BYM2 (not used with GP)
  data.n_spatial_units = 0;
  data.bym2_scale_factor = 1.0;

  // Multi-scale temporal structure
  if (ms_temporal_type_str == "multiscale") {
    data.has_multiscale_temporal = true;

    data.multiscale_temporal_data.n_times = ms_n_times;
    data.multiscale_temporal_data.n_groups = ms_n_groups;
    data.multiscale_temporal_data.n_obs = data.N;
    data.multiscale_temporal_data.time_index = ms_time_index_vec;  // Already std::vector from eager copy
    data.multiscale_temporal_data.group_index = ms_group_index_vec;
    data.multiscale_temporal_data.shared = ms_temporal_shared;
    data.multiscale_temporal_data.seasonal_period = seasonal_period;

    // Parse temporal component types
    data.multiscale_temporal_data.trend_type = tulpa_temporal::parse_temporal_type(trend_type_str);
    data.multiscale_temporal_data.short_term_type = tulpa_temporal::parse_temporal_type(short_term_type_str);

    data.ms_sigma2_trend_prior_U = ms_sigma2_trend_prior_U;
    data.ms_sigma2_trend_prior_alpha = ms_sigma2_trend_prior_alpha;
    data.ms_sigma2_seasonal_prior_U = ms_sigma2_seasonal_prior_U;
    data.ms_sigma2_seasonal_prior_alpha = ms_sigma2_seasonal_prior_alpha;
    data.ms_sigma2_short_prior_U = ms_sigma2_short_prior_U;
    data.ms_sigma2_short_prior_alpha = ms_sigma2_short_prior_alpha;

  } else {
    data.has_multiscale_temporal = false;
    data.multiscale_temporal_data.trend_type = tulpa_temporal::TemporalType::NONE;
    data.multiscale_temporal_data.short_term_type = tulpa_temporal::TemporalType::NONE;
    data.multiscale_temporal_data.seasonal_period = 0;
  }

  // Regular temporal (RW1/RW2/AR1/GP) — now supported in GP interface
  {
    std::string temporal_type_str = Rcpp::as<std::string>(temporal_params["type"]);
    int n_temporal_groups = Rcpp::as<int>(temporal_params["n_groups"]);
    bool temporal_shared = Rcpp::as<bool>(temporal_params["shared"]);

    if (temporal_type_str == "rw1") {
      data.temporal_type = TemporalType::RW1;
    } else if (temporal_type_str == "rw2") {
      data.temporal_type = TemporalType::RW2;
    } else if (temporal_type_str == "ar1") {
      data.temporal_type = TemporalType::AR1;
    } else if (temporal_type_str == "gp") {
      data.temporal_type = TemporalType::GP;

      // GP-specific parameters (same parsing as cpp_hmc_fit)
      data.has_temporal_gp = true;
      data.temporal_gp_data.n_obs = Rcpp::as<int>(temporal_params["n_times"]);
      data.temporal_gp_data.n_groups = n_temporal_groups;
      data.temporal_gp_data.time_values = Rcpp::as<std::vector<double>>(temporal_params["time_values"]);
      data.temporal_gp_data.group_index = Rcpp::as<std::vector<int>>(temporal_params["group_idx"]);

      std::string cov_type_str = Rcpp::as<std::string>(temporal_params["cov_type"]);
      data.temporal_gp_data.cov_type = tulpa_temporal_gp::parse_temporal_cov_type(cov_type_str);
      data.temporal_gp_data.nu = Rcpp::as<double>(temporal_params["nu"]);
      data.temporal_gp_data.period = Rcpp::as<double>(temporal_params["period"]);
      data.temporal_gp_data.shared = temporal_shared;

      data.temporal_gp_sigma2_prior_U = Rcpp::as<double>(temporal_params["gp_sigma2_prior_U"]);
      data.temporal_gp_sigma2_prior_alpha = Rcpp::as<double>(temporal_params["gp_sigma2_prior_alpha"]);
      data.temporal_gp_phi_prior_lower = Rcpp::as<double>(temporal_params["gp_phi_prior_lower"]);
      data.temporal_gp_phi_prior_upper = Rcpp::as<double>(temporal_params["gp_phi_prior_upper"]);

      std::string gp_param_str = "noncentered";
      if (temporal_params.containsElementNamed("gp_parameterization")) {
        gp_param_str = Rcpp::as<std::string>(temporal_params["gp_parameterization"]);
      }
      data.temporal_gp_parameterization = (gp_param_str == "centered") ? 0 : 1;
    } else {
      data.temporal_type = TemporalType::NONE;
    }
    data.temporal_time_idx = Rcpp::as<std::vector<int>>(temporal_params["time_idx"]);
    data.temporal_group_idx = Rcpp::as<std::vector<int>>(temporal_params["group_idx"]);
    data.n_times = Rcpp::as<int>(temporal_params["n_times"]);
    data.n_temporal_groups = n_temporal_groups;
    data.n_temporal_params = Rcpp::as<int>(temporal_params["n_params"]);
    data.temporal_cyclic = Rcpp::as<bool>(temporal_params["cyclic"]);
    data.temporal_shared = temporal_shared;
    data.tau_temporal_shape = Rcpp::as<double>(temporal_params["tau_shape"]);
    data.tau_temporal_rate = Rcpp::as<double>(temporal_params["tau_rate"]);
  }

  // Multi-term RE structure (not used in GP interface - single term only)
  data.total_re_params = 0;
  data.total_sigma_params = 0;
  data.total_chol_params = 0;

  // RSR structure - use pre-copied std::vector
  data.has_rsr = has_rsr;
  if (has_rsr && !rsr_projection_vec.empty()) {
    data.rsr_projection = rsr_projection_vec;
    data.rsr_n = rsr_n;
  } else {
    data.rsr_n = 0;
  }

  // Zero-inflation structure (GP interface: no OI support, use matrix directly)
  data.zi_type = tulpa_zi::parse_zi_type(zi_type_str);
  data.p_zi = X_zi.ncol();
  data.zi_prior_sd = zi_prior_sd;
  data.X_zi_flat.resize(data.N * data.p_zi);
  for (int i = 0; i < data.N; i++) {
    for (int j = 0; j < data.p_zi; j++) {
      data.X_zi_flat[i * data.p_zi + j] = X_zi(i, j);
    }
  }

  // Standard priors
  data.sigma_beta = sigma_beta;
  data.sigma_re_scale = sigma_re_scale;
  data.phi_prior_shape = phi_prior_shape;
  data.phi_prior_rate = phi_prior_rate;
  data.tau_spatial_shape = 1.0;
  data.tau_spatial_rate = 0.01;

  // SVC not used in GP interface
  data.has_svc = false;

  // Latent factors not used in GP interface
  data.has_latent = false;
  data.latent_n_factors = 0;
  data.latent_shared = false;
  data.latent_scale = false;
  data.latent_constraint = 0;
  data.latent_sigma_prior_rate = 1.0;

  // Spatiotemporal not used in GP interface
  data.has_spatiotemporal = false;
  data.spatiotemporal_data.type = STType::NONE;

  // TVC (Temporally-Varying Coefficients) — now supported in GP interface
  {
    bool has_tvc = tvc_params.size() > 0 && tvc_params.containsElementNamed("has_tvc") &&
                   Rcpp::as<bool>(tvc_params["has_tvc"]);
    data.has_tvc = has_tvc;
    if (has_tvc) {
      int tvc_n_tvc = Rcpp::as<int>(tvc_params["n_tvc"]);
      int tvc_n_times = Rcpp::as<int>(tvc_params["n_times"]);
      int tvc_n_groups = Rcpp::as<int>(tvc_params["n_groups"]);
      std::string tvc_structure_str = Rcpp::as<std::string>(tvc_params["structure"]);
      bool tvc_shared = Rcpp::as<bool>(tvc_params["shared"]);
      bool tvc_cyclic = Rcpp::as<bool>(tvc_params["cyclic"]);
      std::vector<int> tvc_indices = Rcpp::as<std::vector<int>>(tvc_params["tvc_indices"]);
      std::vector<int> tvc_time_index = Rcpp::as<std::vector<int>>(tvc_params["time_index"]);
      std::vector<int> tvc_group_index = Rcpp::as<std::vector<int>>(tvc_params["group_index"]);
      std::vector<double> tvc_X_tvc = Rcpp::as<std::vector<double>>(tvc_params["X_tvc"]);
      double tvc_tau_shape = Rcpp::as<double>(tvc_params["tau_shape"]);
      double tvc_tau_rate = Rcpp::as<double>(tvc_params["tau_rate"]);

      data.tvc_data.n_obs = data.N;
      data.tvc_data.n_tvc = tvc_n_tvc;
      data.tvc_data.n_times = tvc_n_times;
      data.tvc_data.n_groups = tvc_n_groups;
      data.tvc_data.shared = tvc_shared;
      data.tvc_data.cyclic = tvc_cyclic;
      data.tvc_data.tvc_indices = tvc_indices;
      data.tvc_data.time_index = tvc_time_index;
      data.tvc_data.group_index = tvc_group_index;
      data.tvc_data.X_tvc = tvc_X_tvc;

      if (tvc_structure_str == "rw1") {
        data.tvc_data.structure = tulpa_temporal::TemporalType::RW1;
      } else if (tvc_structure_str == "rw2") {
        data.tvc_data.structure = tulpa_temporal::TemporalType::RW2;
      } else if (tvc_structure_str == "ar1") {
        data.tvc_data.structure = tulpa_temporal::TemporalType::AR1;
      } else if (tvc_structure_str == "iid") {
        data.tvc_data.structure = tulpa_temporal::TemporalType::IID;
      } else {
        data.tvc_data.structure = tulpa_temporal::TemporalType::RW1;
      }

      data.tvc_tau_shape = tvc_tau_shape;
      data.tvc_tau_rate = tvc_tau_rate;
      data.tvc_data.init_workspace();
    } else {
      data.tvc_data.n_tvc = 0;
      data.tvc_data.n_times = 0;
      data.tvc_data.n_groups = 1;
    }
  }

  // Parallelization
  data.n_threads = n_threads;

  // Final memory barrier before HMC execution
  std::atomic_thread_fence(std::memory_order_seq_cst);

  // Initialize parameters - use explicit std::vector copy from Rcpp
  std::vector<double> q0(q_init.begin(), q_init.end());

  // =========================================================================
  // Gradient check only mode: compare N, A, A_r, H without sampling
  // =========================================================================
  if (gradient_check_only) {
    ParamLayout layout = compute_param_layout(data);

    std::vector<double> grad_N;
    compute_gradient_numerical(q0, data, layout, grad_N);

    std::vector<double> grad_N_impl;
    compute_gradient_numerical_impl(q0, data, layout, grad_N_impl);

    auto max_rel_diff = [](const std::vector<double>& a, const std::vector<double>& b) -> double {
      double mx = 0.0;
      for (size_t i = 0; i < a.size() && i < b.size(); i++) {
        double diff = std::abs(a[i] - b[i]);
        double scale = std::max(1.0, std::max(std::abs(a[i]), std::abs(b[i])));
        mx = std::max(mx, diff / scale);
      }
      return mx;
    };

    auto is_autodiff_fn = [](GradientFn fn) -> bool {
      return fn == &compute_gradient_arena ||
             fn == &compute_gradient_forward ||
             fn == &compute_gradient_autodiff;
    };

    double h_vs_n = -1.0;
    GradientFn h_fn = resolve_gradient_fn(GradientMode::HANDCODED, data, layout);
    if (h_fn != &compute_gradient_numerical && h_fn != &compute_gradient_numerical_impl) {
      const auto& ref = is_autodiff_fn(h_fn) ? grad_N_impl : grad_N;
      std::vector<double> grad_H;
      h_fn(q0, data, layout, grad_H, nullptr);
      h_vs_n = max_rel_diff(grad_H, ref);
    }

    double ar_vs_n = -1.0;
    GradientFn ar_fn = resolve_gradient_fn(GradientMode::AUTODIFF_ARENA, data, layout);
    if (ar_fn != &compute_gradient_numerical && ar_fn != &compute_gradient_numerical_impl) {
      std::vector<double> grad_Ar;
      ar_fn(q0, data, layout, grad_Ar, nullptr);
      ar_vs_n = max_rel_diff(grad_Ar, grad_N_impl);
    }

    double a_vs_n = -1.0;
    GradientFn a_fn = resolve_gradient_fn(GradientMode::AUTODIFF_FWD, data, layout);
    if (a_fn != &compute_gradient_numerical && a_fn != &compute_gradient_numerical_impl) {
      std::vector<double> grad_A;
      a_fn(q0, data, layout, grad_A, nullptr);
      a_vs_n = max_rel_diff(grad_A, grad_N_impl);
    }

    double h_vs_ar = -1.0;
    if (h_vs_n >= 0 && ar_vs_n >= 0) {
      std::vector<double> grad_H2, grad_Ar2;
      h_fn(q0, data, layout, grad_H2, nullptr);
      ar_fn(q0, data, layout, grad_Ar2, nullptr);
      h_vs_ar = max_rel_diff(grad_H2, grad_Ar2);

      if (h_vs_ar > 1e-4) {
        std::vector<std::pair<double,int>> diffs;
        for (size_t i = 0; i < grad_H2.size() && i < grad_Ar2.size(); i++) {
          double diff = std::abs(grad_H2[i] - grad_Ar2[i]);
          double scale = std::max(1.0, std::max(std::abs(grad_H2[i]), std::abs(grad_Ar2[i])));
          diffs.push_back({diff/scale, (int)i});
        }
        std::sort(diffs.begin(), diffs.end(), [](auto&a,auto&b){return a.first>b.first;});
        Rprintf("  H vs A_r cross-check DIVERGE (top 3 of %d params):\n", (int)grad_H2.size());
        for (int k = 0; k < std::min(3,(int)diffs.size()); k++) {
          int idx = diffs[k].second;
          Rprintf("    param[%d]: H=%.8e A_r=%.8e  rel_diff=%.2e\n",
                  idx, grad_H2[idx], grad_Ar2[idx], diffs[k].first);
        }
        // Print MSGP layout info
        if (layout.is_multiscale_gp) {
          Rprintf("  MSGP: sigma2_local[%d] phi_local[%d] sigma2_reg[%d] phi_reg[%d]\n",
                  layout.log_sigma2_gp_local_idx, layout.log_phi_gp_local_idx,
                  layout.log_sigma2_gp_regional_idx, layout.log_phi_gp_regional_idx);
          Rprintf("  MSGP: beta_local[%d-%d] beta_reg[%d-%d]\n",
                  layout.gp_local_start, layout.gp_local_end-1,
                  layout.gp_regional_start, layout.gp_regional_end-1);
        }
        if (layout.has_temporal)
          Rprintf("  temporal[%d-%d] log_tau[%d]\n",
                  layout.temporal_start, layout.temporal_end-1, layout.log_tau_temporal_idx);
      }
    }

    double tol = 1e-4;
    return Rcpp::List::create(
      Rcpp::Named("h_vs_n") = h_vs_n,
      Rcpp::Named("ar_vs_n") = ar_vs_n,
      Rcpp::Named("a_vs_n") = a_vs_n,
      Rcpp::Named("h_vs_ar") = h_vs_ar,
      Rcpp::Named("tol") = tol,
      // h_ok: check h_vs_n first; if it fails but h_vs_ar passes, H is still
      // correct (compute_log_post vs log_post_impl diverge for some ZI models)
      Rcpp::Named("h_ok") = (h_vs_n >= 0)
        ? ((h_vs_n < tol) ? true : (h_vs_ar >= 0 && h_vs_ar < tol))
        : NA_LOGICAL,
      Rcpp::Named("ar_ok") = (ar_vs_n >= 0) ? (ar_vs_n < tol) : NA_LOGICAL,
      Rcpp::Named("a_ok") = (a_vs_n >= 0) ? (a_vs_n < tol) : NA_LOGICAL,
      Rcpp::Named("n_params") = (int)q0.size()
    );
  }

  // Run sampler
  if (n_chains == 1) {
    ParamLayout layout = compute_param_layout(data);
    HMCResult result = run_hmc_chain(
      q0, data, layout, n_iter, n_warmup, L, 0, seed, verbose, max_treedepth,
      metric_type, adapt_delta, -1
    );

    return Rcpp::List::create(
      Rcpp::Named("samples") = result.samples,
      Rcpp::Named("log_prob") = result.log_prob,
      Rcpp::Named("accept_prob") = result.accept_prob,
      Rcpp::Named("n_leapfrog") = result.n_leapfrog,
      Rcpp::Named("treedepth") = result.treedepth,
      Rcpp::Named("divergent") = result.divergent,
      Rcpp::Named("epsilon") = result.epsilon,
      Rcpp::Named("n_warmup") = result.n_warmup,
      Rcpp::Named("n_sample") = result.n_sample,
      Rcpp::Named("n_chains") = 1,
      Rcpp::Named("sampler") = result.sampler.empty()
        ? ((L == 0) ? std::string("NUTS") : std::string("HMC"))
        : result.sampler
    );
  } else {
    // Multiple chains
    std::vector<HMCResult> results = run_hmc_parallel_chains(
      q0, data, n_iter, n_warmup, L, n_chains, seed, verbose, max_treedepth,
      metric_type, adapt_delta, -1
    );

    // Combine results
    int n_sample = results[0].n_sample;
    int n_params = results[0].samples.ncol();

    Rcpp::List samples_list(n_chains);
    Rcpp::List log_prob_list(n_chains);
    Rcpp::List accept_prob_list(n_chains);
    Rcpp::List n_leapfrog_list(n_chains);
    Rcpp::List treedepth_list(n_chains);
    Rcpp::List divergent_list(n_chains);
    Rcpp::NumericVector epsilon_vec(n_chains);

    std::string sampler_name = (L == 0) ? "NUTS" : "HMC";
    for (int c = 0; c < n_chains; c++) {
      samples_list[c] = results[c].samples;
      log_prob_list[c] = results[c].log_prob;
      accept_prob_list[c] = results[c].accept_prob;
      n_leapfrog_list[c] = results[c].n_leapfrog;
      treedepth_list[c] = results[c].treedepth;
      divergent_list[c] = results[c].divergent;
      epsilon_vec[c] = results[c].epsilon;
      if (!results[c].sampler.empty()) {
        sampler_name = results[c].sampler;
      }
    }

    return Rcpp::List::create(
      Rcpp::Named("samples") = samples_list,
      Rcpp::Named("log_prob") = log_prob_list,
      Rcpp::Named("accept_prob") = accept_prob_list,
      Rcpp::Named("n_leapfrog") = n_leapfrog_list,
      Rcpp::Named("treedepth") = treedepth_list,
      Rcpp::Named("divergent") = divergent_list,
      Rcpp::Named("epsilon") = epsilon_vec,
      Rcpp::Named("n_warmup") = n_warmup,
      Rcpp::Named("n_sample") = n_sample,
      Rcpp::Named("n_chains") = n_chains,
      Rcpp::Named("sampler") = sampler_name
    );
  }
}

// [[Rcpp::export]]
Rcpp::List cpp_hmc_fit_gp_v2(Rcpp::List args) {
  // O2-safe interface: single List parameter to minimize Rcpp template instantiation
  // at ABI boundary. All parameter extraction happens inside function body where
  // compiler has full visibility.

  // Extract all parameters from the list - matching cpp_hmc_fit_gp signature
  Rcpp::NumericVector q_init = Rcpp::as<Rcpp::NumericVector>(args["q_init"]);
  Rcpp::IntegerVector y_num = Rcpp::as<Rcpp::IntegerVector>(args["y_num"]);
  Rcpp::IntegerVector y_denom = Rcpp::as<Rcpp::IntegerVector>(args["y_denom"]);
  Rcpp::NumericVector y_num_cont = args.containsElementNamed("y_num_cont")
    ? Rcpp::as<Rcpp::NumericVector>(args["y_num_cont"])
    : Rcpp::NumericVector(y_num.size(), 0.0);
  Rcpp::NumericVector y_denom_cont = Rcpp::as<Rcpp::NumericVector>(args["y_denom_cont"]);
  Rcpp::NumericMatrix X_num = Rcpp::as<Rcpp::NumericMatrix>(args["X_num"]);
  Rcpp::NumericMatrix X_denom = Rcpp::as<Rcpp::NumericMatrix>(args["X_denom"]);
  Rcpp::IntegerVector re_group = Rcpp::as<Rcpp::IntegerVector>(args["re_group"]);
  int n_re_groups = Rcpp::as<int>(args["n_re_groups"]);
  std::string model_type_str = Rcpp::as<std::string>(args["model_type_str"]);
  Rcpp::List gp_params = Rcpp::as<Rcpp::List>(args["gp_params"]);
  Rcpp::List ms_gp_params = Rcpp::as<Rcpp::List>(args["ms_gp_params"]);
  Rcpp::List ms_temporal_params = Rcpp::as<Rcpp::List>(args["ms_temporal_params"]);
  Rcpp::List rsr_params = Rcpp::as<Rcpp::List>(args["rsr_params"]);
  Rcpp::List temporal_params = Rcpp::as<Rcpp::List>(args["temporal_params"]);
  double sigma_beta = Rcpp::as<double>(args["sigma_beta"]);
  double sigma_re_scale = Rcpp::as<double>(args["sigma_re_scale"]);
  double phi_prior_shape = Rcpp::as<double>(args["phi_prior_shape"]);
  double phi_prior_rate = Rcpp::as<double>(args["phi_prior_rate"]);
  std::string zi_type_str = Rcpp::as<std::string>(args["zi_type_str"]);
  Rcpp::NumericMatrix X_zi = Rcpp::as<Rcpp::NumericMatrix>(args["X_zi"]);
  double zi_prior_sd = Rcpp::as<double>(args["zi_prior_sd"]);
  int n_iter = Rcpp::as<int>(args["n_iter"]);
  int n_warmup = Rcpp::as<int>(args["n_warmup"]);
  int L = Rcpp::as<int>(args["L"]);
  int n_chains = Rcpp::as<int>(args["n_chains"]);
  unsigned int seed = Rcpp::as<unsigned int>(args["seed"]);
  int n_threads = Rcpp::as<int>(args["n_threads"]);
  bool verbose = Rcpp::as<bool>(args["verbose"]);
  int max_treedepth = 10;
  if (args.containsElementNamed("max_treedepth")) {
    max_treedepth = Rcpp::as<int>(args["max_treedepth"]);
  }
  double adapt_delta = -1.0;
  if (args.containsElementNamed("adapt_delta")) {
    adapt_delta = Rcpp::as<double>(args["adapt_delta"]);
  }
  std::string metric_str = "auto";
  if (args.containsElementNamed("metric_str")) {
    metric_str = Rcpp::as<std::string>(args["metric_str"]);
  }
  std::string gradient_mode_str = "auto";
  if (args.containsElementNamed("gradient_mode_str")) {
    gradient_mode_str = Rcpp::as<std::string>(args["gradient_mode_str"]);
  }

  // Extract TVC params if present
  Rcpp::List tvc_params_extracted;
  if (args.containsElementNamed("tvc_params")) {
    tvc_params_extracted = Rcpp::as<Rcpp::List>(args["tvc_params"]);
  }

  bool gradient_check_only = false;
  if (args.containsElementNamed("gradient_check_only")) {
    gradient_check_only = Rcpp::as<bool>(args["gradient_check_only"]);
  }

  // Delegate to the original implementation (metric/gradient parsed inside)
  return cpp_hmc_fit_gp(
    q_init, y_num, y_denom, y_num_cont, y_denom_cont,
    X_num, X_denom, re_group, n_re_groups,
    model_type_str, gp_params, ms_gp_params, ms_temporal_params, rsr_params,
    temporal_params,
    sigma_beta, sigma_re_scale, phi_prior_shape, phi_prior_rate,
    zi_type_str, X_zi, zi_prior_sd,
    n_iter, n_warmup, L, n_chains, seed, n_threads, verbose, max_treedepth, adapt_delta,
    metric_str, gradient_mode_str,
    tvc_params_extracted,
    gradient_check_only
  );
}
