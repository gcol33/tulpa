// hmc_rcpp_fit_gp.cpp
// Rcpp-facing HMC fit entry points for GP-based spatial models.
// Split from hmc_rcpp_fit.cpp on 2026-05-02.

#include "hmc_sampler.h"
#include "hmc_gradient_check.h"
#include "hmc_modeldata_builders.h"
#include <Rcpp.h>
#include <atomic>


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
    Rcpp::Nullable<Rcpp::List> tvc_params = R_NilValue,
    bool gradient_check_only = false
) {
  using namespace tulpa_hmc;
  Rcpp::List tvc_params_list = tvc_params.isNotNull()
      ? Rcpp::List(tvc_params)
      : Rcpp::List::create();

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

  populate_legacy_ratio_data(
      data, y_num, y_denom, y_num_cont, y_denom_cont,
      X_num, X_denom, model_type_str);

  // Random effects (single-term legacy path)
  data.re_group = std::vector<int>(re_group.begin(), re_group.end());
  data.n_re_groups = n_re_groups;

  // Initialize multi-term RE fields to indicate single-term mode
  data.n_re_terms = 0;  // 0 means use legacy single-term path
  data.total_re_groups = n_re_groups;
  data.has_re_slopes = false;  // GP interface doesn't support random slopes
  data.has_re_correlated_slopes = false;

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

    data.temporal_type = parse_hmc_temporal_type(temporal_type_str);
    if (data.temporal_type == TemporalType::GP) {

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
  populate_zi_data(data, zi_type_str, X_zi, zi_prior_sd, X_zi.ncol());

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
    bool has_tvc = tvc_params_list.size() > 0 &&
                   tvc_params_list.containsElementNamed("has_tvc") &&
                   Rcpp::as<bool>(tvc_params_list["has_tvc"]);
    data.has_tvc = has_tvc;
    if (has_tvc) {
      int tvc_n_tvc = Rcpp::as<int>(tvc_params_list["n_tvc"]);
      int tvc_n_times = Rcpp::as<int>(tvc_params_list["n_times"]);
      int tvc_n_groups = Rcpp::as<int>(tvc_params_list["n_groups"]);
      std::string tvc_structure_str = Rcpp::as<std::string>(tvc_params_list["structure"]);
      bool tvc_shared = Rcpp::as<bool>(tvc_params_list["shared"]);
      bool tvc_cyclic = Rcpp::as<bool>(tvc_params_list["cyclic"]);
      std::vector<int> tvc_indices = Rcpp::as<std::vector<int>>(tvc_params_list["tvc_indices"]);
      std::vector<int> tvc_time_index = Rcpp::as<std::vector<int>>(tvc_params_list["time_index"]);
      std::vector<int> tvc_group_index = Rcpp::as<std::vector<int>>(tvc_params_list["group_index"]);
      std::vector<double> tvc_X_tvc = Rcpp::as<std::vector<double>>(tvc_params_list["X_tvc"]);
      double tvc_tau_shape = Rcpp::as<double>(tvc_params_list["tau_shape"]);
      double tvc_tau_rate = Rcpp::as<double>(tvc_params_list["tau_rate"]);

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

      data.tvc_data.structure = parse_tvc_temporal_type(tvc_structure_str);

      data.tvc_tau_shape = tvc_tau_shape;
      data.tvc_tau_rate = tvc_tau_rate;
      data.tvc_data.init_workspace();
    } else {
      clear_tvc_data(data);
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
    return run_gradient_check_only(q0, data);
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
