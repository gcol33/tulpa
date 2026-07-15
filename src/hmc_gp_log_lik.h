// -----------------------------------------------------------------------------
// Single-scale GP NNGP likelihood
// -----------------------------------------------------------------------------

// Compute NNGP log-likelihood for single spatial field
// w: spatial effect values at each location (length n_obs)
// sigma2: spatial variance
// phi: spatial range parameter
inline double gp_nngp_log_lik(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data
) {
  int N = gp_data.n_obs;
  int nn = gp_data.nn;

  // Bounds validation (always on - prevents UB from invalid data structures)
  if (gp_data.nn_order.size() < (size_t)N) return -INFINITY;
  if (gp_data.nn_idx.size() < (size_t)(N * nn)) return -INFINITY;
  if (gp_data.nn_dist.size() < (size_t)(N * nn)) return -INFINITY;  // Added: was missing
  if (gp_data.nn_neighbor_dist.size() < (size_t)(N * nn * nn)) return -INFINITY;  // Critical: prevents segfault
  if (w.size() < (size_t)N) return -INFINITY;
  if (gp_data.coords.size() < (size_t)(2 * N)) return -INFINITY;

#if GP_DEBUG_BOUNDS
  Rcpp::Rcout << "[GP_DEBUG] gp_nngp_log_lik called: N=" << N << ", nn=" << nn << "\n";
  Rcpp::Rcout << "[GP_DEBUG] w.size()=" << w.size() << "\n";
  Rcpp::Rcout << "[GP_DEBUG] nn_order.size()=" << gp_data.nn_order.size() << "\n";
  Rcpp::Rcout << "[GP_DEBUG] nn_idx.size()=" << gp_data.nn_idx.size() << "\n";
  Rcpp::Rcout << "[GP_DEBUG] nn_dist.size()=" << gp_data.nn_dist.size() << "\n";
  Rcpp::Rcout << "[GP_DEBUG] coords.size()=" << gp_data.coords.size() << "\n";

  // Validate sizes
  if (gp_data.nn_order.size() < (size_t)N) {
    Rcpp::Rcout << "[GP_DEBUG] ERROR: nn_order too small! size=" << gp_data.nn_order.size() << " < N=" << N << "\n";
    return -INFINITY;
  }
  if (gp_data.nn_idx.size() < (size_t)(N * nn)) {
    Rcpp::Rcout << "[GP_DEBUG] ERROR: nn_idx too small! size=" << gp_data.nn_idx.size() << " < N*nn=" << (N * nn) << "\n";
    return -INFINITY;
  }
  if (w.size() < (size_t)N) {
    Rcpp::Rcout << "[GP_DEBUG] ERROR: w too small! size=" << w.size() << " < N=" << N << "\n";
    return -INFINITY;
  }
#endif

  double log_lik = 0.0;

  // First observation: marginal N(0, sigma2)
#if GP_DEBUG_BOUNDS
  Rcpp::Rcout << "[GP_DEBUG] Accessing nn_order[0]...\n";
#endif
  int first_idx = gp_data.nn_order[0];

#if GP_DEBUG_BOUNDS
  Rcpp::Rcout << "[GP_DEBUG] first_idx=" << first_idx << " (should be 0 to " << (N-1) << ")\n";
  if (first_idx < 0 || first_idx >= N) {
    Rcpp::Rcout << "[GP_DEBUG] ERROR: first_idx out of bounds!\n";
    return -INFINITY;
  }
#endif

  log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
             0.5 * w[first_idx] * w[first_idx] / sigma2;

#if GP_DEBUG_BOUNDS
  Rcpp::Rcout << "[GP_DEBUG] First obs log_lik done, now processing remaining " << (N-1) << " observations\n";
#endif

  // Pre-allocate Eigen matrices/vectors for Cholesky/CG solve
  // Using Eigen avoids hand-rolled linear algebra bugs and leverages SIMD
  Eigen::VectorXd c_vec(nn);
  Eigen::MatrixXd C_mat(nn, nn);
  Eigen::VectorXd alpha(nn);
  Eigen::LLT<Eigen::MatrixXd> llt(nn);

  // Remaining observations: conditional on neighbors
  for (int i = 1; i < N; i++) {
#if GP_DEBUG_BOUNDS
    if (i < 5 || i == N-1) {
      Rcpp::Rcout << "[GP_DEBUG] Processing obs i=" << i << "\n";
    }
#endif

    int obs_idx = gp_data.nn_order[i];

    // Bounds check (always on)
    if (obs_idx < 0 || obs_idx >= N) return -INFINITY;

#if GP_DEBUG_BOUNDS
    if (obs_idx < 0 || obs_idx >= N) {
      Rcpp::Rcout << "[GP_DEBUG] ERROR: obs_idx=" << obs_idx << " out of bounds at i=" << i << "\n";
      return -INFINITY;
    }
#endif

    // Count actual neighbors (early observations have fewer)
    int n_neighbors = 0;
    for (int j = 0; j < nn; j++) {
      int nn_flat_idx = i * nn + j;
      // Bounds check (always on)
      if (nn_flat_idx < 0 || nn_flat_idx >= (int)gp_data.nn_idx.size()) return -INFINITY;
#if GP_DEBUG_BOUNDS
      if (nn_flat_idx < 0 || nn_flat_idx >= (int)gp_data.nn_idx.size()) {
        Rcpp::Rcout << "[GP_DEBUG] ERROR: nn_flat_idx=" << nn_flat_idx << " out of bounds (nn_idx.size=" << gp_data.nn_idx.size() << ")\n";
        return -INFINITY;
      }
#endif
      if (gp_data.nn_idx[nn_flat_idx] > 0) {
        n_neighbors++;
      }
    }

#if GP_DEBUG_BOUNDS
    if (i < 5) {
      Rcpp::Rcout << "[GP_DEBUG]   n_neighbors=" << n_neighbors << "\n";
    }
#endif

    if (n_neighbors == 0) {
      // No neighbors: marginal
      log_lik += -0.5 * std::log(2.0 * M_PI * sigma2) -
                 0.5 * w[obs_idx] * w[obs_idx] / sigma2;
      continue;
    }

    // c_vec: covariances between obs i and its neighbors
    for (int j = 0; j < n_neighbors; j++) {
      int nn_flat_idx = i * nn + j;
      double d = gp_data.nn_dist[nn_flat_idx];
      c_vec(j) = compute_cov(d, sigma2, phi, gp_data.cov_type);
    }

    // C_mat: covariances among neighbors
    for (int j1 = 0; j1 < n_neighbors; j1++) {
      int raw_nn_idx1 = gp_data.nn_idx[i * nn + j1];

      // Bounds check: nn_idx is 1-based from R, so subtract 1
      if (raw_nn_idx1 - 1 < 0 || raw_nn_idx1 - 1 >= (int)gp_data.nn_order.size()) return -INFINITY;

#if GP_DEBUG_BOUNDS
      if (i < 3 && j1 < 3) {
        Rcpp::Rcout << "[GP_DEBUG]   j1=" << j1 << " raw_nn_idx1=" << raw_nn_idx1 << "\n";
      }
      // nn_idx is 1-based from R, so subtract 1
      if (raw_nn_idx1 - 1 < 0 || raw_nn_idx1 - 1 >= (int)gp_data.nn_order.size()) {
        Rcpp::Rcout << "[GP_DEBUG] ERROR: raw_nn_idx1-1=" << (raw_nn_idx1 - 1) << " out of bounds for nn_order (size=" << gp_data.nn_order.size() << ")\n";
        return -INFINITY;
      }
#endif

      int nn_idx1 = gp_data.nn_order[raw_nn_idx1 - 1];

      // Bounds check for coords access
      if (nn_idx1 < 0 || nn_idx1 * 2 + 1 >= (int)gp_data.coords.size()) return -INFINITY;

#if GP_DEBUG_BOUNDS
      if (nn_idx1 < 0 || nn_idx1 * 2 + 1 >= (int)gp_data.coords.size()) {
        Rcpp::Rcout << "[GP_DEBUG] ERROR: nn_idx1=" << nn_idx1 << " leads to coords out of bounds (coords.size=" << gp_data.coords.size() << ")\n";
        return -INFINITY;
      }
#endif

      for (int j2 = 0; j2 < n_neighbors; j2++) {
        int raw_nn_idx2 = gp_data.nn_idx[i * nn + j2];

        // Bounds check
        if (raw_nn_idx2 - 1 < 0 || raw_nn_idx2 - 1 >= (int)gp_data.nn_order.size()) return -INFINITY;

#if GP_DEBUG_BOUNDS
        if (raw_nn_idx2 - 1 < 0 || raw_nn_idx2 - 1 >= (int)gp_data.nn_order.size()) {
          Rcpp::Rcout << "[GP_DEBUG] ERROR: raw_nn_idx2-1=" << (raw_nn_idx2 - 1) << " out of bounds for nn_order\n";
          return -INFINITY;
        }
#endif

        int nn_idx2 = gp_data.nn_order[raw_nn_idx2 - 1];

        if (j1 == j2) {
          C_mat(j1, j2) = sigma2;
        } else {
          // Phase 1.3: Use cached pairwise neighbor distances
          double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
          C_mat(j1, j2) = compute_cov(d12, sigma2, phi, gp_data.cov_type);
        }
      }
    }

    // Solve C_mat * alpha = c_vec via the configured solver (Cholesky default,
    // CG/PCG opt-in via spatial_gp(solver = "cg"|"pcg")).
    // Add small jitter to diagonal for numerical stability — prevents
    // ill-conditioning when phi is very small or sigma2 is near zero.
    for (int j = 0; j < n_neighbors; j++) {
      C_mat(j, j) += kGpJitter;
    }

    if (!solve_neighbor_system(C_mat, n_neighbors, c_vec, alpha, llt,
                               gp_data.solver_config)) {
      // Solver failed (non-PSD or CG non-convergence) — reject step.
      return -INFINITY;
    }

    // Conditional mean and variance
    double cond_mean = 0.0;
    for (int j = 0; j < n_neighbors; j++) {
      int raw_nn_idx = gp_data.nn_idx[i * nn + j];

      // Bounds check
      if (raw_nn_idx - 1 < 0 || raw_nn_idx - 1 >= (int)gp_data.nn_order.size()) return -INFINITY;

#if GP_DEBUG_BOUNDS
      if (raw_nn_idx - 1 < 0 || raw_nn_idx - 1 >= (int)gp_data.nn_order.size()) {
        Rcpp::Rcout << "[GP_DEBUG] ERROR: cond_mean raw_nn_idx-1=" << (raw_nn_idx - 1) << " out of bounds\n";
        return -INFINITY;
      }
#endif

      int nn_orig_idx = gp_data.nn_order[raw_nn_idx - 1];

      // Bounds check for w access
      if (nn_orig_idx < 0 || nn_orig_idx >= (int)w.size()) return -INFINITY;

#if GP_DEBUG_BOUNDS
      if (nn_orig_idx < 0 || nn_orig_idx >= (int)w.size()) {
        Rcpp::Rcout << "[GP_DEBUG] ERROR: nn_orig_idx=" << nn_orig_idx << " out of bounds for w (size=" << w.size() << ")\n";
        return -INFINITY;
      }
#endif

      cond_mean += alpha(j) * w[nn_orig_idx];
    }

    double c_Cinv_c = 0.0;
    for (int j = 0; j < n_neighbors; j++) {
      c_Cinv_c += c_vec(j) * alpha(j);
    }
    double cond_var = std::max(kGpVarFloor, sigma2 - c_Cinv_c);

    // Log-likelihood contribution
    double resid = w[obs_idx] - cond_mean;
    log_lik += -0.5 * std::log(2.0 * M_PI * cond_var) -
               0.5 * resid * resid / cond_var;
  }

#if GP_DEBUG_BOUNDS
  Rcpp::Rcout << "[GP_DEBUG] gp_nngp_log_lik completed, log_lik=" << log_lik << "\n";
#endif

  return log_lik;
}

// -----------------------------------------------------------------------------
// Multi-scale GP likelihood
// -----------------------------------------------------------------------------

// Compute log-likelihood for multi-scale GP (local + regional)
// w_local: local-scale spatial effect (length n_obs)
// w_regional: regional-scale spatial effect (length n_obs)
// Each component evaluated independently with its own range constraint
inline double multiscale_gp_log_lik(
    const std::vector<double>& w_local,
    const std::vector<double>& w_regional,
    double sigma2_local,
    double phi_local,
    double sigma2_regional,
    double phi_regional,
    const MultiscaleGPData& ms_data
) {
  // Create temporary GPData structures for each scale
  GPData gp_local;
  gp_local.n_obs = ms_data.n_obs;
  gp_local.nn = ms_data.nn_local;
  gp_local.coords = ms_data.coords;
  gp_local.nn_idx = ms_data.nn_idx_local;
  gp_local.nn_dist = ms_data.nn_dist_local;
  gp_local.nn_neighbor_dist = ms_data.nn_neighbor_dist_local;
  gp_local.nn_order = ms_data.nn_order_local;
  gp_local.nn_order_inv = ms_data.nn_order_inv_local;
  gp_local.cov_type = ms_data.cov_type;

  GPData gp_regional;
  gp_regional.n_obs = ms_data.n_obs;
  gp_regional.nn = ms_data.nn_regional;
  gp_regional.coords = ms_data.coords;
  gp_regional.nn_idx = ms_data.nn_idx_regional;
  gp_regional.nn_dist = ms_data.nn_dist_regional;
  gp_regional.nn_neighbor_dist = ms_data.nn_neighbor_dist_regional;
  gp_regional.nn_order = ms_data.nn_order_regional;
  gp_regional.nn_order_inv = ms_data.nn_order_inv_regional;
  gp_regional.cov_type = ms_data.cov_type;

  // Compute log-likelihood for each scale
  double ll_local = gp_nngp_log_lik(w_local, sigma2_local, phi_local, gp_local);
  double ll_regional = gp_nngp_log_lik(w_regional, sigma2_regional, phi_regional, gp_regional);

  return ll_local + ll_regional;
}

// -----------------------------------------------------------------------------
// Priors for GP hyperparameters
// -----------------------------------------------------------------------------

// Log prior for range parameter (uniform on log scale within bounds)
inline double log_prior_phi_uniform(double phi, double lower, double upper) {
  if (phi < lower || phi > upper) return -INFINITY;
  // Uniform on [lower, upper]
  return -std::log(upper - lower);
}

// Log prior for range with PC-style (favor larger ranges = simpler models)
inline double log_prior_phi_pc(double phi, double U, double alpha) {
  // P(phi < U) = alpha with 1/phi ~ Exponential(rate): P(phi < U) =
  // exp(-rate/U) = alpha, so rate = -log(alpha)/U (using log(1-alpha) here
  // implements the opposite tail P(phi > U) = alpha).
  if (phi <= 0) return -INFINITY;
  double rate = -std::log(alpha) / U;
  // Prior favors larger phi (simpler, smoother spatial structure)
  return std::log(rate) - rate / phi - 2.0 * std::log(phi);
}
