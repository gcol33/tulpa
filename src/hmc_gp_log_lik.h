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
  if (gp_data.nn_dist.size() < (size_t)(N * nn)) return -INFINITY;
  // nn_neighbor_dist is indexed at i * nn * nn + j1 * nn + j2 below; a short
  // one is an out-of-bounds read, not a wrong number.
  if (gp_data.nn_neighbor_dist.size() < (size_t)(N * nn * nn)) return -INFINITY;
  if (w.size() < (size_t)N) return -INFINITY;
  if (gp_data.coords.size() < (size_t)(2 * N)) return -INFINITY;

  double log_lik = 0.0;

  // First observation: marginal N(0, sigma2)
  int first_idx = gp_data.nn_order[0];

  log_lik += tulpa_nngp::marginal_log_density(w[first_idx], sigma2);

  // Pre-allocate Eigen matrices/vectors for Cholesky/CG solve
  // Using Eigen avoids hand-rolled linear algebra bugs and leverages SIMD
  Eigen::VectorXd c_vec(nn);
  Eigen::MatrixXd C_mat(nn, nn);
  Eigen::VectorXd alpha(nn);
  Eigen::LLT<Eigen::MatrixXd> llt(nn);

  // Remaining observations: conditional on neighbors
  for (int i = 1; i < N; i++) {

    int obs_idx = gp_data.nn_order[i];

    // Bounds check (always on)
    if (obs_idx < 0 || obs_idx >= N) return -INFINITY;

    // Count actual neighbors (early observations have fewer). The shared
    // left-packed scan stops at the first entry outside [1, nn_order.size()],
    // so every column below the count resolves inside nn_order.
    const int n_neighbors = tulpa_nngp::nngp_row_neighbours(
        gp_data.nn_idx.data() + (std::size_t)i * nn, /*stride=*/1, nn,
        (int)gp_data.nn_order.size());

    if (n_neighbors == 0) {
      // No neighbors: marginal
      log_lik += tulpa_nngp::marginal_log_density(w[obs_idx], sigma2);
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

      int nn_idx1 = gp_data.nn_order[raw_nn_idx1 - 1];

      // Bounds check for coords access
      if (nn_idx1 < 0 || nn_idx1 * 2 + 1 >= (int)gp_data.coords.size()) return -INFINITY;

      for (int j2 = 0; j2 < n_neighbors; j2++) {
        int raw_nn_idx2 = gp_data.nn_idx[i * nn + j2];

        // The neighbour distance below is read from the cache rather than
        // recomputed, so only the index itself has to resolve.
        if (raw_nn_idx2 - 1 < 0 || raw_nn_idx2 - 1 >= (int)gp_data.nn_order.size()) return -INFINITY;

        if (j1 == j2) {
          C_mat(j1, j2) = sigma2;
        } else {
          // Use cached pairwise neighbor distances
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

      int nn_orig_idx = gp_data.nn_order[raw_nn_idx - 1];

      // Bounds check for w access
      if (nn_orig_idx < 0 || nn_orig_idx >= (int)w.size()) return -INFINITY;

      cond_mean += alpha(j) * w[nn_orig_idx];
    }

    double c_Cinv_c = 0.0;
    for (int j = 0; j < n_neighbors; j++) {
      c_Cinv_c += c_vec(j) * alpha(j);
    }
    // Floor and density through the shared kernel, at the constants the
    // autodiff twin hands cond_moments. The two are the same function, and the
    // analytic gradients are finite-differenced from this copy.
    double cond_var = tulpa_nngp::apply_var_floor(
        sigma2 - c_Cinv_c, kGpVarFloor, tulpa_nngp::VarFloor::Clamp);
    log_lik += tulpa_nngp::cond_log_density(w[obs_idx], cond_mean, cond_var);
  }

  return log_lik;
}

// The GP hyperparameter priors live in pc_prior.h (the PC densities, shared
// with the SPDE field) and autodiff_utils.h (the bounded-phi map), so the
// sampled coordinate and its density are defined in one place per quantity.
