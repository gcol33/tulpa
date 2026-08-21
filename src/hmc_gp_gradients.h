// -----------------------------------------------------------------------------
// Gradient computation for GP parameters (for HMC)
// -----------------------------------------------------------------------------

#include "omp_threads.h"
#include "nngp_cond.h"   // shared Vecchia conditional-gradient assembly

// Struct to hold NNGP gradient results (for hand-coded gradients)
struct NNGPGradients {
  std::vector<double> grad_w;         // Gradient w.r.t. spatial effects
  double grad_log_sigma2;             // Gradient w.r.t. log(sigma2)
  double grad_log_phi;                // Gradient w.r.t. log(phi)
};

// Covariance derivative w.r.t. phi: dk(d)/dphi. Delegates to
// tulpa_svc::dcov_dphi_svc, the single source of truth for every NNGP
// gradient path.
inline double dcov_dphi(double d, double phi, double cov_val, double sigma2,
                        tulpa_svc::CovType cov_type) {
  return tulpa_svc::dcov_dphi_svc(d, phi, cov_val, sigma2, cov_type);
}

// Fully analytical NNGP gradients: Eigen LLT + OpenMP parallelized. Reads the
// cached nn_neighbor_dist (no coordinate recomputation) and fills C_mat
// symmetrically. Complexity O(N * nn^3), Cholesky-dominated, parallelized
// across observations.
//
// Reference implementation: the only caller is the cpp_test_gp_solver_dispatch
// export in test_helpers.cpp, which test-gp-cg.R drives to score the GP solver
// backends against each other.
inline void gp_nngp_gradients(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data,
    NNGPGradients& grads,
    double /* epsilon */ = 1e-6
) {
  int N = gp_data.n_obs;
  int nn = gp_data.nn;

  grads.grad_w.assign(N, 0.0);
  grads.grad_log_sigma2 = 0.0;
  grads.grad_log_phi = 0.0;

  // Validate
  if ((int)gp_data.nn_order.size() < N || (int)gp_data.nn_idx.size() < N * nn ||
      (int)gp_data.nn_dist.size() < N * nn || (int)w.size() < N ||
      (int)gp_data.coords.size() < 2 * N ||
      (int)gp_data.nn_neighbor_dist.size() < N * nn * nn) return;

  // First observation: marginal N(0, sigma2)
  int first_idx = gp_data.nn_order[0];
  if (first_idx < 0 || first_idx >= N) return;
  double w0 = w[first_idx];
  grads.grad_w[first_idx] = -w0 / sigma2;
  grads.grad_log_sigma2 += 0.5 * (w0 * w0 / sigma2 - 1.0);

  // Team size, and with it the number of chunks the rows are cut into
  int n_threads = tulpa_omp_team_size(N - 1);

  // Per-chunk accumulators: grad_w[t * N + k], sigma2[t], phi[t]
  std::vector<double> tl_grad_w(n_threads * N, 0.0);
  std::vector<double> tl_sigma2(n_threads, 0.0);
  std::vector<double> tl_phi(n_threads, 0.0);

  // Per-chunk Eigen workspaces (avoid per-iteration allocation)
  struct ThreadWS {
    Eigen::MatrixXd C_eigen;
    Eigen::VectorXd c_eigen, dc_eigen, w_nb_eigen;
    Eigen::LLT<Eigen::MatrixXd> llt;
    std::vector<int> nb_idx;
    ThreadWS(int nn_) : C_eigen(nn_, nn_), c_eigen(nn_), dc_eigen(nn_),
                        w_nb_eigen(nn_), llt(nn_), nb_idx(nn_) {}
  };
  std::vector<ThreadWS> ws_vec(n_threads, ThreadWS(nn));

  // The rows are cut into `n_threads` contiguous chunks HERE, by index
  // arithmetic, and each chunk accumulates left to right into its own slot, so
  // the reduction order is a function of (n_threads, N) alone. Letting the
  // runtime hand rows out -- a schedule(dynamic) loop writing into the slot
  // omp_get_thread_num() names -- makes the order a property of the run, and
  // floating-point addition is not associative, so the gradient's last bits move
  // between two runs on identical input. Same policy as tulpa_parallel_sum in
  // omp_threads.h.
  const long long n_rows = static_cast<long long>(N - 1);
  const long long teams  = static_cast<long long>(n_threads);

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(n_threads)
  #endif
  for (int t = 0; t < n_threads; t++) {
    const int lo = 1 + static_cast<int>(n_rows * t / teams);
    const int hi = 1 + static_cast<int>(n_rows * (t + 1) / teams);

    double* my_grad_w = &tl_grad_w[(std::size_t)t * N];
    auto& C_eigen = ws_vec[t].C_eigen;
    auto& c_eigen = ws_vec[t].c_eigen;
    auto& dc_eigen = ws_vec[t].dc_eigen;
    auto& w_nb_eigen = ws_vec[t].w_nb_eigen;
    auto& llt = ws_vec[t].llt;
    auto& nb_idx = ws_vec[t].nb_idx;

    for (int i = lo; i < hi; i++) {
      int obs_idx = gp_data.nn_order[i];
      if (obs_idx < 0 || obs_idx >= N) continue;

      // Count neighbors, through the shared left-packed scan the density
      // kernels also run -- the density and its analytic gradient have to
      // condition on the SAME neighbour set.
      const int n_nb = tulpa_nngp::nngp_row_neighbours(
          gp_data.nn_idx.data() + (std::size_t)i * nn, /*stride=*/1, nn,
          (int)gp_data.nn_order.size());

      if (n_nb == 0) {
        double wi = w[obs_idx];
        my_grad_w[obs_idx] += -wi / sigma2;
        tl_sigma2[t] += 0.5 * (wi * wi / sigma2 - 1.0);
        continue;
      }

      // Build c_vec, dc_vec (covariances and phi derivatives)
      for (int j = 0; j < n_nb; j++) {
        double d = gp_data.nn_dist[i * nn + j];
        c_eigen(j) = compute_cov(d, sigma2, phi, gp_data.cov_type);
        dc_eigen(j) = dcov_dphi(d, phi, c_eigen(j), sigma2, gp_data.cov_type);
      }

      // Validate neighbor indices
      bool ok = true;
      for (int j = 0; j < n_nb && ok; j++) {
        int raw = gp_data.nn_idx[i * nn + j];
        if (raw - 1 < 0 || raw - 1 >= (int)gp_data.nn_order.size()) { ok = false; break; }
        int idx = gp_data.nn_order[raw - 1];
        if (idx < 0 || idx >= N) { ok = false; break; }
        nb_idx[j] = idx;
      }
      if (!ok) {
        double wi = w[obs_idx];
        my_grad_w[obs_idx] += -wi / sigma2;
        tl_sigma2[t] += 0.5 * (wi * wi / sigma2 - 1.0);
        continue;
      }

      // Build C_mat using cached nn_neighbor_dist (symmetric fill, upper triangle only)
      for (int j1 = 0; j1 < n_nb; j1++) {
        C_eigen(j1, j1) = sigma2 + kGpJitter;  // Diagonal + jitter
        for (int j2 = j1 + 1; j2 < n_nb; j2++) {
          double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
          double cov_val = compute_cov(d12, sigma2, phi, gp_data.cov_type);
          C_eigen(j1, j2) = cov_val;
          C_eigen(j2, j1) = cov_val;
        }
      }

      // Configurable solver: factorize once (Cholesky) or run CG twice
      // (alpha = C^{-1}c, beta = C^{-1}w_nb).
      Eigen::VectorXd alpha_vec(n_nb);
      if (!solve_neighbor_system(C_eigen, n_nb, c_eigen, alpha_vec, llt,
                                 gp_data.solver_config)) {
        double wi = w[obs_idx];
        my_grad_w[obs_idx] += -wi / sigma2;
        tl_sigma2[t] += 0.5 * (wi * wi / sigma2 - 1.0);
        continue;
      }

      for (int j = 0; j < n_nb; j++) w_nb_eigen(j) = w[nb_idx[j]];
      Eigen::VectorXd beta_vec(n_nb);
      if (!solve_neighbor_system_second(C_eigen, n_nb, w_nb_eigen, beta_vec,
                                        llt, gp_data.solver_config)) {
        double wi = w[obs_idx];
        my_grad_w[obs_idx] += -wi / sigma2;
        tl_sigma2[t] += 0.5 * (wi * wi / sigma2 - 1.0);
        continue;
      }

      // Pairwise dC/dphi (row-major, zero diagonal) from the cached
      // neighbour-neighbour distances, for the shared gradient assembler.
      std::vector<double> dC(static_cast<std::size_t>(n_nb) * n_nb, 0.0);
      for (int j1 = 0; j1 < n_nb; j1++) {
        for (int j2 = j1 + 1; j2 < n_nb; j2++) {
          double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
          double dC_jk = dcov_dphi(d12, phi, C_eigen(j1, j2), sigma2,
                                   gp_data.cov_type);
          dC[j1 * n_nb + j2] = dC_jk;
          dC[j2 * n_nb + j1] = dC_jk;
        }
      }
      tulpa_nngp::VecchiaGrad g = tulpa_nngp::vecchia_cond_grad(
          n_nb, alpha_vec.data(), beta_vec.data(), c_eigen.data(),
          dc_eigen.data(), dC.data(), w_nb_eigen.data(), w[obs_idx],
          sigma2, phi, kGpVarFloor, tulpa_nngp::VarFloor::Clamp);
      my_grad_w[obs_idx] += g.grad_w_obs;
      for (int j = 0; j < n_nb; j++) my_grad_w[nb_idx[j]] += alpha_vec(j) * g.r_over_v;
      tl_sigma2[t] += g.dlog_sigma2;
      tl_phi[t] += g.dlog_phi;
    }
  }

  // Reduce the per-chunk accumulators, in chunk order
  for (int t = 0; t < n_threads; t++) {
    const double* tg = &tl_grad_w[t * N];
    for (int k = 0; k < N; k++) grads.grad_w[k] += tg[k];
    grads.grad_log_sigma2 += tl_sigma2[t];
    grads.grad_log_phi += tl_phi[t];
  }
}
