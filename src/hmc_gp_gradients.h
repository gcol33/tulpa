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

// Analytical gradient of NNGP log-likelihood w.r.t. w (spatial effects)
// Eigen LLT + OpenMP parallelized. Uses cached nn_neighbor_dist.
// Returns gradients w.r.t. w only; sigma2/phi gradients computed elsewhere.
inline void gp_nngp_gradient_w_analytical(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data,
    std::vector<double>& grad_w  // Output: gradient (length n_obs)
) {
  int N = gp_data.n_obs;
  int nn = gp_data.nn;

  grad_w.assign(N, 0.0);

  // Validate input sizes
  if (gp_data.nn_order.size() < (size_t)N) return;
  if (gp_data.nn_idx.size() < (size_t)(N * nn)) return;
  if (gp_data.nn_dist.size() < (size_t)(N * nn)) return;
  if (w.size() < (size_t)N) return;
  if (gp_data.coords.size() < (size_t)(2 * N)) return;
  if (gp_data.nn_neighbor_dist.size() < (size_t)(N * nn * nn)) return;

  // First observation: marginal N(0, sigma2)
  int first_idx = gp_data.nn_order[0];
  if (first_idx < 0 || first_idx >= N) return;
  grad_w[first_idx] = -w[first_idx] / sigma2;

  // Thread-local workspace setup
  int n_threads = tulpa_omp_team_size(N - 1);

  std::vector<double> tl_grad_w(n_threads * N, 0.0);

  struct ThreadWS {
    Eigen::MatrixXd C_eigen;
    Eigen::VectorXd c_eigen, w_nb_eigen;
    Eigen::LLT<Eigen::MatrixXd> llt;
    std::vector<int> nb_idx;
    ThreadWS(int nn_) : C_eigen(nn_, nn_), c_eigen(nn_),
                        w_nb_eigen(nn_), llt(nn_), nb_idx(nn_) {}
  };
  std::vector<ThreadWS> ws_vec(n_threads, ThreadWS(nn));

  #ifdef _OPENMP
  #pragma omp parallel num_threads(n_threads)
  #endif
  {
    int tid = 0;
    #ifdef _OPENMP
    tid = omp_get_thread_num();
    #endif

    double* my_grad_w = &tl_grad_w[tid * N];
    auto& C_eigen = ws_vec[tid].C_eigen;
    auto& c_eigen = ws_vec[tid].c_eigen;
    auto& w_nb_eigen = ws_vec[tid].w_nb_eigen;
    auto& llt = ws_vec[tid].llt;
    auto& nb_idx = ws_vec[tid].nb_idx;

    #ifdef _OPENMP
    #pragma omp for schedule(dynamic)
    #endif
    for (int i = 1; i < N; i++) {
      int obs_idx = gp_data.nn_order[i];
      if (obs_idx < 0 || obs_idx >= N) continue;

      // Count actual neighbors
      int n_nb = 0;
      for (int j = 0; j < nn && gp_data.nn_idx[i * nn + j] > 0; j++) n_nb++;

      if (n_nb == 0) {
        my_grad_w[obs_idx] += -w[obs_idx] / sigma2;
        continue;
      }

      // Build c_vec
      for (int j = 0; j < n_nb; j++) {
        double d = gp_data.nn_dist[i * nn + j];
        c_eigen(j) = compute_cov(d, sigma2, phi, gp_data.cov_type);
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
        my_grad_w[obs_idx] += -w[obs_idx] / sigma2;
        continue;
      }

      // Build C_mat using cached distances (symmetric fill)
      for (int j1 = 0; j1 < n_nb; j1++) {
        C_eigen(j1, j1) = sigma2 + kGpJitter;
        for (int j2 = j1 + 1; j2 < n_nb; j2++) {
          double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
          double cov_val = compute_cov(d12, sigma2, phi, gp_data.cov_type);
          C_eigen(j1, j2) = cov_val;
          C_eigen(j2, j1) = cov_val;
        }
      }

      // Configurable solver: Cholesky (default) or CG/PCG (opt-in).
      Eigen::VectorXd alpha_vec(n_nb);
      if (!solve_neighbor_system(C_eigen, n_nb, c_eigen, alpha_vec, llt,
                                 gp_data.solver_config)) {
        my_grad_w[obs_idx] += -w[obs_idx] / sigma2;
        continue;
      }

      // Conditional mean and variance
      for (int j = 0; j < n_nb; j++) w_nb_eigen(j) = w[nb_idx[j]];
      double cond_mean = alpha_vec.head(n_nb).dot(w_nb_eigen.head(n_nb));
      double c_Cinv_c = c_eigen.head(n_nb).dot(alpha_vec.head(n_nb));
      double cond_var = std::max(sigma2 - c_Cinv_c, kGpVarFloor);
      double resid = w[obs_idx] - cond_mean;

      // Gradient w.r.t. w
      my_grad_w[obs_idx] += -resid / cond_var;
      double r_over_v = resid / cond_var;
      for (int j = 0; j < n_nb; j++) {
        my_grad_w[nb_idx[j]] += alpha_vec(j) * r_over_v;
      }
    }
  }

  // Reduce thread-local accumulators
  for (int t = 0; t < n_threads; t++) {
    const double* tg = &tl_grad_w[t * N];
    for (int k = 0; k < N; k++) grad_w[k] += tg[k];
  }
}

// Covariance derivative w.r.t. phi: dk(d)/dphi. Delegates to the canonical
// tulpa_svc::dcov_dphi_svc, which is the same math -- this copy had drifted and
// was returning half the true Gaussian derivative (it dropped the factor of 2
// in k*2*d^2/phi^3).
inline double dcov_dphi(double d, double phi, double cov_val, double sigma2,
                        tulpa_svc::CovType cov_type) {
  return tulpa_svc::dcov_dphi_svc(d, phi, cov_val, sigma2, cov_type);
}

// Fully analytical NNGP gradients — Eigen LLT + OpenMP parallelized
// Uses cached nn_neighbor_dist (no coord recomputation), symmetric C_mat fill
// Complexity: O(N * nn³) Cholesky-dominated, parallelized across observations
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

  // Thread-local workspace setup
  int n_threads = tulpa_omp_team_size(N - 1);

  // Per-thread accumulators: grad_w[tid * N + k], sigma2[tid], phi[tid]
  std::vector<double> tl_grad_w(n_threads * N, 0.0);
  std::vector<double> tl_sigma2(n_threads, 0.0);
  std::vector<double> tl_phi(n_threads, 0.0);

  // Per-thread Eigen workspaces (avoid per-iteration allocation)
  struct ThreadWS {
    Eigen::MatrixXd C_eigen;
    Eigen::VectorXd c_eigen, dc_eigen, w_nb_eigen;
    Eigen::LLT<Eigen::MatrixXd> llt;
    std::vector<int> nb_idx;
    ThreadWS(int nn_) : C_eigen(nn_, nn_), c_eigen(nn_), dc_eigen(nn_),
                        w_nb_eigen(nn_), llt(nn_), nb_idx(nn_) {}
  };
  std::vector<ThreadWS> ws_vec(n_threads, ThreadWS(nn));

  #ifdef _OPENMP
  #pragma omp parallel num_threads(n_threads)
  #endif
  {
    int tid = 0;
    #ifdef _OPENMP
    tid = omp_get_thread_num();
    #endif

    double* my_grad_w = &tl_grad_w[tid * N];
    auto& C_eigen = ws_vec[tid].C_eigen;
    auto& c_eigen = ws_vec[tid].c_eigen;
    auto& dc_eigen = ws_vec[tid].dc_eigen;
    auto& w_nb_eigen = ws_vec[tid].w_nb_eigen;
    auto& llt = ws_vec[tid].llt;
    auto& nb_idx = ws_vec[tid].nb_idx;

    #ifdef _OPENMP
    #pragma omp for schedule(dynamic)
    #endif
    for (int i = 1; i < N; i++) {
      int obs_idx = gp_data.nn_order[i];
      if (obs_idx < 0 || obs_idx >= N) continue;

      // Count neighbors
      int n_nb = 0;
      for (int j = 0; j < nn && gp_data.nn_idx[i * nn + j] > 0; j++) n_nb++;

      if (n_nb == 0) {
        double wi = w[obs_idx];
        my_grad_w[obs_idx] += -wi / sigma2;
        tl_sigma2[tid] += 0.5 * (wi * wi / sigma2 - 1.0);
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
        tl_sigma2[tid] += 0.5 * (wi * wi / sigma2 - 1.0);
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
        tl_sigma2[tid] += 0.5 * (wi * wi / sigma2 - 1.0);
        continue;
      }

      for (int j = 0; j < n_nb; j++) w_nb_eigen(j) = w[nb_idx[j]];
      Eigen::VectorXd beta_vec(n_nb);
      if (!solve_neighbor_system_second(C_eigen, n_nb, w_nb_eigen, beta_vec,
                                        llt, gp_data.solver_config)) {
        double wi = w[obs_idx];
        my_grad_w[obs_idx] += -wi / sigma2;
        tl_sigma2[tid] += 0.5 * (wi * wi / sigma2 - 1.0);
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
          sigma2, phi, kGpVarFloor);
      my_grad_w[obs_idx] += g.grad_w_obs;
      for (int j = 0; j < n_nb; j++) my_grad_w[nb_idx[j]] += alpha_vec(j) * g.r_over_v;
      tl_sigma2[tid] += g.dlog_sigma2;
      tl_phi[tid] += g.dlog_phi;
    }
  }

  // Reduce thread-local accumulators
  for (int t = 0; t < n_threads; t++) {
    const double* tg = &tl_grad_w[t * N];
    for (int k = 0; k < N; k++) grads.grad_w[k] += tg[k];
    grads.grad_log_sigma2 += tl_sigma2[t];
    grads.grad_log_phi += tl_phi[t];
  }
}

// Numerical gradient of NNGP log-likelihood w.r.t. w (spatial effects)
// For use in HMC updates
inline void gp_gradient_w(
    const std::vector<double>& w,
    double sigma2,
    double phi,
    const GPData& gp_data,
    std::vector<double>& grad_w,  // Output: gradient (length n_obs)
    double epsilon = 1e-6
) {
  int N = gp_data.n_obs;
  grad_w.resize(N);

  double base_ll = gp_nngp_log_lik(w, sigma2, phi, gp_data);

  // Finite difference for each w[i]
  std::vector<double> w_plus = w;
  for (int i = 0; i < N; i++) {
    w_plus[i] = w[i] + epsilon;
    double ll_plus = gp_nngp_log_lik(w_plus, sigma2, phi, gp_data);
    grad_w[i] = (ll_plus - base_ll) / epsilon;
    w_plus[i] = w[i];  // Reset
  }
}

