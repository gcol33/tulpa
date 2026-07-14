// =============================================================================
// Non-centered NNGP parameterization
// =============================================================================

#include "omp_threads.h"
// Instead of sampling w ~ NNGP(0, sigma2, phi) directly (centered),
// sample z ~ N(0, I) and transform z -> w via the NNGP autoregressive structure:
//   w[order[0]] = sqrt(sigma2) * z[0]
//   w[order[i]] = sum_j B[i,j] * w[nb_j(i)] + sqrt(d_i) * z[i]
// where B[i,:] = C_nb^{-1} c_i (regression coefficients) and
//       d_i = sigma2 - c_i' C_nb^{-1} c_i (conditional variance).
//
// This improves posterior geometry for large N, reducing NUTS treedepth.
// The prior on z is N(0,I), and no Jacobian is needed since we sample in z-space.

struct NNGPNCWorkspace {
    int N = 0, nn = 0;
    std::vector<double> w;          // Transformed spatial effects (N)
    std::vector<double> sqrt_d;     // sqrt(conditional variance) per obs (N)
    std::vector<double> B_flat;     // Regression coefficients (N * nn)
    std::vector<int> B_n_nb;        // Number of actual neighbors per obs (N)
    std::vector<int> nb_idx_flat;   // Neighbor indices per obs (N * nn), 0-based in w
    std::vector<double> adj;        // Adjoint accumulator (N)
    std::vector<double> L_flat;     // Cached Cholesky factors (N * nn * nn) for backward phi grad

    void init(int N_, int nn_) {
        if (N == N_ && nn == nn_) return;
        N = N_; nn = nn_;
        // size_t products: N * nn * nn as int overflows (UB) before the
        // buffers themselves become infeasible.
        const std::size_t Ns  = static_cast<std::size_t>(N);
        const std::size_t nns = static_cast<std::size_t>(nn);
        w.resize(Ns);
        sqrt_d.resize(Ns);
        B_flat.assign(Ns * nns, 0.0);
        B_n_nb.resize(Ns, 0);
        nb_idx_flat.assign(Ns * nns, -1);
        adj.resize(Ns, 0.0);
        L_flat.assign(Ns * nns * nns, 0.0);
    }
};

// Forward pass: z -> w via NNGP autoregressive structure
// z and w are both indexed by LOCATION (0-based), matching the parameter layout.
// Caches B, sqrt_d, nb_idx for backward pass.
// O(N * nn^3) due to per-observation Cholesky. Sequential (causal dependency).
// Uses Eigen LLT for vectorized Cholesky (~2x vs hand-rolled).
inline void nngp_nc_forward(
    const double* z,          // z[loc_idx], indexed by location, length N
    double sigma2, double phi,
    const GPData& gp_data,
    NNGPNCWorkspace& ws
) {
    int N = gp_data.n_obs;
    int nn = gp_data.nn;
    ws.init(N, nn);

    // Pre-allocated Eigen workspace (reused across iterations)
    Eigen::MatrixXd C_eigen(nn, nn);
    Eigen::VectorXd c_eigen(nn);
    Eigen::LLT<Eigen::MatrixXd> llt(nn);

    // First observation: marginal N(0, sigma2)
    int first_loc = gp_data.nn_order[0];
    ws.sqrt_d[0] = std::sqrt(sigma2);
    ws.w[first_loc] = ws.sqrt_d[0] * z[first_loc];
    ws.B_n_nb[0] = 0;

    for (int i = 1; i < N; i++) {
        int obs_loc = gp_data.nn_order[i];
        if (obs_loc < 0 || obs_loc >= N) {
            ws.sqrt_d[i] = std::sqrt(sigma2);
            ws.w[obs_loc] = ws.sqrt_d[i] * z[obs_loc];
            ws.B_n_nb[i] = 0;
            continue;
        }

        // Count neighbors
        int n_nb = 0;
        for (int j = 0; j < nn && gp_data.nn_idx[i * nn + j] > 0; j++) n_nb++;

        if (n_nb == 0) {
            ws.sqrt_d[i] = std::sqrt(sigma2);
            ws.w[obs_loc] = ws.sqrt_d[i] * z[obs_loc];
            ws.B_n_nb[i] = 0;
            continue;
        }

        // Build c_vec (covariance between obs and its neighbors)
        for (int j = 0; j < n_nb; j++) {
            double d = gp_data.nn_dist[i * nn + j];
            c_eigen(j) = compute_cov(d, sigma2, phi, gp_data.cov_type);
        }

        // Validate neighbor indices and build C_mat (symmetric fill)
        bool ok = true;
        for (int j = 0; j < n_nb && ok; j++) {
            int raw = gp_data.nn_idx[i * nn + j];
            if (raw - 1 < 0 || raw - 1 >= (int)gp_data.nn_order.size()) { ok = false; break; }
            int loc = gp_data.nn_order[raw - 1];
            if (loc < 0 || loc >= N) { ok = false; break; }
            ws.nb_idx_flat[i * nn + j] = loc;
        }
        if (!ok) {
            ws.sqrt_d[i] = std::sqrt(sigma2);
            ws.w[obs_loc] = ws.sqrt_d[i] * z[obs_loc];
            ws.B_n_nb[i] = 0;
            continue;
        }

        // Build C_mat using cached distances (symmetric, upper triangle only)
        for (int j1 = 0; j1 < n_nb; j1++) {
            C_eigen(j1, j1) = sigma2 + 1e-8;  // Jitter for stability
            for (int j2 = j1 + 1; j2 < n_nb; j2++) {
                double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
                double cov_val = compute_cov(d12, sigma2, phi, gp_data.cov_type);
                C_eigen(j1, j2) = cov_val;
                C_eigen(j2, j1) = cov_val;
            }
        }

        // Eigen Cholesky: C = LL', alpha = C^{-1}c
        llt.compute(C_eigen.topLeftCorner(n_nb, n_nb));
        if (llt.info() != Eigen::Success) {
            ws.sqrt_d[i] = std::sqrt(sigma2);
            ws.w[obs_loc] = ws.sqrt_d[i] * z[obs_loc];
            ws.B_n_nb[i] = 0;
            continue;
        }

        Eigen::VectorXd alpha_vec = llt.solve(c_eigen.head(n_nb));

        // Cache Cholesky factor L for backward phi gradient
        Eigen::MatrixXd L_mat = llt.matrixL();
        for (int j1 = 0; j1 < n_nb; j1++) {
            for (int j2 = 0; j2 <= j1; j2++) {
                ws.L_flat[i * nn * nn + j1 * nn + j2] = L_mat(j1, j2);
            }
        }

        // Store B and compute conditional variance d_i
        double c_alpha = 0.0;
        for (int j = 0; j < n_nb; j++) {
            ws.B_flat[i * nn + j] = alpha_vec(j);
            c_alpha += c_eigen(j) * alpha_vec(j);
        }
        ws.B_n_nb[i] = n_nb;

        double d_i = std::max(sigma2 - c_alpha, 1e-10);
        ws.sqrt_d[i] = std::sqrt(d_i);

        // Forward transform: w[loc] = B @ w_neighbors + sqrt(d_i) * z[loc]
        double mu = 0.0;
        for (int j = 0; j < n_nb; j++) {
            mu += alpha_vec(j) * ws.w[ws.nb_idx_flat[i * nn + j]];
        }
        ws.w[obs_loc] = mu + ws.sqrt_d[i] * z[obs_loc];
    }
}

// Backward pass: given dL/dw from likelihood, compute gradients for z, log_sigma2, log_phi.
// z and grad_z are indexed by LOCATION (matching parameter layout).
// adj is indexed by NNGP order (internal).
//
// Adjoint propagation (reverse NNGP order) is sequential.
// Phi gradient loop is independent per observation — OpenMP parallelized.
// Uses Eigen for triangular solves (from cached L).
inline void nngp_nc_backward(
    const double* z,            // z[loc_idx], location-indexed
    double sigma2, double phi,
    const GPData& gp_data,
    const NNGPNCWorkspace& ws,
    const double* dL_dw,        // Likelihood gradient w.r.t. w[loc] (location-indexed)
    double* grad_z,             // Output: full gradient for z[loc] (prior + likelihood)
    double& grad_log_sigma2_lik,// Output: likelihood contribution to sigma2 gradient
    double& grad_log_phi_lik,   // Output: likelihood contribution to phi gradient
    double& grad_log_phi_jac    // Output: Jacobian contribution to phi gradient
) {
    int N = gp_data.n_obs;
    int nn = gp_data.nn;
    const std::vector<int>& nn_order_inv = gp_data.nn_order_inv;

    // Initialize adjoint from direct likelihood contribution (NNGP-order indexed)
    std::vector<double>& adj = const_cast<NNGPNCWorkspace&>(ws).adj;
    for (int i = 0; i < N; i++) {
        int loc = gp_data.nn_order[i];
        adj[i] = dL_dw[loc];
    }

    // Backward adjoint propagation (reverse NNGP order) — SEQUENTIAL
    for (int i = N - 1; i >= 1; i--) {
        int n_nb = ws.B_n_nb[i];
        for (int j = 0; j < n_nb; j++) {
            int nb_loc = ws.nb_idx_flat[i * nn + j];
            if (nb_loc >= 0 && nb_loc < N) {
                int nb_nngp = nn_order_inv[nb_loc];
                if (nb_nngp >= 0 && nb_nngp < N) {
                    adj[nb_nngp] += ws.B_flat[i * nn + j] * adj[i];
                }
            }
        }
    }

    // z gradients: prior (-z) + likelihood (sqrt_d * adj)
    for (int i = 0; i < N; i++) {
        int loc = gp_data.nn_order[i];
        grad_z[loc] = -z[loc] + ws.sqrt_d[i] * adj[i];
    }

    // --- Hyperparameter gradients ---

    // sigma2 likelihood gradient
    grad_log_sigma2_lik = 0.0;
    for (int i = 0; i < N; i++) {
        int loc = gp_data.nn_order[i];
        grad_log_sigma2_lik += adj[i] * 0.5 * ws.sqrt_d[i] * z[loc];
    }

    // phi gradients (likelihood + Jacobian) — OpenMP parallelized
    // Each observation's phi contribution is independent.
    grad_log_phi_lik = 0.0;
    grad_log_phi_jac = 0.0;

    // Thread-local workspace setup
    int n_threads = tulpa_omp_team_size(N - 1);

    std::vector<double> tl_phi_lik(n_threads, 0.0);
    std::vector<double> tl_phi_jac(n_threads, 0.0);

    struct BackwardWS {
        Eigen::MatrixXd L_eigen, C_eigen;
        Eigen::VectorXd c_eigen, dc_eigen, rhs_eigen, dalpha_eigen, alpha_eigen;
        std::vector<double> dC_alpha;
        BackwardWS(int nn_) : L_eigen(nn_, nn_), C_eigen(nn_, nn_),
                              c_eigen(nn_), dc_eigen(nn_), rhs_eigen(nn_),
                              dalpha_eigen(nn_), alpha_eigen(nn_), dC_alpha(nn_) {}
    };
    std::vector<BackwardWS> bws_vec(n_threads, BackwardWS(nn));

    #ifdef _OPENMP
    #pragma omp parallel num_threads(n_threads)
    #endif
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif

        auto& L_eigen = bws_vec[tid].L_eigen;
        auto& C_eigen = bws_vec[tid].C_eigen;
        auto& c_eigen = bws_vec[tid].c_eigen;
        auto& dc_eigen = bws_vec[tid].dc_eigen;
        auto& rhs_eigen = bws_vec[tid].rhs_eigen;
        auto& dalpha_eigen = bws_vec[tid].dalpha_eigen;
        auto& alpha_eigen = bws_vec[tid].alpha_eigen;
        auto& dC_alpha = bws_vec[tid].dC_alpha;

        #ifdef _OPENMP
        #pragma omp for schedule(dynamic)
        #endif
        for (int i = 1; i < N; i++) {
            int obs_loc = gp_data.nn_order[i];
            int n_nb = ws.B_n_nb[i];
            if (n_nb == 0 || obs_loc < 0 || obs_loc >= N) continue;

            // Rebuild c_vec, dc_vec, and C_mat for phi derivatives
            for (int j = 0; j < n_nb; j++) {
                double d = gp_data.nn_dist[i * nn + j];
                c_eigen(j) = compute_cov(d, sigma2, phi, gp_data.cov_type);
                dc_eigen(j) = dcov_dphi(d, phi, c_eigen(j), gp_data.cov_type);
                alpha_eigen(j) = ws.B_flat[i * nn + j];
            }

            // Rebuild C_mat from cached distances (needed for dcov_dphi)
            for (int j1 = 0; j1 < n_nb; j1++) {
                C_eigen(j1, j1) = sigma2;
                for (int j2 = j1 + 1; j2 < n_nb; j2++) {
                    double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
                    double cov_val = compute_cov(d12, sigma2, phi, gp_data.cov_type);
                    C_eigen(j1, j2) = cov_val;
                    C_eigen(j2, j1) = cov_val;
                }
            }

            // Restore cached L factor into Eigen matrix
            L_eigen.topLeftCorner(n_nb, n_nb).setZero();
            for (int j1 = 0; j1 < n_nb; j1++) {
                for (int j2 = 0; j2 <= j1; j2++) {
                    L_eigen(j1, j2) = ws.L_flat[i * nn * nn + j1 * nn + j2];
                }
            }

            // dC/dphi * alpha (using properly rebuilt C_mat for dcov_dphi)
            std::fill(dC_alpha.begin(), dC_alpha.begin() + n_nb, 0.0);
            for (int j1 = 0; j1 < n_nb; j1++) {
                for (int j2 = 0; j2 < n_nb; j2++) {
                    if (j1 != j2) {
                        double d12 = gp_data.nn_neighbor_dist[i * nn * nn + j1 * nn + j2];
                        double dC_jk = dcov_dphi(d12, phi, C_eigen(j1, j2), gp_data.cov_type);
                        dC_alpha[j1] += dC_jk * alpha_eigen(j2);
                    }
                }
            }

            // dalpha/dphi = C^{-1} (dc/dphi - dC/dphi * alpha)
            // Use Eigen triangular solve with cached L
            for (int j = 0; j < n_nb; j++) rhs_eigen(j) = dc_eigen(j) - dC_alpha[j];
            auto L_sub = L_eigen.topLeftCorner(n_nb, n_nb);
            Eigen::VectorXd y_temp = L_sub.triangularView<Eigen::Lower>().solve(rhs_eigen.head(n_nb));
            dalpha_eigen.head(n_nb) = L_sub.transpose().triangularView<Eigen::Upper>().solve(y_temp);

            // dd/dphi = -2 * dc' * alpha + alpha' * dC * alpha
            double alpha_dc = 0.0, alpha_dC_alpha = 0.0;
            for (int j = 0; j < n_nb; j++) {
                alpha_dc += alpha_eigen(j) * dc_eigen(j);
                alpha_dC_alpha += alpha_eigen(j) * dC_alpha[j];
            }
            double dd_dphi = -2.0 * alpha_dc + alpha_dC_alpha;

            // Likelihood: dw[loc]/dphi = sum_j dalpha[j]*w[nb_j] + dd_dphi/(2*sqrt(d_i))*z[loc]
            double dw_dphi = 0.0;
            for (int j = 0; j < n_nb; j++) {
                dw_dphi += dalpha_eigen(j) * ws.w[ws.nb_idx_flat[i * nn + j]];
            }
            double sqrt_di = ws.sqrt_d[i];
            if (sqrt_di > 1e-15) {
                dw_dphi += dd_dphi / (2.0 * sqrt_di) * z[obs_loc];
            }
            tl_phi_lik[tid] += adj[i] * dw_dphi * phi;

            // Jacobian: d/d(phi) [0.5*log(d_i)] = 0.5 * dd_dphi / d_i
            double d_i = sqrt_di * sqrt_di;
            if (d_i > 1e-15) {
                tl_phi_jac[tid] += 0.5 * dd_dphi / d_i * phi;
            }
        }
    }

    // Reduce thread-local phi accumulators
    for (int t = 0; t < n_threads; t++) {
        grad_log_phi_lik += tl_phi_lik[t];
        grad_log_phi_jac += tl_phi_jac[t];
    }
}
