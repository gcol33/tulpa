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

// A lightweight, non-owning view over the neighbour-structure fields the
// transform needs. Lets nngp_nc_forward / nngp_nc_backward serve GPData and
// each independent scale of MultiscaleGPData (both cache pairwise neighbour
// distances in nn_neighbor_dist, the fast path) as well as SVCData (which
// does not cache them -- its density kernel, tulpa_svc::nngp_log_lik,
// recomputes neighbour-pair distances from coords instead) without
// duplicating the Cholesky / adjoint-propagation logic per caller. One
// `make_*_view` factory per source struct lives below, next to the view it
// builds, so the two consumers of each -- the log-post transform
// (gp/svc/msgp_nc_apply.cpp) and the draw-storage transform
// (hmc_nuts_chain_iter_store.h) -- read the same field mapping
// (gcol33/tulpa#243).
struct NNGPNCView {
    int n_obs = 0, nn = 0;
    const int* nn_idx = nullptr;              // [n_obs x nn], 1-based, 0 = no neighbor
    const double* nn_dist = nullptr;          // [n_obs x nn]
    const double* nn_neighbor_dist = nullptr; // [n_obs x nn x nn] row-major [i,j1,j2];
                                               // null -> pair_dist() falls back to coords
    const int* nn_order = nullptr;            // [n_obs], 0-based location order
    const int* nn_order_inv = nullptr;        // [n_obs], inverse permutation
    const double* coords = nullptr;           // [n_obs x 2]; only read when
                                               // nn_neighbor_dist is null
    CovType cov_type = CovType::EXPONENTIAL;

    // Conditioning policy of the density kernel this view's field belongs to:
    // the diagonal NUGGET on the neighbour covariance, the floor on the
    // conditional variance, and how that floor is applied. The transform below
    // reads them from here rather than from literals, so a field fitted
    // non-centered and the same field fitted centered condition identically and
    // therefore describe the same posterior. Each make_*_nc_view factory sets
    // all three from its own kernel's constants.
    double jitter = kGpJitter;
    double var_floor = kGpVarFloor;
    tulpa_nngp::VarFloor floor_mode = tulpa_nngp::VarFloor::Clamp;

    // Distance between two already-resolved neighbour locations loc1/loc2 for
    // observation i's j1-th/j2-th neighbour slot. Prefers the cached table;
    // falls back to a direct coordinate lookup (the SVCData path).
    inline double pair_dist(int i, int j1, int j2, int loc1, int loc2) const {
        if (nn_neighbor_dist) {
            return nn_neighbor_dist[(std::size_t)i * nn * nn +
                                     (std::size_t)j1 * nn + j2];
        }
        double dx = coords[(std::size_t)loc1 * 2]     - coords[(std::size_t)loc2 * 2];
        double dy = coords[(std::size_t)loc1 * 2 + 1] - coords[(std::size_t)loc2 * 2 + 1];
        return std::sqrt(dx * dx + dy * dy);
    }
};

// Fast-path view over GPData (cached nn_neighbor_dist).
inline NNGPNCView make_gp_nc_view(const GPData& g) {
    NNGPNCView v;
    v.n_obs = g.n_obs;
    v.nn = g.nn;
    v.nn_idx = g.nn_idx.data();
    v.nn_dist = g.nn_dist.data();
    v.nn_neighbor_dist = g.nn_neighbor_dist.data();
    v.nn_order = g.nn_order.data();
    v.nn_order_inv = g.nn_order_inv.data();
    v.coords = g.coords.data();
    v.cov_type = g.cov_type;
    v.jitter = kGpJitter;
    v.var_floor = kGpVarFloor;
    v.floor_mode = tulpa_nngp::VarFloor::Clamp;
    return v;
}

// MultiscaleGPData holds two independent NNGP scales (local / regional),
// each with its own cached nn_neighbor_dist -- the fast path, same as
// GPData. These build the view for one scale at a time; the caller runs the
// transform once per scale.
inline NNGPNCView make_msgp_nc_view_local(const MultiscaleGPData& g) {
    NNGPNCView v;
    v.n_obs = g.n_obs;
    v.nn = g.nn_local;
    v.nn_idx = g.nn_idx_local.data();
    v.nn_dist = g.nn_dist_local.data();
    v.nn_neighbor_dist = g.nn_neighbor_dist_local.data();
    v.nn_order = g.nn_order_local.data();
    v.nn_order_inv = g.nn_order_inv_local.data();
    v.coords = g.coords.data();
    v.cov_type = g.cov_type;
    v.jitter = kGpJitter;
    v.var_floor = kGpVarFloor;
    v.floor_mode = tulpa_nngp::VarFloor::Clamp;
    return v;
}

inline NNGPNCView make_msgp_nc_view_regional(const MultiscaleGPData& g) {
    NNGPNCView v;
    v.n_obs = g.n_obs;
    v.nn = g.nn_regional;
    v.nn_idx = g.nn_idx_regional.data();
    v.nn_dist = g.nn_dist_regional.data();
    v.nn_neighbor_dist = g.nn_neighbor_dist_regional.data();
    v.nn_order = g.nn_order_regional.data();
    v.nn_order_inv = g.nn_order_inv_regional.data();
    v.coords = g.coords.data();
    v.cov_type = g.cov_type;
    v.jitter = kGpJitter;
    v.var_floor = kGpVarFloor;
    v.floor_mode = tulpa_nngp::VarFloor::Clamp;
    return v;
}

// SVCData does not cache pairwise neighbour distances (its density kernel,
// tulpa_svc::nngp_log_lik, recomputes them from coords), so this view takes
// the coords-fallback branch of pair_dist(). The neighbour topology is shared
// by every SVC term -- only (sigma2_j, phi_j) differ -- so one view serves
// them all.
inline NNGPNCView make_svc_nc_view(const tulpa::SVCData& s) {
    NNGPNCView v;
    v.n_obs = s.n_obs;
    v.nn = s.nn;
    v.nn_idx = s.nn_idx.data();
    v.nn_dist = s.nn_dist.data();
    v.nn_neighbor_dist = nullptr;
    v.nn_order = s.nn_order.data();
    v.nn_order_inv = s.nn_order_inv.data();
    v.coords = s.coords.data();
    v.cov_type = s.cov_type;
    // The SVC density kernel conditions deliberately more loosely than the GP
    // one, and blends at the floor; the transform has to match it or a
    // non-centered SVC fit targets a different posterior than a centered one.
    v.jitter = tulpa_svc::kSvcJitter;
    v.var_floor = tulpa_svc::kSvcVarFloor;
    v.floor_mode = tulpa_nngp::VarFloor::Blend;
    return v;
}

struct NNGPNCWorkspace {
    int N = 0, nn = 0;
    std::vector<double> w;          // Transformed spatial effects (N)
    std::vector<double> sqrt_d;     // sqrt(conditional variance) per obs (N)
    std::vector<double> B_flat;     // Regression coefficients (N * nn)
    std::vector<int> B_n_nb;        // Number of actual neighbors per obs (N)
    std::vector<int> nb_idx_flat;   // Neighbor indices per obs (N * nn), 0-based in w
    std::vector<double> adj;        // Adjoint accumulator (N)
    std::vector<double> L_flat;     // Cached Cholesky factors (N * nn * nn) for backward phi grad
    std::vector<double> d_raw;      // Unfloored conditional variance per obs (N)
    std::vector<double> d_slope;    // d(floored d_i)/d(raw d_i) per obs (N)

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
        d_raw.assign(Ns, 0.0);
        d_slope.assign(Ns, 1.0);
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
    const NNGPNCView& view,
    NNGPNCWorkspace& ws
) {
    int N = view.n_obs;
    int nn = view.nn;
    ws.init(N, nn);

    // Pre-allocated Eigen workspace (reused across iterations)
    Eigen::MatrixXd C_eigen(nn, nn);
    Eigen::VectorXd c_eigen(nn);
    Eigen::LLT<Eigen::MatrixXd> llt(nn);

    // First observation: marginal N(0, sigma2). Guard the location index: a
    // malformed nn_order (e.g. a 1-based ordering leaking through) would make
    // ws.w[first_loc] / z[first_loc] an out-of-bounds access.
    int first_loc = view.nn_order[0];
    ws.sqrt_d[0] = std::sqrt(sigma2);
    ws.d_raw[0] = sigma2;
    ws.d_slope[0] = 1.0;
    ws.B_n_nb[0] = 0;
    if (first_loc >= 0 && first_loc < N) {
        ws.w[first_loc] = ws.sqrt_d[0] * z[first_loc];
    }

    for (int i = 1; i < N; i++) {
        int obs_loc = view.nn_order[i];
        if (obs_loc < 0 || obs_loc >= N) {
            // obs_loc is out of range; set only the i-indexed fields and skip
            // the ws.w[obs_loc] / z[obs_loc] write (matches the gradient path).
            ws.sqrt_d[i] = std::sqrt(sigma2);
            ws.d_raw[i] = sigma2;
            ws.d_slope[i] = 1.0;
            ws.B_n_nb[i] = 0;
            continue;
        }

        // Count neighbors, through the shared left-packed scan the density
        // kernels run.
        const int n_nb = tulpa_nngp::nngp_row_neighbours(
            view.nn_idx + (std::size_t)i * nn, /*stride=*/1, nn, view.n_obs);

        if (n_nb == 0) {
            ws.sqrt_d[i] = std::sqrt(sigma2);
            ws.d_raw[i] = sigma2;
            ws.d_slope[i] = 1.0;
            ws.w[obs_loc] = ws.sqrt_d[i] * z[obs_loc];
            ws.B_n_nb[i] = 0;
            continue;
        }

        // Build c_vec (covariance between obs and its neighbors)
        for (int j = 0; j < n_nb; j++) {
            double d = view.nn_dist[i * nn + j];
            c_eigen(j) = compute_cov(d, sigma2, phi, view.cov_type);
        }

        // Validate neighbor indices and build C_mat (symmetric fill)
        bool ok = true;
        for (int j = 0; j < n_nb && ok; j++) {
            int raw = view.nn_idx[i * nn + j];
            int loc = view.nn_order[raw - 1];
            if (loc < 0 || loc >= N) { ok = false; break; }
            ws.nb_idx_flat[i * nn + j] = loc;
        }
        if (!ok) {
            ws.sqrt_d[i] = std::sqrt(sigma2);
            ws.d_raw[i] = sigma2;
            ws.d_slope[i] = 1.0;
            ws.w[obs_loc] = ws.sqrt_d[i] * z[obs_loc];
            ws.B_n_nb[i] = 0;
            continue;
        }

        // Build C_mat using cached distances (symmetric, upper triangle only)
        for (int j1 = 0; j1 < n_nb; j1++) {
            // Diagonal NUGGET, from the density kernel this field belongs to.
            C_eigen(j1, j1) = sigma2 + view.jitter;
            for (int j2 = j1 + 1; j2 < n_nb; j2++) {
                double d12 = view.pair_dist(i, j1, j2, ws.nb_idx_flat[i * nn + j1],
                                             ws.nb_idx_flat[i * nn + j2]);
                double cov_val = compute_cov(d12, sigma2, phi, view.cov_type);
                C_eigen(j1, j2) = cov_val;
                C_eigen(j2, j1) = cov_val;
            }
        }

        // Eigen Cholesky: C = LL', alpha = C^{-1}c
        llt.compute(C_eigen.topLeftCorner(n_nb, n_nb));
        if (llt.info() != Eigen::Success) {
            ws.sqrt_d[i] = std::sqrt(sigma2);
            ws.d_raw[i] = sigma2;
            ws.d_slope[i] = 1.0;
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

        // Conditional variance, floored by the density kernel's own policy.
        // Its slope through the floor is kept so the backward pass reports the
        // derivative of the variance that was USED, not of the one it replaced.
        double slope = 1.0;
        double d_i = tulpa_nngp::apply_var_floor(sigma2 - c_alpha,
                                                 view.var_floor,
                                                 view.floor_mode, &slope);
        ws.d_raw[i] = sigma2 - c_alpha;
        ws.d_slope[i] = slope;
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
// grad_z carries the LIKELIHOOD/transform gradient only (d(loglik)/dz through
// w). The N(0, I) prior on z is the caller's responsibility -- it is added as
// a templated -0.5 z'z term whose gradient the autodiff tape supplies
// separately -- so this routine does not fold in the -z prior term. This
// matches the SpdeNcTransform::backward contract (adjoint of the transform
// only; the z prior is held by the caller).
//
// grad_log_phi_jac is the derivative of the z->w log-Jacobian and is returned
// for callers that place the transform's density on w. The pure non-centered
// parameterization (the sampler path) samples z ~ N(0, I) and evaluates the
// likelihood at w = f(z, theta) with NO change-of-variables Jacobian, so that
// caller drops grad_log_phi_jac.
//
// Adjoint propagation (reverse NNGP order) is sequential.
// Phi gradient loop is independent per observation — OpenMP parallelized.
// Uses Eigen for triangular solves (from cached L).
inline void nngp_nc_backward(
    const double* z,            // z[loc_idx], location-indexed
    double sigma2, double phi,
    const NNGPNCView& view,
    const NNGPNCWorkspace& ws,
    const double* dL_dw,        // Likelihood gradient w.r.t. w[loc] (location-indexed)
    double* grad_z,             // Output: likelihood/transform gradient for z[loc]
    double& grad_log_sigma2_lik,// Output: likelihood contribution to sigma2 gradient
    double& grad_log_phi_lik,   // Output: likelihood contribution to phi gradient
    double& grad_log_phi_jac    // Output: z->w log-Jacobian contribution to phi gradient
) {
    int N = view.n_obs;
    int nn = view.nn;
    const int* nn_order_inv = view.nn_order_inv;

    // Initialize adjoint from direct likelihood contribution (NNGP-order indexed)
    std::vector<double>& adj = const_cast<NNGPNCWorkspace&>(ws).adj;
    for (int i = 0; i < N; i++) {
        int loc = view.nn_order[i];
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

    // z gradients: likelihood/transform only (sqrt_d * adj). The N(0, I) prior
    // on z is added by the caller (templated -0.5 z'z) and differentiated by
    // the tape, so it is not folded in here -- matching the SpdeNcTransform
    // backward contract.
    for (int i = 0; i < N; i++) {
        int loc = view.nn_order[i];
        grad_z[loc] = ws.sqrt_d[i] * adj[i];
    }

    // --- Hyperparameter gradients ---

    // sigma2 likelihood gradient
    // dw[loc]/d(log sigma2) = (dd_i/d log sigma2) / (2 sqrt(d_i)) * z[loc].
    // With d_i unfloored it is homogeneous of degree 1 in sigma2, so
    // dd_i/d log sigma2 = d_i and the whole term collapses to
    // 0.5 * sqrt(d_i) * z. Where the floor bound, d_i no longer tracks sigma2
    // at that rate: the derivative is the raw variance scaled by the floor's
    // own slope, which is 0 under Clamp.
    grad_log_sigma2_lik = 0.0;
    for (int i = 0; i < N; i++) {
        int loc = view.nn_order[i];
        if (ws.d_slope[i] >= 1.0) {
            grad_log_sigma2_lik += adj[i] * 0.5 * ws.sqrt_d[i] * z[loc];
        } else if (ws.sqrt_d[i] > 0.0) {
            grad_log_sigma2_lik += adj[i] * ws.d_slope[i] * ws.d_raw[i] /
                                   (2.0 * ws.sqrt_d[i]) * z[loc];
        }
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

    // One row's phi contribution, written once and driven from both routes.
    // The workspace slot and the two accumulators are parameters rather than
    // captures, so the parallel route hands each thread its own slot and the
    // serial route reuses slot 0 -- bws_vec is never copied onto a worker
    // stack (the gcol33/tulpa#253 rule).
    auto phi_row = [&](int i, BackwardWS& bw, double& acc_lik, double& acc_jac) {
        auto& L_eigen = bw.L_eigen;
        auto& C_eigen = bw.C_eigen;
        auto& c_eigen = bw.c_eigen;
        auto& dc_eigen = bw.dc_eigen;
        auto& rhs_eigen = bw.rhs_eigen;
        auto& dalpha_eigen = bw.dalpha_eigen;
        auto& alpha_eigen = bw.alpha_eigen;
        auto& dC_alpha = bw.dC_alpha;
        int obs_loc = view.nn_order[i];
        int n_nb = ws.B_n_nb[i];
        if (n_nb == 0 || obs_loc < 0 || obs_loc >= N) return;

        // Rebuild c_vec, dc_vec, and C_mat for phi derivatives
        for (int j = 0; j < n_nb; j++) {
            double d = view.nn_dist[i * nn + j];
            c_eigen(j) = compute_cov(d, sigma2, phi, view.cov_type);
            dc_eigen(j) = dcov_dphi(d, phi, c_eigen(j), sigma2, view.cov_type);
            alpha_eigen(j) = ws.B_flat[i * nn + j];
        }

        // Rebuild C_mat from cached distances (needed for dcov_dphi)
        for (int j1 = 0; j1 < n_nb; j1++) {
            C_eigen(j1, j1) = sigma2;
            for (int j2 = j1 + 1; j2 < n_nb; j2++) {
                double d12 = view.pair_dist(i, j1, j2, ws.nb_idx_flat[i * nn + j1],
                                             ws.nb_idx_flat[i * nn + j2]);
                double cov_val = compute_cov(d12, sigma2, phi, view.cov_type);
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
                    double d12 = view.pair_dist(i, j1, j2, ws.nb_idx_flat[i * nn + j1],
                                                 ws.nb_idx_flat[i * nn + j2]);
                    double dC_jk = dcov_dphi(d12, phi, C_eigen(j1, j2), sigma2,
                                              view.cov_type);
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
        // Same floor slope as the sigma2 term above: a floored d_i does not
        // move with phi at the unfloored rate either.
        double dd_dphi = ws.d_slope[i] * (-2.0 * alpha_dc + alpha_dC_alpha);

        // Likelihood: dw[loc]/dphi = sum_j dalpha[j]*w[nb_j] + dd_dphi/(2*sqrt(d_i))*z[loc]
        double dw_dphi = 0.0;
        for (int j = 0; j < n_nb; j++) {
            dw_dphi += dalpha_eigen(j) * ws.w[ws.nb_idx_flat[i * nn + j]];
        }
        double sqrt_di = ws.sqrt_d[i];
        if (sqrt_di > 1e-15) {
            dw_dphi += dd_dphi / (2.0 * sqrt_di) * z[obs_loc];
        }
        acc_lik += adj[i] * dw_dphi * phi;

        // Jacobian: d/d(phi) [0.5*log(d_i)] = 0.5 * dd_dphi / d_i
        double d_i = sqrt_di * sqrt_di;
        if (d_i > 1e-15) {
            acc_jac += 0.5 * dd_dphi / d_i * phi;
        }
    };

    // At a team of one the region is skipped entirely: entering libgomp costs
    // measurable time per reverse sweep, and this runs on every HMC gradient
    // (gcol33/tulpa#365, #373). One thread means every row already accumulated
    // into slot 0 in index order, which is exactly what the plain loop does,
    // so the two routes are bit-identical. Above one thread the dynamic
    // schedule is unchanged.
    #ifdef _OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(dynamic)
            for (int i = 1; i < N; i++) {
                phi_row(i, bws_vec[tid], tl_phi_lik[tid], tl_phi_jac[tid]);
            }
        }
    } else
    #endif
    {
        for (int i = 1; i < N; i++) {
            phi_row(i, bws_vec[0], tl_phi_lik[0], tl_phi_jac[0]);
        }
    }

    // Reduce thread-local phi accumulators
    for (int t = 0; t < n_threads; t++) {
        grad_log_phi_lik += tl_phi_lik[t];
        grad_log_phi_jac += tl_phi_jac[t];
    }
}
