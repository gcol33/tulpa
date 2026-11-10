// =============================================================================
// Non-centered NNGP parameterization
// =============================================================================
//
// A namespace-body fragment, not a standalone header: hmc_gp.h includes it
// inside `namespace tulpa_gp { }`, and it resolves dcov_dphi / compute_cov from
// what that file has already declared. It carries no include guard and must not
// be included directly. Same pattern as hmc_nuts_chain_setup.h and the
// hmc_nuts_chain_iter_*.h fragments, which hmc_nuts_chain.cpp documents.

#include "omp_threads.h"
// Non-centred Vecchia fields satisfy w_i = sum_j B_ij w_j + sqrt(d_i) z_i.
// The neighbour covariance and marginal variance carry their declared nuggets.
// Likelihood derivatives pass through both the weights and conditional variances.

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

    // A negative marginal nugget couples it to the neighbour nugget.
    double jitter = kGpJitter;
    double marginal_nugget = -1.0;
    double own_nugget() const { return marginal_nugget < 0.0 ? jitter : marginal_nugget; }
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
    v.jitter = s.neighbor_jitter;
    v.marginal_nugget = s.marginal_nugget;
    v.var_floor = s.variance_floor;
    v.floor_mode = tulpa_svc::svc_floor_mode(s);
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

inline double nngp_conditional_log_variance_derivative(
    int i, const NNGPNCView& view, const NNGPNCWorkspace& ws) {
    double norm = 0.0;
    for (int j = 0; j < ws.B_n_nb[i]; ++j) {
        const double value = ws.B_flat[static_cast<size_t>(i) * view.nn + j];
        norm += value * value;
    }
    return ws.d_slope[i] * (ws.d_raw[i] - view.own_nugget() - view.jitter * norm);
}

struct NNGPRowDerivatives {
    Eigen::MatrixXd L;
    Eigen::VectorXd alpha, dc, rhs, dC_alpha, d_alpha_sigma, d_alpha_phi;
    double d_variance_phi = 0.0;

    explicit NNGPRowDerivatives(int nn) : L(nn, nn), alpha(nn), dc(nn), rhs(nn),
        dC_alpha(nn), d_alpha_sigma(nn), d_alpha_phi(nn) {}
};

inline void nngp_row_derivatives(int i, double sigma2, double phi,
    const NNGPNCView& view, const NNGPNCWorkspace& ws, NNGPRowDerivatives& out) {
    const int nn = view.nn, count = ws.B_n_nb[i];
    out.d_variance_phi = 0.0;
    if (count == 0) return;
    out.L.topLeftCorner(count, count).setZero();
    for (int j = 0; j < count; ++j) {
        const double distance = view.nn_dist[static_cast<size_t>(i) * nn + j];
        const double covariance = compute_cov(distance, sigma2, phi, view.cov_type);
        out.dc(j) = dcov_dphi(distance, phi, covariance, sigma2, view.cov_type);
        out.alpha(j) = ws.B_flat[static_cast<size_t>(i) * nn + j];
        for (int k = 0; k <= j; ++k)
            out.L(j, k) = ws.L_flat[static_cast<size_t>(i) * nn * nn + j * nn + k];
    }
    const auto L = out.L.topLeftCorner(count, count);
    out.d_alpha_sigma.head(count).setZero();
    if (view.jitter != 0.0) {
        const Eigen::VectorXd y = L.triangularView<Eigen::Lower>().solve(out.alpha.head(count));
        out.d_alpha_sigma.head(count) = view.jitter *
            L.transpose().triangularView<Eigen::Upper>().solve(y);
    }
    out.dC_alpha.head(count).setZero();
    for (int j = 0; j < count; ++j) for (int k = 0; k < count; ++k) if (j != k) {
        const double distance = view.pair_dist(i, j, k, ws.nb_idx_flat[i * nn + j],
            ws.nb_idx_flat[i * nn + k]);
        const double covariance = compute_cov(distance, sigma2, phi, view.cov_type);
        out.dC_alpha(j) += dcov_dphi(distance, phi, covariance, sigma2, view.cov_type) * out.alpha(k);
    }
    out.rhs.head(count) = out.dc.head(count) - out.dC_alpha.head(count);
    const Eigen::VectorXd y = L.triangularView<Eigen::Lower>().solve(out.rhs.head(count));
    out.d_alpha_phi.head(count) = L.transpose().triangularView<Eigen::Upper>().solve(y);
    out.d_variance_phi = ws.d_slope[i] * (-2.0 * out.alpha.head(count).dot(out.dc.head(count))
        + out.alpha.head(count).dot(out.dC_alpha.head(count)));
}

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

    // An unconditioned location has variance sigma2 plus its marginal nugget.
    const double own_var = sigma2 + view.own_nugget();

    // First observation: marginal N(0, own_var). Guard the location index: a
    // malformed nn_order (e.g. a 1-based ordering leaking through) would make
    // ws.w[first_loc] / z[first_loc] an out-of-bounds access.
    int first_loc = view.nn_order[0];
    ws.sqrt_d[0] = std::sqrt(own_var);
    ws.d_raw[0] = own_var;
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
            ws.sqrt_d[i] = std::sqrt(own_var);
            ws.d_raw[i] = own_var;
            ws.d_slope[i] = 1.0;
            ws.B_n_nb[i] = 0;
            continue;
        }

        // Count neighbors, through the shared left-packed scan the density
        // kernels run.
        const int n_nb = tulpa_nngp::nngp_row_neighbours(
            view.nn_idx + (std::size_t)i * nn, /*stride=*/1, nn, view.n_obs);

        if (n_nb == 0) {
            ws.sqrt_d[i] = std::sqrt(own_var);
            ws.d_raw[i] = own_var;
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
            ws.sqrt_d[i] = std::sqrt(own_var);
            ws.d_raw[i] = own_var;
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
            ws.sqrt_d[i] = std::sqrt(own_var);
            ws.d_raw[i] = own_var;
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
        double d_i = tulpa_nngp::apply_var_floor(own_var - c_alpha,
                                                 view.var_floor,
                                                 view.floor_mode, &slope);
        ws.d_raw[i] = own_var - c_alpha;
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
    //
    // The conditional variance is NOT homogeneous of degree 1 in sigma2 once
    // the neighbour covariance carries a diagonal nugget: the forward builds
    // For C = sigma2 R + j_neighbor I and d_raw = sigma2 + j_marginal - c' alpha,
    // d(d_raw)/d(log sigma2) = d_raw - j_marginal - j_neighbor ||alpha||^2.
    grad_log_sigma2_lik = 0.0;
    for (int i = 0; i < N; i++) {
        if (!(ws.sqrt_d[i] > 0.0)) continue;
        const int loc = view.nn_order[i];
        const double dd_dlog_sigma2 = nngp_conditional_log_variance_derivative(i, view, ws);
        grad_log_sigma2_lik +=
            adj[i] * dd_dlog_sigma2 / (2.0 * ws.sqrt_d[i]) * z[loc];
    }

    // Per-row hyperparameter gradients (phi likelihood + phi Jacobian + the
    // sigma2 channel that runs through alpha) — OpenMP parallelized. Each
    // observation's contribution is independent, and all three read the same
    // restored Cholesky factor and alpha, so they are accumulated in one pass
    // rather than in a second loop that would rebuild both.
    grad_log_phi_lik = 0.0;
    grad_log_phi_jac = 0.0;

    // Thread-local workspace setup
    // Per-row work is an nn x nn neighbour solve; see hmc_gp_gradients.h.
    int n_threads = tulpa_omp_team_size_grain(N - 1, (long long)nn * nn * nn);

    std::vector<double> tl_phi_lik(n_threads, 0.0);
    std::vector<double> tl_phi_jac(n_threads, 0.0);
    std::vector<double> tl_s2_alpha(n_threads, 0.0);

    std::vector<NNGPRowDerivatives> bws_vec =
        tulpa_thread_workspaces<NNGPRowDerivatives>(n_threads, nn);
    auto hyper_row = [&](int i, NNGPRowDerivatives& derivatives, double& acc_lik,
                         double& acc_jac, double& acc_s2) {
        const int loc = view.nn_order[i], count = ws.B_n_nb[i];
        if (count == 0 || loc < 0 || loc >= N) return;
        nngp_row_derivatives(i, sigma2, phi, view, ws, derivatives);
        double ds = 0.0, dp = 0.0;
        for (int j = 0; j < count; ++j) {
            const double field = ws.w[ws.nb_idx_flat[i * nn + j]];
            ds += derivatives.d_alpha_sigma(j) * field;
            dp += derivatives.d_alpha_phi(j) * field;
        }
        acc_s2 += adj[i] * ds;
        const double sd = ws.sqrt_d[i];
        if (sd > 1e-15) dp += derivatives.d_variance_phi / (2.0 * sd) * z[loc];
        acc_lik += adj[i] * dp * phi;
        if (sd * sd > 1e-15)
            acc_jac += 0.5 * derivatives.d_variance_phi / (sd * sd) * phi;
    };
    // At a team of one the region is skipped entirely: entering libgomp costs
    // measurable time per reverse sweep, and this runs on every HMC gradient.
    // One thread means every row accumulates into slot 0 in index order, which
    // is exactly what the plain loop does, so the two routes are bit-identical.
    //
    // Above one thread the rows are cut into `n_threads` contiguous chunks HERE,
    // by index arithmetic, each chunk accumulated left to right into its own
    // slot and the slots added in chunk order after the region. Letting the
    // runtime hand rows out -- a schedule(dynamic) loop writing into the slot
    // omp_get_thread_num() names -- makes the summation order a property of the
    // run rather than of the problem, and floating-point addition is not
    // associative, so the last bits of the phi gradient move between two runs of
    // the SAME fit and a fixed seed stops reproducing a field's draws. Chunking
    // makes the pair a function of (n_threads, N) alone. It is still not the
    // one-thread sum: chunking imposes its own association. Same policy as
    // tulpa_parallel_sum in omp_threads.h.
    #ifdef _OPENMP
    if (n_threads > 1) {
        const long long n_rows = static_cast<long long>(N - 1);
        const long long teams  = static_cast<long long>(n_threads);
        #pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int t = 0; t < n_threads; t++) {
            const int lo = 1 + static_cast<int>(n_rows * t / teams);
            const int hi = 1 + static_cast<int>(n_rows * (t + 1) / teams);
            for (int i = lo; i < hi; i++) {
                hyper_row(i, bws_vec[t], tl_phi_lik[t], tl_phi_jac[t],
                          tl_s2_alpha[t]);
            }
        }
    } else
    #endif
    {
        for (int i = 1; i < N; i++) {
            hyper_row(i, bws_vec[0], tl_phi_lik[0], tl_phi_jac[0],
                      tl_s2_alpha[0]);
        }
    }

    // Reduce thread-local accumulators
    for (int t = 0; t < n_threads; t++) {
        grad_log_phi_lik    += tl_phi_lik[t];
        grad_log_phi_jac    += tl_phi_jac[t];
        grad_log_sigma2_lik += tl_s2_alpha[t];
    }
}
