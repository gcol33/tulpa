// =========================================================================
// Laplace log-det correction via sparse Cholesky
// =========================================================================

// Build W+Q as Eigen sparse matrix and compute log det via Cholesky.
// W = diag(hess_diag), Q = (I-B)^T D^{-1} (I-B)
// Returns -0.5 * log det(W + Q) (the Laplace correction term)
inline double compute_laplace_log_det(CollapsedGPWorkspace& ws) {
    int N = ws.N_gp;
    int nn = ws.nn;

    typedef Eigen::Triplet<double> T;
    std::vector<T> triplets;
    triplets.reserve(N * (2 * nn + 1));

    // Build Q + W as sparse matrix via outer products of (I-B) rows
    for (int k = 0; k < N; k++) {
        double inv_dk = 1.0 / ws.d_cond[k];
        int n_nb_k = ws.n_nb[k];
        int n_entries = 1 + n_nb_k;
        int cols[32];
        double vals[32];
        cols[0] = k;
        vals[0] = 1.0;
        for (int j = 0; j < n_nb_k; j++) {
            cols[1 + j] = ws.nb_idx_flat[k * nn + j];
            vals[1 + j] = -ws.B_flat[k * nn + j];
        }
        for (int a = 0; a < n_entries; a++) {
            for (int b = a; b < n_entries; b++) {
                double v = inv_dk * vals[a] * vals[b];
                triplets.push_back(T(cols[a], cols[b], v));
                if (a != b) {
                    triplets.push_back(T(cols[b], cols[a], v));
                }
            }
        }
    }

    // Add W diagonal
    for (int i = 0; i < N; i++) {
        triplets.push_back(T(i, i, ws.hess_diag[i]));
    }

    // Build sparse matrix and compute Cholesky (all local, no thread_local Eigen)
    Eigen::SparseMatrix<double> WpQ(N, N);
    WpQ.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt;
    llt.compute(WpQ);
    if (llt.info() != Eigen::Success) {
        // Add jitter and retry
        for (int i = 0; i < N; i++) {
            triplets.push_back(T(i, i, 1e-6));
        }
        WpQ.setFromTriplets(triplets.begin(), triplets.end());
        llt.compute(WpQ);
        if (llt.info() != Eigen::Success) {
            ws.laplace_log_det = 0.0;
            ws.H_inv_diag.assign(N, 0.0);
            return 0.0;
        }
    }

    // log det(W+Q) = 2 * sum(log(diag(L)))
    Eigen::SparseMatrix<double> L_sparse = llt.matrixL();
    double log_det = 0.0;
    for (int i = 0; i < N; i++) {
        log_det += std::log(L_sparse.coeff(i, i));
    }
    log_det *= 2.0;

    ws.laplace_log_det = -0.5 * log_det;

    // ---- Takahashi recursion: compute diagonal of (W+Q)^{-1} ----
    // The Cholesky uses fill-reducing permutation P: P*(W+Q)*P^T = L*L^T
    // We compute Z = (L*L^T)^{-1} at L's sparsity pattern, then extract diag.
    ws.H_inv_diag.resize(N);

    const int* outerPtr = L_sparse.outerIndexPtr();
    const int* innerIdx = L_sparse.innerIndexPtr();
    const double* vals_L = L_sparse.valuePtr();
    int nnz_L = L_sparse.nonZeros();

    // Z values stored at same positions as L's nonzeros
    std::vector<double> Z_vals(nnz_L, 0.0);

    // Fix 3: Binary search helpers (Eigen CSC columns have sorted inner indices)
    auto bsearch_idx = [&](int col, int row) -> int {
        int lo = outerPtr[col], hi = outerPtr[col + 1];
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (innerIdx[mid] < row) lo = mid + 1;
            else if (innerIdx[mid] > row) hi = mid;
            else return mid;
        }
        return -1;
    };

    auto find_Z = [&](int row, int col) -> double* {
        int idx = bsearch_idx(col, row);
        return (idx >= 0) ? &Z_vals[idx] : nullptr;
    };

    auto get_L_val = [&](int row, int col) -> double {
        int idx = bsearch_idx(col, row);
        return (idx >= 0) ? vals_L[idx] : 0.0;
    };

    // Pre-build column-level children lists to avoid repeated scanning
    // children[j] = list of (row k, L_kj) for k > j in column j
    struct Child { int k; double L_kj; int idx; };
    std::vector<std::vector<Child>> col_children(N);
    for (int j = 0; j < N; j++) {
        for (int idx = outerPtr[j]; idx < outerPtr[j + 1]; idx++) {
            if (innerIdx[idx] > j) {
                col_children[j].push_back({innerIdx[idx], vals_L[idx], idx});
            }
        }
    }

    // Pre-build parent list: for each column i, which columns j > i have L(j,i) != 0?
    std::vector<std::vector<int>> col_parents(N);
    for (int i = 0; i < N; i++) {
        for (int idx = outerPtr[i]; idx < outerPtr[i + 1]; idx++) {
            int row = innerIdx[idx];
            if (row > i) col_parents[i].push_back(row);
        }
    }

    // Process columns from right to left
    for (int j = N - 1; j >= 0; j--) {
        double L_jj = get_L_val(j, j);
        if (std::abs(L_jj) < 1e-15) L_jj = 1e-10;
        double inv_Ljj = 1.0 / L_jj;

        const auto& children = col_children[j];

        // Z_jj = 1/L_jj^2 - (1/L_jj) * sum_k L_kj * Z_kj
        double sum_diag = 0.0;
        for (const auto& c : children) {
            double* Z_kj = find_Z(c.k, j);
            if (Z_kj) sum_diag += c.L_kj * (*Z_kj);
        }
        double Z_jj = inv_Ljj * inv_Ljj - inv_Ljj * sum_diag;
        double* Z_jj_ptr = find_Z(j, j);
        if (Z_jj_ptr) *Z_jj_ptr = Z_jj;

        // Off-diagonal: only columns i < j that have L(j,i) != 0
        // Use pre-built parent list for column j's parents (actually iterate col j's structure)
        for (int i = 0; i < j; i++) {
            // Binary search for L(j,i) in column i
            int ji_idx = bsearch_idx(i, j);
            if (ji_idx < 0) continue;

            // Z_ji = -(1/L_jj) * sum_k L_kj * Z_ik
            double sum_off = 0.0;
            for (const auto& c : children) {
                double Z_ik = 0.0;
                if (c.k > i) {
                    double* ptr = find_Z(c.k, i);
                    if (ptr) Z_ik = *ptr;
                } else if (c.k < i) {
                    double* ptr = find_Z(i, c.k);
                    if (ptr) Z_ik = *ptr;
                } else {
                    double* ptr = find_Z(i, i);
                    if (ptr) Z_ik = *ptr;
                }
                sum_off += c.L_kj * Z_ik;
            }
            double Z_ji = -inv_Ljj * sum_off;
            double* Z_ji_ptr = find_Z(j, i);
            if (Z_ji_ptr) *Z_ji_ptr = Z_ji;
        }
    }

    // Map from permuted to original ordering
    const auto& P = llt.permutationP();
    for (int i = 0; i < N; i++) {
        int pi = P.indices()[i];
        double* Z_pi = find_Z(pi, pi);
        ws.H_inv_diag[i] = Z_pi ? *Z_pi : 0.0;
    }

    return ws.laplace_log_det;
}

// =========================================================================
// Data likelihood helpers (per-location, aggregated over observations)
// =========================================================================

// Compute data log-likelihood gradient and negative Hessian w.r.t. w[loc]
// Aggregates over all observations at location loc
struct LocLikResult {
    double ll;          // log-likelihood contribution
    double grad;        // d(ll)/d(w[loc])
    double neg_hess;    // -d²(ll)/d(w[loc])²
};

inline LocLikResult compute_loc_lik(
    int loc,
    const double* w,
    const double* beta_num, const double* beta_denom,
    double phi_num, double phi_denom,
    const ModelData& data,
    bool is_binomial,
    const int* obs_list = nullptr,  // pre-filtered obs at this location
    int n_obs = -1                  // number of obs (-1 = scan all)
) {
    LocLikResult res = {0.0, 0.0, 0.0};
    int N_iter = (n_obs >= 0) ? n_obs : data.N;

    // DEBUG: per-obs NaN tracking
    static thread_local int loc_lik_debug = 0;

    for (int idx = 0; idx < N_iter; idx++) {
        int i = (obs_list != nullptr) ? obs_list[idx] : idx;
        if (obs_list == nullptr && data.gp_data.obs_to_loc[i] != loc) continue;

        // Compute eta
        double eta_num_i = 0.0, eta_denom_i = 0.0;
        for (int p = 0; p < data.legacy.p_num; p++)
            eta_num_i += data.legacy.X_num_flat[i * data.legacy.p_num + p] * beta_num[p];
        if (!is_binomial) {
            for (int p = 0; p < data.legacy.p_denom; p++)
                eta_denom_i += data.legacy.X_denom_flat[i * data.legacy.p_denom + p] * beta_denom[p];
        }

        // RE effects are in the outer HMC params, not in the collapsed GP
        // They are added to eta via the outer gradient function

        eta_num_i += w[loc];
        if (!is_binomial && data.gp_data.shared) eta_denom_i += w[loc];

        // Per-family likelihood, gradient, Hessian
        double mu_num = std::exp(eta_num_i);
        int y_num = data.legacy.y_num[i];

        // DEBUG: print first obs details if NaN
        if (loc_lik_debug < 3 && data.N >= 100) {
            double y_dc = (data.legacy.y_denom_cont.size() > (size_t)i) ? data.legacy.y_denom_cont[i] : -999.0;
            int y_d = (data.legacy.y_denom.size() > (size_t)i) ? data.legacy.y_denom[i] : -999;
            Rprintf("[LOC_LIK] i=%d loc=%d eta_num=%.4f eta_den=%.4f mu_num=%.4f y_num=%d "
                    "y_denom=%d y_denom_cont=%.6f phi_num=%.4f phi_den=%.4f shared=%d model=%d\n",
                    i, loc, eta_num_i, eta_denom_i, mu_num, y_num,
                    y_d, y_dc, phi_num, phi_denom,
                    (int)data.gp_data.shared, (int)data.legacy.model_type);
            // Print beta and X values
            Rprintf("  p_num=%d p_denom=%d X_flat_size=%d w[%d]=%.6f\n",
                    data.legacy.p_num, data.legacy.p_denom, (int)data.legacy.X_num_flat.size(), loc, w[loc]);
            // Recompute eta from scratch
            double eta_check = 0.0;
            for (int p = 0; p < std::min(data.legacy.p_num, 4); p++) {
                int idx = i * data.legacy.p_num + p;
                double x_val = (idx < (int)data.legacy.X_num_flat.size()) ? data.legacy.X_num_flat[idx] : -9999.0;
                eta_check += x_val * beta_num[p];
                Rprintf("  beta_num[%d]=%.6f X_num[%d,%d]=%.6f partial_eta=%.6f\n",
                        p, beta_num[p], i, p, x_val, eta_check);
            }
            eta_check += w[loc];
            Rprintf("  eta_check=%.6f (should match eta_num=%.6f)\n", eta_check, eta_num_i);
            loc_lik_debug++;
        }

        switch (data.legacy.model_type) {
            case ModelType::POISSON_GAMMA: {
                // Numerator: Poisson
                res.ll += y_num * eta_num_i - mu_num;
                double resid_num = y_num - mu_num;
                res.grad += resid_num;
                res.neg_hess += mu_num;

                // Denominator: Gamma (log link, continuous y_denom)
                if (data.gp_data.shared) {
                    double mu_denom = std::exp(eta_denom_i);
                    double y_denom = data.legacy.y_denom_cont[i];
                    double shape = phi_denom;
                    double resid_denom = shape * (y_denom / mu_denom - 1.0);
                    res.ll += shape * std::log(shape) - std::lgamma(shape)
                              + (shape - 1.0) * std::log(y_denom)
                              - shape * eta_denom_i - shape * y_denom / mu_denom;
                    res.grad += resid_denom;
                    res.neg_hess += shape * y_denom / mu_denom;
                }
                break;
            }
            case ModelType::NEGBIN_NEGBIN: {
                // Numerator: NegBin
                double r_num = phi_num;
                res.ll += std::lgamma(y_num + r_num) - std::lgamma(r_num) - std::lgamma(y_num + 1)
                          + y_num * eta_num_i - (y_num + r_num) * std::log(mu_num + r_num)
                          + r_num * std::log(r_num);
                double resid_num = y_num - mu_num * (y_num + r_num) / (mu_num + r_num);
                res.grad += resid_num;
                res.neg_hess += mu_num * r_num * (y_num + r_num) / ((mu_num + r_num) * (mu_num + r_num));

                // Denominator: NegBin (shared GP)
                if (data.gp_data.shared) {
                    double mu_denom = std::exp(eta_denom_i);
                    int y_denom = (int)data.legacy.y_denom[i];
                    double r_denom = phi_denom;
                    res.ll += std::lgamma(y_denom + r_denom) - std::lgamma(r_denom) - std::lgamma(y_denom + 1)
                              + y_denom * eta_denom_i - (y_denom + r_denom) * std::log(mu_denom + r_denom)
                              + r_denom * std::log(r_denom);
                    double resid_denom = y_denom - mu_denom * (y_denom + r_denom) / (mu_denom + r_denom);
                    res.grad += resid_denom;
                    res.neg_hess += mu_denom * r_denom * (y_denom + r_denom) / ((mu_denom + r_denom) * (mu_denom + r_denom));
                }
                break;
            }
            case ModelType::BINOMIAL: {
                int n_trials = (int)data.legacy.y_denom[i];
                double p_i = 1.0 / (1.0 + std::exp(-eta_num_i));
                res.ll += y_num * eta_num_i - n_trials * std::log(1.0 + std::exp(eta_num_i));
                res.grad += y_num - n_trials * p_i;
                res.neg_hess += n_trials * p_i * (1.0 - p_i);
                break;
            }
            default:
                // Fallback: Poisson-like
                res.ll += y_num * eta_num_i - mu_num;
                res.grad += y_num - mu_num;
                res.neg_hess += mu_num;
                break;
        }
    }
    return res;
}

// Compute Laplace log-det for given params by finding mode and computing log det(W+Q).
// Used for numerical differentiation of the Laplace correction w.r.t. all params.
// warm_w: initial guess for Newton (warm start from nearby params)
// If base_ws is non-null and rebuild_nngp is false, reuses existing NNGP structure
// (valid when only non-GP params like beta/phi change, not sigma2/phi_gp).
inline double laplace_log_det_full(
    const double* beta_num, const double* beta_denom,
    double sigma2, double phi,
    double phi_num, double phi_denom,
    const ModelData& data,
    const std::vector<double>& warm_w,  // warm start for Newton
    const CollapsedGPWorkspace* base_ws = nullptr,  // existing NNGP to reuse
    bool rebuild_nngp = true
) {
    CollapsedGPWorkspace temp_ws;
    bool is_binomial = (data.legacy.model_type == ModelType::BINOMIAL ||
                        data.legacy.model_type == ModelType::BETA_BINOMIAL);

    if (!rebuild_nngp && base_ws != nullptr) {
        // Reuse existing NNGP structure (B, D, reverse index)
        temp_ws.N_gp = base_ws->N_gp;
        temp_ws.nn = base_ws->nn;
        temp_ws.B_flat = base_ws->B_flat;
        temp_ws.n_nb = base_ws->n_nb;
        temp_ws.nb_idx_flat = base_ws->nb_idx_flat;
        temp_ws.d_cond = base_ws->d_cond;
        temp_ws.rev_ptr = base_ws->rev_ptr;
        temp_ws.rev_col = base_ws->rev_col;
        temp_ws.rev_nb_pos = base_ws->rev_nb_pos;
        temp_ws.structure_built = true;
        // Allocate Newton workspace
        int n = temp_ws.N_gp;
        temp_ws.w_star.resize(n);
        temp_ws.grad_w.resize(n);
        temp_ws.hess_diag.resize(n);
        temp_ws.newton_rhs.resize(n);
        temp_ws.cg_r.resize(n);
        temp_ws.cg_p.resize(n);
        temp_ws.cg_Ap.resize(n);
    } else {
        build_nngp_B_D(sigma2, phi, data.gp_data, temp_ws);
    }

    // Build loc→obs mapping for fast per-location iteration
    build_loc_obs_map(temp_ws, data);

    // Warm-start from provided w
    temp_ws.w_star = warm_w;
    temp_ws.mode_found = true;

    // Run Newton (should converge in 1-2 iterations from warm start)
    int N_gp = temp_ws.N_gp;
    std::vector<double> Qw(N_gp);
    std::vector<double> delta_w(N_gp, 0.0);

    for (int newton_iter = 0; newton_iter < 5; newton_iter++) {
        std::fill(temp_ws.grad_w.begin(), temp_ws.grad_w.end(), 0.0);
        std::fill(temp_ws.hess_diag.begin(), temp_ws.hess_diag.end(), 0.0);

        for (int loc = 0; loc < N_gp; loc++) {
            LocLikResult lr = compute_loc_lik(loc, temp_ws.w_star.data(),
                                               beta_num, beta_denom,
                                               phi_num, phi_denom, data, is_binomial);
            temp_ws.grad_w[loc] = lr.grad;
            temp_ws.hess_diag[loc] = std::max(lr.neg_hess, 1e-8);
        }

        nngp_precision_matvec(temp_ws.w_star.data(), Qw.data(), temp_ws);
        double grad_norm = 0.0;
        for (int i = 0; i < N_gp; i++) {
            temp_ws.grad_w[i] -= Qw[i];
            grad_norm += temp_ws.grad_w[i] * temp_ws.grad_w[i];
        }
        if (std::sqrt(grad_norm) < 1e-6) break;

        std::fill(delta_w.begin(), delta_w.end(), 0.0);
        cg_solve(delta_w.data(), temp_ws.grad_w.data(), temp_ws.hess_diag.data(), temp_ws, 50, 1e-8);
        for (int i = 0; i < N_gp; i++) temp_ws.w_star[i] += delta_w[i];
    }

    // Final hess_diag at the new mode
    std::fill(temp_ws.hess_diag.begin(), temp_ws.hess_diag.end(), 0.0);
    for (int loc = 0; loc < N_gp; loc++) {
        LocLikResult lr = compute_loc_lik(loc, temp_ws.w_star.data(),
                                           beta_num, beta_denom,
                                           phi_num, phi_denom, data, is_binomial);
        temp_ws.hess_diag[loc] = std::max(lr.neg_hess, 1e-8);
    }

    return compute_laplace_log_det(temp_ws);
}
