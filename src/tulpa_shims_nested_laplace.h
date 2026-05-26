// ============================================================================
// Nested-Laplace shims
// ============================================================================
//
// Each shim packs raw pointers into Rcpp::* on entry, calls the matching
// cpp_nested_laplace_<backend> wrapper, and unpacks the universal output
// block (log_marginal[n_grid], n_iter[n_grid], optional modes[n_grid x n_x])
// into the caller's NestedLaplaceShimResult. Backend-specific grid vectors
// are passed in by the caller and not echoed back through the shim.

namespace {

inline Rcpp::Nullable<Rcpp::NumericVector> wrap_x_init(
    const double* x_init, int n_x_init
) {
    if (x_init && n_x_init > 0) {
        return Rcpp::wrap(Rcpp::NumericVector(x_init, x_init + n_x_init));
    }
    return R_NilValue;
}

// Wrap an optional temporal rho grid (ar1) for the cpp entry. nullptr / empty
// -> R_NilValue, matching the Nullable<NumericVector> rho_temporal_grid arg.
inline Rcpp::Nullable<Rcpp::NumericVector> nl_wrap_rho_temporal(
    const double* rho, int n
) {
    if (rho && n > 0) {
        return Rcpp::wrap(Rcpp::NumericVector(rho, rho + n));
    }
    return R_NilValue;
}

// Copy the universal output block out of a nested-Laplace result list.
inline void copy_nested_laplace_result(
    const Rcpp::List& out,
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector lm = out["log_marginal"];
    Rcpp::IntegerVector ni = out["n_iter"];
    int n_grid = (int)lm.size();

    result_out->n_grid = n_grid;
    result_out->log_marginal = new double[n_grid];
    result_out->n_iter       = new int[n_grid];
    for (int k = 0; k < n_grid; k++) {
        result_out->log_marginal[k] = lm[k];
        result_out->n_iter[k]       = ni[k];
    }

    bool has_modes = out.containsElementNamed("modes");
    if (has_modes) {
        Rcpp::NumericMatrix modes = out["modes"];
        int n_x = modes.ncol();
        result_out->store_modes = 1;
        result_out->n_x = n_x;
        result_out->modes = new double[(size_t)n_grid * (size_t)n_x];
        // modes is column-major in Rcpp; emit row-major in the shim buffer
        // so callers can index modes[k*n_x + j] without thinking about it.
        for (int k = 0; k < n_grid; k++) {
            for (int j = 0; j < n_x; j++) {
                result_out->modes[(size_t)k * n_x + j] = modes(k, j);
            }
        }
    } else {
        result_out->store_modes = 0;
        result_out->n_x = 0;
        result_out->modes = nullptr;
    }

    // Per-grid Q (Tulpa ABI v5+). Present only when the underlying
    // cpp_nested_laplace_<backend> returned with store_Q = true.
    bool has_Q = out.containsElementNamed("Q_csc_p_per_grid")
              && out.containsElementNamed("Q_csc_i_per_grid")
              && out.containsElementNamed("Q_csc_x_per_grid");
    if (has_Q) {
        Rcpp::List Qp_list = out["Q_csc_p_per_grid"];
        Rcpp::List Qi_list = out["Q_csc_i_per_grid"];
        Rcpp::List Qx_list = out["Q_csc_x_per_grid"];
        int Q_n = Rcpp::as<int>(out["Q_csc_n"]);

        result_out->store_Q = 1;
        result_out->Q_n     = Q_n;
        result_out->Q_grid_nnz  = new int[n_grid];
        result_out->Q_p_offsets = new int[n_grid + 1];
        result_out->Q_x_offsets = new int[n_grid + 1];

        // First pass: collect per-grid nnz and build offset tables.
        result_out->Q_p_offsets[0] = 0;
        result_out->Q_x_offsets[0] = 0;
        for (int k = 0; k < n_grid; k++) {
            Rcpp::NumericVector xv = Qx_list[k];
            int nnz_k = (int)xv.size();
            result_out->Q_grid_nnz[k]      = nnz_k;
            result_out->Q_p_offsets[k + 1] = (k + 1) * (Q_n + 1);
            result_out->Q_x_offsets[k + 1] = result_out->Q_x_offsets[k] + nnz_k;
        }

        int p_total = result_out->Q_p_offsets[n_grid];
        int x_total = result_out->Q_x_offsets[n_grid];
        result_out->Q_p_flat = new int[p_total];
        result_out->Q_i_flat = new int[x_total];
        result_out->Q_x_flat = new double[x_total];

        // Second pass: copy each per-grid CSC block into the flat buffers.
        for (int k = 0; k < n_grid; k++) {
            Rcpp::IntegerVector pv = Qp_list[k];
            Rcpp::IntegerVector iv = Qi_list[k];
            Rcpp::NumericVector xv = Qx_list[k];

            int p_off = result_out->Q_p_offsets[k];
            int x_off = result_out->Q_x_offsets[k];
            for (int j = 0; j <= Q_n; j++) result_out->Q_p_flat[p_off + j] = pv[j];
            for (int e = 0; e < (int)iv.size(); e++) {
                result_out->Q_i_flat[x_off + e] = iv[e];
                result_out->Q_x_flat[x_off + e] = xv[e];
            }
        }
    } else {
        result_out->store_Q     = 0;
        result_out->Q_n         = 0;
        result_out->Q_grid_nnz  = nullptr;
        result_out->Q_p_offsets = nullptr;
        result_out->Q_x_offsets = nullptr;
        result_out->Q_p_flat    = nullptr;
        result_out->Q_i_flat    = nullptr;
        result_out->Q_x_flat    = nullptr;
    }
}

} // namespace

extern "C" void tulpa_nested_laplace_icar_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* spatial_idx, int n_spatial_units,
    const int* adj_row_ptr, const int* adj_col_idx, const int* n_neighbors,
    const double* tau_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    int store_Q,
    tulpa::NestedLaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);
    Rcpp::NumericVector  tg (tau_grid,    tau_grid + n_grid);

    Rcpp::List out = cpp_nested_laplace_icar(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn,
        tg, in.fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init),
        store_Q != 0
    );
    copy_nested_laplace_result(out, result_out);
}

extern "C" void tulpa_nested_laplace_bym2_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* spatial_idx, int n_spatial_units,
    const int* adj_row_ptr, const int* adj_col_idx, const int* n_neighbors,
    double scale_factor,
    const double* sigma_spatial_grid, const double* rho_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    int store_Q,
    tulpa::NestedLaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);
    Rcpp::NumericVector  sg (sigma_spatial_grid, sigma_spatial_grid + n_grid);
    Rcpp::NumericVector  rg (rho_grid,           rho_grid           + n_grid);

    Rcpp::List out = cpp_nested_laplace_bym2(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn,
        scale_factor, sg, rg,
        in.fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init),
        store_Q != 0
    );
    copy_nested_laplace_result(out, result_out);
}

extern "C" void tulpa_nested_laplace_car_proper_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* spatial_idx, int n_spatial_units,
    const int* adj_row_ptr, const int* adj_col_idx, const int* n_neighbors,
    const double* tau_grid, const double* rho_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    int store_Q,
    tulpa::NestedLaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);
    Rcpp::NumericVector  tg (tau_grid, tau_grid + n_grid);
    Rcpp::NumericVector  rg (rho_grid, rho_grid + n_grid);

    Rcpp::List out = cpp_nested_laplace_car_proper(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn,
        tg, rg, in.fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init),
        store_Q != 0
    );
    copy_nested_laplace_result(out, result_out);
}

extern "C" void tulpa_nested_laplace_rw1_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* temporal_idx, int n_times, int cyclic,
    const double* tau_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    int store_Q,
    tulpa::NestedLaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::IntegerVector  tv(temporal_idx, temporal_idx + N);
    Rcpp::NumericVector  tg(tau_grid, tau_grid + n_grid);

    Rcpp::List out = cpp_nested_laplace_rw1(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        tv, n_times, (cyclic != 0),
        tg, in.fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init),
        store_Q != 0
    );
    copy_nested_laplace_result(out, result_out);
}

extern "C" void tulpa_nested_laplace_rw2_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* temporal_idx, int n_times,
    const double* tau_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    int store_Q,
    tulpa::NestedLaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::IntegerVector  tv(temporal_idx, temporal_idx + N);
    Rcpp::NumericVector  tg(tau_grid, tau_grid + n_grid);

    Rcpp::List out = cpp_nested_laplace_rw2(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        tv, n_times,
        tg, in.fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init),
        store_Q != 0
    );
    copy_nested_laplace_result(out, result_out);
}

extern "C" void tulpa_nested_laplace_ar1_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* temporal_idx, int n_times,
    const double* tau_grid, const double* rho_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    int store_Q,
    tulpa::NestedLaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::IntegerVector  tv(temporal_idx, temporal_idx + N);
    Rcpp::NumericVector  tg(tau_grid, tau_grid + n_grid);
    Rcpp::NumericVector  rg(rho_grid, rho_grid + n_grid);

    Rcpp::List out = cpp_nested_laplace_ar1(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        tv, n_times, tg, rg,
        in.fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init),
        store_Q != 0
    );
    copy_nested_laplace_result(out, result_out);
}

extern "C" void tulpa_nested_laplace_nngp_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* spatial_idx,
    const double* coords_flat, int coord_dim,
    const int* nn_idx_flat, const double* nn_dist_flat,
    const int* nn_order,
    int n_spatial, int nn,
    const double* sigma2_grid, const double* phi_gp_grid, int n_grid,
    int cov_type,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    int store_Q,
    tulpa::NestedLaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::IntegerVector  sidx(spatial_idx, spatial_idx + N);
    Rcpp::NumericMatrix  cm = build_matrix_colmajor(coords_flat, n_spatial, coord_dim);
    Rcpp::IntegerMatrix  nim = build_int_matrix_colmajor(nn_idx_flat, n_spatial, nn);
    Rcpp::NumericMatrix  ndm = build_matrix_colmajor(nn_dist_flat, n_spatial, nn);
    Rcpp::IntegerVector  nord(nn_order, nn_order + n_spatial);
    Rcpp::NumericVector  s2g(sigma2_grid,  sigma2_grid  + n_grid);
    Rcpp::NumericVector  phg(phi_gp_grid,  phi_gp_grid  + n_grid);

    Rcpp::List out = cpp_nested_laplace_nngp(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        sidx, cm, nim, ndm, nord, n_spatial, nn,
        s2g, phg, cov_type,
        in.fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init),
        store_Q != 0
    );
    copy_nested_laplace_result(out, result_out);
}

extern "C" void tulpa_nested_laplace_hsgp_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const double* phi_basis_flat, int n_basis,
    const double* lambda_eig,
    const double* sigma2_grid, const double* lengthscale_grid, int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    int store_Q,
    tulpa::NestedLaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::NumericMatrix  pb = build_matrix_colmajor(phi_basis_flat, N, n_basis);
    Rcpp::NumericVector  le(lambda_eig, lambda_eig + n_basis);
    Rcpp::NumericVector  s2g(sigma2_grid,      sigma2_grid      + n_grid);
    Rcpp::NumericVector  lsg(lengthscale_grid, lengthscale_grid + n_grid);

    Rcpp::List out = cpp_nested_laplace_hsgp(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        pb, le, s2g, lsg,
        in.fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init),
        store_Q != 0
    );
    copy_nested_laplace_result(out, result_out);
}

// ============================================================================
// Spatial × temporal nested-Laplace shims
// ============================================================================
// One extern-C shim per spatial family; the temporal kernel is selected at
// runtime via `temporal_type` ("rw1" / "rw2" / "ar1"). rho_temporal_grid is
// non-null only for ar1; cyclic applies only to rw1. Joint inner Newton over
// [beta] [re] [w_spatial] [w_temporal (n_t)]. Adding a temporal kernel needs
// no new shim -- it routes through the same five entries.

extern "C" void tulpa_nested_laplace_st_icar_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* spatial_idx, int n_spatial_units,
    const int* adj_row_ptr, const int* adj_col_idx, const int* n_neighbors,
    const int* temporal_idx, int n_times,
    const double* tau_spatial_grid,
    const char* temporal_type,
    const double* tau_temporal_grid, const double* rho_temporal_grid, int cyclic,
    int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    int store_Q,
    tulpa::NestedLaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);
    Rcpp::IntegerVector  tv (temporal_idx,     temporal_idx     + N);
    Rcpp::NumericVector  tsg(tau_spatial_grid, tau_spatial_grid + n_grid);
    Rcpp::NumericVector  ttg(tau_temporal_grid, tau_temporal_grid + n_grid);
    Rcpp::Nullable<Rcpp::NumericVector> rtg = nl_wrap_rho_temporal(rho_temporal_grid, n_grid);

    Rcpp::List out = cpp_nested_laplace_st_icar(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn,
        tv, n_times,
        tsg,
        std::string(temporal_type), ttg, rtg, (cyclic != 0),
        in.fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init),
        store_Q != 0
    );
    copy_nested_laplace_result(out, result_out);
}

// ---- CAR_proper ------------------------------------------------------------
extern "C" void tulpa_nested_laplace_st_car_proper_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* spatial_idx, int n_spatial_units,
    const int* adj_row_ptr, const int* adj_col_idx, const int* n_neighbors,
    const int* temporal_idx, int n_times,
    const double* tau_spatial_grid, const double* rho_spatial_grid,
    const char* temporal_type,
    const double* tau_temporal_grid, const double* rho_temporal_grid, int cyclic,
    int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    int store_Q,
    tulpa::NestedLaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);
    Rcpp::IntegerVector  tv (temporal_idx,     temporal_idx     + N);
    Rcpp::NumericVector  tsg(tau_spatial_grid, tau_spatial_grid + n_grid);
    Rcpp::NumericVector  rsg(rho_spatial_grid, rho_spatial_grid + n_grid);
    Rcpp::NumericVector  ttg(tau_temporal_grid, tau_temporal_grid + n_grid);
    Rcpp::Nullable<Rcpp::NumericVector> rtg = nl_wrap_rho_temporal(rho_temporal_grid, n_grid);

    Rcpp::List out = cpp_nested_laplace_st_car_proper(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn,
        tv, n_times,
        tsg, rsg,
        std::string(temporal_type), ttg, rtg, (cyclic != 0),
        in.fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init),
        store_Q != 0
    );
    copy_nested_laplace_result(out, result_out);
}

// ---- BYM2 ------------------------------------------------------------------
extern "C" void tulpa_nested_laplace_st_bym2_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* spatial_idx, int n_spatial_units,
    const int* adj_row_ptr, const int* adj_col_idx, const int* n_neighbors,
    double scale_factor,
    const int* temporal_idx, int n_times,
    const double* sigma_spatial_grid, const double* rho_spatial_grid,
    const char* temporal_type,
    const double* tau_temporal_grid, const double* rho_temporal_grid, int cyclic,
    int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    int store_Q,
    tulpa::NestedLaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);
    Rcpp::IntegerVector  tv (temporal_idx,       temporal_idx       + N);
    Rcpp::NumericVector  ssg(sigma_spatial_grid, sigma_spatial_grid + n_grid);
    Rcpp::NumericVector  rsg(rho_spatial_grid,   rho_spatial_grid   + n_grid);
    Rcpp::NumericVector  ttg(tau_temporal_grid,  tau_temporal_grid  + n_grid);
    Rcpp::Nullable<Rcpp::NumericVector> rtg = nl_wrap_rho_temporal(rho_temporal_grid, n_grid);

    Rcpp::List out = cpp_nested_laplace_st_bym2(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn,
        scale_factor,
        tv, n_times,
        ssg, rsg,
        std::string(temporal_type), ttg, rtg, (cyclic != 0),
        in.fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init),
        store_Q != 0
    );
    copy_nested_laplace_result(out, result_out);
}

// ---- HSGP ------------------------------------------------------------------
extern "C" void tulpa_nested_laplace_st_hsgp_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const double* phi_basis_flat, int n_basis,
    const double* lambda_eig,
    const int* temporal_idx, int n_times,
    const double* sigma2_spatial_grid, const double* lengthscale_spatial_grid,
    const char* temporal_type,
    const double* tau_temporal_grid, const double* rho_temporal_grid, int cyclic,
    int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    int store_Q,
    tulpa::NestedLaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::NumericMatrix  pb = build_matrix_colmajor(phi_basis_flat, N, n_basis);
    Rcpp::NumericVector  le (lambda_eig, lambda_eig + n_basis);
    Rcpp::IntegerVector  tv (temporal_idx, temporal_idx + N);
    Rcpp::NumericVector  s2g(sigma2_spatial_grid,      sigma2_spatial_grid      + n_grid);
    Rcpp::NumericVector  lsg(lengthscale_spatial_grid, lengthscale_spatial_grid + n_grid);
    Rcpp::NumericVector  ttg(tau_temporal_grid,        tau_temporal_grid        + n_grid);
    Rcpp::Nullable<Rcpp::NumericVector> rtg = nl_wrap_rho_temporal(rho_temporal_grid, n_grid);

    Rcpp::List out = cpp_nested_laplace_st_hsgp(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        pb, le,
        tv, n_times,
        s2g, lsg,
        std::string(temporal_type), ttg, rtg, (cyclic != 0),
        in.fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init),
        store_Q != 0
    );
    copy_nested_laplace_result(out, result_out);
}

// ---- NNGP ------------------------------------------------------------------
extern "C" void tulpa_nested_laplace_st_nngp_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const int* spatial_idx, int n_spatial,
    const double* coords_flat, int coord_dim,
    const int* nn_idx_flat, const double* nn_dist_flat,
    const int* nn_order, int nn, int cov_type,
    const int* temporal_idx, int n_times,
    const double* sigma2_spatial_grid, const double* phi_gp_spatial_grid,
    const char* temporal_type,
    const double* tau_temporal_grid, const double* rho_temporal_grid, int cyclic,
    int n_grid,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    int store_Q,
    tulpa::NestedLaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::IntegerVector  sidx(spatial_idx, spatial_idx + N);
    Rcpp::NumericMatrix  cm  = build_matrix_colmajor(coords_flat, n_spatial, coord_dim);
    Rcpp::IntegerMatrix  nim = build_int_matrix_colmajor(nn_idx_flat, n_spatial, nn);
    Rcpp::NumericMatrix  ndm = build_matrix_colmajor(nn_dist_flat, n_spatial, nn);
    Rcpp::IntegerVector  nord(nn_order, nn_order + n_spatial);
    Rcpp::IntegerVector  tv (temporal_idx, temporal_idx + N);
    Rcpp::NumericVector  s2g(sigma2_spatial_grid, sigma2_spatial_grid + n_grid);
    Rcpp::NumericVector  phg(phi_gp_spatial_grid, phi_gp_spatial_grid + n_grid);
    Rcpp::NumericVector  ttg(tau_temporal_grid,   tau_temporal_grid   + n_grid);
    Rcpp::Nullable<Rcpp::NumericVector> rtg = nl_wrap_rho_temporal(rho_temporal_grid, n_grid);

    Rcpp::List out = cpp_nested_laplace_st_nngp(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        sidx, n_spatial,
        cm, nim, ndm, nord, nn, cov_type,
        tv, n_times,
        s2g, phg,
        std::string(temporal_type), ttg, rtg, (cyclic != 0),
        in.fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init),
        store_Q != 0
    );
    copy_nested_laplace_result(out, result_out);
}

// ============================================================================
// SPDE nested-Laplace shim
// ============================================================================
//
// ABI v14 alignment: SPDE now uses the universal NestedLaplaceShimResult
// (modes + optional per-grid Q) like every other nested-Laplace backend.
// Latent layout: [beta (p)] [re (n_re_groups)] [w_mesh (n_mesh)].

extern "C" void tulpa_nested_laplace_spde_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int n_obs, int p, int n_mesh,
    int n_re_groups, double sigma_re,
    const double* A_x, const int* A_i, const int* A_p,
    const double* C0_diag,
    const double* G1_x, const int* G1_i, const int* G1_p,
    const double* range_grid, const double* sigma_grid, int n_grid,
    double nu,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    const double* rational_poles, int n_rational,
    const double* rational_weights,
    int store_Q,
    tulpa::NestedLaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, n_obs, p, family);

    int A_nnz  = A_p  ? A_p [n_mesh] : 0;
    int G1_nnz = G1_p ? G1_p[n_mesh] : 0;
    Rcpp::NumericVector  axv(A_x, A_x + A_nnz);
    Rcpp::IntegerVector  aiv(A_i, A_i + A_nnz);
    Rcpp::IntegerVector  apv(A_p, A_p + n_mesh + 1);
    Rcpp::NumericVector  c0  (C0_diag, C0_diag + n_mesh);
    Rcpp::NumericVector  g1xv(G1_x, G1_x + G1_nnz);
    Rcpp::IntegerVector  g1iv(G1_i, G1_i + G1_nnz);
    Rcpp::IntegerVector  g1pv(G1_p, G1_p + n_mesh + 1);
    Rcpp::NumericVector  rng (range_grid, range_grid + n_grid);
    Rcpp::NumericVector  sig (sigma_grid, sigma_grid + n_grid);

    Rcpp::Nullable<Rcpp::NumericVector> poles_n   = R_NilValue;
    Rcpp::Nullable<Rcpp::NumericVector> weights_n = R_NilValue;
    if (rational_poles && rational_weights && n_rational > 0) {
        poles_n   = Rcpp::wrap(Rcpp::NumericVector(rational_poles,   rational_poles   + n_rational));
        weights_n = Rcpp::wrap(Rcpp::NumericVector(rational_weights, rational_weights + n_rational));
    }

    Rcpp::List out = cpp_nested_laplace_spde(
        in.yv, in.nv, in.Xm,
        in.rv, n_re_groups, sigma_re,
        axv, aiv, apv, n_obs, n_mesh,
        c0, g1xv, g1iv, g1pv,
        rng, sig, nu,
        in.fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init),
        poles_n, weights_n,
        store_Q != 0
    );
    copy_nested_laplace_result(out, result_out);
}

// Joint multi-likelihood nested-Laplace C-ABI shim removed in Phase J-E.
// The legacy `cpp_nested_laplace_joint_bym2` kernel was deleted along
// with the matching `cpp_nested_laplace_joint_{icar,car_proper}` kernels
// when the single-block joint path was routed through
// `cpp_nested_laplace_joint_multi` (`R/nested_laplace_joint.R`'s
// `.joint_call_kernel_via_multi`). External callers that previously
// embedded the BYM2 joint path via the C-ABI should invoke
// `tulpa_nested_laplace_joint()` from R (or rebuild a shim on top of
// `cpp_nested_laplace_joint_multi`).
