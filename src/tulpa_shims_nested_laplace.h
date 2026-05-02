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
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);
    Rcpp::NumericVector  tg (tau_grid,    tau_grid + n_grid);

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_icar(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn,
        tg, fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init)
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
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);
    Rcpp::NumericVector  sg (sigma_spatial_grid, sigma_spatial_grid + n_grid);
    Rcpp::NumericVector  rg (rho_grid,           rho_grid           + n_grid);

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_bym2(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn,
        scale_factor, sg, rg,
        fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init)
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
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);
    Rcpp::NumericVector  tg (tau_grid, tau_grid + n_grid);
    Rcpp::NumericVector  rg (rho_grid, rho_grid + n_grid);

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_car_proper(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn,
        tg, rg, fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init)
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
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::IntegerVector  tv(temporal_idx, temporal_idx + N);
    Rcpp::NumericVector  tg(tau_grid, tau_grid + n_grid);

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_rw1(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        tv, n_times, (cyclic != 0),
        tg, fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init)
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
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::IntegerVector  tv(temporal_idx, temporal_idx + N);
    Rcpp::NumericVector  tg(tau_grid, tau_grid + n_grid);

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_rw2(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        tv, n_times,
        tg, fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init)
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
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::IntegerVector  tv(temporal_idx, temporal_idx + N);
    Rcpp::NumericVector  tg(tau_grid, tau_grid + n_grid);
    Rcpp::NumericVector  rg(rho_grid, rho_grid + n_grid);

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_ar1(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        tv, n_times, tg, rg,
        fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init)
    );
    copy_nested_laplace_result(out, result_out);
}

extern "C" void tulpa_nested_laplace_nngp_impl(
    const double* y, const int* n_trials,
    const double* X_flat, const double* re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const double* coords_flat, int coord_dim,
    const int* nn_idx_flat, const double* nn_dist_flat,
    const int* nn_order,
    int n_spatial, int nn,
    const double* sigma2_grid, const double* phi_gp_grid, int n_grid,
    int cov_type,
    const char* family, double phi,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::NumericMatrix  cm = build_matrix_colmajor(coords_flat, n_spatial, coord_dim);
    Rcpp::IntegerMatrix  nim = build_int_matrix_colmajor(nn_idx_flat, n_spatial, nn);
    Rcpp::NumericMatrix  ndm = build_matrix_colmajor(nn_dist_flat, n_spatial, nn);
    Rcpp::IntegerVector  nord(nn_order, nn_order + n_spatial);
    Rcpp::NumericVector  s2g(sigma2_grid,  sigma2_grid  + n_grid);
    Rcpp::NumericVector  phg(phi_gp_grid,  phi_gp_grid  + n_grid);

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_nngp(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        cm, nim, ndm, nord, n_spatial, nn,
        s2g, phg, cov_type,
        fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init)
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
    tulpa::NestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);
    Rcpp::NumericVector  rv(re_idx, re_idx + N);
    Rcpp::NumericMatrix  pb = build_matrix_colmajor(phi_basis_flat, N, n_basis);
    Rcpp::NumericVector  le(lambda_eig, lambda_eig + n_basis);
    Rcpp::NumericVector  s2g(sigma2_grid,      sigma2_grid      + n_grid);
    Rcpp::NumericVector  lsg(lengthscale_grid, lengthscale_grid + n_grid);

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_hsgp(
        yv, nv, Xm, rv, n_re_groups, sigma_re,
        pb, le, s2g, lsg,
        fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init)
    );
    copy_nested_laplace_result(out, result_out);
}

// ============================================================================
// SPDE nested-Laplace shim
// ============================================================================

extern "C" void tulpa_nested_laplace_spde_impl(
    const double* y, const int* n_trials,
    const double* X_flat,
    int n_obs, int p, int n_mesh,
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
    tulpa::SpdeNestedLaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + n_obs);
    Rcpp::IntegerVector  nv(n_trials, n_trials + n_obs);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, n_obs, p);

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

    std::string fam = family ? std::string(family) : std::string("binomial");

    Rcpp::List out = cpp_nested_laplace_spde(
        yv, nv, Xm,
        axv, aiv, apv, n_obs, n_mesh,
        c0, g1xv, g1iv, g1pv,
        rng, sig, nu,
        fam, phi, max_iter, tol, n_threads,
        wrap_x_init(x_init, n_x_init),
        poles_n, weights_n
    );

    Rcpp::NumericVector lm = out["log_marginal"];
    Rcpp::IntegerVector ni = out["n_iter"];
    int ng = (int)lm.size();
    result_out->n_grid = ng;
    result_out->Q_nnz  = (int)Rcpp::as<int>(out["Q_nnz"]);
    result_out->log_marginal = new double[ng];
    result_out->n_iter       = new int[ng];
    for (int k = 0; k < ng; k++) {
        result_out->log_marginal[k] = lm[k];
        result_out->n_iter[k]       = ni[k];
    }
}

