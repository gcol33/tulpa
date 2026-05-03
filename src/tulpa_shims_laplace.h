// ============================================================================
// Laplace shims
// ============================================================================

extern "C" void tulpa_laplace_mode_dense_impl(
    const double* y,
    const int* n_trials,
    const double* X_flat,
    const double* re_idx,
    int N, int p,
    int n_re_groups,
    double sigma_re,
    const char* family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    const double* /*x_init*/,
    int /*n_x_init*/,
    tulpa::LaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);

    // laplace_mode_dense does not accept x_init — kept in the shim signature
    // for API symmetry with the spatial variant.
    tulpa::LaplaceResult result = tulpa::laplace_mode_dense(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re, in.fam, phi,
        max_iter, tol, n_threads
    );
    copy_mode_result(result, result_out);
}

extern "C" void tulpa_laplace_mode_spatial_impl(
    const double* y,
    const int* n_trials,
    const double* X_flat,
    const double* re_idx,
    int N, int p,
    int n_re_groups,
    double sigma_re,
    const int* spatial_idx,
    int n_spatial_units,
    const int* adj_row_ptr,
    const int* adj_col_idx,
    const int* n_neighbors,
    double tau_spatial,
    const char* family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    const double* x_init,
    int n_x_init,
    tulpa::LaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);
    Rcpp::NumericVector  xinit;
    if (x_init && n_x_init > 0) xinit = Rcpp::NumericVector(x_init, x_init + n_x_init);

    tulpa::LaplaceResult result = tulpa::laplace_mode_spatial(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn, tau_spatial,
        in.fam, phi, max_iter, tol, n_threads, xinit
    );
    copy_mode_result(result, result_out);
}

extern "C" void tulpa_laplace_mode_dense_multi_re_impl(
    const double* y,
    const int* n_trials,
    const double* X_flat,
    int N, int p,
    int n_terms,
    const int* re_idx_flat,
    const int* re_ngroups,
    const int* re_ncoefs,
    const int* sigma_offsets,
    const double* sigma_flat,
    const int* Z_offsets,
    const double* Z_flat,
    const char* family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    const double* weights,
    const double* offset_vec,
    const double* x_init,
    int n_x_init,
    tulpa::LaplaceShimResult* result_out
) {
    Rcpp::NumericVector  yv(y, y + N);
    Rcpp::IntegerVector  nv(n_trials, n_trials + N);
    Rcpp::NumericMatrix  Xm = build_matrix_colmajor(X_flat, N, p);

    Rcpp::List re_idx_list(n_terms);
    for (int k = 0; k < n_terms; k++) {
        Rcpp::IntegerVector v(re_idx_flat + (size_t)k * N,
                              re_idx_flat + (size_t)(k + 1) * N);
        re_idx_list[k] = v;
    }

    Rcpp::IntegerVector re_ngroups_v(re_ngroups, re_ngroups + n_terms);

    Rcpp::List re_sigma_list(n_terms);
    for (int k = 0; k < n_terms; k++) {
        int s = sigma_offsets[k];
        int e = sigma_offsets[k + 1];
        Rcpp::NumericVector v(sigma_flat + s, sigma_flat + e);
        re_sigma_list[k] = v;
    }

    // re_ncoefs is needed both to drive the internal API and to size Z slices.
    std::vector<int> ncoefs_vec(n_terms, 1);
    if (re_ncoefs) {
        for (int k = 0; k < n_terms; k++) ncoefs_vec[k] = re_ncoefs[k];
    }

    Rcpp::Nullable<Rcpp::IntegerVector> ncoefs_nullable = R_NilValue;
    if (re_ncoefs) {
        ncoefs_nullable = Rcpp::Nullable<Rcpp::IntegerVector>(
            Rcpp::IntegerVector(re_ncoefs, re_ncoefs + n_terms));
    }

    Rcpp::Nullable<Rcpp::List> Z_list_nullable = R_NilValue;
    if (Z_offsets && Z_flat) {
        Rcpp::List re_Z_list(n_terms);
        for (int k = 0; k < n_terms; k++) {
            int ck = ncoefs_vec[k];
            int off = Z_offsets[k];
            Rcpp::NumericMatrix Zk(N, ck);
            if (N > 0 && ck > 0) {
                std::memcpy(&Zk[0], Z_flat + off,
                            sizeof(double) * (size_t)N * ck);
            }
            re_Z_list[k] = Zk;
        }
        Z_list_nullable = Rcpp::Nullable<Rcpp::List>(re_Z_list);
    }

    Rcpp::Nullable<Rcpp::NumericVector> weights_nullable = R_NilValue;
    if (weights) {
        weights_nullable = Rcpp::Nullable<Rcpp::NumericVector>(
            Rcpp::NumericVector(weights, weights + N));
    }

    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue;
    if (offset_vec) {
        offset_nullable = Rcpp::Nullable<Rcpp::NumericVector>(
            Rcpp::NumericVector(offset_vec, offset_vec + N));
    }

    Rcpp::Nullable<Rcpp::NumericVector> x_init_nullable = R_NilValue;
    if (x_init && n_x_init > 0) {
        x_init_nullable = Rcpp::Nullable<Rcpp::NumericVector>(
            Rcpp::NumericVector(x_init, x_init + n_x_init));
    }

    std::string fam = family ? std::string(family) : std::string("binomial");

    tulpa::LaplaceResult result = tulpa::laplace_mode_dense_multi_re(
        yv, nv, Xm, re_idx_list, re_ngroups_v, re_sigma_list,
        fam, phi, max_iter, tol, n_threads,
        Z_list_nullable, ncoefs_nullable,
        weights_nullable, offset_nullable, x_init_nullable
    );
    copy_mode_result(result, result_out);
}

extern "C" void tulpa_laplace_mode_bym2_impl(
    const double* y,
    const int* n_trials,
    const double* X_flat,
    const double* re_idx,
    int N, int p,
    int n_re_groups,
    double sigma_re,
    const int* spatial_idx,
    int n_spatial_units,
    const int* adj_row_ptr,
    const int* adj_col_idx,
    const int* n_neighbors,
    double sigma_spatial,
    double rho,
    double scale_factor,
    const char* family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    tulpa::LaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);

    tulpa::LaplaceResult result = tulpa::laplace_mode_bym2(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn,
        sigma_spatial, rho, scale_factor,
        in.fam, phi, max_iter, tol, n_threads
    );
    copy_mode_result(result, result_out);
}

extern "C" void tulpa_laplace_mode_gp_impl(
    const double* y,
    const int* n_trials,
    const double* X_flat,
    const double* re_idx,
    int N, int p,
    int n_re_groups,
    double sigma_re,
    const double* coords_flat,
    int coord_dim,
    const int* nn_idx_flat,
    const double* nn_dist_flat,
    const int* nn_order,
    int n_spatial, int nn,
    double sigma2_gp, double phi_gp, int cov_type,
    const char* family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    tulpa::LaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::NumericMatrix  cm = build_matrix_colmajor(coords_flat, n_spatial, coord_dim);
    Rcpp::IntegerMatrix  nim = build_int_matrix_colmajor(nn_idx_flat, n_spatial, nn);
    Rcpp::NumericMatrix  ndm = build_matrix_colmajor(nn_dist_flat, n_spatial, nn);
    Rcpp::IntegerVector  nord(nn_order, nn_order + n_spatial);

    tulpa::LaplaceResult result = tulpa::laplace_mode_gp(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        cm, nim, ndm, nord, n_spatial, nn,
        sigma2_gp, phi_gp, cov_type,
        in.fam, phi, max_iter, tol, n_threads
    );
    copy_mode_result(result, result_out);
}

extern "C" void tulpa_laplace_mode_multiscale_gp_impl(
    const double* y,
    const int* n_trials,
    const double* X_flat,
    const double* re_idx,
    int N, int p,
    int n_re_groups,
    double sigma_re,
    const double* coords_flat,
    int coord_dim,
    const int* nn_idx_local_flat,
    const double* nn_dist_local_flat,
    const int* nn_order_local,
    int nn_local,
    const int* nn_idx_regional_flat,
    const double* nn_dist_regional_flat,
    const int* nn_order_regional,
    int nn_regional,
    int n_spatial,
    double sigma2_local, double phi_local,
    double sigma2_regional, double phi_regional,
    int cov_type,
    const char* family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    tulpa::LaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::NumericMatrix  cm = build_matrix_colmajor(coords_flat, n_spatial, coord_dim);

    Rcpp::IntegerMatrix  nim_l = build_int_matrix_colmajor(nn_idx_local_flat,    n_spatial, nn_local);
    Rcpp::NumericMatrix  ndm_l = build_matrix_colmajor    (nn_dist_local_flat,   n_spatial, nn_local);
    Rcpp::IntegerVector  nord_l(nn_order_local,    nn_order_local    + n_spatial);
    Rcpp::IntegerMatrix  nim_r = build_int_matrix_colmajor(nn_idx_regional_flat, n_spatial, nn_regional);
    Rcpp::NumericMatrix  ndm_r = build_matrix_colmajor    (nn_dist_regional_flat, n_spatial, nn_regional);
    Rcpp::IntegerVector  nord_r(nn_order_regional, nn_order_regional + n_spatial);

    tulpa::LaplaceResult result = tulpa::laplace_mode_multiscale_gp(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        cm,
        nim_l, ndm_l, nord_l, nn_local,
        nim_r, ndm_r, nord_r, nn_regional,
        n_spatial,
        sigma2_local, phi_local, sigma2_regional, phi_regional,
        cov_type, in.fam, phi, max_iter, tol, n_threads
    );
    copy_mode_result(result, result_out);
}

extern "C" void tulpa_laplace_mode_multiscale_temporal_impl(
    const double* y,
    const int* n_trials,
    const double* X_flat,
    const double* re_idx,
    int N, int p,
    int n_re_groups,
    double sigma_re,
    const int* time_idx,
    int n_times,
    int seasonal_period,
    int trend_type,
    int short_type,
    double sigma2_trend,
    double sigma2_seasonal,
    double sigma2_short,
    double rho_short,
    const char* family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    tulpa::LaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::IntegerVector  tv(time_idx, time_idx + N);

    tulpa::LaplaceResult result = tulpa::laplace_mode_multiscale_temporal(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        tv, n_times, seasonal_period, trend_type, short_type,
        sigma2_trend, sigma2_seasonal, sigma2_short, rho_short,
        in.fam, phi, max_iter, tol, n_threads
    );
    copy_mode_result(result, result_out);
}

extern "C" void tulpa_laplace_mode_rsr_impl(
    const double* y,
    const int* n_trials,
    const double* X_flat,
    const double* re_idx,
    int N, int p,
    int n_re_groups,
    double sigma_re,
    const int* spatial_idx,
    int n_spatial_units,
    const int* adj_row_ptr,
    const int* adj_col_idx,
    const int* n_neighbors,
    double tau_spatial,
    const double* rsr_projection_flat,
    int rsr_n,
    const char* family,
    double phi,
    int max_iter,
    double tol,
    int n_threads,
    tulpa::LaplaceShimResult* result_out
) {
    auto in = pack_laplace_shim_inputs(y, n_trials, X_flat, re_idx, N, p, family);
    Rcpp::IntegerVector  sidx, arp, aci, nn;
    marshal_adj(spatial_idx, N, adj_row_ptr, adj_col_idx, n_neighbors, n_spatial_units,
                sidx, arp, aci, nn);
    Rcpp::NumericVector  proj(rsr_projection_flat,
                              rsr_projection_flat + (size_t)rsr_n * (size_t)rsr_n);

    tulpa::LaplaceResult result = tulpa::laplace_mode_rsr(
        in.yv, in.nv, in.Xm, in.rv, n_re_groups, sigma_re,
        sidx, n_spatial_units, arp, aci, nn,
        tau_spatial, proj, rsr_n,
        in.fam, phi, max_iter, tol, n_threads
    );
    copy_mode_result(result, result_out);
}

