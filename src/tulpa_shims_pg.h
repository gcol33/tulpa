// ============================================================================
// PG shims
// ============================================================================

namespace {

inline void fill_pg_result(
    const Rcpp::List& draws,
    int n_obs, int n_spatial_units, bool has_r,
    tulpa::PGShimResult* out
) {
    Rcpp::NumericMatrix beta = draws["beta"];
    Rcpp::NumericMatrix re   = draws["re"];
    Rcpp::NumericVector sig  = draws["sigma_re"];

    out->n_save = beta.nrow();
    out->n_beta = beta.ncol();
    out->n_re   = re.ncol();
    out->n_obs  = n_obs;
    out->n_spatial_units = n_spatial_units;

    out->beta     = matrix_to_row_major(beta);
    out->re       = matrix_to_row_major(re);
    out->sigma_re = vector_to_buf(sig);
    out->eta      = nullptr;
    out->r_disp   = nullptr;
    out->spatial  = nullptr;
    out->tau_spatial = nullptr;

    if (draws.containsElementNamed("eta")) {
        Rcpp::NumericMatrix e = draws["eta"];
        out->eta = matrix_to_row_major(e);
    }
    if (has_r && draws.containsElementNamed("r")) {
        Rcpp::NumericVector r = draws["r"];
        out->r_disp = vector_to_buf(r);
    }
    if (n_spatial_units > 0 && draws.containsElementNamed("spatial")) {
        Rcpp::NumericMatrix s = draws["spatial"];
        out->spatial = matrix_to_row_major(s);
    }
    if (n_spatial_units > 0 && draws.containsElementNamed("tau")) {
        Rcpp::NumericVector t = draws["tau"];
        out->tau_spatial = vector_to_buf(t);
    }
}

} // namespace

// Forward declaration: defined inside namespace tulpa in pg_binomial.cpp.
namespace tulpa {
Rcpp::List pg_binomial_gibbs_impl(
    Rcpp::IntegerVector y, Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X, Rcpp::IntegerVector group,
    int n_groups, int n_iter, int n_warmup, int thin,
    double prior_beta_sd, double prior_sigma_scale,
    bool store_eta, bool verbose, int n_threads
);
}

extern "C" void tulpa_pg_binomial_gibbs_impl(
    const int* y,
    const int* n_trials,
    const double* X_flat,
    int N_obs, int p,
    const int* group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    int store_eta,
    int verbose,
    int n_threads,
    tulpa::PGShimResult* result_out
) {
    Rcpp::IntegerVector yv(y, y + N_obs);
    Rcpp::IntegerVector nv(n_trials, n_trials + N_obs);
    Rcpp::NumericMatrix Xm = build_matrix_colmajor(X_flat, N_obs, p);
    Rcpp::IntegerVector gv(group, group + N_obs);

    GetRNGstate();
    Rcpp::List draws = tulpa::pg_binomial_gibbs_impl(
        yv, nv, Xm, gv, n_groups,
        n_iter, n_warmup, thin,
        prior_beta_sd, prior_sigma_scale,
        store_eta != 0, verbose != 0, n_threads
    );
    PutRNGstate();

    fill_pg_result(draws, N_obs, /*n_spatial_units=*/0, /*has_r=*/false, result_out);
}

extern "C" void tulpa_pg_negbin_gibbs_impl(
    const int* y,
    const double* X_flat,
    int N_obs, int p,
    const int* group,
    int n_groups,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_scale,
    double prior_r_shape,
    double prior_r_rate,
    double r_init,
    int store_eta,
    int verbose,
    int n_threads,
    tulpa::PGShimResult* result_out
) {
    Rcpp::IntegerVector yv(y, y + N_obs);
    Rcpp::NumericMatrix Xm = build_matrix_colmajor(X_flat, N_obs, p);
    Rcpp::IntegerVector gv(group, group + N_obs);

    GetRNGstate();
    Rcpp::List draws = tulpa::pg_negbin_gibbs(
        yv, Xm, gv, n_groups,
        n_iter, n_warmup, thin,
        prior_beta_sd, prior_sigma_scale,
        prior_r_shape, prior_r_rate, r_init,
        store_eta != 0, verbose != 0, n_threads
    );
    PutRNGstate();

    fill_pg_result(draws, N_obs, /*n_spatial_units=*/0, /*has_r=*/true, result_out);
}

extern "C" void tulpa_pg_negbin_spatial_gibbs_impl(
    const int* y,
    const double* X_flat,
    int N_obs, int p,
    const int* re_group,
    int n_re_groups,
    const int* spatial_group,
    int n_spatial_units,
    const int* adj_row_ptr,
    const int* adj_col_idx,
    const int* n_neighbors,
    int n_iter,
    int n_warmup,
    int thin,
    double prior_beta_sd,
    double prior_sigma_re_scale,
    double prior_tau_shape,
    double prior_tau_rate,
    double prior_r_shape,
    double prior_r_rate,
    double r_init,
    int store_eta,
    int verbose,
    int n_threads,
    tulpa::PGShimResult* result_out
) {
    Rcpp::IntegerVector yv(y, y + N_obs);
    Rcpp::NumericMatrix Xm = build_matrix_colmajor(X_flat, N_obs, p);
    Rcpp::IntegerVector reg(re_group, re_group + N_obs);
    Rcpp::IntegerVector sg(spatial_group, spatial_group + N_obs);
    Rcpp::IntegerVector nn(n_neighbors, n_neighbors + n_spatial_units);

    // Internal API takes adjacency as a List<IntegerVector>; build it from CSR.
    Rcpp::List adj_list(n_spatial_units);
    for (int i = 0; i < n_spatial_units; i++) {
        int s = adj_row_ptr[i], e = adj_row_ptr[i + 1];
        Rcpp::IntegerVector neighbors(e - s);
        for (int k = s; k < e; k++) neighbors[k - s] = adj_col_idx[k];
        adj_list[i] = neighbors;
    }

    GetRNGstate();
    Rcpp::List draws = tulpa::pg_negbin_gibbs_spatial(
        yv, Xm, reg, n_re_groups, sg, n_spatial_units,
        adj_list, nn,
        n_iter, n_warmup, thin,
        prior_beta_sd, prior_sigma_re_scale,
        prior_tau_shape, prior_tau_rate,
        prior_r_shape, prior_r_rate, r_init,
        store_eta != 0, verbose != 0, n_threads
    );
    PutRNGstate();

    fill_pg_result(draws, N_obs, n_spatial_units, /*has_r=*/true, result_out);
}

