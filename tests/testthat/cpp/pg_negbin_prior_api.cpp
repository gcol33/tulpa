#include <Rcpp.h>
#include <tulpa/pg_api.h>

static Rcpp::NumericMatrix nb_api_matrix(const double* x, int nr, int nc) {
  Rcpp::NumericMatrix out(nr, nc);
  for (int i = 0; i < nr; i++) {
    for (int j = 0; j < nc; j++) out(i, j) = x[i * nc + j];
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::List cpp_test_pg_negbin_prior_api(Rcpp::IntegerVector y,
                                       Rcpp::NumericMatrix X,
                                       bool spatial, int n_groups) {
  const int n = y.size();
  Rcpp::IntegerVector groups(n, 1), units(n);
  for (int i = 0; i < n; i++) units[i] = i < n / 2 ? 1 : 2;
  tulpa::PGShimResult out{};
  if (spatial) {
    const int row_ptr[] = {0, 1, 2};
    const int col_idx[] = {2, 1};
    const int degree[] = {1, 1};
    tulpa::get_pg_negbin_spatial_fn()(
      y.begin(), X.begin(), n, X.ncol(), groups.begin(), n_groups,
      units.begin(), 2, row_ptr, col_idx, degree, 80, 20, 1,
      .1, 1, .8, .1, 100, 50, 2, 1, 0, 1, &out);
  } else {
    tulpa::get_pg_negbin_fn()(
      y.begin(), X.begin(), n, X.ncol(), groups.begin(), n_groups,
      80, 20, 1, .1, 1, 100, 50, 2, 1, 0, 1, &out);
  }
  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = nb_api_matrix(out.beta, out.n_save, out.n_beta),
    Rcpp::Named("re") = nb_api_matrix(out.re, out.n_save, out.n_re),
    Rcpp::Named("eta") = nb_api_matrix(out.eta, out.n_save, out.n_obs),
    Rcpp::Named("r") = Rcpp::NumericVector(out.r_disp, out.r_disp + out.n_save));
  if (spatial) {
    result["spatial"] = nb_api_matrix(out.spatial, out.n_save, out.n_spatial_units);
  }
  out.free_buffers();
  return result;
}
