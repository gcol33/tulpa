// pg_binomial_gp_rsr.cpp
// Restricted spatial regression on a CONTINUOUS field: an NNGP Gaussian
// process projected orthogonal to the covariates, sampled by Polya-Gamma Gibbs
// for binomial models (gcol33/tulpa#848).
//
// The areal counterpart is pg_binomial_rsr.cpp. The projection is the same
// construction and the sweep has the same shape; what differs is the field's
// own prior precision, an NNGP one built from the Vecchia factors at the
// current (sigma2, phi) rather than a fixed ICAR adjacency, and the fact that
// the NNGP prior is PROPER -- so nothing has to be supplied on the constant
// direction and the draw needs no centring.
//
// RSR exists for spatially smooth covariates (a climate surface, elevation),
// which is exactly where a continuous field is the natural prior and an areal
// one a discretisation of convenience, so the mitigation was available on the
// field shape that needs it least. What the literature does NOT say is that
// the restriction is free: see ?spatial_rsr for the coverage caveat this
// inherits from the areal case.

#include "pg_shared.h"
#include "pg_rng.h"
#include "linalg_fast.h"
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// ---------------------------------------------------------------------
// Model: eta_i = X_i beta + re_i + (P w)_i with w ~ NNGP(sigma2, phi) over the
// unique locations, one observation per location.
//
// Conditional on the Polya-Gamma weights the field's full conditional is
// Gaussian with precision P W P + Lambda(sigma2, phi) and linear term
// P (kappa - W offset), where W = diag(sum_omega). That is a dense J x J solve
// per sweep: the projector couples every pair of locations, so the sparse
// single-site sweep the unprojected NNGP kernel takes is not available here --
// the same reason the areal RSR kernel is O(J^3) where the plain ICAR one is a
// single-site sweep.
//
// No level step. The unprojected kernel draws the field's level jointly with
// the intercept because the two are collinear; here the projector annihilates
// the column space of the restricted design, which carries the intercept, so
// the field cannot put a level into eta at all. The raw field's own level is
// identified by its (proper) prior alone.
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_gp_rsr(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx,
    Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order,
    int n_spatial,
    int nn,
    Rcpp::NumericVector rsr_projection,  // P_perp (n_spatial x n_spatial, row-major)
    int rsr_n,
    double sigma2_gp_init,
    double phi_gp_init,
    int cov_type,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_sigma_gp_U = 1.0,
    double prior_sigma_gp_alpha = 0.01,
    double prior_phi_lower = 0.01,
    double prior_phi_upper = 10.0,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  const int n_save = tulpa::pg_n_save(n_iter, n_warmup, thin);
  tulpa::PgGibbsCommon C(y, n, X, re_group, n_re_groups, n_save,
                         prior_sigma_re_scale, n_threads, store_eta);
  const int N = C.N;
  const int p = C.p;

  if (N != n_spatial) {
    Rcpp::stop("The NNGP Gibbs kernel maps observation i to location i, so it "
               "needs one observation per location: got %d observation(s) for "
               "%d location(s).", N, n_spatial);
  }
  if (coords.nrow() != n_spatial) {
    Rcpp::stop("`coords` has %d row(s) but `n_spatial` is %d.",
               static_cast<int>(coords.nrow()), n_spatial);
  }
  if (rsr_n != n_spatial) {
    Rcpp::stop("`rsr_n` is %d but `n_spatial` is %d.", rsr_n, n_spatial);
  }
  tulpa::pg_check_pc_prior(prior_sigma_gp_U, prior_sigma_gp_alpha, "gp");
  if (!(prior_phi_lower > 0.0) || !(prior_phi_upper > prior_phi_lower)) {
    Rcpp::stop("The range prior needs 0 < prior_phi_lower < prior_phi_upper; "
               "got [%g, %g].", prior_phi_lower, prior_phi_upper);
  }
  const int J = n_spatial;
  tulpa::pg_check_rsr_projector(rsr_projection, J, /*require_constant=*/false);

  // Per-variant storage
  Rcpp::NumericMatrix gp_raw_draws(n_save, J);
  Rcpp::NumericMatrix gp_proj_draws(n_save, J);
  Rcpp::NumericVector sigma2_gp_draws(n_save);
  Rcpp::NumericVector phi_gp_draws(n_save);

  // Per-variant state
  tulpa::PgNngpScale gp;
  gp.w.assign(J, 0.0);
  gp.sigma2 = sigma2_gp_init;
  gp.phi = phi_gp_init;
  gp.top = tulpa::pg_nngp_topology(nn_idx, nn_dist, nn_order, J, nn);
  Rcpp::NumericVector gp_contrib(N, 0.0);
  std::vector<double> w_proj(J, 0.0);
  std::vector<double> sum_omega_s(J, 0.0), sum_resid_s(J, 0.0);
  const double* P = rsr_projection.begin();
  std::vector<double> Lambda, PW(static_cast<size_t>(J) * J);
  std::vector<double> M(static_cast<size_t>(J) * J), lin(J);
  Rcpp::NumericVector w_draw(J);

  int save_idx = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    if (verbose && (iter + 1) % 200 == 0) {
      Rcpp::Rcout << "  Iteration " << (iter + 1) << "/" << n_iter << "\n";
    }

    // 1. The projected field is what eta sees.
    for (int i = 0; i < N; i++) gp_contrib[i] = w_proj[i];

    // 2-4. Core Gibbs step (eta, omega, beta, RE) -- shared with all variants.
    tulpa::pg_gibbs_core_step(
        N, p, C.beta, C.re, C.sigma_re, C.omega, C.eta, C.X_beta, C.re_contrib,
        gp_contrib, C.offset, C.kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale, C.n_threads_team);

    // 5. Per-location Polya-Gamma statistics against the non-field offset.
    for (int i = 0; i < N; i++) C.offset[i] = C.X_beta[i] + C.re_contrib[i];
    tulpa::pg_accumulate_stats(N, nullptr, J, C.omega.begin(), C.kappa.begin(),
                               C.offset.begin(), sum_omega_s.data(),
                               sum_resid_s.data());

    // 6. The raw field, from the dense conditional the projection induces.
    //    Factors first, so Lambda is the prior at the current range.
    tulpa::pg_nngp_factors(gp.phi, cov_type, coords, nn_dist, gp.top, gp.fac);
    tulpa::pg_nngp_precision_dense(gp.top, gp.fac, gp.sigma2, Lambda);

    for (int a = 0; a < J; a++) {
      const size_t ra = static_cast<size_t>(a) * J;
      for (int s = 0; s < J; s++) PW[ra + s] = P[ra + s] * sum_omega_s[s];
      double v = 0.0;
      for (int s = 0; s < J; s++) v += P[ra + s] * sum_resid_s[s];
      lin[a] = v;
    }
    for (int a = 0; a < J; a++) {
      const size_t ra = static_cast<size_t>(a) * J;
      for (int b = a; b < J; b++) {
        const size_t rb = static_cast<size_t>(b) * J;
        double v = 0.0;
        for (int s = 0; s < J; s++) v += PW[ra + s] * P[rb + s];
        M[ra + b] = v + Lambda[ra + b];
        M[rb + a] = v + Lambda[rb + a];
      }
    }

    tulpa::pg_draw_gaussian_precision(M.data(), J, lin.data(), w_draw.begin(),
                                      "restricted NNGP spatial field");
    for (int s = 0; s < J; s++) gp.w[s] = w_draw[s];

    // 7. The projected field for this draw, so the reported pair and the eta of
    //    the next sweep come from the same field.
    for (int s = 0; s < J; s++) {
      const size_t rs = static_cast<size_t>(s) * J;
      double v = 0.0;
      for (int k = 0; k < J; k++) v += P[rs + k] * gp.w[k];
      w_proj[s] = v;
    }

    // 8. The marginal variance and the range, given the RAW field -- its own
    //    prior is what they parameterize, exactly as the areal kernel updates
    //    tau from the raw ICAR field rather than from the projected one.
    tulpa::pg_nngp_hyper_update(gp, cov_type, coords, nn_dist,
                                prior_sigma_gp_U, prior_sigma_gp_alpha,
                                prior_phi_lower, prior_phi_upper);

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      C.save(save_idx);
      for (int s = 0; s < J; s++) {
        gp_raw_draws(save_idx, s) = gp.w[s];
        gp_proj_draws(save_idx, s) = w_proj[s];
      }
      sigma2_gp_draws[save_idx] = gp.sigma2;
      phi_gp_draws[save_idx] = gp.phi;
      for (int i = 0; i < N; i++) gp_contrib[i] = w_proj[i];
      C.log_prob_draws[save_idx] =
          C.log_joint_common(y, n, gp_contrib.begin(), prior_beta_sd,
                             prior_sigma_re_scale) +
          tulpa::pg_log_nngp_scale(gp, prior_sigma_gp_U, prior_sigma_gp_alpha,
                                   prior_phi_lower, prior_phi_upper);
      save_idx++;
    }

    if ((iter + 1) % 100 == 0) Rcpp::checkUserInterrupt();
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = C.beta_draws,
    Rcpp::Named("re") = C.re_draws,
    Rcpp::Named("sigma_re") = C.sigma_re_draws,
    Rcpp::Named("gp_raw") = gp_raw_draws,
    Rcpp::Named("gp") = gp_proj_draws,
    Rcpp::Named("sigma2_gp") = sigma2_gp_draws,
    Rcpp::Named("phi_gp") = phi_gp_draws,
    Rcpp::Named("log_prob") = C.log_prob_draws
  );

  if (store_eta) {
    result["eta"] = C.eta_draws;
  }

  return result;
}
