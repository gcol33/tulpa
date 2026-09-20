// pg_binomial_rsr.cpp
// RSR (Restricted Spatial Regression) Gibbs sampler for Pólya-Gamma binomial
// models. The spatial field enters the linear predictor projected orthogonal
// to the covariates.

#include "pg_shared.h"
#include "pg_spatial.h"
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
// RSR (Restricted Spatial Regression) Gibbs sampler
//
// Model: eta_i = X_i beta + re_i + (P phi)_{s(i)} with phi ~ ICAR(tau).
// Conditional on the Polya-Gamma weights the field's full conditional is
// Gaussian with precision P W P + tau Q and linear term P (kappa - W offset)
// aggregated to spatial units, where W = diag(sum_omega). That is a dense
// J x J solve per sweep (O(J^3)), not a single-site sweep: the projector
// couples every pair of units.
//
// phi enters the likelihood only through P phi, so its component along the
// constant vector is unidentified. The precision is augmented by 11'/J on
// that direction and the draw is centred afterwards, which is the exact
// constrained draw and leaves both eta and phi'Q phi untouched.
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List cpp_pg_binomial_gibbs_rsr(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector n,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_group,
    int n_re_groups,
    Rcpp::IntegerVector spatial_group,
    int n_spatial_units,
    Rcpp::List adj_list,
    Rcpp::IntegerVector n_neighbors,
    Rcpp::NumericVector rsr_projection,  // P_perp matrix (n_spatial x n_spatial, row-major)
    int rsr_n,
    int n_iter = 2000,
    int n_warmup = 1000,
    int thin = 1,
    double prior_beta_sd = 10.0,
    double prior_sigma_re_scale = 2.5,
    double prior_tau_shape = 1.0,
    double prior_tau_rate = 0.01,
    bool store_eta = false,
    bool verbose = true,
    int n_threads = 1
) {
  const int n_save = tulpa::pg_n_save(n_iter, n_warmup, thin);
  tulpa::PgGibbsCommon C(y, n, X, re_group, n_re_groups, n_save,
                         prior_sigma_re_scale, n_threads, store_eta);
  const int N = C.N;
  const int p = C.p;

  if (rsr_n != n_spatial_units) {
    Rcpp::stop("`rsr_n` is %d but `n_spatial_units` is %d.",
               rsr_n, n_spatial_units);
  }
  const int J = n_spatial_units;
  tulpa::pg_check_index(spatial_group, N, J, "spatial_group");
  tulpa::pg_check_rsr_projector(rsr_projection, J, /*require_constant=*/true);
  const tulpa::PgAdjacency adj =
      tulpa::pg_build_adjacency(adj_list, n_neighbors, J);
  tulpa::pg_check_components_observed(adj, spatial_group);

  // Per-variant storage
  Rcpp::NumericMatrix spatial_raw_draws(n_save, J);
  Rcpp::NumericMatrix spatial_proj_draws(n_save, J);
  Rcpp::NumericVector tau_draws(n_save);

  // Per-variant state
  Rcpp::NumericVector phi(J, 0.0);       // Raw (unprojected)
  Rcpp::NumericVector phi_proj(J, 0.0);  // Projected -- what eta sees
  double tau = 1.0;
  Rcpp::NumericVector spatial_contrib(N, 0.0);

  const double* P = rsr_projection.begin();
  std::vector<double> sum_omega_s(J), sum_resid_s(J);
  std::vector<double> PW(static_cast<size_t>(J) * J);
  std::vector<double> M(static_cast<size_t>(J) * J);
  std::vector<double> lin(J);

  int save_idx = 0;

  for (int iter = 0; iter < n_iter; iter++) {
    // 1. Spatial contribution from the projected field.
    tulpa_parallel_for(C.n_threads_team, N, [&](int i) {
      spatial_contrib[i] = phi_proj[spatial_group[i] - 1];
    });

    // 2-6. Shared core (compute eta, sample omega, update beta/RE)
    tulpa::pg_gibbs_core_step(
        N, p, C.beta, C.re, C.sigma_re, C.omega, C.eta, C.X_beta, C.re_contrib,
        spatial_contrib, C.offset, C.kappa, n, X, re_group, n_re_groups,
        prior_beta_sd, prior_sigma_re_scale, C.n_threads_team);

    // 7. Update the raw field from its own full conditional.
    tulpa_parallel_for(C.n_threads_team, N, [&](int i) {
      C.offset[i] = C.X_beta[i] + C.re_contrib[i];
    });
    tulpa::pg_accumulate_stats(N, spatial_group.begin(), J, C.omega.begin(),
                               C.kappa.begin(), C.offset.begin(),
                               sum_omega_s.data(), sum_resid_s.data());

    for (int a = 0; a < J; a++) {
      const size_t ra = static_cast<size_t>(a) * J;
      for (int s = 0; s < J; s++) PW[ra + s] = P[ra + s] * sum_omega_s[s];
      double v = 0.0;
      for (int s = 0; s < J; s++) v += P[ra + s] * sum_resid_s[s];
      lin[a] = v;
    }
    const double one_over_J = 1.0 / J;
    for (int a = 0; a < J; a++) {
      const size_t ra = static_cast<size_t>(a) * J;
      for (int b = a; b < J; b++) {
        const size_t rb = static_cast<size_t>(b) * J;
        double v = 0.0;
        for (int s = 0; s < J; s++) v += PW[ra + s] * P[rb + s];
        v += one_over_J;
        M[ra + b] = v;
        M[rb + a] = v;
      }
      M[ra + a] += tau * adj.degree(a);
    }
    for (int a = 0; a < J; a++) {
      for (int e = adj.row_ptr[a]; e < adj.row_ptr[a + 1]; e++) {
        M[static_cast<size_t>(a) * J + adj.col_idx[e]] -= tau;
      }
    }

    tulpa::pg_draw_gaussian_precision(M.data(), J, lin.data(), phi.begin(),
                                      "RSR spatial field");

    double mean_phi = 0.0;
    for (int s = 0; s < J; s++) mean_phi += phi[s];
    mean_phi /= J;
    for (int s = 0; s < J; s++) phi[s] -= mean_phi;

    // 8. Projected field for this draw, so the reported pair and the eta of
    //    the next sweep come from the same field.
    for (int s = 0; s < J; s++) {
      const size_t rs = static_cast<size_t>(s) * J;
      double v = 0.0;
      for (int k = 0; k < J; k++) v += P[rs + k] * phi[k];
      phi_proj[s] = v;
    }

    // 9. Update tau (the ICAR precision of the raw field)
    tau = tulpa::update_tau_icar(phi, adj, prior_tau_shape, prior_tau_rate);

    // Save draws
    if (iter >= n_warmup && (iter - n_warmup) % thin == 0) {
      C.save(save_idx);
      for (int s = 0; s < J; s++) {
        spatial_raw_draws(save_idx, s) = phi[s];
        spatial_proj_draws(save_idx, s) = phi_proj[s];
      }
      tau_draws[save_idx] = tau;
      for (int i = 0; i < N; i++) {
        spatial_contrib[i] = phi_proj[spatial_group[i] - 1];
      }
      C.log_prob_draws[save_idx] =
          C.log_joint_common(y, n, spatial_contrib.begin(), prior_beta_sd,
                             prior_sigma_re_scale) +
          tulpa::pg_log_icar(phi.begin(), adj, tau, 0.0) +
          tulpa::pg_log_gamma(tau, prior_tau_shape, prior_tau_rate);
      save_idx++;
    }

    // Progress
    if (verbose && (iter + 1) % 500 == 0) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iter << std::endl;
    }

    // Check for user interrupt
    if ((iter + 1) % 100 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }

  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("beta") = C.beta_draws,
    Rcpp::Named("re") = C.re_draws,
    Rcpp::Named("sigma_re") = C.sigma_re_draws,
    Rcpp::Named("spatial_raw") = spatial_raw_draws,
    Rcpp::Named("spatial") = spatial_proj_draws,
    Rcpp::Named("tau") = tau_draws,
    Rcpp::Named("log_prob") = C.log_prob_draws
  );

  if (store_eta) {
    result["eta"] = C.eta_draws;
  }

  return result;
}
