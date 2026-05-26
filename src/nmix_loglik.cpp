// nmix_loglik.cpp
// Rcpp surface for the N-mixture marginal log-likelihood kernel
// (see nmix_kernel.h). Provides a thin pass-through used both for
// validation against `unmarked::pcount` and as the per-iteration evaluator
// the Laplace Newton driver will call (forthcoming).
//
// Data layout (long form):
//   y          int[n_obs]   observed counts, visits stacked across sites
//   site_idx   int[n_obs]   1-based site index for each visit
//   eta_p      double[n_obs] detection logit linear predictor per visit
//   eta_lambda double[n_sites] abundance log linear predictor per site
//
// All exported quantities aggregate site-level kernel outputs:
//   log_lik           = sum_i log L_i
//   grad_eta_lambda   = (E[N|y_i] - lambda_i)   per site
//   grad_eta_p        = (y_ij - E[N|y_i] p_ij)  per visit
//   info_eta_lambda   = lambda_i                per site  (complete-data Fisher)
//   info_eta_p        = E[N|y_i] p_ij (1-p_ij)  per visit (complete-data Fisher)
//   mean_N, var_N     posterior moments per site (diagnostics, observed-info path)
//   boundary_weight   posterior mass on N = K_max per site (K_max sanity check)

#include "nmix_kernel.h"
#include <Rcpp.h>
#include <vector>

// [[Rcpp::export]]
Rcpp::List cpp_nmix_total_log_lik(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector site_idx,
    Rcpp::NumericVector eta_p,
    Rcpp::NumericVector eta_lambda,
    int K_max,
    double r          // NB size; pass Inf for the Poisson kernel
) {
    const int n_obs = y.size();
    const int n_sites = eta_lambda.size();
    if (site_idx.size() != n_obs) {
        Rcpp::stop("site_idx length must match y length (%d vs %d).",
                   site_idx.size(), n_obs);
    }
    if (eta_p.size() != n_obs) {
        Rcpp::stop("eta_p length must match y length (%d vs %d).",
                   eta_p.size(), n_obs);
    }
    if (K_max < 0) Rcpp::stop("K_max must be >= 0.");
    if (R_finite(r) && r <= 0.0) Rcpp::stop("r (NB size) must be > 0.");

    // Group observations by site (preserves input order within each site).
    std::vector<std::vector<int>> obs_by_site(n_sites);
    for (int o = 0; o < n_obs; ++o) {
        const int s = site_idx[o] - 1;
        if (s < 0 || s >= n_sites) {
            Rcpp::stop("site_idx[%d] = %d out of range [1, %d].",
                       o + 1, site_idx[o], n_sites);
        }
        obs_by_site[s].push_back(o);
    }

    // Outputs.
    double total_log_lik = 0.0;
    double total_grad_theta = 0.0;   // sum_i d log L_i / d theta (NB only; 0 for Poisson)
    Rcpp::NumericVector grad_eta_lambda(n_sites);
    Rcpp::NumericVector info_eta_lambda(n_sites);
    Rcpp::NumericVector mean_N(n_sites);
    Rcpp::NumericVector var_N(n_sites);
    Rcpp::NumericVector boundary_weight(n_sites);
    Rcpp::NumericVector grad_eta_p(n_obs);
    Rcpp::NumericVector info_eta_p(n_obs);
    int n_K_inadmissible = 0;

    for (int s = 0; s < n_sites; ++s) {
        const auto& idx = obs_by_site[s];
        const int J = (int)idx.size();
        if (J == 0) {
            // Site with no visits: marginal collapses to 1 (log_lik = 0),
            // posterior over N equals the Poisson prior. The site contributes
            // *zero* marginal information about lambda (no data), so info = 0
            // even though the complete-data Fisher would be lambda.
            grad_eta_lambda[s] = 0.0;
            info_eta_lambda[s] = 0.0;
            mean_N[s] = std::exp(eta_lambda[s]);
            var_N[s]  = std::exp(eta_lambda[s]);
            boundary_weight[s] = 0.0;
            continue;
        }
        std::vector<int>    y_site(J);
        std::vector<double> eta_p_site(J);
        for (int j = 0; j < J; ++j) {
            y_site[j]     = y[idx[j]];
            eta_p_site[j] = eta_p[idx[j]];
        }
        tulpa::NMixSiteResult res = tulpa::compute_nmix_site(
            y_site.data(), eta_p_site.data(), J,
            eta_lambda[s], K_max, r
        );
        if (!R_finite(res.log_lik)) ++n_K_inadmissible;
        total_log_lik += res.log_lik;
        total_grad_theta += res.grad_theta;
        grad_eta_lambda[s] = res.grad_eta_lambda;
        info_eta_lambda[s] = res.info_eta_lambda;
        mean_N[s] = res.mean_N;
        var_N[s]  = res.var_N;
        boundary_weight[s] = res.boundary_weight;
        for (int j = 0; j < J; ++j) {
            grad_eta_p[idx[j]] = res.grad_eta_p[j];
            info_eta_p[idx[j]] = res.info_eta_p[j];
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("log_lik")          = total_log_lik,
        Rcpp::Named("grad_eta_lambda")  = grad_eta_lambda,
        Rcpp::Named("grad_eta_p")       = grad_eta_p,
        Rcpp::Named("grad_theta")       = total_grad_theta,
        Rcpp::Named("info_eta_lambda")  = info_eta_lambda,
        Rcpp::Named("info_eta_p")       = info_eta_p,
        Rcpp::Named("mean_N")           = mean_N,
        Rcpp::Named("var_N")            = var_N,
        Rcpp::Named("boundary_weight")  = boundary_weight,
        Rcpp::Named("n_K_inadmissible") = n_K_inadmissible
    );
}
