// mclmc.cpp
// Rcpp exports for MCLMC / MAMCLMC sampler
//
// Provides a test function that samples from a diagonal Gaussian target
// to verify the sampler recovers known means and standard deviations.

#include "mclmc.h"
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::List cpp_mclmc_test(
    Rcpp::NumericVector mu_target,
    Rcpp::NumericVector sigma_target,
    Rcpp::NumericVector init,
    int n_iter = 2000,
    int n_warmup = 1000,
    int seed = 42,
    bool adjusted = false
) {
    int dim = mu_target.size();
    if (dim < 1) Rcpp::stop("cpp_mclmc_test: mu_target is empty.");
    if ((int)sigma_target.size() != dim) {
        Rcpp::stop("cpp_mclmc_test: sigma_target has length %d but mu_target "
                   "has length %d.", (int)sigma_target.size(), dim);
    }
    if ((int)init.size() != dim) {
        Rcpp::stop("cpp_mclmc_test: init has length %d but mu_target has "
                   "length %d.", (int)init.size(), dim);
    }
    for (int i = 0; i < dim; i++) {
        if (!(sigma_target[i] > 0.0) || !R_finite(sigma_target[i])) {
            Rcpp::stop("cpp_mclmc_test: sigma_target[%d] must be finite and "
                       "positive.", i + 1);
        }
    }
    std::vector<double> mu_t(mu_target.begin(), mu_target.end());
    std::vector<double> sig_t(sigma_target.begin(), sigma_target.end());

    // Diagonal Gaussian log-posterior and gradient
    auto log_post = [&](const std::vector<double>& x,
                        std::vector<double>& grad) -> double {
        double lp = 0.0;
        for (int i = 0; i < dim; i++) {
            double z = (x[i] - mu_t[i]) / sig_t[i];
            lp -= 0.5 * z * z;
            grad[i] = -z / sig_t[i];
        }
        return lp;
    };

    std::vector<double> x0(init.begin(), init.end());

    tulpa::MCLMCResult result = adjusted ?
        tulpa::mamclmc_sample(log_post, x0, dim, n_iter, n_warmup,
                              0.0, 0, {}, static_cast<unsigned int>(seed)) :
        tulpa::mclmc_sample(log_post, x0, dim, n_iter, n_warmup,
                            0.0, 0, {}, static_cast<unsigned int>(seed));

    // Compute posterior means and standard deviations from draws
    int n_draws = static_cast<int>(result.draws.size());
    if (n_draws < 2) {
        Rcpp::stop("cpp_mclmc_test: the run kept %d draw(s); the posterior SD "
                   "needs at least 2 (raise n_iter).", n_draws);
    }
    Rcpp::NumericVector means(dim, 0.0);
    Rcpp::NumericVector sds(dim, 0.0);

    for (int s = 0; s < n_draws; s++) {
        for (int i = 0; i < dim; i++) {
            means[i] += result.draws[s][i];
        }
    }
    for (int i = 0; i < dim; i++) {
        means[i] /= n_draws;
    }

    for (int s = 0; s < n_draws; s++) {
        for (int i = 0; i < dim; i++) {
            double d = result.draws[s][i] - means[i];
            sds[i] += d * d;
        }
    }
    for (int i = 0; i < dim; i++) {
        sds[i] = std::sqrt(sds[i] / (n_draws - 1));
    }

    // Also return all draws as a matrix for diagnostic use
    Rcpp::NumericMatrix draws_mat(n_draws, dim);
    for (int s = 0; s < n_draws; s++) {
        for (int i = 0; i < dim; i++) {
            draws_mat(s, i) = result.draws[s][i];
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("means") = means,
        Rcpp::Named("sds") = sds,
        Rcpp::Named("draws") = draws_mat,
        Rcpp::Named("lp") = Rcpp::wrap(result.lp),
        Rcpp::Named("n_draws") = n_draws,
        Rcpp::Named("step_size") = result.step_size_final,
        Rcpp::Named("L") = result.L_final,
        Rcpp::Named("accept_rate") = result.accept_rate,
        Rcpp::Named("n_grad_evals") = result.n_grad_evals,
        Rcpp::Named("n_divergences") = result.n_divergences
    );
}
