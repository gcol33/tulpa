// smc_sampler.cpp
// Rcpp interface for Sequential Monte Carlo sampler
//
// Exports:
//   cpp_smc_test  — smoke test with Gaussian prior/likelihood (known analytic Z)

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <random>
#include "smc_sampler.h"

using namespace Rcpp;

// [[Rcpp::export]]
Rcpp::List cpp_smc_test(
    Rcpp::NumericVector mu_target,
    Rcpp::NumericVector sigma_target,
    int n_particles = 500,
    int n_mcmc_steps = 5,
    int seed = 42
) {
    int dim = mu_target.size();
    if (dim != sigma_target.size()) {
        stop("mu_target and sigma_target must have the same length");
    }

    std::vector<double> mu_t(mu_target.begin(), mu_target.end());
    std::vector<double> sig_t(sigma_target.begin(), sigma_target.end());

    // Prior: N(0, sigma_prior^2) with sigma_prior = 10
    double sigma_prior = 10.0;

    // Unnormalized kernels — omit all constants that don't depend on theta.
    // This makes log Z = log integral exp(log_prior + log_lik) d theta
    // which has a known analytic solution for Gaussian-Gaussian conjugacy.

    auto log_prior = [&](const std::vector<double>& x) -> double {
        double lp = 0.0;
        for (int i = 0; i < dim; i++) {
            lp -= 0.5 * x[i] * x[i] / (sigma_prior * sigma_prior);
        }
        return lp;
    };

    auto log_lik = [&](const std::vector<double>& x) -> double {
        double ll = 0.0;
        for (int i = 0; i < dim; i++) {
            double z = (x[i] - mu_t[i]) / sig_t[i];
            ll -= 0.5 * z * z;
        }
        return ll;
    };

    auto prior_sample = [&](std::vector<double>& x, std::mt19937& rng, double) {
        std::normal_distribution<double> normal(0.0, sigma_prior);
        for (int i = 0; i < dim; i++) {
            x[i] = normal(rng);
        }
    };

    // Random-walk Metropolis targeting the tempered posterior.
    // Proposal SD is set analytically from the tempered posterior variance:
    //   v_i = sp^2 * st_i^2 / (st_i^2 + beta * sp^2)
    // Optimal RWM scaling: prop_sd_i = 2.38 * sqrt(v_i) / sqrt(d)
    auto mutation = [&](std::vector<double>& x, double beta, std::mt19937& rng) {
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        double lp_curr = log_prior(x) + beta * log_lik(x);

        double scale = 2.38 / std::sqrt(static_cast<double>(dim));
        std::vector<double> prop = x;
        for (int i = 0; i < dim; i++) {
            double v_i = (sigma_prior * sigma_prior * sig_t[i] * sig_t[i])
                       / (sig_t[i] * sig_t[i] + beta * sigma_prior * sigma_prior);
            std::normal_distribution<double> normal(0.0, scale * std::sqrt(v_i));
            prop[i] += normal(rng);
        }
        double lp_prop = log_prior(prop) + beta * log_lik(prop);

        if (std::log(unif(rng)) < lp_prop - lp_curr) {
            x = prop;
        }
    };

    auto result = tulpa_smc::smc_sample(
        log_prior, log_lik, prior_sample, mutation,
        dim, n_particles, 0.5, n_mcmc_steps,
        static_cast<unsigned int>(seed)
    );

    // Compute weighted posterior mean
    int N = n_particles;
    Rcpp::NumericVector means(dim, 0.0);
    for (int n = 0; n < N; n++) {
        for (int i = 0; i < dim; i++) {
            means[i] += result.weights[n] * result.particles[n][i];
        }
    }

    // Compute weighted posterior SD
    Rcpp::NumericVector sds(dim, 0.0);
    for (int n = 0; n < N; n++) {
        for (int i = 0; i < dim; i++) {
            double diff = result.particles[n][i] - means[i];
            sds[i] += result.weights[n] * diff * diff;
        }
    }
    for (int i = 0; i < dim; i++) {
        sds[i] = std::sqrt(sds[i]);
    }

    return Rcpp::List::create(
        Rcpp::Named("means") = means,
        Rcpp::Named("sds") = sds,
        Rcpp::Named("log_marginal_likelihood") = result.log_marginal_likelihood,
        Rcpp::Named("n_temperatures") = result.n_temperatures,
        Rcpp::Named("n_particles") = N,
        Rcpp::Named("n_resamples") = result.n_resamples,
        Rcpp::Named("n_mutations") = result.n_mutations,
        Rcpp::Named("temperatures") = Rcpp::NumericVector(
            result.temperatures.begin(), result.temperatures.end()
        ),
        Rcpp::Named("ess_history") = Rcpp::NumericVector(
            result.ess_history.begin(), result.ess_history.end()
        )
    );
}
