// smc_sampler.cpp
// Rcpp interface for Sequential Monte Carlo sampler, and the definitions of
// the samplers declared in smc_sampler.h and smc_modeldata.h.
//
// Exports:
//   cpp_smc_test  — smoke test with Gaussian prior/likelihood (known analytic Z)

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <random>
#include "smc_sampler.h"
#include "smc_modeldata.h"

using namespace Rcpp;

// [[Rcpp::export]]
Rcpp::List cpp_smc_test(
    Rcpp::NumericVector mu_target,
    Rcpp::NumericVector sigma_target,
    int n_particles = 500,
    int n_mcmc_steps = 5,
    int seed = 42,
    double bridge_end = 1.0
) {
    int dim = mu_target.size();
    if (dim != sigma_target.size()) {
        stop("mu_target and sigma_target must have the same length");
    }
    if (!(bridge_end > 0.0 && bridge_end <= 1.0)) {
        stop("bridge_end must lie in (0, 1]");
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
    auto mutation = [&](std::vector<double>& x, double tau, std::mt19937& rng) {
        const double beta = tulpa_smc::prior_path_power(tau, bridge_end);
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

    // Prior path: the initial draws are exact prior draws, so the tempering
    // direction is the log-likelihood alone -- to bridge_end on the first leg,
    // then on to 1.
    std::vector<std::function<double(const std::vector<double>&)>> legs;
    legs.push_back([&](const std::vector<double>& x) { return bridge_end * log_lik(x); });
    if (bridge_end < 1.0) {
        legs.push_back([&](const std::vector<double>& x) {
            return (1.0 - bridge_end) * log_lik(x);
        });
    }
    auto result = tulpa_smc::smc_sample(
        legs, prior_sample, mutation,
        dim, n_particles, 0.5, n_mcmc_steps,
        static_cast<unsigned int>(seed)
    );

    // Equally weighted population moments.
    int N = n_particles;
    auto moments = [dim, N](const std::vector<std::vector<double>>& pop,
                            Rcpp::NumericVector& means, Rcpp::NumericVector& sds) {
        means = Rcpp::NumericVector(dim, 0.0);
        sds = Rcpp::NumericVector(dim, 0.0);
        for (int n = 0; n < N; n++)
            for (int i = 0; i < dim; i++) means[i] += pop[n][i] / N;
        for (int n = 0; n < N; n++)
            for (int i = 0; i < dim; i++) {
                double diff = pop[n][i] - means[i];
                sds[i] += diff * diff / N;
            }
        for (int i = 0; i < dim; i++) sds[i] = std::sqrt(sds[i]);
    };
    Rcpp::NumericVector means, sds;
    moments(result.particles, means, sds);

    // The tempered population at bridge_end, and its mean log-likelihood (the
    // WBIC average when bridge_end = 1 / log n).
    Rcpp::NumericVector t_means, t_sds;
    double t_mean_loglik = NA_REAL;
    if (!result.leg_end_particles.empty()) {
        const auto& pop = result.leg_end_particles.front();
        moments(pop, t_means, t_sds);
        t_mean_loglik = 0.0;
        for (int n = 0; n < N; n++) t_mean_loglik += log_lik(pop[n]) / N;
    }

    return Rcpp::List::create(
        Rcpp::Named("means") = means,
        Rcpp::Named("sds") = sds,
        Rcpp::Named("tempered_means") = t_means,
        Rcpp::Named("tempered_sds") = t_sds,
        Rcpp::Named("tempered_mean_loglik") = t_mean_loglik,
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

namespace tulpa_smc {

SMCResult smc_sample(
    const std::vector<std::function<double(const std::vector<double>&)>>& legs,
    const std::function<void(std::vector<double>&, std::mt19937&, double)>& initial_sample,
    const std::function<void(std::vector<double>&, double, std::mt19937&)>& mcmc_mutation,
    int dim,
    int n_particles,
    double ess_threshold,
    int n_mcmc_steps,
    unsigned int seed,
    const std::function<double(const std::vector<double>&)>& initial_log_weight,
    const std::function<void(const std::vector<std::vector<double>>&, double)>&
        prepare_mutation) {
    int N = n_particles;
    std::mt19937 rng(seed);
    const int n_legs = static_cast<int>(legs.size());
    if (n_legs < 1) {
        throw std::invalid_argument("tulpa SMC: the path needs at least one leg.");
    }

    SMCResult result;
    result.log_marginal_likelihood = 0.0;
    result.n_resamples = 0;
    result.n_mutations = 0;
    result.temperatures.push_back(0.0);

    // ------------------------------------------------------------------
    // 1. Initialize: draw N particles from q
    // ------------------------------------------------------------------
    std::vector<std::vector<double>> particles(N, std::vector<double>(dim));
    for (int n = 0; n < N; n++) {
        initial_sample(particles[n], rng, 0.0);
    }

    // Tempering direction h of the current leg at each particle, refreshed
    // after every mutation and when a leg begins.
    int leg = 0;
    std::vector<double> incr(N);
    for (int n = 0; n < N; n++) {
        incr[n] = legs[leg](particles[n]);
    }

    double tau = 0.0;
    double ess_target = ess_threshold * N;

    // Reweight by exp(log_w), accumulate the normalizing-constant increment,
    // and resample. Resampling at every step leaves the particles equally
    // weighted, which is exactly the assumption the equal-weight temperature
    // search (find_next_temperature), the log-mean-weight Z increment, and the
    // uniform final weights all rely on. Adaptive resampling (only when
    // ess < target) would instead require carrying normalized weights across
    // rounds and feeding them back into all three. Resampling unconditionally
    // keeps these consistent (Del Moral et al. 2006); the modest extra Monte
    // Carlo variance is offset by the per-particle MCMC mutations that follow.
    auto reweight_resample = [&](const std::vector<double>& log_w) {
        double max_lw = *std::max_element(log_w.begin(), log_w.end());
        double sum_w = 0.0;
        for (int n = 0; n < N; n++) sum_w += std::exp(log_w[n] - max_lw);
        result.log_marginal_likelihood += max_lw + std::log(sum_w / N);
        std::vector<double> weights(N);
        for (int n = 0; n < N; n++) {
            weights[n] = std::exp(log_w[n] - max_lw) / sum_w;
        }
        result.ess_history.push_back(compute_ess(log_w));

        auto ancestors = systematic_resample(weights, N, rng);
        auto old_particles = particles;
        auto old_incr = incr;
        for (int n = 0; n < N; n++) {
            particles[n] = old_particles[ancestors[n]];
            incr[n] = old_incr[ancestors[n]];
        }
        result.n_resamples++;
    };

    auto mutate = [&]() {
        if (prepare_mutation) prepare_mutation(particles, tau);
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < n_mcmc_steps; k++) {
                mcmc_mutation(particles[n], tau, rng);
            }
            incr[n] = legs[leg](particles[n]);
        }
        result.n_mutations += N * n_mcmc_steps;
    };

    // A population drawn from q but meant to represent pi_0 != q is corrected
    // once, up front, by importance weights pi_0 / q, then moved at beta = 0.
    if (initial_log_weight) {
        std::vector<double> log_w0(N);
        for (int n = 0; n < N; n++) log_w0[n] = initial_log_weight(particles[n]);
        if (!std::isfinite(*std::max_element(log_w0.begin(), log_w0.end()))) {
            throw std::runtime_error(
                "SMC: no initial particle carries a finite pi_0 / q weight.");
        }
        reweight_resample(log_w0);
        mutate();
    }

    // ------------------------------------------------------------------
    // 2. Tempering loop
    // ------------------------------------------------------------------
    int n_steps = 0;
    for (leg = 0; leg < n_legs; leg++) {
        if (leg > 0) {
            for (int n = 0; n < N; n++) incr[n] = legs[leg](particles[n]);
        }
        double t = 0.0;
        while (t < 1.0) {
            Rcpp::checkUserInterrupt();
            if (++n_steps > kMaxTemperatureSteps) {
                throw std::runtime_error(
                    "SMC tempering did not reach the end of its path within " +
                    std::to_string(kMaxTemperatureSteps) + " steps (stalled at "
                    "leg " + std::to_string(leg + 1) + " of " +
                    std::to_string(n_legs) + ", t = " + std::to_string(t) +
                    "). The ESS target is unreachable at every temperature "
                    "above it, which a particle carrying a non-finite "
                    "log-likelihood produces; check the initial population "
                    "and the likelihood at it.");
            }

            // (a) Adaptive temperature selection within the leg
            double t_new = find_next_temperature(incr, t, ess_target, N);
            double delta = t_new - t;

            // (b) Incremental log-weights, reweight and resample
            std::vector<double> log_w(N);
            for (int n = 0; n < N; n++) log_w[n] = delta * incr[n];
            reweight_resample(log_w);

            // (c) MCMC mutations at the new temperature
            t = t_new;
            tau = leg + t;
            result.temperatures.push_back(tau);
            mutate();
        }
        if (leg + 1 < n_legs) result.leg_end_particles.push_back(particles);
    }

    // ------------------------------------------------------------------
    // 3. Final: uniform weights after last resampling
    // ------------------------------------------------------------------
    result.particles = particles;
    result.weights.assign(N, 1.0 / N);
    result.n_temperatures = static_cast<int>(result.temperatures.size()) - 1;  // exclude beta=0

    return result;
}

} // namespace tulpa_smc

namespace tulpa {

SMCDriverResult run_smc_sampler(
    const std::vector<double>& init,
    const ModelData& data,
    const ParamLayout& layout,
    const SMCConfig& cfg,
    SmcMutationFn mutation_fn,
    void* user_data
) {
    SMCDriverResult out;
    int dim = static_cast<int>(init.size());
    if (dim == 0) {
        out.success = false;
        out.error_msg = "init has zero length";
        return out;
    }
    const double b = cfg.bridge_end;
    if (!(b > 0.0 && b <= 1.0)) {
        out.success = false;
        out.error_msg = "bridge_end must lie in (0, 1]";
        return out;
    }
    const bool two_legs = b < 1.0;

    // Closures: prior + likelihood. log_prior + log_lik_only sums exactly to
    // compute_log_post by construction (hmc_sampler_decls.h).
    auto log_prior_fn = [&data, &layout](const std::vector<double>& theta) -> double {
        return smc_finite_or_neg_inf(tulpa_hmc::compute_log_prior(theta, data, layout));
    };
    auto log_lik_fn = [&data, &layout](const std::vector<double>& theta) -> double {
        return smc_finite_or_neg_inf(tulpa_hmc::compute_log_lik_only(theta, data, layout));
    };

    const double sigma_ref = (cfg.prior_sigma > 0.0) ? cfg.prior_sigma : 1.0;
    auto log_ref_fn = [&init, sigma_ref](const std::vector<double>& theta) -> double {
        return smc_gaussian_reference_log_density(theta, init, sigma_ref);
    };
    auto ref_sample = [&init, sigma_ref, dim](
        std::vector<double>& theta, std::mt19937& rng, double /*beta*/
    ) {
        std::normal_distribution<double> normal(0.0, sigma_ref);
        for (int i = 0; i < dim; i++) {
            theta[i] = init[i] + normal(rng);
        }
    };

    using IncrementFn = std::function<double(const std::vector<double>&)>;
    std::vector<IncrementFn> legs;
    std::function<double(const std::vector<double>&)> initial_log_weight;
    std::function<void(std::vector<double>&, double, std::mt19937&)> mutation;
    std::function<void(const std::vector<std::vector<double>>&, double)> prepare;

    SmcPopulationHmc hmc(dim);
    hmc.verbose = cfg.verbose;
    if (mutation_fn != nullptr) {
        const ModelData* data_ptr = &data;
        const ParamLayout* layout_ptr = &layout;
        mutation = [mutation_fn, data_ptr, layout_ptr, user_data, dim, b](
            std::vector<double>& theta, double tau, std::mt19937& rng
        ) {
            // Generate a per-call seed from the rng so the user kernel
            // sees independent randomness across calls but stays
            // deterministic given the outer rng state.
            std::uniform_int_distribution<unsigned int> seed_dist(
                0u, std::numeric_limits<unsigned int>::max());
            unsigned int seed = seed_dist(rng);
            mutation_fn(theta.data(), dim, tulpa_smc::prior_path_power(tau, b),
                        data_ptr, layout_ptr, seed, user_data);
        };
        legs.push_back([&log_lik_fn, b](const std::vector<double>& theta) {
            return smc_finite_or_neg_inf(b * log_lik_fn(theta));
        });
        initial_log_weight = [&log_prior_fn, &log_ref_fn](const std::vector<double>& theta) {
            return log_prior_fn(theta) - log_ref_fn(theta);
        };
    } else {
        legs.push_back([&log_prior_fn, &log_lik_fn, &log_ref_fn, b](
            const std::vector<double>& theta
        ) -> double {
            const double lp = log_prior_fn(theta);
            if (!std::isfinite(lp)) return lp;
            return smc_finite_or_neg_inf(lp + b * log_lik_fn(theta) - log_ref_fn(theta));
        });
        // (1 - s) log q + s (log p + lambda log L) and its gradient. The
        // log-posterior half is one fused compute_gradient pass; off lambda = 1
        // the prior gradient is taken as well, since grad log L is the
        // difference of the two.
        tulpa_hmc::GradientFn prior_grad_fn = two_legs
            ? tulpa_hmc::resolve_prior_gradient_fn(tulpa_hmc::get_gradient_mode(),
                                                   data, layout)
            : nullptr;
        auto log_target = [&data, &layout, &init, sigma_ref, b, prior_grad_fn](
            const std::vector<double>& theta, double tau,
            std::vector<double>& grad
        ) -> double {
            const tulpa_smc::PathPoint pt = tulpa_smc::path_point(tau, b);
            const double s = pt.s;
            const double lambda = pt.lambda;
            const double lq = smc_gaussian_reference_log_density(theta, init, sigma_ref);
            double lpost = 0.0;
            tulpa_hmc::compute_gradient(theta, data, layout, grad, &lpost);
            double lpath = lpost;
            if (lambda != 1.0) {
                std::vector<double> gprior(theta.size());
                double lprior = 0.0;
                prior_grad_fn(theta, data, layout, gprior, &lprior);
                lpath = lprior + lambda * (lpost - lprior);
                for (size_t j = 0; j < theta.size(); j++) {
                    grad[j] = (1.0 - lambda) * gprior[j] + lambda * grad[j];
                }
            }
            const double inv_s2 = 1.0 / (sigma_ref * sigma_ref);
            for (size_t j = 0; j < theta.size(); j++) {
                grad[j] = s * grad[j] - (1.0 - s) * (theta[j] - init[j]) * inv_s2;
            }
            if (!std::isfinite(lpath)) return -std::numeric_limits<double>::infinity();
            return (1.0 - s) * lq + s * lpath;
        };
        mutation = [&hmc, log_target](
            std::vector<double>& theta, double tau, std::mt19937& rng
        ) {
            hmc.step(theta, tau, rng, log_target);
        };
        prepare = [&hmc](const std::vector<std::vector<double>>& particles,
                         double tau) {
            hmc.adapt(particles);
            if (hmc.verbose) Rcpp::Rcout << "SMC path coordinate " << tau << "\n";
        };
    }
    // Leg 2, on either path: the population already represents p L^b, and
    // (1 - b) log L carries it to the posterior.
    if (two_legs) {
        legs.push_back([&log_lik_fn, b](const std::vector<double>& theta) {
            return smc_finite_or_neg_inf((1.0 - b) * log_lik_fn(theta));
        });
    }

    try {
        auto res = tulpa_smc::smc_sample(
            legs, ref_sample, mutation,
            dim,
            cfg.n_particles,
            cfg.ess_threshold,
            cfg.n_mcmc_steps,
            cfg.seed,
            initial_log_weight,
            prepare
        );

        out.particles  = std::move(res.particles);
        if (two_legs) {
            out.tempered_particles = std::move(res.leg_end_particles.front());
            out.tempered_beta = b;
        }
        out.log_weights.resize(res.weights.size());
        for (size_t i = 0; i < res.weights.size(); ++i) {
            out.log_weights[i] = (res.weights[i] > 0.0)
                ? std::log(res.weights[i])
                : -std::numeric_limits<double>::infinity();
        }
        // Not res.log_marginal_likelihood: see SMCDriverResult::log_evidence.
        out.log_evidence = std::numeric_limits<double>::quiet_NaN();
        out.log_evidence_valid = false;
        out.success = true;
    } catch (Rcpp::internal::InterruptedException&) {
        throw;
    } catch (const std::exception& e) {
        out.success = false;
        out.error_msg = std::string("SMC failed: ") + e.what();
    } catch (...) {
        out.success = false;
        out.error_msg = "SMC failed: unknown exception";
    }

    return out;
}

} // namespace tulpa
