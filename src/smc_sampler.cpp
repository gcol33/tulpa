// smc_sampler.cpp
// Definitions of the Sequential Monte Carlo samplers declared in
// smc_sampler.h and smc_modeldata.h.

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <random>
#include "smc_sampler.h"
#include "smc_modeldata.h"
#include "laplace_builtin_family_spec.h"  // builtin_family_ll_is_normalized

using namespace Rcpp;

namespace tulpa_smc {

SMCResult smc_sample(
    const std::vector<std::function<double(const std::vector<double>&)>>& legs,
    const std::function<void(std::vector<double>&, std::mt19937&, double)>& initial_sample,
    const std::function<void(std::vector<double>&, double, std::mt19937&, int)>& mcmc_mutation,
    int dim,
    int n_particles,
    double ess_threshold,
    int min_moves,
    int max_moves,
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
    if (min_moves < 1 || max_moves < min_moves) {
        throw std::invalid_argument(
            "tulpa SMC: the move steps per tempering step need 1 <= min <= max "
            "(min = " + std::to_string(min_moves) + ", max = " +
            std::to_string(max_moves) + ").");
    }

    SMCResult result;
    result.log_marginal_likelihood = 0.0;
    result.log_marginal_likelihood_var = 0.0;
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
        const double ess = compute_ess(log_w);
        result.ess_history.push_back(ess);
        result.log_marginal_likelihood_var += 1.0 / ess - 1.0 / N;

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
        MoveDecorrelation rule(particles);
        int k = 0;
        while (true) {
            for (int n = 0; n < N; n++) mcmc_mutation(particles[n], tau, rng, n);
            k++;
            const bool decorrelated = rule.decorrelated(particles);
            if (k >= max_moves || (k >= min_moves && decorrelated)) break;
            Rcpp::checkUserInterrupt();
        }
        for (int n = 0; n < N; n++) incr[n] = legs[leg](particles[n]);
        result.n_mutations += N * k;
        result.moves_history.push_back(k);
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
    std::function<void(std::vector<double>&, double, std::mt19937&, int)> mutation;
    std::function<void(const std::vector<std::vector<double>>&, double)> prepare;

    SmcPopulationHmc hmc(dim);
    hmc.verbose = cfg.verbose;
    if (mutation_fn != nullptr) {
        const ModelData* data_ptr = &data;
        const ParamLayout* layout_ptr = &layout;
        mutation = [mutation_fn, data_ptr, layout_ptr, user_data, dim, b](
            std::vector<double>& theta, double tau, std::mt19937& rng, int /*particle*/
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
            std::vector<double>& theta, double tau, std::mt19937& rng, int particle
        ) {
            hmc.step(theta, tau, rng, log_target, particle);
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
            cfg.min_moves,
            cfg.max_moves,
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
        std::vector<std::string> uncarried;
        const double prior_norm =
            tulpa_hmc::compute_log_prior_normalizer(data, layout, &uncarried);
        const bool lik_normalized = builtin_family_ll_is_normalized(data);
        if (!lik_normalized) {
            out.log_evidence_declined = "likelihood_not_normalized";
        } else if (!std::isfinite(prior_norm)) {
            out.log_evidence_declined = "prior_normalizer_not_carried:";
            for (size_t k = 0; k < uncarried.size(); k++) {
                out.log_evidence_declined += (k ? "," : "") + uncarried[k];
            }
        } else if (!std::isfinite(res.log_marginal_likelihood)) {
            out.log_evidence_declined = "non_finite_path_estimate";
        } else {
            out.log_evidence = res.log_marginal_likelihood + prior_norm;
            out.log_evidence_se = std::sqrt(res.log_marginal_likelihood_var);
            out.log_evidence_valid = true;
        }
        out.moves_history = std::move(res.moves_history);
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
