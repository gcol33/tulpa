// smc_modeldata.h
// Generic ModelData / ParamLayout driver for the SMC sampler.
//
// Builds log_prior + log_likelihood closures from
// compute_log_prior + compute_log_lik_only (defined in hmc_sampler.h)
// and dispatches to tulpa_smc::smc_sample (see smc_sampler.h).
//
// The mutation kernel is pluggable via a user-supplied SmcMutationFn
// function pointer. If null, a built-in random-walk Metropolis kernel is
// used, scaling the proposal SD by 1 / sqrt(beta) and targeting
//   compute_log_prior(theta) + beta * compute_log_lik_only(theta).
//
// The default prior_sample is a Gaussian perturbation around `init` with
// SD `cfg.prior_sigma` (default 1.0). This is a smoke-test default —
// proper prior draws need per-prior closed-form samplers tulpa lacks
// generically. Because the particles are therefore not draws from p(theta),
// the SMC log-Z accumulator estimates a different integral than the marginal
// likelihood and SMCDriverResult::log_evidence is reported as NaN.

#ifndef TULPA_SMC_MODELDATA_H
#define TULPA_SMC_MODELDATA_H

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <limits>
#include <functional>
#include <utility>

#include "hmc_sampler.h"
#include "smc_sampler.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"

namespace tulpa {

// ----------------------------------------------------------------------------
// Function-pointer mutation kernel
//   Mutates `theta` in place at temperature `beta`. The kernel may inspect
//   `data` and `layout` for problem-specific moves, and `user_data` for
//   any model-side state. `rng_seed` is per-call so the kernel can build
//   its own std::mt19937 deterministically.
// ----------------------------------------------------------------------------
typedef void (*SmcMutationFn)(
    double* theta, int n_params, double beta,
    const ModelData* data, const ParamLayout* layout,
    unsigned int rng_seed, void* user_data);

// ----------------------------------------------------------------------------
// SMC config: mirror smc_sample's tunables, plus the prior_sample default
// SD.
// ----------------------------------------------------------------------------
struct SMCConfig {
    int n_particles    = 500;
    int n_mcmc_steps   = 5;
    double ess_threshold = 0.5;   // resample when ESS < threshold * N
    double prior_sigma   = 1.0;   // default prior_sample SD
    unsigned int seed    = 42;
    bool verbose         = false;
};

struct SMCDriverResult {
    std::vector<std::vector<double>> particles;
    std::vector<double> log_weights;   // log of normalized final weights

    // log Z. The SMC increment log mean_n L(theta_n)^(beta_t - beta_{t-1})
    // estimates Z_t / Z_{t-1} only when the step-0 population represents
    // pi_0(theta) = p(theta), i.e. the particles are prior draws. This driver
    // initializes them from a Gaussian perturbation of `init` and carries no
    // p / q correction in the weights, so no evidence is available: the field
    // is NaN and log_evidence_valid is false. The draws themselves recover,
    // because the MCMC mutations target p(theta) L(theta)^beta.
    double log_evidence = std::numeric_limits<double>::quiet_NaN();
    bool log_evidence_valid = false;

    bool success = true;
    std::string error_msg;
};

// ----------------------------------------------------------------------------
// Built-in random-walk Metropolis kernel (used when mutation_fn == nullptr).
// Proposal SD = 2.38 / sqrt(d) * (1 / sqrt(max(beta, eps))) so the kernel
// gets larger jumps near the prior (beta -> 0) and shrinks toward the
// posterior. Targets log_prior + beta * log_lik.
// ----------------------------------------------------------------------------
inline void smc_default_rwm_step(
    std::vector<double>& theta,
    double beta,
    std::mt19937& rng,
    const std::function<double(const std::vector<double>&)>& log_prior_fn,
    const std::function<double(const std::vector<double>&)>& log_lik_fn
) {
    int d = static_cast<int>(theta.size());
    if (d == 0) return;

    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);

    double beta_eff = std::max(beta, 1e-3);
    double scale = (2.38 / std::sqrt(static_cast<double>(d)))
                 / std::sqrt(beta_eff);

    double lp_curr = log_prior_fn(theta) + beta * log_lik_fn(theta);

    std::vector<double> prop = theta;
    for (int i = 0; i < d; i++) {
        prop[i] += scale * normal(rng);
    }
    double lp_prop = log_prior_fn(prop) + beta * log_lik_fn(prop);

    if (std::isfinite(lp_prop) && std::log(unif(rng)) < lp_prop - lp_curr) {
        theta = prop;
    }
}

// ----------------------------------------------------------------------------
// Main driver: builds closures + invokes tulpa_smc::smc_sample.
// ----------------------------------------------------------------------------
inline SMCDriverResult run_smc_sampler(
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

    // Closures: prior + likelihood. Real factoring landed via
    // prereq (tulpa_hmc::compute_log_prior /
    // compute_log_lik_only in hmc_sampler.h). log_prior + log_lik_only
    // sums exactly to compute_log_post by construction.
    auto log_prior_fn = [&data, &layout](const std::vector<double>& theta) -> double {
        return tulpa_hmc::compute_log_prior(theta, data, layout);
    };
    auto log_lik_fn = [&data, &layout](const std::vector<double>& theta) -> double {
        return tulpa_hmc::compute_log_lik_only(theta, data, layout);
    };

    // prior_sample: Gaussian perturbation around `init`. Smoke-test default
    // — see file header comment.
    double sigma_prior = (cfg.prior_sigma > 0.0) ? cfg.prior_sigma : 1.0;
    auto prior_sample = [&init, sigma_prior, dim](
        std::vector<double>& theta, std::mt19937& rng, double /*beta*/
    ) {
        std::normal_distribution<double> normal(0.0, sigma_prior);
        for (int i = 0; i < dim; i++) {
            theta[i] = init[i] + normal(rng);
        }
    };

    // Mutation: user-supplied or built-in RWM.
    std::function<void(std::vector<double>&, double, std::mt19937&)> mutation;
    if (mutation_fn != nullptr) {
        const ModelData* data_ptr = &data;
        const ParamLayout* layout_ptr = &layout;
        mutation = [mutation_fn, data_ptr, layout_ptr, user_data, dim](
            std::vector<double>& theta, double beta, std::mt19937& rng
        ) {
            // Generate a per-call seed from the rng so the user kernel
            // sees independent randomness across calls but stays
            // deterministic given the outer rng state.
            std::uniform_int_distribution<unsigned int> seed_dist(
                0u, std::numeric_limits<unsigned int>::max());
            unsigned int seed = seed_dist(rng);
            mutation_fn(theta.data(), dim, beta,
                        data_ptr, layout_ptr, seed, user_data);
        };
    } else {
        mutation = [&log_prior_fn, &log_lik_fn](
            std::vector<double>& theta, double beta, std::mt19937& rng
        ) {
            smc_default_rwm_step(theta, beta, rng, log_prior_fn, log_lik_fn);
        };
    }

    try {
        auto res = tulpa_smc::smc_sample(
            log_prior_fn, log_lik_fn, prior_sample, mutation,
            dim,
            cfg.n_particles,
            cfg.ess_threshold,
            cfg.n_mcmc_steps,
            cfg.seed
        );

        out.particles  = std::move(res.particles);
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

#endif // TULPA_SMC_MODELDATA_H
