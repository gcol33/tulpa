// mclmc.cpp
// Rcpp exports for MCLMC / MAMCLMC sampler
//
// Provides a test function that samples from a diagonal Gaussian target
// to verify the sampler recovers known means and standard deviations, and
// defines the samplers declared in mclmc.h and mclmc_modeldata.h.

#include "mclmc.h"
#include "mclmc_modeldata.h"
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

namespace tulpa {

MCLMCResult mclmc_sample(
    const std::function<double(const std::vector<double>&,
                               std::vector<double>&)>& log_prob_grad,
    const std::vector<double>& init,
    int dim,
    int n_iter,
    int n_warmup,
    double step_size,
    int L,
    const std::vector<double>& mass_diag,
    unsigned int seed) {
    using namespace mclmc_detail;

    if (dim < 1) {
        throw std::invalid_argument(
            "tulpa MCLMC: dim must be at least 1, got " + std::to_string(dim) + ".");
    }
    if (static_cast<int>(init.size()) != dim) {
        throw std::invalid_argument(
            "tulpa MCLMC: init has " + std::to_string(init.size()) +
            " entries but dim is " + std::to_string(dim) + ".");
    }
    std::vector<double> mass, inv_mass;
    setup_mass(dim, mass_diag, mass, inv_mass);

    if (L <= 0) {
        // Need enough leapfrog steps per trajectory to give meaningful
        // exploration even in low dim; sqrt(d) alone gives L=1 for d=1.
        L = std::max(5, static_cast<int>(std::sqrt(static_cast<double>(dim))));
    }
    bool adapt_eps = (step_size <= 0.0);
    if (adapt_eps) {
        // Default: aggressive eps for unadjusted (no MH safety net needed)
        step_size = 0.5 / std::pow(static_cast<double>(dim), 0.25);
    }

    std::mt19937 rng(seed);

    std::vector<double> q(init.begin(), init.end());
    std::vector<double> p(dim);
    std::vector<double> grad(dim, 0.0);

    double lp = log_prob_grad(q, grad);
    long long n_grad_evals = 1;

    init_momentum(p, mass, inv_mass, dim, rng);

    int total_iter = n_warmup + n_iter;
    MCLMCResult result;
    result.draws.reserve(n_iter);
    result.lp.reserve(n_iter);

    int n_divergences = 0;
    double smoothed_error_ratio = kTargetEnergyRatio;
    double eps_div_min = std::numeric_limits<double>::infinity();

    std::vector<double> q_prop(dim);
    std::vector<double> p_prop(dim);
    std::vector<double> grad_prop(dim);

    for (int iter = 1; iter <= total_iter; iter++) {
        bool warmup = (iter <= n_warmup);
        double eps = step_size;
        const double alpha = kTrajectoryRefresh;

        // Always accept the leapfrog trajectory (unadjusted).
        std::copy(q.begin(), q.end(), q_prop.begin());
        std::copy(p.begin(), p.end(), p_prop.begin());
        std::copy(grad.begin(), grad.end(), grad_prop.begin());
        double lp_prop = lp;

        double H_old = -lp + kinetic_energy(p_prop, inv_mass, dim);

        for (int step = 0; step < L; step++) {
            lp_prop = leapfrog_step(q_prop, p_prop, grad_prop, inv_mass, eps,
                                    dim, log_prob_grad);
            n_grad_evals++;
        }

        double H_new = -lp_prop + kinetic_energy(p_prop, inv_mass, dim);
        double energy_error = std::abs(H_new - H_old);
        bool diverged = !std::isfinite(lp_prop) || !std::isfinite(H_new) ||
                        energy_error > kDivergenceThresh;

        if (diverged) {
            n_divergences++;
            // Record divergence-derived ceiling and shrink eps immediately.
            if (eps < eps_div_min) eps_div_min = eps;
            step_size = std::max(kEpsMin, 0.5 * step_size);
            // Restart momentum so we can recover from the diverged state
            init_momentum(p, mass, inv_mass, dim, rng);
        } else {
            std::copy(q_prop.begin(), q_prop.end(), q.begin());
            std::copy(p_prop.begin(), p_prop.end(), p.begin());
            std::copy(grad_prop.begin(), grad_prop.end(), grad.begin());
            lp = lp_prop;
        }

        // Canonical Langevin refresh — preserves p ~ N(0, mass).
        // (No energy-shell projection here: doing so would destroy the
        // canonical p-marginal that leapfrog conserves and bias the q-chain.
        // The MH-adjusted variant keeps project_energy because the MH step
        // corrects whatever the proposal kernel does.)
        partial_refresh(p, mass, alpha, dim, rng);

        // Step-size adaptation during warmup:
        // energy error scales as eps^3 for leapfrog, so adjust eps
        // by (target/observed)^{1/3} to move towards the target.
        if (warmup && adapt_eps && !diverged) {
            double observed_ratio = energy_error / static_cast<double>(dim);
            smoothed_error_ratio = kSmoothAlpha * observed_ratio
                                 + (1.0 - kSmoothAlpha) * smoothed_error_ratio;
            if (iter > 10) {
                double ratio = kTargetEnergyRatio / (smoothed_error_ratio + 1e-10);
                double factor = std::pow(std::max(0.2, std::min(ratio, 3.0)),
                                         1.0 / 3.0);
                step_size *= factor;
                step_size = std::max(kEpsMin, std::min(step_size, kEpsMax));
                if (std::isfinite(eps_div_min)) {
                    step_size = std::min(step_size, 0.7 * eps_div_min);
                }
            }
        }

        if (!warmup) {
            result.draws.push_back(q);
            result.lp.push_back(lp);
        }
    }

    // "accept_rate" for unadjusted = fraction of non-divergent trajectories.
    result.accept_rate = static_cast<double>(total_iter - n_divergences)
                         / static_cast<double>(total_iter);
    result.step_size_final = step_size;
    result.L_final = static_cast<double>(L);
    result.n_grad_evals = n_grad_evals;
    result.n_divergences = n_divergences;

    return result;
}

MCLMCResult mamclmc_sample(
    const std::function<double(const std::vector<double>&,
                               std::vector<double>&)>& log_prob_grad,
    const std::vector<double>& init,
    int dim,
    int n_iter,
    int n_warmup,
    double step_size,
    int L,
    const std::vector<double>& mass_diag,
    unsigned int seed) {
    using namespace mclmc_detail;

    if (dim < 1) {
        throw std::invalid_argument(
            "tulpa MCLMC: dim must be at least 1, got " + std::to_string(dim) + ".");
    }
    if (static_cast<int>(init.size()) != dim) {
        throw std::invalid_argument(
            "tulpa MCLMC: init has " + std::to_string(init.size()) +
            " entries but dim is " + std::to_string(dim) + ".");
    }
    std::vector<double> mass, inv_mass;
    setup_mass(dim, mass_diag, mass, inv_mass);

    if (L <= 0) {
        // Need enough leapfrog steps per trajectory to give meaningful
        // exploration even in low dim; sqrt(d) alone gives L=1 for d=1.
        L = std::max(5, static_cast<int>(std::sqrt(static_cast<double>(dim))));
    }
    bool adapt_eps = (step_size <= 0.0);
    if (adapt_eps) {
        step_size = 0.3 / std::pow(static_cast<double>(dim), 0.25);
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    std::vector<double> q(init.begin(), init.end());
    std::vector<double> p(dim);
    std::vector<double> grad(dim, 0.0);

    double lp = log_prob_grad(q, grad);
    long long n_grad_evals = 1;

    init_momentum(p, mass, inv_mass, dim, rng);

    DualAverage da(step_size);
    double target_accept = 0.9;  // MAMCLMC targets high acceptance

    int total_iter = n_warmup + n_iter;
    MCLMCResult result;
    result.draws.reserve(n_iter);
    result.lp.reserve(n_iter);

    int n_accept = 0;
    int n_post_warmup = 0;
    int n_divergences = 0;

    std::vector<double> q_prop(dim);
    std::vector<double> p_prop(dim);
    std::vector<double> grad_prop(dim);

    for (int iter = 1; iter <= total_iter; iter++) {
        bool warmup = (iter <= n_warmup);
        double eps = warmup ? da.eps() : step_size;
        const double alpha = kTrajectoryRefresh;

        std::copy(q.begin(), q.end(), q_prop.begin());
        std::copy(p.begin(), p.end(), p_prop.begin());
        std::copy(grad.begin(), grad.end(), grad_prop.begin());
        double lp_prop = lp;

        double H_old = -lp + kinetic_energy(p_prop, inv_mass, dim);

        for (int step = 0; step < L; step++) {
            lp_prop = leapfrog_step(q_prop, p_prop, grad_prop, inv_mass, eps,
                                    dim, log_prob_grad);
            n_grad_evals++;
        }

        double H_new = -lp_prop + kinetic_energy(p_prop, inv_mass, dim);

        bool diverged = !std::isfinite(lp_prop) || !std::isfinite(H_new) ||
                        std::abs(H_new - H_old) > kDivergenceThresh;
        if (diverged) {
            n_divergences++;
        }

        double log_accept_prob = -(H_new - H_old);
        if (!std::isfinite(log_accept_prob)) log_accept_prob = -1e300;
        log_accept_prob = std::min(log_accept_prob, 0.0);
        double accept_prob = std::exp(log_accept_prob);

        bool accepted = (uniform(rng) < accept_prob);
        if (accepted) {
            std::copy(q_prop.begin(), q_prop.end(), q.begin());
            std::copy(p_prop.begin(), p_prop.end(), p.begin());
            std::copy(grad_prop.begin(), grad_prop.end(), grad.begin());
            lp = lp_prop;
        } else {
            // Generalized-HMC / partial-refresh reversibility (Horowitz 1991;
            // Neal 2011, sec. 5.3): flip the momentum on rejection so the
            // leapfrog proposal followed by the Langevin refresh leaves the
            // canonical distribution invariant.
            for (int i = 0; i < dim; i++) p[i] = -p[i];
        }

        // Canonical Langevin refresh — preserves p ~ N(0, mass).
        // (project_energy was in the original tail but biases the q-marginal
        // because it destroys the canonical p distribution that leapfrog
        // and refresh both preserve. With true microcanonical dynamics it
        // would be a fixed point; with leapfrog it is not.)
        partial_refresh(p, mass, alpha, dim, rng);

        if (warmup && adapt_eps) {
            double error_stat = target_accept - accept_prob;
            da.update(error_stat, iter);
            step_size = da.eps_bar();
        }

        if (!warmup) {
            result.draws.push_back(q);
            result.lp.push_back(lp);
            if (accepted) n_accept++;
            n_post_warmup++;
        }
    }

    result.accept_rate = (n_post_warmup > 0) ?
        static_cast<double>(n_accept) / n_post_warmup : 0.0;
    result.step_size_final = step_size;
    result.L_final = static_cast<double>(L);
    result.n_grad_evals = n_grad_evals;
    result.n_divergences = n_divergences;

    return result;
}

} // namespace tulpa

namespace tulpa_mclmc {

MCLMCFitResult run_mclmc_sampler(
    const std::vector<double>& init_params,
    const ModelData& data,
    const ParamLayout& layout,
    const MCLMCConfig& config
) {
    MCLMCFitResult out;
    out.success = true;

    int n_params = static_cast<int>(init_params.size());

    // Sanity check: initial log-posterior must be finite.
    double lp0 = compute_log_post(init_params, data, layout);
    if (!std::isfinite(lp0)) {
        out.success = false;
        out.error_msg = "Initial log-posterior is not finite";
        out.samples.resize(0, n_params);
        return out;
    }

    if (config.verbose) {
        Rcpp::Rcout << "Running "
                    << (config.adjusted ? "MAMCLMC" : "MCLMC")
                    << " sampler...\n"
                    << "  Parameters: " << n_params << "\n"
                    << "  Initial step size: " << config.step_size
                    << (config.step_size <= 0.0 ? " (adapt)" : "") << "\n"
                    << "  Leapfrog steps L: " << config.L
                    << (config.L <= 0 ? " (auto)" : "") << "\n"
                    << "  Iterations: " << config.n_iter
                    << " (warmup: " << config.n_warmup << ")\n";
    }

    // log_prob_grad closure: fills `grad` with d(log_post)/dq and returns
    // log_post(q). Mirrors the SGHMC pattern (compute_gradient followed by
    // compute_log_post), reusing the existing tulpa_hmc machinery — no
    // duplicate gradient logic.
    auto log_prob_grad = [&](const std::vector<double>& q,
                             std::vector<double>& grad) -> double {
        double lp = 0.0;
        // Fused: compute_gradient writes log_post into lp via log_post_out.
        compute_gradient(q, data, layout, grad, &lp);
        return lp;
    };

    tulpa::MCLMCResult res;
    try {
        if (config.adjusted) {
            res = tulpa::mamclmc_sample(
                log_prob_grad, init_params, n_params,
                config.n_iter, config.n_warmup,
                config.step_size, config.L,
                config.mass_diag, config.seed);
        } else {
            res = tulpa::mclmc_sample(
                log_prob_grad, init_params, n_params,
                config.n_iter, config.n_warmup,
                config.step_size, config.L,
                config.mass_diag, config.seed);
        }
    } catch (const std::exception& e) {
        out.success = false;
        out.error_msg = std::string("MCLMC sampler threw: ") + e.what();
        out.samples.resize(0, n_params);
        return out;
    }

    int n_save = static_cast<int>(res.draws.size());
    out.samples.resize(n_save, n_params);
    for (int i = 0; i < n_save; i++) {
        const std::vector<double>& row = res.draws[i];
        for (int j = 0; j < n_params; j++) {
            out.samples(i, j) = row[j];
        }
    }
    out.log_lik = std::move(res.lp);

    // No per-iteration eps history is exposed by mclmc_sample / mamclmc_sample
    // (only the final adapted step size). Surface that as a single-element
    // history so the SGSamplerShimResult shape can carry it through.
    out.epsilon_history.assign(1, res.step_size_final);

    if (config.verbose) {
        Rcpp::Rcout << (config.adjusted ? "MAMCLMC" : "MCLMC")
                    << " complete. Final eps: " << res.step_size_final
                    << " accept_rate: " << res.accept_rate
                    << " divergences: " << res.n_divergences << "\n";
    }

    return out;
}

} // namespace tulpa_mclmc
