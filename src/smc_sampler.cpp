// smc_sampler.cpp
// Definitions of the Sequential Monte Carlo samplers declared in
// smc_sampler.h and smc_modeldata.h.

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <random>
#include "smc_sampler.h"
#include "smc_modeldata.h"
#include "smc_interweave.h"
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
        result.log_marginal_likelihood_var += std::max(0.0, 1.0 / ess - 1.0 / N);

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

namespace {

// The normal approximation at a point x of the log posterior with gradient g:
// precision the finite-difference curvature -H(x) with its eigenvalues
// (diagonal entries) floored at `min_precision`, mean the Newton step
// x + P^-1 g, which at a mode is the mode itself.
SmcReference smc_normal_at(const std::vector<double>& x,
                           const std::vector<double>& g,
                           const ModelData& data, const ParamLayout& layout,
                           double min_precision, bool full) {
    const int d = static_cast<int>(x.size());
    SmcReference q;
    q.d = d;
    q.full = full;
    Eigen::Map<const Eigen::VectorXd> xv(x.data(), d), gv(g.data(), d);
    if (full) {
        std::vector<double> H;
        tulpa_hmc::compute_hessian_finite_diff(x, data, layout, H);
        Eigen::MatrixXd C(d, d);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) C(i, j) = -H[static_cast<size_t>(i) * d + j];
        if (!C.allFinite()) C.setZero();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);
        const Eigen::VectorXd lam = es.eigenvalues().cwiseMax(min_precision);
        const Eigen::MatrixXd& V = es.eigenvectors();
        q.P = V * lam.asDiagonal() * V.transpose();
        q.P = 0.5 * (q.P + q.P.transpose());
        q.mean = xv + V * (V.transpose() * gv).cwiseQuotient(lam);
    } else {
        std::vector<double> Hd;
        tulpa_hmc::compute_hessian_diag_finite_diff(x, data, layout, Hd);
        q.prec = Eigen::VectorXd(d);
        for (int j = 0; j < d; j++) {
            const double c = -Hd[j];
            q.prec(j) = (std::isfinite(c) && c > min_precision) ? c : min_precision;
        }
        q.mean = xv + gv.cwiseQuotient(q.prec);
    }
    if (!q.mean.allFinite()) q.mean = xv;
    q.finalize();
    return q;
}

}  // namespace

SmcReference smc_pathfinder_reference(
    const std::vector<double>& init,
    const ModelData& data,
    const ParamLayout& layout,
    double min_precision,
    bool full,
    unsigned int seed,
    bool verbose
) {
    const int d = static_cast<int>(init.size());

    // L-BFGS on F = -log posterior with a backtracking (Armijo) line search;
    // a trial point whose density or gradient is not finite shortens the step.
    // Every accepted iterate, the start included, is a candidate.
    const int kMaxIter = 500;
    const int kMaxHalvings = 40;
    auto eval = [&data, &layout](const std::vector<double>& x,
                                 std::vector<double>& G) -> double {
        double lp = 0.0;
        tulpa_hmc::compute_gradient(x, data, layout, G, &lp);
        for (double& g : G) g = -g;
        bool ok = std::isfinite(lp);
        for (double g : G) ok = ok && std::isfinite(g);
        return ok ? -lp : std::numeric_limits<double>::infinity();
    };
    std::vector<double> x = init, G(d), xn(d), Gn(d), dir(d), sv(d), yv(d);
    double F = eval(x, G);
    std::vector<std::vector<double>> path_x, path_g;
    auto record = [&path_x, &path_g](const std::vector<double>& xr,
                                     const std::vector<double>& Gr) {
        path_x.push_back(xr);
        std::vector<double> gr(Gr.size());
        for (size_t j = 0; j < Gr.size(); j++) gr[j] = -Gr[j];
        path_g.push_back(std::move(gr));
    };
    tulpa_gp::LBFGSState lbfgs(10, d);
    int it = 0;
    if (std::isfinite(F)) {
        record(x, G);
        for (; it < kMaxIter; it++) {
            double gnorm = 0.0, ginf = 0.0;
            for (double g : G) { gnorm += g * g; ginf = std::max(ginf, std::abs(g)); }
            gnorm = std::sqrt(gnorm);
            if (ginf <= 1e-6 * std::max(1.0, std::abs(F))) break;
            lbfgs.multiply_H(G, dir);
            double dd = 0.0;
            for (int j = 0; j < d; j++) { dir[j] = -dir[j]; dd += dir[j] * G[j]; }
            double t = 1.0;
            if (!(dd < 0.0) || lbfgs.k == 0) {
                for (int j = 0; j < d; j++) dir[j] = -G[j];
                dd = -gnorm * gnorm;
                t = std::min(1.0, 1.0 / gnorm);
            }
            double Fn = std::numeric_limits<double>::infinity();
            bool accepted = false;
            for (int h = 0; h < kMaxHalvings; h++, t *= 0.5) {
                for (int j = 0; j < d; j++) xn[j] = x[j] + t * dir[j];
                Fn = eval(xn, Gn);
                if (std::isfinite(Fn) && Fn <= F + 1e-4 * t * dd) { accepted = true; break; }
            }
            if (!accepted) break;
            for (int j = 0; j < d; j++) { sv[j] = xn[j] - x[j]; yv[j] = Gn[j] - G[j]; }
            lbfgs.add_pair(sv, yv);
            const double dF = F - Fn;
            x = xn; G = Gn; F = Fn;
            record(x, G);
            if (dF <= 1e-12 * std::max(1.0, std::abs(F))) { it++; break; }
        }
    }
    if (path_x.empty()) {
        // No finite start: the floor-width normal at init, which the path
        // corrects like any other reference.
        return smc_isotropic_reference(init, 1.0 / std::sqrt(min_precision));
    }

    // Score each candidate by its ELBO, E_q[log p L - log q], over common
    // standard-normal draws so that candidates differ by their own geometry
    // rather than by their draws, and keep the best (Zhang, Carpenter, Gelman
    // and Vehtari 2022, Pathfinder, Algorithm 1). A draw outside the support
    // makes the candidate's ELBO -Inf.
    const int kElboDraws = 32;
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<Eigen::VectorXd> Z(kElboDraws, Eigen::VectorXd(d));
    for (auto& z : Z) for (int j = 0; j < d; j++) z(j) = normal(rng);
    SmcReference best;
    double best_elbo = -std::numeric_limits<double>::infinity();
    int best_l = -1;
    std::vector<double> th(d);
    for (size_t l = 0; l < path_x.size(); l++) {
        SmcReference q = smc_normal_at(path_x[l], path_g[l], data, layout,
                                       min_precision, full);
        double elbo = 0.0;
        for (int k = 0; k < kElboDraws && std::isfinite(elbo); k++) {
            const Eigen::VectorXd xk = q.scale_draw(Z[k]);
            for (int j = 0; j < d; j++) th[j] = q.mean(j) + xk(j);
            const double lp = tulpa_hmc::compute_log_post(th, data, layout);
            elbo += std::isfinite(lp) ? (lp - q.log_density(th)) / kElboDraws
                                      : -std::numeric_limits<double>::infinity();
        }
        if (elbo > best_elbo || best_l < 0) {
            best_elbo = elbo;
            best_l = static_cast<int>(l);
            best = std::move(q);
        }
    }
    if (verbose) {
        Rcpp::Rcout << "SMC reference: L-BFGS " << it << " iterations, normal "
                    << "approximation at iterate " << best_l << " of "
                    << path_x.size() - 1 << " (ELBO " << best_elbo << ")" << std::endl;
    }
    return best;
}

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

    // The reference q. A user kernel's prior path importance-corrects q to
    // the prior, so q stands in for the prior there. The built-in path
    // bridges q to the posterior and starts from Pathfinder; before the run
    // that reports, one pass of the same path from that start fits q afresh to
    // the moments of the population it ends with (smc_moment_reference
    // below). Every closure reads q by reference, so the second pass runs from
    // the fitted q.
    const double sigma_ref = (cfg.prior_sigma > 0.0) ? cfg.prior_sigma : 1.0;
    const bool full = dim <= kSmcFullCovMaxDim;
    SmcReference ref = (mutation_fn != nullptr)
        ? smc_isotropic_reference(init, sigma_ref)
        : smc_pathfinder_reference(init, data, layout, 1.0 / (sigma_ref * sigma_ref),
                                   full, cfg.seed, cfg.verbose);
    auto log_ref_fn = [&ref](const std::vector<double>& theta) -> double {
        return ref.log_density(theta);
    };
    auto ref_sample = [&ref](
        std::vector<double>& theta, std::mt19937& rng, double /*beta*/
    ) {
        ref.sample(theta, rng);
    };

    using IncrementFn = std::function<double(const std::vector<double>&)>;
    std::vector<IncrementFn> legs;
    std::function<double(const std::vector<double>&)> initial_log_weight;
    std::function<void(std::vector<double>&, double, std::mt19937&, int)> mutation;
    std::function<void(const std::vector<std::vector<double>>&, double)> prepare;

    SmcPopulationHmc hmc(dim);
    SmcNngpFieldHmc field_hmc(data, layout);
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
        auto log_target = [&data, &layout, &ref, b, prior_grad_fn](
            const std::vector<double>& theta, double tau,
            std::vector<double>& grad
        ) -> double {
            const tulpa_smc::PathPoint pt = tulpa_smc::path_point(tau, b);
            const double s = pt.s;
            const double lambda = pt.lambda;
            const double lq = ref.log_density(theta);
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
            for (size_t j = 0; j < theta.size(); j++) grad[j] *= s;
            ref.add_gradient(theta, 1.0 - s, grad);
            if (!std::isfinite(lpath)) return -std::numeric_limits<double>::infinity();
            return (1.0 - s) * lq + s * lpath;
        };
        auto log_value = [&data, &layout, &ref, b](
            const std::vector<double>& theta, double tau) {
            const auto pt = tulpa_smc::path_point(tau, b);
            if (pt.s == 0.0) return ref.log_density(theta);
            double lp = tulpa_hmc::compute_log_post(theta, data, layout);
            if (pt.lambda != 1.0) lp += (pt.lambda - 1.0) *
                tulpa_hmc::compute_log_lik_only(theta, data, layout);
            return (1.0 - pt.s) * ref.log_density(theta) +
                pt.s * lp;
        };
        mutation = [&hmc, &field_hmc, &data, &layout, log_target, log_value](
            std::vector<double>& theta, double tau, std::mt19937& rng, int particle
        ) {
            hmc.step(theta, tau, rng, log_target, particle);
            field_hmc.step(theta, tau, rng, particle, data, layout, log_target);
            smc_interweave_nngp(theta, tau, rng, data, layout, log_value);
        };
        prepare = [&hmc, &field_hmc, &data, &layout](const std::vector<std::vector<double>>& particles,
                         double tau) {
            hmc.adapt(particles);
            field_hmc.adapt(particles, data, layout);
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
        if (mutation_fn == nullptr) {
            // The fitting pass runs leg 1, so q is matched to the distribution
            // that leg ends at: the posterior, or p L^b on the two-leg path.
            // Its evidence estimate is discarded; the reported run's q is
            // fixed before that run draws anything, so the bridge identity
            // holds for it as for any normalized q.
            std::seed_seq seq{cfg.seed, 1u};
            unsigned int fit_seed = 0;
            seq.generate(&fit_seed, &fit_seed + 1);
            const int n_fit = std::max(1, cfg.n_particles / kSmcReferencePassShare);
            const auto fit_pass = tulpa_smc::smc_sample(
                std::vector<IncrementFn>(legs.begin(), legs.begin() + 1),
                ref_sample, mutation, dim, n_fit, cfg.ess_threshold,
                cfg.min_moves, cfg.max_moves, fit_seed, nullptr, prepare);
            ref = smc_moment_reference(
                fit_pass.particles,
                full && n_fit >= kSmcReferenceParticlesPerDim * dim,
                kSmcReferenceDf);
            out.reference_moves_history = fit_pass.moves_history;
            if (cfg.verbose) {
                Rcpp::Rcout << "SMC reference fitted to the first pass ("
                            << fit_pass.n_temperatures << " tempering steps)"
                            << std::endl;
            }
        }
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
