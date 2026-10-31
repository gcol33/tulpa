// smc_modeldata.h
// Generic ModelData / ParamLayout driver for the SMC sampler.
//
// Builds log_prior + log_likelihood closures from
// compute_log_prior + compute_log_lik_only (defined in hmc_sampler.h)
// and dispatches to tulpa_smc::smc_sample (see smc_sampler.h).
//
// The initial population is drawn from a Gaussian reference
// q = N(init, prior_sigma^2 I), not from the prior: proper prior draws need
// per-prior closed-form samplers tulpa lacks generically. What makes that
// start valid is the path it is tempered along (run_smc_sampler below), never
// the mutations alone -- a population that starts from q and is tempered in
// the likelihood as if it were a prior draw converges to neither q L nor
// p L (gcol33/tulpa#876). The constants in compute_log_prior /
// compute_log_lik_only are not all carried, so no evidence is reported and
// SMCDriverResult::log_evidence is NaN.
//
// The mutation kernel is pluggable via a user-supplied SmcMutationFn
// function pointer, targeting compute_log_prior + beta * compute_log_lik_only.
// If null, the built-in population-covariance HMC kernel runs.

#ifndef TULPA_SMC_MODELDATA_H
#define TULPA_SMC_MODELDATA_H

#include <RcppEigen.h>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <limits>
#include <functional>
#include <utility>
#include <algorithm>

#include "hmc_sampler.h"
#include "smc_sampler.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"

namespace tulpa {

// ----------------------------------------------------------------------------
// Function-pointer mutation kernel
//   Mutates `theta` in place at temperature `beta`, targeting
//   log_prior + beta * log_lik. The kernel may inspect `data` and `layout`
//   for problem-specific moves, and `user_data` for any model-side state.
//   `rng_seed` is per-call so the kernel can build its own std::mt19937
//   deterministically.
// ----------------------------------------------------------------------------
typedef void (*SmcMutationFn)(
    double* theta, int n_params, double beta,
    const ModelData* data, const ParamLayout* layout,
    unsigned int rng_seed, void* user_data);

// ----------------------------------------------------------------------------
// SMC config: mirror smc_sample's tunables, plus the reference SD.
// ----------------------------------------------------------------------------
struct SMCConfig {
    int n_particles    = 500;
    int n_mcmc_steps   = 5;
    double ess_threshold = 0.5;   // ESS target for the adaptive temperature
    double prior_sigma   = 1.0;   // SD of the Gaussian reference q
    unsigned int seed    = 42;
    bool verbose         = false;
    // Likelihood power b in (0, 1] at which the path passes through the
    // tempered posterior p L^b and records its population (see run_smc_sampler).
    // 1 is the plain path to the posterior with nothing recorded.
    double bridge_end    = 1.0;
};

struct SMCDriverResult {
    std::vector<std::vector<double>> particles;
    std::vector<double> log_weights;   // log of normalized final weights

    // Equally weighted draws from the tempered posterior p L^tempered_beta,
    // present when SMCConfig::bridge_end < 1; empty and NaN otherwise.
    std::vector<std::vector<double>> tempered_particles;
    double tempered_beta = std::numeric_limits<double>::quiet_NaN();

    // log Z. The accumulator estimates the integral of pi_0 exp(h) along the
    // path it ran, which is the evidence only when both carry every
    // normalizing constant; compute_log_prior / compute_log_lik_only drop
    // some, so no evidence is available: the field is NaN and
    // log_evidence_valid is false.
    double log_evidence = std::numeric_limits<double>::quiet_NaN();
    bool log_evidence_valid = false;

    bool success = true;
    std::string error_msg;
};

// ----------------------------------------------------------------------------
// Gaussian reference q = N(init, prior_sigma^2 I): the initial population.
// Normalized, so pi_0 = q is a valid start for the reference path.
// ----------------------------------------------------------------------------
inline double smc_gaussian_reference_log_density(
    const std::vector<double>& theta, const std::vector<double>& mu, double sd
) {
    const double d = static_cast<double>(theta.size());
    double ss = 0.0;
    for (size_t i = 0; i < theta.size(); i++) {
        const double z = (theta[i] - mu[i]) / sd;
        ss += z * z;
    }
    return -0.5 * ss - d * std::log(sd) - 0.5 * d * std::log(2.0 * M_PI);
}

// A non-finite density (a draw outside a bounded support, an overflowing
// scale) is a zero-weight particle, never a NaN that poisons the ESS search.
inline double smc_finite_or_neg_inf(double v) {
    return std::isfinite(v) ? v : -std::numeric_limits<double>::infinity();
}

// ----------------------------------------------------------------------------
// Built-in mutation kernel (used when mutation_fn == nullptr): Hamiltonian
// Monte Carlo on the tempered target, with the current population's own
// covariance as the inverse mass (whitened coordinates theta = theta0 + L u)
// and a step size moved each temperature toward a 0.7 acceptance rate.
//
// A random walk cannot serve a hierarchical posterior, even one shaped by the
// population covariance. With non-centered random effects b = sigma z, a
// well-resolved group pins b, so z and log sigma move only together, along
// z ~ 1 / sigma: a curve through J + 1 coordinates that a straight-line
// proposal crosses a step at a time. Five such steps per temperature left the
// resampled population on the few ancestors that happened to sit at small
// sigma, and the RE SD came back 3-10x too small (gcol33/tulpa#876). Leapfrog
// trajectories follow the gradient around the curve. Full covariance up to
// kSmcFullCovMaxDim coordinates, the per-coordinate SDs above it, where the
// O(d^2) apply and O(d^3) factor would outgrow the gradient they ride on.
// ----------------------------------------------------------------------------
const int kSmcFullCovMaxDim = 400;
const double kSmcHmcTargetAccept = 0.7;
const int kSmcHmcMaxLeapfrog = 32;

struct SmcPopulationHmc {
    int d = 0;
    bool full = true;
    Eigen::MatrixXd L;           // population covariance Cholesky (full)
    Eigen::VectorXd sd;          // population SDs (diagonal)
    double eps = 0.5;
    long n_prop = 0;
    double sum_acc = 0.0;
    bool adapted = false;
    bool verbose = false;
    // Cache of the last target evaluation: consecutive steps on one particle
    // start where the previous step ended, so its density and gradient are
    // not re-evaluated.
    std::vector<double> last_theta, last_grad;
    double last_lp = 0.0;
    double last_beta = -1.0;

    explicit SmcPopulationHmc(int dim)
        : d(dim), full(dim <= kSmcFullCovMaxDim),
          eps(std::min(1.0, std::pow(static_cast<double>(std::max(dim, 1)), -0.25))) {}

    // u -> L u (full) or sd .* u (diagonal); the transpose maps a theta-gradient
    // to the whitened coordinates.
    Eigen::VectorXd apply_half(const Eigen::VectorXd& u) const {
        return full ? Eigen::VectorXd(L * u) : Eigen::VectorXd(sd.cwiseProduct(u));
    }
    Eigen::VectorXd apply_half_t(const std::vector<double>& g) const {
        Eigen::Map<const Eigen::VectorXd> gv(g.data(), d);
        return full ? Eigen::VectorXd(L.transpose() * gv)
                    : Eigen::VectorXd(sd.cwiseProduct(gv));
    }

    void adapt(const std::vector<std::vector<double>>& particles) {
        const int N = static_cast<int>(particles.size());
        if (adapted && n_prop > 0) {
            const double acc = sum_acc / n_prop;
            if (verbose) {
                Rcpp::Rcout << "  SMC mutation: accept " << acc
                            << ", step " << eps << "\n";
            }
            eps *= std::exp(2.0 * (acc - kSmcHmcTargetAccept));
            eps = std::min(std::max(eps, 1e-3), 1.5);
        }
        n_prop = 0;
        sum_acc = 0.0;
        adapted = true;
        last_beta = -1.0;

        Eigen::VectorXd mean = Eigen::VectorXd::Zero(d);
        for (int n = 0; n < N; n++)
            for (int j = 0; j < d; j++) mean(j) += particles[n][j];
        mean /= std::max(N, 1);
        if (full) {
            Eigen::MatrixXd C = Eigen::MatrixXd::Zero(d, d);
            Eigen::VectorXd r(d);
            for (int n = 0; n < N; n++) {
                for (int j = 0; j < d; j++) r(j) = particles[n][j] - mean(j);
                C.selfadjointView<Eigen::Lower>().rankUpdate(r);
            }
            C = C.selfadjointView<Eigen::Lower>();
            C /= std::max(N - 1, 1);
            // A resampled population can be rank-deficient (fewer distinct
            // ancestors than coordinates, or a coordinate collapsed onto one
            // value); a ridge scaled to the population's own spread keeps the
            // factor and lets a collapsed coordinate move at all.
            const double ridge = 1e-8 * std::max(C.trace() / d, 1e-12) + 1e-12;
            C.diagonal().array() += ridge;
            Eigen::LLT<Eigen::MatrixXd> llt(C);
            if (llt.info() == Eigen::Success) {
                L = llt.matrixL();
            } else {
                L = Eigen::MatrixXd(C.diagonal().cwiseSqrt().asDiagonal());
            }
        } else {
            sd = Eigen::VectorXd::Zero(d);
            for (int n = 0; n < N; n++)
                for (int j = 0; j < d; j++) {
                    const double r = particles[n][j] - mean(j);
                    sd(j) += r * r;
                }
            sd = (sd / std::max(N - 1, 1)).cwiseSqrt().array() + 1e-6;
        }
    }

    // One HMC transition. `target(theta, beta, grad)` returns the tempered log
    // density and writes its theta-gradient; a non-finite value rejects.
    template <class TargetFn>
    void step(std::vector<double>& theta, double beta, std::mt19937& rng,
              const TargetFn& target) {
        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_real_distribution<double> unif(0.0, 1.0);

        std::vector<double> grad0(d);
        double lp0;
        if (beta == last_beta && theta == last_theta) {
            lp0 = last_lp;
            grad0 = last_grad;
        } else {
            lp0 = target(theta, beta, grad0);
        }
        n_prop++;
        last_theta = theta;
        last_lp = lp0;
        last_grad = grad0;
        last_beta = beta;
        if (!std::isfinite(lp0)) return;

        // Jittered step, trajectory length ~1 in whitened units.
        const double e = eps * (0.8 + 0.4 * unif(rng));
        const int n_leap = std::min(kSmcHmcMaxLeapfrog,
                                    std::max(1, static_cast<int>(std::ceil(1.0 / e))));

        Eigen::VectorXd p(d);
        for (int j = 0; j < d; j++) p(j) = normal(rng);
        const double H0 = -lp0 + 0.5 * p.squaredNorm();

        Eigen::VectorXd u = Eigen::VectorXd::Zero(d);
        Eigen::VectorXd g = apply_half_t(grad0);
        std::vector<double> prop(d), grad(d);
        double lp = lp0;
        bool ok = g.allFinite();
        for (int s = 0; s < n_leap && ok; s++) {
            p += 0.5 * e * g;
            u += e * p;
            const Eigen::VectorXd delta = apply_half(u);
            for (int j = 0; j < d; j++) prop[j] = theta[j] + delta(j);
            lp = target(prop, beta, grad);
            g = apply_half_t(grad);
            ok = std::isfinite(lp) && g.allFinite();
            if (ok) p += 0.5 * e * g;
        }

        double acc = 0.0;
        if (ok) {
            const double H1 = -lp + 0.5 * p.squaredNorm();
            acc = std::isfinite(H1) ? std::min(1.0, std::exp(H0 - H1)) : 0.0;
        }
        sum_acc += acc;
        if (ok && unif(rng) < acc) {
            theta = prop;
            last_theta = prop;
            last_lp = lp;
            last_grad = grad;
        }
    }
};

// ----------------------------------------------------------------------------
// Main driver: builds closures + invokes tulpa_smc::smc_sample.
//
// Two paths, by kernel (smc_sampler.h has why a start from q must be either
// bridged or corrected):
//   * built-in kernel : the REFERENCE path. pi_0 = q, and the population
//     bridges to p L^b along q^(1 - s) (p L^b)^s (tulpa_smc::path_point), so the
//     endpoint is the tempered posterior whatever q is; for b < 1 a second
//     leg then tempers L from b to 1.
//   * user kernel     : its documented target is p L^beta, so the population
//     is corrected to pi_0 = p once, by importance weights p / q, and then
//     tempered in L alone, to b and on to 1 (tulpa_smc::prior_path_power).
// With b = cfg.bridge_end < 1 both paths pass through p L^b exactly at the
// end of leg 1, and the population there is returned as tempered_particles --
// the draws Watanabe's WBIC averages the log-likelihood over.
// ----------------------------------------------------------------------------
SMCDriverResult run_smc_sampler(
    const std::vector<double>& init,
    const ModelData& data,
    const ParamLayout& layout,
    const SMCConfig& cfg,
    SmcMutationFn mutation_fn,
    void* user_data
);

} // namespace tulpa

#endif // TULPA_SMC_MODELDATA_H
