// smc_modeldata.h
// Generic ModelData / ParamLayout driver for the SMC sampler.
//
// Builds log_prior + log_likelihood closures from
// compute_log_prior + compute_log_lik_only (defined in hmc_sampler.h)
// and dispatches to tulpa_smc::smc_sample (see smc_sampler.h).
//
// The initial population is drawn from a normalized reference q
// (SmcReference), not from the prior: proper prior draws need per-prior
// closed-form samplers tulpa lacks generically. What makes that start valid is
// the path it is tempered along (run_smc_sampler below), never the mutations
// alone -- a population that starts from q and is tempered in the likelihood
// as if it were a prior draw converges to neither q L nor p L
// (gcol33/tulpa#876).
//
// The densities tempered are compute_log_prior and compute_log_lik_only, each
// a normalized log-density up to a parameter-independent constant. The path's
// running log Z is therefore the log evidence less those constants, and they
// are added back once at the end (compute_log_prior_normalizer for the prior;
// the built-in families' log-likelihood already carries all of its own),
// rather than inside the tempered densities: there a constant moves no draw in
// exact arithmetic but shifts the rounding of the weights and acceptance
// ratios the draws are computed from. Where
// a constant is not known -- a latent block whose normalizer the engine does
// not carry, a consumer spec's likelihood or extra prior -- no evidence is
// reported and SMCDriverResult::log_evidence_declined says why.
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
    // Bounds on the move steps per tempering step; in between, the count is
    // set by the population autocorrelation rule (tulpa_smc::MoveDecorrelation).
    int min_moves      = 1;
    int max_moves      = 50;
    double ess_threshold = 0.5;   // ESS target for the adaptive temperature
    // Scale of the reference: on the built-in path the floor 1 / prior_sigma^2
    // of the Pathfinder curvature (smc_pathfinder_reference), on a user
    // kernel's prior path the SD of the isotropic reference
    // (smc_isotropic_reference).
    double prior_sigma   = 1.0;
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

    // log p(y): the path's running log Z plus the normalizing constants the
    // tempered densities leave out, with its Monte Carlo standard error
    // (SMCResult::log_marginal_likelihood_var). NaN, with
    // log_evidence_valid false and log_evidence_declined naming the reason,
    // when a constant is not known.
    double log_evidence = std::numeric_limits<double>::quiet_NaN();
    double log_evidence_se = std::numeric_limits<double>::quiet_NaN();
    bool log_evidence_valid = false;
    std::string log_evidence_declined;

    // Move steps each mutation phase ran (SMCResult::moves_history), in the
    // reported run and in the pass the built-in path fits its reference from
    // (empty on a user kernel's path, which has no such pass).
    std::vector<int> moves_history;
    std::vector<int> reference_moves_history;

    bool success = true;
    std::string error_msg;
};

// Full covariance up to this many coordinates, per-coordinate scales above it,
// for both the reference below and the mutation kernel: the O(d^2) apply and
// O(d^3) factor would otherwise outgrow the gradient they ride on.
const int kSmcFullCovMaxDim = 400;

// ----------------------------------------------------------------------------
// Mean and covariance (`full`) or per-coordinate variances (`var`) of an
// equally weighted population, each variance carrying a ridge scaled to the
// population's own spread. A resampled population can be rank-deficient
// (fewer distinct ancestors than coordinates, or a coordinate collapsed onto
// one value); the ridge keeps the factor and lets a collapsed coordinate move
// at all.
// ----------------------------------------------------------------------------
struct SmcPopulationMoments {
    Eigen::VectorXd mean;
    Eigen::MatrixXd cov;   // full
    Eigen::VectorXd var;   // diagonal
};

inline SmcPopulationMoments smc_population_moments(
    const std::vector<std::vector<double>>& particles, bool full
) {
    const int N = static_cast<int>(particles.size());
    const int d = N > 0 ? static_cast<int>(particles[0].size()) : 0;
    SmcPopulationMoments m;
    m.mean = Eigen::VectorXd::Zero(d);
    for (int n = 0; n < N; n++)
        for (int j = 0; j < d; j++) m.mean(j) += particles[n][j];
    m.mean /= std::max(N, 1);
    Eigen::VectorXd r(d);
    if (full) {
        m.cov = Eigen::MatrixXd::Zero(d, d);
        for (int n = 0; n < N; n++) {
            for (int j = 0; j < d; j++) r(j) = particles[n][j] - m.mean(j);
            m.cov.selfadjointView<Eigen::Lower>().rankUpdate(r);
        }
        m.cov = m.cov.selfadjointView<Eigen::Lower>();
        m.cov /= std::max(N - 1, 1);
        const double ridge = 1e-8 * std::max(m.cov.trace() / d, 1e-12) + 1e-12;
        m.cov.diagonal().array() += ridge;
    } else {
        m.var = Eigen::VectorXd::Zero(d);
        for (int n = 0; n < N; n++)
            for (int j = 0; j < d; j++) {
                const double e = particles[n][j] - m.mean(j);
                m.var(j) += e * e;
            }
        m.var /= std::max(N - 1, 1);
        const double ridge = 1e-8 * std::max(m.var.mean(), 1e-12) + 1e-12;
        m.var.array() += ridge;
    }
    return m;
}

// ----------------------------------------------------------------------------
// The reference q the population starts from: a normalized multivariate
// Student-t with location `mean`, scale matrix P^-1 and `df` degrees of
// freedom, the Gaussian N(mean, P^-1) at df = Inf. Normalized, so pi_0 = q is
// a valid start for the reference path whatever its parameters, and the path's
// log Z is the log evidence for any of them. What they decide is how far the
// path has to carry the population, and whether the incremental weights that
// carry it have a variance: a step of size delta near the end reweights by
// (p L / q)^delta, whose second moment under the tempered target needs
// integral (p L)^(1 + delta) q^(-delta) finite, which holds at every delta
// when q's tails are at least as heavy as the posterior's and only below a
// bound when they are lighter (a Gaussian q at a third of the posterior's
// SD: delta < 1/8).
// ----------------------------------------------------------------------------
struct SmcReference {
    int d = 0;
    bool full = true;
    double df = std::numeric_limits<double>::infinity();
    Eigen::VectorXd mean;
    Eigen::MatrixXd P;      // precision of the scale matrix (full)
    Eigen::MatrixXd R;      // lower Cholesky factor, P = R R'
    Eigen::VectorXd prec;   // per-coordinate precision (diagonal)
    double log_norm = 0.0;

    bool gaussian() const { return !std::isfinite(df); }

    void finalize() {
        double half_logdet = 0.0;
        if (full) {
            Eigen::LLT<Eigen::MatrixXd> llt(P);
            R = llt.matrixL();
            half_logdet = R.diagonal().array().log().sum();
        } else {
            half_logdet = 0.5 * prec.array().log().sum();
        }
        log_norm = gaussian()
            ? half_logdet - 0.5 * d * std::log(2.0 * M_PI)
            : half_logdet + std::lgamma(0.5 * (df + d)) - std::lgamma(0.5 * df) -
                  0.5 * d * std::log(df * M_PI);
    }

    double log_density(const std::vector<double>& theta) const {
        Eigen::Map<const Eigen::VectorXd> t(theta.data(), d);
        const Eigen::VectorXd r = t - mean;
        const double q = full ? (R.transpose() * r).squaredNorm()
                              : r.cwiseAbs2().dot(prec);
        return gaussian() ? log_norm - 0.5 * q
                          : log_norm - 0.5 * (df + d) * std::log1p(q / df);
    }

    // grad[j] += w * d log q / d theta_j = -w c (P (theta - mean))_j, with
    // c = 1 for the Gaussian and (df + d) / (df + quad) for the t.
    void add_gradient(const std::vector<double>& theta, double w,
                      std::vector<double>& grad) const {
        Eigen::Map<const Eigen::VectorXd> t(theta.data(), d);
        const Eigen::VectorXd r = t - mean;
        const Eigen::VectorXd g = full ? Eigen::VectorXd(P * r)
                                       : Eigen::VectorXd(prec.cwiseProduct(r));
        const double c = gaussian() ? 1.0 : (df + d) / (df + r.dot(g));
        for (int j = 0; j < d; j++) grad[j] -= w * c * g(j);
    }

    // R'^-1 z (diagonal: z / sqrt(prec)), a N(0, P^-1) draw from z ~ N(0, I).
    Eigen::VectorXd scale_draw(const Eigen::VectorXd& z) const {
        return full ? Eigen::VectorXd(R.transpose().triangularView<Eigen::Upper>().solve(z))
                    : Eigen::VectorXd(z.cwiseQuotient(prec.cwiseSqrt()));
    }

    // theta = mean + R'^-1 z / sqrt(w / df), w ~ chi^2_df (w / df = 1 for the
    // Gaussian): the t as a scale mixture of the Gaussian.
    void sample(std::vector<double>& theta, std::mt19937& rng) const {
        std::normal_distribution<double> normal(0.0, 1.0);
        Eigen::VectorXd z(d);
        for (int j = 0; j < d; j++) z(j) = normal(rng);
        double s = 1.0;
        if (!gaussian()) {
            std::chi_squared_distribution<double> chi2(df);
            s = std::sqrt(df / chi2(rng));
        }
        const Eigen::VectorXd x = scale_draw(z);
        for (int j = 0; j < d; j++) theta[j] = mean(j) + s * x(j);
    }
};

// Degrees of freedom of the reference the built-in path runs from
// (smc_moment_reference): tails polynomial, so heavier than those of any
// posterior whose log density falls at least linearly in its coordinates,
// which a Gaussian reference is not (Owen, Monte Carlo theory, methods and
// examples, Ch. 9: q should have tails at least as heavy as p, and the
// multivariate t at a posterior approximation is the common choice).
const double kSmcReferenceDf = 3.0;

// The pass the reference is fitted from runs 1 / kSmcReferencePassShare of the
// particles: it estimates two moments, not an integral. Measured against
// quadrature on the random-intercept, ICAR, RW1 and AR1 fixtures (20 seeds),
// a quarter matched the full count's accuracy at a quarter of its cost.
const int kSmcReferencePassShare = 4;

// The fitted reference carries the full covariance only when the pass has at
// least this many particles per coordinate. The sample covariance of M draws
// in d dimensions reads its smallest eigenvalue low by up to (1 - sqrt(d/M))^2
// (Marchenko-Pastur), 0.47 at M = 10 d, which the t's covariance
// df / (df - 2) = 3 times its scale still covers; with fewer the reference
// takes the per-coordinate variances, whose estimates need no such ratio.
const int kSmcReferenceParticlesPerDim = 10;

// N(init, sigma^2 I): the reference of a user kernel's prior path, where the
// population is importance-corrected to the prior p rather than bridged to
// the posterior, so q stands in for the prior.
inline SmcReference smc_isotropic_reference(const std::vector<double>& init,
                                            double sigma) {
    SmcReference q;
    q.d = static_cast<int>(init.size());
    q.full = false;
    q.mean = Eigen::Map<const Eigen::VectorXd>(init.data(), q.d);
    q.prec = Eigen::VectorXd::Constant(q.d, 1.0 / (sigma * sigma));
    q.finalize();
    return q;
}

// The t with location the population mean and scale matrix the population
// covariance (full) or variances (diagonal): the generalized stepping-stone
// reference, matched to the moments of a posterior sample (Fan, Wu, Chen, Kuo
// and Lewis 2011; reviewed in Oaks, Cobb, Minin and Leache 2019,
// arXiv:1805.04072, Section 3.1.3), with t tails in place of Gaussian ones.
inline SmcReference smc_moment_reference(
    const std::vector<std::vector<double>>& particles, bool full, double df
) {
    const SmcPopulationMoments m = smc_population_moments(particles, full);
    SmcReference q;
    q.d = static_cast<int>(m.mean.size());
    q.full = full;
    q.df = df;
    q.mean = m.mean;
    if (full) {
        Eigen::LLT<Eigen::MatrixXd> llt(m.cov);
        if (llt.info() == Eigen::Success) {
            q.P = llt.solve(Eigen::MatrixXd::Identity(q.d, q.d));
            q.P = 0.5 * (q.P + q.P.transpose());
        } else {
            q.P = Eigen::MatrixXd(m.cov.diagonal().cwiseInverse().asDiagonal());
        }
    } else {
        q.prec = m.var.cwiseInverse();
    }
    q.finalize();
    return q;
}

// The reference the built-in path's first pass starts from: single-path
// Pathfinder (Zhang, Carpenter, Gelman and Vehtari 2022, JMLR 23(306)). L-BFGS
// climbs the log posterior from `init`; at every iterate the normal
// approximation is the second-order expansion there, and the one with the
// largest ELBO is kept. At a mode that is the Laplace approximation (Dai, Heng,
// Jacob and Whiteley 2022, arXiv:2007.11936, Appendix G, start an SMC sampler
// from it); on a hierarchical posterior the joint mode sits in the neck of the
// funnel, with the field shrunk onto its mean and its scale far out, and the
// ELBO picks an iterate before the climb reaches it. Curvature is the
// finite-difference Hessian, full when `full` and its diagonal otherwise, each
// eigenvalue floored at `min_precision`.
SmcReference smc_pathfinder_reference(
    const std::vector<double>& init,
    const ModelData& data,
    const ParamLayout& layout,
    double min_precision,
    bool full,
    unsigned int seed,
    bool verbose);

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
    // Per-particle cache of the last target evaluation, valid within one
    // mutation phase: a particle's next step starts where its previous step
    // ended, so its density and gradient are not re-evaluated. adapt() runs
    // once per phase, after resampling has reordered the particles, and
    // clears it.
    struct Cached {
        std::vector<double> theta, grad;
        double lp = 0.0;
        bool valid = false;
    };
    std::vector<Cached> cache;

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
        cache.assign(N, Cached());

        const SmcPopulationMoments m = smc_population_moments(particles, full);
        if (full) {
            Eigen::LLT<Eigen::MatrixXd> llt(m.cov);
            if (llt.info() == Eigen::Success) {
                L = llt.matrixL();
            } else {
                L = Eigen::MatrixXd(m.cov.diagonal().cwiseSqrt().asDiagonal());
            }
        } else {
            sd = m.var.cwiseSqrt();
        }
    }

    // One HMC transition of particle `slot`. `target(theta, beta, grad)`
    // returns the tempered log density and writes its theta-gradient; a
    // non-finite value rejects.
    template <class TargetFn>
    void step(std::vector<double>& theta, double beta, std::mt19937& rng,
              const TargetFn& target, int slot) {
        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_real_distribution<double> unif(0.0, 1.0);

        if (slot >= static_cast<int>(cache.size())) cache.resize(slot + 1);
        Cached& c = cache[slot];
        std::vector<double> grad0(d);
        double lp0;
        if (c.valid && theta == c.theta) {
            lp0 = c.lp;
            grad0 = c.grad;
        } else {
            lp0 = target(theta, beta, grad0);
            c.theta = theta;
            c.lp = lp0;
            c.grad = grad0;
            c.valid = true;
        }
        n_prop++;
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
            c.theta = prop;
            c.lp = lp;
            c.grad = grad;
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
//     leg then tempers L from b to 1. q is fitted before the reported run:
//     a pass of leg 1 on a share of the particles from the Pathfinder
//     reference, whose final population gives smc_moment_reference.
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
