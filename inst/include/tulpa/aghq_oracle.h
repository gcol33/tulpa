// aghq_oracle.h
// The structure-agnostic per-group random-effect oracle: the single source of
// truth for "the per-group conditional log-likelihood ell_g(b; theta) in the
// group's random-effect vector b, its score, and its curvature".
//
// One shared inner engine (aghq_re_core) integrates ell_g against N(b; 0, Sigma)
// by adaptive Gauss-Hermite quadrature and is driven entirely through this
// interface; three outer drivers (ML-II optimize, deterministic-integrate,
// stochastic-integrate / Gibbs) consume the same engine + oracle. A model
// package supplies an oracle either natively (compiled, thread-safe) or through
// the R-closure bridge (RClosureOracle, not thread-safe).
//
// Ownership across the R boundary mirrors nested_likelihood.h: the model package
// heap-allocates one REGroupOracle subclass and hands tulpa an
// Rcpp::XPtr<REGroupOracle> with a finalizer; any backing storage the callbacks
// read is parked so it outlives the fit.
//
// Conventions:
//   * b is the group's random-effect vector, length d = sum of the RE-block
//     coefficient counts (the per-group integral is d-dimensional).
//   * theta is the fixed-parameter vector (length n_theta); rebind(theta) sets
//     the active theta before a batch of grad_hess / node_ll / theta_score calls
//     (the analogue of the R make_*(theta) closure capture).
//   * negH is the DATA-only observed information -d^2 ell_g / db^2 (the engine
//     adds the Sigma^{-1} prior curvature). It is symmetric; callers may fill it
//     either storage order.

#ifndef TULPA_AGHQ_ORACLE_H
#define TULPA_AGHQ_ORACLE_H

namespace tulpa {

struct REGroupOracle {
    int n_groups = 0;   // number of groups G
    int d        = 0;   // RE dimension per group (sum of block n_coefs)
    int n_theta  = 0;   // fixed-parameter dimension

    virtual ~REGroupOracle() = default;

    // Set the active fixed-parameter vector for subsequent per-group calls.
    virtual void rebind(const double* theta) = 0;

    // ell_g and its data-only score / observed-info at RE vector b (length d).
    //   logL <- ell_g(b)
    //   grad <- d ell_g / db        (length d)
    //   negH <- -d^2 ell_g / db^2   (d*d, symmetric; prior curvature NOT added)
    virtual void grad_hess(int g, const double* b,
                           double& logL, double* grad, double* negH) const = 0;

    // ell_g at each node row of B (n_nodes x d, row-major); out length n_nodes.
    virtual void node_ll(int g, const double* B, int n_nodes,
                         double* out) const = 0;

    // Optional PSD curvature for the mode-find Newton (d*d, symmetric), filled at
    // the SAME (g, b) as the immediately preceding grad_hess call. A latent-
    // variable marginal (e.g. N-mixture) supplies the complete-data Fisher here
    // so the Newton sees a PD ascent direction even where the observed info
    // `negH` is indefinite away from the mode; `negH` is still what the Laplace
    // marginal uses. Return false (default) to drive the Newton with `negH`
    // itself -- correct for a GLMM, whose observed info is already PD.
    virtual bool newton_hess(int /*g*/, const double* /*b*/,
                             double* /*H*/) const { return false; }

    // Data theta-score of ell_g at b: d ell_g(b; theta) / d theta (length
    // n_theta). Drives the analytic theta-gradient (posterior-weighted score).
    virtual void theta_score(int g, const double* b,
                             double* dl_dtheta) const = 0;

    // Optional theta-space data observed-info A_g = -d^2 ell_g / d theta^2 at b
    // (n_theta*n_theta, symmetric). Enables the fast Laplace Schur-complement SE
    // / Gibbs theta-proposal shape; return false to fall back to the
    // gradient-Jacobian SE.
    virtual bool theta_obs_info(int /*g*/, const double* /*b*/,
                                double* /*A*/) const { return false; }

    // Whether the callbacks are safe to call from an OpenMP-parallel group loop
    // (native C++ oracle: true; R-closure bridge: false -> serial loop).
    virtual bool thread_safe() const { return true; }
};

} // namespace tulpa

#endif // TULPA_AGHQ_ORACLE_H
