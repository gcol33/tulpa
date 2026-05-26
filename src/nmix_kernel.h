// nmix_kernel.h
// Per-site marginal log-likelihood + gradients + Fisher info for the
// Royle (2004) N-mixture model, with a Poisson OR negative-binomial
// abundance mixing distribution.
//
// Per site i with J_i visits and counts y_{ij}:
//   N_i ~ Poisson(lambda_i)                         (mixture = "P",  r = +Inf)
//   N_i ~ NegBin(mean = lambda_i, size = r)         (mixture = "NB", r finite)
//   lambda_i = exp(eta_lambda_i)
//   y_{ij} | N_i ~ Binomial(N_i, p_{ij}),  p_{ij} = plogis(eta_p_{ij})
//
// The NB uses tulpa's neg_binomial_2 convention (size r == phi; mean lambda;
// var = lambda + lambda^2/r). Poisson is the exact r -> Inf limit, recovered
// byte-for-byte by passing r = +Inf (the kernel branches only on isfinite(r)).
//
// Marginal log-lik:
//   log L_i = LSE_{N=max(y_i)..K_max} { log f(N|lambda_i, r)
//                                       + sum_j log Binom(y_{ij}|N, p_{ij}) }
// with f = Poisson or NB. The Binomial block is identical in both cases; the
// abundance prior changes three terms of the per-N weight a_N only (see below).
//
// Gradients (closed form via posterior weights w_N = P(N | y_i)). theta = log r:
//   Poisson:  d log L_i / d eta_lambda = E[N|y_i] - lambda_i
//   NB:       d log L_i / d eta_lambda = r (E[N|y_i] - lambda_i) / (r + lambda_i)
//             d log L_i / d theta      = r * E[ s_r(N) | y_i ]
//                s_r(N) = psi(N+r) - psi(r) - (N+r)/(r+lambda)
//                         + log r + 1 - log(r+lambda)
//   both:     d log L_i / d eta_p_{ij} = y_{ij} - E[N|y_i] * p_{ij}
//
// Complete-data Fisher information (posterior-averaged; block-diagonal across
// arms/visits in the eta coordinates, with a lambda<->theta cross under NB):
//   I_{lambda,lambda} = (E[N|y]+r) q (1-q),  q = lambda/(r+lambda)   (Poisson: lambda)
//   I_{p_ij, p_ij}    = E[N|y] p_{ij}(1-p_{ij})
//   I_{lambda, theta} = -r lambda (E[N|y]-lambda)/(r+lambda)^2       (NB only)
//   I_{theta, theta}  = -dL/dtheta - r^2 E[g''],  g'' below          (NB only)
//
// Observed information (marginal, Laplace curvature) = E[I_c] - Cov(score)
// (Louis 1982). The eta scores are affine in N so their covariance is the
// existing rank-1 Var[N|y] * v v' (v_lambda = 1-q, v_pij = -p_ij). The theta
// score is non-affine (psi(N+r)), so the theta row of Cov(score) needs the
// extra posterior moments accumulated below (cov_N_stheta, var_stheta).
//
// References:
//   Royle (2004) Biometrics 60: 108-115.
//   Dennis, Morgan & Ridout (2015) Biometrics 71: 237-246.
//   Louis (1982) JRSS-B 44: 226-233.

#ifndef TULPA_NMIX_KERNEL_H
#define TULPA_NMIX_KERNEL_H

#include "portable_math.h"   // tulpa::math::portable_digamma / portable_trigamma
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tulpa {

struct NMixSiteResult {
    double log_lik;
    double grad_eta_lambda;
    std::vector<double> grad_eta_p;        // length n_visits
    double info_eta_lambda;                // complete-data Fisher: lambda_i (P) / (E[N]+r)q(1-q) (NB)
    std::vector<double> info_eta_p;        // complete-data Fisher per visit
    double mean_N;                         // E[N | y_i]
    double var_N;                          // Var[N | y_i]
    double boundary_weight;                // posterior mass on N = K_max

    // --- Negative-binomial dispersion outputs (zero / Poisson-neutral when
    //     the Poisson path is taken, i.e. r = +Inf) -----------------------
    double grad_theta;                     // d log L_i / d theta,  theta = log r
    double info_theta;                     // E[I_c, theta theta]
    double info_lambda_theta;              // E[I_c, lambda theta]
    double cov_N_stheta;                   // Cov(N, s_theta | y_i)  (C)
    double var_stheta;                     // Var(s_theta | y_i)     (Vth)
    double score_wt_lambda;                // N-coefficient of s_lambda (1-q); Poisson: 1
};

// Numerically stable log p and log(1-p) under the logit link.
inline void logit_log_probs(double eta, double& log_p, double& log_1mp) {
    if (eta > 0.0) {
        double softplus_neg = std::log1p(std::exp(-eta));   // log(1 + e^{-eta})
        log_p   = -softplus_neg;
        log_1mp = -eta - softplus_neg;
    } else {
        double softplus_pos = std::log1p(std::exp(eta));    // log(1 + e^{eta})
        log_p   = eta - softplus_pos;
        log_1mp = -softplus_pos;
    }
}

// Compute per-site N-mixture marginal log-lik, gradients (wrt linear
// predictors and, under NB, theta = log r), and the complete-data Fisher /
// score-covariance pieces the Laplace driver assembles into the Hessian.
//
// Args:
//   y           length n_visits, observed counts at site i (nonnegative ints)
//   eta_p       length n_visits, detection logit linear predictor per visit
//   n_visits    J_i (only valid visits passed; NA handling is upstream)
//   eta_lambda  log-scale abundance linear predictor at site i
//   K_max       upper truncation for the sum over N (must be >= max(y))
//   r           NB size (dispersion). Pass +Inf for the Poisson kernel.
//
// Notes:
//   - When K_max < max(y), returns log_lik = -Inf and zero gradients.
//   - The sum range is [max(y_i), K_max]; Binom(y, N, p) = 0 for N < y so the
//     truncation is exact, not approximate (Royle 2004, eq. 4).
//   - Poisson path is recovered exactly at r = +Inf; the dispersion outputs
//     are set Poisson-neutral (grad_theta = 0, score_wt_lambda = 1, ...).
inline NMixSiteResult compute_nmix_site(
    const int* y,
    const double* eta_p,
    int n_visits,
    double eta_lambda,
    int K_max,
    double r = std::numeric_limits<double>::infinity()
) {
    const bool is_nb = std::isfinite(r);

    NMixSiteResult res;
    res.grad_eta_p.assign(n_visits, 0.0);
    res.info_eta_p.assign(n_visits, 0.0);
    res.grad_theta        = 0.0;
    res.info_theta        = 0.0;
    res.info_lambda_theta = 0.0;
    res.cov_N_stheta      = 0.0;
    res.var_stheta        = 0.0;
    res.score_wt_lambda   = 1.0;

    int y_max = 0;
    for (int j = 0; j < n_visits; ++j) {
        if (y[j] > y_max) y_max = y[j];
    }
    if (K_max < y_max) {
        res.log_lik = -std::numeric_limits<double>::infinity();
        res.grad_eta_lambda = 0.0;
        res.info_eta_lambda = 0.0;
        res.mean_N = 0.0;
        res.var_N  = 0.0;
        res.boundary_weight = 0.0;
        return res;
    }

    const double lambda = std::exp(eta_lambda);

    // Precompute per-visit p, log p, log(1-p).
    std::vector<double> p_vec(n_visits);
    std::vector<double> log_1mp_vec(n_visits);
    double sum_log_1mp = 0.0;
    double sum_y_eta_p = 0.0;  // sum_j y_j * (log p_j - log(1-p_j)) = sum_j y_j * eta_p_j
    for (int j = 0; j < n_visits; ++j) {
        double lp, l1mp;
        logit_log_probs(eta_p[j], lp, l1mp);
        log_1mp_vec[j] = l1mp;
        sum_log_1mp += l1mp;
        sum_y_eta_p += (double)y[j] * eta_p[j];

        // p_j stable from eta
        if (eta_p[j] > 0.0) {
            p_vec[j] = 1.0 / (1.0 + std::exp(-eta_p[j]));
        } else {
            double e = std::exp(eta_p[j]);
            p_vec[j] = e / (1.0 + e);
        }
    }

    // Constant (in N) part of sum_j log C(N, y_j): -sum_j log Gamma(y_j+1).
    double const_log_yfact = 0.0;
    for (int j = 0; j < n_visits; ++j) {
        const_log_yfact -= R::lgammafn((double)y[j] + 1.0);
    }

    // Sum range and a_N values.
    const int K_lo = y_max;
    const int K_hi = K_max;
    const int K_grid = K_hi - K_lo + 1;
    std::vector<double> a(K_grid);

    // a_N = log f(N|lambda, r) + sum_j log Binom(y_j | N, p_j), grouped by N.
    // Both mixtures share the Binomial block; the abundance prior changes only
    // the N-slope, the N-constant, and (NB) adds a +lgamma(N+r) per-N term.
    //   Poisson: slope = eta_lambda + sum_log_1mp
    //            const = -lambda + sum_y_eta_p + const_log_yfact
    //   NB:      slope = eta_lambda - log(r+lambda) + sum_log_1mp
    //            const = -lgamma(r) + r*log r + sum_y_eta_p + const_log_yfact
    //                    + lgamma(N+r)   (the only N-dependent NB-specific term)
    double slope, base_const;
    if (is_nb) {
        // a_N abundance block: log NB(N|lambda,r) = lgamma(N+r) - lgamma(r)
        //   - lgamma(N+1) + r log r - (N+r) log(r+lambda) + N eta_lambda.
        // The -(N+r) log(r+lambda) splits into an N-slope part (-log(r+lambda))
        // and an N-constant part (-r log(r+lambda)); both must be carried.
        const double log_rpl = std::log(r + lambda);
        slope      = eta_lambda - log_rpl + sum_log_1mp;
        base_const = -std::lgamma(r) + r * std::log(r) - r * log_rpl
                     + sum_y_eta_p + const_log_yfact;
    } else {
        slope      = eta_lambda + sum_log_1mp;
        base_const = -lambda + sum_y_eta_p + const_log_yfact;
    }
    for (int k = 0; k < K_grid; ++k) {
        const int N = K_lo + k;
        double term_lgam = (double)(n_visits - 1) * R::lgammafn((double)N + 1.0);
        for (int j = 0; j < n_visits; ++j) {
            term_lgam -= R::lgammafn((double)(N - y[j]) + 1.0);
        }
        if (is_nb) term_lgam += std::lgamma((double)N + r);
        a[k] = (double)N * slope + base_const + term_lgam;
    }

    // log-sum-exp.
    double max_a = a[0];
    for (int k = 1; k < K_grid; ++k) if (a[k] > max_a) max_a = a[k];
    double sum_exp = 0.0;
    for (int k = 0; k < K_grid; ++k) sum_exp += std::exp(a[k] - max_a);
    const double log_lik = max_a + std::log(sum_exp);

    // Posterior weights w_N and moments. The NB dispersion needs four extra
    // posterior moments of psi(N+r) / psi'(N+r); accumulate them in the same
    // pass when r is finite (no cost on the Poisson path).
    double mean_N = 0.0;
    double mean_N2 = 0.0;
    double w_boundary = 0.0;
    double S_dg = 0.0, S_dg2 = 0.0, S_Ndg = 0.0, S_tg = 0.0;
    for (int k = 0; k < K_grid; ++k) {
        const double w = std::exp(a[k] - log_lik);
        const int N = K_lo + k;
        const double Nd = (double)N;
        mean_N  += w * Nd;
        mean_N2 += w * Nd * Nd;
        if (k == K_grid - 1) w_boundary = w;
        if (is_nb) {
            const double dg = tulpa::math::portable_digamma(Nd + r);
            const double tg = tulpa::math::portable_trigamma(Nd + r);
            S_dg  += w * dg;
            S_dg2 += w * dg * dg;
            S_Ndg += w * Nd * dg;
            S_tg  += w * tg;
        }
    }
    const double var_N = std::max(mean_N2 - mean_N * mean_N, 0.0);

    res.log_lik = log_lik;
    res.mean_N = mean_N;
    res.var_N  = var_N;
    res.boundary_weight = w_boundary;

    // Detection arm: identical Binomial score / info for both mixtures.
    for (int j = 0; j < n_visits; ++j) {
        res.grad_eta_p[j] = (double)y[j] - mean_N * p_vec[j];
        res.info_eta_p[j] = mean_N * p_vec[j] * (1.0 - p_vec[j]);
    }

    if (!is_nb) {
        // Poisson abundance arm (unchanged).
        res.grad_eta_lambda = mean_N - lambda;
        res.info_eta_lambda = lambda;
        res.score_wt_lambda = 1.0;
        return res;
    }

    // ---- Negative-binomial abundance arm ----
    const double rpl   = r + lambda;          // r + lambda
    const double q     = lambda / rpl;        // lambda / (r + lambda)
    const double omq   = r / rpl;             // 1 - q = r / (r + lambda)
    const double dig_r = tulpa::math::portable_digamma(r);
    const double tri_r = tulpa::math::portable_trigamma(r);

    // Score wrt eta_lambda: r (E[N] - lambda) / (r + lambda).
    res.grad_eta_lambda = r * (mean_N - lambda) / rpl;
    // E[I_c, lambda lambda] = (E[N] + r) q (1-q).
    res.info_eta_lambda = (mean_N + r) * q * omq;
    // N-coefficient of s_lambda (for the rank-1 score-covariance vector).
    res.score_wt_lambda = omq;

    // Marginal score wrt theta = log r:  r * E[s_r].
    const double E_sr = S_dg - dig_r - (mean_N + r) / rpl
                        + std::log(r) + 1.0 - std::log(rpl);
    res.grad_theta = r * E_sr;

    // E[g''] for the complete-data theta-theta info.
    //   g''(N) = psi'(N+r) - psi'(r) + 1/r - 1/(r+lambda) + (N - lambda)/(r+lambda)^2
    const double E_gpp = S_tg - tri_r + 1.0 / r - 1.0 / rpl
                         + (mean_N - lambda) / (rpl * rpl);
    // E[I_c, theta theta] = -(dL/dtheta) - r^2 E[g''].
    res.info_theta = -res.grad_theta - r * r * E_gpp;

    // E[I_c, lambda theta] = -r lambda (E[N] - lambda) / (r + lambda)^2.
    res.info_lambda_theta = -r * lambda * (mean_N - lambda) / (rpl * rpl);

    // Score-covariance pieces involving theta. With C := Cov(N, s_theta) and
    // Vth := Var(s_theta):
    //   Cov(N, psi(N+r)) = E[N psi] - E[N] E[psi]
    //   Var(psi(N+r))    = E[psi^2] - E[psi]^2
    //   s_theta = r * s_r, and s_r = psi(N+r) - N/(r+lambda) + const(N-free)
    const double cov_N_dg = S_Ndg - mean_N * S_dg;
    const double var_dg   = std::max(S_dg2 - S_dg * S_dg, 0.0);
    res.cov_N_stheta = r * (cov_N_dg - var_N / rpl);
    res.var_stheta   = r * r * (var_dg + var_N / (rpl * rpl)
                               - 2.0 * cov_N_dg / rpl);
    if (res.var_stheta < 0.0) res.var_stheta = 0.0;

    return res;
}

}  // namespace tulpa

#endif  // TULPA_NMIX_KERNEL_H
