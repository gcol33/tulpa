// nmix_kernel.h
// Per-site marginal log-likelihood + gradients + Fisher info for the
// Royle (2004) N-mixture model.
//
// Per site i with J_i visits and counts y_{ij}:
//   N_i ~ Poisson(lambda_i),   lambda_i = exp(eta_lambda_i)
//   y_{ij} | N_i ~ Binomial(N_i, p_{ij}),  p_{ij} = plogis(eta_p_{ij})
//
// Marginal log-lik:
//   log L_i = LSE_{N=max(y_i)..K_max} { log Pois(N|lambda_i)
//                                       + sum_j log Binom(y_{ij}|N, p_{ij}) }
//
// Gradients (closed form via posterior weights w_N = P(N | y_i)):
//   d log L_i / d eta_lambda_i = E[N|y_i] - lambda_i
//   d log L_i / d eta_p_{ij}   = y_{ij} - E[N|y_i] * p_{ij}
//
// Complete-data Fisher information (block-diagonal across arms and visits):
//   I_{lambda,lambda} = lambda_i
//   I_{p_ij, p_ij}    = E[N|y_i] * p_{ij} * (1 - p_{ij})
//   I_{lambda, p_ij}  = 0
// Used as the Newton curvature -- equivalent to one EM step per Newton
// iteration. PSD by construction; converges to the same mode as observed-info
// Newton but with simpler per-step linear algebra.
//
// Observed information (marginal, for Laplace log-determinant at the mode):
//   I^obs_{lambda,lambda} = lambda_i - Var[N|y_i]
//   I^obs_{p_ij, p_ij}    = E[N|y_i]*p_ij*(1-p_ij) - p_ij^2 * Var[N|y_i]
//   I^obs_{p_ij, p_ij'}   = -p_ij * p_ij' * Var[N|y_i]    (j != j')
//   I^obs_{lambda, p_ij}  = p_ij * Var[N|y_i]
// Cross-coupling enters only through Var[N|y_i], so the full per-site
// observed information has the rank-1 + diagonal structure:
//   I^obs = diag(d_i) + Var[N|y_i] * (-v_i v_i^T)        with sign carefully
// where d_i collects the "complete-data" diagonal terms and v_i collects the
// per-coordinate sensitivities -- materialized only at the mode.
//
// References:
//   Royle (2004) Biometrics 60: 108-115.
//   Dennis, Morgan & Ridout (2015) Biometrics 71: 237-246.

#ifndef TULPA_NMIX_KERNEL_H
#define TULPA_NMIX_KERNEL_H

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
    double info_eta_lambda;                // complete-data Fisher: lambda_i
    std::vector<double> info_eta_p;        // complete-data Fisher per visit
    double mean_N;                         // E[N | y_i]
    double var_N;                          // Var[N | y_i]
    double boundary_weight;                // posterior mass on N = K_max
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
// predictors), and complete-data Fisher info diagonals.
//
// Args:
//   y           length n_visits, observed counts at site i (nonnegative ints)
//   eta_p       length n_visits, detection logit linear predictor per visit
//   n_visits    J_i (only valid visits passed; NA handling is upstream)
//   eta_lambda  log-scale abundance linear predictor at site i
//   K_max       upper truncation for the sum over N (must be >= max(y))
//
// Notes:
//   - When K_max < max(y), returns log_lik = -Inf and zero gradients.
//   - The sum range is [max(y_i), K_max]; Binom(y, N, p) = 0 for N < y so the
//     truncation is exact, not approximate (Royle 2004, eq. 4).
inline NMixSiteResult compute_nmix_site(
    const int* y,
    const double* eta_p,
    int n_visits,
    double eta_lambda,
    int K_max
) {
    NMixSiteResult res;
    res.grad_eta_p.assign(n_visits, 0.0);
    res.info_eta_p.assign(n_visits, 0.0);

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

    // a_N = log Pois(N|lambda) + sum_j log Binom(y_j | N, p_j)
    //     = N*eta_lambda - lambda - lgamma(N+1)
    //       + sum_j [ lgamma(N+1) - lgamma(y_j+1) - lgamma(N-y_j+1) ]
    //       + sum_j y_j * eta_p_j
    //       + N * sum_j log(1-p_j)
    //     = N*(eta_lambda + sum_log_1mp) - lambda
    //       + (J-1)*lgamma(N+1) - sum_j lgamma(N-y_j+1)
    //       + sum_y_eta_p + const_log_yfact
    //
    // (the last two are constants across N, factored out of the LSE)
    const double slope = eta_lambda + sum_log_1mp;
    const double base_const = -lambda + sum_y_eta_p + const_log_yfact;
    for (int k = 0; k < K_grid; ++k) {
        const int N = K_lo + k;
        double term_lgam = (double)(n_visits - 1) * R::lgammafn((double)N + 1.0);
        for (int j = 0; j < n_visits; ++j) {
            term_lgam -= R::lgammafn((double)(N - y[j]) + 1.0);
        }
        a[k] = (double)N * slope + base_const + term_lgam;
    }

    // log-sum-exp.
    double max_a = a[0];
    for (int k = 1; k < K_grid; ++k) if (a[k] > max_a) max_a = a[k];
    double sum_exp = 0.0;
    for (int k = 0; k < K_grid; ++k) sum_exp += std::exp(a[k] - max_a);
    const double log_lik = max_a + std::log(sum_exp);

    // Posterior weights w_N and moments.
    double mean_N = 0.0;
    double mean_N2 = 0.0;
    double w_boundary = 0.0;
    for (int k = 0; k < K_grid; ++k) {
        const double w = std::exp(a[k] - log_lik);
        const int N = K_lo + k;
        mean_N  += w * (double)N;
        mean_N2 += w * (double)N * (double)N;
        if (k == K_grid - 1) w_boundary = w;
    }
    const double var_N = std::max(mean_N2 - mean_N * mean_N, 0.0);

    res.log_lik = log_lik;
    res.grad_eta_lambda = mean_N - lambda;
    res.info_eta_lambda = lambda;
    res.mean_N = mean_N;
    res.var_N  = var_N;
    res.boundary_weight = w_boundary;
    for (int j = 0; j < n_visits; ++j) {
        res.grad_eta_p[j] = (double)y[j] - mean_N * p_vec[j];
        res.info_eta_p[j] = mean_N * p_vec[j] * (1.0 - p_vec[j]);
    }
    return res;
}

}  // namespace tulpa

#endif  // TULPA_NMIX_KERNEL_H
