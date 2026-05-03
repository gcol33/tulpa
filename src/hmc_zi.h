#ifndef TULPA_HMC_ZI_H
#define TULPA_HMC_ZI_H

// Zero-inflation and hurdle model support for tulpa HMC
// Provides log-likelihood functions for ZI-Poisson, ZI-NegBin,
// and hurdle variants

#include <cmath>
#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <utility>

// Use canonical type definitions from exported headers
#include "tulpa/types.h"

namespace tulpa_zi {

using tulpa::ZIType;

// String -> ZIType lookup. Aliases ("none"/"", "zoib"/"zoibinomial") sit on
// their own rows so adding a new type or alias is a one-row change.
inline ZIType parse_zi_type(const std::string& zi_type_str) {
  static constexpr std::array<std::pair<std::string_view, ZIType>, 11> kZITypeTable{{
    {"none",            ZIType::NONE},
    {"",                ZIType::NONE},
    {"zi_poisson",      ZIType::ZI_POISSON},
    {"zi_negbin",       ZIType::ZI_NEGBIN},
    {"hurdle_poisson",  ZIType::HURDLE_POISSON},
    {"hurdle_negbin",   ZIType::HURDLE_NEGBIN},
    {"zi_binomial",     ZIType::ZI_BINOMIAL},
    {"hurdle_binomial", ZIType::HURDLE_BINOMIAL},
    {"oi_binomial",     ZIType::OI_BINOMIAL},
    {"zoib",            ZIType::ZOIB},
    {"zoibinomial",     ZIType::ZOIB},
  }};
  for (const auto& entry : kZITypeTable) {
    if (entry.first == zi_type_str) return entry.second;
  }
  return ZIType::NONE;
}

// Log of 1 + exp(x), numerically stable
inline double log1pexp(double x) {
  if (x > 35.0) return x;
  if (x < -10.0) return std::exp(x);
  return std::log1p(std::exp(x));
}

// Logistic sigmoid function
inline double logistic(double x) {
  if (x > 0) {
    return 1.0 / (1.0 + std::exp(-x));
  } else {
    double ex = std::exp(x);
    return ex / (1.0 + ex);
  }
}

// Log of logistic sigmoid (log(1/(1+exp(-x))))
inline double log_logistic(double x) {
  return -log1pexp(-x);
}

// Log of 1 - logistic(x) = log(exp(-x)/(1+exp(-x))) = -x - log(1+exp(-x))
inline double log1m_logistic(double x) {
  return -log1pexp(x);
}

// Log-gamma function
inline double lgamma_fn(double x) {
  return std::lgamma(x);
}

// Log factorial
inline double lfactorial(int n) {
  return lgamma_fn(n + 1.0);
}

// ============================================================================
// Standard log-likelihoods (for reference)
// ============================================================================

// Poisson log-PMF: y ~ Poisson(mu)
inline double poisson_lpmf(int y, double mu) {
  if (mu <= 0) return -1e10;
  return y * std::log(mu) - mu - lfactorial(y);
}

// Negative binomial log-PMF: y ~ NegBin(mu, phi)
// Using NB2 parameterization: Var(Y) = mu + mu^2/phi
inline double negbin_lpmf(int y, double mu, double phi) {
  if (mu <= 0 || phi <= 0) return -1e10;
  double r = phi;  // size parameter
  double p = phi / (phi + mu);  // success probability
  return lgamma_fn(y + r) - lgamma_fn(r) - lfactorial(y) +
         r * std::log(p) + y * std::log(1 - p);
}

// Binomial log-PMF: y ~ Binomial(n, p)
// p is success probability
inline double binomial_lpmf(int y, int n, double p) {
  if (p <= 0 || p >= 1 || y < 0 || y > n) return -1e10;
  return lfactorial(n) - lfactorial(y) - lfactorial(n - y) +
         y * std::log(p) + (n - y) * std::log(1.0 - p);
}

// ============================================================================
// Zero-inflated likelihoods
// ============================================================================

// Zero-inflated Poisson log-PMF
// zi_prob: probability of structural zero (on probability scale)
// mu: Poisson mean for count process
inline double zi_poisson_lpmf(int y, double mu, double zi_prob) {
  if (y == 0) {
    // P(Y=0) = zi_prob + (1 - zi_prob) * exp(-mu)
    double log_p0_count = -mu;  // log(P(Y=0|count process))
    double log_zi = std::log(zi_prob);
    double log_1m_zi = std::log(1.0 - zi_prob);

    // log(zi_prob + (1-zi_prob)*exp(-mu))
    // = log(zi_prob * (1 + (1-zi_prob)/zi_prob * exp(-mu)))
    // Use log-sum-exp trick
    double a = log_zi;
    double b = log_1m_zi + log_p0_count;
    double max_ab = std::max(a, b);
    return max_ab + std::log(std::exp(a - max_ab) + std::exp(b - max_ab));
  } else {
    // P(Y=y) = (1 - zi_prob) * Poisson(y|mu)
    return std::log(1.0 - zi_prob) + poisson_lpmf(y, mu);
  }
}

// Zero-inflated Poisson log-PMF (logit scale for zi)
// logit_zi: logit of zero-inflation probability
inline double zi_poisson_lpmf_logit(int y, double mu, double logit_zi) {
  if (y == 0) {
    // log(P(Y=0)) = log(sigmoid(logit_zi) + sigmoid(-logit_zi) * exp(-mu))
    double log_zi = log_logistic(logit_zi);
    double log_1m_zi = log1m_logistic(logit_zi);
    double log_p0_count = -mu;

    double a = log_zi;
    double b = log_1m_zi + log_p0_count;
    double max_ab = std::max(a, b);
    return max_ab + std::log(std::exp(a - max_ab) + std::exp(b - max_ab));
  } else {
    return log1m_logistic(logit_zi) + poisson_lpmf(y, mu);
  }
}

// Zero-inflated negative binomial log-PMF (logit scale for zi)
inline double zi_negbin_lpmf_logit(int y, double mu, double phi, double logit_zi) {
  if (y == 0) {
    // log(P(Y=0)) = log(sigmoid(logit_zi) + sigmoid(-logit_zi) * P_NB(0))
    double log_zi = log_logistic(logit_zi);
    double log_1m_zi = log1m_logistic(logit_zi);

    // P_NB(0) = (phi / (phi + mu))^phi
    double log_p0_count = phi * std::log(phi / (phi + mu));

    double a = log_zi;
    double b = log_1m_zi + log_p0_count;
    double max_ab = std::max(a, b);
    return max_ab + std::log(std::exp(a - max_ab) + std::exp(b - max_ab));
  } else {
    return log1m_logistic(logit_zi) + negbin_lpmf(y, mu, phi);
  }
}

// Zero-inflated binomial log-PMF (logit scale for zi)
// y: successes, n: trials, p: success probability, logit_zi: logit of ZI prob
inline double zi_binomial_lpmf_logit(int y, int n, double p, double logit_zi) {
  if (y == 0) {
    // log(P(Y=0)) = log(sigmoid(logit_zi) + sigmoid(-logit_zi) * (1-p)^n)
    double log_zi = log_logistic(logit_zi);
    double log_1m_zi = log1m_logistic(logit_zi);
    double log_p0_count = n * std::log(1.0 - p);  // (1-p)^n

    double a = log_zi;
    double b = log_1m_zi + log_p0_count;
    double max_ab = std::max(a, b);
    return max_ab + std::log(std::exp(a - max_ab) + std::exp(b - max_ab));
  } else {
    return log1m_logistic(logit_zi) + binomial_lpmf(y, n, p);
  }
}

// One-inflated binomial log-PMF (logit scale for oi)
// y: successes, n: trials, p: success probability, logit_oi: logit of OI prob (psi)
// P(Y=n) = psi + (1-psi) * p^n
// P(Y=y) = (1-psi) * Binomial(y; n, p), y < n
inline double oi_binomial_lpmf_logit(int y, int n, double p, double logit_oi) {
  if (y == n) {
    // log(P(Y=n)) = log(sigmoid(logit_oi) + sigmoid(-logit_oi) * p^n)
    double log_oi = log_logistic(logit_oi);
    double log_1m_oi = log1m_logistic(logit_oi);
    double log_pn_count = n * std::log(p);  // p^n

    double a = log_oi;
    double b = log_1m_oi + log_pn_count;
    double max_ab = std::max(a, b);
    return max_ab + std::log(std::exp(a - max_ab) + std::exp(b - max_ab));
  } else {
    return log1m_logistic(logit_oi) + binomial_lpmf(y, n, p);
  }
}

// Zero-and-one inflated binomial log-PMF (logit scale for both zi and oi)
// y: successes, n: trials, p: success probability
// logit_zi: logit of P(structural zero) = pi_0
// logit_oi: logit of P(structural one | not structural zero) = pi_1
//
// P(Y=0) = pi_0
// P(Y=n) = (1 - pi_0) * pi_1
// P(Y=y) = (1 - pi_0) * (1 - pi_1) * Binomial(y; n, p), 0 < y < n
inline double zoib_lpmf_logit(int y, int n, double p, double logit_zi, double logit_oi) {
  double log_pi0 = log_logistic(logit_zi);       // log(pi_0)
  double log_1m_pi0 = log1m_logistic(logit_zi);  // log(1 - pi_0)
  double log_pi1 = log_logistic(logit_oi);       // log(pi_1)
  double log_1m_pi1 = log1m_logistic(logit_oi);  // log(1 - pi_1)

  if (y == 0) {
    // P(Y=0) = pi_0
    return log_pi0;
  } else if (y == n) {
    // P(Y=n) = (1 - pi_0) * pi_1
    return log_1m_pi0 + log_pi1;
  } else {
    // P(Y=y) = (1 - pi_0) * (1 - pi_1) * Binomial(y; n, p)
    return log_1m_pi0 + log_1m_pi1 + binomial_lpmf(y, n, p);
  }
}

// ============================================================================
// Hurdle likelihoods
// ============================================================================

// Truncated Poisson log-PMF (conditional on y > 0)
// P(Y=y|Y>0) = Poisson(y|mu) / (1 - exp(-mu))
inline double truncated_poisson_lpmf(int y, double mu) {
  if (y <= 0) return -1e10;
  double log_p_untrunc = poisson_lpmf(y, mu);
  double log_p_zero = -mu;
  double log_normalizer = std::log(1.0 - std::exp(log_p_zero));
  return log_p_untrunc - log_normalizer;
}

// Truncated negative binomial log-PMF (conditional on y > 0)
inline double truncated_negbin_lpmf(int y, double mu, double phi) {
  if (y <= 0) return -1e10;
  double log_p_untrunc = negbin_lpmf(y, mu, phi);
  double log_p_zero = phi * std::log(phi / (phi + mu));
  double log_normalizer = std::log(1.0 - std::exp(log_p_zero));
  return log_p_untrunc - log_normalizer;
}

// Hurdle Poisson log-PMF (logit scale for hurdle)
// logit_theta: logit of P(Y > 0)
inline double hurdle_poisson_lpmf_logit(int y, double mu, double logit_theta) {
  if (y == 0) {
    // P(Y=0) = 1 - theta = sigmoid(-logit_theta)
    return log1m_logistic(logit_theta);
  } else {
    // P(Y=y|Y>0) * theta
    return log_logistic(logit_theta) + truncated_poisson_lpmf(y, mu);
  }
}

// Hurdle negative binomial log-PMF (logit scale for hurdle)
inline double hurdle_negbin_lpmf_logit(int y, double mu, double phi, double logit_theta) {
  if (y == 0) {
    return log1m_logistic(logit_theta);
  } else {
    return log_logistic(logit_theta) + truncated_negbin_lpmf(y, mu, phi);
  }
}

// Truncated binomial log-PMF (conditional on y > 0)
// P(Y=y|Y>0) = Binomial(y|n,p) / (1 - (1-p)^n)
inline double truncated_binomial_lpmf(int y, int n, double p) {
  if (y <= 0) return -1e10;
  double log_p_untrunc = binomial_lpmf(y, n, p);
  double log_p_zero = n * std::log(1.0 - p);  // (1-p)^n
  double log_normalizer = std::log(1.0 - std::exp(log_p_zero));
  return log_p_untrunc - log_normalizer;
}

// Hurdle binomial log-PMF (logit scale for hurdle)
// logit_theta: logit of P(Y > 0)
inline double hurdle_binomial_lpmf_logit(int y, int n, double p, double logit_theta) {
  if (y == 0) {
    // P(Y=0) = 1 - theta = sigmoid(-logit_theta)
    return log1m_logistic(logit_theta);
  } else {
    // P(Y=y|Y>0) * theta
    return log_logistic(logit_theta) + truncated_binomial_lpmf(y, n, p);
  }
}

// ============================================================================
// Unified interface
// ============================================================================

// Compute ZI/hurdle log-likelihood for a single observation
// logit_zi_or_theta: logit-scale parameter for ZI prob or hurdle prob
// phi: overdispersion (only used for negbin variants)
inline double zi_log_likelihood(
    int y, double mu, double phi, double logit_zi_or_theta,
    ZIType zi_type) {

  switch (zi_type) {
    case ZIType::ZI_POISSON:
      return zi_poisson_lpmf_logit(y, mu, logit_zi_or_theta);

    case ZIType::ZI_NEGBIN:
      return zi_negbin_lpmf_logit(y, mu, phi, logit_zi_or_theta);

    case ZIType::HURDLE_POISSON:
      return hurdle_poisson_lpmf_logit(y, mu, logit_zi_or_theta);

    case ZIType::HURDLE_NEGBIN:
      return hurdle_negbin_lpmf_logit(y, mu, phi, logit_zi_or_theta);

    case ZIType::NONE:
    default:
      // No ZI, return standard likelihood
      if (phi > 0) {
        return negbin_lpmf(y, mu, phi);
      } else {
        return poisson_lpmf(y, mu);
      }
  }
}

// ============================================================================
// Gradients for ZI parameters
// ============================================================================

// Gradient of ZI-Poisson log-likelihood w.r.t. logit_zi
inline double zi_poisson_grad_logit_zi(int y, double mu, double logit_zi) {
  double zi = logistic(logit_zi);
  double one_m_zi = 1.0 - zi;

  if (y == 0) {
    // d/d(logit_zi) log(zi + (1-zi)*exp(-mu))
    // = [zi*(1-zi) - zi*(1-zi)*exp(-mu)] / [zi + (1-zi)*exp(-mu)]
    // = zi*(1-zi)*(1 - exp(-mu)) / [zi + (1-zi)*exp(-mu)]
    double exp_neg_mu = std::exp(-mu);
    double p0 = zi + one_m_zi * exp_neg_mu;
    return zi * one_m_zi * (1.0 - exp_neg_mu) / p0;
  } else {
    // d/d(logit_zi) log(1-zi) = -zi
    return -zi;
  }
}

// Gradient of ZI-NegBin log-likelihood w.r.t. logit_zi
inline double zi_negbin_grad_logit_zi(int y, double mu, double phi, double logit_zi) {
  double zi = logistic(logit_zi);
  double one_m_zi = 1.0 - zi;

  if (y == 0) {
    double p0_nb = std::pow(phi / (phi + mu), phi);
    double p0 = zi + one_m_zi * p0_nb;
    return zi * one_m_zi * (1.0 - p0_nb) / p0;
  } else {
    return -zi;
  }
}

// Gradient of hurdle model log-likelihood w.r.t. logit_theta
inline double hurdle_grad_logit_theta(int y, double logit_theta) {
  double theta = logistic(logit_theta);

  if (y == 0) {
    // d/d(logit_theta) log(1-theta) = -theta
    return -theta;
  } else {
    // d/d(logit_theta) log(theta) = 1-theta
    return 1.0 - theta;
  }
}

// Gradient of ZI-binomial log-likelihood w.r.t. logit_zi
inline double zi_binomial_grad_logit_zi(int y, int n, double p, double logit_zi) {
  double zi = logistic(logit_zi);
  double one_m_zi = 1.0 - zi;

  if (y == 0) {
    // d/d(logit_zi) log(zi + (1-zi)*(1-p)^n)
    // = zi*(1-zi)*(1 - (1-p)^n) / [zi + (1-zi)*(1-p)^n]
    double p0_binom = std::pow(1.0 - p, n);  // (1-p)^n
    double p0 = zi + one_m_zi * p0_binom;
    return zi * one_m_zi * (1.0 - p0_binom) / p0;
  } else {
    // d/d(logit_zi) log(1-zi) = -zi
    return -zi;
  }
}

// Gradient of ZI-binomial log-likelihood w.r.t. logit(p) (the success prob)
// For binomial, we parameterize p via logit: p = sigmoid(eta)
// d(LL)/d(eta) = d(LL)/d(p) * d(p)/d(eta) = d(LL)/d(p) * p*(1-p)
inline double zi_binomial_grad_eta(int y, int n, double p, double logit_zi) {
  double zi = logistic(logit_zi);
  double one_m_zi = 1.0 - zi;

  if (y == 0) {
    // d/d(eta) log(zi + (1-zi)*(1-p)^n)
    // (1-zi) * n * (1-p)^(n-1) * (-1) * p*(1-p) / [zi + (1-zi)*(1-p)^n]
    // = -(1-zi) * n * p * (1-p)^(n-1) / [zi + (1-zi)*(1-p)^n]
    double p0_binom = std::pow(1.0 - p, n);
    double denom = zi + one_m_zi * p0_binom;
    // Avoid division by zero
    if (denom < 1e-12) denom = 1e-12;
    return -one_m_zi * n * p * std::pow(1.0 - p, n - 1) / denom;
  } else {
    // d/d(eta) log(1-zi) + d/d(eta) binomial_lpmf
    // = 0 + d/d(eta) [y*log(p) + (n-y)*log(1-p)]
    // = y * (1-p) - (n-y) * p = y - n*p
    return y - n * p;
  }
}

// Gradient of hurdle-binomial log-likelihood w.r.t. logit(p)
inline double hurdle_binomial_grad_eta(int y, int n, double p, double logit_theta) {
  if (y == 0) {
    // Zero part: no dependence on p
    return 0.0;
  } else {
    // Truncated binomial: d/d(eta) [log(theta) + truncated_binomial_lpmf]
    // truncated_binomial_lpmf = binomial_lpmf - log(1 - (1-p)^n)
    // d/d(eta) binomial_lpmf = y - n*p
    // d/d(eta) log(1 - (1-p)^n) = n*(1-p)^(n-1)*p*(1-p) / (1 - (1-p)^n)
    //                          = n*p*(1-p)^(n-1) / (1 - (1-p)^n)
    double p0 = std::pow(1.0 - p, n);
    double normalizer = 1.0 - p0;
    if (normalizer < 1e-12) normalizer = 1e-12;
    double grad_normalizer = n * p * std::pow(1.0 - p, n - 1) / normalizer;
    return (y - n * p) - grad_normalizer;
  }
}

// ============================================================================
// OI-binomial gradients
// ============================================================================

// Gradient of OI-binomial log-likelihood w.r.t. logit_oi
inline double oi_binomial_grad_logit_oi(int y, int n, double p, double logit_oi) {
  double oi = logistic(logit_oi);
  double one_m_oi = 1.0 - oi;

  if (y == n) {
    // d/d(logit_oi) log(oi + (1-oi)*p^n)
    // = oi*(1-oi)*(1 - p^n) / [oi + (1-oi)*p^n]
    double pn_binom = std::pow(p, n);  // p^n
    double prob_n = oi + one_m_oi * pn_binom;
    if (prob_n < 1e-12) prob_n = 1e-12;
    return oi * one_m_oi * (1.0 - pn_binom) / prob_n;
  } else {
    // d/d(logit_oi) log(1-oi) = -oi
    return -oi;
  }
}

// Gradient of OI-binomial log-likelihood w.r.t. logit(p) (eta)
inline double oi_binomial_grad_eta(int y, int n, double p, double logit_oi) {
  double oi = logistic(logit_oi);
  double one_m_oi = 1.0 - oi;

  if (y == n) {
    // d/d(eta) log(oi + (1-oi)*p^n)
    // d/d(p) [oi + (1-oi)*p^n] = (1-oi)*n*p^(n-1)
    // d/d(eta) = d/d(p) * p*(1-p) = (1-oi)*n*p^(n-1)*p*(1-p)
    //          = (1-oi)*n*p^n*(1-p)
    double pn = std::pow(p, n);
    double prob_n = oi + one_m_oi * pn;
    if (prob_n < 1e-12) prob_n = 1e-12;
    return one_m_oi * n * pn * (1.0 - p) / prob_n;
  } else {
    // d/d(eta) [log(1-oi) + binomial_lpmf]
    // = 0 + (y - n*p)
    return y - n * p;
  }
}

// ============================================================================
// ZOIB gradients
// ============================================================================

// Gradient of ZOIB log-likelihood w.r.t. logit_zi (pi_0)
inline double zoib_grad_logit_zi(int y, int n, double p, double logit_zi, double logit_oi) {
  double pi0 = logistic(logit_zi);

  if (y == 0) {
    // d/d(logit_zi) log(pi_0) = 1 - pi_0
    return 1.0 - pi0;
  } else {
    // d/d(logit_zi) log(1 - pi_0) = -pi_0
    return -pi0;
  }
}

// Gradient of ZOIB log-likelihood w.r.t. logit_oi (pi_1)
inline double zoib_grad_logit_oi(int y, int n, double p, double logit_zi, double logit_oi) {
  double pi1 = logistic(logit_oi);

  if (y == 0) {
    // No dependence on pi_1 when y=0
    return 0.0;
  } else if (y == n) {
    // d/d(logit_oi) log(pi_1) = 1 - pi_1
    return 1.0 - pi1;
  } else {
    // d/d(logit_oi) log(1 - pi_1) = -pi_1
    return -pi1;
  }
}

// Gradient of ZOIB log-likelihood w.r.t. logit(p) (eta)
inline double zoib_grad_eta(int y, int n, double p, double logit_zi, double logit_oi) {
  if (y == 0 || y == n) {
    // At boundaries, likelihood doesn't depend on p
    return 0.0;
  } else {
    // Interior: d/d(eta) binomial_lpmf = y - n*p
    return y - n * p;
  }
}

}  // namespace tulpa_zi

#endif  // QUOTR_HMC_ZI_H
