// pg_rng.cpp
// Pólya-Gamma random number generator implementation
// Based on Polson, Scott & Windle (2013) and BayesLogit package

#include "pg_rng.h"
#include <Rcpp.h>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

namespace tulpa {

// Constants
const double PI = 3.141592653589793;
const double PI_SQ = PI * PI;
const double TRUNC_T = 0.64;  // Truncation point for series

// ---------------------------------------------------------------------
// Inverse Gaussian sampler
// ---------------------------------------------------------------------

double rinvgauss(double mu, double lambda) {
  // Sample from IG(mu, lambda) using the method of Michael, Schucany & Haas
  double y = R::rnorm(0.0, 1.0);
  y = y * y;
  double x = mu + (mu * mu * y) / (2.0 * lambda) -
             (mu / (2.0 * lambda)) * std::sqrt(4.0 * mu * lambda * y + mu * mu * y * y);

  double u = R::runif(0.0, 1.0);
  if (u <= mu / (mu + x)) {
    return x;
  } else {
    return mu * mu / x;
  }
}

// Truncated inverse Gaussian (left truncated at trunc)
double rinvgauss_trunc(double mu, double lambda, double trunc) {
  double x;
  do {
    x = rinvgauss(mu, lambda);
  } while (x < trunc);
  return x;
}

// ---------------------------------------------------------------------
// Jacobi tilted distribution sampler
// ---------------------------------------------------------------------

// Sample from truncated exponential on (0, t)
double rexp_trunc(double rate, double t) {
  double u = R::runif(0.0, 1.0);
  return -std::log(1.0 - u * (1.0 - std::exp(-rate * t))) / rate;
}

// Compute the a_n coefficient
double a_coef(int n, double x, double t) {
  double k = n + 0.5;
  double out;

  if (x <= t) {
    out = PI * k * std::exp(-0.5 * k * k * PI_SQ * x);
  } else {
    out = PI * k * std::exp(-0.5 * k * k * PI_SQ * x);
  }

  return out;
}

// Mass function for J*(1,0)
double mass_j_star(double x) {
  double sum = 0.0;
  double term;
  int n = 0;

  // Alternating series
  do {
    double k = n + 0.5;
    term = std::exp(-0.5 * k * k * PI_SQ * x);
    if (n % 2 == 0) {
      sum += term;
    } else {
      sum -= term;
    }
    n++;
  } while (std::abs(term) > 1e-12 && n < 1000);

  return PI * sum;
}

// Sample from J*(1, z) using Devroye's method
double sample_jacobi_tilted(double z) {
  double z_abs = std::abs(z) / 2.0;
  double K = PI_SQ / 8.0 + z_abs * z_abs / 2.0;

  double x, s;

  if (z_abs < 1e-6) {
    // z ≈ 0: sample from J*(1, 0) directly
    // Use inverse CDF method or series approximation
    do {
      double e1 = R::rexp(1.0);
      double e2 = R::rexp(1.0);
      x = 1.0 / (2.0 * PI_SQ) * std::pow(PI_SQ / 8.0 + e1, -1.0);
      // Simple rejection
    } while (R::runif(0.0, 1.0) > mass_j_star(x));

  } else {
    // General case: use accept-reject with proposal
    // Proposal: mixture of truncated inverse Gaussian and truncated exponential

    double p = std::exp(-z_abs * TRUNC_T) / (std::exp(-z_abs * TRUNC_T) + 2.0 / PI);

    bool accept = false;
    while (!accept) {
      if (R::runif(0.0, 1.0) < p) {
        // Sample from truncated exponential
        x = TRUNC_T + R::rexp(1.0) / K;
        s = a_coef(0, x, TRUNC_T) / K;
      } else {
        // Sample from truncated inverse Gaussian
        double mu = 1.0 / z_abs;
        double lambda = 1.0;
        x = rinvgauss(mu, lambda);
        while (x > TRUNC_T) {
          x = rinvgauss(mu, lambda);
        }
        s = a_coef(0, x, TRUNC_T);
      }

      // Accept-reject step using alternating series
      double u = R::runif(0.0, 1.0) * s;
      int n = 1;
      bool go = true;

      while (go) {
        double a_n = a_coef(n, x, TRUNC_T);
        if (n % 2 == 1) {
          s -= a_n;
          if (u <= s) {
            accept = true;
            go = false;
          }
        } else {
          s += a_n;
          if (u > s) {
            go = false;  // reject
          }
        }
        n++;
        if (n > 1000) go = false;  // safety
      }
    }
  }

  // Apply exponential tilting
  return x * std::exp(-z_abs * z_abs * x / 2.0);
}

// ---------------------------------------------------------------------
// Main PG samplers
// ---------------------------------------------------------------------

// Sample from PG(1, z)
// Uses the sum-of-gammas representation from Polson, Scott & Windle (2013)
// PG(1, z) = (1/2π²) * sum_{k=0}^∞ G_k / ((k+0.5)² + z²/(4π²))
// where G_k ~ Exp(1)
//
// Optimizations:
// - Truncate adaptively based on z (convergence is faster for larger z)
double rpg1(double z) {
  double z_abs = std::abs(z);

  // Sum-of-gammas representation works well for all practical z
  double sum = 0.0;
  double c = z_abs * z_abs / (4.0 * PI_SQ);  // z² / (4π²)

  // Adaptive truncation: fewer terms needed for larger z
  // because the k=0 term dominates more as z increases
  int n_terms;
  if (z_abs < 0.5) {
    n_terms = 50;
  } else if (z_abs < 2.0) {
    n_terms = 30;
  } else if (z_abs < 5.0) {
    n_terms = 20;
  } else {
    n_terms = 15;
  }

  for (int k = 0; k < n_terms; k++) {
    double gk = R::rexp(1.0);  // Gamma(1,1) = Exp(1)
    double k_half = k + 0.5;
    double denom = k_half * k_half + c;
    sum += gk / denom;
  }

  return sum / (2.0 * PI_SQ);
}

// Sample from PG(b, z) for integer b
double rpg_int(int b, double z) {
  if (b <= 0) return 0.0;

  double sum = 0.0;
  for (int i = 0; i < b; i++) {
    sum += rpg1(z);
  }
  return sum;
}

// Vectorized PG(1, z)
NumericVector rpg1_vec(NumericVector z) {
  int n = z.size();
  NumericVector out(n);

  for (int i = 0; i < n; i++) {
    out[i] = rpg1(z[i]);
  }

  return out;
}

// Vectorized PG(b, z)
NumericVector rpg_vec(IntegerVector b, NumericVector z) {
  int n = b.size();
  if (z.size() != n) {
    stop("b and z must have the same length");
  }

  NumericVector out(n);

  for (int i = 0; i < n; i++) {
    out[i] = rpg_int(b[i], z[i]);
  }

  return out;
}

} // namespace tulpa

// ---------------------------------------------------------------------
// R exports
// ---------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::NumericVector cpp_rpg1(Rcpp::NumericVector z) {
  return tulpa::rpg1_vec(z);
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_rpg(Rcpp::IntegerVector b, Rcpp::NumericVector z) {
  return tulpa::rpg_vec(b, z);
}
