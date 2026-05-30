// pg_rng.cpp
// Polya-Gamma random number generator implementation
// Based on Polson, Scott & Windle (2013) JASA and the BayesLogit package.

#include "pg_rng.h"
#include <Rcpp.h>
#include <cmath>
#include <algorithm>

using namespace Rcpp;

namespace tulpa {

// Constants
const double PI = 3.141592653589793238462643383279502884197;
const double PI_SQ = PI * PI;
const double TRUNC_T = 0.64;            // Devroye truncation point t
const double TRUNC_T_RECIP = 1.0 / TRUNC_T;
const double HALF_PI_SQ = 0.5 * PI_SQ;

// ---------------------------------------------------------------------
// Shared sub-computations
// ---------------------------------------------------------------------

// Convex-tilting constant f(z) = pi^2/8 + z^2/2 for the J*(1,z) sampler.
static inline double jacobi_fz(double z) {
  return 0.125 * PI_SQ + 0.5 * z * z;
}

// Coefficient a_n(x) of the alternating series for the J*(1,z) density.
// Two genuinely different analytic continuations meet at x = t: the cosine
// (large-x) block for x > t and the exponential (small-x) block for x <= t.
static inline double a_coef(int n, double x) {
  double k = (n + 0.5) * PI;
  if (x > TRUNC_T) {
    return k * std::exp(-0.5 * k * k * x);
  } else if (x > 0.0) {
    double expnt = -1.5 * (std::log(0.5 * PI) + std::log(x)) +
                   std::log(k) - 2.0 * (n + 0.5) * (n + 0.5) / x;
    return std::exp(expnt);
  }
  return 0.0;
}

// ---------------------------------------------------------------------
// Inverse Gaussian samplers
// ---------------------------------------------------------------------

// Sample from IG(mu, lambda) by the method of Michael, Schucany & Haas.
double rinvgauss(double mu, double lambda) {
  double y = R::rnorm(0.0, 1.0);
  y = y * y;
  double half_mu = 0.5 * mu;
  double mu_y = mu * y;
  double x = mu + half_mu * mu_y / lambda -
             half_mu / lambda * std::sqrt(4.0 * lambda * mu_y + mu_y * mu_y);
  double u = R::runif(0.0, 1.0);
  if (u <= mu / (mu + x)) {
    return x;
  }
  return mu * mu / x;
}

// Inverse Gaussian for the J*(1,z) proposal body, truncated to (0, t).
// For z below 1/t the unconstrained mode lies past t, so the truncated draw
// is taken from the tail of the corresponding stable variate; otherwise a
// direct IG(1/z, 1) draw is rejected until it falls below t.
static inline double rtigauss(double z) {
  z = std::abs(z);
  double t = TRUNC_T;
  double x = t + 1.0;
  if (TRUNC_T_RECIP > z) {
    double alpha = 0.0;
    while (R::runif(0.0, 1.0) > alpha) {
      double e1 = R::rexp(1.0);
      double e2 = R::rexp(1.0);
      while (e1 * e1 > 2.0 * e2 / t) {
        e1 = R::rexp(1.0);
        e2 = R::rexp(1.0);
      }
      x = 1.0 + e1 * t;
      x = t / (x * x);
      alpha = std::exp(-0.5 * z * z * x);
    }
  } else {
    double mu = 1.0 / z;
    while (x > t) {
      x = rinvgauss(mu, 1.0);
    }
  }
  return x;
}

// ---------------------------------------------------------------------
// J*(1, z) tilted-Jacobi sampler (Devroye)
// ---------------------------------------------------------------------

// Probability of choosing the truncated-exponential (right) proposal over the
// truncated inverse-Gaussian (left) proposal, from the proposal-mass ratio.
static inline double mass_texpon(double z) {
  double t = TRUNC_T;
  double fz = jacobi_fz(z);
  double b = std::sqrt(1.0 / t) * (t * z - 1.0);
  double a = std::sqrt(1.0 / t) * (t * z + 1.0) * -1.0;
  double x0 = std::log(fz) + fz * t;
  // log lower-tail standard-normal probabilities
  double xb = x0 - z + R::pnorm(b, 0.0, 1.0, /*lower=*/1, /*log=*/1);
  double xa = x0 + z + R::pnorm(a, 0.0, 1.0, /*lower=*/1, /*log=*/1);
  double qdivp = 4.0 / PI * (std::exp(xb) + std::exp(xa));
  return 1.0 / (1.0 + qdivp);
}

// Draw a single J*(1, z) variate by the alternating-series squeeze.
static inline double sample_jacobi_tilted(double z) {
  z = std::abs(z) * 0.5;
  double fz = jacobi_fz(z);

  while (true) {
    double x;
    if (R::runif(0.0, 1.0) < mass_texpon(z)) {
      x = TRUNC_T + R::rexp(1.0) / fz;
    } else {
      x = rtigauss(z);
    }
    double s = a_coef(0, x);
    double y = R::runif(0.0, 1.0) * s;
    int n = 0;
    bool go = true;
    while (go) {
      ++n;
      if (n % 2 == 1) {
        s -= a_coef(n, x);
        if (y <= s) {
          return x;
        }
      } else {
        s += a_coef(n, x);
        if (y > s) {
          go = false;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------
// Main PG samplers
// ---------------------------------------------------------------------

// Sample from PG(1, z) by the exact Devroye method.
// PG(1, z) = (1/4) * J*(1, z / 2).
double rpg1(double z) {
  return 0.25 * sample_jacobi_tilted(z);
}

// Sample from PG(b, z) for integer b.
// Exact sum of b independent Devroye draws for small/moderate b; a Gaussian
// moment-match PG(b, z) ~= N(b m1, b v1) for large b, where m1 and v1 are the
// closed-form PG(1, z) mean and variance (the latter is the exact second
// central moment used in tests). The CLT error is O(1/sqrt(b)).
static const int RPG_NORMAL_THRESHOLD = 200;

double rpg_int(int b, double z) {
  if (b <= 0) return 0.0;

  if (b >= RPG_NORMAL_THRESHOLD) {
    double az = std::abs(z);
    double m1, v1;
    if (az < 1e-6) {
      m1 = 0.25;
      v1 = 1.0 / 24.0;
    } else {
      double zh = 0.5 * az;
      double th = std::tanh(zh);
      m1 = th / (2.0 * az);
      // var = (sinh(z) - z) / (4 z^3 cosh(z/2)^2)
      double ch = std::cosh(zh);
      v1 = (std::sinh(az) - az) / (4.0 * az * az * az * ch * ch);
    }
    double mean = b * m1;
    double sd = std::sqrt(b * v1);
    double draw = R::rnorm(mean, sd);
    return draw > 0.0 ? draw : 0.0;
  }

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
