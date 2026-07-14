// Vendored from gcol33/SIMP (cc1de1d) by vendor_simp.sh -- do not edit.
// Upstream: SIMP inst/include/simp/adapt.h. Edit there and re-vendor.
// Copyright (c) 2026 Gilles Colling. MIT license (see inst/COPYRIGHTS).

#ifndef SIMP_ADAPT_H
#define SIMP_ADAPT_H

#include "scheme.h"
#include "harmonic.h"
#include <algorithm>
#include <limits>
#include <cmath>

namespace simp {

// Minimize a scalar objective on [lo, hi] by golden-section search. The
// objective is treated as unimodal on the bracket, which holds for the
// band-error curves below once bracketed near their minimum.
template <typename F>
inline double golden_min(F&& f, double lo, double hi, int iters = 80) {
  const double gr = (std::sqrt(5.0) - 1.0) / 2.0;
  double c = hi - gr * (hi - lo);
  double d = lo + gr * (hi - lo);
  double fc = f(c), fd = f(d);
  for (int i = 0; i < iters; ++i) {
    if (fc < fd) {
      hi = d; d = c; fd = fc;
      c = hi - gr * (hi - lo); fc = f(c);
    } else {
      lo = c; c = d; fc = fd;
      d = lo + gr * (hi - lo); fd = f(d);
    }
  }
  return 0.5 * (lo + hi);
}

// Worst-case harmonic energy error of a scheme over the step band (0, nu_max],
// sampled on n_grid points. A separable quadratic target excites modes at
// nu_k = omega_k * eps spanning this band, so the band maximum is the quantity
// a single fixed coefficient must hold down across the whole spectrum.
inline double band_error(const Scheme& s, double nu_max, int n_grid = 64) {
  double worst = 0.0;
  for (int i = 1; i <= n_grid; ++i) {
    const double nu = nu_max * static_cast<double>(i) / n_grid;
    const double e = harmonic_energy_error(s, nu);
    if (e > worst) worst = e;
  }
  return worst;
}

// The two-stage kick weight b that minimizes the worst-case harmonic energy
// error over the band (0, nu_max]. nu_max is the largest dimensionless step in
// play, omega_max * eps, where omega_max is the highest target frequency (the
// square root of the largest Hessian eigenvalue after mass scaling). Feeding an
// adapted mass spectrum in this way yields the minimum-error integrator for the
// problem at hand rather than a fixed compromise. Solved by a coarse grid to
// bracket the optimum, then golden-section refinement.
inline double two_stage_optimal_b(double nu_max) {
  double best_b = 0.25, best = std::numeric_limits<double>::infinity();
  const int G = 64;
  for (int i = 1; i < G; ++i) {
    const double b = 0.5 * i / G;
    const double e = band_error(two_stage(b), nu_max);
    if (e < best) { best = e; best_b = b; }
  }
  const double step = 0.5 / G;
  const double lo = std::max(1e-4, best_b - step);
  const double hi = std::min(0.5 - 1e-4, best_b + step);
  return golden_min([&](double b) { return band_error(two_stage(b), nu_max); },
                    lo, hi);
}

// The step-adapted two-stage scheme for a band of dimensionless steps up to
// nu_max.
inline Scheme two_stage_adaptive(double nu_max) {
  return two_stage(two_stage_optimal_b(nu_max), "adaptive2");
}

// The (b, a) that minimize the worst-case harmonic energy error over
// (0, nu_max] for the three-stage family. The extra free weight lets the
// three-stage member hold a small error over a wider band than the two-stage
// one, at the cost of a third gradient per step. A coarse 2-D grid brackets the
// optimum; a few coordinate golden-section sweeps refine it.
inline void three_stage_optimal(double nu_max, double& b_out, double& a_out) {
  double best_b = 0.25, best_a = 0.25;
  double best = std::numeric_limits<double>::infinity();
  const int G = 24;
  for (int i = 1; i < G; ++i) {
    const double b = 0.5 * i / G;
    for (int j = 1; j < G; ++j) {
      const double a = 0.5 * j / G;
      const double e = band_error(three_stage(b, a), nu_max);
      if (e < best) { best = e; best_b = b; best_a = a; }
    }
  }
  for (int sweep = 0; sweep < 3; ++sweep) {
    best_b = golden_min(
        [&](double b) { return band_error(three_stage(b, best_a), nu_max); },
        1e-4, 0.5 - 1e-4);
    best_a = golden_min(
        [&](double a) { return band_error(three_stage(best_b, a), nu_max); },
        1e-4, 0.5 - 1e-4);
  }
  b_out = best_b;
  a_out = best_a;
}

// The step-adapted three-stage scheme for a band of dimensionless steps up to
// nu_max.
inline Scheme three_stage_adaptive(double nu_max) {
  double b = 0.25, a = 0.25;
  three_stage_optimal(nu_max, b, a);
  return three_stage(b, a, "adaptive3");
}

}  // namespace simp

#endif  // SIMP_ADAPT_H
