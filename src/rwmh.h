// rwmh.h
//
// Random-walk Metropolis primitives shared by every exact-target sampler in the
// engine: the covariance Gibbs sweep (re_cov_gibbs_sweep.h), which runs a
// Laplace-shaped random walk on beta and on each per-(term, group) b, and the
// subspace debias (subspace_debias.h), which runs the same walk on the flagged
// latent coordinates. Both are "Laplace body + Metropolis correction", so the
// scale schedule, the target acceptance and the accept test are one definition
// rather than one per sampler.
//
// The starting scale 2.4 / sqrt(d) and the target acceptance (0.234 for d > 1,
// 0.44 for d = 1) are the Roberts-Gelman-Gilks optimal-scaling results for a
// random-walk Metropolis on a d-dimensional target. The adaptation is the
// Robbins-Monro recursion on log scale with gain 1 / sqrt(sweep), run during
// warmup only and held fixed afterwards, so the recorded draws come from a
// time-homogeneous chain.
//
// Randomness goes through R's RNG (R::unif_rand / R::rnorm), so set.seed in R
// reproduces a run and no sampler here may be called from inside a parallel
// region.

#ifndef TULPA_RWMH_H
#define TULPA_RWMH_H

#include <Rcpp.h>
#include <cmath>

namespace tulpa {

// Starting random-walk scale for a d-dimensional block.
inline double rw_init_scale(int d) {
  return 2.4 / std::sqrt(static_cast<double>(d > 0 ? d : 1));
}

// Target acceptance rate for a d-dimensional block.
inline double rw_target_accept(int d) {
  return (d > 1) ? 0.234 : 0.44;
}

// Robbins-Monro gain at sweep `sweep` (1-based).
inline double rw_adapt_gain(int sweep) {
  return 1.0 / std::sqrt(static_cast<double>(sweep > 0 ? sweep : 1));
}

// One Robbins-Monro update of a log-scale toward `target`, given the realized
// acceptance rate of the sweep just taken.
inline double rw_adapt_scale(double scale, double gain, double acc_rate,
                             double target) {
  return std::exp(std::log(scale) + gain * (acc_rate - target));
}

// Metropolis accept test on a log acceptance ratio.
//
// The uniform is drawn unconditionally, before the ratio is inspected, so a
// sweep consumes exactly one uniform per test whatever the ratio is. That keeps
// the RNG stream a function of the sweep count alone, which is what makes a run
// reproducible from set.seed regardless of whether a proposal happened to land
// outside the likelihood's domain.
//
// A ratio of -Inf (the proposal is outside the domain) rejects; NaN, which no
// finite comparison can order, also rejects. A ratio of +Inf means the CURRENT
// state has zero density, so the move is accepted -- the chain would otherwise
// have no way back out of an impossible state.
inline bool rw_accept(double log_ratio) {
  const double u = R::unif_rand();
  if (std::isnan(log_ratio)) return false;
  return std::log(u) < log_ratio;
}

} // namespace tulpa

#endif // TULPA_RWMH_H
