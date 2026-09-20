// hmc_tvc.h
// Temporally-Varying Coefficients (TVC) for HMC backend
// Supports RW1, RW2, AR1, and GP temporal structures for coefficients

#ifndef TULPA_HMC_TVC_H
#define TULPA_HMC_TVC_H

#include <cstddef>
#include <vector>
#include <cmath>
#include "hmc_temporal.h"  // Reuse RW1/RW2/AR1 implementations
#include "pc_prior.h"      // single-source PC prior on every sampled scale
#include "temporal_gp_kernel.h"  // the same kernels temporal_gp() is built on

// Use canonical type definitions from exported headers
#include "tulpa/sum_to_zero.h"  // s2z_aug_quad / s2z_centre_blocks
#include "tulpa/tvc_data.h"
#include "tulpa/types.h"

namespace tulpa_tvc {

using tulpa::TemporalType;
using tulpa::TVCData;
using tulpa::math::safe_log;

// -----------------------------------------------------------------------------
// TVC log-prior
// -----------------------------------------------------------------------------

// Does this structure's precision annihilate a block's constant direction?
// RW1 and RW2 do, so that direction carries no prior and one has to be
// supplied. AR1 and GP are PROPER -- their constant direction already carries
// precision 1' Q 1 -- so nothing is supplied there, only removed from eta.
inline bool tvc_structure_is_intrinsic(TemporalType s) {
  return s == TemporalType::RW1 || s == TemporalType::RW2;
}

// Offset of the (group, term) block in w_flat: one contiguous run of n_times
// values per (g, j), group-major. Every walk over the field -- the prior, the
// eta accumulator, the centring -- addresses it through here, so the layout is
// written once.
inline int tvc_block_offset(const TVCData& d, int g, int j) {
  return (g * d.n_tvc + j) * d.n_times;
}

// Compute log-prior for all TVC terms
// w_flat: all TVC values (n_groups * n_tvc * n_times, flattened)
// tau: vector of precisions (length n_tvc)
// rho: vector of AR1 correlations (length n_tvc, only for AR1)
//
// An intrinsic block is AUGMENTED, not penalised: Q_aug = Q + 1 1'/n_times
// carries the block's own precision tau on its constant direction, which takes
// the rank to rank(Q) + 1 and makes the direction a free N(0, 1/tau) draw that
// integrates out once tvc_center_eta() has removed it from the likelihood. The
// soft penalty this replaced instead stiffened that direction at a precision
// of its own (s2z_precision(n_times)) which the field never has to traverse
// under a sampler but a diagonal variational family does -- see
// tulpa/sum_to_zero.h, and the same construction on the SVC path in
// hmc_svc_autodiff.h.
template <typename T>
inline T tvc_log_prior(
    const std::vector<T>& w_flat,
    const TVCData& tvc_data,
    const std::vector<T>& tau,
    const std::vector<T>& rho
) {
  int n_times = tvc_data.n_times;
  int n_tvc = tvc_data.n_tvc;
  int n_groups = tvc_data.n_groups;
  const bool intrinsic = tvc_structure_is_intrinsic(tvc_data.structure);

  T log_prior = T(0.0);

  for (int g = 0; g < n_groups; g++) {
    for (int j = 0; j < n_tvc; j++) {
      const T* w_jg = w_flat.data() + tvc_block_offset(tvc_data, g, j);
      T rho_j = (tvc_data.structure == TemporalType::AR1) ? rho[j] : T(0.0);
      log_prior = log_prior + tulpa_temporal::log_prior_temporal(
          w_jg, n_times, tvc_data.structure, tau[j], rho_j, tvc_data.cyclic);
      if (intrinsic) {
        // The augmentation and the ONE rank it fills, per pinned block.
        // gmrf_log_norm is linear in the rank, so the +1 is added here rather
        // than threaded through log_prior_temporal's own normalizer.
        log_prior = log_prior
            + tulpa_temporal::gmrf_log_norm(1, safe_log(tau[j]))
            - T(0.5) * tulpa::s2z_aug_quad(w_jg, 0, n_times, tau[j]);
      }
    }
  }

  return log_prior;
}

// Log-prior for a GP-evolving coefficient: w_j(g, .) ~ N(0, K_j) over the
// DISTINCT time instants, with K_j built from the coefficient's own
// (sigma2_j, phi_j). This is the structure for irregular spacing -- rw1 / rw2 /
// ar1 all read `time_index` as a position on a grid, and only this one reads
// where the instants actually sit.
//
// The field is PROPER, so nothing is supplied on the constant direction; it is
// removed from eta by tvc_center_eta() like every other TVC block's level.
//
// Dispatch is the same `cov_is_markov()` temporal_gp() takes: the exponential
// kernel (equivalently Matern nu = 1/2) is an Ornstein-Uhlenbeck process whose
// density factorizes into an O(T) chain with no matrix; the rest have no
// finite-dimensional state-space form and take a dense T x T Cholesky. Either
// way the covariance depends on the COEFFICIENT and not on the group, so it is
// built once per j and read by all n_groups blocks -- a dense fit costs
// n_tvc factorizations per gradient evaluation, not n_tvc * n_groups.
//
// Returns false when a coefficient's covariance is not numerically PD, which
// the caller turns into -Inf rather than a NaN gradient.
template <typename T>
inline bool tvc_log_prior_gp(
    const std::vector<T>& w_flat,
    const TVCData& tvc_data,
    const std::vector<T>& sigma2,
    const std::vector<T>& phi,
    T& log_prior_out
) {
  const int n_times  = tvc_data.n_times;
  const int n_tvc    = tvc_data.n_tvc;
  const int n_groups = tvc_data.n_groups;
  const bool markov  = tulpa_temporal_gp::cov_is_markov(tvc_data.cov_type,
                                                        tvc_data.nu);

  T log_prior = T(0.0);
  std::vector<T> rho, omr2, L;
  for (int j = 0; j < n_tvc; j++) {
    if (markov) {
      tulpa_temporal_gp::ou_chain(tvc_data.time_values, n_times, phi[j],
                                  rho, omr2);
    } else if (!tulpa_temporal_gp::temporal_cov_chol(
                   tvc_data.time_values, n_times, sigma2[j], phi[j],
                   tvc_data.cov_type, tvc_data.nu, tvc_data.period, L)) {
      return false;
    }
    for (int g = 0; g < n_groups; g++) {
      const int off = tvc_block_offset(tvc_data, g, j);
      log_prior = log_prior + (markov
          ? tulpa_temporal_gp::ou_log_density(w_flat.data() + off, n_times,
                                              sigma2[j], rho, omr2)
          : tulpa_temporal_gp::dense_gp_log_density(L, n_times, w_flat, off));
    }
  }
  log_prior_out = log_prior;
  return true;
}

// -----------------------------------------------------------------------------
// Input validation
// -----------------------------------------------------------------------------

// `time_index` and `group_index` arrive 1-based from R and are used raw as
// subscripts into w_flat by compute_tvc_eta, which runs once per gradient
// evaluation. The whole map is checked here, once, at model-build time.
inline void validate_tvc_data(const TVCData& tvc_data) {
  const int N = tvc_data.n_obs;
  const int n_times = tvc_data.n_times;
  const int n_tvc = tvc_data.n_tvc;
  const int n_groups = tvc_data.n_groups;

  if (n_times < 1 || n_tvc < 1 || n_groups < 1) {
    Rcpp::stop("tulpa: a TVC term needs n_times, n_tvc and n_groups >= 1; got "
               "%d, %d, %d.", n_times, n_tvc, n_groups);
  }
  if ((int)tvc_data.time_index.size() != N) {
    Rcpp::stop("tulpa: TVC `time_index` has length %d but must have one entry "
               "per observation (%d).",
               (int)tvc_data.time_index.size(), N);
  }
  if ((int)tvc_data.group_index.size() != N) {
    Rcpp::stop("tulpa: TVC `group_index` has length %d but must have one entry "
               "per observation (%d).",
               (int)tvc_data.group_index.size(), N);
  }
  if (tvc_data.X_tvc.size() != (std::size_t)N * (std::size_t)n_tvc) {
    Rcpp::stop("tulpa: TVC `X_tvc` holds %d values but must hold n_obs * n_tvc "
               "= %d * %d.", (int)tvc_data.X_tvc.size(), N, n_tvc);
  }
  for (int i = 0; i < N; i++) {
    const int t = tvc_data.time_index[i];
    if (t < 1 || t > n_times) {
      Rcpp::stop("tulpa: TVC `time_index[%d]` is %d; must be a 1-based index "
                 "in [1, %d].", i + 1, t, n_times);
    }
    const int g = tvc_data.group_index[i];
    if (g < 1 || g > n_groups) {
      Rcpp::stop("tulpa: TVC `group_index[%d]` is %d; must be a 1-based index "
                 "in [1, %d].", i + 1, g, n_groups);
    }
  }
  // A GP coefficient is a continuous-time field over the distinct instants,
  // so it needs where they SIT, one value per time index. Every other
  // structure reads the index as a grid position and carries none.
  if (tvc_data.structure == TemporalType::GP &&
      (int)tvc_data.time_values.size() != n_times) {
    Rcpp::stop("tulpa: a GP-evolving TVC needs one `time_values` entry per "
               "distinct time (%d); got %d.",
               n_times, (int)tvc_data.time_values.size());
  }
}

// -----------------------------------------------------------------------------
// TVC contribution to linear predictor
// -----------------------------------------------------------------------------

// Compute TVC contribution to linear predictor for all observations
// eta_tvc[i] = sum_j X_tvc[i,j] * w[group_index[i], j, time_index[i]]
template <typename T>
inline void compute_tvc_eta(
    const std::vector<T>& w_flat,  // n_groups * n_tvc * n_times
    const TVCData& tvc_data,
    std::vector<T>& eta_tvc         // Output: length n_obs
) {
  int N = tvc_data.n_obs;
  int n_times = tvc_data.n_times;
  int n_tvc = tvc_data.n_tvc;

  eta_tvc.assign(N, T(0.0));

  (void)n_times;
  for (int i = 0; i < N; i++) {
    int t = tvc_data.time_index[i] - 1;  // 0-based
    int g = tvc_data.group_index[i] - 1;  // 0-based

    for (int j = 0; j < n_tvc; j++) {
      T w_jgt = w_flat[tvc_block_offset(tvc_data, g, j) + t];
      double x_ij = tvc_data.X_tvc[i * n_tvc + j];
      eta_tvc[i] = eta_tvc[i] + T(x_ij) * w_jgt;
    }
  }
}

// -----------------------------------------------------------------------------
// Identification of each (group, term) block's level
// -----------------------------------------------------------------------------

// beta_j and the level of w_{j,g} are not separately identifiable: the term
// contributes eta_i += X_tvc[i,j] * w_{j,g(i)}(t_i), so w -> w + c together
// with beta_j -> beta_j - c leaves eta exactly unchanged whatever the
// covariate. That direction has to go before the field reaches eta, and
// CENTRING is what removes it -- a penalty on the sum leaves it in the
// likelihood and stiffens it instead.
//
// Unconditional, for both field kinds, and paired with the augmentation
// tvc_log_prior() adds for the intrinsic ones: augmenting a path that does not
// centre leaves the level freer than the penalty did (tulpa/sum_to_zero.h).
// The centring and the eta it feeds are one function so no path can build
// tvc_eta from an uncentred field, exactly as svc_center_eta does for SVC.
template <typename T>
inline void tvc_center_eta(
    std::vector<T>& w_flat,          // centred in place, per (group, term)
    const TVCData& tvc_data,
    std::vector<T>& eta_tvc          // Output: length n_obs
) {
  tulpa::s2z_centre_blocks(w_flat.data(),
                           tvc_data.n_groups * tvc_data.n_tvc,
                           tvc_data.n_times);
  compute_tvc_eta(w_flat, tvc_data, eta_tvc);
}

} // namespace tulpa_tvc

#endif // TULPA_HMC_TVC_H
