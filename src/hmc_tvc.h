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
// supplied. AR1 is PROPER -- its constant direction already carries precision
// 1' Q 1 -- so nothing is supplied there, only removed from eta.
inline bool tvc_structure_is_intrinsic(TemporalType s) {
  return s == TemporalType::RW1 || s == TemporalType::RW2;
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

  // Layout: w_flat[(g * n_tvc + j) * n_times + t]
  for (int g = 0; g < n_groups; g++) {
    for (int j = 0; j < n_tvc; j++) {
      const T* w_jg = w_flat.data() + (g * n_tvc + j) * n_times;
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

  for (int i = 0; i < N; i++) {
    int t = tvc_data.time_index[i] - 1;  // 0-based
    int g = tvc_data.group_index[i] - 1;  // 0-based

    for (int j = 0; j < n_tvc; j++) {
      // w_flat layout: [(g * n_tvc + j) * n_times + t]
      T w_jgt = w_flat[(g * n_tvc + j) * n_times + t];
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
