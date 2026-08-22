// hmc_mass_st_gmrf.cpp
// Implementation of the Type-IV precision-informed diagonal mass override.
// See hmc_mass_st_gmrf.h for what it computes and why.

#include "hmc_mass_st_gmrf.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "hmc_sampler.h"            // ModelData / ParamLayout, LikelihoodSpec
#include "log_post_impl.h"          // GenericLogPostState, generic_eta_at
#include "sparse_cholesky.h"        // takahashi_partial_inverse_csc
#include "tulpa/soft_sum_to_zero.h"  // s2z_precision
#include "st_type_iv_precision.h"

namespace tulpa_hmc {

namespace {

// The closed reason vocabulary. Every early return below names one of these,
// so a declined override is always attributable to a precondition rather than
// reported as an unexplained fallback.
constexpr const char* kNotSpatiotemporal = "not_spatiotemporal";
constexpr const char* kNotTypeIv         = "not_type_iv";
constexpr const char* kHsgpInteraction   = "hsgp_interaction";
constexpr const char* kTemporalNotRw     = "temporal_not_rw";
constexpr const char* kLayoutMismatch    = "layout_mismatch";
constexpr const char* kNoEtaWeightsFn    = "no_eta_weights_fn";
constexpr const char* kBudgetExceeded    = "budget_exceeded";
constexpr const char* kNonfiniteWeights  = "nonfinite_weights";
constexpr const char* kNonfiniteTau      = "nonfinite_tau";
constexpr const char* kFactorizeFailed   = "factorize_failed";
constexpr const char* kSelectedInvFailed = "selected_inverse_failed";
constexpr const char* kNonfiniteVariance = "nonfinite_variance";

// Assembly budget. The two sum-to-zero margins contribute S^2 T + S T^2
// entries, so this is what bounds S rather than the Kronecker term. Past it
// the override declines: a mass matrix is a speed-up, and one whose assembly
// costs more than the warmup it accelerates is not one.
constexpr std::size_t kMaxTriplets = 2000000;

// Relative PD backstop, applied only if the first factorization is refused,
// and reported on the result when it is. Relative to the mean diagonal,
// because an absolute jitter means something different at every tau.
constexpr double kRidgeRel = 1e-8;

}  // namespace

const char* st_gmrf_precondition(const ModelData& data, const ParamLayout& layout) {
  if (!data.has_spatiotemporal || !layout.has_spatiotemporal) {
    return kNotSpatiotemporal;
  }
  const auto& st = data.spatiotemporal_data;
  if (st.type != tulpa::STType::TYPE_IV) return kNotTypeIv;
  // A spectral-basis interaction has precision tau/S_j per basis function
  // times Q_t, not Q_s (x) Q_t, and its coordinates are basis weights rather
  // than a spatial field. A different operator, not a wider case of this one.
  if (data.st_is_hsgp) return kHsgpInteraction;
  // st_kronecker_temporal_quad contributes nothing for any other temporal
  // type, so there is no Kronecker operator to invert.
  if (st.temporal_type != tulpa::TemporalType::RW1 &&
      st.temporal_type != tulpa::TemporalType::RW2) {
    return kTemporalNotRw;
  }

  const int S = st.n_spatial, T = st.n_times;
  if (S <= 0 || T <= 0) return kLayoutMismatch;
  if (layout.st_delta_start < 0 || layout.log_tau_st_idx < 0) {
    return kLayoutMismatch;
  }
  if (layout.st_delta_end - layout.st_delta_start != S * T) {
    return kLayoutMismatch;
  }
  if ((int)st.st_flat.size() != data.N) return kLayoutMismatch;
  // The three arrays the assembly walks, checked against S here rather than
  // indexed on trust: they are filled by a consumer package, and a short one
  // is read past the end for every remaining spatial unit with nothing to say
  // so -- the neighbour covariance is then built from whatever the heap holds.
  if (!st.n_neighbors.empty() && (int)st.n_neighbors.size() < S) {
    return kLayoutMismatch;
  }
  if (!st.adj_row_ptr.empty() && (int)st.adj_row_ptr.size() < S + 1) {
    return kLayoutMismatch;
  }
  if (!st.adj_row_ptr.empty() &&
      (int)st.adj_col_idx.size() < st.adj_row_ptr[S]) {
    return kLayoutMismatch;
  }
  // eta assembly reads sharing.st per process, and it is sized by
  // SharingSpec::init(n_processes).
  if (data.n_processes < 1 ||
      (int)data.sharing.st.size() != data.n_processes) {
    return kLayoutMismatch;
  }

  const auto* spec = static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);
  if (spec == nullptr || spec->eta_weights_fn == nullptr) {
    return kNoEtaWeightsFn;
  }
  if (tulpa_st::st_type_iv_triplet_count(data, S, T) > kMaxTriplets) {
    return kBudgetExceeded;
  }
  return "";
}

StGmrfMassResult st_gmrf_inv_mass(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout
) {
  StGmrfMassResult res;
  res.reason = st_gmrf_precondition(data, layout);
  if (res.reason[0] != 0) return res;

  const auto& st = data.spatiotemporal_data;
  const int S = st.n_spatial, T = st.n_times, ST = S * T;
  res.n_block = ST;
  res.n_spatial = S;
  res.n_times = T;

  if ((int)q.size() < layout.total_params) {
    res.reason = kLayoutMismatch;
    return res;
  }

  const double log_tau = q[layout.log_tau_st_idx];
  if (!std::isfinite(log_tau)) { res.reason = kNonfiniteTau; return res; }
  const double tau = std::exp(log_tau);
  if (!(tau > 0.0) || !std::isfinite(tau)) {
    res.reason = kNonfiniteTau;
    return res;
  }

  // ------------------------------------------------------------------
  // 1. Per-observation eta-space working weights, scattered onto the block.
  // ------------------------------------------------------------------
  const auto* spec = static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);
  const int np = data.n_processes;

  tulpa::GenericLogPostState<double> state;
  tulpa::initialize_generic_state(q, data, layout, state);
  tulpa::precompute_generic_fixed_eta(data, state);

  std::vector<double> h_lik(ST, 0.0);
  std::vector<double> eta(np, 0.0);
  std::vector<double> grad_eta(np, 0.0);
  std::vector<double> neg_hess(static_cast<std::size_t>(np) * np, 0.0);

  // d eta_k / d delta_j is 1 on the processes the interaction is shared into
  // and 0 elsewhere (add_generic_st_effect adds the same scalar to each), so
  // the block's own curvature at observation i is the shared sub-block of the
  // eta-space negative Hessian, summed over both indices.
  const std::vector<bool>& share = data.sharing.st;

  for (int i = 0; i < data.N; i++) {
    const int st_idx = st.st_flat[i];
    if (st_idx <= 0 || st_idx > ST) continue;

    tulpa::generic_eta_at(i, data, layout, state, eta.data());

    std::fill(grad_eta.begin(), grad_eta.end(), 0.0);
    std::fill(neg_hess.begin(), neg_hess.end(), 0.0);
    double logit_zi = 0.0, logit_oi = 0.0;
    tulpa::generic_zi_oi_logits(i, data, layout, state, logit_zi, logit_oi);

    spec->eta_weights_fn(i, eta.data(), logit_zi, logit_oi,
                         q, data, layout, data.model_response_data,
                         grad_eta.data(), neg_hess.data());

    double w = 0.0;
    for (int k = 0; k < np; k++) {
      if (!share[k]) continue;
      for (int l = 0; l < np; l++) {
        if (!share[l]) continue;
        w += neg_hess[(std::size_t)k * np + l];
      }
    }
    if (!std::isfinite(w)) { res.reason = kNonfiniteWeights; return res; }
    h_lik[st_idx - 1] += w;
  }

  // A likelihood need not be concave away from its mode, and a negative total
  // curvature at a coordinate would subtract from the precision the mass is
  // read off. Clamped at zero, which leaves that coordinate's variance set by
  // the prior alone, and counted so the harness can see it happened.
  for (int k = 0; k < ST; k++) {
    if (h_lik[k] < 0.0) { h_lik[k] = 0.0; res.n_curvature_clamped++; }
  }

  // ------------------------------------------------------------------
  // 2. Assemble Q in the sampled coordinate.
  // ------------------------------------------------------------------
  // Centered: delta is sampled, the prior carries tau, likelihood and penalty
  // read delta directly. Non-centered: z is sampled with delta = z/sqrt(tau),
  // so the prior is tau-free and the other two pick up 1/tau.
  const bool nc = (data.st_parameterization == 1);
  const double kron_scale = nc ? 1.0 : tau;
  const double outer_scale = nc ? (1.0 / tau) : 1.0;

  res.lambda_row = outer_scale * tulpa::s2z_precision(T);
  res.lambda_col = outer_scale * tulpa::s2z_precision(S);

  Eigen::SparseMatrix<double> Q;
  if (!tulpa_st::st_type_iv_precision(data, S, T, kron_scale, outer_scale,
                                      h_lik, outer_scale, /*ridge=*/0.0,
                                      kMaxTriplets, Q)) {
    res.reason = kBudgetExceeded;
    return res;
  }

  // ------------------------------------------------------------------
  // 3. Factorize, with one relative backstop if the plain matrix is refused.
  // ------------------------------------------------------------------
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Lower> llt;
  llt.compute(Q);
  if (llt.info() != Eigen::Success) {
    double diag_sum = 0.0;
    for (int k = 0; k < ST; k++) diag_sum += std::abs(Q.coeff(k, k));
    const double ridge = kRidgeRel * (ST > 0 ? diag_sum / ST : 1.0);
    if (!(ridge > 0.0) || !std::isfinite(ridge)) {
      res.reason = kFactorizeFailed;
      return res;
    }
    for (int k = 0; k < ST; k++) Q.coeffRef(k, k) += ridge;
    llt.compute(Q);
    if (llt.info() != Eigen::Success) {
      res.reason = kFactorizeFailed;
      return res;
    }
    res.ridge_applied = ridge;
  }

  // ------------------------------------------------------------------
  // 4. diag(Q^-1) by selected inversion on the factor.
  // ------------------------------------------------------------------
  // takahashi_partial_inverse_csc is the engine's one selected inverse. It
  // reads a plain CSC lower factor, so it serves this Eigen factor exactly as
  // it serves the CHOLMOD one -- there is no second Takahashi recursion here.
  Eigen::SparseMatrix<double> L = llt.matrixL();
  L.makeCompressed();
  const int n = (int)L.rows();
  const int* Lp = L.outerIndexPtr();
  const int* Li = L.innerIndexPtr();
  const double* Lx = L.valuePtr();

  std::vector<double> Zx((std::size_t)Lp[n], 0.0);
  if (!tulpa::takahashi_partial_inverse_csc(n, Lp, Li, Lx, Zx.data())) {
    res.reason = kSelectedInvFailed;
    return res;
  }

  // llt factorizes P Q P', so Q^-1 at original index i sits at permuted i.
  const auto& perm = llt.permutationP().indices();
  res.inv_mass.assign(ST, 0.0);
  for (int i = 0; i < ST; i++) {
    const int pi = perm[i];
    const double v = Zx[(std::size_t)Lp[pi]];   // Z[pi, pi], diagonal first
    if (!std::isfinite(v) || !(v > 0.0)) {
      res.inv_mass.clear();
      res.reason = kNonfiniteVariance;
      return res;
    }
    res.inv_mass[i] = v;
  }

  double ld = 0.0;
  for (int j = 0; j < n; j++) ld += std::log(Lx[Lp[j]]);
  res.log_det_Q = 2.0 * ld;

  res.ok = true;
  res.reason = "";
  return res;
}

}  // namespace tulpa_hmc
