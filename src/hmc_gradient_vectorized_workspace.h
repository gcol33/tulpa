// hmc_gradient_vectorized_workspace.h
// Fragment of hmc_gradient_vectorized.h.
// Included from the hmc_gradient_vectorized.h umbrella header inside
// namespace tulpa_hmc { namespace vectorized { ... } } in hmc_gradients.cpp.
// Do NOT wrap contents in any namespace — already inside namespace vectorized.
// VecGradWorkspace: pre-allocated buffers + caches reused across gradient calls.
#ifndef TULPA_HMC_GRADIENT_VECTORIZED_WORKSPACE_H
#define TULPA_HMC_GRADIENT_VECTORIZED_WORKSPACE_H

// ============================================================================
// Pre-allocated workspace for vectorized gradient (avoids per-call allocation)
// ============================================================================
struct VecGradWorkspace {
  int N = 0;
  std::vector<double> eta_num;
  std::vector<double> eta_denom;
  std::vector<double> mu_num;     // exp(eta_num) — vectorized via Eigen
  std::vector<double> mu_denom;   // exp(eta_denom) — vectorized via Eigen
  std::vector<double> resid_num;
  std::vector<double> resid_denom;
  std::vector<double> effect_dense;  // Pre-expanded RE+spatial+temporal

  // Parameter-dependent intermediates (recomputed each gradient call via Eigen SIMD)
  std::vector<double> log_dn;    // log(mu_num + phi_num) for NB
  std::vector<double> log_dd;    // log(mu_denom + phi_denom) for NB

  // Precomputed data-dependent constants (computed once per dataset)
  uint64_t cached_data_id = 0;
  std::vector<double> lgamma_y_num_p1;    // lgamma(y_num[i] + 1)
  std::vector<double> lgamma_y_denom_p1;  // lgamma(y_denom[i] + 1)
  std::vector<double> lchoose_cache;      // lchoose(n_trials[i], y[i]) for binomial
  std::vector<double> log_y_num_cont;     // log(y_num_cont[i]) for GG, LN
  std::vector<double> log_y_denom_cont;   // log(y_denom_cont[i]) for PG, GG, LN

  // Digamma/lgamma lookup tables for NB/PG: indexed by integer y value.
  // Rebuilt per gradient call (phi changes), but only max_y+1 entries vs N observations.
  // Typically ~50-100 entries vs 500+ observations = 5-10x fewer digamma calls.
  int max_y_num_val = -1;   // max(y_num) across dataset
  int max_y_denom_val = -1; // max(y_denom) across dataset
  std::vector<double> dig_table_num;   // digamma(y + phi_num) for y = 0..max_y_num
  std::vector<double> lg_table_num;    // lgamma(y + phi_num) for y = 0..max_y_num
  std::vector<double> dig_table_denom; // digamma(y + phi_denom) for y = 0..max_y_denom
  std::vector<double> lg_table_denom;  // lgamma(y + phi_denom) for y = 0..max_y_denom

  void init(int n) {
    if (n == N) return;  // Already sized
    N = n;
    eta_num.resize(n);
    eta_denom.resize(n);
    mu_num.resize(n);
    mu_denom.resize(n);
    resid_num.resize(n);
    resid_denom.resize(n);
    effect_dense.resize(n);
    log_dn.resize(n);
    log_dd.resize(n);
    lgamma_y_num_p1.resize(n);
    lgamma_y_denom_p1.resize(n);
    lchoose_cache.resize(n);
    log_y_num_cont.resize(n);
    log_y_denom_cont.resize(n);
    cached_data_id = 0;  // Force recompute on next use
  }

  // Precompute data-only constants (lgamma(y+1), lchoose, log(y)) — called once per dataset
  void precompute(const ModelData& data) {
    if (data.unique_id == cached_data_id) return;  // Already computed for this data
    cached_data_id = data.unique_id;
    max_y_num_val = 0;
    max_y_denom_val = 0;
    for (int i = 0; i < N; i++) {
      lgamma_y_num_p1[i] = std::lgamma(data.legacy.y_num[i] + 1.0);
      lgamma_y_denom_p1[i] = std::lgamma(data.legacy.y_denom[i] + 1.0);
      // lchoose for binomial/beta-binomial: lgamma(n+1) - lgamma(y+1) - lgamma(n-y+1)
      lchoose_cache[i] = std::lgamma(data.legacy.y_denom[i] + 1.0)
                        - std::lgamma(data.legacy.y_num[i] + 1.0)
                        - std::lgamma(data.legacy.y_denom[i] - data.legacy.y_num[i] + 1.0);
      // log(y) for PG/GG/LN families (data-only, never changes)
      log_y_num_cont[i] = (i < static_cast<int>(data.legacy.y_num_cont.size()) && data.legacy.y_num_cont[i] > 0.0)
                         ? std::log(data.legacy.y_num_cont[i]) : 0.0;
      log_y_denom_cont[i] = (i < static_cast<int>(data.legacy.y_denom_cont.size()) && data.legacy.y_denom_cont[i] > 0.0)
                           ? std::log(data.legacy.y_denom_cont[i]) : 0.0;
      // Track max y values for lookup table sizing
      if (data.legacy.y_num[i] > max_y_num_val) max_y_num_val = data.legacy.y_num[i];
      if (data.legacy.y_denom[i] > max_y_denom_val) max_y_denom_val = data.legacy.y_denom[i];
    }
  }

  // Build digamma/lgamma lookup tables for current phi values.
  // Called once per gradient evaluation (phi changes each leapfrog step).
  // Cost: ~(max_y_num + max_y_denom) digamma calls vs N*2 without table.
  void build_digamma_tables(double phi_num, double phi_denom, bool need_denom) {
    int tn = max_y_num_val + 1;
    if (static_cast<int>(dig_table_num.size()) < tn) {
      dig_table_num.resize(tn);
      lg_table_num.resize(tn);
    }
    for (int y = 0; y < tn; y++) {
      auto [d, l] = tulpa::math::portable_digamma_lgamma(y + phi_num);
      dig_table_num[y] = d;
      lg_table_num[y] = l;
    }
    if (need_denom) {
      int td = max_y_denom_val + 1;
      if (static_cast<int>(dig_table_denom.size()) < td) {
        dig_table_denom.resize(td);
        lg_table_denom.resize(td);
      }
      for (int y = 0; y < td; y++) {
        auto [d, l] = tulpa::math::portable_digamma_lgamma(y + phi_denom);
        dig_table_denom[y] = d;
        lg_table_denom[y] = l;
      }
    }
  }
};

#endif  // TULPA_HMC_GRADIENT_VECTORIZED_WORKSPACE_H
