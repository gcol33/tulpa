// hmc_sampler_nuts_infra.h
// Fragment of hmc_sampler.h. Self-contained: defines symbols inside
// namespace tulpa_hmc.
// LeapfrogResult variants, TreeStats, U-turn criterion helpers,
// NUTSWorkspace (zero-allocation tree-building buffer pool).
#ifndef TULPA_HMC_SAMPLER_NUTS_INFRA_H
#define TULPA_HMC_SAMPLER_NUTS_INFRA_H

#include <cstring>
#include <vector>

#include "hmc_sampler_decls.h"  // GradientFn

namespace tulpa_hmc {

// =====================================================================
// HMC/NUTS sampler structures
// =====================================================================

struct LeapfrogResult {
  std::vector<double> q;
  std::vector<double> p;
  double log_prob;
  bool divergent;
};

// NUTS-specific structures
struct LeapfrogResultWithGrad {
  std::vector<double> q, p, grad;
  double log_prob;
  bool divergent;
};

// =====================================================================
// Optimized NUTS infrastructure (zero-allocation tree building)
// =====================================================================

// Result from in-place leapfrog (no vectors allocated)
struct LeapfrogInPlaceResult {
  double log_prob;
  bool divergent;
};

// Lightweight tree statistics (returned by build_tree_fast)
// Endpoints tracked via workspace slot indices, not owned vectors
struct TreeStats {
  int left_slot;            // Workspace slot index for left endpoint
  int right_slot;           // Workspace slot index for right endpoint
  int proposal_slot;        // Workspace slot index for proposal
  double sum_log_weight;
  double sum_accept_prob;
  double log_prob_proposal;
  int n_valid;
  int n_leapfrog;
  bool divergent;
  bool stop;

  // Generalized U-turn criterion (Betancourt 2017, Stan implementation)
  std::vector<double> rho;          // Cumulative momentum sum across trajectory
  std::vector<double> p_sharp_beg;  // M^{-1} * p at trajectory beginning
  std::vector<double> p_sharp_end;  // M^{-1} * p at trajectory end
  std::vector<double> p_beg;        // Raw momentum at trajectory beginning
  std::vector<double> p_end;        // Raw momentum at trajectory end

  // Pre-allocate all U-turn vectors to avoid heap allocation at every leaf node
  void init_vectors(int n) {
    rho.resize(n);
    p_sharp_beg.resize(n);
    p_sharp_end.resize(n);
    p_beg.resize(n);
    p_end.resize(n);
  }
};

// Generalized U-turn criterion: check if momenta are aligned with integrated direction
// Returns true if trajectory should CONTINUE (no U-turn detected)
inline bool compute_criterion(const double* p_sharp_minus, const double* p_sharp_plus,
                              const double* rho, int n) {
  double dot_fwd = 0.0, dot_bwd = 0.0;
  int i = 0;
  for (; i + 3 < n; i += 4) {
    dot_fwd += p_sharp_plus[i]   * rho[i]   + p_sharp_plus[i+1] * rho[i+1]
             + p_sharp_plus[i+2] * rho[i+2] + p_sharp_plus[i+3] * rho[i+3];
    dot_bwd += p_sharp_minus[i]   * rho[i]   + p_sharp_minus[i+1] * rho[i+1]
             + p_sharp_minus[i+2] * rho[i+2] + p_sharp_minus[i+3] * rho[i+3];
  }
  for (; i < n; i++) {
    dot_fwd += p_sharp_plus[i] * rho[i];
    dot_bwd += p_sharp_minus[i] * rho[i];
  }
  return (dot_fwd > 0.0) && (dot_bwd > 0.0);
}

// Fused U-turn criterion: constructs rho = a + b and computes dot products in one O(n) pass.
// Also writes constructed rho to rho_out for downstream use.
inline bool compute_criterion_fused(const double* p_sharp_minus, const double* p_sharp_plus,
                                    const double* a, const double* b,
                                    double* rho_out, int n) {
  double dot_fwd = 0.0, dot_bwd = 0.0;
  int i = 0;
  for (; i + 3 < n; i += 4) {
    double r0 = a[i]   + b[i];
    double r1 = a[i+1] + b[i+1];
    double r2 = a[i+2] + b[i+2];
    double r3 = a[i+3] + b[i+3];
    rho_out[i]   = r0; rho_out[i+1] = r1;
    rho_out[i+2] = r2; rho_out[i+3] = r3;
    dot_fwd += p_sharp_plus[i]   * r0 + p_sharp_plus[i+1]  * r1
             + p_sharp_plus[i+2] * r2 + p_sharp_plus[i+3]  * r3;
    dot_bwd += p_sharp_minus[i]   * r0 + p_sharp_minus[i+1] * r1
             + p_sharp_minus[i+2] * r2 + p_sharp_minus[i+3] * r3;
  }
  for (; i < n; i++) {
    double r = a[i] + b[i];
    rho_out[i] = r;
    dot_fwd += p_sharp_plus[i] * r;
    dot_bwd += p_sharp_minus[i] * r;
  }
  return (dot_fwd > 0.0) && (dot_bwd > 0.0);
}

// Pre-allocated buffer pool for NUTS tree building
// Eliminates all heap allocations in the build_tree hot path
struct NUTSWorkspace {
  int n;                    // n_params
  int max_depth;
  int stride;               // 3 * n (q + p + grad per slot)
  int total_slots;

  // Resolved gradient function pointer (set once, avoids per-call dispatch)
  GradientFn gradient_fn = nullptr;

  // Active symplectic integrator for this chain's trajectory leaves. Set at
  // chain setup from the process-global selection; when an adaptive integrator
  // is selected it is replaced at warmup end by the step-size-adapted
  // minimum-error scheme (simp::two_stage_adaptive / three_stage_adaptive at
  // the resolved nu_max). The in-place leapfrog leaf walks this scheme's op
  // sequence, so the coefficient is per-chain rather than a shared global.
  simp::Scheme scheme = simp::leapfrog();

  // Multiple-time-stepping (RESPA) leaf. When mts is set the leaf splits the
  // force into a cheap stiff prior part (prior_gradient_fn, m inner substeps)
  // and an expensive smooth likelihood part (one full gradient per leaf), so a
  // larger outer step handles the stiff prior without paying the likelihood at
  // the inner rate. prior_gradient_fn is resolved once at setup alongside
  // gradient_fn. mts_gp / mts is scratch reused across leaves.
  bool mts = false;
  int mts_m = 4;
  GradientFn prior_gradient_fn = nullptr;
  std::vector<double> mts_gp;  // prior (fast) gradient scratch, length n

  // Contiguous pool: [slot][q|p|grad], each n doubles wide
  std::vector<double> pool;
  std::vector<double> log_probs;  // One per slot

  // Slot allocator (stack-based, reset per top-level build_tree call)
  int next_slot;

  // Working buffers for compute_gradient (needs std::vector interface)
  std::vector<double> params_buf;
  std::vector<double> grad_buf;

  // Scratch buffer for dense mass matrix matvec (p doubles)
  std::vector<double> dense_scratch;

  // Pre-allocated merge buffers for build_tree_fast (depth-indexed, no per-merge alloc)
  // 4 buffers per depth level: rho_init, p_init_end, p_sharp_init_end, rho_check
  std::vector<double> merge_pool;
  static constexpr int MERGE_RHO_INIT = 0;
  static constexpr int MERGE_P_INIT_END = 1;
  static constexpr int MERGE_PSHARP_INIT_END = 2;
  static constexpr int MERGE_RHO_CHECK = 3;
  static constexpr int MERGE_BUFS_PER_DEPTH = 4;

  double* merge_buf(int depth, int buf_idx) {
    return &merge_pool[static_cast<size_t>(depth * MERGE_BUFS_PER_DEPTH + buf_idx) * n];
  }

  // Pre-allocated iteration-level vectors (reused across NUTS iterations)
  std::vector<double> iter_rho, iter_rho_bck, iter_rho_fwd;
  std::vector<double> iter_p_sharp_init;
  std::vector<double> iter_p_fwd_beg, iter_p_fwd_end;
  std::vector<double> iter_p_bck_beg, iter_p_bck_end;
  std::vector<double> iter_p_sharp_fwd_beg, iter_p_sharp_fwd_end;
  std::vector<double> iter_p_sharp_bck_beg, iter_p_sharp_bck_end;
  std::vector<double> iter_rho_seam;

  // Slot layout:
  //   Slots 0, 1: persistent trajectory endpoints (node_left, node_right)
  //   Slots 2+: allocated by build_tree_fast via alloc_slot()
  static constexpr int NODE_LEFT_SLOT = 0;
  static constexpr int NODE_RIGHT_SLOT = 1;
  static constexpr int TREE_START_SLOT = 2;

  void init(int np, int max_d) {
    n = np;
    max_depth = max_d;
    stride = 3 * np;
    // Max slots: 2 persistent + 2^max_depth for tree (generous upper bound)
    total_slots = 2 + (1 << max_d);
    pool.resize(static_cast<size_t>(total_slots) * stride);
    log_probs.resize(total_slots);
    params_buf.resize(np);
    grad_buf.resize(np);
    dense_scratch.resize(np);
    mts_gp.resize(np);
    next_slot = TREE_START_SLOT;
    // Pre-allocate merge buffers: 4 per depth level
    merge_pool.resize(static_cast<size_t>(MERGE_BUFS_PER_DEPTH) * max_d * np, 0.0);
    // Pre-allocate iteration-level vectors
    iter_rho.resize(np);
    iter_rho_bck.resize(np);
    iter_rho_fwd.resize(np);
    iter_p_sharp_init.resize(np);
    iter_p_fwd_beg.resize(np);
    iter_p_fwd_end.resize(np);
    iter_p_bck_beg.resize(np);
    iter_p_bck_end.resize(np);
    iter_p_sharp_fwd_beg.resize(np);
    iter_p_sharp_fwd_end.resize(np);
    iter_p_sharp_bck_beg.resize(np);
    iter_p_sharp_bck_end.resize(np);
    iter_rho_seam.resize(np);
  }

  // Allocate a fresh slot (stack-based, no deallocation)
  // Returns -1 if workspace is exhausted
  int alloc_slot() {
    if (next_slot >= total_slots) return -1;
    return next_slot++;
  }

  // Reset allocator for new top-level build_tree call
  // Preserves persistent slots 0 and 1
  void reset_tree() {
    next_slot = TREE_START_SLOT;
  }

  // Access helpers (raw pointers into contiguous pool)
  double* q_at(int slot) { return &pool[slot * stride]; }
  double* p_at(int slot) { return &pool[slot * stride + n]; }
  double* grad_at(int slot) { return &pool[slot * stride + 2 * n]; }
  double& logp_at(int slot) { return log_probs[slot]; }

  // Copy full node (q + p + grad + log_prob) between slots
  void copy_node(int dst, int src) {
    std::memcpy(&pool[dst * stride], &pool[src * stride],
                stride * sizeof(double));
    log_probs[dst] = log_probs[src];
  }

  // Load node data from std::vector sources into a slot
  void load_node(int slot, const double* q, const double* p,
                 const double* grad, double log_prob) {
    std::memcpy(q_at(slot), q, n * sizeof(double));
    std::memcpy(p_at(slot), p, n * sizeof(double));
    std::memcpy(grad_at(slot), grad, n * sizeof(double));
    log_probs[slot] = log_prob;
  }
};

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_SAMPLER_NUTS_INFRA_H
