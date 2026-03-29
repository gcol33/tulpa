// hmc_sampler.h
// Full HMC/NUTS backend with spatial, temporal, and zero-inflation support
// Supports ICAR/BYM2 spatial effects, RW/AR1 temporal, and ZI/hurdle models

#ifndef TULPA_HMC_SAMPLER_H
#define TULPA_HMC_SAMPLER_H

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include "linalg_fast.h"
#include "hmc_temporal.h"
#include "hmc_temporal_gp.h"
#include "hmc_zi.h"
#include "hmc_svc.h"
#include "hmc_gp.h"
#include "hmc_temporal_multiscale.h"
#include "hmc_latent.h"
#include "hmc_spatiotemporal.h"
#include "hmc_hsgp.h"
#include "hmc_tvc.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa_hmc {

// Import all canonical types from exported tulpa:: headers
using tulpa::ModelData;
using tulpa::ParamLayout;
using tulpa::ProcessData;
using tulpa::SharingSpec;
using tulpa::TemporalType;
using tulpa::TemporalData;
using tulpa::MultiscaleTemporalData;
using tulpa::TemporalGPData;
using tulpa::TemporalCovType;
using tulpa::ZIType;
using tulpa::GPData;
using tulpa::MultiscaleGPData;
using tulpa::CovType;
using tulpa::STType;
using tulpa::SpatiotemporalData;
using tulpa::NonsepType;
using tulpa::SVCData;
using tulpa::HSGPData;
using tulpa::TVCData;
using tulpa::ModelType;
using tulpa::SpatialType;
using tulpa::GradientMode;
using tulpa::MassMatrixType;
using tulpa::GPSolverConfig;
using tulpa::GPSolver;
using tulpa::MSGPSampler;

// Parse gradient mode from string
inline GradientMode parse_gradient_mode(const std::string& mode_str) {
    static const tulpa::EnumEntry<GradientMode> table[] = {
        {"auto", GradientMode::AUTO}, {"AUTO", GradientMode::AUTO},
        {"N", GradientMode::NUMERICAL}, {"numerical", GradientMode::NUMERICAL},
        {"A_t", GradientMode::AUTODIFF_TAPE}, {"autodiff_tape", GradientMode::AUTODIFF_TAPE},
        {"A_r", GradientMode::AUTODIFF_ARENA}, {"arena", GradientMode::AUTODIFF_ARENA},
        {"autodiff_arena", GradientMode::AUTODIFF_ARENA},
        {"A", GradientMode::AUTODIFF_FWD}, {"autodiff", GradientMode::AUTODIFF_FWD},
        {"forward", GradientMode::AUTODIFF_FWD},
        {"H", GradientMode::HANDCODED}, {"handcoded", GradientMode::HANDCODED},
        {"analytical", GradientMode::HANDCODED}
    };
    return tulpa::parse_enum(mode_str, table, GradientMode::AUTO);
}

// Parse metric type from string
inline MassMatrixType parse_metric_type(const std::string& metric_str) {
    static const tulpa::EnumEntry<MassMatrixType> table[] = {
        {"dense", MassMatrixType::DENSE}, {"DENSE", MassMatrixType::DENSE},
        {"block_diag", MassMatrixType::BLOCK_DIAG}, {"BLOCK_DIAG", MassMatrixType::BLOCK_DIAG},
        {"auto", MassMatrixType::AUTO}, {"AUTO", MassMatrixType::AUTO}
    };
    return tulpa::parse_enum(metric_str, table, MassMatrixType::DIAG);
}

// Human-readable metric name for verbose logging
inline const char* metric_name(MassMatrixType t) {
    switch (t) {
        case MassMatrixType::DIAG: return "DIAG";
        case MassMatrixType::DENSE: return "DENSE";
        case MassMatrixType::BLOCK_DIAG: return "BLOCK_DIAG";
        case MassMatrixType::AUTO: return "AUTO";
    }
    return "UNKNOWN";
}

// Set/get global gradient mode (defined in hmc_sampler.cpp)
void set_gradient_mode(GradientMode mode);
GradientMode get_gradient_mode();

// Reset VecGradWorkspace cache (for new model fit)
void reset_grad_workspace_cache();

// Function pointer type for gradient computation (eliminates per-call dispatch overhead)
using GradientFn = void(*)(
    const std::vector<double>&, const ModelData&, const ParamLayout&,
    std::vector<double>&, double*);

// Resolve the gradient function pointer once based on mode + model config
GradientFn resolve_gradient_fn(GradientMode mode, const ModelData& data, const ParamLayout& layout);

// ModelData and ParamLayout are now defined in exported headers:
//   inst/include/tulpa/model_data.h   (tulpa::ModelData)
//   inst/include/tulpa/param_layout.h (tulpa::ParamLayout)
// Imported into tulpa_hmc namespace via using declarations above.

ParamLayout compute_param_layout(const ModelData& data);
int get_n_params(const ModelData& data);

// =====================================================================
// Log-posterior computation (with OpenMP parallelization)
// =====================================================================

// Main log-posterior function
// When skip_obs_loop=true, returns only prior+structural terms (O(p+S+T)),
// skipping the O(N) observation loop. Used by fused gradient+log_post computation.
double compute_log_post(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    bool skip_obs_loop = false,
    const double* precomputed_st_log_prior = nullptr,
    const double* precomputed_tgp_log_prior = nullptr
);

// Gradient computation (with optional fused log-posterior)
// When log_post_out is non-null, the log-posterior is computed alongside
// the gradient in a single pass, avoiding redundant O(N) computation.
void compute_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
);

// Compute ICAR quadratic form: phi' Q phi
double icar_quadratic_form(
    const std::vector<double>& phi,
    const ModelData& data
);

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

// =====================================================================
// Block-diagonal mass block (max 4×4, stack-allocated)
// =====================================================================

struct MassBlock {
  int start = 0;          // First param index in full parameter vector
  int size = 0;           // Block size (2-4)
  bool adapted = false;

  // Block mass storage (column-major, max 4×4)
  double inv_mass[16] = {};    // C_block (covariance block)
  double L_inv_mass[16] = {};  // Cholesky L where LL^T = C_block

  // Block-local Welford covariance accumulator
  int welford_n = 0;
  double welford_mean[4] = {};
  double welford_M2[16] = {};  // Running sum for covariance (column-major)

  void init(int s, int sz) {
    start = s;
    size = sz;
    adapted = false;
    std::memset(inv_mass, 0, sizeof(inv_mass));
    std::memset(L_inv_mass, 0, sizeof(L_inv_mass));
    // Initialize as identity
    for (int i = 0; i < sz; i++) {
      inv_mass[i * 4 + i] = 1.0;  // Using stride=4 (max block size)
      L_inv_mass[i * 4 + i] = 1.0;
    }
    reset_welford();
  }

  void reset_welford() {
    welford_n = 0;
    std::memset(welford_mean, 0, sizeof(welford_mean));
    std::memset(welford_M2, 0, sizeof(welford_M2));
  }

  // Extract block params and update Welford running stats
  void welford_update(const double* full_params) {
    welford_n++;
    double delta[4];
    for (int i = 0; i < size; i++) {
      delta[i] = full_params[start + i] - welford_mean[i];
      welford_mean[i] += delta[i] / welford_n;
    }
    for (int i = 0; i < size; i++) {
      double dx_new = full_params[start + i] - welford_mean[i];
      for (int j = 0; j <= i; j++) {
        double val = dx_new * delta[j];
        welford_M2[j * 4 + i] += val;  // stride=4
        if (i != j) {
          welford_M2[i * 4 + j] += val;
        }
      }
    }
  }

  // Compute covariance from Welford stats, Cholesky decompose, set adapted
  bool update_from_welford() {
    if (welford_n < 10) return false;

    // Compute sample covariance with small regularization
    double cov[16] = {};
    double scale = 1.0 / (welford_n - 1);
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        cov[i * 4 + j] = welford_M2[j * 4 + i] * scale;  // Note: M2 is col-major with stride 4
      }
      cov[i * 4 + i] += 1e-8;  // Regularization
    }

    // Try Cholesky decomposition
    double L[16] = {};
    if (!cholesky_small(cov, L, size)) return false;

    // Success: copy to block storage
    std::memcpy(inv_mass, cov, sizeof(inv_mass));
    std::memcpy(L_inv_mass, L, sizeof(L_inv_mass));
    adapted = true;
    return true;
  }

  // Tiny Cholesky for k<=4 (direct formula, no Eigen)
  // A and L use stride=4 (max block size)
  static bool cholesky_small(const double* A, double* L, int k) {
    std::memset(L, 0, 16 * sizeof(double));
    for (int i = 0; i < k; i++) {
      double sum = 0.0;
      for (int p = 0; p < i; p++) {
        sum += L[i * 4 + p] * L[i * 4 + p];
      }
      double diag = A[i * 4 + i] - sum;
      if (diag <= 0.0) return false;
      L[i * 4 + i] = std::sqrt(diag);
      for (int j = i + 1; j < k; j++) {
        double s = 0.0;
        for (int p = 0; p < i; p++) {
          s += L[j * 4 + p] * L[i * 4 + p];
        }
        L[j * 4 + i] = (A[j * 4 + i] - s) / L[i * 4 + i];
      }
    }
    return true;
  }

  // result[0..size-1] = C_block * p[start..start+size-1]
  void matvec(const double* p_full, double* result) const {
    const double* pb = p_full + start;
    for (int i = 0; i < size; i++) {
      double sum = 0.0;
      for (int j = 0; j < size; j++) {
        sum += inv_mass[i * 4 + j] * pb[j];
      }
      result[i] = sum;
    }
  }

  // p_block^T * C_block * p_block
  double quadform(const double* p_full) const {
    const double* pb = p_full + start;
    double result = 0.0;
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        result += pb[i] * inv_mass[i * 4 + j] * pb[j];
      }
    }
    return result;
  }

  // Sample momentum for block: p_block = L^{-T} z (back-substitution on tiny L)
  void sample_momentum(double* p_full, std::mt19937& rng) const {
    if (!adapted) return;  // Non-adapted blocks use diagonal path
    std::normal_distribution<double> normal(0.0, 1.0);
    double z[4];
    for (int i = 0; i < size; i++) z[i] = normal(rng);

    // Back-substitution: solve L^T * p = z (upper triangular)
    double* pb = p_full + start;
    for (int i = size - 1; i >= 0; i--) {
      double sum = z[i];
      for (int j = i + 1; j < size; j++) {
        sum -= L_inv_mass[j * 4 + i] * pb[j];  // L^T[i][j] = L[j][i]
      }
      pb[i] = sum / L_inv_mass[i * 4 + i];
    }
  }
};

// =====================================================================
// Precision-informed mass block (heap-allocated, arbitrary size)
// Used for ICAR/BYM2 spatial params where Q (precision) is known analytically.
// Unlike MassBlock (≤4, stack), this handles S×S blocks (S=50 typical).
// NOT adapted from samples — uses fixed analytical precision.
// =====================================================================

struct PrecisionBlock {
  int start = 0;        // First param index in full parameter vector
  int size = 0;         // Block dimension S
  bool active = false;

  // M^{-1} = Q_reg_inv: (Q + lambda*I)^{-1}, column-major S×S
  std::vector<double> Q_inv;
  // L_Q: Cholesky factor where L*L^T = Q + lambda*I, column-major S×S
  // Used for momentum sampling: p_block = L^{-T} * z
  std::vector<double> L_chol;

  void init(int s, int sz, const double* q_inv_data, const double* l_chol_data) {
    start = s;
    size = sz;
    active = true;
    int nn = static_cast<int>(static_cast<size_t>(sz) * sz);
    Q_inv.assign(q_inv_data, q_inv_data + nn);
    L_chol.assign(l_chol_data, l_chol_data + nn);
  }

  // result[0..size-1] = Q_inv * p[start..start+size-1]
  void matvec(const double* p_full, double* result) const {
    const double* pb = p_full + start;
    Eigen::Map<const Eigen::MatrixXd> M(Q_inv.data(), size, size);
    Eigen::Map<const Eigen::VectorXd> pv(pb, size);
    Eigen::Map<Eigen::VectorXd> rv(result, size);
    rv.noalias() = M.selfadjointView<Eigen::Lower>() * pv;
  }

  // p_block^T * Q_inv * p_block
  double quadform(const double* p_full) const {
    const double* pb = p_full + start;
    Eigen::Map<const Eigen::MatrixXd> M(Q_inv.data(), size, size);
    Eigen::Map<const Eigen::VectorXd> pv(pb, size);
    return pv.dot(M.selfadjointView<Eigen::Lower>() * pv);
  }

  // Sample momentum for block: solve L^T * p = z (back-substitution)
  void sample_momentum(double* p_full, std::mt19937& rng) const {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> z(size);
    for (int i = 0; i < size; i++) z[i] = normal(rng);
    double* pb = p_full + start;
    Eigen::Map<const Eigen::MatrixXd> L(L_chol.data(), size, size);
    Eigen::Map<const Eigen::VectorXd> zv(z.data(), size);
    Eigen::Map<Eigen::VectorXd> pv(pb, size);
    pv.noalias() = L.transpose().triangularView<Eigen::Upper>().solve(zv);
  }
};

// =====================================================================
// Kronecker precision block for spatiotemporal (ST) interaction params.
// M = Q_space ⊗ Q_time, M^{-1} = Q_space^{-1} ⊗ Q_time^{-1}
// Never forms the full (S*T)×(S*T) matrix — O(S*T*(S+T)) operations.
// =====================================================================

struct KroneckerBlock {
  int start = 0;        // First ST param index in full parameter vector
  int S = 0;            // Spatial dimension
  int T = 0;            // Temporal dimension
  bool active = false;

  // Spatial: Q_space_inv (S×S), L_space (Cholesky of Q_space, S×S)
  std::vector<double> Qs_inv;  // column-major
  std::vector<double> Ls;      // column-major

  // Temporal: Q_time_inv (T×T), L_time (Cholesky of Q_time, T×T)
  std::vector<double> Qt_inv;  // column-major
  std::vector<double> Lt;      // column-major

  // Scratch buffers (pre-allocated)
  mutable std::vector<double> scratch_ST;  // S*T work buffer

  void init(int st, int ns, int nt,
            const double* qs_inv, const double* ls,
            const double* qt_inv, const double* lt) {
    start = st;
    S = ns;
    T = nt;
    active = true;
    int ss = S * S, tt = T * T;
    Qs_inv.assign(qs_inv, qs_inv + ss);
    Ls.assign(ls, ls + ss);
    Qt_inv.assign(qt_inv, qt_inv + tt);
    Lt.assign(lt, lt + tt);
    scratch_ST.resize(static_cast<size_t>(S) * T);
  }

  // Compute (A ⊗ B) * vec(X) = vec(B * X * A^T)
  // where X is S×T (column-major), A is T×T, B is S×S
  // result = vec(B * X * A^T)
  // This is the standard Kronecker-vector product identity.
  void kron_matvec(const double* As, int na, const double* Bt, int nb,
                   const double* x, double* result) const {
    // Step 1: tmp = X * A^T  (S×T * T×T = S×T)
    // X is S×T column-major, A^T is T×T
    Eigen::Map<const Eigen::MatrixXd> X(x, S, T);
    Eigen::Map<const Eigen::MatrixXd> A(Bt, T, T);  // temporal
    Eigen::Map<const Eigen::MatrixXd> B(As, S, S);   // spatial
    Eigen::Map<Eigen::MatrixXd> R(result, S, T);

    // (B ⊗ A) * vec(X) = vec(B * X * A^T)
    // But our params are stored with spatial varying fastest: param[s + t*S]
    // So X_{s,t} = x[s + t*S] which is column-major S×T.
    // M^{-1} = Qs_inv ⊗ Qt_inv
    // M^{-1} * x = vec(Qs_inv * X * Qt_inv^T)
    R.noalias() = B * X * A.transpose();
  }

  // result[0..S*T-1] = (Qs_inv ⊗ Qt_inv) * p[start..start+S*T-1]
  void matvec(const double* p_full, double* result) const {
    kron_matvec(Qs_inv.data(), S, Qt_inv.data(), T,
                p_full + start, result);
  }

  // p_block^T * (Qs_inv ⊗ Qt_inv) * p_block
  double quadform(const double* p_full) const {
    matvec(p_full, scratch_ST.data());
    const double* pb = p_full + start;
    double qf = 0.0;
    int ST = S * T;
    for (int i = 0; i < ST; i++) qf += pb[i] * scratch_ST[i];
    return qf;
  }

  // Sample momentum: p ~ N(0, M) where M = Qs ⊗ Qt
  // p = (Ls ⊗ Lt)^{-T} * z = vec(Ls^{-T} * Z * Lt^{-1})
  // where Z is S×T standard normal
  void sample_momentum(double* p_full, std::mt19937& rng) const {
    std::normal_distribution<double> normal(0.0, 1.0);
    int ST = S * T;
    // Generate Z ~ N(0,I) as S×T matrix
    std::vector<double> Z(ST);
    for (int i = 0; i < ST; i++) Z[i] = normal(rng);

    double* pb = p_full + start;
    Eigen::Map<Eigen::MatrixXd> Zm(Z.data(), S, T);
    Eigen::Map<const Eigen::MatrixXd> Lsm(Ls.data(), S, S);
    Eigen::Map<const Eigen::MatrixXd> Ltm(Lt.data(), T, T);
    Eigen::Map<Eigen::MatrixXd> Pm(pb, S, T);

    // (Ls ⊗ Lt)^{-T} * z = vec(Ls^{-T} * Z * Lt^{-1})
    // Step 1: solve Ls^T * tmp = Z  →  tmp = Ls^{-T} * Z
    Eigen::MatrixXd tmp = Lsm.transpose().triangularView<Eigen::Upper>().solve(Zm);
    // Step 2: solve tmp2 * Lt^T = tmp  →  tmp2 = tmp * Lt^{-T}
    //   which is (Lt^{-T} * tmp^T)^T = (Lt * tmp^T)^{-T}
    //   Actually: tmp2 * Lt^T = tmp → tmp2 = tmp * Lt^{-T}
    //   Transpose: Lt^{-1} * tmp^T → solve Lt * Y = tmp^T → Y = Lt^{-1} * tmp^T
    //   Then result = Y^T
    Eigen::MatrixXd Y = Ltm.triangularView<Eigen::Lower>().solve(tmp.transpose());
    Pm = Y.transpose();
  }
};

// =====================================================================
// Sparse GMRF block for ST_IV spatiotemporal interaction.
// Uses Eigen sparse Cholesky for:
//   1. Block Gibbs sampling: delta ~ N(Q^{-1}b, Q^{-1})
//   2. Mass matrix operations (momentum, kinetic energy, inv_mass*p)
// Q = tau * (Q_s ⊗ Q_t) + lambda*I (posterior precision for delta block)
// =====================================================================

struct SparseGMRFBlock {
  int start = 0;        // First ST param index in full parameter vector
  int S = 0;            // Spatial dimension
  int T = 0;            // Temporal dimension
  bool active = false;
  bool factorized = false;

  // Sparse precision and its Cholesky factorization
  Eigen::SparseMatrix<double> Q_sparse;  // ST×ST sparse precision matrix
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt;  // Cholesky LL^T = Q

  // Scratch buffers (pre-allocated)
  mutable Eigen::VectorXd scratch_vec;

  void init(int st_start, int ns, int nt) {
    start = st_start;
    S = ns;
    T = nt;
    active = true;
    factorized = false;
    int ST = S * T;
    scratch_vec.resize(ST);
  }

  // Build Q_sparse = tau * (Q_s ⊗ Q_t) + diag_correction
  // adj_row_ptr/adj_col_idx: CSR adjacency for spatial graph (1-based col_idx!)
  // temp_type: RW1, RW2, AR1
  // tau: precision parameter
  // h_lik: diagonal of likelihood Hessian (length ST), or nullptr for prior-only
  // lambda_stz: sum-to-zero penalty (default 0.001)
  void build_and_factorize(
      const std::vector<int>& adj_row_ptr,
      const std::vector<int>& adj_col_idx,
      tulpa_temporal::TemporalType temp_type,
      bool cyclic,
      double tau,
      const double* h_lik,  // diagonal Hessian correction (length S*T), can be nullptr
      double lambda_stz = 0.001
  ) {
    int ST = S * T;

    // Build spatial Laplacian Q_s (S×S)
    // Q_s[i,i] = degree(i), Q_s[i,j] = -1 if adjacent
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(ST * 7);  // Rough estimate: ~7 nonzeros per row

    // Build temporal precision Q_t (T×T) as dense small matrix
    Eigen::MatrixXd Qt = Eigen::MatrixXd::Zero(T, T);
    if (temp_type == tulpa_temporal::TemporalType::RW1) {
      for (int t = 0; t < T - 1; t++) {
        Qt(t, t) += 1.0;
        Qt(t + 1, t + 1) += 1.0;
        Qt(t, t + 1) = -1.0;
        Qt(t + 1, t) = -1.0;
      }
    } else if (temp_type == tulpa_temporal::TemporalType::RW2) {
      for (int t = 0; t < T - 2; t++) {
        Qt(t, t) += 1.0;
        Qt(t + 1, t + 1) += 4.0;
        Qt(t + 2, t + 2) += 1.0;
        Qt(t, t + 1) += -2.0;
        Qt(t + 1, t) += -2.0;
        Qt(t, t + 2) += 1.0;
        Qt(t + 2, t) += 1.0;
        Qt(t + 1, t + 2) += -2.0;
        Qt(t + 2, t + 1) += -2.0;
      }
      // Fix double-counted diagonal
      for (int t = 0; t < T; t++) Qt(t, t) = 0.0;
      Eigen::MatrixXd D = Eigen::MatrixXd::Zero(T - 2, T);
      for (int t = 0; t < T - 2; t++) {
        D(t, t) = 1.0; D(t, t + 1) = -2.0; D(t, t + 2) = 1.0;
      }
      Qt = D.transpose() * D;
    } else {
      // AR1 with rho=0.5 as default approximation
      for (int t = 0; t < T; t++) Qt(t, t) = 1.0;
      for (int t = 0; t < T - 1; t++) {
        Qt(t, t + 1) = -0.5;
        Qt(t + 1, t) = -0.5;
      }
    }

    // Build Kronecker product Q_kron = Q_s ⊗ Q_t as sparse triplets
    // (Q_s ⊗ Q_t)[(s1*T+t1), (s2*T+t2)] = Q_s[s1,s2] * Q_t[t1,t2]
    // For each spatial pair (s1,s2) with Q_s[s1,s2] != 0:
    //   For each temporal pair (t1,t2) with Q_t[t1,t2] != 0:
    //     Add Q_s[s1,s2] * Q_t[t1,t2] at row s1*T+t1, col s2*T+t2
    for (int s1 = 0; s1 < S; s1++) {
      int n_neigh = adj_row_ptr[s1 + 1] - adj_row_ptr[s1];
      double qs_diag = static_cast<double>(n_neigh);

      // Diagonal spatial block: Q_s[s1,s1] = degree
      for (int t1 = 0; t1 < T; t1++) {
        for (int t2 = 0; t2 < T; t2++) {
          if (Qt(t1, t2) != 0.0) {
            triplets.emplace_back(s1 * T + t1, s1 * T + t2, tau * qs_diag * Qt(t1, t2));
          }
        }
      }

      // Off-diagonal spatial: Q_s[s1,s2] = -1 for neighbors
      for (int idx = adj_row_ptr[s1]; idx < adj_row_ptr[s1 + 1]; idx++) {
        int s2 = adj_col_idx[idx] - 1;  // Convert 1-based to 0-based
        for (int t1 = 0; t1 < T; t1++) {
          for (int t2 = 0; t2 < T; t2++) {
            if (Qt(t1, t2) != 0.0) {
              triplets.emplace_back(s1 * T + t1, s2 * T + t2, tau * (-1.0) * Qt(t1, t2));
            }
          }
        }
      }
    }

    // Add diagonal corrections: likelihood Hessian + sum-to-zero penalty + regularization
    double reg = 1e-6;  // Numerical regularization for rank deficiency
    for (int k = 0; k < ST; k++) {
      double diag_add = lambda_stz + reg;  // sum-to-zero soft constraint
      if (h_lik) diag_add += h_lik[k];     // likelihood curvature
      triplets.emplace_back(k, k, diag_add);
    }

    Q_sparse.resize(ST, ST);
    Q_sparse.setFromTriplets(triplets.begin(), triplets.end());
    Q_sparse.makeCompressed();

    // Cholesky factorization
    llt.compute(Q_sparse);
    factorized = (llt.info() == Eigen::Success);
  }

  // Sample delta from GMRF conditional: delta ~ N(Q^{-1}b, Q^{-1})
  // b = grad_delta_lik (likelihood gradient wrt delta)
  // Returns new delta values in delta_out (length S*T)
  void sample_conditional(const double* b, double* delta_out, std::mt19937& rng) const {
    if (!factorized) return;
    int ST = S * T;
    std::normal_distribution<double> normal(0.0, 1.0);

    // Step 1: Solve Q * mean = b  →  mean = Q^{-1} * b
    Eigen::Map<const Eigen::VectorXd> bv(b, ST);
    Eigen::VectorXd mean = llt.solve(bv);

    // Step 2: Sample z ~ N(0, I)
    Eigen::VectorXd z(ST);
    for (int i = 0; i < ST; i++) z[i] = normal(rng);

    // Step 3: delta = mean + perturbation from N(0, Q^{-1})
    // SimplicialLLT factorizes as: P * Q * P^T = L * L^T
    // To sample from N(0, Q^{-1}): pert = P^T * L^{-T} * z
    //   Var(pert) = P^T L^{-T} L^{-1} P = P^T (LL^T)^{-1} P = (P^T LL^T P)^{-1} = Q^{-1} ✓
    auto perm = llt.permutationP();
    // Get L as a concrete sparse matrix (avoids const-view issues)
    Eigen::SparseMatrix<double> L_mat = llt.matrixL();
    // Solve L^T * v = z (upper triangular solve on L^T)
    Eigen::SparseMatrix<double> Lt_mat = L_mat.transpose();
    Eigen::VectorXd v = Lt_mat.triangularView<Eigen::Upper>().solve(z);
    // Un-permute: pert = P^T * v
    Eigen::VectorXd pert = perm.transpose() * v;

    Eigen::Map<Eigen::VectorXd> out(delta_out, ST);
    out = mean + pert;
  }

  // Mass matrix operations (for HMC momentum/kinetic energy)
  // M = Q (posterior precision), M^{-1} = Q^{-1}

  // result = Q^{-1} * p  (inv_mass * momentum)
  void inv_mass_matvec(const double* p_full, double* result) const {
    if (!factorized) return;
    int ST = S * T;
    Eigen::Map<const Eigen::VectorXd> pv(p_full + start, ST);
    Eigen::VectorXd sol = llt.solve(pv);
    std::memcpy(result, sol.data(), ST * sizeof(double));
  }

  // p^T * Q^{-1} * p  (kinetic energy contribution)
  double quadform(const double* p_full) const {
    if (!factorized) return 0.0;
    int ST = S * T;
    Eigen::Map<const Eigen::VectorXd> pv(p_full + start, ST);
    Eigen::VectorXd sol = llt.solve(pv);
    return pv.dot(sol);
  }

  // Sample momentum p ~ N(0, Q):  p = L^T * z where LL^T = Q
  void sample_momentum(double* p_full, std::mt19937& rng) const {
    if (!factorized) return;
    int ST = S * T;
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::VectorXd z(ST);
    for (int i = 0; i < ST; i++) z[i] = normal(rng);

    // P * Q * P^T = L * L^T  →  p ~ N(0, Q) needs p = P^T * L^T * P * z
    // But simpler: p_perm = L^T * z ~ N(0, L^T L) = N(0, PQP^T)
    // Then p = P^T * p_perm ~ N(0, P^T PQP^T P) = N(0, Q) ✓
    auto perm = llt.permutationP();
    Eigen::SparseMatrix<double> L_mat = llt.matrixL();
    Eigen::VectorXd p_perm = L_mat.transpose() * z;  // L^T * z
    Eigen::VectorXd p_vec = perm.transpose() * p_perm;  // P^T * (L^T * z)
    double* pb = p_full + start;
    std::memcpy(pb, p_vec.data(), ST * sizeof(double));
  }
};

// =====================================================================
// Dense mass matrix for NUTS (encapsulates diag + dense state)
// =====================================================================

struct DenseMassMatrix {
  int n = 0;                              // Dimension
  MassMatrixType type = MassMatrixType::DIAG;
  bool adapted = false;

  // Diagonal (always available, used as fallback)
  std::vector<double> inv_mass_diag;      // M^{-1} diagonal = variance
  std::vector<double> sqrt_mass_diag;     // sqrt(M) diagonal = 1/sqrt(variance) for p sampling

  // Dense (only when type == DENSE)
  std::vector<double> inv_mass_dense;     // Full p×p M^{-1} = regularized sample covariance (column-major)
  std::vector<double> L_inv_mass;         // Cholesky factor L where LL^T = M^{-1} (column-major)

  // Scratch buffer for dense matvec results (avoids per-call allocation)
  std::vector<double> scratch;

  // Block-diagonal (only when type == BLOCK_DIAG)
  std::vector<MassBlock> blocks;
  std::vector<bool> in_block;  // in_block[i] = true if param i belongs to a block

  // Precision-informed blocks (optional, independent of type)
  // These override the mass for specific param ranges using known precision structure.
  PrecisionBlock precision_block;    // ICAR/BYM2 spatial block (DISABLED)
  KroneckerBlock kronecker_block;    // ST_IV Kronecker block (DISABLED)
  SparseGMRFBlock sparse_gmrf;       // ST_IV sparse GMRF mass + Gibbs sampling

  void init(int dim, MassMatrixType t) {
    n = dim;
    type = t;
    adapted = false;
    inv_mass_diag.assign(dim, 1.0);
    sqrt_mass_diag.assign(dim, 1.0);
    scratch.resize(dim);
    if (t == MassMatrixType::DENSE) {
      inv_mass_dense.assign(static_cast<size_t>(dim) * dim, 0.0);
      L_inv_mass.assign(static_cast<size_t>(dim) * dim, 0.0);
      // Initialize as identity
      for (int i = 0; i < dim; i++) {
        inv_mass_dense[static_cast<size_t>(i) * dim + i] = 1.0;
        L_inv_mass[static_cast<size_t>(i) * dim + i] = 1.0;
      }
    }
    if (t == MassMatrixType::BLOCK_DIAG) {
      in_block.assign(dim, false);
    }
  }

  // Initialize block-diagonal structure from block specifications
  // block_specs: vector of (start_index, block_size) pairs
  void init_block_diag(int dim, const std::vector<std::pair<int,int>>& block_specs) {
    init(dim, MassMatrixType::BLOCK_DIAG);
    blocks.clear();
    blocks.reserve(block_specs.size());
    for (const auto& spec : block_specs) {
      MassBlock blk;
      blk.init(spec.first, spec.second);
      blocks.push_back(blk);
      for (int i = spec.first; i < spec.first + spec.second; i++) {
        if (i < dim) in_block[i] = true;
      }
    }
  }

  // Update dense mass matrix from sample covariance
  // Returns true on success, false on Cholesky failure (degrades to diagonal)
  // Uses Eigen LLT for Cholesky decomposition
  bool update_from_covariance(const double* cov, int n_samples);

  // Sample momentum: p ~ N(0, M) where M = C^{-1}
  // DIAG: p[i] = z * sqrt_mass_diag[i]
  // BLOCK_DIAG: diagonal for non-block params, L^{-T} z for block params
  // DENSE: solve L^T * p = z  (back-substitution)
  // Uses Eigen triangular solve for dense case (n>=16) for SIMD acceleration.
  void sample_momentum(double* p, std::mt19937& rng) const {
    std::normal_distribution<double> normal(0.0, 1.0);
    if (type == MassMatrixType::BLOCK_DIAG && adapted) {
      // First: diagonal for all params
      for (int i = 0; i < n; i++) {
        p[i] = normal(rng) * sqrt_mass_diag[i];
      }
      // Then: overwrite block params with correlated samples
      for (const auto& blk : blocks) {
        if (blk.adapted) {
          blk.sample_momentum(p, rng);
        }
      }
    } else if (type == MassMatrixType::DIAG || !adapted) {
      for (int i = 0; i < n; i++) {
        p[i] = normal(rng) * sqrt_mass_diag[i];
      }
    } else {
      // Dense: p = L^{-T} z where LL^T = C (inv_mass)
      // We need p ~ N(0, C^{-1}), so sample z ~ N(0, I), then p = L^{-T} z
      std::vector<double> z(n);
      for (int i = 0; i < n; i++) {
        z[i] = normal(rng);
      }
      if (n >= 16) {
        Eigen::Map<const Eigen::MatrixXd> Lm(L_inv_mass.data(), n, n);
        Eigen::Map<const Eigen::VectorXd> zv(z.data(), n);
        Eigen::Map<Eigen::VectorXd> pv(p, n);
        // Solve L^T * p = z: transpose L then use upper-triangular solve
        pv.noalias() = Lm.transpose().triangularView<Eigen::Upper>().solve(zv);
      } else {
        tulpa_linalg::tri_solve_upper_transpose(L_inv_mass.data(), z.data(), p, n);
      }
    }
    // Precision/Kronecker blocks override their param ranges
    if (precision_block.active) precision_block.sample_momentum(p, rng);
    if (kronecker_block.active) kronecker_block.sample_momentum(p, rng);
    if (sparse_gmrf.active && sparse_gmrf.factorized) sparse_gmrf.sample_momentum(p, rng);
  }

  // Check if param i belongs to a precision, kronecker, or sparse GMRF block
  inline bool in_precision_block(int i) const {
    if (precision_block.active &&
        i >= precision_block.start &&
        i < precision_block.start + precision_block.size) return true;
    if (kronecker_block.active &&
        i >= kronecker_block.start &&
        i < kronecker_block.start + kronecker_block.S * kronecker_block.T) return true;
    if (sparse_gmrf.active && sparse_gmrf.factorized &&
        i >= sparse_gmrf.start &&
        i < sparse_gmrf.start + sparse_gmrf.S * sparse_gmrf.T) return true;
    return false;
  }

  // Kinetic energy: 0.5 * p^T * C * p  where C = M^{-1}
  // Uses Eigen BLAS for dense case (n>=16) for SIMD acceleration.
  double kinetic_energy(const double* p) const {
    // Precision/Kronecker/Sparse blocks: compute their contribution separately
    double ke_prec = 0.0;
    if (precision_block.active) ke_prec += precision_block.quadform(p);
    if (kronecker_block.active) ke_prec += kronecker_block.quadform(p);
    if (sparse_gmrf.active && sparse_gmrf.factorized) ke_prec += sparse_gmrf.quadform(p);

    if (type == MassMatrixType::BLOCK_DIAG && adapted) {
      double ke = 0.0;
      for (int i = 0; i < n; i++) {
        if (!in_block[i] && !in_precision_block(i)) {
          ke += inv_mass_diag[i] * p[i] * p[i];
        }
      }
      for (const auto& blk : blocks) {
        if (blk.adapted) {
          ke += blk.quadform(p);
        } else {
          for (int i = blk.start; i < blk.start + blk.size; i++) {
            if (!in_precision_block(i))
              ke += inv_mass_diag[i] * p[i] * p[i];
          }
        }
      }
      return 0.5 * (ke + ke_prec);
    } else if (type == MassMatrixType::DIAG || !adapted) {
      if (!precision_block.active && !kronecker_block.active) {
        return 0.5 * tulpa_linalg::weighted_norm_squared(p, inv_mass_diag.data(), n);
      }
      // Skip precision block params in diagonal sum
      double ke = 0.0;
      for (int i = 0; i < n; i++) {
        if (!in_precision_block(i))
          ke += inv_mass_diag[i] * p[i] * p[i];
      }
      return 0.5 * (ke + ke_prec);
    } else if (n >= 16) {
      // Dense: full matrix handles all params including precision block range
      // But if precision blocks are active, we need to exclude their range
      // from the dense contribution and add the precision block contribution instead.
      // For simplicity: if precision blocks active, fall back to per-element
      if (precision_block.active || kronecker_block.active) {
        double ke = 0.0;
        for (int i = 0; i < n; i++) {
          if (in_precision_block(i)) continue;
          for (int j = 0; j < n; j++) {
            if (in_precision_block(j)) continue;
            ke += p[i] * inv_mass_dense[static_cast<size_t>(j) * n + i] * p[j];
          }
        }
        return 0.5 * (ke + ke_prec);
      }
      Eigen::Map<const Eigen::MatrixXd> Am(inv_mass_dense.data(), n, n);
      Eigen::Map<const Eigen::VectorXd> pv(p, n);
      return 0.5 * pv.dot(Am.selfadjointView<Eigen::Lower>() * pv);
    } else {
      return 0.5 * tulpa_linalg::quadratic_form(p, inv_mass_dense.data(), n);
    }
  }

  // Compute C * p (for leapfrog position update: q += eps * C * p)
  // Result written to `result` buffer.
  // Uses Eigen BLAS for dense case (n>=16) for SIMD acceleration.
  void inv_mass_times_p(const double* p, double* result) const {
    if (type == MassMatrixType::BLOCK_DIAG && adapted) {
      for (int i = 0; i < n; i++) {
        result[i] = inv_mass_diag[i] * p[i];
      }
      for (const auto& blk : blocks) {
        if (blk.adapted) {
          double tmp[4];
          blk.matvec(p, tmp);
          for (int i = 0; i < blk.size; i++) {
            result[blk.start + i] = tmp[i];
          }
        }
      }
    } else if (type == MassMatrixType::DIAG || !adapted) {
      for (int i = 0; i < n; i++) {
        result[i] = inv_mass_diag[i] * p[i];
      }
    } else if (n >= 16) {
      Eigen::Map<const Eigen::MatrixXd> Am(inv_mass_dense.data(), n, n);
      Eigen::Map<const Eigen::VectorXd> pv(p, n);
      Eigen::Map<Eigen::VectorXd> rv(result, n);
      rv.noalias() = Am.selfadjointView<Eigen::Lower>() * pv;
    } else {
      tulpa_linalg::symmatvec(inv_mass_dense.data(), p, result, n);
    }
    // Precision/Kronecker blocks override their param ranges
    if (precision_block.active) {
      std::vector<double> tmp(precision_block.size);
      precision_block.matvec(p, tmp.data());
      for (int i = 0; i < precision_block.size; i++) {
        result[precision_block.start + i] = tmp[i];
      }
    }
    if (kronecker_block.active) {
      int ST = kronecker_block.S * kronecker_block.T;
      std::vector<double> tmp(ST);
      kronecker_block.matvec(p, tmp.data());
      for (int i = 0; i < ST; i++) {
        result[kronecker_block.start + i] = tmp[i];
      }
    }
    if (sparse_gmrf.active && sparse_gmrf.factorized) {
      int ST = sparse_gmrf.S * sparse_gmrf.T;
      std::vector<double> tmp(ST);
      sparse_gmrf.inv_mass_matvec(p, tmp.data());
      for (int i = 0; i < ST; i++) {
        result[sparse_gmrf.start + i] = tmp[i];
      }
    }
  }

  // Compute diag(C) * p — uses diagonal only, even when dense is available.
  // Kept for backwards compatibility / debugging. The NUTS U-turn criterion
  // now uses inv_mass_times_p() for correct geometry with dense mass.
  void inv_mass_diag_times_p(const double* p, double* result) const {
    for (int i = 0; i < n; i++) {
      result[i] = inv_mass_diag[i] * p[i];
    }
  }

  // Set metric directly from precomputed G^{-1} and its Cholesky L.
  // Used by SoftAbs per-trajectory metric retry. No shrinkage applied.
  void set_from_metric(const std::vector<double>& g_inv,
                       const std::vector<double>& l_g_inv) {
    inv_mass_dense = g_inv;
    L_inv_mass = l_g_inv;
    for (int i = 0; i < n; i++) {
      inv_mass_diag[i] = g_inv[static_cast<size_t>(i) * n + i];
      sqrt_mass_diag[i] = 1.0 / std::sqrt(std::max(inv_mass_diag[i], 1e-10));
    }
    adapted = true;
  }

  // Set diagonal mass from WelfordStats output (same interface as before)
  // When type==DENSE, also populate the dense matrices as diagonal so that
  // the dense code paths (sample_momentum, kinetic_energy, inv_mass_times_p)
  // produce correct results even before full covariance is available.
  // When type==BLOCK_DIAG, diagonal is set normally; blocks are adapted separately
  // via their own Welford accumulators.
  void set_diagonal(const std::vector<double>& inv_m, const std::vector<double>& sqrt_m) {
    inv_mass_diag = inv_m;
    sqrt_mass_diag = sqrt_m;
    adapted = true;

    if (type == MassMatrixType::DENSE && !inv_mass_dense.empty()) {
      // Populate dense matrices as diagonal so dense code paths work correctly
      std::fill(inv_mass_dense.begin(), inv_mass_dense.end(), 0.0);
      std::fill(L_inv_mass.begin(), L_inv_mass.end(), 0.0);
      for (int i = 0; i < n; i++) {
        inv_mass_dense[static_cast<size_t>(i) * n + i] = inv_m[i];
        // L where LL^T = inv_mass (diagonal): L[i,i] = sqrt(inv_mass[i])
        L_inv_mass[static_cast<size_t>(i) * n + i] = std::sqrt(inv_m[i]);
      }
    }
  }
};

// =====================================================================
// Online full covariance estimator (Welford's algorithm for outer products)
// =====================================================================

class WelfordCovStats {
public:
  int dim;
  int n;
  std::vector<double> mean;
  std::vector<double> M2;  // dim×dim: running sum of (x - mean_old)(x - mean_new)^T

  WelfordCovStats() : dim(0), n(0) {}

  explicit WelfordCovStats(int d) : dim(d), n(0),
    mean(d, 0.0), M2(static_cast<size_t>(d) * d, 0.0) {}

  void update(const std::vector<double>& x) {
    n++;
    std::vector<double> delta(dim);
    for (int i = 0; i < dim; i++) {
      delta[i] = x[i] - mean[i];
      mean[i] += delta[i] / n;
    }
    // Outer product update: M2 += (x - mean_new) * delta^T
    for (int i = 0; i < dim; i++) {
      double dx_new = x[i] - mean[i];
      for (int j = 0; j <= i; j++) {
        double val = dx_new * delta[j];
        M2[static_cast<size_t>(j) * dim + i] += val;
        if (i != j) {
          M2[static_cast<size_t>(i) * dim + j] += val;  // Symmetric
        }
      }
    }
  }

  // Get covariance with Oracle Approximating Shrinkage (OAS)
  // (Chen, Wiesel, Eldar, Hero 2010)
  //
  // Σ_shrunk = (1 - ρ) * S + ρ * (tr(S)/p) * I
  //
  // OAS automatically adapts shrinkage intensity:
  //   - When n < p (rank-deficient): ρ → 1, shrinks heavily toward scaled identity
  //   - When n >> p: ρ → 0, recovers the unregularized sample covariance
  // Guarantees positive definiteness when ρ > 0.
  //
  // shrinkage_intensity is set as a side-effect for logging.
  mutable double shrinkage_intensity = 0.0;

  std::vector<double> covariance() const {
    std::vector<double> cov(static_cast<size_t>(dim) * dim, 0.0);
    if (n < 2) {
      // Return identity
      for (int i = 0; i < dim; i++) {
        cov[static_cast<size_t>(i) * dim + i] = 1.0;
      }
      shrinkage_intensity = 1.0;
      return cov;
    }

    // Step 1: Compute sample covariance S = M2 / (n - 1)
    double inv_nm1 = 1.0 / (n - 1);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        cov[static_cast<size_t>(j) * dim + i] =
            M2[static_cast<size_t>(j) * dim + i] * inv_nm1;
      }
      // Ensure diagonal is positive
      double& diag = cov[static_cast<size_t>(i) * dim + i];
      diag = std::max(1e-6, diag);
    }

    // Step 2: Compute OAS shrinkage intensity
    // tr(S) and tr(S^2)
    double trS = 0.0;
    double trS2 = 0.0;
    for (int i = 0; i < dim; i++) {
      trS += cov[static_cast<size_t>(i) * dim + i];
    }
    // tr(S^2) = sum of all S_ij^2 (Frobenius norm squared)
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        double s_ij = cov[static_cast<size_t>(j) * dim + i];
        trS2 += s_ij * s_ij;
      }
    }

    // OAS formula (Eq. 23 from Chen et al. 2010):
    // ρ = clamp( ((1 - 2/p) * tr(S²) + tr(S)²) /
    //            ((n + 1 - 2/p) * (tr(S²) - tr(S)²/p)), 0, 1 )
    double p = static_cast<double>(dim);
    double nf = static_cast<double>(n);
    double trS_sq = trS * trS;
    double numer = (1.0 - 2.0 / p) * trS2 + trS_sq;
    double denom = (nf + 1.0 - 2.0 / p) * (trS2 - trS_sq / p);

    double rho;
    if (std::abs(denom) < 1e-12) {
      // Denominator ~0 means S ≈ c*I already, minimal shrinkage needed
      rho = 0.0;
    } else {
      rho = std::max(0.0, std::min(1.0, numer / denom));
    }

    // When n < p, sample covariance is rank-deficient (rank n-1 < p).
    // OAS can underestimate the needed shrinkage for singular matrices.
    // Floor ρ at (1 - n/p) to fill the rank gap: this ensures the
    // p-(n-1) zero eigenvalues get lifted to ρ * tr(S)/p > 0.
    if (n < dim) {
      double rho_floor = 1.0 - nf / p;
      rho = std::max(rho, rho_floor);
    }

    // Even when n > p, moderate n/p ratios (2-5) can produce poorly
    // conditioned matrices. Apply a floor that decays as n/p grows.
    // At n/p=2: floor=0.08, n/p=5: floor=0.05, n/p=10+: floor=0.0
    if (nf < 10.0 * p) {
      double rho_cond_floor = 0.1 * (1.0 - nf / (10.0 * p));
      rho = std::max(rho, rho_cond_floor);
    }
    shrinkage_intensity = rho;

    // Step 3: Apply shrinkage: Σ = (1 - ρ) * S + ρ * (tr(S)/p) * I
    double target_diag = trS / p;  // Scaled identity target
    double one_minus_rho = 1.0 - rho;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        size_t idx = static_cast<size_t>(j) * dim + i;
        if (i == j) {
          cov[idx] = one_minus_rho * cov[idx] + rho * target_diag;
        } else {
          cov[idx] = one_minus_rho * cov[idx];
        }
      }
    }

    return cov;
  }

  void reset() {
    n = 0;
    std::fill(mean.begin(), mean.end(), 0.0);
    std::fill(M2.begin(), M2.end(), 0.0);
  }
};

// =====================================================================
// Welford's online algorithm for mean and variance
// Used for diagonal mass matrix estimation during warmup
// =====================================================================

class WelfordStats {
public:
  int n;
  std::vector<double> mean;
  std::vector<double> M2;  // Sum of squared differences from mean

  WelfordStats(int dim) : n(0), mean(dim, 0.0), M2(dim, 0.0) {}

  void update(const std::vector<double>& x) {
    n++;
    for (size_t i = 0; i < x.size(); i++) {
      double delta = x[i] - mean[i];
      mean[i] += delta / n;
      double delta2 = x[i] - mean[i];
      M2[i] += delta * delta2;
    }
  }

  std::vector<double> variance() const {
    std::vector<double> var(mean.size());
    if (n < 2) {
      // Return unit variance if not enough samples
      std::fill(var.begin(), var.end(), 1.0);
    } else {
      for (size_t i = 0; i < mean.size(); i++) {
        var[i] = M2[i] / (n - 1);
        // Ensure minimum variance to avoid numerical issues
        if (var[i] < 1e-6) var[i] = 1e-6;
      }
    }
    return var;
  }

  // Get inverse mass matrix (= variance, regularized for stability)
  // For HMC: M = diag(1/var), so M^{-1} = diag(var)
  // High variance parameters should move faster in position space
  // Uses Stan-style Bayesian shrinkage toward unit variance to prevent
  // extreme mass matrix entries from small sample sizes
  std::vector<double> inv_mass() const {
    auto var = variance();
    double shrink = (n < 2) ? 0.0 : (double)n / (n + 5.0);
    for (size_t i = 0; i < var.size(); i++) {
      var[i] = shrink * var[i] + 1e-3 * (5.0 / (n + 5.0));
      // Safety clamp for extreme values
      var[i] = std::max(1e-3, std::min(var[i], 1e3));
    }
    return var;
  }

  // Get sqrt of mass matrix for momentum sampling
  // Since M = diag(1/var), sqrt(M) = diag(1/sqrt(var))
  // p ~ N(0, M), so p_i = z_i / sqrt(var_i)
  // Uses the same regularized variance as inv_mass()
  std::vector<double> sqrt_mass() const {
    auto inv_m = inv_mass();
    std::vector<double> sqrt_m(inv_m.size());
    for (size_t i = 0; i < inv_m.size(); i++) {
      sqrt_m[i] = 1.0 / std::sqrt(inv_m[i]);
    }
    return sqrt_m;
  }

  void reset() {
    n = 0;
    std::fill(mean.begin(), mean.end(), 0.0);
    std::fill(M2.begin(), M2.end(), 0.0);
  }
};

struct DualAveraging {
  double mu, log_epsilon_bar, H_bar;
  double gamma, t0, kappa;
  double target_accept;  // Target acceptance rate (dimension-adaptive)
  int m;

  // Default constructor with dimension-adaptive target
  // target_boost: additional boost to target acceptance for challenging models
  //               (e.g., +0.10 for MSGP+temporal combinations)
  DualAveraging(double epsilon_init = 1.0, int n_params = 1, double target_boost = 0.0);
  double update(double alpha);
  double final_epsilon() const;

  // Compute dimension-adaptive target acceptance rate
  // Higher targets (closer to 1) = smaller step sizes = fewer divergences
  // but slower exploration. Stan default is 0.80.
  // target_boost: additional boost for challenging model combinations
  static double compute_target(int n_params, double target_boost = 0.0) {
    // Use higher targets (0.75-0.85) to avoid divergences in
    // challenging models (ICAR, hierarchical negbin, etc.)
    double base_target;
    if (n_params <= 5) base_target = 0.85;
    else if (n_params <= 20) base_target = 0.82;
    else if (n_params <= 50) base_target = 0.80;
    else if (n_params <= 100) base_target = 0.78;
    else base_target = 0.75;

    // Apply boost, cap at 0.99
    return std::min(0.99, base_target + target_boost);
  }
};

struct ChainState {
  std::vector<double> q;
  double log_prob;
  double epsilon;
  DualAveraging da;
  std::mt19937 rng;
  int n_divergent;
};

// Pure C++ result struct (safe for OpenMP parallel regions)
struct HMCResultCpp {
  std::vector<double> samples_flat;  // n_sample × n_params, row-major contiguous
  int n_params_stored = 0;
  std::vector<double> log_prob;
  std::vector<double> accept_prob;
  std::vector<int> n_leapfrog;
  std::vector<int> divergent;
  std::vector<int> treedepth;    // Actual tree depth per iteration (NUTS only)
  double epsilon;
  int n_warmup;
  int n_sample;
  int chain_id;
  int n_max_treedepth = 0;       // Count of iterations hitting max treedepth
  std::string sampler;           // Sampler name (e.g., "NUTS", "HMC", "NUTS->HMC(L=10)")

  // Collapsed mode draws (populated only when collapsed parameterization active)
  int n_gp_collapsed = 0;                         // N_gp if collapsed GP, 0 otherwise
  int n_icar_collapsed = 0;                        // S if collapsed ICAR/BYM2, 0 otherwise
  std::vector<double> gp_w_star_flat;              // w* draws (n_sample x N_gp, row-major)
  std::vector<double> icar_phi_star_flat;          // phi* draws (n_sample x S, row-major)
  std::vector<double> bym2_theta_star_flat;        // theta* draws (n_sample x S, row-major)

  // Row access for flat storage
  double* sample_row(int i) { return &samples_flat[i * n_params_stored]; }
  const double* sample_row(int i) const { return &samples_flat[i * n_params_stored]; }
};

// R-compatible result struct (create outside parallel regions)
struct HMCResult {
  Rcpp::NumericMatrix samples;
  Rcpp::NumericVector log_prob;
  Rcpp::NumericVector accept_prob;
  Rcpp::IntegerVector n_leapfrog;
  Rcpp::IntegerVector treedepth;
  Rcpp::IntegerVector divergent;
  double epsilon;
  int n_warmup;
  int n_sample;
  int chain_id;
  std::string sampler;

  // Collapsed mode draws (populated only when collapsed parameterization active)
  int n_gp_collapsed = 0;
  int n_icar_collapsed = 0;
  Rcpp::NumericMatrix gp_w_star;
  Rcpp::NumericMatrix icar_phi_star;
  Rcpp::NumericMatrix bym2_theta_star;
};

// Convert C++ result to R result (call outside parallel region)
inline HMCResult cpp_to_r_result(const HMCResultCpp& cpp_result, int n_params) {
  HMCResult r_result;
  r_result.samples = Rcpp::NumericMatrix(cpp_result.n_sample, n_params);
  r_result.log_prob = Rcpp::NumericVector(cpp_result.n_sample);
  r_result.accept_prob = Rcpp::NumericVector(cpp_result.n_sample);
  r_result.n_leapfrog = Rcpp::IntegerVector(cpp_result.n_sample);
  r_result.treedepth = Rcpp::IntegerVector(cpp_result.n_sample);
  r_result.divergent = Rcpp::IntegerVector(cpp_result.n_sample);
  r_result.epsilon = cpp_result.epsilon;
  r_result.n_warmup = cpp_result.n_warmup;
  r_result.n_sample = cpp_result.n_sample;
  r_result.chain_id = cpp_result.chain_id;
  r_result.sampler = cpp_result.sampler;

  for (int i = 0; i < cpp_result.n_sample; i++) {
    const double* row = cpp_result.sample_row(i);
    for (int j = 0; j < n_params; j++) {
      r_result.samples(i, j) = row[j];
    }
    r_result.log_prob[i] = cpp_result.log_prob[i];
    r_result.accept_prob[i] = cpp_result.accept_prob[i];
    r_result.n_leapfrog[i] = cpp_result.n_leapfrog[i];
    r_result.treedepth[i] = cpp_result.treedepth[i];
    r_result.divergent[i] = cpp_result.divergent[i];
  }

  // Collapsed GP: copy w* draws
  if (cpp_result.n_gp_collapsed > 0) {
    int n_gp = cpp_result.n_gp_collapsed;
    r_result.n_gp_collapsed = n_gp;
    r_result.gp_w_star = Rcpp::NumericMatrix(cpp_result.n_sample, n_gp);
    for (int i = 0; i < cpp_result.n_sample; i++) {
      for (int j = 0; j < n_gp; j++) {
        r_result.gp_w_star(i, j) = cpp_result.gp_w_star_flat[i * n_gp + j];
      }
    }
  }

  // Collapsed ICAR/BYM2: copy phi* and theta* draws
  if (cpp_result.n_icar_collapsed > 0) {
    int S = cpp_result.n_icar_collapsed;
    r_result.n_icar_collapsed = S;
    r_result.icar_phi_star = Rcpp::NumericMatrix(cpp_result.n_sample, S);
    for (int i = 0; i < cpp_result.n_sample; i++) {
      for (int j = 0; j < S; j++) {
        r_result.icar_phi_star(i, j) = cpp_result.icar_phi_star_flat[i * S + j];
      }
    }
    // BYM2: also copy theta*
    if (!cpp_result.bym2_theta_star_flat.empty()) {
      r_result.bym2_theta_star = Rcpp::NumericMatrix(cpp_result.n_sample, S);
      for (int i = 0; i < cpp_result.n_sample; i++) {
        for (int j = 0; j < S; j++) {
          r_result.bym2_theta_star(i, j) = cpp_result.bym2_theta_star_flat[i * S + j];
        }
      }
    }
  }

  return r_result;
}

// NUTS helper function declarations
double nuts_log_sum_exp(double a, double b);
double nuts_compute_hamiltonian(double log_prob, const std::vector<double>& p,
                                const std::vector<double>& inv_mass, int n);
bool nuts_check_uturn(const std::vector<double>& q_minus, const std::vector<double>& q_plus,
                      const std::vector<double>& p_minus, const std::vector<double>& p_plus,
                      const std::vector<double>& inv_mass, int n);
LeapfrogResultWithGrad leapfrog_step_with_grad(
    const std::vector<double>& q, const std::vector<double>& p,
    const std::vector<double>& grad,
    double epsilon, const std::vector<double>& inv_mass,
    bool use_mass, const ModelData& data, const ParamLayout& layout);
// Optimized NUTS: zero-allocation in-place leapfrog + buffer pool tree building
// Pointer-based Hamiltonian (no vector overhead)
double nuts_compute_hamiltonian_fast(double log_prob, const double* p,
                                     const DenseMassMatrix& mass, int n);
// Pointer-based U-turn check
bool nuts_check_uturn_fast(const double* q_minus, const double* q_plus,
                           const double* p_minus, const double* p_plus,
                           const DenseMassMatrix& mass, double* scratch, int n);
// In-place leapfrog step operating on workspace slot
LeapfrogInPlaceResult leapfrog_step_inplace(
    NUTSWorkspace& ws, int slot, double epsilon,
    const DenseMassMatrix& mass,
    const ModelData& data, const ParamLayout& layout);
// Zero-allocation recursive tree builder
TreeStats build_tree_fast(
    NUTSWorkspace& ws, int input_slot, int direction, int depth,
    double epsilon, const DenseMassMatrix& mass,
    double H0, double delta_max,
    const ModelData& data, const ParamLayout& layout,
    std::mt19937& rng);

// =====================================================================
// Sampler functions
// =====================================================================

// Unified leapfrog step: identity mass when inv_mass is nullptr
LeapfrogResult leapfrog_step(
    const std::vector<double>& q,
    const std::vector<double>& p,
    double epsilon,
    const ModelData& data,
    const ParamLayout& layout,
    const double* inv_mass = nullptr
);

// Find reasonable initial step size
double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng
);

// Mass-aware version: uses diagonal mass matrix for leapfrog and kinetic energy
double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const std::vector<double>& inv_mass
);

// Dense-mass-aware version: uses full DenseMassMatrix for momentum, leapfrog, and kinetic energy
double find_reasonable_epsilon_dense(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const DenseMassMatrix& mass
);

// Run single HMC chain (C++ version - safe for parallel)
// riemannian: -1=auto (retry divergences with SoftAbs for BYM2/ICAR),
//              1=force on, 0=force off
HMCResultCpp run_hmc_chain_cpp(
    const std::vector<double>& q_init,
    const ModelData& data,
    const ParamLayout& layout,
    int n_iter,
    int n_warmup,
    int L,
    int chain_id,
    unsigned int seed,
    bool verbose,
    int max_treedepth = 10,
    MassMatrixType metric_type = MassMatrixType::DIAG,
    double adapt_delta = -1.0,
    int riemannian = -1
);

// Run single HMC chain (R wrapper)
HMCResult run_hmc_chain(
    const std::vector<double>& q_init,
    const ModelData& data,
    const ParamLayout& layout,
    int n_iter,
    int n_warmup,
    int L,
    int chain_id,
    unsigned int seed,
    bool verbose,
    int max_treedepth = 10,
    MassMatrixType metric_type = MassMatrixType::DIAG,
    double adapt_delta = -1.0,
    int riemannian = -1
);

// Run multiple chains in parallel (across-chain parallelization)
std::vector<HMCResult> run_hmc_parallel_chains(
    const std::vector<double>& q_init,
    const ModelData& data,
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    bool verbose,
    int max_treedepth = 10,
    MassMatrixType metric_type = MassMatrixType::DIAG,
    double adapt_delta = -1.0,
    int riemannian = -1
);

// =====================================================================
// SoftAbs per-trajectory metric (Riemannian-like divergence retry)
// =====================================================================

// Compute full Hessian via finite differences of the H-mode gradient.
// H[i,j] = (grad_j(q + h*e_i) - grad_j(q)) / h
// Cost: (p+1) gradient evaluations.
void compute_hessian_finite_diff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& hessian,
    double h = 1e-5
);

// Compute SoftAbs metric from negative Hessian.
// G = Q diag(f(λ_i)) Q^T where f(λ) = λ * coth(α * λ)
// Returns G^{-1} and its Cholesky L. Returns false on failure.
bool compute_softabs_metric(
    const std::vector<double>& neg_hessian,
    int p,
    double alpha,
    std::vector<double>& G_inv,
    std::vector<double>& L_G_inv
);

// =====================================================================
// Mass matrix configuration and helpers (used by hmc_chain.cpp)
// =====================================================================

inline constexpr int DENSE_MAX_PARAMS = 200;

struct MassMatrixConfig {
  MassMatrixType effective_metric;
  bool auto_selected_diag;
  std::vector<std::pair<int,int>> block_specs;
};

// Select mass matrix type (AUTO resolution, block detection, DENSE override)
// and initialize the DenseMassMatrix object.
MassMatrixConfig select_and_init_mass_matrix(
    DenseMassMatrix& mass,
    const ModelData& data,
    const ParamLayout& layout,
    int n_params,
    MassMatrixType metric_type,
    bool verbose
);

// Warm-start mass matrix diagonal from model structure
void warm_start_mass_matrix(
    DenseMassMatrix& mass,
    const ModelData& data,
    const ParamLayout& layout,
    int n_params,
    bool verbose
);

// Runtime gradient verification (compare active gradient vs numerical)
bool verify_gradient_runtime(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    double tol = 1e-4
);

} // namespace tulpa_hmc

#endif // TULPA_HMC_SAMPLER_H
