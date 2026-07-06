// hmc_nuts_optimized.cpp
// Zero-allocation NUTS: in-place leapfrog, build_tree_fast,
// pre-allocated workspace integration.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "hmc_sampler.h"

namespace tulpa_hmc {

// =====================================================================
// Optimized NUTS: zero-allocation infrastructure
// =====================================================================

// Pointer-based Hamiltonian (avoids std::vector overhead)
double nuts_compute_hamiltonian_fast(double log_prob, const double* p,
                                     const DenseMassMatrix& mass, int n) {
  return -log_prob + mass.kinetic_energy(p);
}

// Pointer-based U-turn check
// scratch: temporary buffer of size n (for dense matvec result)
bool nuts_check_uturn_fast(const double* q_minus, const double* q_plus,
                           const double* p_minus, const double* p_plus,
                           const DenseMassMatrix& mass, double* scratch, int n) {
  if (mass.type == MassMatrixType::BLOCK_DIAG && mass.adapted) {
    // Block-diagonal: use inv_mass_times_p which handles blocks correctly
    mass.inv_mass_times_p(p_plus, scratch);
    double dot_fwd = 0.0;
    for (int i = 0; i < n; i++) {
      dot_fwd += (q_plus[i] - q_minus[i]) * scratch[i];
    }
    mass.inv_mass_times_p(p_minus, scratch);
    double dot_bwd = 0.0;
    for (int i = 0; i < n; i++) {
      dot_bwd += (q_plus[i] - q_minus[i]) * scratch[i];
    }
    return (dot_fwd < 0.0) || (dot_bwd < 0.0);
  } else if (mass.type == MassMatrixType::DIAG || !mass.adapted) {
    // Diagonal path (fast, unrolled)
    double dot_fwd = 0.0, dot_bwd = 0.0;
    const double* inv_mass = mass.inv_mass_diag.data();
    int i = 0;
    for (; i + 3 < n; i += 4) {
      double dq0 = q_plus[i]   - q_minus[i];
      double dq1 = q_plus[i+1] - q_minus[i+1];
      double dq2 = q_plus[i+2] - q_minus[i+2];
      double dq3 = q_plus[i+3] - q_minus[i+3];
      dot_fwd += dq0 * (inv_mass[i]   * p_plus[i])
               + dq1 * (inv_mass[i+1] * p_plus[i+1])
               + dq2 * (inv_mass[i+2] * p_plus[i+2])
               + dq3 * (inv_mass[i+3] * p_plus[i+3]);
      dot_bwd += dq0 * (inv_mass[i]   * p_minus[i])
               + dq1 * (inv_mass[i+1] * p_minus[i+1])
               + dq2 * (inv_mass[i+2] * p_minus[i+2])
               + dq3 * (inv_mass[i+3] * p_minus[i+3]);
    }
    for (; i < n; i++) {
      double dq = q_plus[i] - q_minus[i];
      dot_fwd += dq * (inv_mass[i] * p_plus[i]);
      dot_bwd += dq * (inv_mass[i] * p_minus[i]);
    }
    return (dot_fwd < 0.0) || (dot_bwd < 0.0);
  } else {
    // Dense path: compute (q+ - q-) . (C * p+) and (q+ - q-) . (C * p-)
    // Use scratch for C * p
    mass.inv_mass_times_p(p_plus, scratch);
    double dot_fwd = 0.0;
    for (int i = 0; i < n; i++) {
      dot_fwd += (q_plus[i] - q_minus[i]) * scratch[i];
    }
    mass.inv_mass_times_p(p_minus, scratch);
    double dot_bwd = 0.0;
    for (int i = 0; i < n; i++) {
      dot_bwd += (q_plus[i] - q_minus[i]) * scratch[i];
    }
    return (dot_fwd < 0.0) || (dot_bwd < 0.0);
  }
}

// Drift: q += coeff * C * p, where C = M^{-1} carries the full mass structure
// (identity / block-diagonal / diagonal / dense, plus precision and Kronecker
// block corrections). Factored out of the leapfrog step so every scheme's
// drift sub-steps reuse the same fused kernels. coeff is the scheme's drift
// coefficient times the step size; for the default leapfrog it is exactly the
// step size, so this reproduces the historical drift bit-for-bit.
static inline void apply_drift(
    double coeff, double* q, const double* p,
    const DenseMassMatrix& mass, int n) {
  if (!mass.adapted) {
    tulpa_linalg::axpy(coeff, p, q, n);
  } else if (mass.type == MassMatrixType::BLOCK_DIAG) {
    tulpa_linalg::axpy_weighted(coeff, mass.inv_mass_diag.data(), p, q, n);
    for (const auto& blk : mass.blocks) {
      if (blk.adapted) {
        double tmp[4];
        blk.matvec(p, tmp);
        for (int i = 0; i < blk.size; i++) {
          q[blk.start + i] += coeff * (tmp[i] - mass.inv_mass_diag[blk.start + i] * p[blk.start + i]);
        }
      }
    }
  } else if (mass.type == MassMatrixType::DIAG) {
    tulpa_linalg::axpy_weighted(coeff, mass.inv_mass_diag.data(), p, q, n);
  } else {
    if (n >= 16) {
      Eigen::Map<const Eigen::MatrixXd> Am(mass.inv_mass_dense.data(), n, n);
      Eigen::Map<const Eigen::VectorXd> pv(p, n);
      Eigen::Map<Eigen::VectorXd> qv(q, n);
      qv.noalias() += coeff * (Am.selfadjointView<Eigen::Lower>() * pv);
    } else {
      tulpa_linalg::axpy_matvec(coeff, mass.inv_mass_dense.data(), p, q, n);
    }
  }
  if (mass.precision_block.active) {
    const auto& pb = mass.precision_block;
    std::vector<double> tmp(pb.size);
    pb.matvec(p, tmp.data());
    for (int i = 0; i < pb.size; i++) {
      q[pb.start + i] -= coeff * mass.inv_mass_diag[pb.start + i] * p[pb.start + i];
      q[pb.start + i] += coeff * tmp[i];
    }
  }
  if (mass.kronecker_block.active) {
    const auto& kb = mass.kronecker_block;
    int ST = kb.S * kb.T;
    std::vector<double> tmp(ST);
    kb.matvec(p, tmp.data());
    for (int i = 0; i < ST; i++) {
      q[kb.start + i] -= coeff * mass.inv_mass_diag[kb.start + i] * p[kb.start + i];
      q[kb.start + i] += coeff * tmp[i];
    }
  }
}

// In-place leapfrog step operating on a workspace slot.
// Mutates q, p, grad in the slot directly -- zero heap allocation.
//
// Walks the active SIMP scheme's op sequence. The entry gradient is valid at
// the current q (first-same-as-last from the previous step), so a Kick uses it
// directly and only recomputes after an intervening Drift. Each recompute is
// the fused gradient + log_prob pass. For the default leapfrog scheme this is
// exactly half-kick, drift, recompute, half-kick -- the historical step,
// coefficient for coefficient. Higher-order schemes add interior drift/kick
// pairs (one extra gradient each) with no other change.
LeapfrogInPlaceResult leapfrog_step_inplace(
    NUTSWorkspace& ws, int slot, double epsilon,
    const DenseMassMatrix& mass,
    const ModelData& data, const ParamLayout& layout) {

  double* q = ws.q_at(slot);
  double* p = ws.p_at(slot);
  double* grad = ws.grad_at(slot);
  int n = ws.n;

  LeapfrogInPlaceResult result;
  result.divergent = false;

  if (ws.mts) {
    // Multiple-time-stepping (RESPA / Verlet-I) leaf. Force splits as
    // F = F_slow + F_fast: F_fast is the stiff, cheap prior gradient (prior_
    // gradient_fn, m inner substeps), F_slow the smooth, expensive likelihood
    // gradient (grad_full - grad_prior, one full gradient per leaf). The step
    // is symmetric, time reversible, and symplectic, so the NUTS tree / U-turn
    // machinery is unchanged -- only the leaf's internal advance differs.
    const int m = ws.mts_m > 0 ? ws.mts_m : 1;
    const double half = 0.5 * epsilon;
    const double inner = epsilon / static_cast<double>(m);
    const double inner_half = 0.5 * inner;
    double* gp = ws.mts_gp.data();

    // Prior (fast) gradient at the entry position. The entry `grad` is the full
    // gradient here (first-same-as-last from the previous leaf).
    std::memcpy(ws.params_buf.data(), q, n * sizeof(double));
    ws.prior_gradient_fn(ws.params_buf, data, layout, ws.grad_buf, nullptr);
    std::memcpy(gp, ws.grad_buf.data(), n * sizeof(double));

    // Outer half kick with the slow force F_slow(q0) = grad_full - grad_prior.
    for (int i = 0; i < n; i++) p[i] += half * (grad[i] - gp[i]);

    // Inner leapfrog chain driven by the fast (prior) force.
    for (int k = 0; k < m; k++) {
      for (int i = 0; i < n; i++) p[i] += inner_half * gp[i];
      apply_drift(inner, q, p, mass, n);
      std::memcpy(ws.params_buf.data(), q, n * sizeof(double));
      ws.prior_gradient_fn(ws.params_buf, data, layout, ws.grad_buf, nullptr);
      std::memcpy(gp, ws.grad_buf.data(), n * sizeof(double));
      for (int i = 0; i < n; i++) p[i] += inner_half * gp[i];
    }

    // Full gradient + log_prob at the endpoint (the one expensive evaluation per
    // leaf); `grad` becomes grad_full(q_end) for the next leaf's FSAL. gp holds
    // grad_prior(q_end) from the last inner substep.
    std::memcpy(ws.params_buf.data(), q, n * sizeof(double));
    ws.gradient_fn(ws.params_buf, data, layout, ws.grad_buf, &ws.logp_at(slot));
    std::memcpy(grad, ws.grad_buf.data(), n * sizeof(double));

    // Outer half kick with the slow force F_slow(q_end).
    for (int i = 0; i < n; i++) p[i] += half * (grad[i] - gp[i]);

    result.log_prob = ws.logp_at(slot);
  } else {
    const simp::Scheme& scheme = ws.scheme;
    bool grad_fresh = true;  // entry gradient is valid at the current q

    for (const auto& op : scheme.ops) {
      double c = op.second * epsilon;
      if (op.first == simp::Op::Kick) {
        if (!grad_fresh) {
          std::memcpy(ws.params_buf.data(), q, n * sizeof(double));
          ws.gradient_fn(ws.params_buf, data, layout, ws.grad_buf, &ws.logp_at(slot));
          std::memcpy(grad, ws.grad_buf.data(), n * sizeof(double));
          grad_fresh = true;
        }
        tulpa_linalg::axpy(c, grad, p, n);
      } else {
        apply_drift(c, q, p, mass, n);
        grad_fresh = false;
      }
    }

    // Ensure the returned gradient + log_prob are at the trajectory endpoint.
    // Our schemes end in a Kick, so this is already satisfied; the guard covers
    // any drift-terminated scheme without disturbing the common case.
    if (!grad_fresh) {
      std::memcpy(ws.params_buf.data(), q, n * sizeof(double));
      ws.gradient_fn(ws.params_buf, data, layout, ws.grad_buf, &ws.logp_at(slot));
      std::memcpy(grad, ws.grad_buf.data(), n * sizeof(double));
    }
    result.log_prob = ws.logp_at(slot);
  }

  // Divergence check (skip param scan if log_prob already non-finite)
  if (!std::isfinite(result.log_prob)) {
    result.divergent = true;
  } else {
    for (int i = 0; i < n; i++) {
      if (std::abs(q[i]) > 1e10 || !std::isfinite(q[i])) {
        result.divergent = true;
        break;
      }
    }
  }

  return result;
}

// Zero-allocation recursive tree builder
// Uses workspace slot indices instead of vector copies
TreeStats build_tree_fast(
    NUTSWorkspace& ws, int input_slot, int direction, int depth,
    double epsilon, const DenseMassMatrix& mass,
    double H0, double delta_max,
    const ModelData& data, const ParamLayout& layout,
    std::mt19937& rng) {

  int n = ws.n;
  TreeStats stats;

  if (depth == 0) {
    stats.init_vectors(n);  // Pre-allocate U-turn vectors (avoids per-leaf heap allocation)
    // Base case: single leapfrog step in-place on input_slot
    LeapfrogInPlaceResult lf = leapfrog_step_inplace(
      ws, input_slot, direction * epsilon, mass, data, layout
    );

    double H_new = nuts_compute_hamiltonian_fast(
      lf.log_prob, ws.p_at(input_slot), mass, n
    );
    double delta_H = H_new - H0;

    // Both endpoints are the same slot (single node)
    stats.left_slot = input_slot;
    stats.right_slot = input_slot;
    stats.proposal_slot = input_slot;
    stats.log_prob_proposal = lf.log_prob;

    // Multinomial weight: log(weight) = H0 - H_new (relative, Stan-style)
    stats.sum_log_weight = H0 - H_new;

    // Divergence check
    stats.divergent = lf.divergent || (delta_H > delta_max);
    stats.stop = stats.divergent;
    stats.n_valid = stats.divergent ? 0 : 1;

    // Acceptance statistic
    double accept_stat = std::min(1.0, std::exp(-delta_H));
    if (!std::isfinite(accept_stat)) accept_stat = 0.0;
    stats.sum_accept_prob = accept_stat;
    stats.n_leapfrog = 1;

    // Generalized U-turn: track rho, p_sharp, p at this leaf
    const double* p_ptr = ws.p_at(input_slot);

    std::memcpy(stats.rho.data(), p_ptr, n * sizeof(double));
    std::memcpy(stats.p_beg.data(), p_ptr, n * sizeof(double));
    std::memcpy(stats.p_end.data(), p_ptr, n * sizeof(double));

    // p_sharp = M^{-1} * p  ? use full mass matrix for U-turn criterion.
    // Dense mass captures correlation structure; using diagonal p_sharp would
    // make NUTS unable to detect turns in correlated directions, causing
    // trees to grow to max depth on correlated posteriors (slopes, BYM2, HSGP).
    mass.inv_mass_times_p(p_ptr, stats.p_sharp_beg.data());
    std::memcpy(stats.p_sharp_end.data(), stats.p_sharp_beg.data(), n * sizeof(double));

    return stats;
  }

  // Recursive case: build inner subtree
  TreeStats inner = build_tree_fast(
    ws, input_slot, direction, depth - 1,
    epsilon, mass, H0, delta_max, data, layout, rng
  );

  stats = std::move(inner);

  if (stats.stop) return stats;

  // Copy the appropriate endpoint to a fresh slot for outer start
  int start_slot = ws.alloc_slot();
  if (start_slot < 0) {
    stats.stop = true;
    return stats;
  }
  if (direction == 1) {
    ws.copy_node(start_slot, stats.right_slot);
  } else {
    ws.copy_node(start_slot, stats.left_slot);
  }

  // Build outer subtree from the copy
  TreeStats outer = build_tree_fast(
    ws, start_slot, direction, depth - 1,
    epsilon, mass, H0, delta_max, data, layout, rng
  );

  // Combine results
  stats.n_leapfrog += outer.n_leapfrog;
  stats.sum_accept_prob += outer.sum_accept_prob;
  stats.divergent = stats.divergent || outer.divergent;

  // Multinomial sampling
  double new_sum_log_weight = nuts_log_sum_exp(stats.sum_log_weight, outer.sum_log_weight);
  double accept_prob_outer = std::exp(outer.sum_log_weight - new_sum_log_weight);
  if (!std::isfinite(accept_prob_outer)) accept_prob_outer = 0.0;

  std::uniform_real_distribution<double> unif(0.0, 1.0);
  if (unif(rng) < accept_prob_outer) {
    stats.proposal_slot = outer.proposal_slot;
    stats.log_prob_proposal = outer.log_prob_proposal;
  }

  stats.sum_log_weight = new_sum_log_weight;
  stats.n_valid = stats.n_valid + outer.n_valid;

  // === SAVE BOUNDARY VALUES AS COPIES BEFORE MOVES ===
  // "init" = inner (built first), "final" = outer (extends from init)
  // These copies are needed because moves below invalidate the originals
  // Uses pre-allocated depth-indexed merge buffers (no per-merge heap allocation)
  double* p_init_end = ws.merge_buf(depth, NUTSWorkspace::MERGE_P_INIT_END);
  double* p_sharp_init_end = ws.merge_buf(depth, NUTSWorkspace::MERGE_PSHARP_INIT_END);
  double* rho_init = ws.merge_buf(depth, NUTSWorkspace::MERGE_RHO_INIT);
  double* rho_check = ws.merge_buf(depth, NUTSWorkspace::MERGE_RHO_CHECK);

  const double* src_p = (direction == 1) ? stats.p_end.data() : stats.p_beg.data();
  std::memcpy(p_init_end, src_p, n * sizeof(double));
  const double* src_ps = (direction == 1) ? stats.p_sharp_end.data() : stats.p_sharp_beg.data();
  std::memcpy(p_sharp_init_end, src_ps, n * sizeof(double));
  std::memcpy(rho_init, stats.rho.data(), n * sizeof(double));

  // Pointers to final's boundary (safe: these outer members are NOT moved)
  const double* p_final_beg_ptr = (direction == 1) ? outer.p_beg.data() : outer.p_end.data();
  const double* p_sharp_final_beg_ptr = (direction == 1) ? outer.p_sharp_beg.data() : outer.p_sharp_end.data();

  // === UPDATE TREE ENDPOINTS (moves invalidate init's boundary refs) ===
  if (direction == 1) {
    stats.right_slot = outer.right_slot;
    stats.p_sharp_end = std::move(outer.p_sharp_end);
    stats.p_end = std::move(outer.p_end);
  } else {
    stats.left_slot = outer.left_slot;
    stats.p_sharp_beg = std::move(outer.p_sharp_beg);
    stats.p_beg = std::move(outer.p_beg);
  }

  // Combine rho = rho_init + rho_final
  for (int i = 0; i < n; i++) {
    stats.rho[i] = rho_init[i] + outer.rho[i];
  }

  // === GENERALIZED U-TURN CRITERION (Stan-style, 3 juncture checks) ===
  // Check 1: Full merged trajectory ? merged endpoints vs merged rho
  bool persist = compute_criterion(stats.p_sharp_beg.data(), stats.p_sharp_end.data(),
                                   stats.rho.data(), n);

  // After update, far endpoints depend on direction:
  // direction == 1: init's far = stats.beg (left, unchanged), final's far = stats.end (right, updated)
  // direction == -1: init's far = stats.end (right, unchanged), final's far = stats.beg (left, updated)
  const double* init_far_psharp = (direction == 1) ? stats.p_sharp_beg.data() : stats.p_sharp_end.data();
  const double* final_far_psharp = (direction == 1) ? stats.p_sharp_end.data() : stats.p_sharp_beg.data();

  // Check 2: Init subtree + seam from final (rho = rho_init + p_final_beg)
  // Fused: rho construction + dot product in single O(n) pass
  persist &= compute_criterion_fused(init_far_psharp, p_sharp_final_beg_ptr,
                                     rho_init, p_final_beg_ptr, rho_check, n);

  // Check 3: Seam from init + final subtree (rho = rho_final + p_init_end)
  persist &= compute_criterion_fused(p_sharp_init_end, final_far_psharp,
                                     outer.rho.data(), p_init_end, rho_check, n);

  stats.stop = outer.stop || !persist;

  return stats;
}

}  // namespace tulpa_hmc
