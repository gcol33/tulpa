// nested_laplace_joint_batch.h
// Batched (multi-response) fused inner-solve primitives for the joint
// nested-Laplace cell-coupling path (gcol33/tulpa#66).
//
// The B species share one design (X / spatial_idx / cell_obs_map) and one
// sparsity pattern; only the response y / y_pos and per-species dispersion
// differ. Their latent blocks are independent (block-diagonal), so each species
// is its OWN single-species system: B latent vectors x_s, B Hessians H_s over
// the SAME single-species shape, B gradients. The bandwidth win is the FUSED
// scatter -- one pass over cells, the multi-response evaluate_cell loads each
// design row once and the per-species scatter writes into H_s / grad_s -- and
// the per-species linear algebra (solve / line search / convergence) keeps
// every species bit-identical to its independent fit.
//
// This header owns only the fused SCATTER + per-arm eta/response layout. The
// batched inner Newton and the batched outer-grid loop are in
// nested_laplace_joint_batch.cpp (they reuse run_multi_block's block / prior /
// center machinery per species).

#ifndef TULPA_NESTED_LAPLACE_JOINT_BATCH_H
#define TULPA_NESTED_LAPLACE_JOINT_BATCH_H

#include "latent_block.h"
#include "nested_laplace_joint_core.h"
#include "nested_laplace_joint_multi.h"   // scatter_one_arm_row_dense, build_arm_row_chain, ...
#include "sparse_hessian.h"
#include "tulpa/cell_coupling.h"
#include <Rcpp.h>
#include <functional>
#include <memory>
#include <vector>

namespace tulpa {

// Per-arm species-major eta / response storage for a batch of B species. Each
// arm k holds one contiguous buffer of length N_k * B; species s occupies
// [s * N_k, (s + 1) * N_k). This is exactly the layout CellEtas / CellResponse
// read via arm_eta_stride / arm_y_stride = N_k.
struct BatchArmBuffers {
    int B = 1;
    // etas[k]: length N_k * B, species-major. Rewritten each Newton iteration.
    std::vector<std::vector<double>> etas;
    // y[k]: length N_k * B, species-major (coupled data arms). Built once.
    std::vector<std::vector<double>> y;
    // n_trials[k]: length N_k * B (or empty). Built once.
    std::vector<std::vector<int>> n_trials;
    // phi[k * B + s]: per-species dispersion for arm k. Built once / per grid.
    std::vector<double> phi;
    std::vector<int> N;          // per-arm row count N_k
    int n_arms = 0;

    void allocate(const std::vector<JointArm>& arms, int B_) {
        B = B_;
        n_arms = static_cast<int>(arms.size());
        etas.assign(n_arms, {});
        y.assign(n_arms, {});
        n_trials.assign(n_arms, {});
        N.assign(n_arms, 0);
        phi.assign((std::size_t) n_arms * B, 0.0);
        for (int k = 0; k < n_arms; k++) {
            N[k] = arms[k].N;
            etas[k].assign((std::size_t) N[k] * B, 0.0);
        }
    }
};

// Per-species scatter policy for the fused batched cell-coupling pass.
//
// The cell scaffolding (row counts, view setup, the multi-response
// evaluate_cell pass, the species-invariant design chains) is owned by
// scatter_cell_coupling_batch_impl below and identical for every Hessian
// container. The two policies differ only in how a species' per-row / per-pair
// derivatives reach its Hessian:
//
//   * DenseScatterPolicy writes into an n_x x n_x DenseMat through the
//     index-direct dense helpers (no lookup table needed).
//
//   * SparseScatterPolicy writes into a SparseHessianBuilder. Because all B
//     species share one structural pattern, the (row, col) -> flat values[]
//     slot resolution is species-invariant: it is computed ONCE per cell from
//     species 0's pattern and reused for every species, turning each
//     per-species write from a std::map lookup into a flat values[slot] += .
//     This per-cell slot sharing across species is the #69 scatter
//     amortization.
//
// The within-arm row Hessian is the lower triangle of H_row * outer(chain,
// chain) for the row's design chain (the same chain the cross-arm scatter
// walks), so a single chain representation drives the within-arm gradient,
// within-arm Hessian, and cross-arm Hessian for the sparse policy.

// Lower-triangle flat slots of outer(chain, chain): one entry per (a, b) with
// b <= a, resolved against the shared sparse pattern. Cleared and rebuilt per
// cell row; reused across all B species of that cell. The two chain weights are
// stored separately and pre-ordered (w_first applied before w_second) so the
// per-species write reproduces scatter_one_arm_row_sparse's exact multiply
// association `(H_row * w_first) * w_second`.
struct WithinRowSlots {
    std::vector<int>    slot;     // flat values[] index, or -1 if absent
    std::vector<double> w_first;  // first weight in the oracle's association
    std::vector<double> w_second; // second weight in the oracle's association
};

// Flat slots of the full outer(chain_k, chain_l) product for one cross-arm row
// pair. chain_k's weight is stored separately from chain_l's so the write
// reproduces scatter_cross_chain_sparse's `(Hkl * w_k) * w_l`. The diagonal
// (idx_k == idx_l) is marked so its symmetric contribution is written as two
// separate accumulations into the slot, matching the oracle's two H.add calls.
struct CrossPairSlots {
    std::vector<int>    slot;
    std::vector<double> w_k;
    std::vector<double> w_l;
    std::vector<char>   is_diag;
};

struct DenseScatterPolicy {
    using HContainer = DenseMat;

    void chains_ptr(
        const std::vector<std::vector<std::vector<ArmRowChainEntry>>>*) {}

    // Dense path needs no per-cell slot cache; the helpers index H directly.
    template <typename ChainsT>
    void prepare_cell(const std::vector<int>& /*coupled_arms*/,
                      const std::vector<int>& /*arm_row_count*/,
                      const std::vector<const int*>& /*arm_rows_ptr*/,
                      const std::vector<ParsedArm>& /*parsed*/,
                      const std::vector<LatentBlock>& /*blocks*/,
                      const std::vector<std::vector<double>>& /*d_eff_per_arm*/,
                      const ChainsT& /*chains_per_arm*/,
                      std::vector<DenseMat>& /*H_per_sp*/) {}

    void scatter_within_row(int row, double g_row, double H_row,
                            const ParsedArm& pa, int k_arm,
                            const std::vector<LatentBlock>& blocks,
                            const std::vector<double>& d_eff,
                            int /*kk*/, int /*j*/,
                            DenseVec& grad, DenseMat& H,
                            std::vector<int>& active_idx,
                            std::vector<double>& active_d) {
        scatter_one_arm_row_dense(row, g_row, H_row, pa, k_arm, blocks, d_eff,
                                  grad, H, active_idx, active_d);
    }

    void scatter_cross_pair(double Hkl,
                            const std::vector<ArmRowChainEntry>& chain_k,
                            const std::vector<ArmRowChainEntry>& chain_l,
                            int /*kk*/, int /*ll*/, int /*j*/, int /*m*/,
                            DenseMat& H) {
        scatter_cross_chain_dense(Hkl, chain_k, chain_l, H);
    }
};

struct SparseScatterPolicy {
    using HContainer = SparseHessianBuilder;

    // Per-cell slot caches, keyed by within-arm row and cross-arm pair. The
    // outer dims (coupled arm / arm pair) are sized once; the per-row inner
    // storage grows monotonically.
    std::vector<std::vector<WithinRowSlots>> within;            // [kk][j]
    std::vector<std::vector<std::vector<CrossPairSlots>>> cross; // [kk*n+ll][j][m]
    int n_coupled_ = 0;

    // Resolve the species-invariant flat slots for the current cell from the
    // shared sparse pattern (species 0's builder). within[kk][j] holds the LT
    // self-product of chain j; cross[kk*n+ll][j][m] holds the full
    // chain_k x chain_l product for off-diagonal-eligible pairs.
    template <typename ChainsT>
    void prepare_cell(const std::vector<int>& coupled_arms,
                      const std::vector<int>& arm_row_count,
                      const std::vector<const int*>& /*arm_rows_ptr*/,
                      const std::vector<ParsedArm>& /*parsed*/,
                      const std::vector<LatentBlock>& /*blocks*/,
                      const std::vector<std::vector<double>>& /*d_eff_per_arm*/,
                      const ChainsT& chains_per_arm,
                      std::vector<SparseHessianBuilder>& H_per_sp) {
        const int n_coupled = (int) coupled_arms.size();
        n_coupled_ = n_coupled;
        const SparseHessianBuilder& Hp = H_per_sp[0];

        if ((int) within.size() < n_coupled) within.resize(n_coupled);
        if ((int) cross.size() < n_coupled * n_coupled)
            cross.resize((std::size_t) n_coupled * n_coupled);

        for (int kk = 0; kk < n_coupled; kk++) {
            const int rc = arm_row_count[kk];
            auto& wkk = within[kk];
            if ((int) wkk.size() < rc) wkk.resize(rc);
            for (int j = 0; j < rc; j++) {
                const std::vector<ArmRowChainEntry>& ch = chains_per_arm[kk][j];
                const int L = (int) ch.size();
                WithinRowSlots& w = wkk[j];
                w.slot.clear();
                w.w_first.clear();
                w.w_second.clear();
                for (int a = 0; a < L; a++) {
                    for (int b = 0; b <= a; b++) {
                        w.slot.push_back(Hp.lookup(ch[a].idx, ch[b].idx));
                        // Same block group: oracle's outer loop owns ch[a] (the
                        // later chain entry) -> (H_row * w_a) * w_b. Cross group:
                        // the earlier block (ch[b]) owns the outer loop ->
                        // (H_row * w_b) * w_a.
                        if (ch[a].grp == ch[b].grp) {
                            w.w_first.push_back(ch[a].w);
                            w.w_second.push_back(ch[b].w);
                        } else {
                            w.w_first.push_back(ch[b].w);
                            w.w_second.push_back(ch[a].w);
                        }
                    }
                }
            }
        }

        for (int kk = 0; kk < n_coupled; kk++) {
            const int rc_k = arm_row_count[kk];
            for (int ll = kk; ll < n_coupled; ll++) {
                const int rc_l = arm_row_count[ll];
                auto& cpair = cross[(std::size_t) kk * n_coupled + ll];
                if ((int) cpair.size() < rc_k) cpair.resize(rc_k);
                for (int j = 0; j < rc_k; j++) {
                    const std::vector<ArmRowChainEntry>& chain_k =
                        chains_per_arm[kk][j];
                    auto& cj = cpair[j];
                    if ((int) cj.size() < rc_l) cj.resize(rc_l);
                    const int m_start = (kk == ll) ? (j + 1) : 0;
                    for (int m = m_start; m < rc_l; m++) {
                        const std::vector<ArmRowChainEntry>& chain_l =
                            chains_per_arm[ll][m];
                        CrossPairSlots& cs = cj[m];
                        cs.slot.clear();
                        cs.w_k.clear();
                        cs.w_l.clear();
                        cs.is_diag.clear();
                        for (const auto& e_k : chain_k) {
                            for (const auto& e_l : chain_l) {
                                cs.slot.push_back(Hp.lookup(e_k.idx, e_l.idx));
                                cs.w_k.push_back(e_k.w);
                                cs.w_l.push_back(e_l.w);
                                cs.is_diag.push_back(e_k.idx == e_l.idx ? 1 : 0);
                            }
                        }
                    }
                }
            }
        }
    }

    // Within-arm gradient + Hessian for one species' row. Gradient writes use
    // the row's chain (dense grad, no lookup); Hessian writes use the cached LT
    // self-product slots (flat values[] += , no lookup). Matches
    // scatter_one_arm_row_sparse cell-for-cell.
    void scatter_within_row(int /*row*/, double g_row, double H_row,
                            const ParsedArm& /*pa*/, int /*k_arm*/,
                            const std::vector<LatentBlock>& /*blocks*/,
                            const std::vector<double>& /*d_eff*/,
                            int kk, int j,
                            DenseVec& grad, SparseHessianBuilder& H,
                            std::vector<int>& /*active_idx*/,
                            std::vector<double>& /*active_d*/) {
        const std::vector<ArmRowChainEntry>& chain = (*chains_)[kk][j];
        for (const auto& e : chain) grad[e.idx] += g_row * e.w;

        const WithinRowSlots& w = within[kk][j];
        double* __restrict__ Hv = H.values.data();
        const int nw = (int) w.slot.size();
        for (int t = 0; t < nw; t++) {
            const int s = w.slot[t];
            if (s >= 0) Hv[s] += (H_row * w.w_first[t]) * w.w_second[t];
        }
    }

    void scatter_cross_pair(double Hkl,
                            const std::vector<ArmRowChainEntry>& /*chain_k*/,
                            const std::vector<ArmRowChainEntry>& /*chain_l*/,
                            int kk, int ll, int j, int m,
                            SparseHessianBuilder& H) {
        if (Hkl == 0.0) return;
        const CrossPairSlots& cs =
            cross[(std::size_t) kk * n_coupled_ + ll][j][m];
        double* __restrict__ Hv = H.values.data();
        const int nc = (int) cs.slot.size();
        for (int t = 0; t < nc; t++) {
            const int s = cs.slot[t];
            if (s < 0) continue;
            const double val = (Hkl * cs.w_k[t]) * cs.w_l[t];
            Hv[s] += val;
            if (cs.is_diag[t]) Hv[s] += val;
        }
    }

    void chains_ptr(
        const std::vector<std::vector<std::vector<ArmRowChainEntry>>>* c) {
        chains_ = c;
    }

    // Pointer to the cell's chains, set by the impl before per-species scatter.
    const std::vector<std::vector<std::vector<ArmRowChainEntry>>>* chains_ =
        nullptr;
};

// Fused batched cell-coupling scatter, templated on the per-species scatter
// policy (DenseScatterPolicy or SparseScatterPolicy). One pass over cells:
// build the B-batch CellEtas / CellResponse / CellDerivs views (species-major
// buffers, n_batch = B), dispatch the multi-response spec ONCE per cell (it
// loops species inner), then scatter each species s's per-row derivatives into
// its own grad_per_sp[s] / H_per_sp[s].
//
// The species-INVARIANT cell layout (row counts, row pointers, view setup, the
// design-row chains the cross-arm scatter walks, and -- for the sparse policy --
// the (row, col) -> flat-slot resolution against the shared pattern) is computed
// ONCE per cell and reused across all B species; only the per-species
// derivative slice and its scatter into H_s depend on the species. This is the
// #69 amortization: the bandwidth-bound evaluate, the design bookkeeping, and
// the sparse slot lookups are paid once, not B times.
//
// `buf.etas` must already hold the current per-species etas (species-major).
// d_eff is per coupled-arm and shared across species (one outer grid). The
// dense and sparse paths share this body; only the policy differs. B = 1
// reproduces the single-species per-cell branch.
template <typename Policy>
inline void scatter_cell_coupling_batch_impl(
    const CellCouplingSpec&                           spec,
    const std::vector<int>&                           coupled_arms,
    const std::vector<std::vector<std::vector<int>>>& cell_rows,
    int                                               n_cells,
    const std::vector<JointArm>&                      arms,
    const std::vector<ParsedArm>&                     parsed,
    const std::vector<LatentBlock>&                   blocks,
    int                                               k_grid,
    const BatchArmBuffers&                            buf,
    std::vector<DenseVec>&                            grad_per_sp,  // [B]
    std::vector<typename Policy::HContainer>&         H_per_sp,     // [B]
    Policy&                                           policy,
    CurvatureMode                                     curvature,
    bool                                              grad_only
) {
    using HContainer = typename Policy::HContainer;
    const int n_coupled = (int) coupled_arms.size();
    const int Bn        = (int) blocks.size();
    const int B         = buf.B;
    if (n_coupled == 0 || n_cells == 0) return;

    // Per coupled-arm d_eff per block (shared across species).
    std::vector<std::vector<double>> d_eff_per_arm(n_coupled,
                                                   std::vector<double>(Bn));
    for (int kk = 0; kk < n_coupled; kk++) {
        int k = coupled_arms[kk];
        for (int b = 0; b < Bn; b++) {
            double s = blocks[b].arm_scale ? blocks[b].arm_scale(k, k_grid) : 1.0;
            d_eff_per_arm[kk][b] = s * blocks[b].d_fac(k_grid);
        }
    }

    // Per-arm view pointers (species-major buffers; stride = N_k).
    std::vector<const double*> arm_eta_ptr(n_coupled);
    std::vector<const double*> arm_y_ptr(n_coupled);
    std::vector<const int*>    arm_n_trials_ptr(n_coupled);
    std::vector<int>           arm_eta_stride(n_coupled);
    std::vector<int>           arm_y_stride(n_coupled);
    std::vector<std::string>   family_holder(n_coupled);
    std::vector<const char*>   arm_family_ptr(n_coupled);
    std::vector<double>        arm_phi_first(n_coupled);   // phi for species 0
    std::vector<double>        arm_phi_batch(  (std::size_t) n_coupled * B);
    for (int kk = 0; kk < n_coupled; kk++) {
        int k = coupled_arms[kk];
        arm_eta_ptr[kk]      = buf.etas[k].data();
        arm_eta_stride[kk]   = buf.N[k];
        arm_y_ptr[kk]        = buf.y[k].empty() ? nullptr : buf.y[k].data();
        arm_y_stride[kk]     = buf.N[k];
        arm_n_trials_ptr[kk] = buf.n_trials[k].empty() ? nullptr
                                                       : buf.n_trials[k].data();
        family_holder[kk]    = arms[k].family;
        arm_family_ptr[kk]   = family_holder[kk].c_str();
        for (int s = 0; s < B; s++) {
            arm_phi_batch[(std::size_t) kk * B + s] = buf.phi[(std::size_t) k * B + s];
        }
        arm_phi_first[kk]    = arm_phi_batch[(std::size_t) kk * B];
    }

    // Per-cell scratch.
    std::vector<int>            arm_row_count(n_coupled);
    std::vector<const int*>     arm_rows_ptr(n_coupled);
    std::vector<std::vector<double>> arm_grad_buf(n_coupled);          // rc * B
    std::vector<std::vector<double>> arm_neg_hess_diag_buf(n_coupled); // rc * B
    std::vector<double*>        arm_grad_ptr(n_coupled);
    std::vector<double*>        arm_neg_hess_diag_ptr(n_coupled);

    // Cross-arm Hessian scratch: arm_cross_hess[kk][ll] is rc_k * rc_l * B
    // (species-major), kk <= ll only.
    std::vector<std::vector<std::vector<double>>> cross_hess_buf(n_coupled,
        std::vector<std::vector<double>>(n_coupled));
    std::vector<std::vector<double*>> cross_hess_ptr_inner(n_coupled,
        std::vector<double*>(n_coupled, nullptr));
    std::vector<double* const*> cross_hess_outer(n_coupled, nullptr);
    for (int kk = 0; kk < n_coupled; kk++) {
        cross_hess_outer[kk] = cross_hess_ptr_inner[kk].data();
    }

    std::vector<int>    active_idx;
    std::vector<double> active_d;
    active_idx.reserve(Bn);
    active_d.reserve(Bn);

    // Species-invariant design chains for the current cell's cross-arm pairs.
    // chains_per_arm[kk][r] is the eta -> joint-vector chain for the r-th row of
    // coupled arm kk; built once per cell (depends only on parsed / blocks /
    // d_eff, not on species) and reused for every species' cross scatter.
    std::vector<std::vector<std::vector<ArmRowChainEntry>>> chains_per_arm(n_coupled);

    for (int c = 0; c < n_cells; c++) {
        for (int kk = 0; kk < n_coupled; kk++) {
            int rc = (int) cell_rows[kk][c].size();
            arm_row_count[kk] = rc;
            arm_rows_ptr[kk]  = cell_rows[kk][c].data();
            std::size_t need = (std::size_t) rc * B;
            if (arm_grad_buf[kk].size() < need) {
                arm_grad_buf[kk].assign(need, 0.0);
                arm_neg_hess_diag_buf[kk].assign(need, 0.0);
            } else {
                std::fill(arm_grad_buf[kk].begin(), arm_grad_buf[kk].begin() + need, 0.0);
                std::fill(arm_neg_hess_diag_buf[kk].begin(),
                          arm_neg_hess_diag_buf[kk].begin() + need, 0.0);
            }
            arm_grad_ptr[kk]          = arm_grad_buf[kk].data();
            arm_neg_hess_diag_ptr[kk] = arm_neg_hess_diag_buf[kk].data();
        }

        for (int kk = 0; kk < n_coupled; kk++) {
            int rc_k = arm_row_count[kk];
            for (int ll = kk; ll < n_coupled; ll++) {
                int rc_l = arm_row_count[ll];
                std::size_t n_pair = (std::size_t) rc_k * rc_l * B;
                auto& cbuf = cross_hess_buf[kk][ll];
                if (cbuf.size() < n_pair) cbuf.assign(n_pair, 0.0);
                else std::fill(cbuf.begin(), cbuf.begin() + n_pair, 0.0);
                cross_hess_ptr_inner[kk][ll] = cbuf.data();
            }
            for (int ll = 0; ll < kk; ll++) cross_hess_ptr_inner[kk][ll] = nullptr;
        }

        CellEtas etas_view;
        etas_view.arm_eta_ptr   = arm_eta_ptr.data();
        etas_view.arm_rows      = arm_rows_ptr.data();
        etas_view.arm_row_count = arm_row_count.data();
        etas_view.n_arms_       = n_coupled;
        etas_view.arm_eta_stride = arm_eta_stride.data();
        etas_view.n_batch_      = B;

        CellResponse y_view;
        y_view.arm_y           = arm_y_ptr.data();
        y_view.arm_n_trials    = arm_n_trials_ptr.data();
        y_view.arm_family      = arm_family_ptr.data();
        y_view.arm_phi         = arm_phi_first.data();
        y_view.arm_rows        = arm_rows_ptr.data();
        y_view.arm_row_count   = arm_row_count.data();
        y_view.n_arms_         = n_coupled;
        y_view.arm_y_stride    = arm_y_stride.data();
        y_view.arm_phi_batch   = arm_phi_batch.data();
        y_view.n_batch_        = B;

        CellDerivs out;
        out.arm_grad           = arm_grad_ptr.data();
        out.arm_neg_hess_diag  = arm_neg_hess_diag_ptr.data();
        out.arm_cross_hess     = cross_hess_outer.data();
        out.arm_row_count      = arm_row_count.data();
        out.n_arms_            = n_coupled;
        out.n_batch_           = B;
        out.curvature          = curvature;
        out.grad_only          = grad_only;

        spec.evaluate_cell(c, etas_view, y_view, out);

        // Species-invariant design chains for this cell, built once and reused
        // across every species' within-arm and cross-arm scatter. The within-
        // arm row Hessian is the lower triangle of H_row * outer(chain, chain),
        // so the same chain drives within-arm gradient + Hessian and the cross-
        // arm Hessian.
        for (int kk = 0; kk < n_coupled; kk++) {
            int k  = coupled_arms[kk];
            int rc = arm_row_count[kk];
            const int* rows = arm_rows_ptr[kk];
            auto& ch_kk = chains_per_arm[kk];
            if ((int) ch_kk.size() < rc) ch_kk.resize(rc);
            for (int r = 0; r < rc; r++) {
                build_arm_row_chain(rows[r], parsed[k], k, blocks,
                                    d_eff_per_arm[kk], ch_kk[r]);
            }
        }

        // Resolve the policy's species-invariant per-cell scatter caches once
        // (the sparse policy's flat-slot lookups against the shared pattern;
        // the dense policy is a no-op).
        policy.chains_ptr(&chains_per_arm);
        policy.prepare_cell(coupled_arms, arm_row_count, arm_rows_ptr, parsed,
                            blocks, d_eff_per_arm, chains_per_arm, H_per_sp);

        // Per-species scatter into that species' own grad_s / H_s.
        for (int s = 0; s < B; s++) {
            DenseVec& grad = grad_per_sp[s];
            HContainer& H  = H_per_sp[s];

            // Within-arm per-row scatter (gradient + diagonal-curvature
            // Hessian).
            for (int kk = 0; kk < n_coupled; kk++) {
                int k  = coupled_arms[kk];
                int rc = arm_row_count[kk];
                const int* rows = arm_rows_ptr[kk];
                const double* g = arm_grad_buf[kk].data() + (std::size_t) s * rc;
                const double* h = arm_neg_hess_diag_buf[kk].data() + (std::size_t) s * rc;
                for (int j = 0; j < rc; j++) {
                    policy.scatter_within_row(
                        rows[j], g[j], h[j],
                        parsed[k], k, blocks, d_eff_per_arm[kk], kk, j,
                        grad, H, active_idx, active_d);
                }
            }

            // Cross-arm Hessian scatter (species s's slice), reusing the
            // per-cell design chains / slot caches.
            if (!grad_only) {
                for (int kk = 0; kk < n_coupled; kk++) {
                    int rc_k = arm_row_count[kk];
                    for (int ll = kk; ll < n_coupled; ll++) {
                        const double* ch = cross_hess_ptr_inner[kk][ll];
                        if (!ch) continue;
                        int rc_l = arm_row_count[ll];
                        const std::size_t base = (std::size_t) s * rc_k * rc_l;
                        for (int j = 0; j < rc_k; j++) {
                            const std::vector<ArmRowChainEntry>& chain_k =
                                chains_per_arm[kk][j];
                            int m_start = (kk == ll) ? (j + 1) : 0;
                            for (int m = m_start; m < rc_l; m++) {
                                double Hkl = ch[base + (std::size_t) j * rc_l + m];
                                if (Hkl == 0.0) continue;
                                policy.scatter_cross_pair(
                                    Hkl, chain_k, chains_per_arm[ll][m],
                                    kk, ll, j, m, H);
                            }
                        }
                    }
                }
            }
        }
    }
}

// Dense wrapper: fused batched scatter into per-species DenseMat H_per_sp via
// scatter_one_arm_row_dense + scatter_cross_chain_dense.
inline void scatter_cell_coupling_batch_dense(
    const CellCouplingSpec&                           spec,
    const std::vector<int>&                           coupled_arms,
    const std::vector<std::vector<std::vector<int>>>& cell_rows,
    int                                               n_cells,
    const std::vector<JointArm>&                      arms,
    const std::vector<ParsedArm>&                     parsed,
    const std::vector<LatentBlock>&                   blocks,
    int                                               k_grid,
    const BatchArmBuffers&                            buf,
    std::vector<DenseVec>&                            grad_per_sp,  // [B]
    std::vector<DenseMat>&                            H_per_sp,     // [B]
    CurvatureMode                                     curvature = CurvatureMode::Observed,
    bool                                              grad_only = false
) {
    DenseScatterPolicy policy;
    scatter_cell_coupling_batch_impl(
        spec, coupled_arms, cell_rows, n_cells, arms, parsed, blocks, k_grid,
        buf, grad_per_sp, H_per_sp, policy, curvature, grad_only);
}

// Sparse wrapper: fused batched scatter into per-species SparseHessianBuilder
// H_per_sp via scatter_one_arm_row_sparse + scatter_cross_chain_sparse. Each
// builder must already carry the joint structural pattern (built once by
// build_joint_hessian_pattern) so H.add() lands every (row, col) the per-row /
// cross helpers touch -- the same pattern, hence the same nnz, as the
// single-species sparse oracle.
inline void scatter_cell_coupling_batch_sparse(
    const CellCouplingSpec&                           spec,
    const std::vector<int>&                           coupled_arms,
    const std::vector<std::vector<std::vector<int>>>& cell_rows,
    int                                               n_cells,
    const std::vector<JointArm>&                      arms,
    const std::vector<ParsedArm>&                     parsed,
    const std::vector<LatentBlock>&                   blocks,
    int                                               k_grid,
    const BatchArmBuffers&                            buf,
    std::vector<DenseVec>&                            grad_per_sp,  // [B]
    std::vector<SparseHessianBuilder>&                H_per_sp,     // [B]
    SparseScatterPolicy&                              policy,
    CurvatureMode                                     curvature = CurvatureMode::Observed,
    bool                                              grad_only = false
) {
    scatter_cell_coupling_batch_impl(
        spec, coupled_arms, cell_rows, n_cells, arms, parsed, blocks, k_grid,
        buf, grad_per_sp, H_per_sp, policy, curvature, grad_only);
}

// Batched outer-grid driver (dense). Defined in nested_laplace_joint_batch.cpp.
// Returns an Rcpp::List of length n_batch; element s is
// List(log_marginal[n_grid], weights[n_grid], modes[n_grid x n_x], n_iter).
// All-coupled cell-coupling families only (occu_cover); errors otherwise.
Rcpp::List run_multi_block_nested_laplace_joint_batch(
    int                              n_grid,
    int                              n_batch,
    std::vector<JointArm>&           arms,
    const std::vector<ParsedArm>&    parsed,
    const std::vector<LatentBlock>&  blocks,
    int                              n_x_after_re,
    const BatchArmBuffers&           buf,
    int                              max_iter,
    double                           tol,
    std::function<void(int)>         prep_at_grid,
    std::shared_ptr<CellCouplingSpec> spec,
    bool                             store_Q = true
);

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_JOINT_BATCH_H
