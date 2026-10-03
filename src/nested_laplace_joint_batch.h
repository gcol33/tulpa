// nested_laplace_joint_batch.h
// Batched (multi-response) fused inner-solve primitives for the joint
// nested-Laplace cell-coupling path.
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
    // n_trials[k]: length N_k, or empty for a no-data arm. Built once.
    //
    // Trial counts are shared DESIGN, not per-species response: the B species
    // differ in `y` / `y_pos` and in dispersion and in nothing else (the header
    // note above), so every species of arm k reads arm k's own vector. That is
    // why this buffer is NOT species-major the way `etas` / `y` are -- a
    // per-species fill is not expressible, so a reader cannot silently pick up
    // species 0's effort for every species. A consumer needing species-specific
    // effort (a binomial arm whose trial counts differ by species) needs an
    // `n_trials_batch` input alongside `y_batch` and a stride here, which
    // changes the exported signature.
    std::vector<std::vector<int>> n_trials;
    // phi[k * B + s]: per-species dispersion for arm k. Built once; an arm with
    // a dispersion axis has it rewritten at every outer-grid cell from
    // `phi_grid` (load_grid_cell).
    std::vector<double> phi;
    // Per-arm dispersion axes on the outer grid, one column per species
    // (n_cols == B). An arm with no axis has an empty entry and keeps `phi`.
    ArmGridTable phi_grid;
    std::vector<int> N;          // per-arm row count N_k
    int n_arms = 0;

    // Load every species' dispersion at outer-grid cell kg for each arm that
    // carries an axis. Every reader of `phi` (the fused scatter and the
    // per-species objective) then sees the cell's dispersion.
    void load_grid_cell(int kg) {
        for (int k = 0; k < n_arms; k++) {
            if (phi_grid.active(k))
                phi_grid.load_cell(k, kg, phi.data() + (std::size_t) k * B);
        }
    }

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

// Per-species scatter targets for the fused batched cell-coupling pass. The
// cell scaffolding (row counts, view setup, the multi-response evaluate_cell
// pass, the species-invariant design chains) is owned by
// scatter_cell_coupling_batch_impl below and identical for every Hessian
// container; a target decides where species s's row and cross-row writes land:
//   within_row(s, kk, k, row, g, h, pa, blocks, d_eff)
//   cross(s, kk, ll, Hkl, chain_k, chain_l)
// with begin_cell(c) before a cell's first write. Each is the single-species
// target (CoupledDenseScatter / CoupledPlannedScatter) applied to species s, so
// a species is written exactly as its own single-species fit writes it.

struct BatchDenseScatter {
    std::vector<DenseVec>& grad;   // [B]
    std::vector<DenseMat>& H;      // [B]
    std::vector<int>       active_idx;
    std::vector<double>    active_d;

    BatchDenseScatter(std::vector<DenseVec>& g, std::vector<DenseMat>& h,
                      int n_blocks)
        : grad(g), H(h) {
        active_idx.reserve(n_blocks);
        active_d.reserve(n_blocks);
    }
    void begin_cell(int) {}
    void within_row(int s, int, int k, int row, double g, double h,
                    const ParsedArm& pa, const std::vector<LatentBlock>& blocks,
                    const std::vector<double>& d_eff) {
        scatter_one_arm_row_dense(row, g, h, pa, k, blocks, d_eff, grad[s], H[s],
                                  active_idx, active_d);
    }
    void cross(int s, int, int, double Hkl,
               const std::vector<ArmRowChainEntry>& ck,
               const std::vector<ArmRowChainEntry>& cl) {
        scatter_cross_chain_dense(Hkl, ck, cl, H[s]);
    }
};

// One chunk's planned sparse targets, one per species. All B species share one
// pattern and therefore one plan; each writes through its own chunk sink.
struct BatchPlannedScatter {
    std::vector<CoupledPlannedScatter> sp;   // [B]

    void begin_cell(int c) { for (auto& t : sp) t.begin_cell(c); }
    void within_row(int s, int kk, int k, int row, double g, double h,
                    const ParsedArm& pa, const std::vector<LatentBlock>& blocks,
                    const std::vector<double>& d_eff) {
        sp[s].row(kk, k, row, g, h, pa, blocks, d_eff);
    }
    void cross(int s, int kk, int ll, double Hkl,
               const std::vector<ArmRowChainEntry>& ck,
               const std::vector<ArmRowChainEntry>& cl) {
        sp[s].cross(kk, ll, Hkl, ck, cl);
    }
};

// Fused batched cell-coupling scatter over the cells [c0, c1), templated on
// the per-species scatter target (BatchDenseScatter or BatchPlannedScatter).
// One pass over cells: build the B-batch CellEtas / CellResponse / CellDerivs
// views (species-major buffers, n_batch = B), dispatch the multi-response spec
// ONCE per cell (it loops species inner), then hand each species s's per-row
// derivatives and cross-row entries to the target, which writes them into that
// species' own gradient and Hessian.
//
// The species-INVARIANT cell layout (row counts, row pointers, view setup, the
// design-row chains the cross-arm scatter walks) is computed ONCE per cell and
// reused across all B species; only the per-species derivative slice and its
// scatter depend on the species. That is the amortization: the
// bandwidth-bound evaluate and the design bookkeeping are paid once, not B
// times. The sparse target's flat offsets come from the coupled plan, resolved
// once per fit and shared by every species, since all B share one pattern.
//
// `buf.etas` must already hold the current per-species etas (species-major).
// d_eff is per coupled-arm and shared across species (one outer grid). B = 1
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
    Policy&                                           policy,
    CurvatureMode                                     curvature,
    bool                                              grad_only,
    int                                               c0,
    int                                               c1
) {
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
            d_eff_per_arm[kk][b] = s * blocks[b].d_fac_at(k_grid);
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

    // Species-invariant design chains for the current cell's cross-arm pairs.
    // chains_per_arm[kk][r] is the eta -> joint-vector chain for the r-th row of
    // coupled arm kk; built once per cell (depends only on parsed / blocks /
    // d_eff, not on species) and reused for every species' cross scatter.
    std::vector<std::vector<std::vector<ArmRowChainEntry>>> chains_per_arm(n_coupled);

    for (int c = c0; c < c1; c++) {
        policy.begin_cell(c);
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
        // across every species' cross-arm scatter.
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

        // Per-species scatter into that species' own gradient and Hessian.
        for (int s = 0; s < B; s++) {

            // Within-arm per-row scatter (gradient + diagonal-curvature
            // Hessian).
            for (int kk = 0; kk < n_coupled; kk++) {
                int k  = coupled_arms[kk];
                int rc = arm_row_count[kk];
                const int* rows = arm_rows_ptr[kk];
                const double* g = arm_grad_buf[kk].data() + (std::size_t) s * rc;
                const double* h = arm_neg_hess_diag_buf[kk].data() + (std::size_t) s * rc;
                for (int j = 0; j < rc; j++) {
                    policy.within_row(s, kk, k, rows[j], g[j], h[j],
                                      parsed[k], blocks, d_eff_per_arm[kk]);
                }
            }

            // Cross-arm Hessian scatter (species s's slice), reusing the
            // per-cell design chains.
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
                                policy.cross(s, kk, ll, Hkl, chain_k,
                                             chains_per_arm[ll][m]);
                            }
                        }
                    }
                }
            }
        }
    }
}

// Dense wrapper: fused batched scatter into per-species DenseMat H_per_sp, every
// cell in one pass.
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
    BatchDenseScatter policy(grad_per_sp, H_per_sp,
                             static_cast<int>(blocks.size()));
    scatter_cell_coupling_batch_impl(
        spec, coupled_arms, cell_rows, n_cells, arms, parsed, blocks, k_grid,
        buf, policy, curvature, grad_only, 0, n_cells);
}

// Sparse wrapper: fused batched scatter into per-species SparseHessianBuilder
// H_per_sp over the coupled plan's fixed chunks, each species' shared entries
// reduced in chunk order -- the single-species sparse branch's partition and
// reduction, so each species reproduces its own single-species fit. Every
// builder must carry the pattern the plan was resolved against.
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
    const CoupledScatterPlan&                         plan,
    CurvatureMode                                     curvature = CurvatureMode::Observed,
    bool                                              grad_only = false
) {
    const int B = buf.B;
    if (coupled_arms.empty() || n_cells == 0) return;
    for (int s = 0; s < B; s++) {
        if (!plan.valid_for(H_per_sp[s], n_cells)) {
            throw std::logic_error(
                "coupled scatter plan was not resolved against this Hessian "
                "pattern");
        }
    }
    std::vector<ArmIndexedView> views;
    views.reserve(plan.arm.size());
    for (const auto& ac : plan.arm) views.emplace_back(ac);

    std::vector<CoupledChunkPartials> partials;
    partials.reserve(B);
    for (int s = 0; s < B; s++) partials.emplace_back(plan);

    for (int ch = 0; ch < plan.n_chunks; ch++) {
        BatchPlannedScatter policy;
        policy.sp.reserve(B);
        for (int s = 0; s < B; s++) {
            policy.sp.emplace_back(
                plan, views,
                partials[s].sink(ch, grad_per_sp[s].data(),
                                 H_per_sp[s].values.data()));
        }
        scatter_cell_coupling_batch_impl(
            spec, coupled_arms, cell_rows, n_cells, arms, parsed, blocks,
            k_grid, buf, policy, curvature, grad_only,
            plan.chunk_lo(ch), plan.chunk_lo(ch + 1));
    }
    for (int s = 0; s < B; s++) {
        partials[s].reduce(grad_per_sp[s].data(), H_per_sp[s].values.data());
    }
}

// Batched outer-grid driver. Defined in nested_laplace_joint_batch.cpp.
// Returns an Rcpp::List of length n_batch; element s is species s's outer-grid
// result as nl_pack_grid_results packs it, the list the single-species joint
// grid returns for that species' responses. `pd_mode`, `step_curvature`,
// `force_sparse`, `fixed_block` and `compute_fitted_var` carry the
// single-species driver's meaning -- the last of them gated on a ONE-ARM fit
// there and here alike, since a per-row predictive variance of eta is a
// statement about the one predictor such a fit has.
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
    std::shared_ptr<CellCouplingSpec> spec,
    bool                             store_Q,
    JointPDMode                      pd_mode,
    StepCurvature                    step_curvature,
    bool                             force_sparse,
    const JointFixedBlockRequest*    fixed_block,
    bool                             compute_fitted_var = true
);

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_JOINT_BATCH_H
