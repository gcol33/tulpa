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

// Fused batched cell-coupling scatter (dense). One pass over cells: build the
// B-batch CellEtas / CellResponse / CellDerivs views (species-major buffers,
// n_batch = B), dispatch the multi-response spec ONCE per cell (it loops
// species inner), then scatter each species s's per-row derivatives into its
// own grad_per_sp[s] / H_per_sp[s] via the shared single-species
// scatter_one_arm_row_dense + build_arm_row_chain helpers. B = 1 reproduces
// scatter_cell_coupling_dense_branch exactly.
//
// `buf.etas` must already hold the current per-species etas (species-major).
// d_eff is per coupled-arm and shared across species (one outer grid).
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
    std::vector<ArmRowChainEntry> chain_k_scratch;
    std::vector<ArmRowChainEntry> chain_l_scratch;
    chain_k_scratch.reserve(32);
    chain_l_scratch.reserve(32);

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

        // Per-species scatter into that species' own grad_s / H_s.
        for (int s = 0; s < B; s++) {
            DenseVec& grad = grad_per_sp[s];
            DenseMat& H    = H_per_sp[s];

            // Within-arm per-row scatter.
            for (int kk = 0; kk < n_coupled; kk++) {
                int k  = coupled_arms[kk];
                int rc = arm_row_count[kk];
                const int* rows = arm_rows_ptr[kk];
                const double* g = arm_grad_buf[kk].data() + (std::size_t) s * rc;
                const double* h = arm_neg_hess_diag_buf[kk].data() + (std::size_t) s * rc;
                for (int j = 0; j < rc; j++) {
                    scatter_one_arm_row_dense(
                        rows[j], g[j], h[j],
                        parsed[k], k, blocks, d_eff_per_arm[kk],
                        grad, H, active_idx, active_d);
                }
            }

            // Cross-arm Hessian scatter (species s's slice).
            if (!grad_only) {
                for (int kk = 0; kk < n_coupled; kk++) {
                    int k    = coupled_arms[kk];
                    int rc_k = arm_row_count[kk];
                    const int* rows_k = arm_rows_ptr[kk];
                    for (int ll = kk; ll < n_coupled; ll++) {
                        const double* ch = cross_hess_ptr_inner[kk][ll];
                        if (!ch) continue;
                        int l    = coupled_arms[ll];
                        int rc_l = arm_row_count[ll];
                        const int* rows_l = arm_rows_ptr[ll];
                        const std::size_t base = (std::size_t) s * rc_k * rc_l;
                        for (int j = 0; j < rc_k; j++) {
                            build_arm_row_chain(rows_k[j], parsed[k], k, blocks,
                                                d_eff_per_arm[kk], chain_k_scratch);
                            int m_start = (kk == ll) ? (j + 1) : 0;
                            for (int m = m_start; m < rc_l; m++) {
                                double Hkl = ch[base + (std::size_t) j * rc_l + m];
                                if (Hkl == 0.0) continue;
                                build_arm_row_chain(rows_l[m], parsed[l], l, blocks,
                                                    d_eff_per_arm[ll], chain_l_scratch);
                                scatter_cross_chain_dense(Hkl, chain_k_scratch,
                                                          chain_l_scratch, H);
                            }
                        }
                    }
                }
            }
        }
    }
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
