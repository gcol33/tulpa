// scatter_dense_basis.h
// Batched BLAS-3 scatter for DENSE_BASIS LatentBlock contributions.
//
// Stage 2.1 of the Stage 2 perf plan. Replaces the per-obs scalar inner
// loops for HSGP / HSGP-MO / HSGP-SVC. For one arm at one outer-grid cell,
// scatters:
//   * H[block, block] += d_eff^2 * Q_k^T diag(neg_hess) Q_k         (SYRK)
//   * H[block, beta]  += d_eff   * Q_k^T diag(neg_hess) X           (GEMM)
//   * H[block, RE_g]  += d_eff   * sum_{i in g} neg_hess_i Q_k(i,:) (per-group accumulate)
//   * grad[block]     += d_eff   * Q_k^T grad_obs                   (GEMV)
//
// where Q_k(i, j) = Phi_k(i, j) * sqrt_S[j] is the per-arm scaled basis.
//
// The DENSE_BASIS block × block sub-pattern is fully populated by
// build_joint_hessian_pattern (joint_hessian_pattern.h), so every entry
// this helper writes is guaranteed to exist in the SparseHessianBuilder
// pattern. H.add normalizes (r, c) -> (max, min) internally; we always
// write lower triangle (j_row >= j_col within the block).
//
// What this helper does NOT handle (left to the per-obs scatter):
//   * DENSE_BASIS x INDEXED_* cross terms (different blocks)
//   * DENSE_BASIS x DENSE_BASIS inter-block cross (different blocks)
// These remain in scatter_arm_obs_joint_multi_sparse with the block's
// per-obs basis_eval; this is the rare case (multi-scale HSGP or HSGP +
// ICAR) and the cross cost is small compared to the M^2 intra-block win.

#ifndef TULPA_SCATTER_DENSE_BASIS_H
#define TULPA_SCATTER_DENSE_BASIS_H

#include "laplace_family_link.h"
#include "latent_block.h"
#include "nested_laplace_joint_core.h"
#include "sparse_hessian.h"
#include <Eigen/Core>
#include <Rcpp.h>
#include <vector>

namespace tulpa {

// Per-obs cache entry for DENSE_BASIS dofs kept alongside INDEXED active
// dofs in the per-obs scatter loop. Only populated when cross terms must
// be emitted from per-obs (DENSE_BASIS x INDEXED, or DENSE_BASIS x
// DENSE_BASIS across different blocks); in the pure single-DB case the
// per-obs loop skips DB entirely and scatter_dense_basis_block handles
// everything.
//
//   has_batch  : block has a dense_basis_batch hook so the batch helper
//                covers block-internal contributions (block x block,
//                block x beta, block x RE, block gradient). The per-obs
//                loop emits ONLY cross terms with other blocks and skips
//                the gradient + same-block H entries.
//   has_batch == false (legacy): the block does not provide a batch hook,
//                so the per-obs loop must also emit block-internal
//                contributions, matching pre-Stage-2.1 behavior.
struct DenseBasisActive {
    int    dof;
    double weight;
    int    block_idx;
    bool   has_batch;
};

// Per-(arm, DENSE_BASIS block) scratch. One per outer-grid thread.
// Buffers are resized lazily by scatter_dense_basis_block.
//
// Stage 2.2a (scatter index cache): the per-call H.add() map lookups for
// the M*(M+1)/2 SYRK output, M*p_k GEMM output, and M*n_re_k per-group
// RE writes are replaced with direct values[idx] += val writes via
// precomputed flat-index arrays. The cache is built lazily and keyed on
// (H_builder pointer, M, p_k, n_re_k, block start + m_offset, bstart,
// rstart); any mismatch triggers a rebuild. In the common case (same
// H_builder reused across Newton iters and outer-grid cells, same block /
// arm shape) the cache is built once and reused.
struct DenseBasisScratch {
    Eigen::VectorXd g_obs;        // length N_k
    Eigen::VectorXd w_obs;        // length N_k
    Eigen::VectorXd w_sqrt;       // length N_k
    Eigen::MatrixXd WQ;           // N_k x M (W^{1/2} Q)
    Eigen::MatrixXd H_block;      // M x M (SYRK output, lower)
    Eigen::MatrixXd H_beta;       // M x p_k (GEMM output)
    Eigen::MatrixXd H_re;         // M x n_re (per-group accumulate)
    Eigen::VectorXd grad_block;   // length M

    // ---- Scatter index cache ----
    // idx_bb stores flat values[] indices for block x block (lower-tri
    // including diagonal) entries in column-major order matching the
    // write loop below: for j2 in [0,M), for j1 in [j2,M), entry is
    // (blk_row + j1, blk_col + j2). Length M*(M+1)/2.
    std::vector<int> idx_bb;
    // idx_bbeta stores flat indices for block x beta in column-major
    // order: for l in [0,p_k), for j in [0,M), entry is (blk_row+j, bstart+l).
    // Length M * p_k.
    std::vector<int> idx_bbeta;
    // idx_bre stores flat indices for block x RE in column-major order:
    // for g in [0,n_re_k), for j in [0,M), entry is (blk_row+j, rstart+g).
    // Length M * n_re_k.
    std::vector<int> idx_bre;

    // Cache key. -1 sentinel means "not yet built".
    const void* cache_H_ptr = nullptr;
    int cache_H_nnz   = -1;
    int cache_blk_off = -1;   // blk.start + m_off (row anchor)
    int cache_M       = -1;
    int cache_p_k     = -1;
    int cache_n_re_k  = -1;
    int cache_bstart  = -1;
    int cache_rstart  = -1;
};

// Batched scatter for one DENSE_BASIS block on one arm. Returns true if
// any work was done (block was DENSE_BASIS with a dense_basis_batch hook
// and d_eff != 0); false otherwise so the caller can fall back to the
// per-obs path.
inline bool scatter_dense_basis_block(
    const LatentBlock&            blk,
    int                            k_arm,
    const ParsedArm&               pa,
    const JointArm&                arm,
    const ArmSpecView&             view,
    const Rcpp::NumericVector&     eta,
    double                         d_eff,
    DenseVec&                      grad,
    SparseHessianBuilder&          H,
    DenseBasisScratch&             scratch
) {
    if (blk.contrib_kind != BlockContribKind::DENSE_BASIS) return false;
    if (!blk.dense_basis_batch) return false;
    if (d_eff == 0.0) return true;  // nothing to scatter, but skip per-obs fallback

    const LatentBlock::DenseBasisBatch batch = blk.dense_basis_batch(k_arm);
    if (batch.data == nullptr) return false;

    const int N_k       = batch.N_k;
    const int M         = batch.m_per_arm;
    const int m_off     = batch.m_offset_in_block;
    const int p_k       = pa.p;
    const int n_re_k    = pa.n_re_groups;
    const int bstart    = pa.beta_start;
    const int rstart    = pa.re_start;

    if (N_k == 0 || M == 0) return true;
    if (N_k != arm.N) {
        Rcpp::stop("scatter_dense_basis_block: batch.N_k (%d) != arm.N (%d) "
                   "for k_arm = %d.", N_k, arm.N, k_arm);
    }

    // ---- Scatter index cache: build / refresh if stale ----
    // The cache replaces per-write std::map lookups (~50 ns each) with
    // direct values[idx] += val writes. Built lazily; rebuilt only when
    // the H_builder pointer / nnz changes or the shape parameters shift.
    const int blk_off_row = blk.start + m_off;
    const void* H_ptr     = static_cast<const void*>(&H);
    const bool cache_hit  =
        (scratch.cache_H_ptr   == H_ptr)
        && (scratch.cache_H_nnz   == H.nnz)
        && (scratch.cache_blk_off == blk_off_row)
        && (scratch.cache_M       == M)
        && (scratch.cache_p_k     == p_k)
        && (scratch.cache_n_re_k  == n_re_k)
        && (scratch.cache_bstart  == bstart)
        && (scratch.cache_rstart  == rstart);
    if (!cache_hit) {
        // Block x block, lower triangle column-major: for j2 in [0,M),
        // for j1 in [j2,M).
        scratch.idx_bb.resize(static_cast<size_t>(M) * (M + 1) / 2);
        {
            int t = 0;
            for (int j2 = 0; j2 < M; j2++) {
                const int c = blk_off_row + j2;
                for (int j1 = j2; j1 < M; j1++) {
                    const int r = blk_off_row + j1;
                    scratch.idx_bb[t++] = H.lookup(r, c);
                }
            }
        }
        // Block x beta, column-major: for l in [0,p_k), for j in [0,M).
        scratch.idx_bbeta.resize(static_cast<size_t>(M) * p_k);
        for (int l = 0; l < p_k; l++) {
            const int c = bstart + l;
            for (int j = 0; j < M; j++) {
                const int r = blk_off_row + j;
                scratch.idx_bbeta[static_cast<size_t>(l) * M + j] =
                    H.lookup(r, c);
            }
        }
        // Block x RE, column-major: for g in [0,n_re_k), for j in [0,M).
        scratch.idx_bre.resize(static_cast<size_t>(M) * n_re_k);
        for (int g = 0; g < n_re_k; g++) {
            const int c = rstart + g;
            for (int j = 0; j < M; j++) {
                const int r = blk_off_row + j;
                scratch.idx_bre[static_cast<size_t>(g) * M + j] =
                    H.lookup(r, c);
            }
        }

        scratch.cache_H_ptr   = H_ptr;
        scratch.cache_H_nnz   = H.nnz;
        scratch.cache_blk_off = blk_off_row;
        scratch.cache_M       = M;
        scratch.cache_p_k     = p_k;
        scratch.cache_n_re_k  = n_re_k;
        scratch.cache_bstart  = bstart;
        scratch.cache_rstart  = rstart;
    }

    // Per-obs gradient and curvature (sign matches grad_hess_for_family:
    // gh.grad is +d log p / d eta, gh.neg_hess is the Fisher curvature).
    scratch.g_obs.resize(N_k);
    scratch.w_obs.resize(N_k);
    scratch.w_sqrt.resize(N_k);
    for (int i = 0; i < N_k; i++) {
        auto gh = arm_grad_hess(view, i, eta[i]);
        scratch.g_obs(i) = gh.grad;
        scratch.w_obs(i) = gh.neg_hess;
        const double w = gh.neg_hess;
        scratch.w_sqrt(i) = (w > 0.0) ? std::sqrt(w) : 0.0;
    }

    // Map raw Phi (row-major) and apply column scaling sqrt_S inside WQ.
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        Phi(batch.data, N_k, M);

    Eigen::Map<const Eigen::VectorXd> sqrt_S_vec(
        batch.sqrt_S, batch.sqrt_S ? M : 0);

    scratch.WQ.resize(N_k, M);
    if (batch.sqrt_S) {
        // WQ = diag(sqrt(w)) * Phi * diag(sqrt_S)
        scratch.WQ.noalias() = scratch.w_sqrt.asDiagonal() * Phi
                                * sqrt_S_vec.asDiagonal();
    } else {
        scratch.WQ.noalias() = scratch.w_sqrt.asDiagonal() * Phi;
    }

    // ---- Block x block (SYRK, lower triangle) ----
    // H_block = WQ^T WQ (M x M, symmetric); only lower written.
    scratch.H_block.setZero(M, M);
    scratch.H_block.template selfadjointView<Eigen::Lower>()
        .rankUpdate(scratch.WQ.transpose());

    const double dd = d_eff * d_eff;
    // Cached write order: column-major lower triangle (j2 outer, j1 inner
    // over [j2, M)), matching scratch.idx_bb layout above.
    {
        double* __restrict__ Hv = H.values.data();
        const int* __restrict__ idx_bb = scratch.idx_bb.data();
        int t = 0;
        for (int j2 = 0; j2 < M; j2++) {
            for (int j1 = j2; j1 < M; j1++) {
                const int k = idx_bb[t++];
                if (k >= 0) Hv[k] += dd * scratch.H_block(j1, j2);
            }
        }
    }

    // ---- Block x beta (GEMM) ----
    // H_beta[j, l] = sum_i neg_hess_i * Q_k(i, j) * X(i, l)
    //              = (W^{1/2} Q)^T (W^{1/2} X) [j, l]
    // Building Q (= Phi * diag(sqrt_S)) at full scale would re-multiply,
    // so we fold sqrt_S in via Q_k = WQ / sqrt(w_i). Simpler form:
    //   H_beta = Q^T diag(w) X = Phi^T diag(sqrt_S) diag(w) X
    // We already have WQ = diag(sqrt(w)) Phi diag(sqrt_S), so
    //   WQ^T * (diag(sqrt(w)) X) = Phi^T diag(sqrt_S) diag(w) X = H_beta.
    if (p_k > 0) {
        // X is column-major Rcpp::NumericMatrix [N_k x p_k]. Map directly.
        Eigen::Map<const Eigen::MatrixXd> X_map(&pa.X(0, 0), N_k, p_k);
        // W^{1/2} X
        Eigen::MatrixXd WX(N_k, p_k);
        WX.noalias() = scratch.w_sqrt.asDiagonal() * X_map;
        scratch.H_beta.resize(M, p_k);
        scratch.H_beta.noalias() = scratch.WQ.transpose() * WX;

        // Cached write order matches scratch.idx_bbeta layout (column-major
        // over (l, j)).
        double* __restrict__ Hv = H.values.data();
        const int* __restrict__ idx_bbeta = scratch.idx_bbeta.data();
        for (int l = 0; l < p_k; l++) {
            for (int j = 0; j < M; j++) {
                const int k = idx_bbeta[static_cast<size_t>(l) * M + j];
                if (k >= 0) Hv[k] += d_eff * scratch.H_beta(j, l);
            }
        }
    }

    // ---- Block x RE (per-group accumulate; cheap, BLAS-2-ish) ----
    if (n_re_k > 0) {
        scratch.H_re.setZero(M, n_re_k);
        for (int i = 0; i < N_k; i++) {
            const int gi = static_cast<int>(pa.re_idx[i]) - 1;
            if (gi < 0 || gi >= n_re_k) continue;
            const double w_i = scratch.w_obs(i);
            if (w_i == 0.0) continue;
            // Add w_i * Q_k(i, :) to column gi of H_re.
            // Q_k(i, j) = Phi(i, j) * sqrt_S[j].
            // Using WQ here would multiply by sqrt(w) twice; use WQ /
            // sqrt(w_i) for the scaling (or re-derive Q):
            //   H_re(:, gi) += w_i * Q(i, :)^T
            //                = sqrt(w_i) * WQ(i, :)^T
            // since WQ(i, :) = sqrt(w_i) * Q(i, :).
            const double s = scratch.w_sqrt(i);
            scratch.H_re.col(gi).noalias() += s * scratch.WQ.row(i).transpose();
        }
        // Cached write order matches scratch.idx_bre layout (column-major
        // over (g, j)).
        double* __restrict__ Hv = H.values.data();
        const int* __restrict__ idx_bre = scratch.idx_bre.data();
        for (int g = 0; g < n_re_k; g++) {
            for (int j = 0; j < M; j++) {
                const int k = idx_bre[static_cast<size_t>(g) * M + j];
                if (k >= 0) Hv[k] += d_eff * scratch.H_re(j, g);
            }
        }
    }

    // ---- Gradient: grad[block] += d_eff * Q^T g_obs ----
    // Q^T g_obs = diag(sqrt_S) Phi^T g_obs.
    scratch.grad_block.resize(M);
    scratch.grad_block.noalias() = Phi.transpose() * scratch.g_obs;
    if (batch.sqrt_S) {
        scratch.grad_block.array() *= sqrt_S_vec.array();
    }
    for (int j = 0; j < M; j++) {
        grad[blk.start + m_off + j] += d_eff * scratch.grad_block(j);
    }

    return true;
}

} // namespace tulpa

#endif // TULPA_SCATTER_DENSE_BASIS_H
