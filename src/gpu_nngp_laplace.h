// gpu_nngp_laplace.h
// GPU-accelerated NNGP for Laplace approximation.
// Batches the m×m neighbor covariance Cholesky factorizations on GPU.
// Falls back to CPU transparently if CUDA unavailable.

#ifndef TULPA_GPU_NNGP_LAPLACE_H
#define TULPA_GPU_NNGP_LAPLACE_H

#include "gpu_backend.h"  // single entry point; it owns the CUDA/stub choice
#include "linalg_fast.h"  // shared small-dense Cholesky / NNGP solve core
#include "nngp_cond.h"    // nngp_row_neighbours: the shared left-packed scan
#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>

namespace tulpa {

// Covariance value at distance `d`.
//
// PRECONDITION: sigma2 > 0 and phi > 0. Both Matern branches divide by phi, so
// phi = 0 sends x to infinity and returns inf * 0 = NaN, and a negative phi
// makes the exponential branch diverge and both Matern branches negative, which
// takes the assembled neighbour matrix out of the PSD cone before it reaches
// the Cholesky. batch_nngp_scatter, the only caller, rejects both at entry, so
// no covariance is ever formed from a non-positive parameter.
inline double nngp_cov_gpu(double d, double sigma2, double phi, int cov_type) {
    if (d < 1e-10) return sigma2;
    if (cov_type == 0) return sigma2 * std::exp(-d / phi);
    if (cov_type == 1) {
        double x = std::sqrt(3.0) * d / phi;
        return sigma2 * (1.0 + x) * std::exp(-x);
    }
    double x = std::sqrt(5.0) * d / phi;
    return sigma2 * (1.0 + x + x * x / 3.0) * std::exp(-x);
}

// Batch-compute all NNGP conditional means, variances, and (optionally) the
// conditional regression weights `alpha`.
//
// Uses GPU for the Cholesky step if available, CPU otherwise.
//
// Outputs:
//   cond_mean_out, cond_var_out: length n_spatial, indexed by ORIGINAL obs idx.
//   alpha_out (optional, nullable): length n_spatial * nn, flat row-major,
//       indexed by NNGP-order index (NOT by obs idx). alpha_out[i * nn + k]
//       is the conditional regression coefficient for NNGP-order-i's k-th
//       neighbor (where the k-th neighbor's obs idx is
//       nn_order[nn_idx(i, k) - 1]). Inactive slots (nn_idx(i, k) == 0) and
//       trailing pad columns are zeroed.
//
// alpha_out enables callers (e.g., laplace_mode_gp) to assemble the full
// NNGP precision matrix Λ = (I-A)' D⁻¹ (I-A) instead of the diagonal-on-w
// approximation. The conditional mean satisfies cond_mean[obs] =
//     sum_k alpha_out[i_nngp * nn + k] * w[ nn_order[nn_idx(i_nngp, k) - 1] ]
// where obs = nn_order[i_nngp].
inline void batch_nngp_scatter(
    const std::vector<double>& w,
    int n_spatial, int nn,
    double sigma2, double phi_gp, int cov_type,
    const Rcpp::NumericMatrix& coords,
    const Rcpp::IntegerMatrix& nn_idx,
    const Rcpp::NumericMatrix& nn_dist,
    const Rcpp::IntegerVector& nn_order,
    std::vector<double>& cond_mean_out,
    std::vector<double>& cond_var_out,
    bool& gpu_used,
    std::vector<double>* alpha_out = nullptr
) {
    // The neighbour tables, the ordering and the coordinates all arrive from R
    // and are used below as raw array offsets -- one of them (`obs_idx`) as the
    // offset of a WRITE into cond_mean_out / cond_var_out, where an out-of-range
    // value is silent heap corruption rather than an error. Check the whole
    // contract once here, so the loops can index without re-deriving it.
    if (n_spatial < 0 || nn < 0) {
        Rcpp::stop("batch_nngp_scatter: n_spatial (%d) and nn (%d) must be "
                   "non-negative.", n_spatial, nn);
    }
    if (!(sigma2 > 0.0) || !(phi_gp > 0.0)) {
        Rcpp::stop("batch_nngp_scatter: sigma2 (%g) and phi_gp (%g) must both "
                   "be positive.", sigma2, phi_gp);
    }
    if (nn_order.size() != n_spatial) {
        Rcpp::stop("batch_nngp_scatter: length(nn_order) (%d) must equal "
                   "n_spatial (%d).",
                   static_cast<int>(nn_order.size()), n_spatial);
    }
    for (int i = 0; i < n_spatial; i++) {
        const int o = nn_order[i];
        if (o < 0 || o >= n_spatial) {
            Rcpp::stop("batch_nngp_scatter: nn_order[%d] is %d; must be a "
                       "0-based location index in [0, %d).", i + 1, o, n_spatial);
        }
    }
    if (nn_idx.nrow() != n_spatial || nn_idx.ncol() != nn) {
        Rcpp::stop("batch_nngp_scatter: nn_idx is %d x %d; must be %d x %d.",
                   static_cast<int>(nn_idx.nrow()),
                   static_cast<int>(nn_idx.ncol()), n_spatial, nn);
    }
    if (nn_dist.nrow() != n_spatial || nn_dist.ncol() != nn) {
        Rcpp::stop("batch_nngp_scatter: nn_dist is %d x %d; must be %d x %d.",
                   static_cast<int>(nn_dist.nrow()),
                   static_cast<int>(nn_dist.ncol()), n_spatial, nn);
    }
    if (coords.nrow() < n_spatial) {
        Rcpp::stop("batch_nngp_scatter: nrow(coords) (%d) must be at least "
                   "n_spatial (%d).",
                   static_cast<int>(coords.nrow()), n_spatial);
    }

    cond_mean_out.assign(n_spatial, 0.0);
    cond_var_out.assign(n_spatial, sigma2);
    if (alpha_out) alpha_out->assign(static_cast<size_t>(n_spatial) * nn, 0.0);
    gpu_used = false;

    // Phase 1: Build all C matrices and c vectors on CPU
    struct LocData {
        int nngp_idx;     // index in NNGP ordering
        int obs_idx;      // original location index
        int n_nb;         // actual neighbor count
        bool factored;    // its neighbour covariance factorized
    };
    std::vector<LocData> locs;
    std::vector<std::vector<double>> C_mats;  // flattened nn×nn
    std::vector<std::vector<double>> c_vecs;  // length nn

    for (int i = 0; i < n_spatial; i++) {
        int obs_idx = nn_order[i];
        // Shared left-packed scan: the leading run of entries in [1, n_spatial].
        // Counting every positive entry instead reads the padding sentinel as a
        // neighbour on any row whose valid entries are not leading, and
        // nn_order[0 - 1] is then an out-of-bounds read whose result indexes
        // coords. This is the rule phase 3 below and apply_nngp_full_prior_sparse
        // already applied.
        const int n_nb = tulpa_nngp::nngp_row_neighbours(
            &nn_idx(i, 0), /*stride=*/nn_idx.nrow(), nn, n_spatial);
        if (n_nb == 0) continue;

        std::vector<double> C(nn * nn, 0.0);
        std::vector<double> c(nn, 0.0);

        for (int j = 0; j < n_nb; j++) {
            c[j] = nngp_cov_gpu(nn_dist(i, j), sigma2, phi_gp, cov_type);
        }
        for (int j1 = 0; j1 < n_nb; j1++) {
            int o1 = nn_order[nn_idx(i, j1) - 1];
            // The diagonal NUGGET rides on the MATRIX rather than on the
            // factorization, so the CPU fallback and the batched cuSOLVER call
            // factorize the same matrix, and so this kernel conditions the
            // field the way the sampler density (tulpa_gp::kGpJitter) does.
            C[j1 * nn + j1] = sigma2 + tulpa_linalg::kNngpNugget;
            for (int j2 = j1 + 1; j2 < n_nb; j2++) {
                int o2 = nn_order[nn_idx(i, j2) - 1];
                double d12 = tulpa_linalg::coords_dist(coords, o1, o2);
                double cv = nngp_cov_gpu(d12, sigma2, phi_gp, cov_type);
                C[j1 * nn + j2] = cv;
                C[j2 * nn + j1] = cv;
            }
        }
        // Pad unused diagonal for stability
        for (int j = n_nb; j < nn; j++) C[j * nn + j] = 1.0;

        locs.push_back({i, obs_idx, n_nb, true});
        C_mats.push_back(std::move(C));
        c_vecs.push_back(std::move(c));
    }

    int batch_size = static_cast<int>(locs.size());
    if (batch_size == 0) return;

    // Phase 2: Cholesky factorize — GPU or CPU
    // After this, C_mats[b] contains L (lower triangular, ROW-major, upper
    // triangle zeroed) — the layout the TriLayout::RowMajor solves read.
    //
    // The GPU result is verified against the CPU factorization of one batch
    // element before the batch is accepted. This is not defensive padding: the
    // batched cuSOLVER call returned a COLUMN-major factor that every consumer
    // here read row-major, so each conditional variance came back wrong by
    // ~0.25 on a unit-square fixture while staying finite and ordinary-looking,
    // and nothing downstream could tell. One extra m^3 factorization per batch
    // (m <= nn, against 50+ matrices) buys a check that the accepted factor is
    // the one the CPU would have produced.
    bool chol_ok = false;
    if (batch_size >= 50) {  // GPU overhead threshold
        // cuda_batched_cholesky writes into C_mats, and can also fail PARTWAY
        // (a per-matrix non-PD `info`, or a failed copy-back after some
        // matrices have landed). Either way the CPU fallback below must see the
        // original covariances, not half-factorized ones, so keep them.
        std::vector<std::vector<double>> C_orig(C_mats);

        const int probe = batch_size / 2;
        const int probe_n = locs[probe].n_nb;
        std::vector<double> probe_ref(C_orig[probe]);
        const bool probe_ref_ok =
            tulpa_linalg::chol_factor_lower<tulpa_linalg::TriLayout::RowMajor>(
                probe_ref.data(), probe_ref.data(), probe_n, nn,
                /*nugget=*/0.0);

        chol_ok = probe_ref_ok && tulpa_gpu::cuda_batched_cholesky(C_mats, nn);
        if (chol_ok) {
            // Compare the lower triangle only; the reference's upper triangle
            // still holds its input (chol_factor_lower leaves it untouched).
            double scale = 0.0, diff = 0.0;
            for (int j = 0; j < probe_n; j++) {
                for (int k = 0; k <= j; k++) {
                    const double ref = probe_ref[j * nn + k];
                    const double got = C_mats[probe][j * nn + k];
                    scale = std::max(scale, std::abs(ref));
                    diff  = std::max(diff, std::abs(ref - got));
                }
            }
            if (!(diff <= 1e-8 * std::max(scale, 1e-12))) chol_ok = false;
        }
        // The probe above compares ONE matrix against an exact CPU
        // refactorization, which catches a whole-batch failure -- a layout
        // convention, a call that did nothing. It cannot catch a PER-MATRIX one,
        // and the call documents that it can fail partway (a per-matrix non-PD
        // `info`, a copy-back that stops after some matrices have landed), so a
        // probe at a fixed index accepts every failure that begins after it.
        // The matrices also have different effective sizes -- `locs[b].n_nb` is
        // smaller at the neighbour-poor first locations -- so the probe is not
        // representative of the batch either.
        //
        // So EVERY matrix is checked against the factorization's own defining
        // identity on the diagonal: `sum_k L[j][k]^2 == C[j][j]`, plus a
        // positive pivot. That is O(n_nb^2) per matrix against the O(n_nb^3)
        // the factorization itself costs, so full coverage stays an order below
        // the work being verified.
        if (chol_ok) {
            for (int b = 0; b < batch_size && chol_ok; b++) {
                const auto& L = C_mats[b];
                const int nb = locs[b].n_nb;
                for (int j = 0; j < nb; j++) {
                    const double piv = L[j * nn + j];
                    if (!(piv > 0.0) || !std::isfinite(piv)) { chol_ok = false; break; }
                    double acc = 0.0;
                    for (int k = 0; k <= j; k++) {
                        const double v = L[j * nn + k];
                        acc += v * v;
                    }
                    const double want = C_orig[b][j * nn + j];
                    // The nugget is already on C_orig's diagonal, so this is the
                    // factorization's exact defining identity up to rounding.
                    const double tol = 1e-8 * std::max(std::abs(want), 1e-12);
                    if (!(std::abs(acc - want) <= tol)) { chol_ok = false; break; }
                }
            }
        }
        if (chol_ok) gpu_used = true;
        else C_mats.swap(C_orig);
    }

    if (!chol_ok) {
        // CPU Cholesky per matrix: shared core, in-place on the nn-strided
        // n_nb×n_nb block; zero the upper triangle afterwards, which is what
        // makes the buffer a row-major lower factor for the solves below.
        for (int b = 0; b < batch_size; b++) {
            auto& L = C_mats[b];
            int n_nb = locs[b].n_nb;
            // The nugget is already on the diagonal, so this factorizes the
            // same matrix the batched call was handed.
            locs[b].factored =
                tulpa_linalg::chol_factor_lower<tulpa_linalg::TriLayout::RowMajor>(
                    L.data(), L.data(), n_nb, nn, /*nugget=*/0.0);
            if (!locs[b].factored) continue;
            for (int j = 0; j < n_nb; j++) {
                for (int k = j + 1; k < n_nb; k++) L[j * nn + k] = 0.0;
            }
        }
    }

    // Phase 3: Solve and extract conditional mean/variance (CPU, O(m²) per loc)
    for (int b = 0; b < batch_size; b++) {
        auto& L = C_mats[b];
        auto& c = c_vecs[b];
        int n_nb = locs[b].n_nb;
        int obs_idx = locs[b].obs_idx;
        int nngp_idx = locs[b].nngp_idx;

        // A neighbour covariance that would not factorize leaves the location
        // conditioned on NOTHING -- the marginal moments the output buffers were
        // seeded with, and a zero alpha row -- rather than kriged against an
        // unusable factor.
        if (!locs[b].factored) continue;

        // Gather neighbor values. Every slot below n_nb resolves inside
        // nn_order by construction (nngp_row_neighbours bounded it) and
        // nn_order's own range was checked at entry.
        std::vector<double> w_nb(n_nb, 0.0);
        for (int j = 0; j < n_nb; j++) {
            w_nb[j] = w[nn_order[nn_idx(nngp_idx, j) - 1]];
        }

        std::vector<double> alpha(n_nb);
        tulpa_linalg::nngp_moments_from_chol<tulpa_linalg::TriLayout::RowMajor>(
            L.data(), n_nb, nn, c.data(), w_nb.data(), sigma2,
            tulpa_linalg::kNngpVarFloor,
            cond_mean_out[obs_idx], cond_var_out[obs_idx], alpha.data());

        if (alpha_out) {
            double* row = alpha_out->data() + static_cast<size_t>(nngp_idx) * nn;
            for (int j = 0; j < n_nb; j++) row[j] = alpha[j];
        }
    }
}

// =====================================================================
// NNGP full-precision prior scatter.
//
// Assembles the full NNGP precision Λ = (I - A)' D⁻¹ (I - A) contribution
// into (grad, H), instead of the diagonal-on-w approximation. Here `A` is
// the lower-triangular NNGP coefficient matrix whose rows are the
// conditional regression coefficients on the conditioning set, and
// `D = diag(cv)` is the diagonal of conditional variances.
//
// The NNGP log-prior is sum_i [ -½ log(2π v_i) - ½ q_i² / v_i ] where
// q_i = w_i - sum_k a_{i,k} w_{N(i)_k}. Differentiating:
//   ∂(-½ q_i²/v_i)/∂w_i        = -q_i / v_i
//   ∂(-½ q_i²/v_i)/∂w_{N(i)_k} = +a_{i,k} q_i / v_i
//   ∂²(-½ q_i²/v_i)/∂w_∂w'     = -(1/v_i) · ∂q_i/∂w · ∂q_i/∂w'
// which yields the off-diagonal Hessian terms across each conditioning
// set. The current latent w is read through w_block (length n_spatial,
// indexed by obs idx, i.e., x[gp_start + obs_idx]); grad and H are
// indexed by the global latent layout starting at gp_start.
//
// Hessian entries go through SparseHessianBuilder::add(). The sparsity
// pattern must include all (focal, neighbor_k) and (neighbor_k, neighbor_kp)
// pairs for every row; see make_nngp_prior_sparsity_pattern below. This is
// the only container the NNGP prior scatters into: make_nngp_block declares
// no dense `add_prior`, so blocks_require_sparse() pins every NNGP fit to the
// sparse Newton path regardless of n_x.
//
// Inputs:
//   alpha    : length n_spatial * nn, flat row-major, indexed by NNGP-order
//              index. Output of batch_nngp_scatter(..., &alpha).
//   cv       : length n_spatial, indexed by obs idx. Output of the same.
//   w_block  : length n_spatial, indexed by obs idx (latent w slice).
//   nn_idx   : n_spatial × nn, 1-based NNGP-order indices of neighbors
//              (0 = no neighbor sentinel).
//   nn_order : length n_spatial, 0-based permutation NNGP-order → obs idx.
//   gp_start : offset in the global latent layout where the spatial block
//              begins. grad[gp_start + obs_idx] and H[gp_start + obs_idx][.]
//              are scattered into.
//
// The scatter reproduces Λ to ~1e-16 relative, asserted against an
// independently assembled (I - A)' D⁻¹ (I - A) in test-nngp-prior-scatter.R.
// Compare it RELATIVELY: Λ's entries scale as 1/cond_var, so a fixture where
// the conditional variance collapses puts them at 1e13 and an absolute
// comparison reads machine epsilon as a large discrepancy.
template <typename DenseVec, typename SparseBuilder>
inline void apply_nngp_full_prior_sparse(
    DenseVec& grad, SparseBuilder& H,
    const std::vector<double>& w_block,
    const std::vector<double>& alpha,
    const std::vector<double>& cv,
    const Rcpp::IntegerMatrix& nn_idx,
    const Rcpp::IntegerVector& nn_order,
    int n_spatial, int nn, int gp_start
) {
    std::vector<int> nb_obs(nn);
    std::vector<double> a_row(nn);
    for (int i_nngp = 0; i_nngp < n_spatial; i_nngp++) {
        int obs_focal = nn_order[i_nngp];
        if (obs_focal < 0 || obs_focal >= n_spatial) continue;
        double v_i = cv[obs_focal];
        if (!(v_i > 0.0)) continue;
        double tau_i = 1.0 / v_i;
        int idx_i = gp_start + obs_focal;

        // Same left-packed scan batch_nngp_scatter counted `alpha`'s columns
        // with, so a_row[k] is the coefficient that was written for column k.
        const int n_row = tulpa_nngp::nngp_row_neighbours(
            &nn_idx(i_nngp, 0), /*stride=*/nn_idx.nrow(), nn, n_spatial);
        int n_nb = 0;
        const double* arow = alpha.data() + static_cast<size_t>(i_nngp) * nn;
        for (int k = 0; k < n_row; k++) {
            int obs_k = nn_order[nn_idx(i_nngp, k) - 1];
            if (obs_k < 0 || obs_k >= n_spatial) break;
            nb_obs[n_nb] = obs_k;
            a_row[n_nb] = arow[k];
            n_nb++;
        }

        double q_i = w_block[obs_focal];
        for (int k = 0; k < n_nb; k++) q_i -= a_row[k] * w_block[nb_obs[k]];

        grad[idx_i] -= q_i * tau_i;
        for (int k = 0; k < n_nb; k++) {
            grad[gp_start + nb_obs[k]] += a_row[k] * q_i * tau_i;
        }

        H.add(idx_i, idx_i, tau_i);
        for (int k = 0; k < n_nb; k++) {
            int idx_k = gp_start + nb_obs[k];
            double a_k = a_row[k];
            H.add(idx_i, idx_k, -a_k * tau_i);  // builder symmetrises
            double ak_tau = a_k * tau_i;
            // The builder stores only the lower triangle (stype = -1) and
            // normalises (row, col) to (max, min). Iterating all (k, kp)
            // pairs would hit each off-diagonal slot twice — once as
            // (k, kp) and once as (kp, k) — doubling the value. Walk only
            // the unique pairs with kp <= k.
            for (int kp = 0; kp <= k; kp++) {
                int idx_kp = gp_start + nb_obs[kp];
                H.add(idx_k, idx_kp, ak_tau * a_row[kp]);
            }
        }
    }
}

// =====================================================================
// NNGP sparsity pattern: emits the (row, col) pairs needed by
// SparseHessianBuilder to represent the full NNGP precision matrix.
// Pushes pairs into `pattern` (the builder dedups + sorts).
inline void make_nngp_prior_sparsity_pattern(
    std::vector<std::pair<int,int>>& pattern,
    const Rcpp::IntegerMatrix& nn_idx,
    const Rcpp::IntegerVector& nn_order,
    int n_spatial, int nn, int gp_start
) {
    for (int i_nngp = 0; i_nngp < n_spatial; i_nngp++) {
        int obs_focal = nn_order[i_nngp];
        if (obs_focal < 0 || obs_focal >= n_spatial) continue;
        int idx_i = gp_start + obs_focal;
        pattern.push_back({idx_i, idx_i});
        // Same left-packed scan the scatter runs, so the pattern covers exactly
        // the pairs the scatter writes into.
        const int n_row = tulpa_nngp::nngp_row_neighbours(
            &nn_idx(i_nngp, 0), /*stride=*/nn_idx.nrow(), nn, n_spatial);
        for (int k = 0; k < n_row; k++) {
            int obs_k = nn_order[nn_idx(i_nngp, k) - 1];
            if (obs_k < 0 || obs_k >= n_spatial) break;
            int idx_k = gp_start + obs_k;
            pattern.push_back({idx_i, idx_k});
            pattern.push_back({idx_k, idx_k});
            for (int kp = 0; kp < k; kp++) {
                int obs_kp = nn_order[nn_idx(i_nngp, kp) - 1];
                if (obs_kp < 0 || obs_kp >= n_spatial) break;
                int idx_kp = gp_start + obs_kp;
                pattern.push_back({idx_k, idx_kp});
            }
        }
    }
}

} // namespace tulpa

#endif // TULPA_GPU_NNGP_LAPLACE_H
