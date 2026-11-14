// gpu_nngp_laplace.cpp
// Definitions of the NNGP prior scatter declared in gpu_nngp_laplace.h.

#include <RcppEigen.h>
#include "gpu_nngp_laplace.h"

namespace tulpa {

void batch_nngp_scatter(
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
    std::vector<double>* alpha_out) {
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
    // A location conditioned on nothing has the marginal variance, with the
    // nugget the neighbour blocks carry on every diagonal entry.
    cond_var_out.assign(n_spatial, sigma2 + tulpa_linalg::kNngpNugget);
    if (alpha_out) alpha_out->assign(static_cast<size_t>(n_spatial) * nn, 0.0);
    gpu_used = false;

    // Phase 1: Build all C matrices and c vectors on CPU
    struct LocData {
        int nngp_idx;     // index in NNGP ordering
        int obs_idx;      // original location index
        int n_nb;         // actual neighbor count
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

        locs.push_back({i, obs_idx, n_nb});
        C_mats.push_back(std::move(C));
        c_vecs.push_back(std::move(c));
    }

    int batch_size = static_cast<int>(locs.size());
    if (batch_size == 0) return;

    auto policy = device::current_policy();
    std::vector<int> widths;
    widths.reserve(locs.size());
    for (const auto& loc : locs) widths.push_back(loc.n_nb);
    const auto status = device::cholesky_batch(C_mats, nn, *policy, gpu_used, &widths);
    if (!status) {
        policy->fail(status);
        return;
    }

    // Phase 3: Solve and extract conditional mean/variance (CPU, O(m²) per loc)
    for (int b = 0; b < batch_size; b++) {
        auto& L = C_mats[b];
        auto& c = c_vecs[b];
        int n_nb = locs[b].n_nb;
        int obs_idx = locs[b].obs_idx;
        int nngp_idx = locs[b].nngp_idx;

        // Gather neighbor values. Every slot below n_nb resolves inside
        // nn_order by construction (nngp_row_neighbours bounded it) and
        // nn_order's own range was checked at entry.
        std::vector<double> w_nb(n_nb, 0.0);
        for (int j = 0; j < n_nb; j++) {
            w_nb[j] = w[nn_order[nn_idx(nngp_idx, j) - 1]];
        }

        std::vector<double> alpha(n_nb);
        tulpa_linalg::nngp_moments_from_chol<tulpa_linalg::TriLayout::RowMajor>(
            L.data(), n_nb, nn, c.data(), w_nb.data(),
            sigma2 + tulpa_linalg::kNngpNugget, tulpa_linalg::kNngpVarFloor,
            cond_mean_out[obs_idx], cond_var_out[obs_idx], alpha.data());

        if (alpha_out) {
            double* row = alpha_out->data() + static_cast<size_t>(nngp_idx) * nn;
            for (int j = 0; j < n_nb; j++) row[j] = alpha[j];
        }
    }
}

void make_nngp_prior_sparsity_pattern(
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
