// scatter_st_cache.h
// Stage 2 follow-up: scatter index cache for the spatial × indexed-temporal
// (ST) sparse path. Same lookup-elimination idea as
// scatter_indexed_cache.h, adapted to the ST layout in
// nl_scatter_obs_spatial_x_indexed_temporal_sparse.
//
// Latent layout (see build_st_hessian_pattern in nested_laplace.cpp):
//   [beta(p)] [re(n_re_groups)] [w_spatial(s_block_len)] [w_temporal(n_t)]
//
// Per-obs scatter touches:
//   * β / β lower triangle  (p*(p+1)/2)
//   * β / RE                (p when this obs has a RE group)
//   * RE / RE diagonal      (1 when this obs has a RE group)
//   * spatial diag          (S_i = obs_p[i+1] - obs_p[i])
//   * spatial / spatial     (S_i*(S_i-1)/2 within obs)
//   * spatial × β / RE      (S_i * p,   S_i)
//   * spatial × temporal    (S_i, when this obs has a temporal index)
//   * temporal diag         (1, when has_t)
//   * temporal × β / RE     (p,  1)
//
// All flat values[] indices for these entries are determined by the
// fit-time pattern; only the per-iter scatter values change. The cache
// stores every index up-front so the inner scatter writes via
// `values[idx] += val` and skips the per-write std::map lookup that
// dominates ICAR/BYM2 scatter wall-time on the joint path.
//
// Cache validity is keyed on (H_builder ptr, H.nnz, p, n_re_groups,
// s_start, t_start, n_t, N). The ST runner builds the cache once at
// fit-time alongside the H pattern and reuses it across every outer-grid
// cell.

#ifndef TULPA_SCATTER_ST_CACHE_H
#define TULPA_SCATTER_ST_CACHE_H

#include "laplace_family_link.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <cstddef>
#include <vector>

namespace tulpa {

struct STScatterIndexCache {
    bool enabled = false;

    // ---- Validity ----
    const void* cache_H_ptr  = nullptr;
    int         cache_H_nnz  = -1;
    int         cache_p      = -1;
    int         cache_n_re   = -1;
    int         cache_s_start = -1;
    int         cache_t_start = -1;
    int         cache_n_t     = -1;
    int         cache_N       = -1;

    // ---- Fixed per-fit indices ----
    // β/β lower triangle, column-major (l outer, j inner over [l, p)).
    // Length p*(p+1)/2.
    std::vector<int> idx_bb;
    // RE/RE diagonal (one entry per RE group). Length n_re_groups.
    std::vector<int> idx_re_diag;
    // β × RE, layout [g * p + j]. Length n_re_groups * p.
    std::vector<int> idx_beta_re;
    // β × temporal (per-t, length p) — used only when an obs has has_t.
    // Layout [t_local * p + j] for t_local in [0, n_t). Length n_t * p.
    std::vector<int> idx_beta_t;
    // RE × temporal: layout [t_local * n_re_groups + g]. Length n_t * n_re_groups.
    // Only entries hit by some obs with both a temporal index and that RE
    // group are non-(-1); others stay -1 because the pattern omits them.
    std::vector<int> idx_re_t;
    // Temporal diagonal. Length n_t.
    std::vector<int> idx_t_diag;

    // ---- Per-obs records ----
    // g_re_global = -1 when this obs has no RE group; otherwise the global
    // RE-block index (p + gi). Length N.
    std::vector<int> g_re_global;
    // idx_t_global = -1 when this obs has no temporal index; otherwise
    // t_start + t. Length N.
    std::vector<int> idx_t_global;
    // For each obs i, the spatial entries [obs_p[i] .. obs_p[i+1]) carry
    // S_i spatial active dofs. The cache stores the flat indices for all
    // per-spatial-dof entries in concatenated arrays sized to obs_p[N].
    //
    // Layout (sized to total_S = obs_p[N]):
    //   idx_s_diag[c]        — spatial diag (s_start+s, s_start+s)
    //   idx_s_t   [c]        — spatial × temporal cross (only when obs has_t)
    //   idx_s_re  [c]        — spatial × RE  cross (only when obs has g_re)
    //
    // Spatial × spatial cross (within obs): per obs i with S_i spatial
    // dofs, S_i*(S_i-1)/2 entries. Stored in idx_s_s, indexed by a per-
    // obs prefix in obs_ss_off (length N+1; obs_ss_off[N] = total_ss).
    std::vector<int> idx_s_diag;
    std::vector<int> idx_s_t;
    std::vector<int> idx_s_re;
    std::vector<int> idx_s_s;
    std::vector<int> obs_ss_off;   // length N + 1

    // β × spatial cross: per obs S_i * p entries. Layout per obs:
    //   [j * S_i + e] for j in [0, p), e in [0, S_i)
    // Stored in idx_beta_s, indexed by obs_bs_off (length N + 1).
    std::vector<int> idx_beta_s;
    std::vector<int> obs_bs_off;   // length N + 1
};

// Build the cache. Must be called AFTER build_st_hessian_pattern (so H
// has its final entry_map). obs_p/obs_local_idx come straight from the
// SpatialBlockOps bundle.
inline void build_st_scatter_index_cache(
    int N, int p, int n_re_groups,
    const Rcpp::NumericVector&   re_idx,
    int s_start,
    const std::vector<int>&      obs_p,
    const std::vector<int>&      obs_local_idx,
    int t_start, int n_t,
    const Rcpp::IntegerVector&   t_idx,
    const SparseHessianBuilder&  H,
    STScatterIndexCache&         cache
) {
    cache.enabled       = true;
    cache.cache_H_ptr   = static_cast<const void*>(&H);
    cache.cache_H_nnz   = H.nnz;
    cache.cache_p       = p;
    cache.cache_n_re    = n_re_groups;
    cache.cache_s_start = s_start;
    cache.cache_t_start = t_start;
    cache.cache_n_t     = n_t;
    cache.cache_N       = N;

    // β/β lower triangle, column-major (l outer, j inner over [l, p)).
    cache.idx_bb.resize(static_cast<size_t>(p) * (p + 1) / 2);
    {
        int t = 0;
        for (int l = 0; l < p; l++) {
            for (int j = l; j < p; j++) {
                cache.idx_bb[t++] = H.lookup(j, l);
            }
        }
    }

    // RE diagonal.
    cache.idx_re_diag.resize(static_cast<size_t>(n_re_groups));
    for (int g = 0; g < n_re_groups; g++) {
        cache.idx_re_diag[g] = H.lookup(p + g, p + g);
    }

    // β × RE: [g * p + j].
    cache.idx_beta_re.resize(static_cast<size_t>(n_re_groups) * p);
    for (int g = 0; g < n_re_groups; g++) {
        const int c = p + g;
        for (int j = 0; j < p; j++) {
            cache.idx_beta_re[static_cast<size_t>(g) * p + j] =
                H.lookup(c, j);
        }
    }

    // Temporal diag.
    cache.idx_t_diag.resize(static_cast<size_t>(n_t));
    for (int t = 0; t < n_t; t++) {
        const int idx = t_start + t;
        cache.idx_t_diag[t] = H.lookup(idx, idx);
    }

    // β × temporal: [t * p + j].
    cache.idx_beta_t.resize(static_cast<size_t>(n_t) * p);
    for (int t = 0; t < n_t; t++) {
        const int r = t_start + t;
        for (int j = 0; j < p; j++) {
            cache.idx_beta_t[static_cast<size_t>(t) * p + j] =
                H.lookup(r, j);
        }
    }

    // RE × temporal: [t * n_re + g]. Pattern may not include all
    // combinations; -1 entries are silently dropped at scatter time.
    cache.idx_re_t.resize(static_cast<size_t>(n_t) * n_re_groups);
    for (int t = 0; t < n_t; t++) {
        const int idx_t = t_start + t;
        for (int g = 0; g < n_re_groups; g++) {
            cache.idx_re_t[static_cast<size_t>(t) * n_re_groups + g] =
                H.lookup(idx_t, p + g);
        }
    }

    // Per-obs records.
    cache.g_re_global.assign(N, -1);
    cache.idx_t_global.assign(N, -1);
    cache.obs_bs_off.assign(N + 1, 0);
    cache.obs_ss_off.assign(N + 1, 0);

    for (int i = 0; i < N; i++) {
        if (n_re_groups > 0) {
            int gi = static_cast<int>(re_idx[i]) - 1;
            if (gi >= 0 && gi < n_re_groups) cache.g_re_global[i] = p + gi;
        }
        int t = t_idx[i] - 1;
        if (t >= 0 && t < n_t) cache.idx_t_global[i] = t_start + t;

        const int S_i = obs_p[i + 1] - obs_p[i];
        cache.obs_bs_off[i + 1] = cache.obs_bs_off[i]
            + static_cast<int>(S_i) * p;
        cache.obs_ss_off[i + 1] = cache.obs_ss_off[i]
            + static_cast<int>(S_i) * static_cast<int>(S_i - 1) / 2;
    }

    const int total_S  = (N > 0) ? obs_p[N] : 0;
    const int total_bs = cache.obs_bs_off[N];
    const int total_ss = cache.obs_ss_off[N];

    cache.idx_s_diag.assign(total_S, -1);
    cache.idx_s_t   .assign(total_S, -1);
    cache.idx_s_re  .assign(total_S, -1);
    cache.idx_s_s   .assign(total_ss, -1);
    cache.idx_beta_s.assign(total_bs, -1);

    for (int i = 0; i < N; i++) {
        const int s_beg = obs_p[i];
        const int s_end = obs_p[i + 1];
        const int S_i   = s_end - s_beg;
        if (S_i == 0) continue;

        const int idx_t_glb = cache.idx_t_global[i];
        const int g_re_glb  = cache.g_re_global[i];
        const int bs_off    = cache.obs_bs_off[i];
        const int ss_off    = cache.obs_ss_off[i];

        // First: lookup all spatial dof globals for this obs (S_i entries).
        // Used both for diag and the within-obs cross loop.
        // No need to store these explicitly — they reproduce as
        //   s_start + obs_local_idx[s_beg + e]
        // and we only need indices here.

        // Spatial diag + s × t + s × RE.
        for (int e = 0; e < S_i; e++) {
            const int idx_a = s_start + obs_local_idx[s_beg + e];
            cache.idx_s_diag[s_beg + e] = H.lookup(idx_a, idx_a);
            if (idx_t_glb >= 0) {
                cache.idx_s_t[s_beg + e] = H.lookup(idx_a, idx_t_glb);
            }
            if (g_re_glb >= 0) {
                cache.idx_s_re[s_beg + e] = H.lookup(idx_a, g_re_glb);
            }
        }

        // β × spatial: layout [j * S_i + e].
        for (int j = 0; j < p; j++) {
            for (int e = 0; e < S_i; e++) {
                const int idx_a = s_start + obs_local_idx[s_beg + e];
                cache.idx_beta_s[bs_off + j * S_i + e] = H.lookup(idx_a, j);
            }
        }

        // Spatial × spatial within obs (upper triangle e1 < e2):
        // layout row-major over (e1, e2), e1 in [0, S_i-1), e2 in [e1+1, S_i).
        int t = 0;
        for (int e1 = 0; e1 < S_i; e1++) {
            const int idx_a = s_start + obs_local_idx[s_beg + e1];
            for (int e2 = e1 + 1; e2 < S_i; e2++) {
                const int idx_b = s_start + obs_local_idx[s_beg + e2];
                cache.idx_s_s[ss_off + t++] = H.lookup(
                    std::max(idx_a, idx_b), std::min(idx_a, idx_b));
            }
        }
    }
}

inline bool st_scatter_index_cache_valid(
    const STScatterIndexCache&  cache,
    const SparseHessianBuilder& H,
    int N, int p, int n_re_groups, int s_start, int t_start, int n_t
) {
    if (!cache.enabled) return false;
    if (cache.cache_H_ptr   != static_cast<const void*>(&H)) return false;
    if (cache.cache_H_nnz   != H.nnz)         return false;
    if (cache.cache_p       != p)             return false;
    if (cache.cache_n_re    != n_re_groups)   return false;
    if (cache.cache_s_start != s_start)       return false;
    if (cache.cache_t_start != t_start)       return false;
    if (cache.cache_n_t     != n_t)           return false;
    if (cache.cache_N       != N)             return false;
    return true;
}

// Cached ST scatter. Mirrors nl_scatter_obs_spatial_x_indexed_temporal_sparse
// entry-for-entry, but replaces every H.add() with a direct
// values[idx] += val using flat indices from STScatterIndexCache.
inline void nl_scatter_obs_spatial_x_indexed_temporal_cached(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X,
    int N, int p, int n_re_groups,
    const Rcpp::NumericVector& eta,
    const std::string& family, double phi,
    int s_start,
    const std::vector<int>&    obs_p,
    const std::vector<int>&    obs_local_idx,
    const std::vector<double>& obs_weight,
    const STScatterIndexCache& cache,
    DenseVec& grad, SparseHessianBuilder& H
) {
    double* __restrict__       Hv  = H.values.data();
    const int* __restrict__    bb  = cache.idx_bb.data();
    const int* __restrict__    bre = cache.idx_beta_re.data();
    const int* __restrict__    red = cache.idx_re_diag.data();
    const int* __restrict__    bt  = cache.idx_beta_t.data();
    const int* __restrict__    ret = cache.idx_re_t.data();
    const int* __restrict__    td  = cache.idx_t_diag.data();
    const int* __restrict__    sd  = cache.idx_s_diag.data();
    const int* __restrict__    st  = cache.idx_s_t.data();
    const int* __restrict__    sr  = cache.idx_s_re.data();
    const int* __restrict__    ss  = cache.idx_s_s.data();
    const int* __restrict__    bs  = cache.idx_beta_s.data();
    const int* __restrict__    g_re_arr  = cache.g_re_global.data();
    const int* __restrict__    idx_t_arr = cache.idx_t_global.data();
    const int* __restrict__    bs_off    = cache.obs_bs_off.data();
    const int* __restrict__    ss_off    = cache.obs_ss_off.data();

    for (int i = 0; i < N; i++) {
        auto gh = grad_hess_for_family(y[i], n_trials[i], eta[i], family, phi);

        // β block: gradient + β/β diagonal lower triangle (column-major).
        for (int j = 0; j < p; j++) {
            const double Xij = X(i, j);
            grad[j] += gh.grad * Xij;
        }
        {
            int t = 0;
            for (int l = 0; l < p; l++) {
                const double Xil = X(i, l);
                for (int j = l; j < p; j++) {
                    const int k = bb[t++];
                    if (k >= 0) Hv[k] += gh.neg_hess * X(i, j) * Xil;
                }
            }
        }

        const int g_re   = g_re_arr[i];
        const int idx_t  = idx_t_arr[i];
        // β × RE + RE diag + RE grad.
        if (g_re >= 0) {
            grad[g_re] += gh.grad;
            const int g_local = g_re - p;
            const int k_re_diag = red[g_local];
            if (k_re_diag >= 0) Hv[k_re_diag] += gh.neg_hess;
            const int row_off = g_local * p;
            for (int j = 0; j < p; j++) {
                const int k = bre[row_off + j];
                if (k >= 0) Hv[k] += gh.neg_hess * X(i, j);
            }
        }

        const int s_beg = obs_p[i];
        const int s_end = obs_p[i + 1];
        const int S_i   = s_end - s_beg;
        const bool has_s = (S_i > 0);
        const bool has_t = (idx_t >= 0);
        if (!has_s && !has_t) continue;

        // Spatial: diagonal + within-spatial cross + β/RE/temporal cross +
        // gradient.
        if (has_s) {
            const int bs_o = bs_off[i];
            const int ss_o = ss_off[i];
            for (int e = 0; e < S_i; e++) {
                const double w_a   = obs_weight[s_beg + e];
                const int    idx_a = s_start + obs_local_idx[s_beg + e];
                // spatial gradient (only x-loc that needs the global dof)
                grad[idx_a] += gh.grad * w_a;
                // spatial diagonal
                const int k_diag = sd[s_beg + e];
                if (k_diag >= 0) Hv[k_diag] += gh.neg_hess * w_a * w_a;
                // β × spatial
                for (int j = 0; j < p; j++) {
                    const int k = bs[bs_o + j * S_i + e];
                    if (k >= 0) Hv[k] += gh.neg_hess * w_a * X(i, j);
                }
                // RE × spatial
                if (g_re >= 0) {
                    const int k = sr[s_beg + e];
                    if (k >= 0) Hv[k] += gh.neg_hess * w_a;
                }
                // temporal × spatial
                if (has_t) {
                    const int k = st[s_beg + e];
                    if (k >= 0) Hv[k] += gh.neg_hess * w_a;
                }
            }
            // Within-spatial cross (upper triangle e1 < e2): row-major.
            {
                int t = 0;
                for (int e1 = 0; e1 < S_i; e1++) {
                    const double w_a = obs_weight[s_beg + e1];
                    for (int e2 = e1 + 1; e2 < S_i; e2++) {
                        const double w_b = obs_weight[s_beg + e2];
                        const int k = ss[ss_o + t++];
                        if (k >= 0) Hv[k] += gh.neg_hess * w_a * w_b;
                    }
                }
            }
        }

        // Temporal-only diag + β × temporal + RE × temporal + gradient.
        if (has_t) {
            grad[idx_t] += gh.grad;
            const int t_local = idx_t - cache.cache_t_start;
            const int k_t_diag = td[t_local];
            if (k_t_diag >= 0) Hv[k_t_diag] += gh.neg_hess;
            const int row_off = t_local * p;
            for (int j = 0; j < p; j++) {
                const int k = bt[row_off + j];
                if (k >= 0) Hv[k] += gh.neg_hess * X(i, j);
            }
            if (g_re >= 0) {
                const int g_local = g_re - p;
                const int k = ret[static_cast<size_t>(t_local) * cache.cache_n_re + g_local];
                if (k >= 0) Hv[k] += gh.neg_hess;
            }
        }
    }
}

} // namespace tulpa

#endif // TULPA_SCATTER_ST_CACHE_H
