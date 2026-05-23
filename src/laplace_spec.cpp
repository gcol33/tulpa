// laplace_spec.cpp
// LikelihoodSpec-driven Laplace mode finder.
//
// Mirrors the laplace_mode_dense pattern in laplace_core.cpp but routes
// the per-observation log-likelihood and IRLS weights through a
// LikelihoodSpec rather than the family-enum dispatch in
// laplace_family_link.h. Lets a downstream package (tulpaGlmm,
// tulpaObs, tulpaRatio, ...) pin its own log-likelihood through Laplace
// without adding a family enum to tulpa for every new family it ships.
//
// Covers n_processes >= 1 with multi-term, multi-coefficient (random
// slope) RE structure. Each term t has q_t = re_n_coefs[t] coefficients
// per group (q_t == 1 for `(1|g)`, q_t > 1 for slopes). Per-obs RE
// contribution at observation i, into process k whose
// data.sharing.re[k] is true:
//
//     eta_k_i += sum_t  z_{t,i}^T b_{t, g_t(i)}
//
// where z_{t,i,0} = 1 (intercept) and z_{t,i,c} for c >= 1 is read from
// data.re_slope_matrices[t][i*(q_t-1) + (c-1)]. Prior:
//   uncorrelated (`(x||g)`):  Σ_t = diag(σ_{t,0}², ..., σ_{t,q_t-1}²)
//   correlated   (`(x|g)`) :  Σ_t = D_t L_t L_t^T D_t with D_t = diag(σ_t)
//                             and L_t parameterized via tanh of
//                             chol_re_start_multi[t] : chol_re_end_multi[t]
//                             (matches tulpa_priors_re.h conventions).
//
// Single-process / single-term / intercept-only is the trivial reduction
// (n_processes == 1, n_re_terms == 1, q_0 == 1) and stays bit-identical
// to the previous single-term code path.

#include "tulpa/likelihood.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include "laplace_cholesky.h"
#include "laplace_cholesky_dispatch.h"
#include "laplace_newton_loop.h"
#include "linalg_fast.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

namespace {

// Layout helper: sub-vector that Newton actually moves on. For n_processes
// >= 1 the latent slice is the concatenation of every process's beta block
// followed by every RE term's block in term order. Extra parameters
// (log_sigma_re_*, dispersion, ZI betas, ...) are held fixed at their
// input value — Laplace integrates over the latent field, not the
// hyperparameters.
//
// `latent_offset[k]` gives the offset in the n_x-long latent vector where
// process k's beta block lives. `latent_offset[np]` is where the first RE
// term block starts (or n_x when no RE). Each RE term t occupies a
// q_t * G_t span at `re_terms[t].latent_offset` in the latent vector
// and at `re_terms[t].param_start` in the parameter vector.
struct ReTermSlot {
    int n_coefs = 1;            // q_t (1 for intercept-only)
    int n_groups = 0;           // G_t
    int param_start = -1;       // absolute idx into params (== layout.re_start_multi[t])
    int latent_offset = 0;      // offset into the n_x-long latent vec
    bool correlated = false;    // (x|g) when true and n_coefs > 1
    int chol_start = -1;        // absolute idx into params for tanh-Cholesky raw values
    std::vector<int> sigma_slots; // length n_coefs; absolute idxs
};

struct SpecLatentLayout {
    int np = 1;
    std::vector<int> beta_start;        // [np] absolute index into params
    std::vector<int> beta_count;        // [np] block size
    std::vector<int> latent_offset;     // [np + 1] prefix into n_x for beta blocks

    bool has_re = false;
    std::vector<ReTermSlot> re_terms;   // empty when no RE; size = K otherwise
    int n_x = 0;
};

// Build the per-Laplace SpecLatentLayout from the (already-populated)
// ParamLayout + ModelData. Reads multi-term RE fields when present and
// falls back to the legacy single-term fields when n_re_terms == 0.
inline SpecLatentLayout build_latent_layout(
    const ModelData& data,
    const ParamLayout& layout
) {
    SpecLatentLayout L;
    if (layout.process_beta_start.empty()) {
        Rcpp::stop("laplace_spec_dense: ParamLayout.process_beta_start is empty");
    }
    L.np = data.n_processes;
    if (L.np <= 0) {
        Rcpp::stop("laplace_spec_dense: requires n_processes >= 1 (got %d)", L.np);
    }
    if ((int)layout.process_beta_start.size() < L.np ||
        (int)layout.process_beta_count.size() < L.np) {
        Rcpp::stop("laplace_spec_dense: ParamLayout has %d/%d process_beta blocks "
                   "but data.n_processes == %d",
                   (int)layout.process_beta_start.size(),
                   (int)layout.process_beta_count.size(), L.np);
    }
    L.beta_start.assign(L.np, 0);
    L.beta_count.assign(L.np, 0);
    L.latent_offset.assign(L.np + 1, 0);
    int running = 0;
    for (int k = 0; k < L.np; k++) {
        L.beta_start[k] = layout.process_beta_start[k];
        L.beta_count[k] = layout.process_beta_count[k];
        L.latent_offset[k] = running;
        running += L.beta_count[k];
    }
    L.latent_offset[L.np] = running;

    L.has_re = layout.has_re;
    if (L.has_re) {
        // n_terms unification. data.n_re_terms is the canonical count when
        // populated by the modern populate_re; legacy single-term callers
        // leave it at 0 but set data.n_re_groups + layout.re_start.
        const int n_terms_unified = (data.n_re_terms > 0) ? data.n_re_terms : 1;
        L.re_terms.resize(n_terms_unified);

        for (int t = 0; t < n_terms_unified; t++) {
            ReTermSlot& s = L.re_terms[t];

            // n_coefs: prefer multi-term layout, else fall back to 1.
            if ((int)layout.re_n_coefs_multi.size() > t) {
                s.n_coefs = layout.re_n_coefs_multi[t];
            } else {
                s.n_coefs = 1;
            }
            if (s.n_coefs <= 0) s.n_coefs = 1;

            // n_groups: prefer multi-term data, else legacy n_re_groups.
            if ((int)data.re_n_groups_multi.size() > t) {
                s.n_groups = data.re_n_groups_multi[t];
            } else {
                s.n_groups = data.n_re_groups;
            }

            // param_start: prefer multi-term layout, else legacy re_start.
            if ((int)layout.re_start_multi.size() > t) {
                s.param_start = layout.re_start_multi[t];
            } else {
                s.param_start = layout.re_start;
            }

            // correlated: only meaningful when q_t > 1.
            s.correlated = (s.n_coefs > 1
                            && (int)layout.re_correlated_multi.size() > t
                            && layout.re_correlated_multi[t]);

            // chol_start: only when correlated.
            if (s.correlated && (int)layout.chol_re_start_multi.size() > t) {
                s.chol_start = layout.chol_re_start_multi[t];
            } else {
                s.chol_start = -1;
            }

            // sigma_slots: prefer the per-coef slopes layout, else fall back
            // to the multi-term scalar (q_t == 1) or legacy log_sigma_re_idx.
            s.sigma_slots.assign(s.n_coefs, -1);
            if ((int)layout.log_sigma_re_slopes.size() > t
                && (int)layout.log_sigma_re_slopes[t].size() == s.n_coefs) {
                for (int c = 0; c < s.n_coefs; c++) {
                    s.sigma_slots[c] = layout.log_sigma_re_slopes[t][c];
                }
            } else if ((int)layout.log_sigma_re_multi.size() > t) {
                s.sigma_slots[0] = layout.log_sigma_re_multi[t];
            } else {
                s.sigma_slots[0] = layout.log_sigma_re_idx;
            }

            s.latent_offset = running;
            running += s.n_groups * s.n_coefs;
        }
    }
    L.n_x = running;
    return L;
}

// Pull the per-coefficient sigma values for term t out of the params
// vector. Returns a length-q_t vector of standard deviations.
inline std::vector<double> term_sigmas(
    const std::vector<double>& params,
    const ReTermSlot& s
) {
    std::vector<double> sig(s.n_coefs, 1.0);
    for (int c = 0; c < s.n_coefs; c++) {
        int slot = s.sigma_slots[c];
        sig[c] = (slot < 0) ? 1.0 : std::exp(params[slot]);
    }
    return sig;
}

// Build the q_t x q_t lower-triangular Cholesky factor L_t from the
// tanh-Cholesky raw parameters in `params[chol_start .. chol_start +
// q_t*(q_t-1)/2)`. Mirrors tulpa_priors_re.h::compute_re_prior. Diagonal
// entries are derived as sqrt(1 - sum_off-diag^2) per row, guaranteed
// positive because tanh^2 < 1.
inline void build_chol_L(
    const std::vector<double>& params,
    const ReTermSlot& s,
    std::vector<double>& L_flat   // q x q row-major, lower triangular; resized here
) {
    const int q = s.n_coefs;
    L_flat.assign((size_t)q * q, 0.0);
    int idx = s.chol_start;
    for (int row = 0; row < q; row++) {
        double row_sum_sq = 0.0;
        for (int col = 0; col < row; col++) {
            double l_ij = std::tanh(params[idx++]);
            L_flat[(size_t)row * q + col] = l_ij;
            row_sum_sq += l_ij * l_ij;
        }
        double diag_sq = 1.0 - row_sum_sq;
        if (diag_sq < 1e-12) diag_sq = 1e-12;
        L_flat[(size_t)row * q + row] = std::sqrt(diag_sq);
    }
}

// Build the q_t x q_t precision matrix Q_t = Σ_t^{-1} for term t and its
// log-determinant log|Q_t|. Σ_t = D L L^T D with D = diag(σ_t):
//   uncorrelated: L = I; Q_t = diag(1/σ_{t,c}²); log|Q_t| = -2 sum_c log σ_{t,c}.
//   correlated  : Q_t = D^{-1} L^{-T} L^{-1} D^{-1};
//                 log|Q_t| = -2 sum_c log σ_{t,c} - 2 sum_c log L_{cc}.
// Q is dense q x q (small, q == 2 or 3 in typical use).
inline void build_term_precision(
    const std::vector<double>& sigmas,    // length q
    const std::vector<double>& L_flat,    // empty when uncorrelated; else q*q row-major
    int q,
    bool correlated,
    std::vector<double>& Q_flat,          // out: q*q row-major (resized)
    double& log_det_Q                     // out
) {
    Q_flat.assign((size_t)q * q, 0.0);
    log_det_Q = 0.0;
    if (!correlated || q == 1) {
        for (int c = 0; c < q; c++) {
            double tau_c = 1.0 / (sigmas[c] * sigmas[c] + 1e-300);
            Q_flat[(size_t)c * q + c] = tau_c;
            log_det_Q += std::log(tau_c);
        }
        return;
    }
    // Compute Linv (lower triangular) by forward substitution on the columns
    // of the identity. Linv is q x q row-major (lower triangular).
    std::vector<double> Linv((size_t)q * q, 0.0);
    for (int j = 0; j < q; j++) {
        // Solve L * x = e_j for x, store in Linv column j (i.e. Linv[i,j]).
        std::vector<double> x(q, 0.0);
        x[j] = 1.0 / L_flat[(size_t)j * q + j];
        for (int i = j + 1; i < q; i++) {
            double s = 0.0;
            for (int k = j; k < i; k++) {
                s += L_flat[(size_t)i * q + k] * x[k];
            }
            x[i] = -s / L_flat[(size_t)i * q + i];
        }
        for (int i = 0; i < q; i++) Linv[(size_t)i * q + j] = x[i];
    }
    // Σ^{-1} = (D L L^T D)^{-1} = D^{-1} L^{-T} L^{-1} D^{-1}.
    // Form M = L^{-T} L^{-1} = Linv^T * Linv.
    std::vector<double> M((size_t)q * q, 0.0);
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < q; j++) {
            double s = 0.0;
            for (int k = 0; k < q; k++) {
                s += Linv[(size_t)k * q + i] * Linv[(size_t)k * q + j];
            }
            M[(size_t)i * q + j] = s;
        }
    }
    // Q = D^{-1} M D^{-1}.
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < q; j++) {
            Q_flat[(size_t)i * q + j] = M[(size_t)i * q + j]
                / (sigmas[i] * sigmas[j]);
        }
    }
    // log|Q| = -2 sum_c log σ_c - 2 sum_c log L_{cc}.
    double s = 0.0;
    for (int c = 0; c < q; c++) {
        s += std::log(sigmas[c]);
        s += std::log(L_flat[(size_t)c * q + c]);
    }
    log_det_Q = -2.0 * s;
}

// Resolve the 1-based group index at obs i for term t. Reads the
// multi-term flat layout when populated, else the legacy single-term
// re_group field. Returns -1 if the obs has no group for this term.
inline int re_group_at(
    const ModelData& data,
    int i,
    int t,
    int n_terms_unified
) {
    if (!data.re_group_multi_flat.empty()) {
        int g1 = data.re_group_multi_flat[(size_t)i * n_terms_unified + t];
        return (g1 >= 1) ? (g1 - 1) : -1;
    }
    if (t == 0 && !data.re_group.empty()) {
        int g1 = data.re_group[i];
        return (g1 >= 1) ? (g1 - 1) : -1;
    }
    return -1;
}

// Read the design value z_{t,i,c} for coefficient c of RE term t. When the
// term carries the implicit group intercept (the common case) coef 0 is the
// intercept (z = 1, not stored) and coefs 1..q_t-1 are slopes read from
// data.re_slope_matrices[t] at column c-1. When the term has no intercept
// (lme4 `(0 + x | g)`) every coef 0..q_t-1 is a slope read at column c.
// Returns 0 when the term is intercept-only or the index is out of range.
inline double slope_at(
    const ModelData& data,
    int t,
    int i,
    int c        // coefficient index in 0..q_t-1
) {
    const bool has_int = re_term_has_intercept(data, t);
    if (has_int && c <= 0) return 1.0; // implicit intercept
    const int col = has_int ? (c - 1) : c;
    if (col < 0) return 0.0;
    if ((int)data.re_slope_matrices.size() <= t) return 0.0;
    const auto& M = data.re_slope_matrices[t];
    if (M.empty()) return 0.0;
    if ((int)data.re_n_slopes.size() <= t) return 0.0;
    const int n_slopes = data.re_n_slopes[t];
    if (n_slopes <= 0 || col >= n_slopes) return 0.0;
    return M[(size_t)i * n_slopes + col];
}

// True iff the RE contributes to process k's linear predictor. Falls back
// to "share into every process" when SharingSpec.re hasn't been initialised
// (e.g. legacy callers building ModelData without sharing.init()). Sharing
// is per-process and applied uniformly to all RE terms (matches HMC).
inline bool re_shared_into(const ModelData& data, int k) {
    if ((int)data.sharing.re.size() != data.n_processes) return true;
    return data.sharing.re[k];
}

// Compute the per-obs RE contribution sum_t z_{t,i}^T b_{t, g_t(i)} once,
// and return whether any RE term contributed at this obs (so the caller
// can skip the sharing step entirely when not).
inline double obs_re_contrib(
    const ModelData& data,
    const std::vector<double>& params,
    const SpecLatentLayout& L,
    int i,
    int n_terms_unified,
    bool& used_out
) {
    used_out = false;
    if (!L.has_re) return 0.0;
    double re_eff = 0.0;
    for (int t = 0; t < (int)L.re_terms.size(); t++) {
        int g = re_group_at(data, i, t, n_terms_unified);
        if (g < 0) continue;
        const ReTermSlot& s = L.re_terms[t];
        if (g >= s.n_groups) continue;
        const int q = s.n_coefs;
        const int base = s.param_start + g * q;
        // z_{t,i,c} = 1 for the implicit intercept (when present) and the
        // slope design value otherwise; slope_at() encodes both cases.
        double contrib = 0.0;
        for (int c = 0; c < q; c++) {
            contrib += params[base + c] * slope_at(data, t, i, c);
        }
        re_eff += contrib;
        used_out = true;
    }
    return re_eff;
}

// Build eta_flat for all processes. eta_flat is laid out as [N x np] in
// observation-major order so eta_flat[i*np + k] is process k's linear
// predictor at observation i — the same layout the LikelihoodSpec eta
// pointer expects (one obs at a time, &eta_flat[i*np]).
//
// Per process: eta_k_i = (X_k beta_k)_i + offset_k_i + (RE if shared into k).
inline void compute_eta_spec(
    const ModelData& data,
    const std::vector<double>& params,
    const SpecLatentLayout& L,
    const std::vector<int>& /*re_group_1based*/,
    int N,
    std::vector<double>& eta_flat,
    int n_threads
) {
    const int np = L.np;
    const int n_terms_unified = (data.n_re_terms > 0) ? data.n_re_terms : 1;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
    #endif
    for (int i = 0; i < N; i++) {
        bool re_used = false;
        double re_eff = obs_re_contrib(data, params, L, i, n_terms_unified, re_used);

        for (int k = 0; k < np; k++) {
            const ProcessData& proc = data.processes[k];
            double e = 0.0;
            if (proc.p > 0) {
                const double* row = proc.X_flat.data() + (std::ptrdiff_t)i * proc.p;
                const double* beta = &params[L.beta_start[k]];
                for (int j = 0; j < proc.p; j++) e += row[j] * beta[j];
            }
            if (!proc.offset.empty()) e += proc.offset[i];
            if (re_used && re_shared_into(data, k)) e += re_eff;
            eta_flat[(std::ptrdiff_t)i * np + k] = e;
        }
    }
}

// Per-observation gradient + neg-Hessian assembled into the latent gradient
// and Hessian in (beta_0..beta_{np-1}, re_term_0..re_term_{K-1}) space.
// eta_weights_fn writes
//   grad_eta[k]                       = d log_lik_i / d eta_i_k
//   neg_hess_eta[k * np + l]          = -d^2 log_lik_i / (d eta_i_k d eta_i_l)
// At the end the prior contribution is added: beta ridge + per-term RE
// precision Q_t (computed once from σ_t and L_t).
inline void scatter_spec(
    const std::vector<double>& params,
    const std::vector<double>& eta_flat,
    const std::vector<int>& /*re_group_1based*/,
    const SpecLatentLayout& L,
    const ModelData& data,
    const ParamLayout& layout,
    const LikelihoodSpec& spec,
    const void* response_data,
    int N,
    double /*tau_re_legacy*/,
    DenseVec& grad,
    DenseMat& H,
    int /*n_threads*/
) {
    std::fill(grad.begin(), grad.end(), 0.0);
    H.zero();

    const int np = L.np;
    std::vector<double> grad_eta(np, 0.0);
    std::vector<double> neg_hess_eta((size_t)np * np, 0.0);
    const int K = (int)L.re_terms.size();
    const int n_terms_unified = (data.n_re_terms > 0) ? data.n_re_terms : 1;

    // Per-obs caches reused across the obs loop.
    std::vector<int>     g_term(K, -1);    // 0-based group at obs i for each term
    std::vector<int>     q_term(K, 0);     // q_t cached
    // Slope-row z_{t,i,c} for c = 0..q_t-1 packed contiguously per term.
    std::vector<std::vector<double>> z_term(K);
    for (int t = 0; t < K; t++) z_term[t].assign(L.re_terms[t].n_coefs, 0.0);

    for (int i = 0; i < N; i++) {
        std::fill(grad_eta.begin(), grad_eta.end(), 0.0);
        std::fill(neg_hess_eta.begin(), neg_hess_eta.end(), 0.0);

        spec.eta_weights_fn(
            i, &eta_flat[(std::ptrdiff_t)i * np], 0.0, 0.0,
            params, data, layout, response_data,
            grad_eta.data(), neg_hess_eta.data()
        );

        // Resolve per-term groups + slope rows once.
        for (int t = 0; t < K; t++) {
            const ReTermSlot& s = L.re_terms[t];
            int g = re_group_at(data, i, t, n_terms_unified);
            if (g >= s.n_groups) g = -1;
            g_term[t] = g;
            q_term[t] = s.n_coefs;
            if (g < 0) continue;
            for (int c = 0; c < s.n_coefs; c++) {
                z_term[t][c] = slope_at(data, t, i, c);
            }
        }

        // ============= GRADIENT scatter =============
        // Per-process beta gradient: g_{beta_k_j} += X_k(i, j) * grad_eta[k]
        // Per-term RE gradient at coef c, group g_t(i):
        //   g_{b_{t,g,c}} += z_{t,i,c} · sum_{k: shared} grad_eta[k]
        double s_grad = 0.0;
        for (int k = 0; k < np; k++) {
            const ProcessData& proc = data.processes[k];
            const double gk = grad_eta[k];
            if (proc.p > 0) {
                const double* row = proc.X_flat.data() + (std::ptrdiff_t)i * proc.p;
                double* gbeta = &grad[L.latent_offset[k]];
                for (int j = 0; j < proc.p; j++) gbeta[j] += row[j] * gk;
            }
            if (re_shared_into(data, k)) s_grad += gk;
        }
        for (int t = 0; t < K; t++) {
            int g = g_term[t]; if (g < 0) continue;
            const ReTermSlot& s = L.re_terms[t];
            const int q = q_term[t];
            const int row_base = s.latent_offset + g * q;
            for (int c = 0; c < q; c++) {
                grad[row_base + c] += z_term[t][c] * s_grad;
            }
        }

        // ============= HESSIAN scatter (lower triangle) =============
        // beta × beta block: unchanged.
        for (int k = 0; k < np; k++) {
            const ProcessData& pk = data.processes[k];
            const double* xk = (pk.p > 0)
                ? pk.X_flat.data() + (std::ptrdiff_t)i * pk.p
                : nullptr;
            const int off_k = L.latent_offset[k];

            for (int l = 0; l <= k; l++) {
                const ProcessData& pl = data.processes[l];
                const double* xl = (pl.p > 0)
                    ? pl.X_flat.data() + (std::ptrdiff_t)i * pl.p
                    : nullptr;
                const int off_l = L.latent_offset[l];
                const double w_kl = neg_hess_eta[(size_t)k * np + l];

                if (xk && xl) {
                    if (k == l) {
                        for (int j = 0; j < pk.p; j++) {
                            const double w_xj = w_kl * xk[j];
                            double* row_j = H[off_k + j];
                            for (int m = 0; m <= j; m++) {
                                row_j[off_l + m] += w_xj * xl[m];
                            }
                        }
                    } else {
                        for (int j = 0; j < pk.p; j++) {
                            const double w_xj = w_kl * xk[j];
                            double* row_j = H[off_k + j];
                            for (int m = 0; m < pl.p; m++) {
                                row_j[off_l + m] += w_xj * xl[m];
                            }
                        }
                    }
                }
            }
        }

        // RE × beta cross block. For each term t with active group g,
        // for each process l shared into k via the RE: weight w_l =
        // sum_{k: shared} neg_hess_eta[k*np + l]. Then for coef c:
        //   H[re_row(t,g,c), beta_l_m] += z_{t,i,c} · w_l · X_l(i, m)
        // The RE row is always larger than any beta column (latent_offset
        // for RE blocks sits after all beta blocks), so this is lower-tri.
        // Precompute w_l once per obs.
        std::vector<double> w_l_vec(np, 0.0);
        for (int l = 0; l < np; l++) {
            double w_l = 0.0;
            for (int k = 0; k < np; k++) {
                if (!re_shared_into(data, k)) continue;
                w_l += neg_hess_eta[(size_t)k * np + l];
            }
            w_l_vec[l] = w_l;
        }
        for (int t = 0; t < K; t++) {
            int g = g_term[t]; if (g < 0) continue;
            const ReTermSlot& s = L.re_terms[t];
            const int q = q_term[t];
            const int re_row_base = s.latent_offset + g * q;
            for (int c = 0; c < q; c++) {
                const double zc = z_term[t][c];
                if (zc == 0.0) continue;
                double* row = H[re_row_base + c];
                for (int l = 0; l < np; l++) {
                    const ProcessData& pl = data.processes[l];
                    if (pl.p == 0) continue;
                    const double w_l = w_l_vec[l];
                    if (w_l == 0.0) continue;
                    const double* xl =
                        pl.X_flat.data() + (std::ptrdiff_t)i * pl.p;
                    const double zc_w = zc * w_l;
                    const int off_l = L.latent_offset[l];
                    for (int m = 0; m < pl.p; m++) {
                        row[off_l + m] += zc_w * xl[m];
                    }
                }
            }
        }

        // RE × RE block (term × term). For terms t, t' both with active
        // groups, the contribution to H[re_row(t,g,c), re_row(t',g',c')] is
        //   z_{t,i,c} · z_{t',i,c'} · sum_{k,l: shared,shared} w_{i,kl}
        // Compute s_hess once per obs.
        double s_hess = 0.0;
        for (int k = 0; k < np; k++) {
            if (!re_shared_into(data, k)) continue;
            for (int l = 0; l < np; l++) {
                if (!re_shared_into(data, l)) continue;
                s_hess += neg_hess_eta[(size_t)k * np + l];
            }
        }
        if (s_hess != 0.0) {
            for (int t = 0; t < K; t++) {
                int g = g_term[t]; if (g < 0) continue;
                const ReTermSlot& st = L.re_terms[t];
                const int qt = q_term[t];
                const int row_base_t = st.latent_offset + g * qt;
                for (int tp = 0; tp <= t; tp++) {
                    int gp = g_term[tp]; if (gp < 0) continue;
                    const ReTermSlot& stp = L.re_terms[tp];
                    const int qtp = q_term[tp];
                    const int row_base_tp = stp.latent_offset + gp * qtp;
                    for (int c = 0; c < qt; c++) {
                        const double zc = z_term[t][c];
                        if (zc == 0.0) continue;
                        double* row = H[row_base_t + c];
                        if (t == tp) {
                            // Same term, same group? Block is on the
                            // diagonal of the H lower triangle and we
                            // need cp <= c.
                            if (g == gp) {
                                for (int cp = 0; cp <= c; cp++) {
                                    row[row_base_tp + cp] +=
                                        zc * z_term[tp][cp] * s_hess;
                                }
                            } else if (gp < g) {
                                // Same term, gp < g: full q_tp row.
                                for (int cp = 0; cp < qtp; cp++) {
                                    row[row_base_tp + cp] +=
                                        zc * z_term[tp][cp] * s_hess;
                                }
                            }
                            // gp > g: would be upper triangle for same
                            // term; symmetrise pass handles it.
                        } else {
                            // Different term, tp < t: we are below the
                            // diagonal block by construction.
                            for (int cp = 0; cp < qtp; cp++) {
                                row[row_base_tp + cp] +=
                                    zc * z_term[tp][cp] * s_hess;
                            }
                        }
                    }
                }
            }
        }
    }

    // Symmetrise lower → upper triangle.
    for (int j = 0; j < L.n_x; j++) {
        for (int k = j + 1; k < L.n_x; k++) {
            H[j][k] = H[k][j];
        }
    }

    // beta prior: N(0, sigma_beta^2 I), applied per process.
    double tau_beta = 1.0 / (data.sigma_beta * data.sigma_beta + 1e-300);
    for (int k = 0; k < np; k++) {
        const int off_k = L.latent_offset[k];
        for (int j = 0; j < L.beta_count[k]; j++) {
            grad[off_k + j] += -tau_beta * params[L.beta_start[k] + j];
            H[off_k + j][off_k + j] += tau_beta;
        }
    }

    // RE prior: per term build Q_t (q_t × q_t) and apply to each group.
    if (!L.re_terms.empty()) {
        std::vector<double> L_flat;
        std::vector<double> Q_flat;
        double log_det_Q_unused = 0.0;
        for (int t = 0; t < K; t++) {
            const ReTermSlot& s = L.re_terms[t];
            const int q = s.n_coefs;
            std::vector<double> sigmas = term_sigmas(params, s);
            if (s.correlated && q > 1) {
                build_chol_L(params, s, L_flat);
            } else {
                L_flat.clear();
            }
            build_term_precision(sigmas, L_flat, q, s.correlated && q > 1,
                                 Q_flat, log_det_Q_unused);
            for (int g = 0; g < s.n_groups; g++) {
                const int base_lat   = s.latent_offset + g * q;
                const int base_param = s.param_start  + g * q;
                // grad += -Q b
                for (int c = 0; c < q; c++) {
                    double acc = 0.0;
                    for (int cp = 0; cp < q; cp++) {
                        acc += Q_flat[(size_t)c * q + cp]
                               * params[base_param + cp];
                    }
                    grad[base_lat + c] += -acc;
                }
                // H += Q on the q×q sub-block
                for (int c = 0; c < q; c++) {
                    for (int cp = 0; cp < q; cp++) {
                        H[base_lat + c][base_lat + cp]
                            += Q_flat[(size_t)c * q + cp];
                    }
                }
            }
        }
    }
}

inline double total_log_lik_spec(
    const std::vector<double>& params,
    const std::vector<double>& eta_flat,
    const ModelData& data,
    const ParamLayout& layout,
    const LikelihoodSpec& spec,
    const void* response_data,
    int N
) {
    const int np = data.n_processes;
    double ll = 0.0;
    for (int i = 0; i < N; i++) {
        ll += spec.ll_double(
            i, &eta_flat[(std::ptrdiff_t)i * np], 0.0, 0.0,
            params, data, layout, response_data
        );
    }
    return ll;
}

inline double log_prior_latent(
    const std::vector<double>& params,
    const SpecLatentLayout& L,
    double sigma_beta,
    double /*tau_re_legacy*/
) {
    double lp = 0.0;
    double tau_beta = 1.0 / (sigma_beta * sigma_beta + 1e-300);
    int p_total = 0;
    for (int k = 0; k < L.np; k++) {
        for (int j = 0; j < L.beta_count[k]; j++) {
            double b = params[L.beta_start[k] + j];
            lp += -0.5 * tau_beta * b * b;
        }
        p_total += L.beta_count[k];
    }
    lp += 0.5 * p_total * std::log(tau_beta / (2.0 * M_PI));

    if (!L.re_terms.empty()) {
        std::vector<double> L_flat, Q_flat;
        double log_det_Q = 0.0;
        for (int t = 0; t < (int)L.re_terms.size(); t++) {
            const ReTermSlot& s = L.re_terms[t];
            const int q = s.n_coefs;
            std::vector<double> sigmas = term_sigmas(params, s);
            if (s.correlated && q > 1) {
                build_chol_L(params, s, L_flat);
            } else {
                L_flat.clear();
            }
            build_term_precision(sigmas, L_flat, q, s.correlated && q > 1,
                                 Q_flat, log_det_Q);
            for (int g = 0; g < s.n_groups; g++) {
                const int base = s.param_start + g * q;
                // -0.5 b^T Q b
                double quad = 0.0;
                for (int c = 0; c < q; c++) {
                    double acc = 0.0;
                    for (int cp = 0; cp < q; cp++) {
                        acc += Q_flat[(size_t)c * q + cp] * params[base + cp];
                    }
                    quad += params[base + c] * acc;
                }
                lp += -0.5 * quad;
            }
            // log normalisation: 0.5 G_t log|Q_t| - 0.5 G_t q log(2π)
            lp += 0.5 * (double)s.n_groups * log_det_Q;
            lp += -0.5 * (double)s.n_groups * (double)q * std::log(2.0 * M_PI);
        }
    }
    return lp;
}

} // namespace (anonymous)

// ----------------------------------------------------------------------------
// Public entry. Mirrors laplace_mode_dense's contract but reads the
// log-likelihood and IRLS weights from data.likelihood_spec.
//
//   data, layout: filled exactly as for the NUTS shim (n_processes == 1).
//   params_inout: full parameter vector. On entry, hyperparameter slots
//     (sigma_re, extras) supply their fixed values; latent slots may carry
//     a warm start. On exit, the latent slots are overwritten with the
//     mode; hyperparameter slots are untouched.
//   re_group_1based: per-obs 1-based RE group index. Empty / nullptr if no RE.
//
// Returns the same LaplaceShimResult shape as the family-enum shims so
// downstream code can switch between paths without re-wiring its result
// handling.
// ----------------------------------------------------------------------------
struct LaplaceShimResult;

// Helper: write a trial latent step into the parameter vector. Used by both
// the Newton step and the gradient-only fall-back when factorisation fails.
inline void apply_latent_step(
    const SpecLatentLayout& L,
    const std::vector<double>& base,
    const std::vector<double>& step,
    double scale,
    std::vector<double>& out
) {
    out = base;
    for (int k = 0; k < L.np; k++) {
        const int off_k = L.latent_offset[k];
        for (int j = 0; j < L.beta_count[k]; j++) {
            out[L.beta_start[k] + j] = base[L.beta_start[k] + j]
                                       + scale * step[off_k + j];
        }
    }
    for (int t = 0; t < (int)L.re_terms.size(); t++) {
        const ReTermSlot& s = L.re_terms[t];
        const int n = s.n_groups * s.n_coefs;
        for (int j = 0; j < n; j++) {
            out[s.param_start + j] = base[s.param_start + j]
                                     + scale * step[s.latent_offset + j];
        }
    }
}

void laplace_mode_spec_dense_impl(
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& params_inout,
    const std::vector<int>& re_group_1based,
    int max_iter, double tol, int n_threads,
    int* n_iter_out,
    int* converged_out,
    double* log_det_Q_out,
    double* log_marginal_out
) {
    if (data.n_processes < 1) {
        Rcpp::stop("laplace_spec_dense: requires n_processes >= 1 (got %d)",
                   data.n_processes);
    }
    const LikelihoodSpec* spec =
        static_cast<const LikelihoodSpec*>(data.likelihood_spec);
    if (spec == nullptr) {
        Rcpp::stop("laplace_spec_dense: data.likelihood_spec is null");
    }
    if (spec->eta_weights_fn == nullptr) {
        Rcpp::stop("laplace_spec_dense: LikelihoodSpec.eta_weights_fn is null "
                   "(required for spec-driven Laplace)");
    }
    if (spec->ll_double == nullptr) {
        Rcpp::stop("laplace_spec_dense: LikelihoodSpec.ll_double is null");
    }
    if ((int)data.processes.size() != data.n_processes) {
        Rcpp::stop("laplace_spec_dense: data.processes.size() (%d) != n_processes (%d)",
                   (int)data.processes.size(), data.n_processes);
    }

    SpecLatentLayout L = build_latent_layout(data, layout);
    const int np = L.np;
    for (int k = 0; k < np; k++) {
        const ProcessData& proc = data.processes[k];
        if (proc.p != L.beta_count[k]) {
            Rcpp::stop("laplace_spec_dense: process[%d].p (%d) != "
                       "ParamLayout.process_beta_count[%d] (%d)",
                       k, proc.p, k, L.beta_count[k]);
        }
        if (proc.p > 0 && (int)proc.X_flat.size() != data.N * proc.p) {
            Rcpp::stop("laplace_spec_dense: process[%d] design matrix shape "
                       "(%d) inconsistent with N * p (%d * %d)",
                       k, (int)proc.X_flat.size(), data.N, proc.p);
        }
    }
    // Multi-term path resolution. Three legitimate input shapes:
    //   (a) Modern: data.re_group_multi_flat populated (shape N * n_terms).
    //       This drives the per-term group lookup directly.
    //   (b) Legacy single-term: data.re_group populated. Used when
    //       n_re_terms <= 1 and the caller did not set
    //       re_group_multi_flat.
    //   (c) Shim-only: caller passes a length-N re_group_1based but
    //       neither data.re_group nor data.re_group_multi_flat was
    //       written (tulpaGlmm:glmm_laplace.cpp does both, but we keep
    //       the fallback to be defensive).
    // We synthesize (a) on a local data copy when only (c) is present so
    // the multi-term scatter path always reads from the same place.
    std::unique_ptr<ModelData> data_owned;
    const ModelData* data_ptr = &data;
    if (L.has_re && data.re_group_multi_flat.empty()) {
        const int n_terms_unified = (data.n_re_terms > 0) ? data.n_re_terms : 1;
        if (n_terms_unified == 1
            && (int)data.re_group.size() != data.N
            && (int)re_group_1based.size() == data.N) {
            data_owned.reset(new ModelData(data));
            data_owned->re_group.assign(re_group_1based.begin(),
                                        re_group_1based.end());
            if (data_owned->n_re_groups <= 0) {
                int max_g = 0;
                for (int v : re_group_1based) if (v > max_g) max_g = v;
                data_owned->n_re_groups = max_g;
            }
            data_ptr = data_owned.get();
        } else if (n_terms_unified > 1) {
            Rcpp::stop(
                "laplace_spec_dense: data.n_re_terms == %d but "
                "data.re_group_multi_flat is empty. "
                "Set data.re_group_multi_flat (length N * n_terms).",
                data.n_re_terms);
        } else if ((int)data.re_group.size() != data.N
                   && (int)re_group_1based.size() != data.N) {
            Rcpp::stop(
                "laplace_spec_dense: layout.has_re but neither "
                "data.re_group_multi_flat, data.re_group, nor the shim "
                "re_group argument has length N (= %d). Got "
                "re_group_1based size %d.",
                data.N, (int)re_group_1based.size());
        }
    }
    const ModelData& data_use = *data_ptr;

    int N = data_use.N;
    int n_x = L.n_x;

    bool use_sparse = (n_x >= 200);
    SparseCholeskySolver sparse_solver;
    DenseCholeskyScratch dense_scratch;
    dense_scratch.ensure(n_x);
    const void* response_data = data.model_response_data;

    #ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    #endif

    std::vector<double> eta_flat((size_t)N * np, 0.0);

    auto eval_objective = [&](const std::vector<double>& trial_params) -> double {
        compute_eta_spec(data_use, trial_params, L, re_group_1based, N, eta_flat, n_threads);
        double ll = total_log_lik_spec(
            trial_params, eta_flat, data_use, layout, *spec, response_data, N
        );
        double lp = log_prior_latent(trial_params, L, data_use.sigma_beta, 1.0);
        return ll + lp;
    };

    double obj_current = eval_objective(params_inout);
    int n_iter = 0;
    bool converged = false;

    DenseVec grad(n_x, 0.0);
    DenseMat H(n_x, DenseVec(n_x, 0.0));
    std::vector<double> delta(n_x, 0.0);

    for (int iter = 0; iter < max_iter; iter++) {
        compute_eta_spec(data_use, params_inout, L, re_group_1based, N, eta_flat, n_threads);
        scatter_spec(params_inout, eta_flat, re_group_1based, L,
                     data_use, layout, *spec, response_data, N, 1.0,
                     grad, H, n_threads);

        std::fill(delta.begin(), delta.end(), 0.0);
        bool solve_ok = dispatch_factor_solve(H, grad, delta, n_x,
                                              sparse_solver, use_sparse,
                                              dense_scratch);
        if (!solve_ok) {
            // Damped fall-back: tiny gradient step keeps the search moving so a
            // bad starting point doesn't kill the run.
            std::vector<double> grad_step(n_x, 0.0);
            for (int j = 0; j < n_x; j++) {
                grad_step[j] = std::isfinite(grad[j]) ? 0.01 * grad[j] : 0.0;
            }
            std::vector<double> trial;
            apply_latent_step(L, params_inout, grad_step, 1.0, trial);
            params_inout = trial;
            obj_current = eval_objective(params_inout);
            n_iter = iter + 1;
            continue;
        }

        // Step halving in latent slots only — hyperparameter slots are pinned.
        double step_scale = 1.0;
        std::vector<double> trial;
        for (int half = 0; half <= 12; half++) {
            apply_latent_step(L, params_inout, delta, step_scale, trial);
            double obj_try = eval_objective(trial);
            if (obj_try >= obj_current - 1e-8 || half == 12) {
                params_inout = trial;
                obj_current = obj_try;
                break;
            }
            step_scale *= 0.5;
        }

        n_iter = iter + 1;
        double m = 0.0;
        for (int j = 0; j < n_x; j++) {
            m = std::max(m, std::abs(step_scale * delta[j]));
        }
        if (m < tol) {
            converged = true;
            break;
        }
    }

    // Final Hessian + log-marginal at the mode.
    compute_eta_spec(data_use, params_inout, L, re_group_1based, N, eta_flat, n_threads);
    scatter_spec(params_inout, eta_flat, re_group_1based, L,
                 data_use, layout, *spec, response_data, N, 1.0,
                 grad, H, n_threads);
    double log_det_Q = 0.0;
    dispatch_factor_log_det(H, n_x, sparse_solver, use_sparse,
                             dense_scratch, log_det_Q);

    double ll = total_log_lik_spec(
        params_inout, eta_flat, data_use, layout, *spec, response_data, N
    );
    double lp = log_prior_latent(params_inout, L, data_use.sigma_beta, 1.0);
    double log_marginal = finalize_log_marginal(ll, lp, log_det_Q, n_x);

    if (n_iter_out)      *n_iter_out      = n_iter;
    if (converged_out)   *converged_out   = converged ? 1 : 0;
    if (log_det_Q_out)   *log_det_Q_out   = log_det_Q;
    if (log_marginal_out)*log_marginal_out= log_marginal;
}

} // namespace tulpa

// ============================================================================
// Test harness: Gaussian LikelihoodSpec + iid RE driven through
// laplace_mode_spec_dense_impl. Lets the testthat suite exercise the
// spec-driven Laplace path against the family-enum reference without
// requiring a downstream package. Internal — not part of the public ABI.
// ============================================================================

namespace {

struct GaussianResponse {
    const double* y;
    int N;
    double phi;   // observation sd
};

template<typename T>
T gaussian_ll_T(
    int i,
    const T* eta,
    const T& /*logit_zi*/,
    const T& /*logit_oi*/,
    const std::vector<T>& /*params*/,
    const tulpa::ModelData& /*data*/,
    const tulpa::ParamLayout& /*layout*/,
    const void* model_data
) {
    auto* resp = static_cast<const GaussianResponse*>(model_data);
    T r = T(resp->y[i]) - eta[0];
    T phi2 = T(resp->phi * resp->phi);
    return T(-0.5) * std::log(T(2.0 * M_PI) * phi2) - r * r / (T(2.0) * phi2);
}

void gaussian_eta_weights(
    int i,
    const double* eta,
    double /*logit_zi*/,
    double /*logit_oi*/,
    const std::vector<double>& /*params*/,
    const tulpa::ModelData& /*data*/,
    const tulpa::ParamLayout& /*layout*/,
    const void* model_data,
    double* grad_eta,
    double* neg_hess_eta
) {
    auto* resp = static_cast<const GaussianResponse*>(model_data);
    double inv_phi2 = 1.0 / (resp->phi * resp->phi);
    grad_eta[0]     = (resp->y[i] - eta[0]) * inv_phi2;
    neg_hess_eta[0] = inv_phi2;
}

} // namespace (anonymous, test only)

// [[Rcpp::export]]
Rcpp::List cpp_laplace_spec_test_gaussian(
    Rcpp::NumericVector y,
    Rcpp::NumericMatrix X,
    Rcpp::IntegerVector re_idx,    // 1-based; pass integer(0) for no RE
    int n_re_groups,
    double sigma_re,
    double sigma_beta,
    double phi,
    int max_iter = 100,
    double tol = 1e-8,
    int n_threads = 1
) {
    int N = y.size();
    int p = X.ncol();
    bool has_re = (n_re_groups > 0) && (re_idx.size() == N);

    // Build ProcessData (row-major X_flat).
    tulpa::ProcessData proc;
    proc.p = p;
    proc.X_flat.resize((size_t)N * p);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < p; j++) {
            proc.X_flat[(size_t)i * p + j] = X(i, j);
        }
    }

    // Build LikelihoodSpec.
    tulpa::LikelihoodSpec spec;
    spec.n_processes      = 1;
    spec.name             = "gaussian_spec_test";
    spec.ll_double        = &gaussian_ll_T<double>;
    spec.eta_weights_fn   = &gaussian_eta_weights;
    spec.n_extra_params   = 0;

    // Build response payload.
    GaussianResponse resp{y.begin(), N, phi};

    // Build ModelData.
    tulpa::ModelData data;
    data.n_processes          = 1;
    data.processes.push_back(proc);
    data.N                    = N;
    data.sigma_beta           = sigma_beta;
    data.likelihood_spec      = &spec;
    data.model_response_data  = &resp;
    data.sharing.init(1);
    if (has_re) {
        data.n_re_groups = n_re_groups;
        data.re_group.assign(re_idx.begin(), re_idx.end());
    }

    // Build ParamLayout (manually — bypasses tulpa_compute_param_layout
    // because the test pins the latent layout we want to exercise).
    tulpa::ParamLayout layout;
    layout.process_beta_start.push_back(0);
    layout.process_beta_count.push_back(p);
    int next = p;
    if (has_re) {
        layout.has_re = true;
        layout.log_sigma_re_idx = next++;
        layout.re_start = next;
        layout.re_end   = next + n_re_groups;
        next = layout.re_end;
    }
    layout.total_params = next;

    // Initialise params: zeros for latent, log(sigma_re) for the precision slot.
    std::vector<double> params(layout.total_params, 0.0);
    if (has_re) params[layout.log_sigma_re_idx] = std::log(sigma_re);

    // Build re_group_1based slice for the impl.
    std::vector<int> re_group_1based;
    if (has_re) re_group_1based.assign(re_idx.begin(), re_idx.end());

    int n_iter = 0;
    int converged = 0;
    double log_det_Q = 0.0;
    double log_marginal = 0.0;
    tulpa::laplace_mode_spec_dense_impl(
        data, layout, params, re_group_1based,
        max_iter, tol, n_threads,
        &n_iter, &converged, &log_det_Q, &log_marginal
    );

    // Return mode + diagnostics in the same shape as cpp_laplace_fit's list.
    Rcpp::NumericVector mode(p + (has_re ? n_re_groups : 0));
    for (int j = 0; j < p; j++) mode[j] = params[j];
    if (has_re) {
        for (int g = 0; g < n_re_groups; g++) mode[p + g] = params[layout.re_start + g];
    }
    return Rcpp::List::create(
        Rcpp::Named("mode")         = mode,
        Rcpp::Named("log_det_Q")    = log_det_Q,
        Rcpp::Named("log_marginal") = log_marginal,
        Rcpp::Named("n_iter")       = n_iter,
        Rcpp::Named("converged")    = (converged != 0)
    );
}

// ============================================================================
// Multi-process Gaussian test fixture: n_processes == 2 independent Gaussian
// likelihoods on (y1, y2) with separate (X1, X2) and a shared iid RE that
// enters both processes' linear predictors. Exercises the generalized
// laplace_mode_spec_dense_impl path (np >= 2 + cross-process Hessian zeros
// + RE shared into multiple processes) against a hand-derived Newton step
// the test computes in R.
// ============================================================================

namespace {

struct Gaussian2pResponse {
    const double* y1;
    const double* y2;
    int N;
    double phi1;
    double phi2;
};

template<typename T>
T gaussian2p_ll_T(
    int i,
    const T* eta,
    const T& /*logit_zi*/,
    const T& /*logit_oi*/,
    const std::vector<T>& /*params*/,
    const tulpa::ModelData& /*data*/,
    const tulpa::ParamLayout& /*layout*/,
    const void* model_data
) {
    auto* resp = static_cast<const Gaussian2pResponse*>(model_data);
    T r1 = T(resp->y1[i]) - eta[0];
    T r2 = T(resp->y2[i]) - eta[1];
    T phi1_2 = T(resp->phi1 * resp->phi1);
    T phi2_2 = T(resp->phi2 * resp->phi2);
    T ll = T(-0.5) * std::log(T(2.0 * M_PI) * phi1_2)
           - r1 * r1 / (T(2.0) * phi1_2)
           + T(-0.5) * std::log(T(2.0 * M_PI) * phi2_2)
           - r2 * r2 / (T(2.0) * phi2_2);
    return ll;
}

// neg_hess_eta is row-major [np x np]. Independent Gaussians: cross terms 0.
void gaussian2p_eta_weights(
    int i,
    const double* eta,
    double /*logit_zi*/,
    double /*logit_oi*/,
    const std::vector<double>& /*params*/,
    const tulpa::ModelData& /*data*/,
    const tulpa::ParamLayout& /*layout*/,
    const void* model_data,
    double* grad_eta,
    double* neg_hess_eta
) {
    auto* resp = static_cast<const Gaussian2pResponse*>(model_data);
    double inv1 = 1.0 / (resp->phi1 * resp->phi1);
    double inv2 = 1.0 / (resp->phi2 * resp->phi2);
    grad_eta[0] = (resp->y1[i] - eta[0]) * inv1;
    grad_eta[1] = (resp->y2[i] - eta[1]) * inv2;
    neg_hess_eta[0] = inv1;   // (0, 0)
    neg_hess_eta[1] = 0.0;    // (0, 1)
    neg_hess_eta[2] = 0.0;    // (1, 0)
    neg_hess_eta[3] = inv2;   // (1, 1)
}

} // namespace (anonymous, test only)

// [[Rcpp::export]]
Rcpp::List cpp_laplace_spec_test_gaussian2p(
    Rcpp::NumericVector y1,
    Rcpp::NumericVector y2,
    Rcpp::NumericMatrix X1,
    Rcpp::NumericMatrix X2,
    Rcpp::NumericVector offset1,
    Rcpp::NumericVector offset2,
    Rcpp::IntegerVector re_idx,
    int n_re_groups,
    double sigma_re,
    double sigma_beta,
    double phi1,
    double phi2,
    bool re_into_proc0 = true,
    bool re_into_proc1 = true,
    int max_iter = 100,
    double tol = 1e-10,
    int n_threads = 1
) {
    int N = y1.size();
    int p1 = X1.ncol();
    int p2 = X2.ncol();
    bool has_re = (n_re_groups > 0) && (re_idx.size() == N);

    auto build_proc = [N](Rcpp::NumericMatrix Xr,
                          Rcpp::NumericVector offr) {
        tulpa::ProcessData proc;
        proc.p = Xr.ncol();
        proc.X_flat.resize((size_t)N * proc.p);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < proc.p; j++) {
                proc.X_flat[(size_t)i * proc.p + j] = Xr(i, j);
            }
        }
        if (offr.size() == N) {
            proc.offset.assign(offr.begin(), offr.end());
        }
        return proc;
    };

    tulpa::ProcessData proc1 = build_proc(X1, offset1);
    tulpa::ProcessData proc2 = build_proc(X2, offset2);

    Gaussian2pResponse resp{y1.begin(), y2.begin(), N, phi1, phi2};

    tulpa::LikelihoodSpec spec;
    spec.n_processes      = 2;
    spec.name             = "gaussian2p_spec_test";
    spec.ll_double        = &gaussian2p_ll_T<double>;
    spec.eta_weights_fn   = &gaussian2p_eta_weights;

    tulpa::ModelData data;
    data.n_processes         = 2;
    data.processes.push_back(proc1);
    data.processes.push_back(proc2);
    data.N                   = N;
    data.sigma_beta          = sigma_beta;
    data.likelihood_spec     = &spec;
    data.model_response_data = &resp;
    data.sharing.init(2);
    data.sharing.re[0] = re_into_proc0;
    data.sharing.re[1] = re_into_proc1;
    if (has_re) {
        data.n_re_groups = n_re_groups;
        data.re_group.assign(re_idx.begin(), re_idx.end());
    }

    tulpa::ParamLayout layout;
    layout.process_beta_start.push_back(0);
    layout.process_beta_count.push_back(p1);
    layout.process_beta_start.push_back(p1);
    layout.process_beta_count.push_back(p2);
    int next = p1 + p2;
    if (has_re) {
        layout.has_re = true;
        layout.log_sigma_re_idx = next++;
        layout.re_start = next;
        layout.re_end   = next + n_re_groups;
        next = layout.re_end;
    }
    layout.total_params = next;

    std::vector<double> params(layout.total_params, 0.0);
    if (has_re) params[layout.log_sigma_re_idx] = std::log(sigma_re);

    std::vector<int> re_group_1based;
    if (has_re) re_group_1based.assign(re_idx.begin(), re_idx.end());

    int n_iter = 0;
    int converged = 0;
    double log_det_Q = 0.0;
    double log_marginal = 0.0;
    tulpa::laplace_mode_spec_dense_impl(
        data, layout, params, re_group_1based,
        max_iter, tol, n_threads,
        &n_iter, &converged, &log_det_Q, &log_marginal
    );

    int n_x = p1 + p2 + (has_re ? n_re_groups : 0);
    Rcpp::NumericVector mode(n_x);
    for (int j = 0; j < p1; j++) mode[j] = params[j];
    for (int j = 0; j < p2; j++) mode[p1 + j] = params[p1 + j];
    if (has_re) {
        for (int g = 0; g < n_re_groups; g++) {
            mode[p1 + p2 + g] = params[layout.re_start + g];
        }
    }
    return Rcpp::List::create(
        Rcpp::Named("mode")         = mode,
        Rcpp::Named("log_det_Q")    = log_det_Q,
        Rcpp::Named("log_marginal") = log_marginal,
        Rcpp::Named("n_iter")       = n_iter,
        Rcpp::Named("converged")    = (converged != 0)
    );
}

// ============================================================================
// Multi-term + slope RE test fixture. Accepts a list of RE terms
// (group_idx, n_groups, n_coefs, sigma vector, correlated, optional slope
// matrix, optional chol_raw vector) and runs the spec-Laplace path on a
// Gaussian likelihood. Returns the mode in the canonical concatenation
// order [beta | term_0 | term_1 | ...].
// ============================================================================

// [[Rcpp::export]]
Rcpp::List cpp_laplace_spec_test_multi_re(
    Rcpp::NumericVector y,
    Rcpp::NumericMatrix X,
    Rcpp::List re_terms,         // list of lists per term (see below)
    double sigma_beta,
    double phi,
    int max_iter = 200,
    double tol = 1e-12,
    int n_threads = 1
) {
    // Each element of re_terms is a list with fields:
    //   group_idx   IntegerVector length N, 1-based
    //   n_groups    int
    //   n_coefs     int (1 = intercept-only, >1 = intercept + slopes)
    //   sigma       NumericVector length n_coefs (positive sds)
    //   correlated  logical scalar
    //   slope_mat   NumericMatrix N x (n_coefs - 1) (or NULL if n_coefs == 1)
    //   chol_raw    NumericVector length n_coefs*(n_coefs-1)/2 (only when correlated)
    int N = y.size();
    int p = X.ncol();
    int K = re_terms.size();

    // ---- ProcessData ----
    tulpa::ProcessData proc;
    proc.p = p;
    proc.X_flat.resize((size_t)N * p);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < p; j++) {
            proc.X_flat[(size_t)i * p + j] = X(i, j);
        }
    }

    tulpa::LikelihoodSpec spec;
    spec.n_processes    = 1;
    spec.name           = "gaussian_multire_test";
    spec.ll_double      = &gaussian_ll_T<double>;
    spec.eta_weights_fn = &gaussian_eta_weights;

    GaussianResponse resp{y.begin(), N, phi};

    tulpa::ModelData data;
    data.n_processes         = 1;
    data.processes.push_back(proc);
    data.N                   = N;
    data.sigma_beta          = sigma_beta;
    data.likelihood_spec     = &spec;
    data.model_response_data = &resp;
    data.sharing.init(1);

    // ---- Multi-term RE structure ----
    data.n_re_terms = K;
    data.re_parameterization = 1; // unused on the Laplace path (centered)
    data.re_n_groups_multi.assign(K, 0);
    data.re_n_coefs.assign(K, 1);
    data.re_n_slopes.assign(K, 0);
    data.re_has_intercept.assign(K, 1);  // this harness always carries intercept
    data.re_correlated.assign(K, false);
    data.re_n_chol.assign(K, 0);
    data.re_offsets.assign(K, 0);
    data.re_slope_matrices.assign(K, std::vector<double>());
    data.re_group_multi_flat.assign((size_t)N * K, 0);

    // Precompute total RE shape for asserts; also build manual ParamLayout.
    int total_re_groups = 0;
    int total_sigma_params = 0;
    int total_chol_params = 0;
    bool any_slopes = false;
    bool any_correlated = false;

    // First pass: pull static term shape from each list element.
    std::vector<Rcpp::NumericVector> sigma_per_term(K);
    std::vector<Rcpp::NumericVector> chol_per_term(K);
    for (int t = 0; t < K; t++) {
        Rcpp::List term = Rcpp::as<Rcpp::List>(re_terms[t]);
        Rcpp::IntegerVector gi = Rcpp::as<Rcpp::IntegerVector>(term["group_idx"]);
        int n_g = Rcpp::as<int>(term["n_groups"]);
        int n_c = Rcpp::as<int>(term["n_coefs"]);
        bool corr = Rcpp::as<bool>(term["correlated"]);
        if ((int)gi.size() != N) {
            Rcpp::stop("term %d: group_idx length %d != N (%d)", t + 1,
                       (int)gi.size(), N);
        }
        data.re_n_groups_multi[t] = n_g;
        data.re_n_coefs[t]        = n_c;
        data.re_n_slopes[t]       = std::max(0, n_c - 1);
        data.re_correlated[t]     = (n_c > 1) && corr;
        data.re_n_chol[t]         = data.re_correlated[t] ? n_c * (n_c - 1) / 2 : 0;
        data.re_offsets[t]        = total_re_groups;
        for (int i = 0; i < N; i++) {
            data.re_group_multi_flat[(size_t)i * K + t] = gi[i];
        }
        if (data.re_n_slopes[t] > 0) {
            any_slopes = true;
            Rcpp::NumericMatrix sm = Rcpp::as<Rcpp::NumericMatrix>(term["slope_mat"]);
            if (sm.nrow() != N || sm.ncol() != data.re_n_slopes[t]) {
                Rcpp::stop("term %d: slope_mat is %dx%d, expected %dx%d",
                           t + 1, sm.nrow(), sm.ncol(), N, data.re_n_slopes[t]);
            }
            data.re_slope_matrices[t].assign(
                (size_t)N * data.re_n_slopes[t], 0.0);
            for (int i = 0; i < N; i++) {
                for (int s = 0; s < data.re_n_slopes[t]; s++) {
                    data.re_slope_matrices[t][(size_t)i * data.re_n_slopes[t] + s]
                        = sm(i, s);
                }
            }
        }
        if (data.re_correlated[t]) any_correlated = true;
        sigma_per_term[t] = Rcpp::as<Rcpp::NumericVector>(term["sigma"]);
        if ((int)sigma_per_term[t].size() != n_c) {
            Rcpp::stop("term %d: sigma length %d != n_coefs (%d)", t + 1,
                       (int)sigma_per_term[t].size(), n_c);
        }
        if (data.re_correlated[t]) {
            chol_per_term[t] = Rcpp::as<Rcpp::NumericVector>(term["chol_raw"]);
            if ((int)chol_per_term[t].size() != data.re_n_chol[t]) {
                Rcpp::stop("term %d: chol_raw length %d != n_coefs*(n_coefs-1)/2 (%d)",
                           t + 1, (int)chol_per_term[t].size(), data.re_n_chol[t]);
            }
        }
        total_re_groups    += n_g;
        total_sigma_params += n_c;
        total_chol_params  += data.re_n_chol[t];
    }
    data.total_re_groups    = total_re_groups;
    data.total_re_params    = 0;
    for (int t = 0; t < K; t++) {
        data.total_re_params += data.re_n_groups_multi[t] * data.re_n_coefs[t];
    }
    data.total_sigma_params = total_sigma_params;
    data.total_chol_params  = total_chol_params;
    data.has_re_slopes              = any_slopes;
    data.has_re_correlated_slopes   = any_correlated;

    // ---- Build ParamLayout manually to mirror hmc_param_layout's
    // multi-term branches. We use the slopes-or-not branch because that's
    // the schema the new spec-Laplace path consumes.
    tulpa::ParamLayout layout;
    layout.process_beta_start.push_back(0);
    layout.process_beta_count.push_back(p);
    int next = p;
    layout.has_re                 = true;
    layout.has_re_slopes          = any_slopes;
    layout.has_re_correlated_slopes = any_correlated;
    layout.log_sigma_re_multi.resize(K);
    layout.log_sigma_re_slopes.resize(K);
    layout.re_start_multi.resize(K);
    layout.re_end_multi.resize(K);
    layout.re_n_coefs_multi.resize(K);
    layout.re_correlated_multi.resize(K);
    layout.chol_re_start_multi.assign(K, -1);
    layout.chol_re_end_multi.assign(K, -1);

    // 1. sigma slots (q_t per term)
    for (int t = 0; t < K; t++) {
        int q = data.re_n_coefs[t];
        layout.re_n_coefs_multi[t] = q;
        layout.re_correlated_multi[t] = data.re_correlated[t];
        layout.log_sigma_re_slopes[t].resize(q);
        for (int c = 0; c < q; c++) {
            layout.log_sigma_re_slopes[t][c] = next++;
        }
        layout.log_sigma_re_multi[t] = layout.log_sigma_re_slopes[t][0];
    }
    // 2. chol slots (only for correlated terms)
    for (int t = 0; t < K; t++) {
        if (data.re_n_chol[t] > 0) {
            layout.chol_re_start_multi[t] = next;
            next += data.re_n_chol[t];
            layout.chol_re_end_multi[t] = next;
        }
    }
    // 3. RE effects (group * coef per term)
    for (int t = 0; t < K; t++) {
        layout.re_start_multi[t] = next;
        next += data.re_n_groups_multi[t] * data.re_n_coefs[t];
        layout.re_end_multi[t] = next;
    }
    // legacy mirrors
    layout.log_sigma_re_idx = layout.log_sigma_re_multi[0];
    layout.re_start = layout.re_start_multi[0];
    layout.re_end   = layout.re_end_multi[0];
    layout.total_params = next;

    // ---- params: hyperparams from term spec, latent zero-init ----
    std::vector<double> params(layout.total_params, 0.0);
    for (int t = 0; t < K; t++) {
        int q = data.re_n_coefs[t];
        for (int c = 0; c < q; c++) {
            params[layout.log_sigma_re_slopes[t][c]] =
                std::log(sigma_per_term[t][c]);
        }
        if (data.re_correlated[t]) {
            for (int j = 0; j < data.re_n_chol[t]; j++) {
                params[layout.chol_re_start_multi[t] + j] = chol_per_term[t][j];
            }
        }
    }

    int n_iter = 0;
    int converged = 0;
    double log_det_Q = 0.0;
    double log_marginal = 0.0;
    std::vector<int> empty_group;
    tulpa::laplace_mode_spec_dense_impl(
        data, layout, params, empty_group,
        max_iter, tol, n_threads,
        &n_iter, &converged, &log_det_Q, &log_marginal
    );

    // Build mode = [beta | term_0_block | term_1_block | ...]
    int n_x = p;
    for (int t = 0; t < K; t++) n_x += data.re_n_groups_multi[t] * data.re_n_coefs[t];
    Rcpp::NumericVector mode(n_x);
    int off = 0;
    for (int j = 0; j < p; j++) mode[off++] = params[j];
    for (int t = 0; t < K; t++) {
        int q = data.re_n_coefs[t];
        int n_g = data.re_n_groups_multi[t];
        for (int j = 0; j < n_g * q; j++) {
            mode[off++] = params[layout.re_start_multi[t] + j];
        }
    }
    return Rcpp::List::create(
        Rcpp::Named("mode")         = mode,
        Rcpp::Named("log_det_Q")    = log_det_Q,
        Rcpp::Named("log_marginal") = log_marginal,
        Rcpp::Named("n_iter")       = n_iter,
        Rcpp::Named("converged")    = (converged != 0)
    );
}
