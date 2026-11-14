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
#include "laplace_builtin_family_spec.h"
#include "laplace_cholesky.h"
#include "laplace_cholesky_dispatch.h"
#include "laplace_newton.h"       // shared single-arm loop (np == 1 delegates here)
#include "laplace_newton_loop.h"
#include "laplace_re_priors.h"
#include "laplace_spec_curvature3.h"  // build_spec_curvature3_oracle (inner-skew diagnostic)
#include "laplace_spec_solve.h"   // spec_inner_solve (defined below, shared with driver)
#include "hard_constraints.h"     // HardConstraintState, level_prior_conditioning
#include "laplace_spatial_priors.h"
#include "latent_block.h"
#include "linalg_fast.h"
#include "nested_laplace_grid.h"   // nl_check_positive
#include "omp_threads.h"          // tulpa_parallel_for (serial route at one thread)
#include "sparse_cholesky.h"
#include "sparse_hessian.h"         // SparseHessianBuilder (sparse spec assembly)
#include <Rcpp.h>
#include "lkj_chol_helpers.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// Additive guard on a squared scale before it is inverted into a precision. A
// scale of exactly zero is a point mass, which the inverted form cannot carry;
// the guard turns it into a very large finite precision instead of infinity.
constexpr double kPrecisionDenomGuard = 1e-300;

// Below this, a centering fold moves the intercept by less than the rounding of
// the values it is folded into, so the fold is skipped rather than applied.
constexpr double kCenterFoldCutoff = 1e-15;

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

    // GMRF latent blocks (icar/bym2/car_proper/rw1/rw2/ar1/iid/nngp/tgmrf).
    // Appended to the compacted latent vector AFTER all beta + RE blocks, so
    // the compacted layout is [beta per proc | RE terms | blocks] -- bit-
    // identical to the single-arm nested kernel's `x`. Each block b lives at
    // [block_latent_offset[b], + block_size[b]) in the compacted n_x vector and
    // at [block_param_start[b], + block_size[b]) in the params vector. The
    // contiguous-latent contract (enforced in build_latent_layout) requires
    // block_param_start[b] == blocks[b].start == block_latent_offset[b], i.e.
    // each LatentBlock's own `start` offset (used by its idx/d_fac/add_prior/
    // log_prior/center callbacks) coincides with the compacted offset. Empty
    // when no blocks (the conditional-Laplace path used by tulpaRatio/tulpaObs).
    const std::vector<LatentBlock>* blocks = nullptr;
    int n_blocks = 0;
    std::vector<int> block_latent_offset;  // [n_blocks] compacted offset
    std::vector<int> block_param_start;    // [n_blocks] params offset (== block.start)
    std::vector<int> block_size;           // [n_blocks]

    int n_x = 0;
};

// Build the per-Laplace SpecLatentLayout from the (already-populated)
// ParamLayout + ModelData. Reads multi-term RE fields when present and
// falls back to the legacy single-term fields when n_re_terms == 0.
inline SpecLatentLayout build_latent_layout(
    const ModelData& data,
    const ParamLayout& layout,
    const std::vector<LatentBlock>* blocks = nullptr
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

    // GMRF blocks follow [beta | RE] contiguously in the compacted latent
    // vector. The contiguous-latent contract: each block's own `start` offset
    // (the position its callbacks index into x) must equal the compacted offset
    // we compute here, so the block callbacks operate directly on the gathered
    // x_latent and scatter into the matching grad/H rows.
    if (blocks != nullptr && !blocks->empty()) {
        L.blocks   = blocks;
        L.n_blocks = static_cast<int>(blocks->size());
        L.block_latent_offset.resize(L.n_blocks);
        L.block_param_start.resize(L.n_blocks);
        L.block_size.resize(L.n_blocks);
        for (int b = 0; b < L.n_blocks; b++) {
            const LatentBlock& blk = (*blocks)[b];
            if (blk.start != running) {
                Rcpp::stop("laplace_spec_dense: block %d start (%d) != computed "
                           "compacted latent offset (%d). Blocks must follow "
                           "[beta | RE] contiguously in both the params and "
                           "latent vectors.", b, blk.start, running);
            }
            L.block_latent_offset[b] = running;
            L.block_param_start[b]   = blk.start;
            L.block_size[b]          = blk.size;
            running += blk.size;
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

// Correlation Cholesky for a correlated RE term, from the term's raw slots.
// One build for every backend: tulpa::build_L_from_raw is what the HMC prior
// differentiates and what the spec solver evaluates, so the same raw vector
// gives the same Sigma whichever path a user takes. It succeeds for every raw
// vector, so there is no support boundary here to clamp at or reject on.
inline void build_chol_L(
    const std::vector<double>& params,
    const ReTermSlot& s,
    std::vector<double>& L_flat   // q x q row-major, lower triangular; resized here
) {
    const int q = s.n_coefs;
    L_flat.assign((size_t)q * q, 0.0);
    tulpa::build_L_from_raw(params.data() + s.chol_start, q, L_flat.data());
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
            double tau_c = 1.0 / (sigmas[c] * sigmas[c] + kPrecisionDenomGuard);
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

struct SpecRePrior {
    std::vector<double> precision;
    double log_det = 0.0;
};

// Covariance hyperparameters are pinned for one inner solve, including every
// density evaluation along its subspace correction.
inline std::vector<SpecRePrior> spec_re_priors(
    const std::vector<double>& params, const SpecLatentLayout& L
) {
    std::vector<SpecRePrior> priors(L.re_terms.size());
    std::vector<double> L_flat;
    for (std::size_t t = 0; t < L.re_terms.size(); t++) {
        const ReTermSlot& s = L.re_terms[t];
        const bool correlated = s.correlated && s.n_coefs > 1;
        if (correlated) build_chol_L(params, s, L_flat);
        else L_flat.clear();
        build_term_precision(term_sigmas(params, s), L_flat, s.n_coefs,
                             correlated, priors[t].precision, priors[t].log_det);
    }
    return priors;
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
//
// TermDesign resolves term t's intercept flag and slope matrix once, so a pass
// over the observations reads z_{t,i,c} with one branch and one load.
struct TermDesign {
    bool has_int = true;
    const double* M = nullptr;   // N x n_slopes row-major, or null
    int n_slopes = 0;

    double z(int i, int c) const {
        if (has_int && c <= 0) return 1.0; // implicit intercept
        const int col = has_int ? (c - 1) : c;
        if (col < 0 || M == nullptr || col >= n_slopes) return 0.0;
        return M[(size_t)i * n_slopes + col];
    }
};

inline TermDesign term_design(const ModelData& data, int t) {
    TermDesign d;
    d.has_int = re_term_has_intercept(data, t);
    if ((int)data.re_slope_matrices.size() <= t) return d;
    const auto& M = data.re_slope_matrices[t];
    if (M.empty() || (int)data.re_n_slopes.size() <= t) return d;
    const int n_slopes = data.re_n_slopes[t];
    if (n_slopes <= 0) return d;
    d.M = M.data();
    d.n_slopes = n_slopes;
    return d;
}

inline std::vector<TermDesign> term_designs(const ModelData& data, int K) {
    std::vector<TermDesign> out(K);
    for (int t = 0; t < K; t++) out[t] = term_design(data, t);
    return out;
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
template<class T> inline T obs_re_contrib(
    const ModelData& data,
    const std::vector<T>& params,
    const SpecLatentLayout& L,
    const std::vector<TermDesign>& designs,
    int i,
    int n_terms_unified,
    bool& used_out
) {
    used_out = false;
    if (!L.has_re) return T(0.0);
    T re_eff(0.0);
    for (int t = 0; t < (int)L.re_terms.size(); t++) {
        int g = re_group_at(data, i, t, n_terms_unified);
        if (g < 0) continue;
        const ReTermSlot& s = L.re_terms[t];
        if (g >= s.n_groups) continue;
        const int q = s.n_coefs;
        const int base = s.param_start + g * q;
        // z_{t,i,c} = 1 for the implicit intercept (when present) and the
        // slope design value otherwise; TermDesign::z encodes both cases.
        T contrib(0.0);
        for (int c = 0; c < q; c++) {
            contrib += params[base + c] * designs[t].z(i, c);
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
template<class T> inline void compute_eta_spec(
    const ModelData& data,
    const std::vector<T>& params,
    const SpecLatentLayout& L,
    const std::vector<int>& /*re_group_1based*/,
    int N,
    int k_grid,
    std::vector<T>& eta_flat,
    int n_threads
) {
    const int np = L.np;
    const int n_terms_unified = (data.n_re_terms > 0) ? data.n_re_terms : 1;
    const std::vector<TermDesign> designs =
        term_designs(data, static_cast<int>(L.re_terms.size()));
    std::vector<char> shared(np);
    for (int k = 0; k < np; k++) shared[k] = re_shared_into(data, k);

    // Per-block grid-mixing coefficient d_fac(k_grid) (BYM2/IID reparam; 1.0
    // for plain indexed blocks). Constant across observations, so cache once.
    std::vector<double> d_fac_cache(L.n_blocks, 1.0);
    for (int b = 0; b < L.n_blocks; b++) {
        const LatentBlock& blk = (*L.blocks)[b];
        d_fac_cache[b] = blk.d_fac_at(k_grid);
    }

    // Reached once per objective evaluation (eval_penalized_log_lik_ll ->
    // compute_eta), so the region entry itself is on the hot path: at one
    // thread tulpa_parallel_for takes a plain loop rather than entering
    // libgomp. Rows write disjoint eta_flat slots, so the two routes are
    // bit-identical.
    tulpa_parallel_for(n_threads, N, [&](int i) {
        bool re_used = false;
        T re_eff = obs_re_contrib(data, params, L, designs, i,
                                       n_terms_unified, re_used);

        // Block contribution to this obs's predictor (single-process: blocks
        // are guarded to np == 1 at the impl entry, so they enter process 0).
        // INDEXED_SINGLE blocks touch one unit (idx); INDEXED_MULTI blocks
        // (SPDE: ~3 mesh nodes via the FEM projector A) touch several units
        // with barycentric weights read from obs_indices.
        T blk_eff(0.0);
        std::vector<std::pair<int,double>> blk_multi;
        std::vector<double> blk_basis;
        for (int b = 0; b < L.n_blocks; b++) {
            const LatentBlock& blk = (*L.blocks)[b];
            // Per-row design weight (SVC field): weight[i] * z[.], folded into
            // the block amplitude once, ahead of the kind split, so an SPDE
            // (INDEXED_MULTI) or HSGP (DENSE_BASIS) block carries it as an
            // areal (INDEXED_SINGLE) one does. Unset -> 1.0, and multiplying
            // the amplitude by exactly 1.0 leaves the unweighted path
            // byte-identical.
            const double d_b = d_fac_cache[b] * block_row_weight(blk, i, 0);
            if (blk.contrib_kind == BlockContribKind::INDEXED_MULTI) {
                blk.fill_obs_indices(i, /*k_arm=*/0, blk_multi);
                for (const auto& nw : blk_multi) {
                    int l = nw.first;
                    if (l >= 1 && l <= L.block_size[b]) {
                        blk_eff += d_b * nw.second
                                 * params[L.block_param_start[b] + l - 1];
                    }
                }
            } else if (blk.contrib_kind == BlockContribKind::DENSE_BASIS) {
                // Every block coefficient is touched by obs i with weight
                // basis_eval(i)[l] (HSGP folds sqrt(S_l) into the weight). The
                // block's prior is added separately in scatter_spec / log_prior.
                const int sz = L.block_size[b];
                blk_basis.assign(sz, 0.0);
                blk.basis_eval(i, /*k_arm=*/0, k_grid, blk_basis.data());
                const T* xp = &params[L.block_param_start[b]];
                T acc(0.0);
                for (int l = 0; l < sz; l++) acc += blk_basis[l] * xp[l];
                blk_eff += d_b * acc;
            } else {
                int l = blk.idx(i, /*k_arm=*/0);
                if (l >= 1 && l <= L.block_size[b]) {
                    blk_eff += d_b * params[L.block_param_start[b] + l - 1];
                }
            }
        }

        for (int k = 0; k < np; k++) {
            const ProcessData& proc = data.processes[k];
            T e(0.0);
            if (proc.p > 0) {
                const double* row = proc.X_flat.data() + (std::ptrdiff_t)i * proc.p;
                const T* beta = &params[L.beta_start[k]];
                for (int j = 0; j < proc.p; j++) e += row[j] * beta[j];
            }
            if (!proc.offset.empty()) e += proc.offset[i];
            if (re_used && shared[k]) e += re_eff;
            if (k == 0) e += blk_eff;
            eta_flat[(std::ptrdiff_t)i * np + k] = e;
        }
    });
}

// Where scatter_spec writes. Both sinks take lower-triangle (row >= col)
// entries.
//
// DenseSpecSink writes into the row-major dense Hessian; a block's prior goes
// through its dense add_prior.
struct DenseSpecSink {
    DenseMat& H;
    void add(int r, int c, double v) { H[r][c] += v; }
    void add_block_prior(const LatentBlock& blk, DenseVec& grad,
                         const Rcpp::NumericVector& x, int k_grid) {
        if (blk.add_prior) blk.add_prior(grad, H, x, k_grid);
    }
};

// SparseSpecSink writes into a SparseHessianBuilder, resolving each entry's slot
// through csc_slot; a nonzero write off the pattern is counted for the
// enclosing HessianPatternGuard. A block's prior goes through add_prior_sparse.
struct SparseSpecSink {
    SparseHessianBuilder& H;
    void add(int r, int c, double v) {
        const int k = H.csc_slot(r, c);
        if (k >= 0) {
            H.values[k] += v;
        } else if (v != 0.0) {
            record_hessian_pattern_drop();
        }
    }
    void add_block_prior(const LatentBlock& blk, DenseVec& grad,
                         const Rcpp::NumericVector& x, int k_grid) {
        if (blk.add_prior_sparse) blk.add_prior_sparse(H, grad, x, k_grid);
    }
};

inline DenseSpecSink spec_sink(DenseMat& H) { return DenseSpecSink{H}; }
inline SparseSpecSink spec_sink(SparseHessianBuilder& H) {
    return SparseSpecSink{H};
}

// Per-observation gradient + neg-Hessian assembled into the latent gradient
// and Hessian in (beta_0..beta_{np-1}, re_term_0..re_term_{K-1}) space.
// eta_weights_fn writes
//   grad_eta[k]                       = d log_lik_i / d eta_i_k
//   neg_hess_eta[k * np + l]          = -d^2 log_lik_i / (d eta_i_k d eta_i_l)
// At the end the prior contribution is added: beta ridge + per-term RE
// precision Q_t (computed once from σ_t and L_t).
//
// `grad` and `H` arrive zeroed (the Newton loop zeroes its Hessian store
// immediately before every scatter) and only H's LOWER triangle is assembled:
// every reader of the spec Hessian -- the dense Cholesky, CHOLMOD, the store_Q
// export -- reads the lower triangle alone.
//
// `H` is a sink (DenseSpecSink / SparseSpecSink below), so one assembly serves
// the dense scratch Hessian and a SparseHessianBuilder on the pattern
// build_spec_hessian_pattern derives.
template <typename Sink>
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
    int k_grid,
    const Rcpp::NumericVector* x_latent,
    double /*tau_re_legacy*/,
    DenseVec& grad,
    Sink H,
    int /*n_threads*/,
    const BetaPrior* beta_prior,
    const std::vector<SpecRePrior>& re_priors,
    // Per-observation weights already evaluated at exactly this eta
    // (SpecFusedWeights), N * np and N * np * np, or null to call
    // spec.eta_weights_fn.
    const double* cached_grad_eta = nullptr,
    const double* cached_neg_hess_eta = nullptr
) {
    const int np = L.np;
    std::vector<double> grad_eta(np, 0.0);
    std::vector<double> neg_hess_eta((size_t)np * np, 0.0);
    std::vector<double> w_l_vec(np, 0.0);
    const int K = (int)L.re_terms.size();
    const int n_terms_unified = (data.n_re_terms > 0) ? data.n_re_terms : 1;
    const std::vector<TermDesign> designs = term_designs(data, K);
    std::vector<char> shared(np);
    for (int k = 0; k < np; k++) shared[k] = re_shared_into(data, k);

    // Per-obs caches reused across the obs loop.
    std::vector<int>     g_term(K, -1);    // 0-based group at obs i for each term
    std::vector<int>     q_term(K, 0);     // q_t cached
    // Slope-row z_{t,i,c} for c = 0..q_t-1 packed contiguously per term.
    std::vector<std::vector<double>> z_term(K);
    for (int t = 0; t < K; t++) z_term[t].assign(L.re_terms[t].n_coefs, 0.0);

    // Block grid-mixing coefficients d_fac(k_grid) (cached once) and per-obs
    // active-unit scratch (compacted latent index of the unit obs i loads onto,
    // or -1).
    std::vector<double> blk_dfac(L.n_blocks, 1.0);
    for (int b = 0; b < L.n_blocks; b++) {
        const LatentBlock& blk = (*L.blocks)[b];
        blk_dfac[b] = blk.d_fac_at(k_grid);
    }
    std::vector<int> blk_active_idx(L.n_blocks, -1);
    // Per-obs flat list of latent contributions (compacted latent index,
    // effective weight d_fac * a). One entry per INDEXED_SINGLE block (weight
    // d_fac), several per INDEXED_MULTI block (one per FEM mesh node, weight
    // d_fac * barycentric_weight). The block scatter walks this list so the
    // many-to-one SPDE projection and the one-to-one areal index share a path.
    std::vector<std::pair<int,double>> blk_contrib;
    std::vector<std::pair<int,double>> blk_multi_scratch;
    std::vector<double> blk_basis_scratch;

    for (int i = 0; i < N; i++) {
        if (cached_grad_eta) {
            std::copy_n(cached_grad_eta + (std::ptrdiff_t)i * np, np,
                        grad_eta.begin());
            std::copy_n(cached_neg_hess_eta + (std::ptrdiff_t)i * np * np,
                        (std::ptrdiff_t)np * np, neg_hess_eta.begin());
        } else {
            std::fill(grad_eta.begin(), grad_eta.end(), 0.0);
            std::fill(neg_hess_eta.begin(), neg_hess_eta.end(), 0.0);

            spec.eta_weights_fn(
                i, &eta_flat[(std::ptrdiff_t)i * np], 0.0, 0.0,
                params, data, layout, response_data,
                grad_eta.data(), neg_hess_eta.data()
            );
        }

        // Resolve per-term groups + slope rows once.
        for (int t = 0; t < K; t++) {
            const ReTermSlot& s = L.re_terms[t];
            int g = re_group_at(data, i, t, n_terms_unified);
            if (g >= s.n_groups) g = -1;
            g_term[t] = g;
            q_term[t] = s.n_coefs;
            if (g < 0) continue;
            for (int c = 0; c < s.n_coefs; c++) {
                z_term[t][c] = designs[t].z(i, c);
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
            if (shared[k]) s_grad += gk;
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
                            for (int m = 0; m <= j; m++) {
                                H.add(off_k + j, off_l + m, w_xj * xl[m]);
                            }
                        }
                    } else {
                        for (int j = 0; j < pk.p; j++) {
                            const double w_xj = w_kl * xk[j];
                            for (int m = 0; m < pl.p; m++) {
                                H.add(off_k + j, off_l + m, w_xj * xl[m]);
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
        for (int l = 0; l < np; l++) {
            double w_l = 0.0;
            for (int k = 0; k < np; k++) {
                if (!shared[k]) continue;
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
                const int r = re_row_base + c;
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
                        H.add(r, off_l + m, zc_w * xl[m]);
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
            if (!shared[k]) continue;
            for (int l = 0; l < np; l++) {
                if (!shared[l]) continue;
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
                        const int r = row_base_t + c;
                        if (t == tp) {
                            // A term reads one group per observation, so
                            // tp == t carries gp == g: the block sits on the
                            // diagonal of the H lower triangle and cp <= c.
                            for (int cp = 0; cp <= c; cp++) {
                                H.add(r, row_base_tp + cp,
                                      zc * z_term[tp][cp] * s_hess);
                            }
                        } else {
                            // Different term, tp < t: we are below the
                            // diagonal block by construction.
                            for (int cp = 0; cp < qtp; cp++) {
                                H.add(r, row_base_tp + cp,
                                      zc * z_term[tp][cp] * s_hess);
                            }
                        }
                    }
                }
            }
        }

        // ============= LATENT BLOCK scatter (lower triangle) =============
        // Single-process (np == 1): blocks enter process 0, so the effective
        // eta-space score / weight summed over shared processes are exactly
        // s_grad / s_hess / w_l_vec computed above. Mirrors the multi-block
        // nested kernel's accumulate_latent_cross_terms, but the weights come
        // from the LikelihoodSpec rather than grad_hess_for_family.
        //
        // The obs contributes through a flat list of (compacted latent index,
        // effective weight) pairs: one pair per INDEXED_SINGLE block (weight
        // d_fac), several per INDEXED_MULTI block (one per FEM mesh node, weight
        // d_fac * barycentric_weight). For each contribution (idx_c, a_c):
        //   grad[idx_c]               += a_c * s_grad
        //   H[idx_c, beta_l]          += a_c * w_l * X_l(i, .)
        //   H[idx_c, re_row]          += a_c * z * s_hess
        //   H[idx_c, idx_d] (lower)   += a_c * a_d * s_hess  (every pair, incl.
        //                                the diagonal and cross-node SPDE terms)
        // The block prior Q(theta) is added once, after the observation loop.
        if (L.n_blocks > 0) {
            blk_contrib.clear();
            for (int b = 0; b < L.n_blocks; b++) {
                const LatentBlock& blk = (*L.blocks)[b];
                // The per-row SVC weight rides the block amplitude, so the
                // contribution weight a_c is d_fac * weight[i] on every kind:
                // grad scales by weight and the block x block Hessian by
                // weight^2 through the existing chain rule, exactly as the
                // joint multi-arm scatter does. Unset -> 1.0.
                const double d_b = blk_dfac[b] * block_row_weight(blk, i, 0);
                if (blk.contrib_kind == BlockContribKind::INDEXED_MULTI) {
                    blk.fill_obs_indices(i, /*k_arm=*/0, blk_multi_scratch);
                    for (const auto& nw : blk_multi_scratch) {
                        int l = nw.first;
                        if (l >= 1 && l <= L.block_size[b]) {
                            blk_contrib.emplace_back(
                                L.block_latent_offset[b] + l - 1,
                                d_b * nw.second);
                        }
                    }
                } else if (blk.contrib_kind == BlockContribKind::DENSE_BASIS) {
                    // Obs i touches every block coefficient l with weight
                    // basis_eval(i)[l]; the block x beta / RE / block scatter
                    // below walks the resulting contribution list, so the dense
                    // rank-N data fill shares the indexed path (the O(size^2)
                    // block x block inner loop is the intended dense update).
                    const int sz = L.block_size[b];
                    blk_basis_scratch.assign(sz, 0.0);
                    blk.basis_eval(i, /*k_arm=*/0, k_grid,
                                   blk_basis_scratch.data());
                    const int off = L.block_latent_offset[b];
                    for (int l = 0; l < sz; l++) {
                        const double w = blk_basis_scratch[l];
                        if (w != 0.0) blk_contrib.emplace_back(off + l, d_b * w);
                    }
                } else {
                    int l = blk.idx(i, /*k_arm=*/0);
                    if (l >= 1 && l <= L.block_size[b]) {
                        blk_contrib.emplace_back(
                            L.block_latent_offset[b] + l - 1, d_b);
                    }
                }
            }
            const int n_contrib = static_cast<int>(blk_contrib.size());
            for (int c = 0; c < n_contrib; c++) {
                const int    idx_c = blk_contrib[c].first;
                const double a_c   = blk_contrib[c].second;

                grad[idx_c] += a_c * s_grad;

                // block x beta (block row > beta col -> lower triangle)
                for (int l = 0; l < np; l++) {
                    const ProcessData& pl = data.processes[l];
                    if (pl.p == 0) continue;
                    const double w_l = w_l_vec[l];
                    if (w_l == 0.0) continue;
                    const double* xl = pl.X_flat.data() + (std::ptrdiff_t)i * pl.p;
                    const double a_w = a_c * w_l;
                    const int off_l = L.latent_offset[l];
                    for (int m = 0; m < pl.p; m++) {
                        H.add(idx_c, off_l + m, a_w * xl[m]);
                    }
                }

                if (s_hess != 0.0) {
                    // block x RE (block row > re row -> lower triangle)
                    for (int t = 0; t < K; t++) {
                        int g = g_term[t]; if (g < 0) continue;
                        const ReTermSlot& s = L.re_terms[t];
                        const int qn = q_term[t];
                        const int re_row_base = s.latent_offset + g * qn;
                        for (int cc = 0; cc < qn; cc++) {
                            const double zc = z_term[t][cc];
                            if (zc == 0.0) continue;
                            H.add(idx_c, re_row_base + cc, a_c * zc * s_hess);
                        }
                    }
                    // block x block over every contribution pair (route to the
                    // lower triangle by compacted index; includes the c == d
                    // diagonal and the cross-node SPDE interactions).
                    for (int d = 0; d < n_contrib; d++) {
                        const int    idx_d = blk_contrib[d].first;
                        if (idx_d > idx_c) continue;   // upper handled when c,d swap
                        const double v = a_c * blk_contrib[d].second * s_hess;
                        H.add(idx_c, idx_d, v);
                    }
                }
            }
        }
    }

    // Fixed-effect Gaussian prior. With an explicit BetaPrior (per-coef mean +
    // precision -- e.g. the EM block prior threaded from cpp_laplace_fit_multi_re)
    // each coefficient gets its own tau_j / mean_j; otherwise the legacy scalar
    // ridge N(0, sigma_beta^2 I). bj is the global beta index across processes,
    // matching the length-p BetaPrior (single-process for the multi-RE caller).
    {
        const double tau_scalar = 1.0 / (data.sigma_beta * data.sigma_beta + kPrecisionDenomGuard);
        int bj = 0;
        for (int k = 0; k < np; k++) {
            const int off_k = L.latent_offset[k];
            for (int j = 0; j < L.beta_count[k]; j++, bj++) {
                const double tau  = beta_prior ? beta_prior->tau_at(bj)  : tau_scalar;
                const double mean = beta_prior ? beta_prior->mean_at(bj) : 0.0;
                grad[off_k + j] += -tau * (params[L.beta_start[k] + j] - mean);
                H.add(off_k + j, off_k + j, tau);
            }
        }
    }

    // RE prior: apply the cell's Q_t to each group.
    if (!L.re_terms.empty()) {
        for (int t = 0; t < K; t++) {
            const ReTermSlot& s = L.re_terms[t];
            const int q = s.n_coefs;
            const std::vector<double>& Q_flat = re_priors[t].precision;
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
                // H += Q on the lower triangle of the q x q sub-block
                for (int c = 0; c < q; c++) {
                    for (int cp = 0; cp <= c; cp++) {
                        H.add(base_lat + c, base_lat + cp,
                              Q_flat[(size_t)c * q + cp]);
                    }
                }
            }
        }
    }

    // Block prior Q(theta). The block reads the current field from x_latent
    // (gathered in compacted order, so x_latent[block.start + s] holds unit s's
    // value) and scatters -Q.field into grad and +Q into H at the same compacted
    // offsets.
    if (L.n_blocks > 0 && x_latent != nullptr) {
        for (int b = 0; b < L.n_blocks; b++) {
            H.add_block_prior((*L.blocks)[b], grad, *x_latent, k_grid);
        }
    }
}

// Structural pattern of the spec Hessian over a [beta | RE] latent layout, as
// lower-triangle CSC: every entry scatter_spec can write, independent of the
// values at any iterate.
//
//   * beta x beta: the full lower triangle. A beta column also takes every RE
//     row, since an observation in group g of any term couples b_g with every
//     fixed effect. Each beta column is therefore contiguous from its diagonal
//     to n_x - 1.
//   * RE term t, group g: the lower triangle of its q_t x q_t block (data and
//     prior), and, for a later term t' > t, the q_t' x q_t block of every group
//     of t' that shares an observation with g.
//
// Latent blocks are not described (their fill comes from each block's own
// pattern callbacks, which this layout does not carry), so a layout holding any
// returns false, as does one whose RE spans do not tile [P, n_x).
inline bool build_spec_hessian_pattern(
    const SpecLatentLayout& L, const ModelData& data, int N,
    std::vector<int>& csc_p, std::vector<int>& csc_i
) {
    if (L.n_blocks != 0) return false;
    const int n_x = L.n_x;
    const int P = L.latent_offset[L.np];
    const int K = static_cast<int>(L.re_terms.size());
    {
        int running = P;
        for (const ReTermSlot& s : L.re_terms) {
            if (s.latent_offset != running) return false;
            running += s.n_groups * s.n_coefs;
        }
        if (running != n_x) return false;
    }

    // Groups of later terms sharing an observation with each (term, group),
    // stored as the first latent row of the partner group.
    std::vector<int> group_base(K + 1, 0);
    for (int t = 0; t < K; t++)
        group_base[t + 1] = group_base[t] + L.re_terms[t].n_groups;
    std::vector<std::vector<int>> partners(K > 1 ? group_base[K] : 0);
    if (K > 1) {
        const int n_terms_unified = (data.n_re_terms > 0) ? data.n_re_terms : 1;
        std::vector<int> g_term(K, -1);
        for (int i = 0; i < N; i++) {
            for (int t = 0; t < K; t++) {
                int g = re_group_at(data, i, t, n_terms_unified);
                if (g >= L.re_terms[t].n_groups) g = -1;
                g_term[t] = g;
            }
            for (int tp = 0; tp < K; tp++) {
                if (g_term[tp] < 0) continue;
                std::vector<int>& out = partners[group_base[tp] + g_term[tp]];
                for (int t = tp + 1; t < K; t++) {
                    if (g_term[t] < 0) continue;
                    const ReTermSlot& s = L.re_terms[t];
                    out.push_back(s.latent_offset + g_term[t] * s.n_coefs);
                }
            }
        }
        for (std::vector<int>& v : partners) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }
    }
    // A partner's coefficient count, read off its first row.
    auto partner_q = [&](int row_base) {
        int t = K - 1;
        while (t > 0 && L.re_terms[t].latent_offset > row_base) t--;
        return L.re_terms[t].n_coefs;
    };

    csc_p.assign(n_x + 1, 0);
    csc_i.clear();
    for (int c = 0; c < P; c++) {
        csc_p[c] = static_cast<int>(csc_i.size());
        for (int r = c; r < n_x; r++) csc_i.push_back(r);
    }
    for (int t = 0; t < K; t++) {
        const ReTermSlot& s = L.re_terms[t];
        const int q = s.n_coefs;
        for (int g = 0; g < s.n_groups; g++) {
            const int base = s.latent_offset + g * q;
            const std::vector<int>* part =
                (K > 1) ? &partners[group_base[t] + g] : nullptr;
            for (int cp = 0; cp < q; cp++) {
                csc_p[base + cp] = static_cast<int>(csc_i.size());
                for (int c = cp; c < q; c++) csc_i.push_back(base + c);
                if (part) {
                    for (int row_base : *part) {
                        const int qp = partner_q(row_base);
                        for (int c = 0; c < qp; c++) csc_i.push_back(row_base + c);
                    }
                }
            }
        }
    }
    csc_p[n_x] = static_cast<int>(csc_i.size());
    return true;
}

// The spec path's data log-likelihood, threaded the way compute_eta_spec and
// the family-enum compute_total_log_lik are. It is evaluated once per
// line-search trial, so the sum is the same shape as the eta assembly beside it
// and reads the same n_threads. tulpa_parallel_sum cuts the range into
// contiguous chunks, so at more than one thread the summation ORDER differs from
// the serial one and the two agree to floating-point tolerance rather than
// bitwise -- the invariant test-nested-laplace-joint-threading.R states for the
// joint path.
inline double total_log_lik_spec(
    const std::vector<double>& params,
    const std::vector<double>& eta_flat,
    const ModelData& data,
    const ParamLayout& layout,
    const LikelihoodSpec& spec,
    const void* response_data,
    int N,
    int n_threads
) {
    const int np = data.n_processes;
    return tulpa_parallel_sum(n_threads, N, [&](int i) {
        return spec.ll_double(
            i, &eta_flat[(std::ptrdiff_t)i * np], 0.0, 0.0,
            params, data, layout, response_data
        );
    });
}

// The weights a fused log-likelihood pass (LikelihoodSpec::ll_eta_weights_fn)
// produced, and the eta they belong to. The Newton loop's next scatter is at
// the eta of its last objective evaluation, so it reads them here instead of
// calling eta_weights_fn; the key is eta's bits, so a hit hands the scatter
// exactly the weights the call would have returned.
struct SpecFusedWeights {
    std::vector<double> eta;        // the key, N * np
    std::vector<double> grad;       // N * np
    std::vector<double> neg_hess;   // N * np * np

    bool matches(const std::vector<double>& eta_flat) const {
        return !eta.empty() && eta.size() == eta_flat.size() &&
               std::memcmp(eta.data(), eta_flat.data(),
                           eta.size() * sizeof(double)) == 0;
    }
};

// total_log_lik_spec through the fused callback, recording each observation's
// weights in `fw`. Each value is ll_double's for the same arguments and the
// reduction is the same, so the sum is total_log_lik_spec's bit for bit.
inline double total_log_lik_spec_fused(
    const std::vector<double>& params,
    const std::vector<double>& eta_flat,
    const ModelData& data,
    const ParamLayout& layout,
    const LikelihoodSpec& spec,
    const void* response_data,
    int N,
    int n_threads,
    SpecFusedWeights& fw
) {
    const int np = data.n_processes;
    const std::size_t nn = (std::size_t)np * np;
    fw.eta.clear();
    fw.grad.assign((std::size_t)N * np, 0.0);
    fw.neg_hess.assign((std::size_t)N * nn, 0.0);
    const double ll = tulpa_parallel_sum(n_threads, N, [&](int i) {
        return spec.ll_eta_weights_fn(
            i, &eta_flat[(std::ptrdiff_t)i * np], 0.0, 0.0,
            params, data, layout, response_data,
            &fw.grad[(std::size_t)i * np], &fw.neg_hess[(std::size_t)i * nn]
        );
    });
    fw.eta = eta_flat;
    return ll;
}

inline double log_prior_latent(
    const std::vector<double>& params,
    const SpecLatentLayout& L,
    const std::vector<SpecRePrior>& re_priors,
    double sigma_beta,
    double /*tau_re_legacy*/,
    const Rcpp::NumericVector* x_latent = nullptr,
    int k_grid = 0,
    const BetaPrior* beta_prior = nullptr
) {
    double lp = 0.0;
    // Fixed-effect Gaussian log-prior (per-coef tau_j / mean_j when a BetaPrior
    // is supplied, else the scalar N(0, sigma_beta^2 I) ridge). Mirrors the
    // scatter beta-prior block so the mode and this log-density agree.
    const double tau_scalar = 1.0 / (sigma_beta * sigma_beta + kPrecisionDenomGuard);
    int bj = 0;
    for (int k = 0; k < L.np; k++) {
        for (int j = 0; j < L.beta_count[k]; j++, bj++) {
            const double tau  = beta_prior ? beta_prior->tau_at(bj)  : tau_scalar;
            const double mean = beta_prior ? beta_prior->mean_at(bj) : 0.0;
            // tau = 0 (sd = Inf) is a flat improper prior: no density and no
            // normalizer. Skipping avoids 0.5*log(0) = -Inf poisoning both the
            // log-marginal and the line-search objective (which would then
            // accept every full Newton step, disabling the monotone safeguard).
            if (tau == 0.0) continue;
            const double d = params[L.beta_start[k] + j] - mean;
            lp += -0.5 * tau * d * d + 0.5 * std::log(tau / (2.0 * M_PI));
        }
    }

    if (!L.re_terms.empty()) {
        for (int t = 0; t < (int)L.re_terms.size(); t++) {
            const ReTermSlot& s = L.re_terms[t];
            const int q = s.n_coefs;
            const std::vector<double>& Q_flat = re_priors[t].precision;
            const double log_det_Q = re_priors[t].log_det;
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

    // Block log-priors log p(x_block | theta_k), summed across blocks.
    if (L.n_blocks > 0 && x_latent != nullptr) {
        for (int b = 0; b < L.n_blocks; b++) {
            const LatentBlock& blk = (*L.blocks)[b];
            if (blk.log_prior) lp += blk.log_prior(*x_latent, k_grid);
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

// Coordinate map between the compacted latent vector (length n_x, layout
// [beta | RE | blocks]) the shared Newton loop drives and the full params vector
// the spec helpers read (latent slots interleaved with pinned hyperparameters).
// Mutual inverses; single source of truth for the bridge used by
// spec_inner_solve and its callers.
static inline void scatter_compacted_latent(
    const SpecLatentLayout& L, const double* x, std::vector<double>& p
) {
    for (int k = 0; k < L.np; k++)
        for (int j = 0; j < L.beta_count[k]; j++)
            p[L.beta_start[k] + j] = x[L.latent_offset[k] + j];
    for (const ReTermSlot& s : L.re_terms) {
        const int n = s.n_groups * s.n_coefs;
        for (int j = 0; j < n; j++) p[s.param_start + j] = x[s.latent_offset + j];
    }
    for (int b = 0; b < L.n_blocks; b++)
        for (int j = 0; j < L.block_size[b]; j++)
            p[L.block_param_start[b] + j] = x[L.block_latent_offset[b] + j];
}
static inline void gather_compacted_latent(
    const SpecLatentLayout& L, const std::vector<double>& p, double* x
) {
    for (int k = 0; k < L.np; k++)
        for (int j = 0; j < L.beta_count[k]; j++)
            x[L.latent_offset[k] + j] = p[L.beta_start[k] + j];
    for (const ReTermSlot& s : L.re_terms) {
        const int n = s.n_groups * s.n_coefs;
        for (int j = 0; j < n; j++) x[s.latent_offset + j] = p[s.param_start + j];
    }
    for (int b = 0; b < L.n_blocks; b++)
        for (int j = 0; j < L.block_size[b]; j++)
            x[L.block_latent_offset[b] + j] = p[L.block_param_start[b] + j];
}

// Spec-driven Laplace inner solve for any np >= 1: the one place the
// LikelihoodSpec helpers are wrapped as the shared loop's closures. Declared in
// laplace_spec_solve.h; the standalone entry (laplace_mode_spec_dense_impl, any
// np) and the single-arm nested outer-grid driver both route through it. The eta
// buffer is N * np (the spec helpers own the multi-process coupling); the caller
// must size scratch.eta to at least N * np. The returned mode is in compacted
// [beta | RE | blocks] coordinates; the live Cholesky factor is left resident in
// `scratch`/`solver` for the caller's predictive-variance back-solves.
LaplaceResult spec_inner_solve(
    const ModelData& data,
    const ParamLayout& layout,
    const std::vector<LatentBlock>* blocks,
    int k_grid,
    const LikelihoodSpec& spec,
    const void* response_data,
    const std::vector<int>& re_group_1based,
    int max_iter, double tol, int n_threads,
    const std::vector<double>& base_params,
    NewtonScratch& scratch,
    SparseCholeskySolver* solver,
    bool store_Q,
    const std::vector<std::pair<int, int>>* inv_block_layout,
    const BetaPrior* beta_prior,
    int sparse_override,
    bool compute_skew,
    const std::vector<int>* skew_probe_idx,
    const SubspaceDebiasOptions* debias,
    const CilaOptions* cila,
    std::uint64_t cila_cell_key
) {
    const SpecLatentLayout L = build_latent_layout(data, layout, blocks);
    const int N = data.N;
    const int np = L.np;
    const int n_eta = N * np;        // flattened per-process eta [i*np + k]
    const int n_x = L.n_x;

    std::vector<double> params_work = base_params;   // pinned hyperparams + warm start
    const std::vector<SpecRePrior> re_priors = spec_re_priors(base_params, L);
    std::vector<double> eta_flat((size_t)n_eta, 0.0);
    SpecFusedWeights fused_weights;
    std::shared_ptr<device::LikelihoodWorkspace> device_workspace;
    auto device_policy=device::current_policy();
    device::Status device_status;
    DeviceLikelihood device_kernel;
    if (device::select_likelihood(data,layout,device_kernel,*device_policy,device_status,true)) {
        device::ObservationPlan plan; plan.n=N;
        plan.np=np+static_cast<int>(device_kernel.parameter_indices.size());
        plan.width=static_cast<int>(params_work.size());
        std::vector<device::Projection> symbolic_params;
        symbolic_params.reserve(params_work.size());
        for (std::size_t j=0; j<params_work.size(); ++j)
            symbolic_params.push_back(device::Projection::leaf(static_cast<int>(j)));
        for (int k=0; k<np; ++k) {
            const auto& process=data.processes[k];
            plan.add_design(process.X_flat.data(),process.p,L.beta_start[k]);
            for (int j=0; j<process.p; ++j) {
                symbolic_params[L.beta_start[k]+j]=device::Projection(0);
            }
        }
        for (std::size_t j=0; j<device_kernel.parameter_indices.size(); ++j) {
            plan.add_design(nullptr,0,0);
        }
        std::vector<device::Projection> symbolic_eta(n_eta);
        compute_eta_spec(data,symbolic_params,L,re_group_1based,N,k_grid,symbolic_eta,1);
        for (int k=0; k<plan.np; ++k) for (int i=0; i<N; ++i) {
            const auto e=k<np ? symbolic_eta[static_cast<std::size_t>(i)*np+k]
                : device::Projection::leaf(device_kernel.parameter_indices[k-np]);
            plan.append_row(e);
        }
        bool created=false;
        device_workspace=std::static_pointer_cast<device::LikelihoodWorkspace>(device_policy->workspace(
            &data,&layout,"laplace_observation",[&]() {
                created=true;
                if (!spec.device_fn(data,layout,response_data,true,device_kernel))
                    throw std::logic_error("device likelihood declined its initialized descriptor");
                return std::make_shared<device::LikelihoodWorkspace>(std::move(plan),std::move(device_kernel));
            }));
        if (!created) device_status=device_workspace->refresh_projection(std::move(plan));
    }
    if (!device_status) {
        device_policy->fail(device_status); throw std::runtime_error(device_status.message);
    }
    double device_log_lik=0;

    auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta_out) {
        scatter_compacted_latent(L, x.begin(), params_work);
        if (device_workspace) {
            spec.device_fn(data,layout,response_data,false,device_workspace->kernel());
            device::LikelihoodResult result;
            const auto status=device_workspace->evaluate(params_work,1,true,true,result);
            if (!status) { device_policy->fail(status); throw std::runtime_error(status.message); }
            device_policy->record("likelihood_weights",true);
            device_log_lik=result.value[0];
            const int channels=device_workspace->plan().np;
            for (int i=0; i<N; ++i) for (int k=0; k<np; ++k)
                eta_flat[static_cast<std::size_t>(i)*np+k]=result.eta[static_cast<std::size_t>(i)*channels+k];
            fused_weights.eta=eta_flat;
            fused_weights.grad.resize(n_eta);
            for (int i=0; i<N; ++i) for (int k=0; k<np; ++k)
                fused_weights.grad[static_cast<std::size_t>(i)*np+k]=result.score[static_cast<std::size_t>(k)*N+i];
            fused_weights.neg_hess.resize(static_cast<std::size_t>(N)*np*np);
            for (int i=0; i<N; ++i) for (int k=0; k<np; ++k) for (int l=0; l<np; ++l)
                fused_weights.neg_hess[(static_cast<std::size_t>(i)*np+k)*np+l]=
                    result.curvature[(static_cast<std::size_t>(i)*channels+k)*channels+l];
            std::copy(eta_flat.begin(),eta_flat.end(),eta_out.begin());
            return;
        }
        compute_eta_spec(data, params_work, L, re_group_1based, N, k_grid,
                         eta_flat, n_threads);
        for (int i = 0; i < n_eta; i++) eta_out[i] = eta_flat[i];
    };
    // Generic over the Hessian store: the dense scratch matrix, or the
    // structural-pattern builder below.
    auto scatter_grad_hess = [&](const Rcpp::NumericVector& x,
                                 const Rcpp::NumericVector& eta,
                                 DenseVec& grad, auto& H) {
        scatter_compacted_latent(L, x.begin(), params_work);
        for (int i = 0; i < n_eta; i++) eta_flat[i] = eta[i];
        // x is the compacted latent the block callbacks index -> pass &x as x_latent.
        const bool cached = fused_weights.matches(eta_flat);
        scatter_spec(params_work, eta_flat, re_group_1based, L,
                     data, layout, spec, response_data, N, k_grid,
                     &x, 1.0, grad, spec_sink(H), n_threads, beta_prior, re_priors,
                     cached ? fused_weights.grad.data() : nullptr,
                     cached ? fused_weights.neg_hess.data() : nullptr);
    };
    auto center_effects_fn = [&](Rcpp::NumericVector& x) {
        for (int b = 0; b < L.n_blocks; b++) {
            const LatentBlock& blk = (*blocks)[b];
            if (!blk.center) continue;
            // Fold each removed constant into the coefficient it aliases with,
            // so eta is preserved (compute_eta_spec enters the block only into
            // process 0; mirrors center_joint in the joint driver). Without
            // this an intrinsic block with an unpenalized constant null space
            // (RW1/RW2) silently deletes the level from the linear predictor,
            // so the reported intercept collapses to zero and the corrupted
            // level leaks into the integrated log-marginal.
            const double d_fac = blk.d_fac_at(k_grid);
            for (const auto& fold : blk.center(x)) {
                if (std::abs(fold.amount) < kCenterFoldCutoff) continue;
                if (fold.beta_offset < 0 || fold.beta_offset >= L.beta_count[0])
                    continue;
                x[L.latent_offset[0] + fold.beta_offset] += d_fac * fold.amount;
            }
        }
    };
    auto compute_log_prior = [&](const Rcpp::NumericVector& x,
                                 const Rcpp::NumericVector& /*eta*/) -> double {
        scatter_compacted_latent(L, x.begin(), params_work);
        return log_prior_latent(params_work, L, re_priors, data.sigma_beta, 1.0, &x, k_grid,
                                beta_prior);
    };
    auto log_lik_fn = [&](const Rcpp::NumericVector& eta,
                          bool need_weights = true) -> double {
        for (int i = 0; i < n_eta; i++) eta_flat[i] = eta[i];
        if (device_workspace && fused_weights.matches(eta_flat)) return device_log_lik;
        if (need_weights && spec.ll_eta_weights_fn) {
            return total_log_lik_spec_fused(params_work, eta_flat, data, layout,
                                            spec, response_data, N, n_threads,
                                            fused_weights);
        }
        return total_log_lik_spec(params_work, eta_flat, data, layout,
                                  spec, response_data, N, n_threads);
    };

    std::vector<double> x_init(n_x, 0.0);
    gather_compacted_latent(L, base_params, x_init.data());   // latent warm start

    // Slots the feasibility sweep may shift when x_init lies outside the
    // likelihood's domain: the first beta coefficient of each process. That is
    // the intercept whenever the caller's design carries one -- the same
    // assumption center_effects_fn above makes when it folds a removed field
    // level into L.latent_offset[0]. Under a design without an intercept column
    // the slot is a covariate instead, so the shift moves eta non-uniformly; the
    // sweep then either still finds an interior point or reports infeasibility,
    // and in neither case does the solve proceed on an undefined objective.
    //
    // Passing these costs nothing on a feasible start: the sweep's single
    // establishing evaluation replaces the lazy one the loop used to make at
    // iteration 0, at the same x, for the same value.
    std::vector<int> feasible_start_coords;
    for (int k = 0; k < np; k++) {
        if (L.beta_count[k] > 0) feasible_start_coords.push_back(L.latent_offset[k]);
    }

    Curvature3Oracle curvature3;
    if (compute_skew) {
        curvature3 = build_spec_curvature3_oracle(spec, response_data, data,
                                                  layout, params_work);
    }

    // From STRUCTURAL_SPARSE_THRESHOLD latents on, a [beta | RE] layout
    // assembles straight into a builder on the structural pattern, so a Newton
    // step never zeroes or reads an n_x x n_x matrix. The scratch keeps the
    // installed pattern; a solve on the same structure (every draw of a batch,
    // every cell of a grid) finds it equal and reuses it, together with the
    // solver's symbolic factor. A layout with latent blocks assembles densely.
    SparseHessianBuilder* sparse_H = nullptr;
    if (newton_use_sparse(n_x, sparse_override, STRUCTURAL_SPARSE_THRESHOLD)) {
        std::vector<int> csc_p, csc_i;
        if (build_spec_hessian_pattern(L, data, N, csc_p, csc_i)) {
            SparseHessianBuilder& B = scratch.H_sparse;
            if (B.n != n_x || B.col_ptr != csc_p || B.row_idx != csc_i) {
                B.init_csc(n_x, std::move(csc_p), std::move(csc_i));
            }
            sparse_H = &B;
        }
    }

    // The hard constraints the blocks carry: an intrinsic field's level row,
    // which puts the level in the coefficient it folds into so that
    // coefficient's prior acts on it.
    HardConstraintState hard;
    if (blocks) {
        hard.init(n_x, *blocks);
        if (hard.active())
            hard.prior_conditioning =
                level_prior_conditioning(*blocks, k_grid, scratch.x_try);
    }

    return laplace_newton_solve_ll(
        n_eta, n_x, max_iter, tol,
        compute_eta, scatter_grad_hess, center_effects_fn, compute_log_prior,
        log_lik_fn, scratch, x_init, solver, store_Q, inv_block_layout,
        sparse_override, &feasible_start_coords,
        compute_skew, skew_probe_idx, &curvature3, debias, cila, cila_cell_key,
        sparse_H, hard.active() ? &hard : nullptr
    );
}

namespace {

// The structural checks a spec solve needs before it can run, and the latent
// layout they validate. Every condition here is a property of the model
// description rather than of a hyperparameter value, so a caller solving one
// description at many values (laplace_spec_dense_check) runs them once, on the
// main thread, where an Rcpp::stop is an R error rather than std::terminate.
SpecLatentLayout spec_dense_checked_layout(
    const ModelData& data,
    const ParamLayout& layout,
    const std::vector<LatentBlock>* blocks
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

    SpecLatentLayout L = build_latent_layout(data, layout, blocks);
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

    // GMRF blocks are wired for the single-process (single-arm) path only;
    // multi-process / joint blocks (arm_scale, per-arm sharing) are the L4 step
    // of the solver unification.
    if (L.n_blocks > 0 && np != 1) {
        Rcpp::stop("laplace_spec_dense: GMRF latent blocks currently require "
                   "n_processes == 1 (got %d); joint/multi-arm blocks land at "
                   "L4 of the solver unification.", np);
    }
    return L;
}

} // namespace

int laplace_spec_dense_check(
    const ModelData& data,
    const ParamLayout& layout,
    const std::vector<LatentBlock>* blocks
) {
    return spec_dense_checked_layout(data, layout, blocks).n_x;
}

// Result-returning standalone spec Laplace. Carries the full LaplaceResult
// (compacted [beta | RE | blocks] mode, log_marginal, and -- when return_re_cov
// -- the per-(term,group) marginal covariance blocks the EM M-step consumes).
// beta_prior overrides the scalar sigma_beta ridge with a full per-coef Gaussian.
// The void laplace_mode_spec_dense_impl (the cross-package shim entry) is a thin
// wrapper over this; the standalone single-point Laplace R exports
// (cpp_laplace_fit_multi_re{,_spatial,_bym2}) call it directly.
LaplaceResult laplace_mode_spec_dense_solve(
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& params_inout,
    const std::vector<int>& re_group_1based,
    int max_iter, double tol, int n_threads,
    const std::vector<LatentBlock>* blocks,
    int k_grid,
    const BetaPrior* beta_prior,
    bool return_re_cov,
    int sparse_override,
    bool store_Q,
    bool compute_skew,
    const std::vector<int>* skew_probe_idx,
    const SubspaceDebiasOptions* debias
) {
    const SpecLatentLayout L = spec_dense_checked_layout(data, layout, blocks);
    const int np = L.np;
    const LikelihoodSpec* spec =
        static_cast<const LikelihoodSpec*>(data.likelihood_spec);

    // Per-block feasibility at this grid cell (e.g. proper-CAR PD interval).
    // Mirror the nested kernel: infeasible -> log_marginal = -inf, no solve.
    if (L.n_blocks > 0) {
        for (int b = 0; b < L.n_blocks; b++) {
            const LatentBlock& blk = (*blocks)[b];
            if (blk.prep && !blk.prep(k_grid)) {
                LaplaceResult infeasible;
                infeasible.mode.assign(L.n_x, 0.0);
                gather_compacted_latent(L, params_inout, infeasible.mode.data());
                infeasible.log_det_Q    = 0.0;
                infeasible.log_marginal = -std::numeric_limits<double>::infinity();
                infeasible.n_iter       = 0;
                infeasible.converged    = false;
                return infeasible;
            }
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

    // Per-(term,group) marginal-covariance block layout for the EM M-step
    // (return_re_cov): one block of side n_coefs per RE group, in compacted
    // latent coords -- mirrors the retired family-enum multi-RE solver's
    // inv_block_layout. spec_inner_solve fills LaplaceResult.re_cov_flat from it.
    std::vector<std::pair<int, int>> inv_block_layout;
    if (return_re_cov) {
        const SpecLatentLayout L_use = build_latent_layout(data_use, layout, blocks);
        for (const ReTermSlot& s : L_use.re_terms) {
            for (int g = 0; g < s.n_groups; g++) {
                inv_block_layout.emplace_back(s.latent_offset + g * s.n_coefs,
                                              s.n_coefs);
            }
        }
    }

    // L3/L4: every np routes through the one shared spec inner solve.
    // spec_inner_solve wraps the spec helpers (compute_eta_spec / scatter_spec /
    // total_log_lik_spec / log_prior_latent) as the shared Newton loop's closures
    // over an N*np eta buffer, so this standalone entry (any np) and the
    // single-arm nested outer-grid driver (nested_laplace_multi.h) run the same
    // loop body. GMRF blocks are gated to np == 1
    // above, so for np >= 2 there are no blocks and centering is a no-op.
    NewtonScratch scratch;
    scratch.allocate(n_x, N * np);
    SparseCholeskySolver newton_solver;
    LaplaceResult res = spec_inner_solve(
        data_use, layout, blocks, k_grid, *spec, data_use.model_response_data,
        re_group_1based, max_iter, tol, n_threads,
        params_inout, scratch, &newton_solver,
        store_Q,
        return_re_cov ? &inv_block_layout : nullptr,
        beta_prior, sparse_override,
        compute_skew, skew_probe_idx, debias
    );
    scatter_compacted_latent(L, res.mode.data(), params_inout);  // mode -> params latent
    return res;
}

// Void shim wrapper over laplace_mode_spec_dense_solve (declared in
// laplace_spec_solve.h). Keeps the historical signature (no beta_prior /
// re_cov) so the cross-package C-callable's ABI is unchanged.
void laplace_mode_spec_dense_impl(
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& params_inout,
    const std::vector<int>& re_group_1based,
    int max_iter, double tol, int n_threads,
    int* n_iter_out,
    int* converged_out,
    double* log_det_Q_out,
    double* log_marginal_out,
    const std::vector<LatentBlock>* blocks,
    int k_grid
) {
    LaplaceResult res = laplace_mode_spec_dense_solve(
        data, layout, params_inout, re_group_1based,
        max_iter, tol, n_threads, blocks, k_grid);
    if (n_iter_out)       *n_iter_out       = res.n_iter;
    if (converged_out)    *converged_out    = res.converged ? 1 : 0;
    if (log_det_Q_out)    *log_det_Q_out    = res.log_det_Q;
    if (log_marginal_out) *log_marginal_out = res.log_marginal;
}

} // namespace tulpa
