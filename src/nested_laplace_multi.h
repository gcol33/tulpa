// nested_laplace_multi.h
// Shared helpers for the multi-block nested-Laplace driver. Used by both
// nested_laplace.cpp (single-block kernels routing through length-1
// LatentBlock vectors) and nested_laplace_multi.cpp (genuine multi-block
// dispatch from R).
//
// The driver assembles the inner solve at each outer-grid point k as:
//
//   eta_i  = X_i β + RE_g(i) + Σ_b d_fac_b(k) * x[start_b + idx_b(i) - 1]
//   grad/H from the spec solver's observation scatter (β, RE)
//          + accumulate_latent_cross_terms (latent, latent x β, latent x RE)
//          + Σ_b add_prior_b(k)
//          + add_re_beta_priors
//   center : each block's centerer applied to its sub-vector
//   log_prior = Σ_b log_prior_b(k) + compute_log_prior_re
//
// Per-block prep is invoked once at grid point k before the inner solve; if
// any block reports infeasible (e.g. proper CAR with rho outside the PD
// interval), the inner solve short-circuits with log_marginal = -inf.

#ifndef TULPA_NESTED_LAPLACE_MULTI_H
#define TULPA_NESTED_LAPLACE_MULTI_H

#include "hessian_pattern_guard.h"        // HessianPatternGuard (see its placement rule)
#include "laplace_builtin_family_spec.h"  // builtin_family_spec, BuiltinFamilyResponse
#include "laplace_core.h"
#include "laplace_family_link.h"
#include "laplace_newton.h"
#include "laplace_re_priors.h"
#include "laplace_spec_solve.h"           // spec_inner_solve (the unified inner solve)
#include "latent_block.h"
#include "nested_laplace_grid.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// Per-observation contribution walk over the [beta (p) | RE (n_re_groups) |
// blocks] latent layout: for every latent index observation i touches, call
// sink(latent_index, weight). One definition behind the per-row predictive
// variance vector a_i and the per-row fitted eta, which differ only in what
// they accumulate into.
template <typename Sink>
inline void nl_multi_obs_contribs(
    int i, int p, bool has_re, int n_re_groups,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    const std::vector<LatentBlock>& blocks,
    const std::vector<double>& d_fac,
    std::vector<std::pair<int,double>>& scratch,
    Sink&& sink
) {
    for (int j = 0; j < p; j++) sink(j, X(i, j));
    if (has_re) {
        const int g = static_cast<int>(re_idx[i]) - 1;
        if (g >= 0 && g < n_re_groups) sink(p + g, 1.0);
    }
    for (std::size_t b = 0; b < blocks.size(); b++) {
        if (blocks[b].contrib_kind == BlockContribKind::INDEXED_MULTI) {
            blocks[b].fill_obs_indices(i, /*k_arm=*/0, scratch);
            for (const auto& nw : scratch) {
                const int l = nw.first;
                if (l > 0 && l <= blocks[b].size) {
                    sink(blocks[b].start + l - 1, d_fac[b] * nw.second);
                }
            }
        } else {
            const int l = blocks[b].idx(i, /*k_arm=*/0);
            if (l > 0 && l <= blocks[b].size) {
                const double w = blocks[b].row_weight
                                 ? blocks[b].row_weight(i, /*k_arm=*/0)
                                 : 1.0;
                sink(blocks[b].start + l - 1, d_fac[b] * w);
            }
        }
    }
}

// Rows sharing a loading vector, and therefore a predictive variance.
//
// nl_multi_obs_contribs builds row i's loading vector a_i out of the p values
// of X(i, .), the RE group re_idx[i], and, per block, either the (index,
// weight) list the block's obs_indices fills or the pair (idx(i, 0),
// row_weight(i, 0)). None of those move with the outer-grid cell. The only
// per-cell quantity in the walk is the block scalar d_fac_b(k), which
// multiplies every entry block b contributes, uniformly across rows.
//
// The latent index ranges of beta, the RE block and each latent block are
// disjoint, so a latent index identifies which d_fac scales it. Two rows whose
// walks emit the same (index, weight) sequence at d_fac == 1 therefore emit
// bit-identical a_i at EVERY cell, and share the variance a_i' H_k^{-1} a_i
// exactly. One back-solve per class then serves every member of it, at every
// cell, for one O(N * (p + n_blocks)) pass over the design.
struct NlRowClasses {
    std::vector<int> row_class;  // observation -> class id in [0, n_class)
    std::vector<int> class_rep;  // class id -> the observation solved for
    std::size_t size() const { return class_rep.size(); }
};

// FNV-1a, folded a 64-bit word at a time. The hash only groups CANDIDATES:
// every merge is confirmed word by word against the representative's key, so a
// collision costs one comparison rather than fusing two distinct rows.
inline std::uint64_t nl_row_key_mix(std::uint64_t h, std::uint64_t word) {
    for (int byte = 0; byte < 8; byte++) {
        h ^= (word >> (8 * byte)) & 0xFFULL;
        h *= 1099511628211ULL;
    }
    return h;
}

inline NlRowClasses nl_build_row_classes(
    int N, int p, bool has_re, int n_re_groups,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    const std::vector<LatentBlock>& blocks
) {
    NlRowClasses out;
    if (N <= 0) return out;
    out.row_class.assign(N, 0);

    // The walk is driven at d_fac == 1 so the recorded weights carry no cell
    // dependence. Running it through nl_multi_obs_contribs itself keeps the key
    // and the loading vector one definition: a change to what a row reads
    // changes both together.
    const std::vector<double> unit_d_fac(blocks.size(), 1.0);

    // Keys packed end to end, one (index, weight-bits) pair per contribution.
    // Comparison is on exact bit patterns -- a tolerance would merge rows whose
    // loading vectors differ, and the variance is read off the same factor for
    // every member of a class.
    std::vector<int> key_idx;
    std::vector<std::uint64_t> key_w;
    std::vector<std::size_t> key_off(static_cast<std::size_t>(N) + 1, 0);
    const std::size_t key_guess =
        static_cast<std::size_t>(N) * (static_cast<std::size_t>(p) +
                                       blocks.size() + 1);
    key_idx.reserve(key_guess);
    key_w.reserve(key_guess);

    std::vector<std::uint64_t> row_hash(N, 0);
    std::vector<std::pair<int, double>> scratch;
    for (int i = 0; i < N; i++) {
        nl_multi_obs_contribs(
            i, p, has_re, n_re_groups, X, re_idx, blocks, unit_d_fac, scratch,
            [&](int idx, double w) {
                std::uint64_t bits;
                std::memcpy(&bits, &w, sizeof(bits));
                key_idx.push_back(idx);
                key_w.push_back(bits);
            });
        key_off[static_cast<std::size_t>(i) + 1] = key_idx.size();

        std::uint64_t h = 1469598103934665603ULL;
        for (std::size_t s = key_off[i]; s < key_off[i + 1]; s++) {
            h = nl_row_key_mix(h, static_cast<std::uint64_t>(
                                      static_cast<std::uint32_t>(key_idx[s])));
            h = nl_row_key_mix(h, key_w[s]);
        }
        row_hash[i] = h;
    }

    auto key_equal = [&](int i, int j) {
        const std::size_t oi = key_off[i], oj = key_off[j];
        const std::size_t ni = key_off[i + 1] - oi;
        if (ni != key_off[j + 1] - oj) return false;
        for (std::size_t s = 0; s < ni; s++) {
            if (key_idx[oi + s] != key_idx[oj + s]) return false;
            if (key_w[oi + s] != key_w[oj + s]) return false;
        }
        return true;
    };

    std::unordered_map<std::uint64_t, std::vector<int>> buckets;
    buckets.reserve(static_cast<std::size_t>(N));
    for (int i = 0; i < N; i++) {
        std::vector<int>& candidates = buckets[row_hash[i]];
        int cls = -1;
        for (int c : candidates) {
            if (key_equal(i, out.class_rep[c])) { cls = c; break; }
        }
        if (cls < 0) {
            cls = static_cast<int>(out.class_rep.size());
            out.class_rep.push_back(i);
            candidates.push_back(cls);
        }
        out.row_class[i] = cls;
    }
    return out;
}

// Generic outer-grid driver over a vector of LatentBlocks.
//
// `n_threads_outer` controls the outer-grid parallelism (1 = serial, the
// pre-refactor default). `n_threads_inner` controls per-observation parallel
// kernels (compute_eta, scatter, log-lik reduction) inside each cell. When
// `n_threads_outer > 1`, the driver pre-allocates n_threads_outer worth of
// NewtonScratch (one per OpenMP thread) and one SparseCholeskySolver per
// thread, then parallelises the grid via run_nested_laplace_grid.
inline Rcpp::List run_multi_block_nested_laplace(
    int n_grid,
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, const Rcpp::NumericVector& re_idx,
    int N, int p, int n_re_groups, double sigma_re,
    const std::vector<LatentBlock>& blocks,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    bool store_modes,
    const Rcpp::NumericVector& x_init,
    bool store_Q = false,
    int n_threads_outer = 1,
    double prune_tol = 0.0,
    const LikelihoodSpec* ext_spec = nullptr,
    void* ext_response = nullptr,
    tulpa_progress::GridProgress* progress = nullptr,
    GridCheckpoint* ckpt = nullptr,
    // Inner-Laplace skewness diagnostic (inner_laplace_skew.h), opt-in like
    // store_Q. Applied only to the FULL per-cell solve, never the cheap-pass
    // warm-start screen; the caller is expected to pass a length-1 theta grid
    // (n_grid == 1) built at a single target theta (typically the outer
    // grid's MAP cell) when requesting this, mirroring the outer Pareto-k
    // diagnostic's re-dispatch-at-a-point convention -- see
    // .nl_inner_skew_at_theta() in R/laplace_diagnostics.R.
    bool compute_skew = false,
    const std::vector<int>* skew_probe_idx = nullptr,
    // Subspace debias (subspace_debias.h). Unlike the
    // diagnostics above this runs on EVERY integrated cell, because the
    // correction it produces enters the reported marginal as a mixture over the
    // outer grid -- one node's corrected shape is not the grid's. The cheap
    // warm-start screen never runs it. nullptr or an empty index set never
    // reaches the sampler, so the grid is unchanged and consumes no random
    // number.
    const SubspaceDebiasOptions* debias = nullptr,
    // Corrected integrated Laplace (inner_cila.h). Runs on
    // every fully-solved cell for the same reason the debias does -- the
    // corrected marginal is a reweighting of each cell's own particle set, so
    // all of them travel out -- and never on the cheap warm-start screen. Its
    // auxiliary points come from an engine-owned stream keyed by the cell index,
    // so the outer grid stays parallel-integrable with the correction on.
    const CilaOptions* cila = nullptr,
    // Inner Newton steps per cell in the cheap screening sweep, forwarded to
    // run_nested_laplace_grid.
    int screen_iters = CHEAP_SCREEN_ITERS,
    // Whether to fill `fitted_eta_var`. The per-row predictive variance costs
    // one back-solve per DISTINCT loading vector per cell (see NlRowClasses
    // above), which on a design with few repeated rows is the dominant cost of
    // a cell. A caller that reads only `fitted_eta` -- or only the marginal
    // summaries -- passes false and the pass is skipped outright; the returned
    // list then carries no `fitted_eta_var` element.
    bool compute_fitted_var = true
) {
    int n_x = p + n_re_groups;
    for (const auto& b : blocks) {
        n_x = std::max(n_x, b.start + b.size);
    }

    // ---- Likelihood: built-in family or model-supplied spec -----------------
    // Each single-block (and np==1 multi-block) nested kernel routes its inner
    // Laplace solve through spec_inner_solve, which reads the per-observation
    // score / Fisher weight / log-lik entirely from a single-process
    // LikelihoodSpec; scatter_spec assembles the latent gradient + Hessian. The
    // driver therefore carries no copy of the obs + latent-cross scatter.
    //
    // By default tulpa builds builtin_family_spec(family) over (y, n_trials,
    // family, phi). A model package (tulpaObs occupancy, tulpaGlmm custom
    // families) instead supplies its own spec + response via ext_spec /
    // ext_response (passed from R as an XPtr<NestedLikelihood>); then `family` /
    // `phi` / `y` / `n_trials` are unused here and the likelihood -- including any
    // per-observation scaling, e.g. the marginalized occupancy state -- lives
    // wholly in the model's spec. Built once; read-only inside the (possibly
    // parallel) grid. sigma_beta = 100 makes the spec's beta ridge tau_beta =
    // 1e-4, identical to the nested kernel's DEFAULT_TAU_BETA; the spec also
    // folds the beta-prior log-density into the log-marginal.
    // n_re_groups > 0 with an re_idx of the wrong length is a caller error, not
    // a configuration: silently dropping the block would fit a model with no
    // random effect and still return a log-marginal for it, and the per-row
    // walks below read re_idx[i] for every i < N.
    if (n_re_groups > 0 && (int) re_idx.size() != N) {
        Rcpp::stop("length(re_idx) (%d) must equal n_obs (%d) when "
                   "n_re_groups (%d) is positive.",
                   (int) re_idx.size(), N, n_re_groups);
    }
    const bool has_re = (n_re_groups > 0);

    ProcessData proc;
    proc.p = p;
    proc.X_flat.resize((size_t)N * p);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < p; j++)
            proc.X_flat[(size_t)i * p + j] = X(i, j);

    LikelihoodSpec builtin_spec;
    std::vector<int> n_trials_vec;
    BuiltinFamilyResponse resp;
    const LikelihoodSpec* spec_ptr = ext_spec;
    void*                 resp_ptr = ext_response;
    if (!ext_spec) {
        builtin_spec  = builtin_family_spec(family);
        n_trials_vec.assign(n_trials.begin(), n_trials.end());
        resp.y        = y.begin();
        resp.n_trials = n_trials_vec.data();
        resp.N        = N;
        resp.family   = family;
        resp.phi      = phi;
        resp.prepare();
        spec_ptr      = &builtin_spec;
        resp_ptr      = &resp;
    }

    ModelData data;
    data.n_processes         = 1;
    data.processes.push_back(proc);
    data.N                   = N;
    data.sigma_beta          = DEFAULT_SIGMA_BETA;  // == 1 / DEFAULT_TAU_BETA
    data.likelihood_spec     = spec_ptr;
    data.model_response_data = resp_ptr;
    data.sharing.init(1);
    std::vector<int> re_group_1based;
    if (has_re) {
        data.n_re_groups = n_re_groups;
        re_group_1based.resize(N);
        for (int i = 0; i < N; i++) re_group_1based[i] = static_cast<int>(re_idx[i]);
        data.re_group = re_group_1based;
    }

    ParamLayout layout;
    layout.process_beta_start.push_back(0);
    layout.process_beta_count.push_back(p);
    if (has_re) {
        layout.has_re           = true;
        layout.re_start         = p;
        layout.re_end           = p + n_re_groups;
        layout.log_sigma_re_idx = n_x;   // hyperparam slot AFTER all latent
    }
    const int total_params = n_x + (has_re ? 1 : 0);
    layout.total_params = total_params;
    if (has_re) tulpa::nl_check_positive("sigma_re", sigma_re);
    const double log_sigma_re = has_re ? std::log(sigma_re) : 0.0;

    // Per-outer-thread NewtonScratch. omp_get_thread_num() returns 0 outside
    // parallel regions so the serial path correctly picks scratch_pool[0].
    //
    // The subspace sampler draws from R's RNG (rwmh.h), which is not reentrant
    // and would make a run's draws a function of the thread schedule, so a
    // debiased grid is integrated serially whatever the caller asked for.
    const bool debias_active = (debias != nullptr) && !debias->idx.empty();
    if (debias_active) n_threads_outer = 1;
    int n_outer = std::max(1, n_threads_outer);
    std::vector<NewtonScratch> scratch_pool(n_outer);
    for (auto& s : scratch_pool) s.allocate(n_x, N);

    // When the outer grid is parallel, force inner kernels to single-thread
    // OpenMP (compute_eta / scatter / log-lik reduction). Nested OpenMP at
    // this problem size is overhead-dominated.
    int n_threads_inner_eff = (n_outer > 1) ? 1 : n_threads;

    // Per-cell per-row predictive variance of the linear predictor,
    // var(eta_i | theta_k) = a_i' H_k^{-1} a_i, where a_i is the row's loading
    // vector (the same beta + RE + sum_b d_fac_b * e_{idx_b(i)} accumulation as
    // compute_eta, but as coefficients rather than multiplied by the mode). It
    // is read off the live Cholesky factor laplace_newton_solve leaves resident
    // at the converged mode (the same factor inv_block_layout reuses for the RE
    // posterior-covariance blocks), so there is no refactorization. Filled only
    // on the full solves (not the max_iter=1 cheap screen), only when modes are
    // stored, and only when the caller asked for it via compute_fitted_var;
    // packed into `fitted_eta_var` [n_grid x N] below so callers can
    // marginalise psi intervals over the hyperparameter grid. Buffer is
    // k-sliced (disjoint per cell) so the outer-parallel writes do not race.
    const bool want_fitted_var = store_modes && compute_fitted_var;
    std::vector<double> fitted_var_buf(
        want_fitted_var ? static_cast<std::size_t>(n_grid) * N : 0, 0.0);

    // Rows sharing a loading vector share the variance at every cell, so the
    // grid solves one representative per class. Built once from the design,
    // read-only inside the (possibly parallel) grid.
    const NlRowClasses row_classes =
        want_fitted_var
            ? nl_build_row_classes(N, p, has_re, n_re_groups, X, re_idx, blocks)
            : NlRowClasses{};

    // Inner implementation: takes max_iter as a parameter so the cheap-pass
    // path can call with max_iter=1 for a one-Newton-step screen. See the
    // joint analogue in nested_laplace_joint_multi.h for the rationale.
    auto solve_at_theta_impl = [&](int k,
                                   const std::vector<double>& prev_mode,
                                   SparseCholeskySolver* solver,
                                   int max_iter_use,
                                   NewtonScratch* scratch_override
                                   = nullptr,
                                   bool want_var = false,
                                   bool allow_probe = false) -> LaplaceResult
    {
        for (const auto& b : blocks) {
            if (b.prep && !b.prep(k)) {
                LaplaceResult bad;
                bad.mode = (static_cast<int>(prev_mode.size()) == n_x)
                           ? prev_mode
                           : std::vector<double>(n_x, 0.0);
                bad.log_marginal = -std::numeric_limits<double>::infinity();
                bad.n_iter = 0;
                bad.converged = false;
                bad.log_det_Q = 0.0;
                return bad;
            }
        }

        std::vector<double> d_fac_cache(blocks.size());
        for (size_t b = 0; b < blocks.size(); b++) {
            d_fac_cache[b] = blocks[b].d_fac_at(k);
        }

        int tid;
        #ifdef _OPENMP
        tid = omp_in_parallel() ? omp_get_thread_num() : 0;
        #else
        tid = 0;
        #endif

        NewtonScratch& scratch = scratch_override ? *scratch_override
                                                  : scratch_pool[tid];

        // base_params: latent warm start in [0, n_x); log_sigma_re hyperparam
        // after. spec_inner_solve builds the [beta | RE | blocks] layout,
        // wraps the spec helpers as the shared Newton loop's closures, and leaves
        // the live Cholesky factor resident in scratch/solver for the predictive-
        // variance back-solves below.
        std::vector<double> base_params(total_params, 0.0);
        if (static_cast<int>(prev_mode.size()) == n_x)
            std::copy(prev_mode.begin(), prev_mode.end(), base_params.begin());
        if (has_re) base_params[layout.log_sigma_re_idx] = log_sigma_re;

        LaplaceResult res = spec_inner_solve(
            data, layout, &blocks, k, *spec_ptr, resp_ptr, re_group_1based,
            max_iter_use, tol, n_threads_inner_eff, base_params,
            scratch, solver, store_Q, /*inv_block_layout=*/nullptr,
            /*beta_prior=*/nullptr, /*sparse_override=*/0,
            allow_probe && compute_skew, skew_probe_idx,
            allow_probe ? debias : nullptr,
            allow_probe ? cila : nullptr,
            static_cast<std::uint64_t>(k) + 1ULL
        );

        // Per-row predictive variance, var(eta_i | theta_k) = a_i' H^{-1} a_i,
        // read off the live Cholesky factor laplace_newton_solve left resident
        // at the converged mode (sparse path solves the CHOLMOD factor in
        // `solver`, dense path back-substitutes scratch.chol.L) -- one
        // back-solve per DISTINCT loading vector, no refactorization. H is the
        // spec's Fisher curvature at the mode; when
        // the model spec already returns the marginal information (e.g. tulpaObs'
        // occupancy q*sigma*(1-sigma)^2/(1-q*sigma)), no rescaling is needed --
        // the calibrated variance falls out directly.
        if (want_var && std::isfinite(res.log_marginal) &&
            static_cast<std::size_t>(k + 1) * N <= fitted_var_buf.size()) {
            const bool use_sparse = (n_x >= SPARSE_THRESHOLD);
            const bool used_sparse_factor =
                use_sparse && solver && solver->factored();
            std::vector<double> a(n_x, 0.0), z(n_x, 0.0), zwork;
            if (!used_sparse_factor) zwork.assign(n_x, 0.0);
            std::vector<std::pair<int,double>> a_multi;
            // CHOLMOD workspace for the back-solves, local to this cell: the
            // outer grid runs cells on separate threads, each against its own
            // solver, and a workspace holds handles owned by one solver's
            // cholmod_common.
            SparseCholeskySolver::SolveWorkspace ws;
            const std::size_t n_class = row_classes.size();
            const double failed = std::numeric_limits<double>::quiet_NaN();
            std::vector<double> class_var(n_class, failed);
            const std::size_t base = static_cast<std::size_t>(k) * N;
            for (std::size_t c = 0; c < n_class; c++) {
                const int i = row_classes.class_rep[c];
                std::fill(a.begin(), a.end(), 0.0);
                nl_multi_obs_contribs(
                    i, p, has_re, n_re_groups, X, re_idx, blocks,
                    d_fac_cache, a_multi,
                    [&](int idx, double w) { a[idx] += w; });
                bool ok = true;
                if (used_sparse_factor) {
                    ok = solver->solve(a.data(), z.data(), n_x, ws);
                } else {
                    ok = chol_substitute_raw(scratch.chol.L.data(), n_x,
                                             a.data(), z.data(), zwork.data());
                }
                double v = 0.0;
                if (ok) for (int j = 0; j < n_x; j++) v += a[j] * z[j];
                // H is positive definite, so a' H^-1 a is negative only when
                // the back-substitution lost accuracy; reporting 0 there gives
                // a downstream interval zero width and no way to tell that
                // apart from a genuinely zero loading vector (which does give
                // exactly 0).
                const double var = (ok && v >= 0.0) ? v : failed;
                class_var[c] = var;
                fitted_var_buf[base + i] = var;
            }
            // The rows that are not their class representative read what that
            // single back-solve produced, a failed solve's NaN included: their
            // loading vector is bit-identical, so an individual solve would
            // have failed the same way.
            if (n_class < static_cast<std::size_t>(N)) {
                for (int i = 0; i < N; i++) {
                    const int c = row_classes.row_class[i];
                    if (row_classes.class_rep[c] != i)
                        fitted_var_buf[base + i] = class_var[c];
                }
            }
        }

        return res;
    };

    // 3-arg adapter for run_nested_laplace_grid.
    auto solve_at_theta = [&](int k,
                              const std::vector<double>& prev_mode,
                              SparseCholeskySolver* solver) -> LaplaceResult
    {
        return solve_at_theta_impl(k, prev_mode, solver, max_iter, nullptr,
                                   /*want_var=*/want_fitted_var,
                                   /*allow_probe=*/true);
    };

    // A checkpoint carries the `LaplaceResult`; the per-row predictive variance
    // lives in this driver's own buffer, so a loaded cell never reaches the
    // pass that fills it. NaN goes down first, because the buffer's initial 0
    // is a silently-wrong "zero predictive variance" a grid mixture reads as a
    // real number and understates the reported variance by, in proportion to
    // the resumed mass -- the engine's convention everywhere else is that a
    // decline returns NaN precisely so this cannot happen.
    //
    // The refill then re-solves at the restored mode with ZERO Newton steps.
    // The loop cannot move the iterate, and the post-loop factorization every
    // solve ends with still runs, so the factor the back-solves read is the one
    // the original solve read at the same point: the refilled row is
    // bit-identical to the uninterrupted fit's, which is what "a resumed fit
    // equals the uninterrupted one" already promises for every other field.
    auto resume_refill = [&](int k, const std::vector<double>& mode,
                             SparseCholeskySolver* solver) {
        if (!want_fitted_var ||
            static_cast<std::size_t>(k + 1) * N > fitted_var_buf.size()) return;
        const std::size_t base = static_cast<std::size_t>(k) * N;
        std::fill(fitted_var_buf.begin() + base,
                  fitted_var_buf.begin() + base + N,
                  std::numeric_limits<double>::quiet_NaN());
        if (static_cast<int>(mode.size()) != n_x) return;
        solve_at_theta_impl(k, mode, solver, /*max_iter_use=*/0, nullptr,
                            /*want_var=*/true, /*allow_probe=*/false);
    };

    // Cheap-pass screening: a short inner Newton run warm-started from the
    // neighbour quasi-mode the driver chains across the lattice, returning
    // the quasi-mode and the Laplace log-marginal at it. Calling
    // solve_at_theta_impl with the driver-supplied `n_steps` reuses all the
    // per-cell callbacks without duplicating scatter/eta logic. Dedicated
    // thread-local solver + scratch keep cheap_eval independent of the
    // parallel fan-out's pool.
    // One cheap solver + scratch per outer worker slot: the cheap screen may
    // run per-tile chains concurrently, and CHOLMOD's
    // cholmod_common is not thread-safe, so each worker needs its own. This
    // path passes no tile metadata, so the screen stays serial (worker 0), but
    // the pool keeps the CheapEval contract uniform across call sites.
    const int n_cheap_workers = std::max(1, n_outer);
    std::vector<SparseCholeskySolver> cheap_solvers(n_cheap_workers);
    std::vector<NewtonScratch> cheap_scratches(n_cheap_workers);
    for (auto& cs : cheap_scratches) cs.allocate(n_x, N);
    auto cheap_eval = [&](int k_grid,
                          const std::vector<double>& warm,
                          int n_steps, int worker) -> LaplaceResult {
        return solve_at_theta_impl(
            k_grid, warm, &cheap_solvers[worker], n_steps,
            &cheap_scratches[worker]);
    };

    const HessianPatternGuard pattern_guard;
    Rcpp::List out = run_nested_laplace_grid(
        n_grid, n_x, solve_at_theta, x_init, store_modes, n_outer,
        /*tile_ids=*/std::vector<int>(),
        /*tile_pilot_cells=*/std::vector<int>(),
        cheap_eval, prune_tol, progress, ckpt,
        /*x_init_per_cell=*/std::vector<double>(), screen_iters,
        resume_refill
    );
    pattern_guard.check("the single-block nested-Laplace outer grid");

    // Per-row fitted linear predictor at every grid cell, reconstructed from the
    // stored modes with the SAME accumulation as the inner solve's compute_eta
    // (beta + RE + sum_b d_fac_b(k) * x_block[idx_b(i)]). `run_nested_laplace_
    // grid` stores modes row k for grid cell k, so d_fac(k) aligns. Returned as
    // `fitted_eta` [n_grid x N] so callers can marginalise mu/psi over the
    // hyperparameter grid for every prior -- including those whose predictor
    // mixes components with hyperparameter-dependent scales (e.g. BYM2) -- and
    // for held-out (n_trials = 0) rows, without re-deriving each prior's scales.
    if (store_modes && out.containsElementNamed("modes")) {
        Rcpp::NumericMatrix modes = out["modes"];
        int ng = modes.nrow();
        Rcpp::NumericMatrix fitted_eta(ng, N);
        std::vector<double> dfac(blocks.size());
        std::vector<std::pair<int,double>> e_multi;
        for (int k = 0; k < ng; k++) {
            for (size_t b = 0; b < blocks.size(); b++) dfac[b] = blocks[b].d_fac_at(k);
            for (int i = 0; i < N; i++) {
                double e = 0.0;
                nl_multi_obs_contribs(
                    i, p, has_re, n_re_groups, X, re_idx, blocks,
                    dfac, e_multi,
                    [&](int idx, double w) { e += w * modes(k, idx); });
                fitted_eta(k, i) = e;
            }
        }
        out["fitted_eta"] = fitted_eta;

        // Per-cell predictive variance of eta, var(eta_i | theta_k) = a_i'
        // H_k^{-1} a_i, filled by the solves above (0 for pruned cells, which
        // carry zero grid weight). Paired with `fitted_eta` so callers can
        // marginalise the per-row eta posterior as a Gaussian mixture over the
        // grid (mean fitted_eta[k,i], variance fitted_eta_var[k,i], weight w_k)
        // and map monotone through plogis for calibrated psi intervals.
        if (want_fitted_var &&
            fitted_var_buf.size() == static_cast<std::size_t>(ng) * N) {
            Rcpp::NumericMatrix fitted_eta_var(ng, N);
            for (int k = 0; k < ng; k++) {
                const std::size_t base = static_cast<std::size_t>(k) * N;
                for (int i = 0; i < N; i++) {
                    fitted_eta_var(k, i) = fitted_var_buf[base + i];
                }
            }
            out["fitted_eta_var"] = fitted_eta_var;
        }
    }
    return out;
}

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_MULTI_H
