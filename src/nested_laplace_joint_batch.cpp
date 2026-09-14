// nested_laplace_joint_batch.cpp
// Batched (multi-response) joint nested-Laplace driver + entry.
//
// The B species share one design + sparsity pattern; their latent systems are
// independent (block-diagonal). This driver runs B per-species Newton solves
// sharing the FUSED cell-coupling scatter, so each species' trajectory is
// bit-identical to its independent single-species fit while the
// bandwidth-bound per-cell evaluate is amortised across species.
//
// Both scatter paths are live and selected as the single-species driver selects
// them (force_sparse, the latent dimension against SPARSE_THRESHOLD, or a block
// that only scatters sparsely): the dense path carries an n_x by n_x DenseMat
// per species and scatters through scatter_cell_coupling_batch_dense; the sparse
// path carries a SparseHessianBuilder per species seeded from one fit-level
// joint pattern and scatters through scatter_cell_coupling_batch_sparse, whose
// SparseScatterPolicy resolves the (row, col) -> flat-slot caches once per cell
// for all B species.
//
// Cell-coupling families with ALL arms coupled (occu_cover) only; plain
// outer-grid sweep with per-species warm-start chaining, as the single-species
// serial grid chains. An arm dispersion axis crossed onto the grid is loaded
// per species at each cell (the species' own nodes over the shared cell
// layout). The final pass at each species' mode is the single-species loop's
// own (joint_newton_finalize_dense / _sparse), and each species' cells are packed
// by nl_pack_grid_results, so a species returns the list the single-species
// entry returns for its own responses.

#include "nested_laplace_joint_batch.h"
#include "nested_laplace_grid.h"
#include "nested_laplace_joint_core.h"
#include "nested_laplace_joint_multi.h"
#include "joint_hessian_pattern.h"
#include "laplace_newton_joint.h"
#include "laplace_newton_joint_sparse.h"
#include "laplace_newton_loop.h"
#include "laplace_cholesky.h"
#include "laplace_cholesky_dispatch.h"
#include "laplace_core.h"
#include "latent_block.h"
#include "sparse_cholesky.h"
#include "sparse_hessian.h"
#include "tulpa/cell_coupling.h"
#include <Rcpp.h>
#include <cmath>
#include <limits>
#include <vector>

using namespace tulpa;

namespace {

// Per-species inner-solve state: the single-species Newton scratch of the path
// the fit takes, plus the species' own CHOLMOD solver. The Hessian container
// itself lives at driver scope (dense DenseMat or sparse SparseHessianBuilder,
// by path) so the fused scatter can write all B species in one pass. Allocated
// once, reused across grid cells.
struct SpeciesState {
    NewtonScratchJoint       dense;
    NewtonScratchJointSparse sparse;
    SparseCholeskySolver     solver;

    void allocate(int n_x, const std::vector<JointArm>& arms, bool use_sparse,
                  bool want_extract) {
        if (use_sparse) sparse.allocate(n_x, arms, want_extract);
        else            dense.allocate(n_x, arms, want_extract);
    }
};

// The iterate, line-search trial, eta buffers and step of one species, read off
// whichever scratch its path allocated.
struct SpeciesView {
    Rcpp::NumericVector&              x;
    Rcpp::NumericVector&              x_try;
    std::vector<Rcpp::NumericVector>& etas;
    std::vector<Rcpp::NumericVector>& etas_tmp;
    std::vector<double>&              delta;
};

inline SpeciesView species_view(SpeciesState& st, bool use_sparse) {
    if (use_sparse)
        return SpeciesView{st.sparse.x, st.sparse.x_try, st.sparse.etas,
                           st.sparse.etas_tmp, st.sparse.delta};
    return SpeciesView{st.dense.x, st.dense.x_try, st.dense.etas,
                       st.dense.etas_tmp, st.dense.delta};
}

// Compute species s's per-arm eta from x into `etas_out` (single-species
// vectors). Identical to the single-species compute_eta_joint inner body.
inline void compute_eta_species(
    const Rcpp::NumericVector& x,
    std::vector<Rcpp::NumericVector>& etas_out,
    const std::vector<JointArm>& arms,
    const std::vector<ParsedArm>& parsed,
    const std::vector<LatentBlock>& blocks,
    int k_grid,
    const std::vector<double>& d_fac_cache
) {
    const int n_arms = (int) arms.size();
    const int Bn = (int) blocks.size();
    for (int k_arm = 0; k_arm < n_arms; k_arm++) {
        const ParsedArm& pa = parsed[k_arm];
        const int N_k = arms[k_arm].N;
        const int p_k = pa.p;
        const int n_re_k = pa.n_re_groups;
        const int bstart = pa.beta_start;
        const int rstart = pa.re_start;
        std::vector<double> d_eff(Bn);
        for (int b = 0; b < Bn; b++) {
            double s = blocks[b].arm_scale ? blocks[b].arm_scale(k_arm, k_grid) : 1.0;
            d_eff[b] = s * d_fac_cache[b];
        }
        for (int i = 0; i < N_k; i++) {
            double e = (pa.offset.size() != 0) ? pa.offset[i] : 0.0;
            for (int j = 0; j < p_k; j++) e += pa.X(i, j) * x[bstart + j];
            if (n_re_k > 0) {
                int g = (int) pa.re_idx[i] - 1;
                if (g >= 0 && g < n_re_k) e += x[rstart + g];
            }
            for (int b = 0; b < Bn; b++) {
                if (d_eff[b] == 0.0) continue;
                int l = blocks[b].idx(i, k_arm);
                if (l > 0 && l <= blocks[b].size) {
                    e += d_eff[b] * block_row_weight(blocks[b], i, k_arm)
                                  * x[blocks[b].start + l - 1];
                }
            }
            etas_out[k_arm][i] = e;
        }
    }
}

// Per-species cell-coupling log-lik at the given per-arm etas (single-species
// B=1 views into the species eta + the species y column of `buf`). The cell
// loop is eval_cell_coupling_log_lik_impl's; only where the response comes
// from differs, which is what its arm_response callback is for.
inline double species_cell_loglik(
    const CellCouplingSpec& spec,
    const std::vector<int>& coupled_arms,
    const std::vector<std::vector<std::vector<int>>>& cell_rows,
    int n_cells,
    const std::vector<JointArm>& arms,
    const std::vector<Rcpp::NumericVector>& etas_s,  // per-arm species eta
    const BatchArmBuffers& buf,
    int s
) {
    return eval_cell_coupling_log_lik_impl(
        spec, coupled_arms, cell_rows, n_cells, arms, etas_s,
        [&](int kk, int k) {
            (void) kk;
            CellArmResponse r;
            r.y = buf.y[k].empty() ? nullptr
                  : buf.y[k].data() + (std::size_t) s * buf.N[k];
            // No species offset: trial counts are shared design.
            r.n_trials = buf.n_trials[k].empty() ? nullptr
                                                 : buf.n_trials[k].data();
            r.phi = buf.phi[(std::size_t) k * buf.B + s];
            return r;
        });
}

// Add the block priors + per-arm beta/RE priors into a freshly-scattered sparse
// Hessian: the tail of the single-species sparse scatter, which loads the base
// ridge separately (the Newton step and the final pass each load it).
inline void add_species_priors_sparse(
    SparseHessianBuilder&            H,
    DenseVec&                        grad,
    const Rcpp::NumericVector&       x,
    const std::vector<LatentBlock>&  blocks,
    const std::vector<ParsedArm>&    parsed,
    int                              k_grid
) {
    for (const auto& b : blocks) {
        if (b.add_prior_sparse) b.add_prior_sparse(H, grad, x, k_grid);
    }
    add_per_arm_beta_re_priors_sparse(grad, H, x, parsed);
}

} // anonymous namespace

namespace tulpa {

// Batched outer-grid driver. Returns an Rcpp::List of length B; element s is
// the nl_pack_grid_results list of species s. All-coupled cell-coupling
// families only.
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
    CurvatureMode                    step_curvature,
    bool                             force_sparse,
    const JointFixedBlockRequest*    fixed_block
) {
    const int n_arms = (int) arms.size();
    const int B = n_batch;
    const HessianPatternGuard pattern_guard;

    std::vector<int> coupled_arms;
    if (spec) coupled_arms = spec->arm_ids();
    if (coupled_arms.empty()) {
        Rcpp::stop("batched joint path requires a cell-coupling spec with "
                   "coupled arms (occu_cover); none registered.");
    }
    for (int k = 0; k < n_arms; k++) {
        bool found = false;
        for (int c : coupled_arms) if (c == k) { found = true; break; }
        if (!found && arms[k].N > 0) {
            Rcpp::stop("batched joint path requires every "
                       "data arm to be cell-coupled; arm %d is not.", k + 1);
        }
    }

    std::vector<std::vector<std::vector<int>>> cell_rows;
    int n_cells = build_cell_rows_from_arms(arms, coupled_arms, cell_rows);

    int n_x = n_x_after_re;
    for (const auto& b : blocks) n_x = std::max(n_x, b.start + b.size);
    const bool use_sparse = force_sparse || (n_x >= SPARSE_THRESHOLD) ||
                            blocks_require_sparse(blocks);
    // The dense single-species loop's own factorization choice, which below the
    // threshold is the dense Cholesky.
    const bool dense_factor_sparse = (n_x >= SPARSE_THRESHOLD);
    const bool want_fixed_block = fixed_block && fixed_block->active();

    std::vector<SpeciesState> st(B);
    for (int s = 0; s < B; s++)
        st[s].allocate(n_x, arms, use_sparse, want_fixed_block);

    BatchArmBuffers wbuf = buf;  // working copy carries etas (species-major)
    std::vector<std::vector<LaplaceResult>> cell_results(
        B, std::vector<LaplaceResult>(n_grid));
    // The warm start each species' next cell takes: the previous cell's
    // reported mode, as the single-species serial grid chains it.
    std::vector<std::vector<double>> prev_mode(B);

    std::vector<DenseVec> grad_per_sp(B);
    std::vector<DenseMat> H_per_sp;
    std::vector<SparseHessianBuilder> H_sparse_per_sp;
    SparseScatterPolicy sparse_policy;
    if (use_sparse) {
        SparseHessianBuilder pattern;
        build_joint_hessian_pattern(parsed, arms, blocks, n_x, pattern,
                                    coupled_arms, cell_rows, n_cells);
        H_sparse_per_sp.assign(B, pattern);
    } else {
        H_per_sp.assign(B, DenseMat());
    }

    auto fused_scatter = [&](int kg, CurvatureMode curvature) {
        for (int s = 0; s < B; s++) grad_per_sp[s].assign(n_x, 0.0);
        if (use_sparse) {
            for (int s = 0; s < B; s++) H_sparse_per_sp[s].zero();
            scatter_cell_coupling_batch_sparse(
                *spec, coupled_arms, cell_rows, n_cells, arms, parsed, blocks,
                kg, wbuf, grad_per_sp, H_sparse_per_sp, sparse_policy,
                curvature, false);
        } else {
            for (int s = 0; s < B; s++)
                H_per_sp[s].assign(n_x, DenseVec(n_x, 0.0));
            scatter_cell_coupling_batch_dense(
                *spec, coupled_arms, cell_rows, n_cells, arms, parsed, blocks,
                kg, wbuf, grad_per_sp, H_per_sp, curvature, false);
        }
    };

    auto load_species_etas = [&](int s, const std::vector<Rcpp::NumericVector>& etas) {
        for (int k = 0; k < n_arms; k++) {
            const int N_k = arms[k].N;
            double* dst = wbuf.etas[k].data() + (std::size_t) s * N_k;
            const double* src = REAL(etas[k]);
            for (int i = 0; i < N_k; i++) dst[i] = src[i];
        }
    };

    for (int kg = 0; kg < n_grid; kg++) {
        wbuf.load_grid_cell(kg);
        bool feasible = true;
        for (const auto& b : blocks) {
            if (b.prep && !b.prep(kg)) { feasible = false; break; }
        }
        if (!feasible) {
            for (int s = 0; s < B; s++) {
                LaplaceResult bad;
                bad.mode = (static_cast<int>(prev_mode[s].size()) == n_x)
                           ? prev_mode[s] : std::vector<double>(n_x, 0.0);
                bad.log_marginal = -std::numeric_limits<double>::infinity();
                bad.n_iter = 0;
                bad.converged = false;
                bad.log_det_Q = 0.0;
                cell_results[s][kg] = bad;
                prev_mode[s] = cell_results[s][kg].mode;
            }
            continue;
        }

        std::vector<double> d_fac_cache((int) blocks.size());
        for (int b = 0; b < (int) blocks.size(); b++) d_fac_cache[b] = blocks[b].d_fac_at(kg);

        for (int s = 0; s < B; s++) {
            SpeciesView v = species_view(st[s], use_sparse);
            if (static_cast<int>(prev_mode[s].size()) == n_x)
                for (int j = 0; j < n_x; j++) v.x[j] = prev_mode[s][j];
            else
                for (int j = 0; j < n_x; j++) v.x[j] = 0.0;
        }

        std::vector<LaplaceResult> res(B);
        std::vector<bool> converged(B, false);
        // Sentinel for "this species' objective has not been evaluated at the
        // current iterate yet"; obj_valid gates every read.
        std::vector<double> obj(B, -std::numeric_limits<double>::infinity());
        std::vector<bool> obj_valid(B, false);
        std::vector<NewtonConvState> conv_state(B);

        for (int iter = 0; iter < max_iter; iter++) {
            for (int s = 0; s < B; s++) {
                if (converged[s]) continue;
                SpeciesView v = species_view(st[s], use_sparse);
                compute_eta_species(v.x, v.etas, arms, parsed, blocks, kg, d_fac_cache);
                load_species_etas(s, v.etas);
            }
            fused_scatter(kg, step_curvature);
            for (int s = 0; s < B; s++) {
                if (converged[s]) continue;
                SpeciesView v = species_view(st[s], use_sparse);
                DenseVec& grad = grad_per_sp[s];
                bool ok;
                if (use_sparse) {
                    SparseHessianBuilder& H = H_sparse_per_sp[s];
                    add_species_priors_sparse(H, grad, v.x, blocks, parsed, kg);
                    H.add_uniform_ridge(LAPLACE_UNIFORM_RIDGE);
                    bool used_block_schur = false;
                    ok = s2z_newton_step(H, st[s].solver, n_x, pd_mode, grad.data(),
                                         v.delta.data(), used_block_schur,
                                         &st[s].sparse.s2z_block_schur_cache);
                } else {
                    DenseMat& H = H_per_sp[s];
                    for (const auto& b : blocks) if (b.add_prior) b.add_prior(grad, H, v.x, kg);
                    add_per_arm_beta_re_priors(grad, H, v.x, parsed);
                    ok = joint_pd_step_solve_dense(H, grad, v.delta, n_x,
                                                   st[s].solver, dense_factor_sparse,
                                                   st[s].dense.chol, pd_mode);
                }
                if (!ok) {
                    // The step was never solved. Move a short way along
                    // whatever finite part of it came back, so the next
                    // iteration starts somewhere else, and record the cell as
                    // not converged.
                    constexpr double kFailedStepDamping = 0.1;
                    for (int j = 0; j < n_x; j++)
                        if (std::isfinite(v.delta[j]))
                            v.x[j] += kFailedStepDamping * v.delta[j];
                    obj_valid[s] = false;
                    converged[s] = false;
                    res[s].n_iter = iter + 1;
                    continue;
                }
                auto eval_obj = [&](const Rcpp::NumericVector& xv) -> double {
                    return eval_penalized_log_lik_joint_ll(
                        xv,
                        [&](const Rcpp::NumericVector& xe,
                            std::vector<Rcpp::NumericVector>& e) {
                            compute_eta_species(xe, e, arms, parsed, blocks, kg,
                                                d_fac_cache);
                        },
                        [&](const Rcpp::NumericVector& xe,
                            const std::vector<Rcpp::NumericVector>&) {
                            return log_prior_joint_blocks(xe, blocks, parsed, kg);
                        },
                        [&](const std::vector<Rcpp::NumericVector>& e) {
                            return species_cell_loglik(*spec, coupled_arms,
                                                       cell_rows, n_cells, arms,
                                                       e, wbuf, s);
                        },
                        v.etas_tmp);
                };
                if (!obj_valid[s]) { obj[s] = eval_obj(v.x); obj_valid[s] = true; }
                double slope = newton_decrement(grad, v.delta, n_x);
                double step = line_search_backtrack(v.x, v.delta, n_x,
                                                    obj[s], slope, eval_obj,
                                                    obj[s], v.x_try, nullptr,
                                                    newton_trust_scale(conv_state[s], slope));
                res[s].n_iter = iter + 1;
                if (newton_converged(v.delta, grad, step, n_x, tol, conv_state[s]))
                    converged[s] = true;
            }
            bool all_conv = true;
            for (int s = 0; s < B; s++) if (!converged[s]) { all_conv = false; break; }
            if (all_conv) break;
        }

        // Final pass at each species' mode: the fused observed-curvature
        // scatter, the priors, then the single-species loop's own final pass.
        for (int s = 0; s < B; s++) {
            SpeciesView v = species_view(st[s], use_sparse);
            compute_eta_species(v.x, v.etas, arms, parsed, blocks, kg, d_fac_cache);
            load_species_etas(s, v.etas);
        }
        fused_scatter(kg, CurvatureMode::Observed);
        for (int s = 0; s < B; s++) {
            SpeciesView v = species_view(st[s], use_sparse);
            LaplaceResult& r = res[s];
            r.mode.assign(n_x, 0.0);
            r.converged = converged[s];
            auto compute_eta = [&](const Rcpp::NumericVector& xe,
                                   std::vector<Rcpp::NumericVector>& e) {
                compute_eta_species(xe, e, arms, parsed, blocks, kg, d_fac_cache);
            };
            auto center = [&](Rcpp::NumericVector& xc) {
                center_joint_blocks(xc, blocks, parsed, n_arms, kg,
                                    [&](int b) { return d_fac_cache[b]; });
            };
            auto log_prior = [&](const Rcpp::NumericVector& xe,
                                 const std::vector<Rcpp::NumericVector>&) {
                return log_prior_joint_blocks(xe, blocks, parsed, kg);
            };
            auto log_lik = [&](const std::vector<Rcpp::NumericVector>& e) {
                return species_cell_loglik(*spec, coupled_arms, cell_rows,
                                           n_cells, arms, e, wbuf, s);
            };
            auto eval_objective = [&](const Rcpp::NumericVector& xv) -> double {
                return eval_penalized_log_lik_joint_ll(xv, compute_eta, log_prior,
                                                       log_lik, v.etas_tmp);
            };
            DenseVec& grad = grad_per_sp[s];
            if (use_sparse) {
                SparseHessianBuilder& H = H_sparse_per_sp[s];
                add_species_priors_sparse(H, grad, v.x, blocks, parsed, kg);
                joint_newton_finalize_sparse(
                    r, n_x, st[s].sparse, H, grad, st[s].solver,
                    compute_eta, center, log_prior, log_lik, eval_objective,
                    store_Q, pd_mode, false, nullptr, nullptr, fixed_block,
                    nullptr, nullptr, static_cast<std::uint64_t>(kg) + 1ULL,
                    nullptr);
            } else {
                DenseMat& H = H_per_sp[s];
                for (const auto& b : blocks) if (b.add_prior) b.add_prior(grad, H, v.x, kg);
                add_per_arm_beta_re_priors(grad, H, v.x, parsed);
                joint_newton_finalize_dense(
                    r, n_x, st[s].dense, H, grad, st[s].solver,
                    dense_factor_sparse, compute_eta, center, log_prior,
                    log_lik, eval_objective, store_Q, pd_mode, false, nullptr,
                    nullptr, fixed_block, nullptr, nullptr,
                    static_cast<std::uint64_t>(kg) + 1ULL, nullptr);
            }
            cell_results[s][kg] = std::move(r);
            prev_mode[s] = cell_results[s][kg].mode;
        }
    }

    Rcpp::List out(B);
    for (int s = 0; s < B; s++) {
        Rcpp::List sp = nl_pack_grid_results(cell_results[s], n_grid, n_x,
                                             /*store_modes=*/true,
                                             /*n_threads_outer_realised=*/1);
        // The sparse single-species grid reports the outer width its scatter
        // partition ran at; every species here ran serially.
        if (use_sparse) sp["n_outer"] = 1;
        out[s] = sp;
    }
    pattern_guard.check("the batched joint nested-Laplace grid");
    return out;
}

} // namespace tulpa
