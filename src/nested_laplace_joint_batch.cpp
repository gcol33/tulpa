// nested_laplace_joint_batch.cpp
// Batched (multi-response) joint nested-Laplace driver + entry (gcol33/tulpa#66).
//
// The B species share one design + sparsity pattern; their latent systems are
// independent (block-diagonal). This driver runs B per-species Newton solves
// sharing the FUSED cell-coupling scatter (scatter_cell_coupling_batch_dense),
// so each species' trajectory is bit-identical to its independent single-species
// fit while the bandwidth-bound per-cell evaluate is amortised across species.
//
// First cut: dense path only (small/medium fields), cell-coupling families with
// ALL arms coupled (occu_cover), plain outer-grid sweep with per-species
// warm-start chaining. Returns per-species { log_marginal, modes, weights,
// n_iter } so R unpacks each through the existing single-species post-processing.
// (store_Q for the per-grid inner covariance -> SDs is a follow-up; the first
// gate checks means / field / log_marginal.)

#include "nested_laplace_joint_batch.h"
#include "nested_laplace_joint_core.h"
#include "nested_laplace_joint_multi.h"
#include "laplace_newton_joint.h"
#include "laplace_newton_loop.h"
#include "laplace_cholesky_dispatch.h"
#include "laplace_core.h"
#include "latent_block.h"
#include "sparse_cholesky.h"
#include "tulpa/cell_coupling.h"
#include <Rcpp.h>
#include <cmath>
#include <limits>
#include <vector>

using namespace tulpa;

namespace {

// Per-species inner-solve state. One DenseMat H_s + per-arm eta vectors + the
// scratch the single-species Newton uses. Allocated once, reused across grid
// cells.
struct SpeciesState {
    Rcpp::NumericVector x;        // n_x latent
    Rcpp::NumericVector x_try;    // line-search trial
    std::vector<Rcpp::NumericVector> etas;      // per-arm, N_k each (species view)
    std::vector<Rcpp::NumericVector> etas_tmp;  // line-search buffer
    DenseVec grad;
    DenseMat H;
    DenseVec delta;
    DenseCholeskyScratch chol;
    SparseCholeskySolver sparse;

    void allocate(int n_x, const std::vector<JointArm>& arms) {
        x = Rcpp::NumericVector(n_x, 0.0);
        x_try = Rcpp::NumericVector(n_x, 0.0);
        etas.clear(); etas_tmp.clear();
        for (const JointArm& a : arms) {
            etas.emplace_back(a.N, 0.0);
            etas_tmp.emplace_back(a.N, 0.0);
        }
        grad.assign(n_x, 0.0);
        H.assign(n_x, DenseVec(n_x, 0.0));
        delta.assign(n_x, 0.0);
        chol.ensure(n_x);
    }
};

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
            double e = 0.0;
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
// B=1 views into the species' eta + the species' y column of `buf`). Mirrors
// eval_cell_coupling_log_lik but reads species s's y column.
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
    const int n_coupled = (int) coupled_arms.size();
    if (n_coupled == 0 || n_cells == 0) return 0.0;

    std::vector<const double*> arm_eta_ptr(n_coupled);
    std::vector<const double*> arm_y_ptr(n_coupled);
    std::vector<const int*>    arm_n_trials_ptr(n_coupled);
    std::vector<std::string>   family_holder(n_coupled);
    std::vector<const char*>   arm_family_ptr(n_coupled);
    std::vector<double>        arm_phi_vec(n_coupled);
    for (int kk = 0; kk < n_coupled; kk++) {
        int k = coupled_arms[kk];
        arm_eta_ptr[kk]      = REAL(etas_s[k]);
        arm_y_ptr[kk]        = buf.y[k].empty() ? nullptr
                               : buf.y[k].data() + (std::size_t) s * buf.N[k];
        arm_n_trials_ptr[kk] = buf.n_trials[k].empty() ? nullptr
                               : buf.n_trials[k].data() + (std::size_t) s * buf.N[k];
        family_holder[kk]    = arms[k].family;
        arm_family_ptr[kk]   = family_holder[kk].c_str();
        arm_phi_vec[kk]      = buf.phi[(std::size_t) k * buf.B + s];
    }

    std::vector<int>            arm_row_count(n_coupled);
    std::vector<const int*>     arm_rows_ptr(n_coupled);
    std::vector<std::vector<double>> grad_buf(n_coupled), nh_buf(n_coupled);
    std::vector<double*>        grad_ptr(n_coupled), nh_ptr(n_coupled);

    double total = 0.0;
    for (int c = 0; c < n_cells; c++) {
        for (int kk = 0; kk < n_coupled; kk++) {
            int rc = (int) cell_rows[kk][c].size();
            arm_row_count[kk] = rc;
            arm_rows_ptr[kk]  = cell_rows[kk][c].data();
            if ((int) grad_buf[kk].size() < rc) {
                grad_buf[kk].assign(rc, 0.0); nh_buf[kk].assign(rc, 0.0);
            } else {
                std::fill(grad_buf[kk].begin(), grad_buf[kk].begin() + rc, 0.0);
                std::fill(nh_buf[kk].begin(), nh_buf[kk].begin() + rc, 0.0);
            }
            grad_ptr[kk] = grad_buf[kk].data();
            nh_ptr[kk]   = nh_buf[kk].data();
        }
        CellEtas ev; ev.arm_eta_ptr = arm_eta_ptr.data(); ev.arm_rows = arm_rows_ptr.data();
        ev.arm_row_count = arm_row_count.data(); ev.n_arms_ = n_coupled;
        CellResponse yv; yv.arm_y = arm_y_ptr.data(); yv.arm_n_trials = arm_n_trials_ptr.data();
        yv.arm_family = arm_family_ptr.data(); yv.arm_phi = arm_phi_vec.data();
        yv.arm_rows = arm_rows_ptr.data(); yv.arm_row_count = arm_row_count.data();
        yv.n_arms_ = n_coupled;
        CellDerivs out; out.arm_grad = grad_ptr.data(); out.arm_neg_hess_diag = nh_ptr.data();
        out.arm_cross_hess = nullptr; out.arm_row_count = arm_row_count.data();
        out.n_arms_ = n_coupled;
        total += spec.evaluate_cell(c, ev, yv, out);
    }
    return total;
}

// Penalised per-species log-posterior at x (for the line search): cell-coupling
// log-lik + block log-priors + per-arm RE/beta log-prior.
inline double species_penalized_logpost(
    const Rcpp::NumericVector& x,
    std::vector<Rcpp::NumericVector>& etas_tmp,
    const CellCouplingSpec& spec,
    const std::vector<int>& coupled_arms,
    const std::vector<std::vector<std::vector<int>>>& cell_rows,
    int n_cells,
    const std::vector<JointArm>& arms,
    const std::vector<ParsedArm>& parsed,
    const std::vector<LatentBlock>& blocks,
    int k_grid,
    const std::vector<double>& d_fac_cache,
    const BatchArmBuffers& buf,
    int s
) {
    compute_eta_species(x, etas_tmp, arms, parsed, blocks, k_grid, d_fac_cache);
    double ll = species_cell_loglik(spec, coupled_arms, cell_rows, n_cells,
                                    arms, etas_tmp, buf, s);
    double lp = log_prior_per_arm_re(x, parsed);
    for (const auto& b : blocks) if (b.log_prior) lp += b.log_prior(x, k_grid);
    return ll + lp;
}

} // anonymous namespace

namespace tulpa {

// Batched outer-grid driver (dense). Returns an Rcpp::List of length B; element
// s is a List(log_marginal[n_grid], modes[n_grid x n_x], weights[n_grid],
// n_iter[n_grid]). All-coupled cell-coupling families only.
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
    std::shared_ptr<CellCouplingSpec> spec
) {
    const int n_arms = (int) arms.size();
    const int B = n_batch;

    std::vector<int> coupled_arms;
    if (spec) coupled_arms = spec->arm_ids();
    if (coupled_arms.empty()) {
        Rcpp::stop("batched joint path requires a cell-coupling spec with "
                   "coupled arms (occu_cover); none registered.");
    }
    // All arms must be coupled in this first cut.
    for (int k = 0; k < n_arms; k++) {
        bool found = false;
        for (int c : coupled_arms) if (c == k) { found = true; break; }
        if (!found && arms[k].N > 0) {
            Rcpp::stop("batched joint path (gcol33/tulpa#66) requires every "
                       "data arm to be cell-coupled; arm %d is not.", k + 1);
        }
    }

    std::vector<std::vector<std::vector<int>>> cell_rows;
    int n_cells = build_cell_rows_from_arms(arms, coupled_arms, cell_rows);

    int n_x = n_x_after_re;
    for (const auto& b : blocks) n_x = std::max(n_x, b.start + b.size);
    const bool use_sparse = (n_x >= SPARSE_THRESHOLD);

    // Per-species states + persistent etas buffers (species-major) for the
    // fused scatter.
    std::vector<SpeciesState> st(B);
    for (int s = 0; s < B; s++) st[s].allocate(n_x, arms);

    BatchArmBuffers wbuf = buf;  // working copy carries etas (species-major)
    // Per-species accumulators.
    std::vector<std::vector<double>> log_marg(B, std::vector<double>(n_grid,
                                       -std::numeric_limits<double>::infinity()));
    std::vector<std::vector<double>> modes_flat(B,
        std::vector<double>((std::size_t) n_grid * n_x, 0.0));
    std::vector<std::vector<int>> n_iter(B, std::vector<int>(n_grid, 0));
    std::vector<std::vector<double>> prev_mode(B, std::vector<double>(n_x, 0.0));
    std::vector<bool> have_prev(B, false);

    std::vector<DenseVec> grad_per_sp(B);
    std::vector<DenseMat> H_per_sp(B);

    for (int kg = 0; kg < n_grid; kg++) {
        if (prep_at_grid) prep_at_grid(kg);
        bool feasible = true;
        for (const auto& b : blocks) {
            if (b.prep && !b.prep(kg)) { feasible = false; break; }
        }
        if (!feasible) continue;  // -inf log_marginal stays

        std::vector<double> d_fac_cache((int) blocks.size());
        for (int b = 0; b < (int) blocks.size(); b++) d_fac_cache[b] = blocks[b].d_fac(kg);

        // Warm starts.
        for (int s = 0; s < B; s++) {
            if (have_prev[s]) for (int j = 0; j < n_x; j++) st[s].x[j] = prev_mode[s][j];
            else              for (int j = 0; j < n_x; j++) st[s].x[j] = 0.0;
        }

        std::vector<bool> converged(B, false);
        std::vector<double> obj(B, -1e300);
        std::vector<bool> obj_valid(B, false);

        for (int iter = 0; iter < max_iter; iter++) {
            // 1. compute per-species etas into wbuf (species-major).
            for (int s = 0; s < B; s++) {
                if (converged[s]) continue;
                compute_eta_species(st[s].x, st[s].etas, arms, parsed, blocks,
                                    kg, d_fac_cache);
                for (int k = 0; k < n_arms; k++) {
                    const int N_k = arms[k].N;
                    double* dst = wbuf.etas[k].data() + (std::size_t) s * N_k;
                    const double* src = REAL(st[s].etas[k]);
                    for (int i = 0; i < N_k; i++) dst[i] = src[i];
                }
            }
            // 2. zero per-species grad/H, fused scatter.
            for (int s = 0; s < B; s++) {
                grad_per_sp[s].assign(n_x, 0.0);
                H_per_sp[s].assign(n_x, DenseVec(n_x, 0.0));
            }
            scatter_cell_coupling_batch_dense(
                *spec, coupled_arms, cell_rows, n_cells, arms, parsed, blocks,
                kg, wbuf, grad_per_sp, H_per_sp, CurvatureMode::Observed, false);
            // 3. per-species priors + solve + line search + convergence.
            for (int s = 0; s < B; s++) {
                if (converged[s]) continue;
                DenseVec& grad = grad_per_sp[s];
                DenseMat& H = H_per_sp[s];
                for (const auto& b : blocks) if (b.add_prior) b.add_prior(grad, H, st[s].x, kg);
                add_per_arm_beta_re_priors(grad, H, st[s].x, parsed);

                bool ok = dispatch_factor_solve(H, grad, st[s].delta, n_x,
                                                st[s].sparse, use_sparse, st[s].chol);
                if (!ok) {
                    for (int j = 0; j < n_x; j++)
                        if (std::isfinite(st[s].delta[j])) st[s].x[j] += 0.1 * st[s].delta[j];
                    obj_valid[s] = false;
                    n_iter[s][kg] = iter + 1;
                    continue;
                }
                auto eval_obj = [&](const Rcpp::NumericVector& xv) -> double {
                    return species_penalized_logpost(
                        xv, st[s].etas_tmp, *spec, coupled_arms, cell_rows, n_cells,
                        arms, parsed, blocks, kg, d_fac_cache, wbuf, s);
                };
                if (!obj_valid[s]) { obj[s] = eval_obj(st[s].x); obj_valid[s] = true; }
                double step = step_halving_update(st[s].x, st[s].delta, n_x,
                                                  obj[s], eval_obj, obj[s], st[s].x_try);
                n_iter[s][kg] = iter + 1;
                if (max_abs_step(st[s].delta, step, n_x) < tol) converged[s] = true;
            }
            bool all_conv = true;
            for (int s = 0; s < B; s++) if (!converged[s]) { all_conv = false; break; }
            if (all_conv) break;
        }

        // Final mode-pass per species: observed Hessian -> log_det -> log_marginal.
        for (int s = 0; s < B; s++) {
            compute_eta_species(st[s].x, st[s].etas, arms, parsed, blocks, kg, d_fac_cache);
            for (int k = 0; k < n_arms; k++) {
                const int N_k = arms[k].N;
                double* dst = wbuf.etas[k].data() + (std::size_t) s * N_k;
                const double* src = REAL(st[s].etas[k]);
                for (int i = 0; i < N_k; i++) dst[i] = src[i];
            }
        }
        for (int s = 0; s < B; s++) { grad_per_sp[s].assign(n_x, 0.0);
                                      H_per_sp[s].assign(n_x, DenseVec(n_x, 0.0)); }
        scatter_cell_coupling_batch_dense(*spec, coupled_arms, cell_rows, n_cells,
            arms, parsed, blocks, kg, wbuf, grad_per_sp, H_per_sp,
            CurvatureMode::Observed, false);
        for (int s = 0; s < B; s++) {
            DenseVec& grad = grad_per_sp[s]; DenseMat& H = H_per_sp[s];
            for (const auto& b : blocks) if (b.add_prior) b.add_prior(grad, H, st[s].x, kg);
            add_per_arm_beta_re_priors(grad, H, st[s].x, parsed);
            double log_det = 0.0;
            dispatch_factor_log_det(H, n_x, st[s].sparse, use_sparse, st[s].chol, log_det);
            double ll = species_cell_loglik(*spec, coupled_arms, cell_rows, n_cells,
                                            arms, st[s].etas, wbuf, s);
            double lp = log_prior_per_arm_re(st[s].x, parsed);
            for (const auto& b : blocks) if (b.log_prior) lp += b.log_prior(st[s].x, kg);
            log_marg[s][kg] = finalize_log_marginal(ll, lp, log_det, n_x);

            // Center (sum-to-zero) with per-arm intercept compensation, then store.
            for (int b = 0; b < (int) blocks.size(); b++) {
                if (!blocks[b].center) continue;
                double c_b = blocks[b].center(st[s].x);
                if (std::abs(c_b) < 1e-15) continue;
                for (int k_arm = 0; k_arm < n_arms; k_arm++) {
                    if (parsed[k_arm].p == 0) continue;
                    double sc = blocks[b].arm_scale ? blocks[b].arm_scale(k_arm, kg) : 1.0;
                    st[s].x[parsed[k_arm].beta_start] += sc * d_fac_cache[b] * c_b;
                }
            }
            double* mr = modes_flat[s].data() + (std::size_t) kg * n_x;
            for (int j = 0; j < n_x; j++) mr[j] = st[s].x[j];
            for (int j = 0; j < n_x; j++) prev_mode[s][j] = st[s].x[j];
            have_prev[s] = true;
        }
    }

    // Pack per-species results: modes as [n_grid x n_x] row-major matrix.
    Rcpp::List out(B);
    for (int s = 0; s < B; s++) {
        Rcpp::NumericVector lm(n_grid);
        Rcpp::IntegerVector ni(n_grid);
        double mx = -std::numeric_limits<double>::infinity();
        for (int k = 0; k < n_grid; k++) { lm[k] = log_marg[s][k]; ni[k] = n_iter[s][k];
                                           if (std::isfinite(lm[k]) && lm[k] > mx) mx = lm[k]; }
        Rcpp::NumericVector w(n_grid, 0.0);
        double wsum = 0.0;
        for (int k = 0; k < n_grid; k++)
            if (std::isfinite(lm[k])) { w[k] = std::exp(lm[k] - mx); wsum += w[k]; }
        if (wsum > 0) for (int k = 0; k < n_grid; k++) w[k] /= wsum;
        Rcpp::NumericMatrix md(n_grid, n_x);
        for (int k = 0; k < n_grid; k++)
            for (int j = 0; j < n_x; j++)
                md(k, j) = modes_flat[s][(std::size_t) k * n_x + j];
        out[s] = Rcpp::List::create(
            Rcpp::Named("log_marginal") = lm,
            Rcpp::Named("weights")      = w,
            Rcpp::Named("modes")        = md,
            Rcpp::Named("n_iter")       = ni
        );
    }
    return out;
}

} // namespace tulpa
