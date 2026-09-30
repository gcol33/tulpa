// coupled_scatter_plan.h
// Flat-offset plan and fixed cell partition for the cell-coupled sparse
// scatter.
//
// The coupled scatter walks cells and, per cell, writes three kinds of
// Hessian entries: each row's within-row block (its fixed effects, RE group
// and latent dofs against each other), the cross-row products of two rows of
// coupled arms sharing the cell, and the rank-1 self-cross an arm may declare.
// Every entry lies in the fit-level sparsity pattern, and which entries a
// cell reaches depends on the design alone, never on the outer-grid point or
// the latent vector. The plan resolves the flat values[] offset of all of
// them once per fit, so the Newton loop writes `values[slot] += v` and never
// looks an entry up.
//
//   * Row writes reuse the per-observation layout (ArmIndexedCache) and the
//     row body it drives (scatter_row_indexed), so a coupled row and an
//     uncoupled observation are written by the same code.
//   * Cross-row and rank-1 writes read per-cell tables. A cell's dofs split
//     into the coupled arms' fixed effects, shared by every cell (positions
//     [0, G)), and the cell's own RE / latent dofs (positions [G, G + L_c)).
//     `bb` holds the G x G fixed-effect block once, `bl` the G x L_c block and
//     `ll` the L_c x L_c block of each cell.
//
// The cell loop is split into a FIXED number of contiguous chunks,
// `coupled_chunk_count(n_cells)`, a function of the cell count alone. Chunks
// run concurrently when threads are free, and the answer does not depend on
// how many are: an entry written by one chunk only is written in place, in the
// order a single pass would write it; an entry written by several chunks (the
// fixed-effect block, a latent dof several cells share) is accumulated in each
// chunk's own partial and added in chunk order afterwards. The encoding is
// baked into the plan's offsets: `slot >= 0` is an in-place write, `slot <= -2`
// is partial index `-2 - slot`, and `-1` is an entry absent from the pattern.
// At one chunk nothing is shared and every write is the single pass's own.

#ifndef TULPA_COUPLED_SCATTER_PLAN_H
#define TULPA_COUPLED_SCATTER_PLAN_H

#include "hessian_pattern_guard.h"
#include "joint_hessian_pattern.h"
#include "latent_block.h"
#include "nested_laplace_joint_core.h"
#include "omp_threads.h"
#include "scatter_indexed_cache.h"
#include "sparse_hessian.h"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// Cells per chunk before a second chunk is cut, and the most chunks the loop is
// ever split into. A chunk carries a partial of every shared entry, so the
// split stops where a chunk's own work is small against that partial.
inline constexpr int kCoupledCellsPerChunk = 64;
inline constexpr int kCoupledMaxChunks     = 64;

inline int coupled_chunk_count(int n_cells) {
    const int c = n_cells / kCoupledCellsPerChunk;
    return std::max(1, std::min(kCoupledMaxChunks, c));
}

// First cell of chunk `ch` (ch == n_chunks gives n_cells).
inline int coupled_chunk_lo(int ch, int n_chunks, int n_cells) {
    return static_cast<int>(static_cast<long long>(ch) * n_cells / n_chunks);
}

// Run `fn(ch)` for every chunk. Inside a parallel region the chunks become
// tasks, so any team thread with nothing of its own to do takes one -- the
// threads that finished their grid cells drain a long cell's chunks. Outside
// one, a team of up to `n_threads` runs them. `concurrent` false (a coupling
// spec that is not thread-safe) runs them in order on the caller. Which thread
// runs a chunk never changes what it writes, so all three routes give the same
// numbers. A failure inside a chunk is rethrown on the caller once every chunk
// has finished, since an exception may not leave a task or a parallel region.
template <class Fn>
inline void run_coupled_chunks(int n_chunks, int n_threads, bool concurrent,
                               Fn&& fn) {
#ifdef _OPENMP
    const bool nested = omp_in_parallel() != 0;
    if (n_chunks > 1 && concurrent && (nested || n_threads > 1)) {
        std::atomic<bool> failed{false};
        std::string       err;
        auto guarded = [&](int ch) {
            try {
                fn(ch);
            } catch (const std::exception& e) {
                if (!failed.exchange(true)) {
                    #pragma omp critical(tulpa_coupled_chunk_err)
                    err = e.what();
                }
            } catch (...) {
                failed.store(true);
            }
        };
        if (nested) {
            #pragma omp taskgroup
            {
                for (int ch = 0; ch < n_chunks; ch++) {
                    #pragma omp task default(shared) firstprivate(ch)
                    guarded(ch);
                }
            }
        } else {
            const int T = tulpa_omp_team_size_req(n_threads, n_chunks);
            #pragma omp parallel for schedule(dynamic, 1) num_threads(T)
            for (int ch = 0; ch < n_chunks; ch++) guarded(ch);
        }
        if (failed.load()) {
            throw std::runtime_error(
                "coupled-cell chunk failed: " +
                (err.empty() ? std::string("unknown error") : err));
        }
        return;
    }
#else
    (void) n_threads; (void) concurrent;
#endif
    for (int ch = 0; ch < n_chunks; ch++) fn(ch);
}

// One cell's slot tables. Positions [0, G) are the coupled fixed effects,
// [G, G + L) the cell's own dofs in `local` order.
struct CoupledCellSlots {
    const int* bb    = nullptr;  // G x G
    const int* bl    = nullptr;  // G x L, row = fixed-effect position
    const int* ll    = nullptr;  // L x L
    const int* local = nullptr;  // L sorted joint dofs
    int        G     = 0;
    int        L     = 0;

    // Position of a cell-local joint dof, or -1 when the cell does not reach it.
    int local_pos(int dof) const {
        const int* it = std::lower_bound(local, local + L, dof);
        return (it != local + L && *it == dof)
               ? G + static_cast<int>(it - local) : -1;
    }

    // Encoded slot of the entry at positions (pa, pb); symmetric.
    int slot(int pa, int pb) const {
        if (pa < 0 || pb < 0) return -1;
        if (pa < G) {
            if (pb < G) return bb[static_cast<std::size_t>(pa) * G + pb];
            return bl[static_cast<std::size_t>(pa) * L + (pb - G)];
        }
        if (pb < G) return bl[static_cast<std::size_t>(pb) * L + (pa - G)];
        return ll[static_cast<std::size_t>(pa - G) * L + (pb - G)];
    }
};

struct CoupledScatterPlan {
    // Validity key: the pattern the offsets were resolved against.
    std::shared_ptr<const SparseHessianBuilder::EntryMap> pattern;

    int n_cells  = 0;
    int n_chunks = 1;

    // Row layout per coupled arm (index kk into coupled_arms), encoded.
    std::vector<ArmIndexedCache> arm;

    // Coupled fixed effects: joint dof d of arm kk sits at position
    // d + beta_delta[kk].
    int G = 0;
    std::vector<int> beta_delta;
    std::vector<int> bb;

    // Per-cell dofs and tables (flat, offset by cell).
    std::vector<std::size_t> local_off;  // n_cells + 1
    std::vector<int>         local_dof;
    std::vector<std::size_t> bl_off;     // n_cells + 1
    std::vector<int>         bl;
    std::vector<std::size_t> ll_off;     // n_cells + 1
    std::vector<int>         ll;

    // Entries and gradient coordinates more than one chunk writes, in the
    // order their partials are laid out.
    std::vector<int> shared_slot;
    std::vector<int> shared_dof;
    // Per joint dof: -1 when a single chunk writes its gradient coordinate
    // (in place), else its partial index.
    std::vector<int> grad_enc;

    bool valid_for(const SparseHessianBuilder& H, int n_cells_now) const {
        return pattern && pattern == H.entry_map && n_cells == n_cells_now;
    }

    int chunk_lo(int ch) const { return coupled_chunk_lo(ch, n_chunks, n_cells); }

    CoupledCellSlots cell(int c) const {
        CoupledCellSlots s;
        s.bb    = bb.data();
        s.G     = G;
        s.local = local_dof.data() + local_off[c];
        s.L     = static_cast<int>(local_off[c + 1] - local_off[c]);
        s.bl    = bl.data() + bl_off[c];
        s.ll    = ll.data() + ll_off[c];
        return s;
    }
};

// Resolve the plan against the pattern in `H` (after H.init()). The owner walk
// visits every entry a chunk can write -- the design's structural set, which
// contains every entry a Newton step writes, since a zero weight only ever
// removes one -- and marks an entry shared as soon as a second chunk reaches
// it.
inline void build_coupled_scatter_plan(
    const std::vector<ParsedArm>&                     parsed,
    const std::vector<JointArm>&                      arms,
    const std::vector<LatentBlock>&                   blocks,
    const std::vector<int>&                           coupled_arms,
    const std::vector<std::vector<std::vector<int>>>& cell_rows,
    int                                               n_cells,
    int                                               n_x,
    const SparseHessianBuilder&                       H,
    CoupledScatterPlan&                               plan
) {
    plan = CoupledScatterPlan{};
    plan.pattern  = H.entry_map;
    plan.n_cells  = n_cells;
    plan.n_chunks = coupled_chunk_count(n_cells);
    const int n_coupled = static_cast<int>(coupled_arms.size());

    std::vector<std::pair<int,double>> scratch;
    plan.arm.resize(n_coupled);
    for (int kk = 0; kk < n_coupled; kk++) {
        const int k = coupled_arms[kk];
        build_arm_indexed_cache(parsed[k], arms[k].N, k, blocks, H,
                                /*with_bilinear=*/false, plan.arm[kk], scratch);
    }

    std::vector<int> beta_dof;
    plan.beta_delta.resize(n_coupled);
    for (int kk = 0; kk < n_coupled; kk++) {
        const ParsedArm& pa = parsed[coupled_arms[kk]];
        plan.beta_delta[kk] = static_cast<int>(beta_dof.size()) - pa.beta_start;
        for (int j = 0; j < pa.p; j++) beta_dof.push_back(pa.beta_start + j);
    }
    const int G = static_cast<int>(beta_dof.size());
    plan.G = G;
    plan.bb.resize(static_cast<std::size_t>(G) * G);
    for (int a = 0; a < G; a++)
        for (int b = 0; b < G; b++)
            plan.bb[static_cast<std::size_t>(a) * G + b] =
                H.lookup(beta_dof[a], beta_dof[b]);

    plan.local_off.assign(n_cells + 1, 0);
    plan.bl_off.assign(n_cells + 1, 0);
    plan.ll_off.assign(n_cells + 1, 0);
    std::vector<int> arm_dofs, cell_dofs;
    for (int c = 0; c < n_cells; c++) {
        cell_dofs.clear();
        for (int kk = 0; kk < n_coupled; kk++) {
            const int k = coupled_arms[kk];
            coupled_cell_arm_dofs(parsed[k], k, blocks, cell_rows[kk][c],
                                  /*with_beta=*/false, scratch, arm_dofs);
            cell_dofs.insert(cell_dofs.end(), arm_dofs.begin(), arm_dofs.end());
        }
        std::sort(cell_dofs.begin(), cell_dofs.end());
        cell_dofs.erase(std::unique(cell_dofs.begin(), cell_dofs.end()),
                        cell_dofs.end());
        const int L = static_cast<int>(cell_dofs.size());
        plan.local_dof.insert(plan.local_dof.end(), cell_dofs.begin(),
                              cell_dofs.end());
        for (int a = 0; a < G; a++)
            for (int l = 0; l < L; l++)
                plan.bl.push_back(H.lookup(beta_dof[a], cell_dofs[l]));
        for (int a = 0; a < L; a++)
            for (int b = 0; b < L; b++)
                plan.ll.push_back(H.lookup(cell_dofs[a], cell_dofs[b]));
        plan.local_off[c + 1] = plan.local_dof.size();
        plan.bl_off[c + 1]    = plan.bl.size();
        plan.ll_off[c + 1]    = plan.ll.size();
    }

    // Owner walk: -1 unwritten, ch >= 0 written by chunk ch alone, -2 shared.
    std::vector<int> h_own(H.values.size(), -1);
    std::vector<int> g_own(static_cast<std::size_t>(n_x), -1);
    auto mark = [](std::vector<int>& own, int s, int ch) {
        if (s < 0) return;
        int& o = own[static_cast<std::size_t>(s)];
        if (o == -1)      o = ch;
        else if (o != ch) o = -2;
    };
    auto mark_range = [&](std::vector<int>& own, const std::vector<int>& v,
                          std::size_t lo, std::size_t hi, int ch) {
        for (std::size_t t = lo; t < hi; t++) mark(own, v[t], ch);
    };
    for (int ch = 0; ch < plan.n_chunks; ch++) {
        const int c_lo = plan.chunk_lo(ch);
        const int c_hi = plan.chunk_lo(ch + 1);
        if (c_lo == c_hi) continue;
        mark_range(h_own, plan.bb, 0, plan.bb.size(), ch);
        for (int kk = 0; kk < n_coupled; kk++) {
            const ParsedArm&       pa = parsed[coupled_arms[kk]];
            const ArmIndexedCache& ac = plan.arm[kk];
            const int p_k = pa.p;
            bool any_row = false;
            for (int c = c_lo; c < c_hi; c++) {
                for (int row : cell_rows[kk][c]) {
                    any_row = true;
                    const PerObsScatterPlan& pl = ac.plans[row];
                    const std::size_t A = static_cast<std::size_t>(pl.A_idx);
                    for (int j = 0; j < p_k; j++)
                        mark(g_own, pa.beta_start + j, ch);
                    if (pl.g_re_global >= 0) {
                        const int g_local = pl.g_re_global - pa.re_start;
                        mark(g_own, pl.g_re_global, ch);
                        mark(h_own, ac.idx_re_diag[g_local], ch);
                        mark_range(h_own, ac.idx_beta_re,
                                   static_cast<std::size_t>(g_local) * p_k,
                                   static_cast<std::size_t>(g_local + 1) * p_k,
                                   ch);
                        mark_range(h_own, ac.idx_re_active, pl.rxa_start,
                                   pl.rxa_start + A, ch);
                    }
                    for (std::size_t a = 0; a < A; a++)
                        mark(g_own, ac.active_dof_global[pl.act_start + a], ch);
                    mark_range(h_own, ac.idx_beta_active, pl.bxa_start,
                               pl.bxa_start + static_cast<std::size_t>(p_k) * A,
                               ch);
                    mark_range(h_own, ac.idx_act_act, pl.axa_start,
                               pl.axa_start + A * (A + 1) / 2, ch);
                }
            }
            if (any_row) mark_range(h_own, ac.idx_bb, 0, ac.idx_bb.size(), ch);
        }
        for (int c = c_lo; c < c_hi; c++) {
            mark_range(h_own, plan.bl, plan.bl_off[c], plan.bl_off[c + 1], ch);
            mark_range(h_own, plan.ll, plan.ll_off[c], plan.ll_off[c + 1], ch);
        }
    }

    for (std::size_t s = 0; s < h_own.size(); s++) {
        if (h_own[s] == -2) {
            h_own[s] = -2 - static_cast<int>(plan.shared_slot.size());
            plan.shared_slot.push_back(static_cast<int>(s));
        }
    }
    plan.grad_enc.assign(static_cast<std::size_t>(n_x), -1);
    for (int d = 0; d < n_x; d++) {
        if (g_own[static_cast<std::size_t>(d)] == -2) {
            plan.grad_enc[static_cast<std::size_t>(d)] =
                static_cast<int>(plan.shared_dof.size());
            plan.shared_dof.push_back(d);
        }
    }

    auto encode = [&](std::vector<int>& v) {
        for (int& s : v) {
            if (s >= 0 && h_own[static_cast<std::size_t>(s)] <= -2)
                s = h_own[static_cast<std::size_t>(s)];
        }
    };
    for (ArmIndexedCache& ac : plan.arm) {
        encode(ac.idx_bb);
        encode(ac.idx_re_diag);
        encode(ac.idx_beta_re);
        encode(ac.idx_beta_active);
        encode(ac.idx_re_active);
        encode(ac.idx_act_act);
    }
    encode(plan.bb);
    encode(plan.bl);
    encode(plan.ll);
}

// Writes one chunk's contributions: an entry the chunk owns in place, a shared
// one into the chunk's partial, an absent one discarded and counted.
struct CoupledChunkSink {
    double* __restrict__    grad;
    double* __restrict__    Hv;
    double* __restrict__    Hp;
    double* __restrict__    gp;
    const int* __restrict__ genc;

    void hess(int e, double v) const {
        if (e >= 0)       Hv[e] += v;
        else if (e <= -2) Hp[-2 - e] += v;
        else if (v != 0.0) record_hessian_pattern_drop();
    }
    void grad_add(int d, double v) const {
        const int t = genc[d];
        if (t < 0) grad[d] += v;
        else       gp[t] += v;
    }
};

// Per-chunk partials of the shared entries, for one Hessian and gradient.
class CoupledChunkPartials {
public:
    explicit CoupledChunkPartials(const CoupledScatterPlan& plan)
        : plan_(plan),
          n_h_(plan.shared_slot.size()),
          n_g_(plan.shared_dof.size()),
          h_(static_cast<std::size_t>(plan.n_chunks) * n_h_, 0.0),
          g_(static_cast<std::size_t>(plan.n_chunks) * n_g_, 0.0) {}

    CoupledChunkSink sink(int ch, double* grad, double* Hv) {
        return CoupledChunkSink{grad, Hv,
                                h_.data() + static_cast<std::size_t>(ch) * n_h_,
                                g_.data() + static_cast<std::size_t>(ch) * n_g_,
                                plan_.grad_enc.data()};
    }

    // Add every chunk's partials into the targets, chunk by chunk, so each
    // shared entry receives its chunk sums in chunk order.
    void reduce(double* grad, double* Hv) const {
        for (int ch = 0; ch < plan_.n_chunks; ch++) {
            const double* hp = h_.data() + static_cast<std::size_t>(ch) * n_h_;
            for (std::size_t t = 0; t < n_h_; t++)
                Hv[plan_.shared_slot[t]] += hp[t];
            const double* gp = g_.data() + static_cast<std::size_t>(ch) * n_g_;
            for (std::size_t t = 0; t < n_g_; t++)
                grad[plan_.shared_dof[t]] += gp[t];
        }
    }

private:
    const CoupledScatterPlan& plan_;
    std::size_t               n_h_;
    std::size_t               n_g_;
    std::vector<double>       h_;
    std::vector<double>       g_;
};

} // namespace tulpa

#endif // TULPA_COUPLED_SCATTER_PLAN_H
