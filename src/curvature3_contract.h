// curvature3_contract.h
//
// The contraction the inner-Laplace skewness diagnostic performs once the third
// derivative of a log-density stops being a scalar per observation.
//
// inner_laplace_skew.h expands the joint log density along the Gaussian
// conditional-mean curve at a probed latent index i. For a log-density that is a
// separable sum of one-eta terms the cubic coefficient is
//
//   sum_j l_j'''(eta_j) u_{i,j}^3,
//
// which is what the scalar oracle (laplace_spec_curvature3.h) supplies. A UNIT
// whose log-density depends on several linear predictors at once -- a
// zero-inflation mixture's (count, zi) pair, or a CellCouplingSpec cell's arms --
// has no such per-eta term. The SAME expansion along the SAME curve gives the
// wider contraction
//
//   sum_units sum_{a,b,c} T^{abc}_unit u^a u^b u^c,
//
// with T^{abc} = d^3 log p_unit / (d eta_a d eta_b d eta_c) and u^x the eta
// response of coordinate x to v_i = Sigma e_i. The separable case is K = 1: one
// coordinate per unit, T^{111} = l''', and the sum collapses term for term.
//
// T IS NEVER MATERIALISED. Partition the unit's coordinates into K BLOCKS (the
// processes of a multi-process likelihood; the arms of a coupled cell) and write
// u^(a) for u restricted to block a, so u = sum_a u^(a). Then
//
//   sum_{a,b,c} T^{abc} u^a u^b u^c = sum_a d/ds [ u' L''(e + s u^(a)) u ]_{s=0},
//
// because moving along u^(a) differentiates exactly the coordinates of block a,
// each weighted by its own u. Every derivative on the right is ONE central
// difference of the unit's own Hessian -- the quantity the likelihood already
// returns for the Newton solve -- so the whole tensor costs 2K extra evaluations
// of that Hessian per unit and no storage, for any block sizes.
//
// PER-BLOCK STEP. The blocks sit on different scales (an occupancy logit and a
// detection logit do not carry the same curvature magnitude), so a single global
// step is too coarse on one of them. curvature3_block_step() sizes block a's step
// so the ETA displacement it produces on block a is
// CURVATURE3_FD_STEP * max(1, max|eta|) over that block -- the same eta-space
// step the scalar working-weight fallback in laplace_spec_curvature3.h takes.
//
// SYMMETRISATION. Third partials commute, so T[u^a, u^b, u^c] can be read off any
// of the three difference quotients G_a, G_b, G_c, and
// curvature3_symmetrized_sum() averages the three. For this block decomposition
// the symmetrised total is ALGEBRAICALLY the plain total: summing
// (bf[a][b][c] + bf[b][a][c] + bf[c][a][b]) / 3 over all ordered triples relabels
// each of the three sums into sum_{a,b,c} bf[a][b][c]. The averaging is therefore
// exact rather than an approximation, and it is not a variance reduction on the
// finite difference either. What it does buy is robustness at a block whose own
// difference quotient could not be formed (a zero or non-finite step): its
// triples still reach the sum through the two permutations that difference a
// different block, instead of dropping out.

#ifndef TULPA_CURVATURE3_CONTRACT_H
#define TULPA_CURVATURE3_CONTRACT_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace tulpa {

// Relative eta-space finite-difference step for every third-derivative estimate
// that has no analytic ladder: the scalar working-weight fallback
// (laplace_spec_curvature3.h) and the block contraction here.
constexpr double CURVATURE3_FD_STEP = 1e-4;

// Central-difference step along block a's own direction u^(a). `max_abs_eta` and
// `max_abs_u` are the sup norms of eta and of u over that block's coordinates.
// The step is chosen so the eta displacement h * max|u| equals
// CURVATURE3_FD_STEP * max(1, max|eta|). Returns 0 when the block carries no
// direction (u is zero there) or the inputs are not finite -- the caller then
// leaves that block's difference quotient out, and symmetrisation recovers its
// triples from the other two permutations.
inline double curvature3_block_step(double max_abs_eta, double max_abs_u) {
    if (!(max_abs_u > 0.0) || !std::isfinite(max_abs_u)) return 0.0;
    if (!std::isfinite(max_abs_eta)) return 0.0;
    return CURVATURE3_FD_STEP * std::max(1.0, max_abs_eta) / max_abs_u;
}

// Global counterpart of curvature3_block_step(): one step for every block,
// sized off the sup norms taken across the whole unit. Selected by
// `per_arm_step = false` on the cell tensor, which exists so the per-block
// policy can be measured against the coarser one rather than asserted.
//
// One step serves every block, so a single non-finite sup norm makes the shared
// scale unreadable: it returns 0 (the same decline curvature3_block_step
// returns on its own non-finite input) rather than sizing the step off the
// finite entries and applying it to the block that could not be read.
inline double curvature3_global_step(const std::vector<double>& max_abs_eta,
                                     const std::vector<double>& max_abs_u) {
    double e = 0.0, u = 0.0;
    for (double v : max_abs_eta) {
        if (!std::isfinite(v)) return 0.0;
        e = std::max(e, v);
    }
    for (double v : max_abs_u) {
        if (!std::isfinite(v)) return 0.0;
        u = std::max(u, v);
    }
    return curvature3_block_step(e, u);
}

// Permutation-symmetrised contraction of the block difference quotients.
//
// `bf` is the K x K x K array bf[(a * K + b) * K + c] = G_a[u^(b), u^(c)], the
// ordered bilinear form of the a-th difference quotient over blocks (b, c),
// already contracted with u. `have` marks which blocks produced a quotient at
// all (length K); an entry that is false contributes nothing as the DIFFERENCED
// index but is still reachable as one of the other two.
//
// Returns sum over ordered triples of the average of the available permutations,
// or NaN when no triple had any permutation available (never 0, which would read
// as "no skew" rather than "not computable").
inline double curvature3_symmetrized_sum(const std::vector<double>& bf,
                                         const std::vector<char>& have,
                                         int K) {
    double total = 0.0;
    bool any = false;
    for (int a = 0; a < K; a++) {
        for (int b = 0; b < K; b++) {
            for (int c = 0; c < K; c++) {
                double sum = 0.0;
                int n = 0;
                const int perm[3][3] = {{a, b, c}, {b, a, c}, {c, a, b}};
                for (int t = 0; t < 3; t++) {
                    const int d = perm[t][0];
                    if (!have[d]) continue;
                    const double v =
                        bf[(std::size_t)(d * K + perm[t][1]) * K + perm[t][2]];
                    if (!std::isfinite(v)) continue;
                    sum += v;
                    n++;
                }
                if (n == 0) continue;
                total += sum / n;
                any = true;
            }
        }
    }
    if (!any) return std::numeric_limits<double>::quiet_NaN();
    return total;
}

// Cubic contraction at one unit of a multi-coordinate likelihood: given the unit
// index, the unit's coordinates' eta at the mode, and their eta response u,
// return sum_{a,b,c} T^{abc} u^a u^b u^c. NaN when this unit is not computable.
using UnitCubic3Fn = std::function<double(int, const double*, const double*)>;

// The third-derivative oracle for one likelihood, in whichever of the two shapes
// its unit takes. Exactly one of `scalar` (one eta per unit: l'''(eta_j)) and
// `unit` (`n_coords` etas per unit: the contraction above) is set on success;
// `declined` carries the closed-vocabulary reason when neither is, and
// is empty otherwise. Reported alongside the decision rather than re-derived by a
// second predicate, so the reason and the decision cannot drift apart.
struct Curvature3Oracle {
    std::function<double(int, double)> scalar;
    UnitCubic3Fn                       unit;
    int                                n_coords = 1;
    std::string                        declined;

    bool any() const {
        return static_cast<bool>(scalar) || static_cast<bool>(unit);
    }
};

} // namespace tulpa

#endif // TULPA_CURVATURE3_CONTRACT_H
