// nested_laplace_multi.h
// Per-cell properties of a LatentBlock vector the joint driver reads: whether
// the row loadings are cell-invariant, and the cheap screen's Newton depth.

#ifndef TULPA_NESTED_LAPLACE_MULTI_H
#define TULPA_NESTED_LAPLACE_MULTI_H

#include "hard_constraints.h"             // collect_hard_constraints
#include "tulpa_stop.h"
#include "laplace_newton.h"               // SPARSE_THRESHOLD
#include "latent_block.h"
#include "nested_laplace_grid.h"
#include "nested_laplace_joint_core.h"     // make_single_arm
#include "nested_laplace_joint_multi.h"    // the joint driver
#include <Rcpp.h>
#include <algorithm>
#include <vector>

namespace tulpa {

// Whether every row's loading vector is the same at every cell up to the
// per-block scalars d_fac_b(k). A DENSE_BASIS block's weights are its basis
// row at the cell (HSGP folds sqrt(S_j(theta_k)) into weight j), which scales
// each coefficient by its own factor, so rows are classed per cell there.
inline bool nl_loadings_cell_invariant(const std::vector<LatentBlock>& blocks) {
    for (const auto& b : blocks) {
        if (b.contrib_kind == BlockContribKind::DENSE_BASIS) return false;
    }
    return true;
}

// Cell-dependent bases change the coordinates carried by a neighbour's warm
// start. Their screening density uses the converged inner mode, so its ranking
// is independent of that coordinate map.
inline int nl_screen_iterations(const std::vector<LatentBlock>& blocks,
                                 int requested, int max_iter) {
    return nl_loadings_cell_invariant(blocks)
        ? requested : std::max(requested, max_iter);
}

} // namespace tulpa

#endif // TULPA_NESTED_LAPLACE_MULTI_H
