// laplace_spatial_priors.cpp
// Spatial prior gradient/Hessian and log-prior helpers for Laplace engines.

#include "laplace_spatial_priors.h"
#include "sparse_hessian.h"
#include <cmath>

using namespace Rcpp;

namespace tulpa {

namespace {

// Shared kernel: add tau * Q(rho) contributions to (grad, H) for any
// CAR/ICAR-shaped precision Q(rho) = D - rho*W. ICAR is the special case
// rho = 1.0; proper-CAR uses rho in (rho_lower, rho_upper).
inline void add_car_grad_hess(
    DenseVec& grad, DenseMat& H, const NumericVector& x,
    int spatial_start, int n_spatial_units, double tau, double rho,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors
) {
    for (int s = 0; s < n_spatial_units; s++) {
        int sp_idx = spatial_start + s;
        double phi_s = x[sp_idx];

        double neighbor_sum = 0.0;
        for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
            int neighbor = adj_col_idx[k];
            neighbor_sum += x[spatial_start + neighbor];
        }
        grad[sp_idx] -= tau * (n_neighbors[s] * phi_s - rho * neighbor_sum);
        H[sp_idx][sp_idx] += tau * n_neighbors[s];

        for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
            int neighbor = adj_col_idx[k];
            H[sp_idx][spatial_start + neighbor] -= tau * rho;
        }
    }
}

// Sparse twin of `add_car_grad_hess`. Same math; writes lower-triangle
// entries into a SparseHessianBuilder via add(row, col, val) which
// internally normalizes orientation. We pass each (s, neighbor) pair once
// (skip neighbor < s, since the same edge is visited from both ends) to
// avoid double-counting off-diagonal contributions.
inline void add_car_grad_hess_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const NumericVector& x,
    int spatial_start, int n_spatial_units, double tau, double rho,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors
) {
    for (int s = 0; s < n_spatial_units; s++) {
        int sp_idx = spatial_start + s;
        double phi_s = x[sp_idx];

        double neighbor_sum = 0.0;
        for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
            int neighbor = adj_col_idx[k];
            neighbor_sum += x[spatial_start + neighbor];
        }
        grad[sp_idx] -= tau * (n_neighbors[s] * phi_s - rho * neighbor_sum);
        H.add(sp_idx, sp_idx, tau * n_neighbors[s]);

        for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
            int neighbor = adj_col_idx[k];
            if (neighbor < s) continue;  // visit each edge once
            H.add(sp_idx, spatial_start + neighbor, -tau * rho);
        }
    }
}

// Shared kernel: phi' Q(rho) phi for the same Q family.
// Uses the symmetry of the adjacency to halve work (only neighbor > s).
inline double car_quadratic_form(
    const NumericVector& x, int spatial_start, int n_spatial_units, double rho,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors
) {
    double quad_form = 0.0;
    for (int s = 0; s < n_spatial_units; s++) {
        double phi_s = x[spatial_start + s];
        quad_form += n_neighbors[s] * phi_s * phi_s;
        for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
            int neighbor = adj_col_idx[k];
            if (neighbor > s) {
                quad_form -= 2.0 * rho * phi_s * x[spatial_start + neighbor];
            }
        }
    }
    return quad_form;
}

// Penalty-method sum-to-zero constraint for the intrinsic (rank-deficient)
// ICAR field. Q(rho=1) has a constant null-space, so (intercept, field-mean)
// is a flat direction of the joint posterior: an informative prior on the
// intercept is evaded by letting the level live in the unpenalised field
// constant, and a post-hoc centering then folds that constant back into the
// intercept -- so the intercept prior never regularises the level. Pinning the
// field constant to ~0 *during* the solve (this penalty) forces the level into
// the intercept, where the prior acts. Besag-standard ICAR treatment; mirrors
// the (sum field)^2 penalty the N-mixture spatial path uses. CAR_proper is full
// rank (Q(rho) identifies the intercept) and is left untouched.
//
// The penalty is 0.5 * SUM2ZERO_TAU * (sum_s phi_s)^2. Its Hessian is the rank-1
// SUM2ZERO_TAU * 11', which lives entirely in the constant eigenspace and is
// orthogonal to every spatial deviation, so it pins the mean without shrinking
// the field pattern. The dense path adds it exactly. The sparse path adds the
// exact gradient plus the diagonal of 11' (= SUM2ZERO_TAU per node, the only
// part that fits the fixed adjacency pattern) for a stable quasi-Newton step;
// the off-diagonals are dropped, but the line search on the exact penalty
// converges to the same constrained mode. A modest precision suffices: the
// pinned field mean is s / n with s ~ data / (SUM2ZERO_TAU * n), so the level
// in the field shrinks like 1 / (SUM2ZERO_TAU * n^2) -- already ~0 at n^2 scale.
constexpr double SUM2ZERO_TAU = 1.0;

inline double field_sum(const NumericVector& x, int start, int n) {
    double s = 0.0;
    for (int j = 0; j < n; j++) s += x[start + j];
    return s;
}

// Visit each of an intrinsic field's `n_components` disjoint, equal-size,
// contiguous connected components, calling comp(comp_start_absolute,
// comp_size). A connected graph has one component spanning [start, start + n)
// (n_components <= 1, byte-identical to the historical single-component path);
// a replicated field over the block-diagonal I_L (x) Q has L equal-size
// components (the `by =` replicated CAR). The single source of the per-component
// loop so the gradient, Hessian, pattern and log-prior treat the null space
// identically.
template <typename F>
inline void for_each_icar_component(int start, int n, int n_components, F&& comp) {
    const int L = (n_components > 1) ? n_components : 1;
    const int csize = n / L;
    for (int c = 0; c < L; c++) comp(start + c * csize, csize);
}

} // anonymous namespace

void add_icar_prior(
    DenseVec& grad, DenseMat& H, const NumericVector& x,
    int spatial_start, int n_spatial_units, double tau_spatial,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors, int n_components
) {
    // ICAR = CAR(rho = 1). The quadratic form is over the whole (block-diagonal
    // for a replicated field) graph; the CSR already carries the per-component
    // edge structure, so no per-component handling is needed here.
    add_car_grad_hess(grad, H, x, spatial_start, n_spatial_units,
                      tau_spatial, /*rho=*/1.0,
                      adj_row_ptr, adj_col_idx, n_neighbors);
    // One sum-to-zero penalty per connected component (exact rank-1 11' on the
    // dense Hessian, within the component's diagonal block).
    for_each_icar_component(spatial_start, n_spatial_units, n_components,
        [&](int cstart, int csize) {
            double s = field_sum(x, cstart, csize);
            for (int i = 0; i < csize; i++) {
                grad[cstart + i] -= SUM2ZERO_TAU * s;
                for (int j = 0; j < csize; j++)
                    H[cstart + i][cstart + j] += SUM2ZERO_TAU;
            }
        });
}

void add_icar_prior_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const NumericVector& x,
    int spatial_start, int n_spatial_units, double tau_spatial,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors, int n_components
) {
    add_car_grad_hess_sparse(grad, H, x, spatial_start, n_spatial_units,
                              tau_spatial, /*rho=*/1.0,
                              adj_row_ptr, adj_col_idx, n_neighbors);
    // One sum-to-zero penalty per connected component:
    // -0.5 SUM2ZERO_TAU sum_c (sum_{i in c} phi_i)^2, Hessian SUM2ZERO_TAU 11'
    // per component block. The exact gradient is always added. The rank-1 11'
    // Hessian is handled by per-component field size (see icar_s2z_densify): a
    // densified component block (laid out by add_icar_pattern) stores the full
    // 11' exactly; a large component leaves the off-diagonals off the stored
    // Hessian and registers a per-component rank-1 11' for the solver to fold in
    // at solve time (Sherman-Morrison step + matrix-determinant-lemma log-det).
    for_each_icar_component(spatial_start, n_spatial_units, n_components,
        [&](int cstart, int csize) {
            const double s = field_sum(x, cstart, csize);
            for (int i = 0; i < csize; i++)
                grad[cstart + i] -= SUM2ZERO_TAU * s;
            if (icar_s2z_densify(csize)) {
                for (int i = 0; i < csize; i++)
                    for (int j = 0; j <= i; j++)
                        H.add(cstart + i, cstart + j, SUM2ZERO_TAU);
            } else {
                H.add_s2z_rank1(cstart, csize, SUM2ZERO_TAU);
            }
        });
}

void add_icar_pattern(
    std::vector<std::pair<int,int>>& out,
    int spatial_start, int n_spatial_units,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    int n_components
) {
    const int L = (n_components > 1) ? n_components : 1;
    if (icar_s2z_densify(n_spatial_units / L)) {
        // Dense lower-triangle block per component so each component's
        // sum-to-zero 11' fits exactly.
        for_each_icar_component(spatial_start, n_spatial_units, n_components,
            [&](int cstart, int csize) {
                for (int i = 0; i < csize; i++)
                    for (int j = 0; j <= i; j++)
                        out.emplace_back(cstart + i, cstart + j);
            });
    } else {
        // The adjacency edges over the whole (block-diagonal) graph already
        // contain only within-component edges, so the plain adjacency pattern
        // covers every component; the per-component rank-1 11' is folded at
        // solve time (off the stored pattern).
        add_car_pattern(out, spatial_start, n_spatial_units,
                        adj_row_ptr, adj_col_idx);
    }
}

double log_prior_icar_structured(
    const NumericVector& x, int spatial_start, int n_spatial_units,
    double tau_spatial,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors, int n_components
) {
    double quad_form = car_quadratic_form(
        x, spatial_start, n_spatial_units, /*rho=*/1.0,
        adj_row_ptr, adj_col_idx, n_neighbors);
    // One sum-to-zero penalty per connected component (matches the gradient in
    // add_icar_prior[_sparse]): sum_c (sum_{i in c} phi_i)^2.
    double s2z = 0.0;
    for_each_icar_component(spatial_start, n_spatial_units, n_components,
        [&](int cstart, int csize) {
            double s = field_sum(x, cstart, csize);
            s2z += s * s;
        });
    // -0.5 tau phi'Q phi (the intrinsic quadratic) and the per-component
    // sum-to-zero penalty that pins the constant null-space.
    return -0.5 * tau_spatial * quad_form - 0.5 * SUM2ZERO_TAU * s2z;
}

double log_prior_icar(
    const NumericVector& x, int spatial_start, int n_spatial_units,
    double tau_spatial,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors, int n_components
) {
    // ICAR is rank-deficient: Q over a graph with `n_components` connected
    // components has rank (n - n_components), so only that many eigenvalues
    // contribute to log|tau Q| (one constant null direction per component).
    return log_prior_icar_structured(x, spatial_start, n_spatial_units,
                                     tau_spatial, adj_row_ptr, adj_col_idx,
                                     n_neighbors, n_components)
         + 0.5 * (n_spatial_units - n_components)
               * std::log(tau_spatial / (2.0 * M_PI));
}

void add_car_proper_prior(
    DenseVec& grad, DenseMat& H, const NumericVector& x,
    int spatial_start, int n_spatial_units, double tau, double rho,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors
) {
    add_car_grad_hess(grad, H, x, spatial_start, n_spatial_units,
                      tau, rho,
                      adj_row_ptr, adj_col_idx, n_neighbors);
}

void add_car_proper_prior_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const NumericVector& x,
    int spatial_start, int n_spatial_units, double tau, double rho,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors
) {
    add_car_grad_hess_sparse(grad, H, x, spatial_start, n_spatial_units,
                              tau, rho,
                              adj_row_ptr, adj_col_idx, n_neighbors);
}

void add_car_pattern(
    std::vector<std::pair<int,int>>& out,
    int spatial_start, int n_spatial_units,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx
) {
    // Diagonal entries are added unconditionally by the pattern builder.
    // We only emit the off-diagonal adjacency edges (s, neighbor), s != neighbor.
    // Normalize to lower triangle (hi >= lo) — SparseHessianBuilder dedupes.
    for (int s = 0; s < n_spatial_units; s++) {
        int sp_idx = spatial_start + s;
        for (int k = adj_row_ptr[s]; k < adj_row_ptr[s + 1]; k++) {
            int neighbor = adj_col_idx[k];
            if (neighbor == s) continue;
            int n_idx = spatial_start + neighbor;
            int hi = (sp_idx > n_idx) ? sp_idx : n_idx;
            int lo = (sp_idx > n_idx) ? n_idx : sp_idx;
            out.emplace_back(hi, lo);
        }
    }
}

double log_prior_car_proper(
    const NumericVector& x, int spatial_start, int n_spatial_units,
    double tau, double rho, double log_det_Q_rho,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors
) {
    double quad_form = car_quadratic_form(
        x, spatial_start, n_spatial_units, rho,
        adj_row_ptr, adj_col_idx, n_neighbors);
    // log p(phi | tau, rho) = 0.5 * log|tau * Q(rho)| - 0.5 * tau * phi'Qphi
    double lp = 0.5 * log_det_Q_rho
              + 0.5 * n_spatial_units * std::log(tau)
              - 0.5 * tau * quad_form
              - 0.5 * n_spatial_units * std::log(2.0 * M_PI);
    return lp;
}

} // namespace tulpa
