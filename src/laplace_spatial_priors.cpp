// laplace_spatial_priors.cpp
// Spatial prior gradient/Hessian and log-prior helpers for Laplace engines.

#include "laplace_spatial_priors.h"
#include "sparse_hessian.h"
#include "icar_kernel.h"                  // for_each_icar_component
#include "laplace_s2z.h"                  // add_s2z_pin* / s2z_pin_quad
#include "tulpa/sum_to_zero.h"            // s2z_aug_rank
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

// Sum-to-zero identification of the intrinsic (rank-deficient) ICAR field.
// Q(rho=1) has a constant null-space per component, so (intercept, field-mean)
// is a flat direction of the joint posterior: an informative prior on the
// intercept is evaded by letting the level live in the unaugmented field
// constant. Pinning that constant to ~0 *during* the solve forces the level
// into the intercept, where the prior acts. CAR_proper is full rank (Q(rho)
// identifies the intercept) and is left untouched.
//
// The pin itself -- augmented coefficient, dense/sparse scatter, densify-vs-fold
// storage and the pattern it needs -- is laplace_s2z.h, shared with the temporal
// kernels so both intrinsic families pin the same direction to the same width.

} // anonymous namespace

void add_icar_prior(
    DenseVec& grad, DenseMat& H, const NumericVector& x,
    int spatial_start, int n_spatial_units, double tau_spatial,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors, const GraphPartition& partition
) {
    // ICAR = CAR(rho = 1). The quadratic form is over the whole (block-diagonal
    // for a replicated field) graph; the CSR already carries the per-component
    // edge structure, so no per-component handling is needed here.
    add_car_grad_hess(grad, H, x, spatial_start, n_spatial_units,
                      tau_spatial, /*rho=*/1.0,
                      adj_row_ptr, adj_col_idx, n_neighbors);
    // Augmented Q_aug = Q + sum_c 1_c 1_c'/J_c: the component's constant
    // direction carries the field's own tau (exact rank-1 tau/J_c * 11' on the
    // dense Hessian, over the component's nodes).
    for_each_icar_component(spatial_start, partition,
        [&](int start, const int* idx, int csize) {
            add_s2z_pin(grad, H, x, start, idx, csize, tau_spatial);
        });
}

void add_icar_prior_sparse(
    DenseVec& grad, SparseHessianBuilder& H, const NumericVector& x,
    int spatial_start, int n_spatial_units, double tau_spatial,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors, const GraphPartition& partition
) {
    add_car_grad_hess_sparse(grad, H, x, spatial_start, n_spatial_units,
                              tau_spatial, /*rho=*/1.0,
                              adj_row_ptr, adj_col_idx, n_neighbors);
    // Augmented Q_aug = Q + sum_c 1_c 1_c'/J_c:
    // -0.5 tau sum_c (sum_{i in c} phi_i)^2 / J_c, Hessian (tau/J_c) 11' per
    // component block. Storage is switched per component by size (s2z_densify):
    // a densified component block (laid out by add_icar_pattern) stores the full
    // 11' exactly; a large component leaves the off-diagonals off the stored
    // Hessian and registers a per-component rank-1 11' for the solver to fold in
    // at solve time (Sherman-Morrison step + matrix-determinant-lemma log-det).
    for_each_icar_component(spatial_start, partition,
        [&](int start, const int* idx, int csize) {
            add_s2z_pin_sparse(grad, H, x, start, idx, csize, tau_spatial);
        });
}

void add_icar_pattern(
    std::vector<std::pair<int,int>>& out,
    int spatial_start, int n_spatial_units,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const GraphPartition& partition
) {
    // The adjacency edges over the whole (block-diagonal) graph contain only
    // within-component edges, so the plain adjacency pattern covers every
    // component. Where a component densifies, its sum-to-zero block subsumes
    // those edges and the builder's entry map collapses the overlap; where it
    // does not, the rank-1 11' is folded at solve time and adds no entries.
    add_car_pattern(out, spatial_start, n_spatial_units,
                    adj_row_ptr, adj_col_idx);
    for_each_icar_component(spatial_start, partition,
        [&](int start, const int* idx, int csize) {
            add_s2z_pin_pattern(out, start, idx, csize);
        });
}

double log_prior_icar_structured(
    const NumericVector& x, int spatial_start, int n_spatial_units,
    double tau_spatial,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors, const GraphPartition& partition
) {
    double quad_form = car_quadratic_form(
        x, spatial_start, n_spatial_units, /*rho=*/1.0,
        adj_row_ptr, adj_col_idx, n_neighbors);
    // Augmentation to the quadratic form (matches the gradient in
    // add_icar_prior[_sparse]): tau sum_c (sum_{i in c} phi_i)^2 / J_c.
    double s2z = 0.0;
    for_each_icar_component(spatial_start, partition,
        [&](int start, const int* idx, int csize) {
            s2z += s2z_pin_quad(x, start, idx, csize, tau_spatial);
        });
    // -0.5 tau phi'Q_aug phi, Q_aug = Q + sum_c 1_c 1_c'/J_c.
    return -0.5 * tau_spatial * quad_form - 0.5 * s2z;
}

double log_prior_icar(
    const NumericVector& x, int spatial_start, int n_spatial_units,
    double tau_spatial,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors, const GraphPartition& partition
) {
    // Q is rank (n - n_components), and ICAR's null space is exactly those
    // n_components constants, every one of which the augmentation Q_aug = Q +
    // sum_c 1_c 1_c'/J_c fills. So Q_aug is FULL RANK and all n eigenvalues
    // contribute to log|tau Q_aug|. Keeping the deficient rank here while the
    // quadratic carries the augmentation would make the tau-marginal wrong and
    // bias the variance component low.
    const int L = partition.n_components();
    return log_prior_icar_structured(x, spatial_start, n_spatial_units,
                                     tau_spatial, adj_row_ptr, adj_col_idx,
                                     n_neighbors, partition)
         + 0.5 * s2z_aug_rank(n_spatial_units - L, L)
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
