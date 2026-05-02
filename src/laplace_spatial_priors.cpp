// laplace_spatial_priors.cpp
// Spatial prior gradient/Hessian and log-prior helpers for Laplace engines.

#include "laplace_spatial_priors.h"
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

} // anonymous namespace

void add_icar_prior(
    DenseVec& grad, DenseMat& H, const NumericVector& x,
    int spatial_start, int n_spatial_units, double tau_spatial,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors
) {
    // ICAR = CAR(rho = 1).
    add_car_grad_hess(grad, H, x, spatial_start, n_spatial_units,
                      tau_spatial, /*rho=*/1.0,
                      adj_row_ptr, adj_col_idx, n_neighbors);
}

double log_prior_icar(
    const NumericVector& x, int spatial_start, int n_spatial_units,
    double tau_spatial,
    const IntegerVector& adj_row_ptr, const IntegerVector& adj_col_idx,
    const IntegerVector& n_neighbors
) {
    double quad_form = car_quadratic_form(
        x, spatial_start, n_spatial_units, /*rho=*/1.0,
        adj_row_ptr, adj_col_idx, n_neighbors);
    // ICAR is rank-deficient: only (n - 1) eigenvalues contribute to log|tau Q|.
    double lp = -0.5 * tau_spatial * quad_form;
    lp += 0.5 * (n_spatial_units - 1) * std::log(tau_spatial / (2.0 * M_PI));
    return lp;
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
