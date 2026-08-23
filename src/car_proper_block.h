// car_proper_block.h
// Prior callbacks for a proper-CAR latent block.
//
// Q(tau, rho) = tau (D - rho W) is full rank for rho inside the adjacency
// eigenvalue interval, so the block needs a per-cell feasibility gate and a
// per-cell log|Q| the density reads. Those five callbacks -- prep, the dense
// and sparse prior scatters, the log-density and the sparsity pattern -- are
// the same at every entry that carries such a block; only where the cell's
// (tau, rho) come from differs, which is what the two accessors carry:
//
//   * the single-arm kernel reads them off its own tau_grid / rho_grid,
//   * the joint driver's non-copy branch off two theta_grid columns,
//   * its copy branch off one theta_grid column at tau = 1, the amplitude
//     riding arm_scale instead.
//
// No centerer is set. Q is full rank, so the field carries no null direction
// to identify against the intercept, and shifting it would report a
// (field, intercept) pair whose joint posterior density is below the mode's.

#ifndef TULPA_CAR_PROPER_BLOCK_H
#define TULPA_CAR_PROPER_BLOCK_H

#include "latent_block.h"
#include "laplace_spatial_priors.h"
#include "hmc_car_proper.h"
#include "nl_cell_cache.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <cmath>
#include <memory>
#include <vector>

namespace tulpa {

// Fill `block`'s prep / add_prior / add_prior_sparse / log_prior /
// add_prior_pattern for a proper CAR over [start, start + size).
//
// `tau_at(k)` and `rho_at(k)` return the cell's hyperparameters. The adjacency
// is captured BY VALUE: an Rcpp vector is a handle onto a preserved SEXP, so
// the copy is cheap and the closures do not depend on the caller's locals
// outliving the fit.
//
// log|Q(rho)| is cell-keyed (NlCellCache) rather than a scalar: prep runs
// lock-free in the parallel joint grid driver, so a shared scalar would race a
// concurrent cell's log_prior read. The single-arm kernels run their grid
// serially today, but nothing at the type level stops such a block from
// meeting a parallel one, and cell-keyed state costs nothing.
template <class TauAt, class RhoAt>
inline void set_car_proper_block_priors(
    LatentBlock& block, int start, int size,
    TauAt tau_at, RhoAt rho_at,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors
) {
    // The dense log|Q(rho)| helper takes plain int vectors; owned by
    // shared_ptr so the prep closure carries no reference to a caller local.
    auto adj_rp_v = std::make_shared<std::vector<int>>(adj_row_ptr.begin(),
                                                       adj_row_ptr.end());
    auto adj_ci_v = std::make_shared<std::vector<int>>(adj_col_idx.begin(),
                                                       adj_col_idx.end());
    auto n_nbr_v  = std::make_shared<std::vector<int>>(n_neighbors.begin(),
                                                       n_neighbors.end());
    auto log_det_Q_rho = std::make_shared<NlCellCache<double>>();

    block.prep = [size, rho_at, adj_rp_v, adj_ci_v, n_nbr_v,
                  log_det_Q_rho](int k) -> bool {
        std::vector<double> Qmat = tulpa_car_proper::compute_car_precision(
            size, *adj_rp_v, *adj_ci_v, *n_nbr_v, rho_at(k));
        const double ld_val = tulpa_car_proper::car_log_det(size, Qmat);
        log_det_Q_rho->claim() = ld_val;
        log_det_Q_rho->publish(k);
        return std::isfinite(ld_val);
    };
    block.add_prior = [start, size, tau_at, rho_at,
                       adj_row_ptr, adj_col_idx, n_neighbors](
        DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x, int k) {
        add_car_proper_prior(grad, H, x, start, size, tau_at(k), rho_at(k),
                             adj_row_ptr, adj_col_idx, n_neighbors);
    };
    block.add_prior_sparse = [start, size, tau_at, rho_at,
                              adj_row_ptr, adj_col_idx, n_neighbors](
        SparseHessianBuilder& H, DenseVec& grad,
        const Rcpp::NumericVector& x, int k) {
        add_car_proper_prior_sparse(grad, H, x, start, size,
                                    tau_at(k), rho_at(k),
                                    adj_row_ptr, adj_col_idx, n_neighbors);
    };
    block.log_prior = [start, size, tau_at, rho_at, adj_row_ptr, adj_col_idx,
                       n_neighbors, log_det_Q_rho](
        const Rcpp::NumericVector& x, int k) -> double {
        return log_prior_car_proper(x, start, size, tau_at(k), rho_at(k),
                                    log_det_Q_rho->find(k),
                                    adj_row_ptr, adj_col_idx, n_neighbors);
    };
    block.add_prior_pattern = [start, size, adj_row_ptr, adj_col_idx](
        std::vector<std::pair<int,int>>& out) {
        add_car_pattern(out, start, size, adj_row_ptr, adj_col_idx);
    };
}

} // namespace tulpa

#endif // TULPA_CAR_PROPER_BLOCK_H
