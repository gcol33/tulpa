// field_block_priors.h
// Prior callbacks for the intrinsic-CAR and the indexed temporal latent blocks.
//
// An ICAR field (plain, or BYM2's structured component) and an RW1 / RW2 / AR1
// chain carry the same four prior callbacks at every entry that builds one --
// the dense and sparse scatters, the sparsity pattern and the log-density --
// and differ only in where a cell's hyperparameters come from, which the
// accessors carry:
//
//   * the single-arm kernels read them off their own tau / rho grids,
//   * the multi-block and joint drivers off theta_grid columns,
//   * a joint copy block holds tau at 1, the amplitude riding arm_scale.
//
// Filling all four from here keeps the sparse twin of a block present wherever
// its dense scatter is, so whichever Newton path a driver takes, the prior is
// the same. Centering is left to the caller: whether a block's level is folded
// into an intercept, a covariate coefficient or a projected design is a
// property of where the block sits, not of its prior.
//
// Rcpp vectors are captured BY VALUE: each is a handle onto a preserved SEXP,
// so the copy is cheap and the closures do not depend on the caller's locals
// outliving the fit.

#ifndef TULPA_FIELD_BLOCK_PRIORS_H
#define TULPA_FIELD_BLOCK_PRIORS_H

#include "latent_block.h"
#include "laplace_spatial_priors.h"
#include "laplace_temporal_priors.h"
#include "sparse_hessian.h"
#include "tulpa/graph_components.h"
#include <Rcpp.h>
#include <string>
#include <utility>
#include <vector>

namespace tulpa {

// Intrinsic CAR over [start, start + size) at precision tau_at(k).
//
// `node_prec` is the per-node precision multiplier (empty: none), the
// per-component BYM2 scaling (gcol33/tulpa#902). `structured` selects the
// density without the rank-deficient normalizer (log_prior_icar_structured),
// which is the one BYM2's structured component carries; a plain ICAR field
// carries log_prior_icar.
template <class TauAt>
inline void set_icar_block_priors(
    LatentBlock& block, int start, int size, TauAt tau_at,
    Rcpp::IntegerVector adj_row_ptr,
    Rcpp::IntegerVector adj_col_idx,
    Rcpp::IntegerVector n_neighbors,
    GraphPartition partition,
    std::vector<double> node_prec = std::vector<double>(),
    bool structured = false
) {
    block.add_prior = [start, size, tau_at, adj_row_ptr, adj_col_idx,
                       n_neighbors, partition, node_prec](
        DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x, int k) {
        add_icar_prior(grad, H, x, start, size, tau_at(k), adj_row_ptr,
                       adj_col_idx, n_neighbors, partition,
                       node_prec_ptr(node_prec));
    };
    block.add_prior_sparse = [start, size, tau_at, adj_row_ptr, adj_col_idx,
                              n_neighbors, partition, node_prec](
        SparseHessianBuilder& H, DenseVec& grad,
        const Rcpp::NumericVector& x, int k) {
        add_icar_prior_sparse(grad, H, x, start, size, tau_at(k), adj_row_ptr,
                              adj_col_idx, n_neighbors, partition,
                              node_prec_ptr(node_prec));
    };
    block.add_prior_pattern = [start, size, adj_row_ptr, adj_col_idx,
                               partition](std::vector<std::pair<int,int>>& out) {
        add_icar_pattern(out, start, size, adj_row_ptr, adj_col_idx, partition);
    };
    block.log_prior = [start, size, tau_at, adj_row_ptr, adj_col_idx,
                       n_neighbors, partition, node_prec, structured](
        const Rcpp::NumericVector& x, int k) -> double {
        if (structured) {
            return log_prior_icar_structured(x, start, size, tau_at(k),
                                             adj_row_ptr, adj_col_idx,
                                             n_neighbors, partition,
                                             node_prec_ptr(node_prec));
        }
        return log_prior_icar(x, start, size, tau_at(k), adj_row_ptr,
                              adj_col_idx, n_neighbors, partition,
                              node_prec_ptr(node_prec));
    };
    block.prior_kind = PriorFillKind::ADJACENCY;
}

// Indexed temporal chains over [start, start + n_groups * n_times): one walk
// of n_times per group, the walks never connected across a group boundary.
// `type` is "rw1", "rw2" or "ar1"; rho_at is read for "ar1" only and `cyclic`
// closes each RW chain into a ring. RW1 / RW2 carry the sum-to-zero pin that
// identifies their level (add_rw*_field); AR1 is proper at |rho| < 1.
template <class TauAt, class RhoAt>
inline void set_temporal_block_priors(
    LatentBlock& block, const std::string& type,
    int start, int n_groups, int n_times,
    TauAt tau_at, RhoAt rho_at, bool cyclic
) {
    if (type == "rw1" || type == "rw2") {
        const bool rw2 = (type == "rw2");
        block.add_prior = [start, n_groups, n_times, tau_at, cyclic, rw2](
            DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x, int k) {
            if (rw2) add_rw2_field(grad, H, x, start, n_groups, n_times,
                                   tau_at(k), cyclic);
            else     add_rw1_field(grad, H, x, start, n_groups, n_times,
                                   tau_at(k), cyclic);
        };
        block.add_prior_sparse = [start, n_groups, n_times, tau_at, cyclic,
                                  rw2](SparseHessianBuilder& H, DenseVec& grad,
                                       const Rcpp::NumericVector& x, int k) {
            if (rw2) add_rw2_field_sparse(grad, H, x, start, n_groups, n_times,
                                          tau_at(k), cyclic);
            else     add_rw1_field_sparse(grad, H, x, start, n_groups, n_times,
                                          tau_at(k), cyclic);
        };
        block.add_prior_pattern = [start, n_groups, n_times, cyclic, rw2](
            std::vector<std::pair<int,int>>& out) {
            if (rw2) add_rw2_field_pattern(out, start, n_groups, n_times, cyclic);
            else     add_rw1_field_pattern(out, start, n_groups, n_times, cyclic);
        };
        block.log_prior = [start, n_groups, n_times, tau_at, cyclic, rw2](
            const Rcpp::NumericVector& x, int k) -> double {
            return rw2 ? log_prior_rw2_field(x, start, n_groups, n_times,
                                             tau_at(k), cyclic)
                       : log_prior_rw1_field(x, start, n_groups, n_times,
                                             tau_at(k), cyclic);
        };
    } else if (type == "ar1") {
        block.add_prior = [start, n_groups, n_times, tau_at, rho_at](
            DenseVec& grad, DenseMat& H, const Rcpp::NumericVector& x, int k) {
            for (int g = 0; g < n_groups; g++)
                add_ar1_precision(grad, H, x, start + g * n_times, n_times,
                                  tau_at(k), rho_at(k));
        };
        block.add_prior_sparse = [start, n_groups, n_times, tau_at, rho_at](
            SparseHessianBuilder& H, DenseVec& grad,
            const Rcpp::NumericVector& x, int k) {
            for (int g = 0; g < n_groups; g++)
                add_ar1_precision_sparse(grad, H, x, start + g * n_times,
                                         n_times, tau_at(k), rho_at(k));
        };
        block.add_prior_pattern = [start, n_groups, n_times](
            std::vector<std::pair<int,int>>& out) {
            for (int g = 0; g < n_groups; g++)
                add_ar1_pattern(out, start + g * n_times, n_times);
        };
        block.log_prior = [start, n_groups, n_times, tau_at, rho_at](
            const Rcpp::NumericVector& x, int k) -> double {
            double lp = 0.0;
            for (int g = 0; g < n_groups; g++)
                lp += log_prior_ar1(x, start + g * n_times, n_times,
                                    tau_at(k), rho_at(k));
            return lp;
        };
    } else {
        Rcpp::stop("unknown temporal_type '%s' (expected 'rw1', 'rw2', or "
                   "'ar1')", type.c_str());
    }
    block.prior_kind = PriorFillKind::ADJACENCY;
}

} // namespace tulpa

#endif // TULPA_FIELD_BLOCK_PRIORS_H
