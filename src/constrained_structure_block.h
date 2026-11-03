// constrained_structure_block.h
// A latent block whose prior is an intrinsic GMRF of fixed structure,
//
//   x ~ N(0, (tau R)^-)  conditioned on  A x = 0,
//
// with R a sparse positive semi-definite structure whose null space is exactly
// the row space of A. The Knorr-Held space-time interactions are this shape:
// R = R_t (x) R_s (type IV), R_t (x) I (type II), I (x) R_s (type III) or I
// (type I), and A spans null(R) -- a sum over the areas at each time, a sum
// over the times at each area -- so the interaction carries nothing the main
// effects already do.
//
// The constraint is HARD: the rows go to LatentBlock::hard_constraints and the
// sparse joint Newton solves the bordered system on A x = 0
// (hard_constraints.h). The stored precision is tau R, which carries no
// curvature along null(R): a direction only the constraint identifies (a time
// no observation reaches, an area aliased with its main effect) is identified
// by the KKT system, through one anchor per row on the factored matrix's
// diagonal, not by a term in the prior. An A'A term would do it too, but its
// rows span whole time slices and whole series and fill the factor. The
// log-density is the constrained one,
//
//   log p(x | tau) = 0.5 (n - r) log(tau / (2 pi)) + 0.5 log|R|_+
//                    - 0.5 tau x' R x,
//
// r the number of constraint rows and |R|_+ the generalized determinant, both
// supplied by the caller, who builds R.
//
// The structure arrives as a FULL symmetric CSC (both triangles, 0-based row
// indices). It may be R or R plus any term zero on the constraint set (a
// multiple of A'A): the quadratic form read off it is x' R x on every feasible
// x either way.
//
// No dense scatter is set: the block reaches the sparse joint Newton only
// (blocks_have_sparse_only_prior), which is the one carrying the constraint.

#ifndef TULPA_CONSTRAINED_STRUCTURE_BLOCK_H
#define TULPA_CONSTRAINED_STRUCTURE_BLOCK_H

#include "latent_block.h"
#include "linear_constraint_rows.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {

template <class TauAt>
inline LatentBlock make_constrained_structure_block(
    int start, int size,
    Rcpp::IntegerVector obs_idx,
    const Rcpp::IntegerVector& S_p, const Rcpp::IntegerVector& S_i,
    const Rcpp::NumericVector& S_x,
    double log_pdet,
    const std::vector<LinearConstraintRow>& rows_local,
    TauAt tau_at
) {
    if (S_p.size() != size + 1 || S_i.size() != S_x.size() ||
        S_p[0] != 0 || S_p[size] != S_i.size()) {
        Rcpp::stop("constrained structure block: the structure CSC is not a "
                   "%d x %d matrix.", size, size);
    }
    auto p_v = std::make_shared<std::vector<int>>(S_p.begin(), S_p.end());
    auto i_v = std::make_shared<std::vector<int>>(S_i.begin(), S_i.end());
    auto x_v = std::make_shared<std::vector<double>>(S_x.begin(), S_x.end());
    for (int v : *i_v) {
        if (v < 0 || v >= size)
            Rcpp::stop("constrained structure block: a structure row index "
                       "lies outside [0, %d).", size);
    }

    LatentBlock block;
    block.start = start;
    block.size  = size;
    block.contrib_kind = BlockContribKind::INDEXED_SINGLE;
    block.prior_kind   = PriorFillKind::USER_CSC;
    block.idx   = [obs_idx](int i, int /*k_arm*/) { return obs_idx[i]; };
    block.d_fac = [](int) { return 1.0; };

    block.add_prior_sparse = [start, size, p_v, i_v, x_v, tau_at](
        SparseHessianBuilder& H, DenseVec& grad,
        const Rcpp::NumericVector& x, int k
    ) {
        const double tau = tau_at(k);
        for (int j = 0; j < size; j++) {
            const double xj = x[start + j];
            for (int e = (*p_v)[j]; e < (*p_v)[j + 1]; e++) {
                const int i = (*i_v)[e];
                const double q = tau * (*x_v)[e];
                grad[start + i] -= q * xj;
                if (i >= j) H.add(start + i, start + j, q);
            }
        }
    };

    block.add_prior_pattern = [start, size, p_v, i_v](
        std::vector<std::pair<int,int>>& out
    ) {
        for (int j = 0; j < size; j++) {
            for (int e = (*p_v)[j]; e < (*p_v)[j + 1]; e++) {
                const int i = (*i_v)[e];
                if (i > j) out.emplace_back(start + i, start + j);
            }
        }
    };

    const int rank = size - static_cast<int>(rows_local.size());
    block.log_prior = [start, size, p_v, i_v, x_v, tau_at, rank, log_pdet](
        const Rcpp::NumericVector& x, int k
    ) -> double {
        const double tau = tau_at(k);
        double quad = 0.0;
        for (int j = 0; j < size; j++) {
            const double xj = x[start + j];
            for (int e = (*p_v)[j]; e < (*p_v)[j + 1]; e++)
                quad += x[start + (*i_v)[e]] * (*x_v)[e] * xj;
        }
        return 0.5 * rank * std::log(tau / (2.0 * M_PI)) + 0.5 * log_pdet
             - 0.5 * tau * quad;
    };

    block.hard_constraints.reserve(rows_local.size());
    for (const auto& r : rows_local) {
        LinearConstraintRow g = r;
        for (int& c : g.cols) {
            if (c < 0 || c >= size)
                Rcpp::stop("constrained structure block: a constraint index "
                           "lies outside the block.");
            c += start;
        }
        block.hard_constraints.push_back(std::move(g));
    }
    return block;
}

}  // namespace tulpa

#endif  // TULPA_CONSTRAINED_STRUCTURE_BLOCK_H
