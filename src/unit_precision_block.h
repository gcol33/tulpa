// unit_precision_block.h
// Prior callbacks for a standard-normal latent block.
//
// A block whose prior is N(0, I) over its own range contributes grad -= x
// there, a unit diagonal to the Hessian, and -0.5 sum(x^2) - 0.5 size log(2 pi)
// to the log-density. BYM2's unstructured component and the plain IID block are
// both exactly that, in the single-arm and the joint multi-arm kernels alike, so
// the three callbacks are filled from here rather than written out per block.
//
// The prior is diagonal, so no add_prior_pattern is set: the sparse builder adds
// the block diagonal unconditionally.

#ifndef TULPA_UNIT_PRECISION_BLOCK_H
#define TULPA_UNIT_PRECISION_BLOCK_H

#include "latent_block.h"
#include "sparse_hessian.h"
#include <Rcpp.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tulpa {

// Fill `block`'s add_prior / add_prior_sparse / log_prior for a unit-precision
// Gaussian over [start, start + size). Leaves every other field alone, so the
// caller still sets idx / d_fac / contrib_kind and pushes the block.
inline void set_unit_precision_block_priors(LatentBlock& block,
                                            int start, int size) {
    block.add_prior = [start, size](DenseVec& grad, DenseMat& H,
                                    const Rcpp::NumericVector& x, int /*k*/) {
        for (int s = 0; s < size; s++) {
            const int idx = start + s;
            grad[idx] -= x[idx];
            H[idx][idx] += 1.0;
        }
    };
    block.add_prior_sparse = [start, size](SparseHessianBuilder& H,
                                           DenseVec& grad,
                                           const Rcpp::NumericVector& x,
                                           int /*k*/) {
        for (int s = 0; s < size; s++) {
            const int idx = start + s;
            grad[idx] -= x[idx];
            H.add(idx, idx, 1.0);
        }
    };
    block.log_prior = [start, size](const Rcpp::NumericVector& x,
                                    int /*k*/) -> double {
        double lp = 0.0;
        for (int s = 0; s < size; s++) {
            const double v = x[start + s];
            lp -= 0.5 * v * v;
        }
        lp -= 0.5 * size * std::log(2.0 * M_PI);
        return lp;
    };
}

} // namespace tulpa

#endif // TULPA_UNIT_PRECISION_BLOCK_H
