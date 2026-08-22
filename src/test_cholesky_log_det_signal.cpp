// test_cholesky_log_det_signal.cpp
// Test-only entry point: the failure signal on the factorize-for-log-determinant
// dispatch.
//
// dense_cholesky_log_det_raw deliberately does not clamp its pivots -- an
// indefinite H leaves a NaN and an exactly singular one a -Inf -- so whether a
// failed factorization is reported at all is a property of the return value,
// and the Newton paths that carry it into a log-marginal are the callers that
// have to read it. This drives both the dispatch and the dense core on a
// caller-supplied symmetric matrix.
//
// `prefer_sparse` selects the same arm the Newton loops select, so the two
// backends can be shown to agree on which matrices they refuse.

#include <Rcpp.h>
#include <limits>

#include "laplace_cholesky.h"
#include "laplace_cholesky_dispatch.h"
#include "sparse_cholesky.h"

// [[Rcpp::export]]
Rcpp::List cpp_test_log_det_signal(Rcpp::NumericMatrix H_in,
                                    bool prefer_sparse = false,
                                    bool add_ridge = false) {
    const int n = static_cast<int>(H_in.nrow());
    if (static_cast<int>(H_in.ncol()) != n) {
        Rcpp::stop("H must be square; got %d x %d.", n,
                   static_cast<int>(H_in.ncol()));
    }

    tulpa::DenseVec fill(n, 0.0);
    tulpa::DenseMat H(static_cast<std::size_t>(n), fill);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) H[i][j] = H_in(i, j);
    }

    tulpa::SparseCholeskySolver solver;
    tulpa::DenseCholeskyScratch scratch;

    double log_det_dispatch = std::numeric_limits<double>::quiet_NaN();
    tulpa::DenseMat Hd = H;
    const bool ok_dispatch =
        add_ridge
            ? tulpa::dispatch_factor_log_det(Hd, n, solver, prefer_sparse,
                                             scratch, log_det_dispatch)
            : tulpa::dispatch_factor_log_det_ridged(Hd, n, solver, prefer_sparse,
                                                    scratch, log_det_dispatch);

    // The dense core on its own, with no sparse arm in front of it.
    double log_det_dense = std::numeric_limits<double>::quiet_NaN();
    tulpa::DenseMat He = H;
    const bool ok_dense =
        tulpa::dense_cholesky_log_det_raw(He, n, scratch, log_det_dense);

    return Rcpp::List::create(
        Rcpp::_["ok"]            = ok_dispatch,
        Rcpp::_["log_det"]       = log_det_dispatch,
        Rcpp::_["ok_dense"]      = ok_dense,
        Rcpp::_["log_det_dense"] = log_det_dense
    );
}
