// spde_field_se.cpp
// Per-cell linear-predictor SE for an SPDE fit with the field included:
//   se_j = sqrt(c_j' H^{-1} c_j),  c_j = column j of Cq = [x*_j; a*_j],
// with H the joint (beta, field) posterior precision at the fitted
// hyperparameters. The joint precision is factorized once and each query
// column is solved on its own, so the dense working set stays O(p + n_mesh)
// regardless of the number of query cells -- a large prediction grid never
// materializes the (p + n_mesh) x n_cells dense H^{-1} Cq.

#include <RcppEigen.h>
#include <cmath>

// [[Rcpp::export]]
Rcpp::NumericVector cpp_spde_field_se(
    const Eigen::Map<Eigen::SparseMatrix<double>> H,    // (m x m) joint precision
    const Eigen::Map<Eigen::SparseMatrix<double>> Cq)   // (m x n_cells) query cols
{
    typedef Eigen::Map<Eigen::SparseMatrix<double> > SpMap;
    const int m = static_cast<int>(H.rows());
    const int n = static_cast<int>(Cq.cols());
    Rcpp::NumericVector se(n);
    if (n == 0) return se;
    if (static_cast<int>(Cq.rows()) != m) {
        Rcpp::stop("cpp_spde_field_se: H and Cq row dimensions disagree.");
    }

    // LL^T of the lower triangle (H is symmetric); fails loudly on a
    // non-positive-definite joint precision, matching the R Cholesky path.
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > chol;
    chol.compute(H);
    if (chol.info() != Eigen::Success) {
        Rcpp::stop("cpp_spde_field_se: joint precision is not positive definite.");
    }

    Eigen::VectorXd b(m);
    for (int j = 0; j < n; ++j) {
        b.setZero();
        for (SpMap::InnerIterator it(Cq, j); it; ++it) b[it.row()] = it.value();
        const Eigen::VectorXd x = chol.solve(b);
        double q = 0.0;               // c_j' H^{-1} c_j over the nonzeros of c_j
        for (SpMap::InnerIterator it(Cq, j); it; ++it) q += it.value() * x[it.row()];
        se[j] = q > 0.0 ? std::sqrt(q) : 0.0;
    }
    return se;
}
