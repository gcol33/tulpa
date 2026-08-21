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

    // H is positive definite, so c' H^{-1} c is negative only when the solve
    // lost accuracy. Reporting sqrt(0) there presents a numerical failure as
    // certainty -- an interval built from it has zero width -- so the count is
    // raised alongside the factorization failure above. A query column that is
    // entirely zero gives exactly 0 and is a genuine zero standard error.
    int n_negative = 0;
    Eigen::VectorXd b(m);
    for (int j = 0; j < n; ++j) {
        b.setZero();
        for (SpMap::InnerIterator it(Cq, j); it; ++it) b[it.row()] = it.value();
        const Eigen::VectorXd x = chol.solve(b);
        double q = 0.0;               // c_j' H^{-1} c_j over the nonzeros of c_j
        for (SpMap::InnerIterator it(Cq, j); it; ++it) q += it.value() * x[it.row()];
        if (q < 0.0) { n_negative++; q = 0.0; }
        se[j] = std::sqrt(q);
    }
    if (n_negative > 0) {
        Rcpp::stop("cpp_spde_field_se: c' H^-1 c is negative for %d of %d "
                   "query cells; the joint precision solve lost accuracy.",
                   n_negative, n);
    }
    return se;
}
