// intrinsic_log_pdet.h
// The generalized log-determinant log|Q|_+ of an intrinsic GMRF's structure
// matrix: the sum of the logs of its non-zero eigenvalues. It is the part of an
// intrinsic field's density normalizer that carries no hyperparameter
// (0.5 rank log tau carries the precision), so a posterior never reads it, but
// a log marginal likelihood does: without it the evidence of a model with an
// intrinsic field is off by a graph- or length-dependent constant and is not
// comparable with any other model's.
//
// One builder for every intrinsic structure. For a PSD Q whose null space is
// spanned by the columns of N (n x r),
//
//   |Q|_+ = det(Q[-S, -S]) * det(N'N) / det(N[S, ])^2
//
// for any r rows S at which N[S, ] is invertible: deleting S leaves an SPD
// principal submatrix, factored by one sparse Cholesky. With N = 1 on a
// connected graph Laplacian this is the matrix-tree theorem, J * det(any
// (J-1)-minor). S is chosen by column-pivoted QR of N' so N[S, ] is the
// best-conditioned r x r block.

#ifndef TULPA_INTRINSIC_LOG_PDET_H
#define TULPA_INTRINSIC_LOG_PDET_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Rcpp.h>
#include <cmath>
#include <vector>

namespace tulpa {

inline double intrinsic_log_pdet(const Eigen::SparseMatrix<double>& Q,
                                 const Eigen::MatrixXd& N) {
    const int n = static_cast<int>(Q.rows());
    const int r = static_cast<int>(N.cols());
    if (n == 0 || r >= n) return 0.0;
    std::vector<int> drop;
    double log_ns = 0.0, log_ntn = 0.0;
    if (r > 0) {
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(N.transpose());
        Eigen::MatrixXd NS(r, r);
        for (int a = 0; a < r; a++) {
            drop.push_back(qr.colsPermutation().indices()[a]);
            NS.row(a) = N.row(drop.back());
        }
        log_ns = std::log(std::abs(NS.partialPivLu().determinant()));
        Eigen::LLT<Eigen::MatrixXd> ntn(N.transpose() * N);
        const Eigen::MatrixXd Lnt = ntn.matrixL();
        for (int a = 0; a < r; a++) log_ntn += 2.0 * std::log(Lnt(a, a));
    }
    std::vector<int> pos(n, 0);
    for (int s : drop) pos[s] = -1;
    int m = 0;
    for (int i = 0; i < n; i++) if (pos[i] == 0) pos[i] = m++; else pos[i] = -1;
    std::vector<Eigen::Triplet<double>> trip;
    for (int c = 0; c < Q.outerSize(); c++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(Q, c); it; ++it) {
            const int i = pos[it.row()], j = pos[it.col()];
            if (i >= 0 && j >= 0) trip.emplace_back(i, j, it.value());
        }
    }
    Eigen::SparseMatrix<double> M(m, m);
    M.setFromTriplets(trip.begin(), trip.end());
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt(M);
    if (llt.info() != Eigen::Success) {
        Rcpp::stop("intrinsic_log_pdet: the structure matrix with its null "
                   "space removed is not positive definite.");
    }
    const Eigen::SparseMatrix<double> L = llt.matrixL();
    double ld = 0.0;
    for (int j = 0; j < m; j++) ld += std::log(L.coeff(j, j));
    return 2.0 * ld + log_ntn - 2.0 * log_ns;
}

// log|Q|_+ of one random walk of order 1 or 2 over n_times nodes, as the
// structure the RW priors scatter (laplace_temporal_priors.cpp): the row-wise
// differences of order `order`, closed into a ring when `cyclic`. Its null
// space is the constants, plus the linear trend for an acyclic RW2.
inline double rw_log_pdet(int n_times, int order, bool cyclic) {
    if (n_times < order + 1) return 0.0;
    std::vector<std::vector<std::pair<int, double>>> rows;
    if (order == 1) {
        for (int t = 1; t < n_times; t++) rows.push_back({{t - 1, -1.0}, {t, 1.0}});
        if (cyclic) rows.push_back({{n_times - 1, -1.0}, {0, 1.0}});
    } else {
        for (int t = 1; t < n_times - 1; t++)
            rows.push_back({{t - 1, 1.0}, {t, -2.0}, {t + 1, 1.0}});
        if (cyclic) {
            rows.push_back({{n_times - 2, 1.0}, {n_times - 1, -2.0}, {0, 1.0}});
            rows.push_back({{n_times - 1, 1.0}, {0, -2.0}, {1, 1.0}});
        }
    }
    std::vector<Eigen::Triplet<double>> trip;
    for (const auto& row : rows)
        for (const auto& a : row)
            for (const auto& b : row)
                trip.emplace_back(a.first, b.first, a.second * b.second);
    Eigen::SparseMatrix<double> Q(n_times, n_times);
    Q.setFromTriplets(trip.begin(), trip.end());
    const int r = (order == 2 && !cyclic) ? 2 : 1;
    Eigen::MatrixXd N(n_times, r);
    for (int t = 0; t < n_times; t++) {
        N(t, 0) = 1.0;
        if (r == 2) N(t, 1) = t - 0.5 * (n_times - 1);
    }
    return intrinsic_log_pdet(Q, N);
}

} // namespace tulpa

#endif // TULPA_INTRINSIC_LOG_PDET_H
