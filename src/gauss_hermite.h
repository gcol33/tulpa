// gauss_hermite.h
// Probabilist's Gauss-Hermite quadrature via Golub-Welsch. C++ port of
// gauss_hermite_prob() in R/agq.R: nodes z_k and weights w_k (sum w_k = 1) such
// that sum_k w_k f(z_k) approximates E_{Z~N(0,1)}[f(Z)]. Self-contained so the
// AGHQ engine carries no R round-trip in the inner loop.

#ifndef TULPA_GAUSS_HERMITE_H
#define TULPA_GAUSS_HERMITE_H

#include <RcppEigen.h>
#include <cmath>

namespace tulpa {

struct GaussHermite {
    Eigen::VectorXd nodes;     // length n, ascending
    Eigen::VectorXd weights;   // length n, sum = 1
};

// Probabilist's Hermite three-term recurrence z H_n = H_{n+1} + n H_{n-1} ->
// symmetric tridiagonal Jacobi with zero diagonal and off-diagonal sqrt(k).
// Eigenvalues are the nodes; the squared first component of each eigenvector is
// the weight (probabilist's measure normalises to 1). n = 1 -> {0}, {1}.
inline GaussHermite gauss_hermite_prob(int n) {
    GaussHermite gh;
    if (n <= 1) {
        gh.nodes   = Eigen::VectorXd::Zero(1);
        gh.weights = Eigen::VectorXd::Ones(1);
        return gh;
    }
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);
    for (int k = 1; k <= n - 1; ++k) {
        const double b = std::sqrt((double)k);
        J(k - 1, k) = b;
        J(k, k - 1) = b;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J);   // ascending eigenvalues
    gh.nodes   = es.eigenvalues();
    gh.weights = es.eigenvectors().row(0).array().square();
    return gh;
}

} // namespace tulpa

#endif // TULPA_GAUSS_HERMITE_H
