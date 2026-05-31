// stochastic_logdet.h
// Stochastic Lanczos Quadrature (SLQ) for log-determinant estimation.
// For N > 100K where exact Cholesky is infeasible.
//
// Algorithm: Ubaru, Chen & Saad (2017)
//   log|H| ≈ n * mean_i [ z_i' f(H) z_i ]
// where z_i are random probe vectors and f(H) z_i is computed via
// Lanczos iteration (only needs H*v products, not factorization).
//
// Cost: O(n_probes * n_lanczos * nnz(H)) — linear in matrix size.

#ifndef TULPA_STOCHASTIC_LOGDET_H
#define TULPA_STOCHASTIC_LOGDET_H

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <random>

namespace tulpa {

// Sparse matrix-vector product: y = A*x where A is in CSC (lower triangle, symmetric)
// A is symmetric with stype=-1 (lower stored), so we compute both lower and upper contributions
inline void sparse_sym_matvec(
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_idx,
    const std::vector<double>& values,
    int n,
    const std::vector<double>& x,
    std::vector<double>& y
) {
    std::fill(y.begin(), y.end(), 0.0);
    for (int col = 0; col < n; col++) {
        for (int idx = col_ptr[col]; idx < col_ptr[col + 1]; idx++) {
            int row = row_idx[idx];
            double val = values[idx];
            y[row] += val * x[col];  // lower triangle
            if (row != col) {
                y[col] += val * x[row];  // symmetric upper
            }
        }
    }
}

// Lanczos iteration: compute tridiagonal decomposition T such that
// H ≈ Q T Q' where Q is orthonormal. Returns alpha (diagonal) and
// beta (off-diagonal) of T.
inline void lanczos(
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_idx,
    const std::vector<double>& values,
    int n,
    const std::vector<double>& z,  // starting vector (unit norm)
    int m,                          // number of Lanczos steps
    std::vector<double>& alpha,     // diagonal of T (length m)
    std::vector<double>& beta       // off-diagonal of T (length m-1)
) {
    alpha.resize(m, 0.0);
    beta.resize(m > 0 ? m - 1 : 0, 0.0);

    std::vector<double> q_prev(n, 0.0);
    std::vector<double> q_curr = z;
    std::vector<double> w(n);

    for (int j = 0; j < m; j++) {
        sparse_sym_matvec(col_ptr, row_idx, values, n, q_curr, w);

        alpha[j] = 0.0;
        for (int i = 0; i < n; i++) alpha[j] += q_curr[i] * w[i];

        for (int i = 0; i < n; i++) {
            w[i] -= alpha[j] * q_curr[i];
            if (j > 0) w[i] -= beta[j - 1] * q_prev[i];
        }

        double beta_j = 0.0;
        for (int i = 0; i < n; i++) beta_j += w[i] * w[i];
        beta_j = std::sqrt(beta_j);

        if (j < m - 1) {
            beta[j] = beta_j;
            if (beta_j < 1e-12) break;  // invariant subspace found

            q_prev = q_curr;
            for (int i = 0; i < n; i++) q_curr[i] = w[i] / beta_j;
        }
    }
}

// Eigendecomposition of a symmetric tridiagonal matrix T (m x m) with diagonal
// alpha and off-diagonal beta, via the implicit-shift QL algorithm (EISPACK
// tql2). Returns the eigenvalues theta_j in eval and, for each eigenvalue, the
// first component s_j[0] of its (orthonormal) eigenvector in first_comp.
//
// The full eigenvector matrix is never formed: tracking only its first row is
// all the Gauss-quadrature weight tau_j = (e_1 . s_j)^2 needs.
inline bool tridiag_eig_first_component(
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    int m,
    std::vector<double>& eval,
    std::vector<double>& first_comp
) {
    eval.assign(alpha.begin(), alpha.begin() + m);
    std::vector<double> e(m, 0.0);
    for (int i = 0; i < m - 1; i++) e[i] = beta[i];
    e[m - 1] = 0.0;

    // first_comp accumulates the first row of the accumulated rotation matrix,
    // initialized to the identity's first row (e_1).
    first_comp.assign(m, 0.0);
    first_comp[0] = 1.0;

    const int max_iter = 50;
    for (int l = 0; l < m; l++) {
        int iter = 0;
        int mm;
        do {
            // Find a small sub-diagonal element to split the matrix.
            for (mm = l; mm < m - 1; mm++) {
                double dd = std::fabs(eval[mm]) + std::fabs(eval[mm + 1]);
                if (std::fabs(e[mm]) <= 1e-300 + 1e-15 * dd) break;
            }
            if (mm != l) {
                if (iter++ == max_iter) return false;
                double g = (eval[l + 1] - eval[l]) / (2.0 * e[l]);
                double r = std::hypot(g, 1.0);
                double sign_g = (g >= 0.0) ? std::fabs(r) : -std::fabs(r);
                g = eval[mm] - eval[l] + e[l] / (g + sign_g);
                double s = 1.0, c = 1.0, p = 0.0;
                int i;
                for (i = mm - 1; i >= l; i--) {
                    double f = s * e[i];
                    double b = c * e[i];
                    r = std::hypot(f, g);
                    e[i + 1] = r;
                    if (r == 0.0) {
                        eval[i + 1] -= p;
                        e[mm] = 0.0;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = eval[i + 1] - p;
                    r = (eval[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    eval[i + 1] = g + p;
                    g = c * r - b;
                    // Accumulate the rotation into the first row only.
                    double fc = first_comp[i + 1];
                    first_comp[i + 1] = s * first_comp[i] + c * fc;
                    first_comp[i] = c * first_comp[i] - s * fc;
                }
                if (r == 0.0 && i >= l) continue;
                eval[l] -= p;
                e[l] = g;
                e[mm] = 0.0;
            }
        } while (mm != l);
    }
    return true;
}

// Stochastic Lanczos Quadrature for log|H|.
// H is a sparse symmetric positive definite matrix in CSC format (lower triangle).
// Returns estimate of log|H| using n_probes random vectors and n_lanczos steps.
inline double stochastic_log_determinant(
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_idx,
    const std::vector<double>& values,
    int n,
    int n_probes = 30,
    int n_lanczos = 50,
    unsigned int seed = 42
) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    double log_det_sum = 0.0;

    for (int probe = 0; probe < n_probes; probe++) {
        // Generate random probe vector z ~ N(0, I), normalize
        std::vector<double> z(n);
        double norm = 0.0;
        for (int i = 0; i < n; i++) {
            z[i] = normal(rng);
            norm += z[i] * z[i];
        }
        norm = std::sqrt(norm);
        for (int i = 0; i < n; i++) z[i] /= norm;

        // Lanczos iteration
        std::vector<double> alpha, beta;
        int m = std::min(n_lanczos, n);
        lanczos(col_ptr, row_idx, values, n, z, m, alpha, beta);

        // SLQ Gauss-quadrature estimate of z' log(H) z (Ubaru, Chen & Saad 2017):
        // eigendecompose T_m to get Ritz values theta_j and the squared first
        // eigenvector components tau_j = (e_1 . s_j)^2 (the quadrature weights,
        // sum_j tau_j = 1), then z' log(H) z ~ sum_j tau_j * log(theta_j) since z
        // is unit norm. The per-probe contribution to log|H| is n times this.
        std::vector<double> theta, first_comp;
        if (!tridiag_eig_first_component(alpha, beta, m, theta, first_comp)) {
            Rcpp::stop("stochastic_log_determinant: tridiagonal eigensolve failed to converge");
        }

        double quad = 0.0;
        for (int j = 0; j < m; j++) {
            double tau = first_comp[j] * first_comp[j];
            double lam = theta[j];
            if (lam <= 0.0) {
                Rcpp::stop("stochastic_log_determinant: non-positive Ritz value; matrix is not positive definite");
            }
            quad += tau * std::log(lam);
        }

        log_det_sum += static_cast<double>(n) * quad;
    }

    return log_det_sum / n_probes;
}

} // namespace tulpa

#endif // TULPA_STOCHASTIC_LOGDET_H
