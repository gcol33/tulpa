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
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>

namespace tulpa {

// Check a CSC lower triangle against `n` before anything indexes off it. Every
// loop below uses col_ptr[col + 1] as a bound and row_idx[idx] as an unchecked
// WRITE offset into a length-n vector, so a mis-sized or out-of-range triple is
// heap corruption rather than a wrong number, and neither entry point (the Rcpp
// export or the C-ABI shim) inspects the arrays it forwards.
inline void validate_csc_lower(
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_idx,
    const std::vector<double>& values,
    int n
) {
    if (n < 0) {
        Rcpp::stop("stochastic_log_determinant: n (%d) must be non-negative.", n);
    }
    if (static_cast<int>(col_ptr.size()) != n + 1) {
        Rcpp::stop("stochastic_log_determinant: length(Q_p) (%d) must be n + 1 "
                   "(%d).", static_cast<int>(col_ptr.size()), n + 1);
    }
    if (row_idx.size() != values.size()) {
        Rcpp::stop("stochastic_log_determinant: length(Q_i) (%d) must equal "
                   "length(Q_x) (%d).", static_cast<int>(row_idx.size()),
                   static_cast<int>(values.size()));
    }
    if (col_ptr[0] != 0) {
        Rcpp::stop("stochastic_log_determinant: Q_p[0] (%d) must be 0.",
                   col_ptr[0]);
    }
    if (col_ptr[n] != static_cast<int>(values.size())) {
        Rcpp::stop("stochastic_log_determinant: Q_p[n] (%d) must equal "
                   "length(Q_x) (%d).", col_ptr[n],
                   static_cast<int>(values.size()));
    }
    for (int col = 0; col < n; col++) {
        if (col_ptr[col + 1] < col_ptr[col]) {
            Rcpp::stop("stochastic_log_determinant: Q_p is not non-decreasing "
                       "at column %d (%d then %d).", col + 1, col_ptr[col],
                       col_ptr[col + 1]);
        }
    }
    for (std::size_t k = 0; k < row_idx.size(); k++) {
        if (row_idx[k] < 0 || row_idx[k] >= n) {
            Rcpp::stop("stochastic_log_determinant: Q_i[%d] (%d) is outside "
                       "[0, %d).", static_cast<int>(k), row_idx[k], n);
        }
    }
}

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

// Lanczos iteration: compute the tridiagonal decomposition T with
// H ~ Q T Q_transpose, Q orthonormal. Fills alpha (diagonal) and beta
// (off-diagonal) of T and RETURNS the number of steps actually taken.
//
// Two things the plain three-term recurrence does not give:
//
// Full reorthogonalisation. Ubaru, Chen & Saad (2017), the reference this file
// implements, run Lanczos with it. Without it the computed q_j lose
// orthogonality within a handful of steps, Ritz values duplicate, and the Gauss
// weights stop summing to 1 -- a BIAS in log|H| rather than an error, on the
// N > 100K path where there is no Cholesky to cross-check against. It is also a
// second route into a spurious non-positive Ritz value on a genuinely PD
// matrix. The stored basis costs n * m doubles, the cost the reference assumes.
//
// The step count. A breakdown (beta_j below tolerance) means the Krylov space is
// exhausted and the quadrature on the leading block is EXACT. Padding alpha and
// beta with zeros out to m instead hands the eigensolver a trailing zero block
// whose Ritz values are all 0, which a caller cannot tell apart from a genuinely
// non-positive one.
inline int lanczos(
    const std::vector<int>& col_ptr,
    const std::vector<int>& row_idx,
    const std::vector<double>& values,
    int n,
    const std::vector<double>& z,  // starting vector (unit norm)
    int m,                          // maximum number of Lanczos steps
    std::vector<double>& alpha,     // diagonal of T (length m_eff on return)
    std::vector<double>& beta       // off-diagonal of T (length m_eff - 1)
) {
    alpha.assign(m > 0 ? m : 0, 0.0);
    beta.assign(m > 1 ? m - 1 : 0, 0.0);
    if (m <= 0 || n <= 0) return 0;

    // Column j of the Lanczos basis lives at &Q[(size_t)j * n].
    std::vector<double> Q(static_cast<std::size_t>(n) * m, 0.0);
    std::copy(z.begin(), z.end(), Q.begin());
    std::vector<double> q_curr(n), w(n);

    int m_eff = 0;
    for (int j = 0; j < m; j++) {
        const std::size_t base_j = static_cast<std::size_t>(j) * n;
        std::copy(Q.begin() + base_j, Q.begin() + base_j + n, q_curr.begin());

        sparse_sym_matvec(col_ptr, row_idx, values, n, q_curr, w);

        alpha[j] = 0.0;
        for (int i = 0; i < n; i++) alpha[j] += q_curr[i] * w[i];
        m_eff = j + 1;
        if (j == m - 1) break;

        for (int i = 0; i < n; i++) {
            w[i] -= alpha[j] * q_curr[i];
            if (j > 0) w[i] -= beta[j - 1] * Q[static_cast<std::size_t>(j - 1) * n + i];
        }

        // Reorthogonalise against every stored basis vector, twice: one
        // Gram-Schmidt pass leaves a residual of the same order as the loss it
        // corrects, and the second is the standard twice-is-enough sweep.
        for (int pass = 0; pass < 2; pass++) {
            for (int k = 0; k <= j; k++) {
                const std::size_t base_k = static_cast<std::size_t>(k) * n;
                double c = 0.0;
                for (int i = 0; i < n; i++) c += Q[base_k + i] * w[i];
                for (int i = 0; i < n; i++) w[i] -= c * Q[base_k + i];
            }
        }

        double beta_j = 0.0;
        for (int i = 0; i < n; i++) beta_j += w[i] * w[i];
        beta_j = std::sqrt(beta_j);

        beta[j] = beta_j;
        if (beta_j < 1e-12) break;  // invariant subspace found; T_{m_eff} is exact

        const std::size_t base_next = static_cast<std::size_t>(j + 1) * n;
        for (int i = 0; i < n; i++) Q[base_next + i] = w[i] / beta_j;
    }

    alpha.resize(m_eff);
    beta.resize(m_eff > 0 ? m_eff - 1 : 0);
    return m_eff;
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
    validate_csc_lower(col_ptr, row_idx, values, n);

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

        // Lanczos iteration. m_eff is below m when the recurrence broke down,
        // which is the case where T_{m_eff} spans the whole Krylov space and the
        // quadrature below is exact rather than approximate.
        std::vector<double> alpha, beta;
        int m = std::min(n_lanczos, n);
        int m_eff = lanczos(col_ptr, row_idx, values, n, z, m, alpha, beta);
        if (m_eff <= 0) continue;

        // SLQ Gauss-quadrature estimate of z' log(H) z (Ubaru, Chen & Saad 2017):
        // eigendecompose T_m to get Ritz values theta_j and the squared first
        // eigenvector components tau_j = (e_1 . s_j)^2 (the quadrature weights,
        // sum_j tau_j = 1), then z' log(H) z ~ sum_j tau_j * log(theta_j) since z
        // is unit norm. The per-probe contribution to log|H| is n times this.
        std::vector<double> theta, first_comp;
        if (!tridiag_eig_first_component(alpha, beta, m_eff, theta, first_comp)) {
            Rcpp::stop("stochastic_log_determinant: tridiagonal eigensolve failed to converge");
        }

        double quad = 0.0;
        for (int j = 0; j < m_eff; j++) {
            double tau = first_comp[j] * first_comp[j];
            double lam = theta[j];
            // A Ritz value carrying no quadrature weight contributes nothing to
            // the estimate, so it says nothing about definiteness either. Only a
            // WEIGHTED non-positive one does.
            if (tau == 0.0) continue;
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
