// small_dense_sym.h
// Small dense symmetric factorizations (row-major) for the K x K capacitance
// matrices of the low-rank folds: SmallChol for an SPD matrix, SmallSymEig for
// a symmetric one whose inertia matters.

#ifndef TULPA_SMALL_DENSE_SYM_H
#define TULPA_SMALL_DENSE_SYM_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tulpa {

// Small dense SPD Cholesky (row-major), for the K x K capacitance and the
// n_s x n_s Schur complement in the block-Schur log-determinant / solve.
struct SmallChol {
    int m = 0;
    std::vector<double> L;   // lower-triangle row-major
    bool ok = false;
    void factor(const std::vector<double>& M, int m_) {
        m = m_; L.assign((std::size_t) m * m, 0.0); ok = true;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j <= i; ++j) {
                double s = M[(std::size_t) i * m + j];
                for (int p = 0; p < j; ++p) s -= L[(std::size_t) i*m+p] * L[(std::size_t) j*m+p];
                if (i == j) {
                    if (s <= 0.0) { ok = false; return; }
                    L[(std::size_t) i*m+i] = std::sqrt(s);
                } else {
                    L[(std::size_t) i*m+j] = s / L[(std::size_t) j*m+j];
                }
            }
        }
    }
    void solve(const double* b, double* x) const {     // L L' x = b
        std::vector<double> y(m);
        for (int i = 0; i < m; ++i) {
            double s = b[i];
            for (int p = 0; p < i; ++p) s -= L[(std::size_t) i*m+p] * y[p];
            y[i] = s / L[(std::size_t) i*m+i];
        }
        for (int i = m - 1; i >= 0; --i) {
            double s = y[i];
            for (int p = i + 1; p < m; ++p) s -= L[(std::size_t) p*m+i] * x[p];
            x[i] = s / L[(std::size_t) i*m+i];
        }
    }
    double logdet() const {
        double d = 0.0;
        for (int i = 0; i < m; ++i) d += 2.0 * std::log(L[(std::size_t) i*m+i]);
        return d;
    }
};

// Small dense symmetric eigendecomposition (cyclic Jacobi, row-major), for the
// K x K capacitance when the field block is indefinite and the capacitance may
// be too: its inertia decides whether the pinned field block is PD, and the
// solve stands in for SmallChol's. M = V diag(w) V'.
struct SmallSymEig {
    int m = 0;
    std::vector<double> V, w;
    bool ok = false;
    void factor(const std::vector<double>& M, int m_) {
        m = m_; ok = false;
        std::vector<double> a = M;
        V.assign((std::size_t) m * m, 0.0);
        for (int i = 0; i < m; ++i) V[(std::size_t) i*m+i] = 1.0;
        double scale = 0.0;
        for (double v : a) scale = std::max(scale, std::fabs(v));
        if (!(scale > 0.0) || !std::isfinite(scale)) return;
        for (int sweep = 0; sweep < 100; ++sweep) {
            double off = 0.0;
            for (int p = 0; p < m; ++p)
                for (int q = p + 1; q < m; ++q) off = std::max(off, std::fabs(a[(std::size_t) p*m+q]));
            if (off <= 1e-15 * scale) {
                w.assign(m, 0.0);
                for (int i = 0; i < m; ++i) {
                    w[i] = a[(std::size_t) i*m+i];
                    if (w[i] == 0.0 || !std::isfinite(w[i])) return;
                }
                ok = true;
                return;
            }
            for (int p = 0; p < m; ++p)
                for (int q = p + 1; q < m; ++q) {
                    const double apq = a[(std::size_t) p*m+q];
                    if (apq == 0.0) continue;
                    const double theta = (a[(std::size_t) q*m+q] - a[(std::size_t) p*m+p]) / (2.0 * apq);
                    const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                     (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                    const double c = 1.0 / std::sqrt(t * t + 1.0), sn = t * c;
                    for (int k = 0; k < m; ++k) {           // A <- A J (columns p, q)
                        const double akp = a[(std::size_t) k*m+p], akq = a[(std::size_t) k*m+q];
                        a[(std::size_t) k*m+p] = c * akp - sn * akq;
                        a[(std::size_t) k*m+q] = sn * akp + c * akq;
                    }
                    for (int k = 0; k < m; ++k) {           // A <- J' A (rows p, q)
                        const double apk = a[(std::size_t) p*m+k], aqk = a[(std::size_t) q*m+k];
                        a[(std::size_t) p*m+k] = c * apk - sn * aqk;
                        a[(std::size_t) q*m+k] = sn * apk + c * aqk;
                    }
                    for (int k = 0; k < m; ++k) {           // V <- V J
                        const double vkp = V[(std::size_t) k*m+p], vkq = V[(std::size_t) k*m+q];
                        V[(std::size_t) k*m+p] = c * vkp - sn * vkq;
                        V[(std::size_t) k*m+q] = sn * vkp + c * vkq;
                    }
                }
        }
    }
    int n_neg() const {
        int c = 0;
        for (double v : w) if (v < 0.0) ++c;
        return c;
    }
    double logabsdet() const {
        double d = 0.0;
        for (double v : w) d += std::log(std::fabs(v));
        return d;
    }
    void solve(const double* b, double* x) const {     // V diag(w)^-1 V' x = b
        std::vector<double> y(m, 0.0);
        for (int j = 0; j < m; ++j) {
            double s = 0.0;
            for (int i = 0; i < m; ++i) s += V[(std::size_t) i*m+j] * b[i];
            y[j] = s / w[j];
        }
        for (int i = 0; i < m; ++i) {
            double s = 0.0;
            for (int j = 0; j < m; ++j) s += V[(std::size_t) i*m+j] * y[j];
            x[i] = s;
        }
    }
};

}  // namespace tulpa

#endif  // TULPA_SMALL_DENSE_SYM_H
