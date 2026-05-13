// spde_qbuilder.h
// SPDE precision matrix builder and sparse A helpers
// Precomputes sparsity pattern from FEM matrices, rebuilds Q values per theta.
// Used by spde_laplace.cpp for single-fit and nested Laplace SPDE.

#ifndef TULPA_SPDE_QBUILDER_H
#define TULPA_SPDE_QBUILDER_H

#include "laplace_core.h"
#include "laplace_family_link.h"
#include "laplace_newton.h"
#include "laplace_re_priors.h"
#include <Rcpp.h>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

namespace tulpa {

// =====================================================================
// SpdeQBuilder: pattern-preserving sparse Q construction
// =====================================================================
//
// Stores three per-entry contribution arrays (C0, G1, GDG) that combine as:
//   Q(kappa, tau) = tau² * (kappa⁴ * c0[e] + 2*kappa² * g1[e] + gdg[e])
// The sparsity pattern (Q_p, Q_i) is fixed. Only Q_x changes with theta.

struct SpdeQBuilder {
    int n_mesh;
    std::vector<int> Q_p, Q_i;
    std::vector<double> c0_contrib;
    std::vector<double> g1_contrib;
    std::vector<double> gdg_contrib;
    // Orphan ridge: per-entry contribution that is theta-independent. Equals
    // 1.0 on the diagonal of orphan mesh nodes (nodes with C0_diag ~ 0 AND no
    // off-diagonal G1 connectivity — i.e. vertices the mesh refiner inserted
    // but never wired into a triangle). Adds a unit precision to those nodes
    // so Q remains PD. Zero on every other entry.
    std::vector<double> orphan_contrib;
    std::vector<double> Q_x;

    void init(int n, const Rcpp::NumericVector& C0_diag,
              const Rcpp::NumericVector& G1_x, const Rcpp::IntegerVector& G1_i,
              const Rcpp::IntegerVector& G1_p) {
        n_mesh = n;

        // Detect orphan mesh nodes: any vertex with zero (or near-zero) FEM
        // mass. Two flavors land here:
        //   1. Truly disconnected vertices (zero mass AND no G1 connectivity)
        //      from upstream mesh refiners that emit Steiner points but fail
        //      to retriangulate (see tulpaMesh fix-mesh-zero-triangles).
        //   2. Zero-mass vertices that DO carry G1 connectivity (e.g. a
        //      Steiner point on a constraint edge with degenerate incident-
        //      triangle area). The SPDE precision Q = tau² (κ⁴C + 2κ²G + G C⁻¹G)
        //      uses C⁻¹ in the GDG term, which is undefined where C_diag = 0:
        //      the GDG contribution for that row silently zeros out, leaving
        //      Q rank-deficient at the node. CHOLMOD then warns "not positive
        //      definite" and the Newton solver makes no progress.
        // The defensive fix in both flavors: place a unit precision ridge on
        // the orphan diagonal so Q stays PD. A is built only from triangles
        // with valid area, so the orphan latent has no likelihood weight and
        // is effectively pinned at zero.
        const double c0_eps = 1e-15;
        std::vector<bool> is_orphan(n, false);
        for (int i = 0; i < n; i++) {
            if (C0_diag[i] <= c0_eps) is_orphan[i] = true;
        }

        std::vector<double> c0_inv(n);
        for (int i = 0; i < n; i++) {
            c0_inv[i] = (C0_diag[i] > c0_eps) ? 1.0 / C0_diag[i] : 0.0;
        }

        struct Triple { double c0 = 0, g1 = 0, gdg = 0, orph = 0; };
        std::map<std::pair<int,int>, Triple> entries;

        // C0 diagonal
        for (int i = 0; i < n; i++) entries[{i, i}].c0 += C0_diag[i];

        // G1 sparse
        for (int j = 0; j < n; j++) {
            for (int idx = G1_p[j]; idx < G1_p[j + 1]; idx++) {
                entries[{G1_i[idx], j}].g1 += G1_x[idx];
            }
        }

        // GDG = G1 * diag(c0_inv) * G1
        for (int j = 0; j < n; j++) {
            std::vector<std::pair<int, double>> scaled;
            for (int idx = G1_p[j]; idx < G1_p[j + 1]; idx++) {
                int k = G1_i[idx];
                double val = G1_x[idx] * c0_inv[k];
                if (std::abs(val) > c0_eps) scaled.push_back({k, val});
            }
            for (auto& [k, sc] : scaled) {
                for (int idx2 = G1_p[k]; idx2 < G1_p[k + 1]; idx2++) {
                    entries[{G1_i[idx2], j}].gdg += G1_x[idx2] * sc;
                }
            }
        }

        // Orphan ridge: place 1.0 on the diagonal of every orphan node.
        // This is a theta-independent term — see rebuild() for how it's mixed
        // into Q_x. Pinning to ~zero is fine because A never references
        // orphan vertices, so the orphan latent has no likelihood contribution.
        for (int i = 0; i < n; i++) {
            if (is_orphan[i]) entries[{i, i}].orph += 1.0;
        }

        // Convert to CSC
        struct Entry { int row, col; double c0, g1, gdg, orph; };
        std::vector<Entry> sorted;
        sorted.reserve(entries.size());
        for (auto& [key, t] : entries) {
            if (std::abs(t.c0) + std::abs(t.g1) + std::abs(t.gdg) + std::abs(t.orph) > c0_eps) {
                sorted.push_back({key.first, key.second, t.c0, t.g1, t.gdg, t.orph});
            }
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const Entry& a, const Entry& b) {
                      return a.col < b.col || (a.col == b.col && a.row < b.row);
                  });

        int nnz = static_cast<int>(sorted.size());
        Q_p.assign(n + 1, 0);
        Q_i.resize(nnz);
        c0_contrib.resize(nnz);
        g1_contrib.resize(nnz);
        gdg_contrib.resize(nnz);
        orphan_contrib.resize(nnz);
        Q_x.resize(nnz);

        int cur_col = 0;
        for (int e = 0; e < nnz; e++) {
            while (cur_col <= sorted[e].col) { Q_p[cur_col] = e; cur_col++; }
            Q_i[e] = sorted[e].row;
            c0_contrib[e] = sorted[e].c0;
            g1_contrib[e] = sorted[e].g1;
            gdg_contrib[e] = sorted[e].gdg;
            orphan_contrib[e] = sorted[e].orph;
        }
        while (cur_col <= n) { Q_p[cur_col] = nnz; cur_col++; }
    }

    // Rebuild Q values for given (kappa, tau) and operator order alpha.
    // alpha = nu + d/2 where d=2 (spatial dimension), nu = Matérn smoothness.
    //   alpha=1 (nu=0): Q = tau² * (κ²C + G)  — very rough
    //   alpha=2 (nu=1): Q = tau² * (κ⁴C + 2κ²G + GC⁻¹G)  — standard
    //   Fractional alpha: rational approximation via weighted sum of shifted terms
    void rebuild(double kappa, double tau_spde, int alpha = 2) {
        double k2 = kappa * kappa;
        double tau2 = tau_spde * tau_spde;
        int nnz_val = static_cast<int>(Q_i.size());

        if (alpha == 1) {
            // Q = tau² * (κ²C + G + eps*C) — ridge for positive definiteness
            // alpha=1 operator is rank-deficient; needs more regularization
            double eps_ridge = 1e-2;
            for (int e = 0; e < nnz_val; e++) {
                Q_x[e] = tau2 * ((k2 + eps_ridge) * c0_contrib[e] + g1_contrib[e])
                       + orphan_contrib[e];
            }
        } else {
            // alpha == 2 (default): Q = tau² * L·C⁻¹·L
            double k4 = k2 * k2;
            for (int e = 0; e < nnz_val; e++) {
                Q_x[e] = tau2 * (k4 * c0_contrib[e] + 2.0 * k2 * g1_contrib[e] + gdg_contrib[e])
                       + orphan_contrib[e];
            }
        }
    }

    // Rebuild for fractional alpha using rational approximation.
    // poles[] and weights[] are the m pairs (r_k, w_k) from the best
    // rational approximation of x^(-beta), beta = alpha/2.
    // Q ≈ tau² * Σ_k w_k * ((κ²+r_k)²C + 2(κ²+r_k)G + GC⁻¹G)
    // Each term has the same sparsity pattern as alpha=2 but with shifted κ².
    void rebuild_rational(double kappa, double tau_spde,
                          const std::vector<double>& poles,
                          const std::vector<double>& weights) {
        double k2 = kappa * kappa;
        double tau2 = tau_spde * tau_spde;
        int nnz_val = static_cast<int>(Q_i.size());
        int m = static_cast<int>(poles.size());

        // Zero out
        for (int e = 0; e < nnz_val; e++) Q_x[e] = 0.0;

        // Accumulate weighted shifted terms
        for (int j = 0; j < m; j++) {
            double k2_shifted = k2 + poles[j];
            double k4_shifted = k2_shifted * k2_shifted;
            for (int e = 0; e < nnz_val; e++) {
                Q_x[e] += tau2 * weights[j] * (
                    k4_shifted * c0_contrib[e] +
                    2.0 * k2_shifted * g1_contrib[e] +
                    gdg_contrib[e]
                );
            }
        }
        // Theta-independent orphan ridge.
        for (int e = 0; e < nnz_val; e++) Q_x[e] += orphan_contrib[e];
    }

    int nnz() const { return static_cast<int>(Q_i.size()); }
};

// =====================================================================
// Sparse per-row A storage for fast eta/scatter
// =====================================================================

struct ARowEntry { int mesh_idx; double weight; };
using ARows = std::vector<std::vector<ARowEntry>>;

inline ARows build_A_rows(int N, int n_mesh,
                           const Rcpp::NumericVector& A_x,
                           const Rcpp::IntegerVector& A_i,
                           const Rcpp::IntegerVector& A_p) {
    ARows rows(N);
    for (int j = 0; j < n_mesh; j++) {
        for (int idx = A_p[j]; idx < A_p[j + 1]; idx++) {
            int i = A_i[idx];
            if (i < N && std::abs(A_x[idx]) > 1e-15) {
                rows[i].push_back({j, A_x[idx]});
            }
        }
    }
    return rows;
}

// =====================================================================
// Shared SPDE Laplace runner (used by single-fit and nested)
// =====================================================================

template<typename F>
void run_spde_laplace(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, int N, int p, int n_mesh, int mesh_start, int n_x,
    const ARows& A_rows, const SpdeQBuilder& qb,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    const Rcpp::NumericVector& x_init,
    SparseCholeskySolver* shared_solver,
    F callback
) {
    auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
        for (int i = 0; i < N; i++) {
            eta[i] = 0.0;
            for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
            for (const auto& ae : A_rows[i]) {
                eta[i] += ae.weight * x[mesh_start + ae.mesh_idx];
            }
        }
    };

    auto scatter = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector& eta,
                       DenseVec& grad, DenseMat& H) {
        for (int i = 0; i < N; i++) {
            auto gh = grad_hess_for_family(y[i], n_trials[i], eta[i], family, phi);
            for (int j = 0; j < p; j++) {
                grad[j] += gh.grad * X(i, j);
                for (int k = 0; k < p; k++) H[j][k] += gh.neg_hess * X(i, j) * X(i, k);
            }
            const auto& row = A_rows[i];
            for (size_t s1 = 0; s1 < row.size(); s1++) {
                int idx1 = mesh_start + row[s1].mesh_idx;
                double a1 = row[s1].weight;
                grad[idx1] += gh.grad * a1;
                H[idx1][idx1] += gh.neg_hess * a1 * a1;
                for (int j = 0; j < p; j++) {
                    H[j][idx1] += gh.neg_hess * X(i, j) * a1;
                    H[idx1][j] += gh.neg_hess * X(i, j) * a1;
                }
                for (size_t s2 = s1 + 1; s2 < row.size(); s2++) {
                    int idx2 = mesh_start + row[s2].mesh_idx;
                    double cross = gh.neg_hess * a1 * row[s2].weight;
                    H[idx1][idx2] += cross;
                    H[idx2][idx1] += cross;
                }
            }
        }
        for (int j = 0; j < n_mesh; j++) {
            for (int qidx = qb.Q_p[j]; qidx < qb.Q_p[j + 1]; qidx++) {
                int qi = qb.Q_i[qidx];
                double q = qb.Q_x[qidx];
                grad[mesh_start + qi] -= q * x[mesh_start + j];
                H[mesh_start + qi][mesh_start + j] += q;
            }
        }
        double tau_beta = 1e-4;
        for (int j = 0; j < p; j++) { grad[j] -= tau_beta * x[j]; H[j][j] += tau_beta; }
    };

    auto center = [&](Rcpp::NumericVector& x) {
        center_effects(x, mesh_start, n_mesh);
    };

    auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
        double qf = 0.0;
        for (int j = 0; j < n_mesh; j++) {
            for (int qidx = qb.Q_p[j]; qidx < qb.Q_p[j + 1]; qidx++) {
                qf += x[mesh_start + qb.Q_i[qidx]] * qb.Q_x[qidx] * x[mesh_start + j];
            }
        }
        return -0.5 * qf;
    };

    LaplaceResult result = laplace_newton_solve(
        y, n_trials, family, phi, N, n_x,
        max_iter, tol, n_threads,
        compute_eta, scatter, center, log_prior,
        x_init, shared_solver
    );

    callback(result);
}

} // namespace tulpa

#endif // TULPA_SPDE_QBUILDER_H
