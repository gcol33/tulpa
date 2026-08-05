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
#include "tulpa/spde_model_data.h"
#include <Rcpp.h>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

namespace tulpa {

// Matern (range, sigma) -> SPDE operator (kappa, tau) for the d = 2
// construction. Single source of truth for the conversion every caller
// applies before touching SpdeQBuilder; the R counterpart is .spde_kappa_tau()
// (R/marginal_se_spatial.R) and the two must agree formula for formula.
//
// The marginal variance is sigma^2 = Gamma(nu) / (Gamma(nu+1) (4 pi)
// kappa^(2 nu) tau^2) = 1 / (4 pi nu kappa^(2 nu) tau^2) (Lindgren, Rue &
// Lindstrom 2011, d = 2), so nu enters BOTH kappa and tau. At nu = 1 the tau
// factor collapses to sqrt(4 pi) kappa.
inline std::pair<double, double> spde_range_sigma_to_kappa_tau(
    double range, double sigma, double nu
) {
    const double kappa = std::sqrt(8.0 * nu) / range;
    const double tau   = 1.0 / (std::sqrt(4.0 * M_PI * nu) *
                                std::pow(kappa, nu) * sigma);
    return {kappa, tau};
}

// =====================================================================
// SpdeQBuilder: pattern-preserving sparse Q construction
// =====================================================================
//
// The FEM precision of operator order alpha is Q = tau² K (C⁻¹K)^(alpha-1)
// with K = kappa²C + G (Lindgren, Rue & Lindstrom 2011). Expanding the
// product over the operator chain
//
//   M_0 = C,   M_1 = G,   M_j = G (C⁻¹G)^(j-1)
//
// gives a binomial sum whose only theta dependence is a scalar per level:
//
//   Q(kappa, tau) = tau² * sum_{j=0}^{alpha} C(alpha, j) kappa^(2(alpha-j)) M_j
//
// so one per-entry array per level (m_contrib[j]) covers EVERY integer alpha:
// alpha = 1 is kappa²C + G, alpha = 2 is kappa⁴C + 2kappa²G + GC⁻¹G, alpha = 3
// adds 3kappa²GC⁻¹G + GC⁻¹GC⁻¹G, and so on. The chain is built once by init()
// for the requested alpha; the sparsity pattern (Q_p, Q_i) is its union and is
// fixed, so only Q_x changes with theta.

struct SpdeQBuilder {
    int n_mesh;
    std::vector<int> Q_p, Q_i;
    // Operator order the chain was built for. rebuild() expands over levels
    // 0..alpha_order, so m_contrib.size() == alpha_order + 1.
    int alpha_order = 2;
    // m_contrib[j][e] = entry e of M_j. Levels 0/1/2 are C, G and GC⁻¹G; the
    // named accessors below expose them for the callers that need those three
    // specifically (the rational assembly and the analytic dQ/dtheta kernel).
    std::vector<std::vector<double>> m_contrib;
    // Orphan ridge: per-entry contribution that is theta-independent. Equals
    // 1.0 on the diagonal of orphan mesh nodes (nodes with C0_diag ~ 0 AND no
    // off-diagonal G1 connectivity — i.e. vertices the mesh refiner inserted
    // but never wired into a triangle). Adds a unit precision to those nodes
    // so Q remains PD. Zero on every other entry.
    std::vector<double> orphan_contrib;
    std::vector<double> Q_x;

    const std::vector<double>& c0_contrib()  const { return m_contrib[0]; }
    const std::vector<double>& g1_contrib()  const { return m_contrib[1]; }
    const std::vector<double>& gdg_contrib() const { return m_contrib[2]; }

    // `alpha` is the operator order the chain is built for. The rational
    // assembly shifts the alpha = 2 stencil, so rational callers pass 2.
    void init(int n, const Rcpp::NumericVector& C0_diag,
              const Rcpp::NumericVector& G1_x, const Rcpp::IntegerVector& G1_i,
              const Rcpp::IntegerVector& G1_p, int alpha = 2) {
        if (alpha < 1) {
            Rcpp::stop("SPDE FEM assembly requires alpha >= 1 (nu >= 0); got %d.",
                       alpha);
        }
        n_mesh = n;
        alpha_order = alpha;

        // Detect orphan mesh nodes: any vertex with zero (or near-zero) FEM
        // mass. Two flavors land here:
        //   1. Truly disconnected vertices (zero mass AND no G1 connectivity)
        //      from upstream mesh refiners that emit Steiner points but fail
        //      to retriangulate (see tulpaMesh fix-mesh-zero-triangles).
        //   2. Zero-mass vertices that DO carry G1 connectivity (e.g. a
        //      Steiner point on a constraint edge with degenerate incident-
        //      triangle area). Every operator level above M_1 carries a C⁻¹,
        //      which is undefined where C_diag = 0: the GC⁻¹G contribution for
        //      that row silently zeros out, leaving
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

        // Operator chain M_0 = C, M_1 = G, M_{l+1} = M_l C⁻¹ G, held
        // column-major as row -> value maps. Every M_l is symmetric
        // (M_l = G (C⁻¹G)^(l-1)), so column k of M_l also reads its row k,
        // which is what the product below indexes.
        const int n_levels = alpha + 1;
        std::vector<std::vector<std::map<int, double>>> M(
            n_levels, std::vector<std::map<int, double>>(n));

        for (int i = 0; i < n; i++) M[0][i][i] += C0_diag[i];

        for (int j = 0; j < n; j++) {
            for (int idx = G1_p[j]; idx < G1_p[j + 1]; idx++) {
                M[1][j][G1_i[idx]] += G1_x[idx];
            }
        }

        // (M_l C⁻¹ G)[r, c] = sum_k M_l[r, k] * c0_inv[k] * G[k, c]. The k loop
        // walks column c of G (G is symmetric, so G[k, c] is its entry at row
        // k), and r walks column k of M_l. A node with zero FEM mass has
        // c0_inv = 0, so paths through an orphan drop out of every level.
        for (int l = 1; l < alpha; l++) {
            for (int c = 0; c < n; c++) {
                for (int idx = G1_p[c]; idx < G1_p[c + 1]; idx++) {
                    int k = G1_i[idx];
                    double w = G1_x[idx] * c0_inv[k];
                    if (std::abs(w) <= c0_eps) continue;
                    auto& out_col = M[l + 1][c];
                    for (const auto& [r, v] : M[l][k]) out_col[r] += v * w;
                }
            }
        }

        // Orphan ridge: place 1.0 on the diagonal of every orphan node.
        // This is a theta-independent term — see rebuild() for how it's mixed
        // into Q_x. Pinning to ~zero is fine because A never references
        // orphan vertices, so the orphan latent has no likelihood contribution.
        std::vector<double> orph(n, 0.0);
        for (int i = 0; i < n; i++) {
            if (is_orphan[i]) orph[i] += 1.0;
        }

        // Convert to CSC over the union of the chain's patterns, dropping
        // entries that are numerically absent at every level.
        Q_p.assign(n + 1, 0);
        Q_i.clear();
        m_contrib.assign(n_levels, std::vector<double>());
        orphan_contrib.clear();

        std::vector<double> vals(n_levels);
        for (int col = 0; col < n; col++) {
            Q_p[col] = static_cast<int>(Q_i.size());
            std::set<int> rows;
            for (int l = 0; l < n_levels; l++) {
                for (const auto& [r, v] : M[l][col]) rows.insert(r);
            }
            if (orph[col] != 0.0) rows.insert(col);
            for (int row : rows) {
                double mag = 0.0;
                for (int l = 0; l < n_levels; l++) {
                    auto it = M[l][col].find(row);
                    vals[l] = (it == M[l][col].end()) ? 0.0 : it->second;
                    mag += std::abs(vals[l]);
                }
                double o = (row == col) ? orph[col] : 0.0;
                if (mag + std::abs(o) <= c0_eps) continue;
                Q_i.push_back(row);
                for (int l = 0; l < n_levels; l++) m_contrib[l].push_back(vals[l]);
                orphan_contrib.push_back(o);
            }
        }
        Q_p[n] = static_cast<int>(Q_i.size());
        Q_x.assign(Q_i.size(), 0.0);
    }

    // Rebuild Q values for given (kappa, tau) at the operator order this
    // builder was init()'d for. alpha = nu + d/2 with d = 2 (spatial
    // dimension), nu = Matérn smoothness:
    //   alpha=1 (nu=0): Q = tau² * (κ²C + G)
    //   alpha=2 (nu=1): Q = tau² * (κ⁴C + 2κ²G + GC⁻¹G)  — standard
    //   alpha=3 (nu=2): Q = tau² * (κ⁶C + 3κ⁴G + 3κ²GC⁻¹G + GC⁻¹GC⁻¹G)
    //   ... and so on, the binomial expansion of K(C⁻¹K)^(alpha-1).
    // Fractional alpha goes through rebuild_rational instead.
    void rebuild(double kappa, double tau_spde) {
        const double k2 = kappa * kappa;
        const double tau2 = tau_spde * tau_spde;
        const int nnz_val = static_cast<int>(Q_i.size());
        const int n_levels = alpha_order + 1;

        // coef[j] = C(alpha, j) * kappa^(2 * (alpha - j)). Powers accumulate by
        // repeated multiplication so alpha = 2 reproduces k4 = k2 * k2 exactly.
        std::vector<double> kpow(n_levels);
        kpow[0] = 1.0;
        for (int m = 1; m < n_levels; m++) kpow[m] = kpow[m - 1] * k2;

        std::vector<double> coef(n_levels);
        double binom = 1.0;
        for (int j = 0; j < n_levels; j++) {
            if (j > 0) binom = binom * (alpha_order - j + 1) / j;
            coef[j] = binom * kpow[alpha_order - j];
        }

        for (int e = 0; e < nnz_val; e++) {
            double acc = 0.0;
            for (int j = 0; j < n_levels; j++) acc += coef[j] * m_contrib[j][e];
            Q_x[e] = tau2 * acc + orphan_contrib[e];
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
        if (alpha_order < 2) {
            Rcpp::stop("The rational assembly shifts the alpha = 2 stencil, so "
                       "the builder must be init()'d with alpha >= 2; got %d.",
                       alpha_order);
        }
        const std::vector<double>& c0  = c0_contrib();
        const std::vector<double>& g1  = g1_contrib();
        const std::vector<double>& gdg = gdg_contrib();
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
                    k4_shifted * c0[e] +
                    2.0 * k2_shifted * g1[e] +
                    gdg[e]
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
// Per-row entry and container are exported in tulpa/spde_model_data.h so
// the same struct is reused by ModelData::spde_data. Aliasing here keeps
// internal call sites (build_A_rows, run_spde_laplace, etc.) untouched.

using ARowEntry = SpdeARowEntry;
using ARows     = std::vector<std::vector<ARowEntry>>;

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

// `re_idx` / `n_re_groups` / `sigma_re` carry an optional iid random-effect
// block laid out between the fixed effects and the mesh field, matching the
// nested SPDE layout [beta (p), re (n_re_groups), w_mesh (n_mesh)]. The caller
// passes mesh_start = p + n_re_groups and n_x = p + n_re_groups + n_mesh. The
// defaults reproduce the spatial-only case, so callers that carry no RE block
// (implicit_diff) need no change.
template<typename F>
void run_spde_laplace(
    const Rcpp::NumericVector& y, const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X, int N, int p, int n_mesh, int mesh_start, int n_x,
    const ARows& A_rows, const SpdeQBuilder& qb,
    const std::string& family, double phi,
    int max_iter, double tol, int n_threads,
    const Rcpp::NumericVector& x_init,
    SparseCholeskySolver* shared_solver,
    const double* offset,
    F callback,
    const Rcpp::NumericVector& re_idx = Rcpp::NumericVector(),
    int n_re_groups = 0, double sigma_re = 1.0,
    bool center_mesh = true, double prior_lognorm = 0.0,
    // Inner-Laplace skewness diagnostic (gcol33/tulpa#273 item 3): forwarded
    // unchanged to laplace_newton_solve, which already supports it (the
    // dense/auto-sparse family-enum path every other single-block kernel
    // shares). skew_probe_idx == nullptr with compute_skew = true probes
    // every latent index.
    bool compute_skew = false,
    const std::vector<int>* skew_probe_idx = nullptr
) {
    const double tau_re = (n_re_groups > 0)
                          ? 1.0 / (sigma_re * sigma_re + 1e-10) : 0.0;
    auto re_group = [&](int i) -> int {
        if (n_re_groups <= 0) return -1;
        int g = (int)re_idx[i] - 1;
        return (g >= 0 && g < n_re_groups) ? g : -1;
    };

    auto compute_eta = [&](const Rcpp::NumericVector& x, Rcpp::NumericVector& eta) {
        for (int i = 0; i < N; i++) {
            eta[i] = offset ? offset[i] : 0.0;
            for (int j = 0; j < p; j++) eta[i] += X(i, j) * x[j];
            int g = re_group(i);
            if (g >= 0) eta[i] += x[p + g];
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
            int g = re_group(i);
            if (g >= 0) {
                int re_i = p + g;
                grad[re_i] += gh.grad;
                H[re_i][re_i] += gh.neg_hess;
                for (int j = 0; j < p; j++) {
                    double c = gh.neg_hess * X(i, j);
                    H[re_i][j] += c;
                    H[j][re_i] += c;
                }
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
                if (g >= 0) {
                    int re_i = p + g;
                    double c = gh.neg_hess * a1;
                    H[re_i][idx1] += c;
                    H[idx1][re_i] += c;
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
        for (int g = 0; g < n_re_groups; g++) {
            grad[p + g] -= tau_re * x[p + g];
            H[p + g][p + g] += tau_re;
        }
    };

    // Integer-alpha field is sum-to-zero centred for intercept identifiability;
    // the rational path's latent is the auxiliary weights x (field u = Pr x),
    // which must NOT be centred -- the proper SPDE prior (kappa^2 > 0) already
    // identifies the constant mode.
    auto center = [&](Rcpp::NumericVector& x) {
        if (center_mesh) center_effects(x, mesh_start, n_mesh);
    };

    auto log_prior = [&](const Rcpp::NumericVector& x, const Rcpp::NumericVector&) {
        double qf = 0.0;
        for (int j = 0; j < n_mesh; j++) {
            for (int qidx = qb.Q_p[j]; qidx < qb.Q_p[j + 1]; qidx++) {
                qf += x[mesh_start + qb.Q_i[qidx]] * qb.Q_x[qidx] * x[mesh_start + j];
            }
        }
        for (int g = 0; g < n_re_groups; g++) qf += tau_re * x[p + g] * x[p + g];
        // prior_lognorm = 0.5 log|Q(theta)|: the SPDE prior normalizer, supplied
        // by the caller (constant in x). Required whenever this fit's
        // log_marginal is compared across (range, sigma); see spde_logdet.h.
        return prior_lognorm - 0.5 * qf;
    };

    LaplaceResult result = laplace_newton_solve(
        y, n_trials, family, phi, N, n_x,
        max_iter, tol, n_threads,
        compute_eta, scatter, center, log_prior,
        x_init, shared_solver, /*store_Q=*/false, /*inv_block_layout=*/nullptr,
        compute_skew, skew_probe_idx
    );

    callback(result);
}

} // namespace tulpa

#endif // TULPA_SPDE_QBUILDER_H
