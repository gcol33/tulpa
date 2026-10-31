// spde_qbuilder.cpp
// Definitions of the SPDE precision builder declared in spde_qbuilder.h.

#include <RcppEigen.h>
#include "spde_qbuilder.h"

namespace tulpa {

void SpdeQBuilder::init(int n, const Rcpp::NumericVector& C0_diag,
                         const Rcpp::NumericVector& G1_x, const Rcpp::IntegerVector& G1_i,
                         const Rcpp::IntegerVector& G1_p, int alpha) {
    if (alpha < 1) {
        Rcpp::stop("SPDE FEM assembly requires alpha >= 1 (nu >= 0); got %d.",
                   alpha);
    }
    n_mesh = n;
    alpha_order = alpha;

    // Orphan mesh nodes (zero or near-zero FEM mass) and the unit ridge
    // that keeps Q positive definite at them: spde_zero_mass.h carries the
    // floor, the ridge and why both are needed.
    const double c0_eps = SPDE_C0_EPS;
    std::vector<bool> is_orphan(n, false);
    for (int i = 0; i < n; i++) {
        is_orphan[i] = spde_is_orphan_mass(C0_diag[i]);
    }

    std::vector<double> c0_inv(n);
    for (int i = 0; i < n; i++) {
        c0_inv[i] = spde_c0_inv(C0_diag[i]);
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

    // Orphan ridge, theta-independent — see rebuild() for how it is mixed
    // into Q_x.
    std::vector<double> orph(n, 0.0);
    for (int i = 0; i < n; i++) {
        if (is_orphan[i]) orph[i] += SPDE_ORPHAN_RIDGE;
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

void spde_validate_csc(int n_col, const Rcpp::IntegerVector& p_vec,
                              const Rcpp::IntegerVector& i_vec, int nnz,
                              int n_row, const char* what) {
    if ((int) p_vec.size() != n_col + 1) {
        Rcpp::stop("%s: column pointer vector must have length %d; got %d.",
                   what, n_col + 1, (int) p_vec.size());
    }
    if ((int) i_vec.size() != nnz) {
        Rcpp::stop("%s: row-index vector must have length %d; got %d.",
                   what, nnz, (int) i_vec.size());
    }
    if (p_vec[0] != 0) {
        Rcpp::stop("%s: column pointers must start at 0; got %d.",
                   what, (int)p_vec[0]);
    }
    for (int c = 0; c < n_col; c++) {
        if (p_vec[c + 1] < p_vec[c]) {
            Rcpp::stop("%s: column pointers must be non-decreasing; entry %d "
                       "(%d) is below entry %d (%d).",
                       what, c + 2, (int)p_vec[c + 1], c + 1, (int)p_vec[c]);
        }
    }
    if (p_vec[n_col] != nnz) {
        Rcpp::stop("%s: last column pointer (%d) must equal the number of "
                   "stored values (%d).", what, (int)p_vec[n_col], nnz);
    }
    for (int e = 0; e < nnz; e++) {
        if (i_vec[e] < 0 || i_vec[e] >= n_row) {
            Rcpp::stop("%s: row index %d at slot %d is outside [0, %d).",
                       what, (int)i_vec[e], e + 1, n_row);
        }
    }
}

} // namespace tulpa
