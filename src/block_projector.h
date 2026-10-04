// block_projector.h
// A per-arm linear projector from a latent block onto the observations:
// arm k's linear predictor receives A_k x_block, with A_k an N_k x size matrix
// stored column-compressed (A_x / A_i / A_p, 0-based). The SPDE block reaches
// its mesh through one (barycentric, ~3 nonzeros per row) and an areal block
// carrying a `projector` reaches its units through one (restricted spatial
// regression: A_k = P_k S_k, dense). Both are INDEXED_MULTI blocks: each row's
// nonzeros are materialized once at factory time and handed to every walker
// through `obs_indices`.

#ifndef TULPA_BLOCK_PROJECTOR_H
#define TULPA_BLOCK_PROJECTOR_H

#include "latent_block.h"
#include "spde_qbuilder.h"   // ARows, build_A_rows, spde_validate_projector
#include <Rcpp.h>
#include <cstdio>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace tulpa {

// Per-arm row lists of the projectors, validated against each arm's
// observation count and the block's column count.
inline std::shared_ptr<std::vector<ARows>> build_projector_rows_per_arm(
    const Rcpp::List&          A_x_per_arm,
    const Rcpp::List&          A_i_per_arm,
    const Rcpp::List&          A_p_per_arm,
    const Rcpp::IntegerVector& n_obs_per_arm,
    int                        n_arms,
    int                        n_cols,
    int                        block_index,
    const char*                type
) {
    if (static_cast<int>(A_x_per_arm.size()) != n_arms ||
        static_cast<int>(A_i_per_arm.size()) != n_arms ||
        static_cast<int>(A_p_per_arm.size()) != n_arms ||
        n_obs_per_arm.size() != n_arms) {
        Rcpp::stop("Block %d (type '%s'): A_x/A_i/A_p/n_obs_per_arm must "
                   "each have length n_arms (%d).",
                   block_index + 1, type, n_arms);
    }
    auto rows = std::make_shared<std::vector<ARows>>(n_arms);
    for (int k = 0; k < n_arms; k++) {
        Rcpp::NumericVector A_x = A_x_per_arm[k];
        Rcpp::IntegerVector A_i = A_i_per_arm[k];
        Rcpp::IntegerVector A_p = A_p_per_arm[k];
        char arm_label[96];
        std::snprintf(arm_label, sizeof(arm_label),
                      "Block %d (type '%s'): A[[%d]]", block_index + 1, type,
                      k + 1);
        spde_validate_projector(n_cols, n_obs_per_arm[k], A_x, A_i, A_p,
                                arm_label);
        (*rows)[k] = build_A_rows(n_obs_per_arm[k], n_cols, A_x, A_i, A_p);
    }
    return rows;
}

// The INDEXED_MULTI `obs_indices` of a projected block: row i of A_{k_arm} as
// (block-local 1-based column, weight) pairs.
inline std::function<void(int, int, std::vector<std::pair<int, double>>&)>
make_projector_obs_indices(std::shared_ptr<std::vector<ARows>> rows_per_arm) {
    return [rows_per_arm](int i, int k_arm,
                          std::vector<std::pair<int, double>>& out) {
        const ARows& rows = (*rows_per_arm)[k_arm];
        if (i < 0 || i >= static_cast<int>(rows.size())) return;
        const auto& row = rows[i];
        out.reserve(row.size());
        for (const auto& ae : row) out.emplace_back(ae.mesh_idx + 1, ae.weight);
    };
}

// Row sums A_k 1 of each arm's projector: what a constant added to the block
// moves arm k's predictor by, row by row.
inline std::vector<std::vector<double>> projector_row_sums(
    const std::vector<ARows>& rows_per_arm
) {
    std::vector<std::vector<double>> out(rows_per_arm.size());
    for (std::size_t k = 0; k < rows_per_arm.size(); k++) {
        const ARows& rows = rows_per_arm[k];
        out[k].assign(rows.size(), 0.0);
        for (std::size_t i = 0; i < rows.size(); i++)
            for (const auto& ae : rows[i]) out[k][i] += ae.weight;
    }
    return out;
}

} // namespace tulpa

#endif // TULPA_BLOCK_PROJECTOR_H
