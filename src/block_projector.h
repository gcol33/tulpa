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
#include <cmath>
#include <cstdio>
#include <string>
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

// The per-arm projector of an areal block spec (`projector`, the column-
// compressed lists A_x / A_i / A_p, one entry per arm), or null when the block
// reads its units through `spatial_idx`. A projector replaces the gather, so any
// per-row weight belongs inside A_k: a spec carrying `svc_weight` as well is
// refused rather than weighted twice.
inline std::shared_ptr<std::vector<ARows>> read_block_projector(
    const Rcpp::List&          bs,
    const Rcpp::IntegerVector& n_obs_per_arm,
    int                        n_cols,
    int                        block_index,
    const std::string&         type
) {
    if (!bs.containsElementNamed("projector") || Rf_isNull(bs["projector"]))
        return nullptr;
    if (bs.containsElementNamed("svc_weight") && !Rf_isNull(bs["svc_weight"])) {
        Rcpp::stop("Block %d (type '%s'): `projector` and `svc_weight` cannot "
                   "be combined; scale the projector's rows by the weight.",
                   block_index + 1, type.c_str());
    }
    Rcpp::List pr = bs["projector"];
    return build_projector_rows_per_arm(
        pr["A_x"], pr["A_i"], pr["A_p"], n_obs_per_arm,
        static_cast<int>(n_obs_per_arm.size()), n_cols, block_index,
        type.c_str());
}

// Read a block through its projector instead of an index gather.
inline void apply_block_projector(
    LatentBlock& block, const std::shared_ptr<std::vector<ARows>>& rows
) {
    block.idx          = std::function<int(int, int)>();
    block.obs_indices  = make_projector_obs_indices(rows);
    block.contrib_kind = BlockContribKind::INDEXED_MULTI;
}

// Where a projected intrinsic field's level belongs. A constant c added to the
// field moves arm k's predictor by c * A_k 1, so it is read off the row sums:
//   * ABSENT    -- A_k 1 = 0 on every arm (a projector orthogonal to an
//                  intercept, as the restricted-spatial-regression one is): the
//                  level never reaches the predictor and is removed with no
//                  fold;
//   * INTERCEPT -- A_k 1 equals the intercept column on every arm the field
//                  reaches: the level folds into the intercept, as for a
//                  gathered field;
//   * NONE      -- no column carries it, and the precision augmentation
//                  identifies the level in the field.
// `intercept_at(k, i, x0)` writes arm k's intercept-column entry at row i and
// returns false when the arm has no such column.
enum class ProjectedLevel { ABSENT, INTERCEPT, NONE };

inline ProjectedLevel projected_level(
    const std::vector<ARows>& rows_per_arm,
    const std::function<bool(int, int, double&)>& intercept_at
) {
    const auto sums = projector_row_sums(rows_per_arm);
    bool absent    = true;
    bool intercept = true;
    for (std::size_t k = 0; k < sums.size(); k++) {
        for (std::size_t i = 0; i < sums[k].size(); i++) {
            double scale = 1.0;
            for (const auto& ae : rows_per_arm[k][i]) scale += std::abs(ae.weight);
            const double s = sums[k][i];
            if (std::abs(s) > kAliasColumnTol * scale) absent = false;
            double x0 = 0.0;
            if (s != 0.0 &&
                (!intercept_at(static_cast<int>(k), static_cast<int>(i), x0) ||
                 std::abs(x0 - s) > kAliasColumnTol * (1.0 + std::abs(s))))
                intercept = false;
        }
    }
    if (absent) return ProjectedLevel::ABSENT;
    if (intercept) return ProjectedLevel::INTERCEPT;
    return ProjectedLevel::NONE;
}

} // namespace tulpa

#endif // TULPA_BLOCK_PROJECTOR_H
