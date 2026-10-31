// block_projector.cpp
// Definition of build_projector_rows_per_arm, declared in block_projector.h.

#include <RcppEigen.h>
#include "block_projector.h"

namespace tulpa {

std::shared_ptr<std::vector<ARows>> build_projector_rows_per_arm(
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

} // namespace tulpa
