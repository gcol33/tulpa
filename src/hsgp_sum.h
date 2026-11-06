#ifndef TULPA_HSGP_SUM_H
#define TULPA_HSGP_SUM_H

#include "hsgp_block_factory.h"
#include "hsgp_observation.h"
#include "nested_laplace_grid.h"

namespace tulpa {

// Independent Gaussian components enter the likelihood through their sum.
// QR of the concatenated loading integrates their joint null space exactly.
inline std::vector<int> hsgp_sum_indices(const Rcpp::List& specs) {
    std::vector<int> indices;
    int width = 0, rows = 0;
    for (int b = 0; b < specs.size(); ++b) {
        Rcpp::List p = specs[b];
        if (Rcpp::as<std::string>(p["type"]) != "hsgp" ||
            (p.containsElementNamed("observation_space") &&
             !Rcpp::as<bool>(p["observation_space"]))) continue;
        Rcpp::List phi = p.containsElementNamed("phi")
            ? Rcpp::as<Rcpp::List>(p["phi"])
            : Rcpp::List::create(p["phi_basis"]);
        Rcpp::List train = p.containsElementNamed("training_phi")
            ? Rcpp::as<Rcpp::List>(p["training_phi"]) : phi;
        int n = 0;
        for (int k = 0; k < train.size(); ++k)
            n += Rcpp::as<Rcpp::NumericMatrix>(train[k]).nrow();
        if (!indices.empty() && n != rows)
            Rcpp::stop("HSGP components must have the same pooled training rows.");
        rows = n;
        width += Rcpp::as<Rcpp::NumericMatrix>(phi[0]).ncol();
        indices.push_back(b);
    }
    if (indices.size() < 2 || width <= rows) indices.clear();
    return indices;
}

LatentBlock make_hsgp_sum_block(int start, const Rcpp::List& specs,
    const std::vector<int>& indices, const Rcpp::IntegerVector& axis_offsets,
    const Rcpp::NumericMatrix& theta_grid, const Rcpp::IntegerVector& n_obs);

} // namespace tulpa
#endif
