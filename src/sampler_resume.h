#ifndef TULPA_SAMPLER_RESUME_H
#define TULPA_SAMPLER_RESUME_H

#include "hmc_sampler.h"

namespace tulpa {

inline std::vector<double> read_epsilon_init(
    Rcpp::Nullable<Rcpp::NumericVector> input, int n_chains) {
  if (input.isNull()) return {};
  Rcpp::NumericVector values(input);
  if (values.size() != 1 && values.size() != n_chains)
    Rcpp::stop("epsilon_init must have length 1 or n_chains");
  std::vector<double> out(n_chains);
  for (int c = 0; c < n_chains; ++c) {
    out[c] = values[values.size() == 1 ? 0 : c];
    tulpa_hmc::validate_epsilon_init(out[c]);
  }
  return out;
}

// A vector supplies one shared diagonal; a matrix carries one row per chain.
template <typename T>
inline std::vector<std::vector<double>> read_inv_metric_init(
    Rcpp::Nullable<T> input, int n_chains, int n_params) {
  if (input.isNull()) return {};
  Rcpp::NumericVector values(input.get());
  const bool matrix = values.hasAttribute("dim");
  if (matrix) {
    Rcpp::IntegerVector dims = values.attr("dim");
    if (dims.size() != 2 || dims[0] != n_chains || dims[1] != n_params)
      Rcpp::stop("inv_metric_init must have n_chains rows and n_params columns");
  } else if (values.size() != n_params) {
    Rcpp::stop("inv_metric_init length must equal n_params");
  }
  std::vector<std::vector<double>> out(n_chains, std::vector<double>(n_params));
  for (int c = 0; c < n_chains; ++c) for (int j = 0; j < n_params; ++j) {
    const double value = values[matrix ? c + n_chains * j : j];
    out[c][j] = value;
  }
  for (const auto& row : out) tulpa_hmc::validate_inv_metric_init(row, n_params);
  return out;
}

} // namespace tulpa

#endif
