// hmc_chain_stack.cpp
// Definition of stack_hmc_chains, declared in hmc_chain_stack.h.

#include <Rcpp.h>
#include "hmc_chain_stack.h"

namespace tulpa_hmc {

StackedChains stack_hmc_chains(
    const std::vector<HMCResultCpp>& chains,
    int n_chains,
    int n_params
) {
  if (n_chains < 1) Rcpp::stop("n_chains must be >= 1");
  if (static_cast<int>(chains.size()) != n_chains) {
    Rcpp::stop("multi-chain run returned %d chains, expected %d",
               static_cast<int>(chains.size()), n_chains);
  }

  std::vector<int> ns(n_chains);
  int n_total = 0;
  bool equal_counts = true;
  for (int c = 0; c < n_chains; c++) {
    ns[c] = usable_chain_samples(chains[c], n_params);
    if (ns[c] != ns[0]) equal_counts = false;
    n_total += ns[c];
  }

  StackedChains out;
  out.n_total = n_total;
  out.n_sample_per_chain = equal_counts ? ns[0] : -1;
  out.draws = Rcpp::NumericMatrix(n_total, n_params);
  out.chain_id = Rcpp::IntegerVector(n_total);
  out.log_prob = Rcpp::NumericVector(n_total);
  out.accept_prob = Rcpp::NumericVector(n_total);
  out.divergent = Rcpp::IntegerVector(n_total);
  out.treedepth = Rcpp::IntegerVector(n_total);
  out.energy = Rcpp::NumericVector(n_total);
  out.epsilon = Rcpp::NumericVector(n_chains);
  out.inv_metric = Rcpp::NumericMatrix(n_chains, n_params);
  out.final_position = Rcpp::NumericMatrix(n_chains, n_params);

  int r = 0;
  for (int c = 0; c < n_chains; c++) {
    const HMCResultCpp& ch = chains[c];
    for (int s = 0; s < ns[c]; s++) {
      const double* row = ch.sample_row(s);
      for (int j = 0; j < n_params; j++) out.draws(r, j) = row[j];
      out.chain_id[r] = c + 1;
      out.log_prob[r] = ch.log_prob[s];
      out.accept_prob[r] = ch.accept_prob[s];
      out.divergent[r] = ch.divergent[s];
      out.treedepth[r] = ch.treedepth[s];
      out.energy[r] = ch.energy[s];
      r++;
    }
    out.epsilon[c] = ch.epsilon;
    out.n_max_treedepth += ch.n_max_treedepth;
    // A short vector defaults to the identity metric / the origin rather than
    // reading past the end.
    for (int j = 0; j < n_params; j++) {
      out.inv_metric(c, j) = (j < static_cast<int>(ch.inv_metric_diag.size()))
                                 ? ch.inv_metric_diag[j] : 1.0;
      out.final_position(c, j) = (j < static_cast<int>(ch.final_position.size()))
                                     ? ch.final_position[j] : 0.0;
    }
  }

  return out;
}

} // namespace tulpa_hmc
