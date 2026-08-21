// hmc_chain_stack.h
// Chain-major stacking of a multi-chain NUTS run into R vectors/matrices.
// Shared by every Rcpp entry point that returns a multi-chain fit, so the
// output sizing and the per-chain bounds checks have one definition.

#ifndef TULPA_HMC_CHAIN_STACK_H
#define TULPA_HMC_CHAIN_STACK_H

#include <Rcpp.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "hmc_sampler.h"

namespace tulpa_hmc {

// Draws stacked chain-major: chain 1's iterations, then chain 2's, and so on.
//
// The row count is the SUM of the per-chain sample counts. Sizing it as
// n_chains * chains[0].n_sample assumes every chain returned the same number of
// draws, which per-chain checkpoint resume does not guarantee: a chain loaded
// from a file written under a different n_iter carries its own count, and the
// write then runs past the allocation.
struct StackedChains {
  Rcpp::NumericMatrix draws;
  Rcpp::IntegerVector chain_id;
  Rcpp::NumericVector log_prob;
  Rcpp::NumericVector accept_prob;
  Rcpp::IntegerVector divergent;
  Rcpp::IntegerVector treedepth;
  Rcpp::NumericVector epsilon;
  Rcpp::NumericMatrix inv_metric;
  Rcpp::NumericMatrix final_position;
  int n_total = 0;
  int n_sample_per_chain = 0;  // -1 when the chains disagree
};

// Sample count chain `c` can actually be read for: every per-iteration vector
// has to reach it, and so does the flat draw storage.
inline int usable_chain_samples(const HMCResultCpp& ch, int n_params) {
  int ns = ch.n_sample;
  if (ns < 0) ns = 0;
  const std::size_t need = static_cast<std::size_t>(ns);
  if (ch.log_prob.size() < need) ns = static_cast<int>(ch.log_prob.size());
  if (ch.accept_prob.size() < need) ns = std::min<int>(ns, static_cast<int>(ch.accept_prob.size()));
  if (ch.divergent.size() < need) ns = std::min<int>(ns, static_cast<int>(ch.divergent.size()));
  if (ch.treedepth.size() < need) ns = std::min<int>(ns, static_cast<int>(ch.treedepth.size()));
  const int stride = ch.n_params_stored;
  if (stride < n_params) return 0;
  const int rows_stored = (stride > 0)
      ? static_cast<int>(ch.samples_flat.size() / static_cast<std::size_t>(stride))
      : 0;
  return std::min(ns, rows_stored);
}

inline StackedChains stack_hmc_chains(
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
      r++;
    }
    out.epsilon[c] = ch.epsilon;
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

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_CHAIN_STACK_H
