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
  Rcpp::NumericVector energy;
  Rcpp::NumericVector epsilon;
  Rcpp::NumericMatrix inv_metric;
  Rcpp::NumericMatrix final_position;
  // Post-warmup max-treedepth saturations, summed over chains. The per-sample
  // `treedepth` vector beside it covers the same iterations; this is the count
  // the chain itself keeps and checkpoints (gcol33/tulpa#703).
  int n_max_treedepth = 0;
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
  if (ch.energy.size() < need) ns = std::min<int>(ns, static_cast<int>(ch.energy.size()));
  const int stride = ch.n_params_stored;
  if (stride < n_params) return 0;
  const int rows_stored = (stride > 0)
      ? static_cast<int>(ch.samples_flat.size() / static_cast<std::size_t>(stride))
      : 0;
  return std::min(ns, rows_stored);
}

StackedChains stack_hmc_chains(
    const std::vector<HMCResultCpp>& chains,
    int n_chains,
    int n_params
);

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_CHAIN_STACK_H
