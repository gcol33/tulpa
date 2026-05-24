// hmc_sampler_chain_state.h
// Fragment of hmc_sampler.h. Self-contained: defines symbols inside
// namespace tulpa_hmc.
// ChainState, HMCResultCpp, HMCResult, cpp_to_r_result,
// NUTS helper function declarations.
#ifndef TULPA_HMC_SAMPLER_CHAIN_STATE_H
#define TULPA_HMC_SAMPLER_CHAIN_STATE_H

#include <random>
#include <string>
#include <vector>

#include <Rcpp.h>

#include "hmc_sampler_adapt.h"        // DualAveraging
#include "hmc_sampler_decls.h"        // ModelData, ParamLayout
#include "hmc_sampler_mass_blocks.h"  // DenseMassMatrix
#include "hmc_sampler_nuts_infra.h"   // LeapfrogResultWithGrad, NUTSWorkspace, etc.

namespace tulpa_hmc {

struct ChainState {
  std::vector<double> q;
  double log_prob;
  double epsilon;
  DualAveraging da;
  std::mt19937 rng;
  int n_divergent;
};

// Pure C++ result struct (safe for OpenMP parallel regions)
struct HMCResultCpp {
  std::vector<double> samples_flat;  // n_sample × n_params, row-major contiguous
  int n_params_stored = 0;
  std::vector<double> log_prob;
  std::vector<double> accept_prob;
  std::vector<int> n_leapfrog;
  std::vector<int> divergent;
  std::vector<int> treedepth;    // Actual tree depth per iteration (NUTS only)
  double epsilon;
  int n_warmup;
  int n_sample;
  int chain_id;
  int n_max_treedepth = 0;       // Count of iterations hitting max treedepth
  std::string sampler;           // Sampler name (e.g., "NUTS", "HMC", "NUTS->HMC(L=10)")

  // Warm-start / resume outputs (gcol33/tulpa#29). Both length n_params after
  // a completed run. `inv_metric_diag` is the adapted inverse-mass diagonal at
  // end of warmup; `final_position` is the last raw sampler state (sampling
  // parameterization, e.g. z for an NC GP, not the stored w transform). Feeding
  // them back as inv_metric_init + q_init with n_warmup=0 continues the chain.
  std::vector<double> inv_metric_diag;
  std::vector<double> final_position;

  // Collapsed mode draws (populated only when collapsed parameterization active)
  int n_gp_collapsed = 0;                         // N_gp if collapsed GP, 0 otherwise
  int n_icar_collapsed = 0;                        // S if collapsed ICAR/BYM2, 0 otherwise
  std::vector<double> gp_w_star_flat;              // w* draws (n_sample x N_gp, row-major)
  std::vector<double> icar_phi_star_flat;          // phi* draws (n_sample x S, row-major)
  std::vector<double> bym2_theta_star_flat;        // theta* draws (n_sample x S, row-major)

  // Row access for flat storage
  double* sample_row(int i) { return &samples_flat[i * n_params_stored]; }
  const double* sample_row(int i) const { return &samples_flat[i * n_params_stored]; }
};

// R-compatible result struct (create outside parallel regions)
struct HMCResult {
  Rcpp::NumericMatrix samples;
  Rcpp::NumericVector log_prob;
  Rcpp::NumericVector accept_prob;
  Rcpp::IntegerVector n_leapfrog;
  Rcpp::IntegerVector treedepth;
  Rcpp::IntegerVector divergent;
  double epsilon;
  int n_warmup;
  int n_sample;
  int chain_id;
  std::string sampler;

  // Collapsed mode draws (populated only when collapsed parameterization active)
  int n_gp_collapsed = 0;
  int n_icar_collapsed = 0;
  Rcpp::NumericMatrix gp_w_star;
  Rcpp::NumericMatrix icar_phi_star;
  Rcpp::NumericMatrix bym2_theta_star;
};

// Convert C++ result to R result (call outside parallel region)
inline HMCResult cpp_to_r_result(const HMCResultCpp& cpp_result, int n_params) {
  HMCResult r_result;
  r_result.samples = Rcpp::NumericMatrix(cpp_result.n_sample, n_params);
  r_result.log_prob = Rcpp::NumericVector(cpp_result.n_sample);
  r_result.accept_prob = Rcpp::NumericVector(cpp_result.n_sample);
  r_result.n_leapfrog = Rcpp::IntegerVector(cpp_result.n_sample);
  r_result.treedepth = Rcpp::IntegerVector(cpp_result.n_sample);
  r_result.divergent = Rcpp::IntegerVector(cpp_result.n_sample);
  r_result.epsilon = cpp_result.epsilon;
  r_result.n_warmup = cpp_result.n_warmup;
  r_result.n_sample = cpp_result.n_sample;
  r_result.chain_id = cpp_result.chain_id;
  r_result.sampler = cpp_result.sampler;

  for (int i = 0; i < cpp_result.n_sample; i++) {
    const double* row = cpp_result.sample_row(i);
    for (int j = 0; j < n_params; j++) {
      r_result.samples(i, j) = row[j];
    }
    r_result.log_prob[i] = cpp_result.log_prob[i];
    r_result.accept_prob[i] = cpp_result.accept_prob[i];
    r_result.n_leapfrog[i] = cpp_result.n_leapfrog[i];
    r_result.treedepth[i] = cpp_result.treedepth[i];
    r_result.divergent[i] = cpp_result.divergent[i];
  }

  // Collapsed GP: copy w* draws
  if (cpp_result.n_gp_collapsed > 0) {
    int n_gp = cpp_result.n_gp_collapsed;
    r_result.n_gp_collapsed = n_gp;
    r_result.gp_w_star = Rcpp::NumericMatrix(cpp_result.n_sample, n_gp);
    for (int i = 0; i < cpp_result.n_sample; i++) {
      for (int j = 0; j < n_gp; j++) {
        r_result.gp_w_star(i, j) = cpp_result.gp_w_star_flat[i * n_gp + j];
      }
    }
  }

  // Collapsed ICAR/BYM2: copy phi* and theta* draws
  if (cpp_result.n_icar_collapsed > 0) {
    int S = cpp_result.n_icar_collapsed;
    r_result.n_icar_collapsed = S;
    r_result.icar_phi_star = Rcpp::NumericMatrix(cpp_result.n_sample, S);
    for (int i = 0; i < cpp_result.n_sample; i++) {
      for (int j = 0; j < S; j++) {
        r_result.icar_phi_star(i, j) = cpp_result.icar_phi_star_flat[i * S + j];
      }
    }
    // BYM2: also copy theta*
    if (!cpp_result.bym2_theta_star_flat.empty()) {
      r_result.bym2_theta_star = Rcpp::NumericMatrix(cpp_result.n_sample, S);
      for (int i = 0; i < cpp_result.n_sample; i++) {
        for (int j = 0; j < S; j++) {
          r_result.bym2_theta_star(i, j) = cpp_result.bym2_theta_star_flat[i * S + j];
        }
      }
    }
  }

  return r_result;
}

// NUTS helper function declarations
double nuts_log_sum_exp(double a, double b);
double nuts_compute_hamiltonian(double log_prob, const std::vector<double>& p,
                                const std::vector<double>& inv_mass, int n);
bool nuts_check_uturn(const std::vector<double>& q_minus, const std::vector<double>& q_plus,
                      const std::vector<double>& p_minus, const std::vector<double>& p_plus,
                      const std::vector<double>& inv_mass, int n);
LeapfrogResultWithGrad leapfrog_step_with_grad(
    const std::vector<double>& q, const std::vector<double>& p,
    const std::vector<double>& grad,
    double epsilon, const std::vector<double>& inv_mass,
    bool use_mass, const ModelData& data, const ParamLayout& layout);
// Optimized NUTS: zero-allocation in-place leapfrog + buffer pool tree building
// Pointer-based Hamiltonian (no vector overhead)
double nuts_compute_hamiltonian_fast(double log_prob, const double* p,
                                     const DenseMassMatrix& mass, int n);
// Pointer-based U-turn check
bool nuts_check_uturn_fast(const double* q_minus, const double* q_plus,
                           const double* p_minus, const double* p_plus,
                           const DenseMassMatrix& mass, double* scratch, int n);
// In-place leapfrog step operating on workspace slot
LeapfrogInPlaceResult leapfrog_step_inplace(
    NUTSWorkspace& ws, int slot, double epsilon,
    const DenseMassMatrix& mass,
    const ModelData& data, const ParamLayout& layout);
// Zero-allocation recursive tree builder
TreeStats build_tree_fast(
    NUTSWorkspace& ws, int input_slot, int direction, int depth,
    double epsilon, const DenseMassMatrix& mass,
    double H0, double delta_max,
    const ModelData& data, const ParamLayout& layout,
    std::mt19937& rng);

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_SAMPLER_CHAIN_STATE_H
