// hmc_sampler_funcs.h
// Fragment of hmc_sampler.h. Self-contained: defines symbols inside
// namespace tulpa_hmc.
// Sampler function declarations (leapfrog, find_reasonable_epsilon,
// run_hmc_*) and SoftAbs metric helpers.
#ifndef TULPA_HMC_SAMPLER_FUNCS_H
#define TULPA_HMC_SAMPLER_FUNCS_H

#include <random>
#include <vector>

#include "hmc_sampler_chain_state.h"  // HMCResultCpp, HMCResult
#include "hmc_sampler_decls.h"        // ModelData, ParamLayout
#include "hmc_sampler_mass_blocks.h"  // DenseMassMatrix, MassMatrixType
#include "hmc_sampler_nuts_infra.h"   // LeapfrogResult

namespace tulpa_hmc {

// =====================================================================
// Sampler functions
// =====================================================================

// Unified leapfrog step: identity mass when inv_mass is nullptr
LeapfrogResult leapfrog_step(
    const std::vector<double>& q,
    const std::vector<double>& p,
    double epsilon,
    const ModelData& data,
    const ParamLayout& layout,
    const double* inv_mass = nullptr
);

// Find reasonable initial step size
double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng
);

// Mass-aware version: uses diagonal mass matrix for leapfrog and kinetic energy
double find_reasonable_epsilon(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const std::vector<double>& inv_mass
);

// Dense-mass-aware version: uses full DenseMassMatrix for momentum, leapfrog, and kinetic energy
double find_reasonable_epsilon_dense(
    const std::vector<double>& q,
    const ModelData& data,
    const ParamLayout& layout,
    std::mt19937& rng,
    const DenseMassMatrix& mass
);

// Run single HMC chain (C++ version - safe for parallel)
// riemannian: -1=auto (retry divergences with SoftAbs for BYM2/ICAR),
//              1=force on, 0=force off
// inv_metric_init: optional caller-supplied initial diagonal inv-mass
//                  (length n_params). Empty -> default structural warm-start.
HMCResultCpp run_hmc_chain_cpp(
    const std::vector<double>& q_init,
    const ModelData& data,
    const ParamLayout& layout,
    int n_iter,
    int n_warmup,
    int L,
    int chain_id,
    unsigned int seed,
    bool verbose,
    int max_treedepth = 10,
    MassMatrixType metric_type = MassMatrixType::DIAG,
    double adapt_delta = -1.0,
    int riemannian = -1,
    const std::vector<double>& inv_metric_init = std::vector<double>()
);

// Run single HMC chain (R wrapper)
HMCResult run_hmc_chain(
    const std::vector<double>& q_init,
    const ModelData& data,
    const ParamLayout& layout,
    int n_iter,
    int n_warmup,
    int L,
    int chain_id,
    unsigned int seed,
    bool verbose,
    int max_treedepth = 10,
    MassMatrixType metric_type = MassMatrixType::DIAG,
    double adapt_delta = -1.0,
    int riemannian = -1,
    const std::vector<double>& inv_metric_init = std::vector<double>()
);

// Run multiple chains in parallel (across-chain parallelization)
std::vector<HMCResult> run_hmc_parallel_chains(
    const std::vector<double>& q_init,
    const ModelData& data,
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    bool verbose,
    int max_treedepth = 10,
    MassMatrixType metric_type = MassMatrixType::DIAG,
    double adapt_delta = -1.0,
    int riemannian = -1
);

// Pure-C++ across-chain core (no Rcpp types -> callable from the C ABI).
// q_init_per_chain: length n_chains (chain c starts at entry c).
// inv_metric_per_chain: empty (all default) or length n_chains (entry c may
// itself be empty). Together these let one call do a fresh fit (broadcast
// init) or a warm-started resume (per-chain final_position + inv_metric,
// n_warmup = 0). See gcol33/tulpa#30.
std::vector<HMCResultCpp> run_hmc_parallel_chains_cpp(
    const std::vector<std::vector<double>>& q_init_per_chain,
    const std::vector<std::vector<double>>& inv_metric_per_chain,
    const ModelData& data,
    int n_iter,
    int n_warmup,
    int L,
    int n_chains,
    unsigned int seed,
    bool verbose,
    int max_treedepth = 10,
    MassMatrixType metric_type = MassMatrixType::DIAG,
    double adapt_delta = -1.0,
    int riemannian = -1,
    const std::string& checkpoint_path = "",
    // Optional caller-supplied layout (gcol33/tulpa#70). When non-null it is used
    // verbatim instead of compute_param_layout(data), so a model fitting through
    // a FullGradFn with a hand-built ParamLayout (total_params set, no processes)
    // gets the correct parameter count -- the single-chain runner already honours
    // the passed layout; this brings the multi-chain runner in line.
    const ParamLayout* layout_override = nullptr
);

// =====================================================================
// SoftAbs per-trajectory metric (Riemannian-like divergence retry)
// =====================================================================

// Compute full Hessian via finite differences of the H-mode gradient.
// H[i,j] = (grad_j(q + h*e_i) - grad_j(q)) / h
// Cost: (p+1) gradient evaluations.
void compute_hessian_finite_diff(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& hessian,
    double h = 1e-5
);

// Compute SoftAbs metric from negative Hessian.
// G = Q diag(f(λ_i)) Q^T where f(λ) = λ * coth(α * λ)
// Returns G^{-1} and its Cholesky L. Returns false on failure.
bool compute_softabs_metric(
    const std::vector<double>& neg_hessian,
    int p,
    double alpha,
    std::vector<double>& G_inv,
    std::vector<double>& L_G_inv
);

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_SAMPLER_FUNCS_H
