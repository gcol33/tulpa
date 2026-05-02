// hmc_gp.h
// Gaussian Process spatial effects with NNGP approximation
// Supports single-scale GP and multi-scale (local + regional) GP

#ifndef TULPA_HMC_GP_H
#define TULPA_HMC_GP_H

#include <vector>
#include <cmath>
#include <random>
#include <RcppEigen.h>
#include "hmc_svc.h"  // Reuse covariance functions and NNGP infrastructure
#include "hmc_gp_cg.h"  // Iterative CG/PCG solvers (dense_cg_solve, dense_pcg_solve)
#include "tulpa/gp_data.h"
#include "tulpa/types.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// Verbose debug output (set to false for production)
#define GP_DEBUG_BOUNDS false

namespace tulpa_gp {

using tulpa::CovType;
using tulpa::GPData;
using tulpa::MultiscaleGPData;
using tulpa::MSGPSampler;
using tulpa::GPSolver;
using tulpa::GPSolverConfig;
using tulpa_svc::compute_cov;

// Parse sampler string to enum
inline MSGPSampler parse_msgp_sampler(const std::string& s) {
  if (s == "noncentered" || s == "auto") return MSGPSampler::NONCENTERED;
  if (s == "centered") return MSGPSampler::CENTERED;
  if (s == "interweaved") return MSGPSampler::INTERWEAVED;
  if (s == "adaptive") return MSGPSampler::ADAPTIVE;
  if (s == "riemannian") return MSGPSampler::RIEMANNIAN;
  if (s == "lbfgs") return MSGPSampler::LBFGS;
  return MSGPSampler::NONCENTERED;  // Default fallback
}

#include "hmc_gp_lbfgs.h"

#include "hmc_gp_solvers.h"

#include "hmc_gp_log_lik.h"

#include "hmc_gp_gradients.h"

#include "hmc_gp_nc.h"

} // namespace tulpa_gp

#endif // TULPA_HMC_GP_H
