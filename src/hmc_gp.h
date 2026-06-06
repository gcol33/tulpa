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
#include "pc_prior.h"  // tulpa::log_prior_sigma2_pc

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
  static const tulpa::EnumEntry<MSGPSampler> table[] = {
      {"noncentered", MSGPSampler::NONCENTERED}, {"auto", MSGPSampler::NONCENTERED},
      {"centered", MSGPSampler::CENTERED},
      {"interweaved", MSGPSampler::INTERWEAVED},
      {"adaptive", MSGPSampler::ADAPTIVE},
      {"riemannian", MSGPSampler::RIEMANNIAN},
      {"lbfgs", MSGPSampler::LBFGS}
  };
  return tulpa::parse_enum(s, table, MSGPSampler::NONCENTERED);
}

#include "hmc_gp_lbfgs.h"

#include "hmc_gp_solvers.h"

#include "hmc_gp_log_lik.h"

#include "hmc_gp_gradients.h"

#include "hmc_gp_nc.h"

} // namespace tulpa_gp

#endif // TULPA_HMC_GP_H
