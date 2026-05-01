// hmc_gradient_check.h
// Diagnostics for comparing HMC gradient implementations.
//
// Used by gradient_check_only Rcpp paths. The function returns the same R list
// schema that cpp_hmc_fit() and cpp_hmc_fit_gp() returned before the diagnostic
// code was split out.

#ifndef TULPA_HMC_GRADIENT_CHECK_H
#define TULPA_HMC_GRADIENT_CHECK_H

#include "hmc_sampler.h"
#include <Rcpp.h>
#include <vector>

namespace tulpa_hmc {

Rcpp::List run_gradient_check_only(const std::vector<double>& q0,
                                   const ModelData& data);

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_GRADIENT_CHECK_H
