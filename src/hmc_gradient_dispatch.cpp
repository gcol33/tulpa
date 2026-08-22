// hmc_gradient_dispatch.cpp
// Gradient-mode dispatch: the only translation unit that defines
// resolve_gradient_fn() and resolve_prior_gradient_fn(). Their bodies are the
// textual fragment hmc_gradient_dispatch.h, included below inside
// namespace tulpa_hmc; declarations are in hmc_sampler_decls.h.

#include "hmc_sampler.h"
#include "tulpa/likelihood.h"
#include <Rcpp.h>

namespace tulpa_hmc {

#include "hmc_gradient_dispatch.h"

} // namespace tulpa_hmc
