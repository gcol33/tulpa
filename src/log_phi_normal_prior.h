#ifndef TULPA_LOG_PHI_NORMAL_PRIOR_H
#define TULPA_LOG_PHI_NORMAL_PRIOR_H

// Normal(0, sd) prior on a likelihood's single extra parameter log_phi, read
// at layout.extra_offset, with sd taken from the model data's
// `log_phi_prior_sd` field. Returned up to additive constants (the
// -log(sd) - 0.5*log(2*pi) term cancels in NUTS). The two overloads are the
// LikelihoodSpec extra_prior / extra_prior_arena callbacks for any model data
// type carrying that field.

#include <vector>
#include "tulpa/param_layout.h"
#include "tulpa/autodiff_arena.h"

namespace tulpa {

template<class Data>
double log_phi_normal_prior_double(
    const std::vector<double>& params,
    const ParamLayout& layout,
    const void* model_data
) {
    const auto* d  = static_cast<const Data*>(model_data);
    double log_phi = params[layout.extra_offset];
    double sd      = d->log_phi_prior_sd;
    return -0.5 * (log_phi / sd) * (log_phi / sd);
}

template<class Data>
tulpa::arena::Var log_phi_normal_prior_arena(
    const std::vector<tulpa::arena::Var>& params,
    const ParamLayout& layout,
    const void* model_data
) {
    using tulpa::arena::Var;
    const auto* d = static_cast<const Data*>(model_data);
    Var log_phi   = params[layout.extra_offset];
    double sd     = d->log_phi_prior_sd;
    Var z         = log_phi * Var(1.0 / sd);
    return Var(-0.5) * z * z;
}

}  // namespace tulpa

#endif
