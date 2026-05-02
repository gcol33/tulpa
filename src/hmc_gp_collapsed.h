// hmc_gp_collapsed.h
// Collapsed/marginalized GP spatial effects via inner Laplace optimization
//
// Instead of sampling N_gp spatial effects w alongside hyperparameters,
// we marginalize w out by finding w* = argmax_w [log p(y|w,theta) + log p_NNGP(w|sigma2,phi)]
// at each HMC gradient evaluation. This reduces dimensionality from
// N_gp + 2 (hyperparams) to just 2 hyperparams.
//
// The collapsed log-posterior is:
//   log p(theta|y) ~ log p(y|w*,theta) + log p_NNGP(w*|sigma2,phi) + log p(theta)
//                    - 0.5 * log det(W + Q)  [optional log-det correction]
//
// where W = diag(neg_hessian_data) and Q = NNGP precision matrix.
//
// The gradient dL/dtheta uses the implicit function theorem:
// at the mode w*, dL/dw = 0, so dw*/dtheta terms vanish and
// dL/dtheta = partial_L/partial_theta |_{w=w*}

#ifndef TULPA_HMC_GP_COLLAPSED_H
#define TULPA_HMC_GP_COLLAPSED_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppEigen.h>
// NOTE: This header must be included AFTER hmc_sampler.h (which defines ModelData/ModelType)
// and hmc_gp.h (which defines GPData in namespace tulpa_gp).
// It is included from hmc_sampler.cpp in the correct order.

using tulpa_gp::GPData;
using tulpa_svc::compute_cov;
using tulpa_hmc::ModelData;
using tulpa_hmc::ModelType;

#include "hmc_gp_collapsed_ops.h"

#include "hmc_gp_collapsed_logdet.h"

#include "hmc_gp_collapsed_mode.h"

#include "hmc_gp_collapsed_grad.h"

#include "hmc_gp_collapsed_post.h"

#endif // TULPA_HMC_GP_COLLAPSED_H
