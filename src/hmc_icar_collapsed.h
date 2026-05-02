// hmc_icar_collapsed.h
// Collapsed/marginalized ICAR and BYM2 spatial effects via inner Laplace optimization
//
// Instead of sampling S (ICAR) or 2S (BYM2) spatial effects alongside hyperparameters,
// we marginalize them out by finding phi* = argmax [log p(y|phi,theta_outer) + log p(phi|tau)]
// at each HMC gradient evaluation.
//
// ICAR: reduces S+1 params (log_tau + S phi) to just 1 (log_tau)
// BYM2: reduces 2S+2 params (log_sigma, logit_rho, S phi, S theta) to 2 (log_sigma, logit_rho)
//
// The collapsed log-posterior is:
//   log p(theta|y) ~ log p(y|phi*,theta) + log p(phi*|tau) + log p(theta)
//                    - 0.5 * log det(W + tau*Q)  [Laplace correction]
//
// Key advantage over collapsed GP: Q is FIXED (adjacency-based), doesn't depend on
// hyperparameters. Only tau*Q changes. This makes numerical Laplace gradient cheaper.
//
// This umbrella header is a thin re-export. Each fragment is fully
// self-contained: it includes its own dependencies and defines symbols
// inside namespace tulpa_hmc. The fragments can be included individually
// in any order — header guards make repeats free.

#ifndef TULPA_HMC_ICAR_COLLAPSED_H
#define TULPA_HMC_ICAR_COLLAPSED_H

#include "hmc_icar_collapsed_workspace.h"
#include "hmc_icar_collapsed_unit_lik.h"
#include "hmc_icar_collapsed_kernels.h"
#include "hmc_icar_collapsed_logdet.h"
#include "hmc_icar_collapsed_grad.h"
#include "hmc_icar_collapsed_mode.h"
#include "hmc_icar_collapsed_full.h"
#include "hmc_icar_collapsed_grad_H.h"
#include "hmc_icar_collapsed_log_post.h"

#endif // TULPA_HMC_ICAR_COLLAPSED_H
