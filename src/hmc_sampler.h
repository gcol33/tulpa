// hmc_sampler.h
// Full HMC/NUTS backend with spatial, temporal, and zero-inflation support
// Supports ICAR/BYM2 spatial effects, RW/AR1 temporal, and ZI/hurdle models
//
// This umbrella is a thin re-export. Each fragment is self-contained:
// it includes its own dependencies and defines symbols inside namespace
// tulpa_hmc. The umbrella additionally pulls in the sibling latent-structure
// headers (tulpa_temporal, tulpa_spatiotemporal, etc.) that the
// rest of the backend expects to find through `hmc_sampler.h`.

#ifndef TULPA_HMC_SAMPLER_H
#define TULPA_HMC_SAMPLER_H

// Sibling latent-structure headers — separate namespaces, brought in here
// for callers (log_post_impl.h, gradient kernels, ESS, ...) that include
// hmc_sampler.h to access tulpa_temporal::, etc.
#include "hmc_temporal.h"
#include "hmc_temporal_gp.h"
#include "hmc_svc.h"
#include "hmc_gp.h"
#include "hmc_temporal_multiscale.h"
#include "hmc_latent.h"
#include "hmc_spatiotemporal.h"
#include "hmc_hsgp.h"
#include "hmc_tvc.h"

// tulpa_hmc:: fragments — each self-contained.
#include "hmc_sampler_decls.h"
#include "hmc_sampler_nuts_infra.h"
#include "hmc_sampler_mass_blocks.h"
#include "hmc_sampler_adapt.h"
#include "hmc_sampler_chain_state.h"
#include "hmc_sampler_funcs.h"
#include "hmc_sampler_config.h"

#endif // TULPA_HMC_SAMPLER_H
