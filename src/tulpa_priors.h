// tulpa_priors.h
// Umbrella header: re-exports the 12 sliced prior headers. The split mirrors
// the laplace_helpers.h decomposition (commit 1fa2a45) so each prior family
// lives in a focused file while downstream callers (#include "tulpa_priors.h")
// see no API change.
//
// Prerequisite: ModelData and ParamLayout must be defined before this
// header (normally via hmc_sampler.h).

#ifndef TULPA_PRIORS_H
#define TULPA_PRIORS_H

#include "tulpa_priors_re.h"
#include "tulpa_priors_icar.h"
#include "tulpa_priors_gp.h"
#include "tulpa_priors_msgp.h"
#include "tulpa_priors_hsgp.h"
#include "tulpa_priors_temporal.h"
#include "tulpa_priors_mstemporal.h"
#include "tulpa_priors_svc.h"
#include "tulpa_priors_tvc.h"
#include "tulpa_priors_latent.h"
#include "tulpa_priors_st.h"
#include "tulpa_priors_zioi.h"

#endif // TULPA_PRIORS_H
