// hmc_gradient_gp_impl.h
// Umbrella for GP / collapsed-GP / ICAR-collapsed / temporal-GP hand-coded
// gradient functions. Split on 2026-05-02 from a 1306-line monolith.
// Each fragment contains complete function definitions; they are not
// standalone-compilable (they rely on namespace tulpa_hmc and the helper
// includes set up at the top of hmc_gradients.cpp). Each fragment is
// `#include`d exactly once below, so no header guards.

#include "hmc_gradient_gp_handcoded.h"
#include "hmc_gradient_gp_collapsed.h"
#include "hmc_gradient_icar_collapsed_grad.h"
#include "hmc_gradient_temporal_gp.h"
