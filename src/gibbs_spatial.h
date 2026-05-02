// gibbs_spatial.h
// Umbrella header: re-exports the four sliced gibbs_spatial_* headers.
// Component-wise Gibbs sampler for ICAR / BYM2 / HSGP spatial models.
// Designed for large S where HMC struggles with dimensionality.
//
// Update scheme (per family):
//   1. spatial effects | rest    (univariate MH per site, or block for HSGP)
//   2. variance hypers | spatial (conjugate Gamma where possible)
//   3. beta            | rest    (block MH with Gaussian proposal)
//   4. dispersion      | rest    (univariate MH on log scale)

#pragma once

#include "gibbs_spatial_data.h"
#include "gibbs_spatial_icar.h"
#include "gibbs_spatial_bym2.h"
#include "gibbs_spatial_hsgp.h"
