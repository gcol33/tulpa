// hmc_latent.h
// Data types for the latent-factor block: observation-level random effects
// shared between the arms of a model. The densities, constraints and
// parameter transforms over these types are templated on the scalar type and
// live in hmc_latent_autodiff.h.

#ifndef TULPA_HMC_LATENT_H
#define TULPA_HMC_LATENT_H

namespace tulpa_latent {

// Identifiability constraint on a factor column. SUM_TO_ZERO centres the
// column; FIRST_ZERO shifts it so its first entry is exactly zero.
enum class LatentConstraint { SUM_TO_ZERO, FIRST_ZERO };

struct LatentFactorData {
  int n_factors;                    // Number of latent factors (K)
  int n_obs;                        // Number of observations (N)
  bool shared;                      // Whether factors enter both num and denom
  bool scale;                       // Whether to standardize factors
  LatentConstraint constraint;      // Identifiability constraint
  double sigma_prior_rate;          // Exponential rate for PC prior on sigma
};

}  // namespace tulpa_latent

#endif  // TULPA_HMC_LATENT_H
