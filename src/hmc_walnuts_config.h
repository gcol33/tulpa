// hmc_walnuts_config.h
// Tuning constants of the WALNUTS transition (hmc_walnuts.h). Split out so the
// chain-driver declarations can take one without pulling the transition in.
#ifndef TULPA_HMC_WALNUTS_CONFIG_H
#define TULPA_HMC_WALNUTS_CONFIG_H

namespace tulpa_hmc {

// min_micro_steps and max_error are walnutpie's SamplingConfig defaults. Its
// max_step_halvings default of 5 caps a macro step at 16 micro steps, which the
// neck of Neal's funnel (K = 9, gamma = 3) outruns: 1393 failed macro steps a
// 100k-iteration chain and sd(v) 2.89 against 3 over 20 seeds. At 10 the same
// chains fail 2.8 times, read sd(v) 2.98, and take 40.5 gradients an iteration
// against 40.0.
struct WalnutsConfig {
  int max_step_halvings = 10;  // subdivisions tried per macro step: 1, 2, .., 2^(h-1)
  int min_micro_steps = 1;     // micro steps in the coarsest subdivision
  double max_error = 0.5;      // tolerated |change| in joint log density per macro step
};

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_WALNUTS_CONFIG_H
