// hmc_walnuts.h
// WALNUTS transition: NUTS with a locally adapted step size per macro step
// (Bou-Rabee, Carpenter, Kleppe, Liu, Marsden & Shaw 2025, arXiv 2506.18746).
//
// Each macro step of the orbit is integrated with the coarsest leapfrog
// subdivision -- min_micro_steps * 2^k micro steps of size step / 2^k -- whose
// joint log density moves by at most max_error, and is kept only if the
// subdivision is the one the reversed step would pick (no coarser count also
// meets the tolerance from the end state with the momentum flipped). The
// selection rule is a deterministic function of the state, so together with
// that check the orbit map is an involution and the kernel leaves the target
// invariant. This is the WALNUTS-D variant of section 4.1, ported from the
// reference implementation in flatironinstitute/walnutpie (walnuts.hpp,
// transition_w; MIT, see inst/COPYRIGHTS), generalized from a diagonal to any
// metric DenseMassMatrix carries.
//
// Within a subtree states are selected by the Barker rule, at the top level by
// the biased-progressive Metropolis rule. A subtree whose leaf cannot be built
// (no subdivision within tolerance, or a non-reversible one) or that contains a
// sub-U-turn ends the orbit and is discarded, as in NUTS.
//
// The leapfrog is the plain Stormer-Verlet step regardless of the process-global
// integrator selection: the error-controlled subdivision is defined on it.

#ifndef TULPA_HMC_WALNUTS_H
#define TULPA_HMC_WALNUTS_H

#include <random>
#include <vector>

#include "hmc_sampler.h"
#include "hmc_walnuts_config.h"

namespace tulpa_hmc {

// One end (or the selected point) of an orbit segment.
struct WalnutsState {
  std::vector<double> theta;
  std::vector<double> rho;
  std::vector<double> grad;
  double logp_pos = 0.0;    // target log density at theta
  double logp_joint = 0.0;  // logp_pos - kinetic energy(rho)

  void resize(int n) { theta.resize(n); rho.resize(n); grad.resize(n); }
};

// A contiguous orbit segment: its two ends, the state selected from it, and
// the log of the summed joint densities over its states.
struct WalnutsSpan {
  WalnutsState bk;
  WalnutsState fw;
  WalnutsState select;  // rho unused
  double logp = 0.0;

  void resize(int n) { bk.resize(n); fw.resize(n); select.resize(n); }
};

// Per-chain buffers, sized once: one temporary span per tree level, the
// accumulated orbit, the segment being added, and the macro-step scratch.
struct WalnutsWorkspace {
  int n = 0;
  std::vector<WalnutsSpan> level_tmp;  // index = subtree depth
  WalnutsSpan accum;
  WalnutsSpan next;
  WalnutsState step_state;  // macro-step result
  WalnutsState rev_state;   // reversibility-check scratch
  std::vector<double> diff;
  std::vector<double> diff_sharp;
  std::vector<double> drift_scratch;

  // Per-transition tallies.
  int n_grad = 0;
  double accept_sum = 0.0;
  int accept_count = 0;
  bool divergent = false;

  void init(int n_params, int max_depth);
};

struct WalnutsTransitionResult {
  int depth = 0;           // doublings in the final orbit
  int n_grad = 0;          // gradient evaluations, reversibility checks included
  double mean_accept = 0;  // mean first-subdivision acceptance, for step-size adaptation
  bool divergent = false;  // some macro step met no subdivision within tolerance
  double H0 = 0.0;         // starting Hamiltonian
};

// One WALNUTS transition from (q, grad, log_post), which are overwritten with
// the selected state. `step` is the macro step size.
WalnutsTransitionResult walnuts_transition(
    std::vector<double>& q, std::vector<double>& grad, double& log_post,
    double step, int max_depth, const WalnutsConfig& cfg,
    const DenseMassMatrix& mass, GradientFn gradient_fn,
    const ModelData& data, const ParamLayout& layout,
    WalnutsWorkspace& ws, std::mt19937& rng);

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_WALNUTS_H
