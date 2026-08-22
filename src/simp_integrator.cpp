// simp_integrator.cpp
// Active symplectic integrator for the HMC/NUTS trajectory, backed by SIMP.
// The scheme (op sequence + coefficients) is the single source of truth shared
// by both leapfrog steppers; selecting an integrator swaps the scheme here.

#include <Rcpp.h>
#include <algorithm>
#include <string>

#include "hmc_sampler_decls.h"  // simp/scheme.h, the extern declarations
#include "simp/adapt.h"         // step-adapted multistage constructors (placeholder)

namespace tulpa_hmc {

// Half-width of the dimensionless step band the fixed three-stage placeholder
// is optimised over. "adaptive3" resolves its own band per chain at warmup end
// (compute_adaptive_nu_max); this is the band the placeholder runs with until
// then, and on the fixed-trajectory HMC path that never resolves one.
constexpr double kThreeStagePlaceholderBand = 2.0;

// Ceiling on the RESPA inner-substep count. Each substep is one extra prior
// gradient per trajectory leaf, so a mistyped value multiplies the whole
// trajectory's cost with nothing to stop it; the bound is far above any useful
// splitting ratio and exists to turn a typo into a message.
constexpr int kMaxMtsSubsteps = 1024;

// Default is plain leapfrog, so an unset integrator reproduces the historical
// behaviour byte-for-byte.
simp::Scheme g_integrator_scheme = simp::leapfrog();
IntegratorAdaptive g_integrator_adaptive = IntegratorAdaptive::NONE;
bool g_integrator_mts = false;
int g_mts_substeps = 4;
std::string g_integrator_name = "leapfrog";

void set_integrator_scheme(const std::string& name, int mts_substeps) {
  // Validate everything before any of the five globals moves, so a rejected
  // call leaves the process on the integrator it was already using rather than
  // on a half-reset one. scheme_by_name throws on an unknown name, so the
  // unnamed branches resolve their scheme here too.
  if (name == "mts" &&
      (mts_substeps < 1 || mts_substeps > kMaxMtsSubsteps)) {
    Rcpp::stop("tulpa_integrator: `mts_substeps` is %d; the RESPA inner "
               "substep count must lie in [1, %d]. Each substep is one extra "
               "prior gradient per trajectory step.",
               mts_substeps, kMaxMtsSubsteps);
  }
  const bool named_scheme = (name != "adaptive2" && name != "adaptive3" &&
                             name != "mts");
  simp::Scheme resolved =
      named_scheme ? simp::scheme_by_name(name) : simp::leapfrog();

  // Reset the orthogonal selectors; each branch sets the ones it needs.
  g_integrator_adaptive = IntegratorAdaptive::NONE;
  g_integrator_mts = false;

  if (name == "adaptive2") {
    // Adaptive selections defer the coefficient to warmup end (per chain). The
    // placeholder is a fixed member of the SAME stage family, so warmup runs it
    // and the dual-averaged step size transfers to the resolved scheme.
    g_integrator_adaptive = IntegratorAdaptive::TWO_STAGE;
    g_integrator_scheme = simp::minerror2();          // two-stage, nu->0 optimum
  } else if (name == "adaptive3") {
    g_integrator_adaptive = IntegratorAdaptive::THREE_STAGE;
    g_integrator_scheme =
        simp::three_stage_adaptive(kThreeStagePlaceholderBand);
  } else if (name == "mts") {
    // Multiple-time-stepping: the RESPA leaf splits prior (fast) from
    // likelihood (slow). The inner substeps use a leapfrog structure; the
    // scheme field is unused by the MTS leaf but kept valid.
    g_integrator_mts = true;
    g_mts_substeps = mts_substeps;
    g_integrator_scheme = simp::leapfrog();
  } else {
    g_integrator_scheme = resolved;
  }
  g_integrator_name = name;
}

const simp::Scheme& get_integrator_scheme() { return g_integrator_scheme; }
IntegratorAdaptive get_integrator_adaptive() { return g_integrator_adaptive; }
bool get_integrator_mts() { return g_integrator_mts; }
int get_mts_substeps() { return g_mts_substeps; }
const std::string& get_integrator_name() { return g_integrator_name; }

}  // namespace tulpa_hmc

// Select the trajectory integrator by name. mts_substeps sets the inner-substep
// count for the "mts" integrator (ignored otherwise). Returns the previous name
// so the caller can restore it.
//
// The selection is process-global, so a caller that changes it owns restoring
// it: with_tulpa_integrator() (R/integrator.R) is the scoped form, and it
// restores on error as well as on success. Throws (propagated to R) on an
// unknown name or an out-of-range substep count -- validated before any of the
// five globals is touched, so a rejected call leaves the process on the
// integrator it was already using.
// [[Rcpp::export]]
std::string tulpa_set_integrator_cpp(std::string name, int mts_substeps = 4) {
  std::string previous = tulpa_hmc::get_integrator_name();
  tulpa_hmc::set_integrator_scheme(name, mts_substeps);
  return previous;
}

// [[Rcpp::export]]
std::string tulpa_get_integrator_cpp() {
  return tulpa_hmc::get_integrator_name();
}
