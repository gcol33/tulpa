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

// Default is plain leapfrog, so an unset integrator reproduces the historical
// behaviour byte-for-byte.
simp::Scheme g_integrator_scheme = simp::leapfrog();
IntegratorAdaptive g_integrator_adaptive = IntegratorAdaptive::NONE;
bool g_integrator_mts = false;
int g_mts_substeps = 4;
std::string g_integrator_name = "leapfrog";

void set_integrator_scheme(const std::string& name, int mts_substeps) {
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
    g_integrator_scheme = simp::three_stage_adaptive(2.0);  // three-stage, default band
  } else if (name == "mts") {
    // Multiple-time-stepping: the RESPA leaf splits prior (fast) from
    // likelihood (slow). The inner substeps use a leapfrog structure; the
    // scheme field is unused by the MTS leaf but kept valid.
    g_integrator_mts = true;
    g_mts_substeps = std::max(1, mts_substeps);
    g_integrator_scheme = simp::leapfrog();
  } else {
    // Validate/resolve before mutating any state (throws on unknown name).
    simp::Scheme resolved = simp::scheme_by_name(name);
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
// so the caller can restore it. Throws (propagated to R) on an unknown name.
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
