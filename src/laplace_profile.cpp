// laplace_profile.cpp
// R-facing entry points for the per-thread Laplace phase accumulator.
// See laplace_profile.h for the underlying mechanism.

#include "laplace_profile.h"
#include <Rcpp.h>

// Phase names keyed by PhaseIdx in laplace_profile.h. Keep in sync.
static const char* const kPhaseNames[] = {
    "pattern_build",    // PHASE_PATTERN_BUILD
    "prep",             // PHASE_PREP
    "eta",              // PHASE_ETA
    "scatter",          // PHASE_SCATTER
    "analyze",          // PHASE_ANALYZE
    "factorize",        // PHASE_FACTORIZE
    "solve",            // PHASE_SOLVE
    "line_search",      // PHASE_LINE_SEARCH
    "log_det",          // PHASE_LOG_DET
    "log_lik_prior"     // PHASE_LOG_LIK_PRIOR
};
static_assert(
    sizeof(kPhaseNames) / sizeof(kPhaseNames[0]) == tulpa::PHASE_COUNT,
    "kPhaseNames must match PhaseIdx ordering in laplace_profile.h"
);

// [[Rcpp::export]]
void cpp_profile_reset() {
    tulpa::global_phase_accumulator().reset();
}

// [[Rcpp::export]]
Rcpp::List cpp_profile_read() {
    const auto& acc = tulpa::global_phase_accumulator();
    Rcpp::NumericVector us(tulpa::PHASE_COUNT);
    Rcpp::IntegerVector ns(tulpa::PHASE_COUNT);
    Rcpp::CharacterVector names(tulpa::PHASE_COUNT);
    for (int i = 0; i < tulpa::PHASE_COUNT; ++i) {
        us[i] = acc.us[i];
        ns[i] = static_cast<int>(acc.n[i]);
        names[i] = kPhaseNames[i];
    }
    us.attr("names") = names;
    ns.attr("names") = names;
    return Rcpp::List::create(
        Rcpp::Named("us")    = us,
        Rcpp::Named("calls") = ns,
        Rcpp::Named("names") = names
    );
}
