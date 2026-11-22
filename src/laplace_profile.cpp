// laplace_profile.cpp
// R-facing entry points for the phase accumulator behind tulpa_profile(),
// which is process-global, mutex-guarded and off unless switched on.
// See laplace_profile.h for the underlying mechanism.

#include "laplace_profile.h"
#include "laplace_newton_loop.h"   // line_search_backtrack (probe)
#include <Rcpp.h>
#include <vector>

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
    "log_lik_prior",    // PHASE_LOG_LIK_PRIOR
    "hessian_extract",  // PHASE_HESSIAN_EXTRACT
    "inner_diagnostics",// PHASE_INNER_DIAG
    "gradient",         // PHASE_GRADIENT
    "outer_grid_cell",  // PHASE_OUTER_CELL
    "nuts_warmup",      // PHASE_NUTS_WARMUP
    "nuts_sampling"     // PHASE_NUTS_SAMPLING
};
static_assert(
    sizeof(kPhaseNames) / sizeof(kPhaseNames[0]) == tulpa::PHASE_COUNT,
    "kPhaseNames must match PhaseIdx ordering in laplace_profile.h"
);

// Switch phase timing on or off; returns the previous state so a caller can
// restore it. Called from the R main thread only.
// [[Rcpp::export]]
bool cpp_profile_enable(bool on) {
    return tulpa::phase_profiling_enabled().exchange(on);
}

// [[Rcpp::export]]
void cpp_profile_reset() {
    std::lock_guard<std::mutex> guard(tulpa::phase_mutex());
    tulpa::global_phase_accumulator().reset();
}

// [[Rcpp::export]]
Rcpp::List cpp_profile_read() {
    std::lock_guard<std::mutex> guard(tulpa::phase_mutex());
    const auto& acc = tulpa::global_phase_accumulator();
    Rcpp::NumericVector us(tulpa::PHASE_COUNT);
    Rcpp::IntegerVector ns(tulpa::PHASE_COUNT);
    Rcpp::CharacterVector names(tulpa::PHASE_COUNT);
    Rcpp::LogicalVector enclosing(tulpa::PHASE_COUNT);
    for (int i = 0; i < tulpa::PHASE_COUNT; ++i) {
        us[i] = acc.us[i];
        ns[i] = static_cast<int>(acc.n[i]);
        names[i] = kPhaseNames[i];
        enclosing[i] = tulpa::phase_is_enclosing(i);
    }
    us.attr("names") = names;
    ns.attr("names") = names;
    return Rcpp::List::create(
        Rcpp::Named("us")    = us,
        Rcpp::Named("calls") = ns,
        Rcpp::Named("names") = names,
        Rcpp::Named("enclosing") = enclosing
    );
}
