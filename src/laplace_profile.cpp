// laplace_profile.cpp
// R-facing entry points for the per-thread Laplace phase accumulator.
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
    "log_lik_prior"     // PHASE_LOG_LIK_PRIOR
};
static_assert(
    sizeof(kPhaseNames) / sizeof(kPhaseNames[0]) == tulpa::PHASE_COUNT,
    "kPhaseNames must match PhaseIdx ordering in laplace_profile.h"
);

// Line-search probe: drive the shared backtracking line search on a 1-D
// quadratic phi(a) = slope*a + c*a^2 (obj_old = 0) along delta = 1 from x = 0,
// and report the accepted step plus the number of objective evaluations. Lets
// tests exercise the interpolation backtrack directly: for an overshoot
// (c < 0 with phi(1) < 0) the line optimum is a* = -slope/(2c), and one
// safeguarded interpolation should land within [0.1, 0.5] of it in far fewer
// sweeps than fixed halving.
// [[Rcpp::export]]
Rcpp::List cpp_line_search_probe(double slope, double c) {
    Rcpp::NumericVector x(1, 0.0), x_try(1, 0.0);
    std::vector<double> delta(1, 1.0);
    double obj_old = 0.0, obj_out = 0.0;
    int n_evals = 0;
    auto eval_obj = [&](const Rcpp::NumericVector& xv) -> double {
        const double a = xv[0];
        return slope * a + c * a * a;
    };
    double step = tulpa::line_search_backtrack(
        x, delta, 1, obj_old, slope, eval_obj, obj_out, x_try, &n_evals);
    return Rcpp::List::create(
        Rcpp::Named("step")    = step,
        Rcpp::Named("obj")     = obj_out,
        Rcpp::Named("n_evals") = n_evals
    );
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
