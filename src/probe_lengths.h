// probe_lengths.h -- the length contract of the vectorized family probes.

#ifndef TULPA_PROBE_LENGTHS_H
#define TULPA_PROBE_LENGTHS_H

#include <Rcpp.h>
#include "tulpa_stop.h"

// Every vectorized family probe reads y, n_trials and (where it takes one)
// logit_zi to the length of eta, and Rcpp's operator[] does not bounds-check,
// so a short vector is a read past the end of an R vector rather than an error.
// n_trials recycles from length 1 and must otherwise match. One helper, called
// from all of them: the check was written out per function, and the one that
// skipped it returned whatever was next in memory.
inline void check_probe_lengths(const char* fn, R_xlen_t n,
                                       const Rcpp::NumericVector& y,
                                       const Rcpp::IntegerVector& n_trials,
                                       const Rcpp::NumericVector* logit_zi) {
  if (y.size() != n) {
    tulpa::stop("%s: y (%d) and eta (%d) differ in length.",
               {fn, (int)y.size(), (int)n});
  }
  if (logit_zi && logit_zi->size() != n) {
    tulpa::stop("%s: logit_zi (%d) and eta (%d) differ in length.",
               {fn, (int)logit_zi->size(), (int)n});
  }
  if (n_trials.size() != 1 && n_trials.size() != n) {
    tulpa::stop("%s: n_trials must be length 1 or %d (got %d).",
               {fn, (int)n, (int)n_trials.size()});
  }
}

#endif  // TULPA_PROBE_LENGTHS_H
