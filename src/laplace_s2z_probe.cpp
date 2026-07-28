// laplace_s2z_probe.cpp
// R-visible view of the sum-to-zero densify cutoff.
//
// The cutoff decides how the augmentation's rank-1 11' is STORED, not what the
// fit answers -- both storages are exact and agree -- so which one a given
// TULPA_S2Z_DENSIFY_MAX setting selects is otherwise invisible from R. Exported
// so a test can assert that an unparseable setting leaves the documented
// default in place instead of silently reading as 0, which is the meaningful
// "always fold" setting.

#include "laplace_s2z.h"

// [[Rcpp::export]]
int cpp_s2z_densify_max() {
  return tulpa::s2z_densify_max();
}
