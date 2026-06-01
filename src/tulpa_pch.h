#ifndef TULPA_PCH_H
#define TULPA_PCH_H

// Precompiled-header payload: the Rcpp + Eigen headers every translation unit
// parses. Compiled once to tulpa_pch.h.gch and force-included into each TU so
// the ~1.3s RcppEigen parse is paid once per build instead of once per file.
#include <RcppEigen.h>

#endif  // TULPA_PCH_H
