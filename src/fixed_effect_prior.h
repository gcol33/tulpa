// fixed_effect_prior.h
// The fixed-effect prior SD a compiled entry receives from R.

#ifndef TULPA_FIXED_EFFECT_PRIOR_H
#define TULPA_FIXED_EFFECT_PRIOR_H

#include <Rcpp.h>
#include "tulpa/model_data.h"

namespace tulpa {

// NULL is the engine default, DEFAULT_SIGMA_BETA (tulpa/model_data.h); anything
// else must be one positive number. An entry takes the SD as a nullable rather
// than as a defaulted double so its R signature carries no restated copy of the
// default.
inline double read_sigma_beta(Rcpp::Nullable<Rcpp::NumericVector> sigma_beta,
                              const char* arg = "sigma_beta") {
    if (sigma_beta.isNull()) return DEFAULT_SIGMA_BETA;
    Rcpp::NumericVector s(sigma_beta);
    if (s.size() != 1 || ISNAN(s[0]) || !(s[0] > 0.0)) {
        Rcpp::stop("`%s` must be NULL or one positive number.", arg);
    }
    return s[0];
}

}  // namespace tulpa

#endif  // TULPA_FIXED_EFFECT_PRIOR_H
