#include <Rcpp.h>
#include "hmc_sampler.h"
#include "tulpa/nuts_api.h"

// [[Rcpp::export]]
int cpp_test_gradient_mode(SEXP name, bool c_abi = false) {
    const char* mode = Rf_isNull(name) ? nullptr : CHAR(Rf_asChar(name));
    if (c_abi) return tulpa::set_gradient_mode_str(mode);
    return static_cast<int>(tulpa_hmc::parse_gradient_mode(mode ? mode : "<null>"));
}
