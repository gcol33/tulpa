// test_cell_coupling_register.cpp
// Test-only Rcpp export that registers `TestSeparableBernoulliCoupling`
// under the name "test_separable_bernoulli" in tulpa's CellCouplingSpec
// registry. The test-cell-coupling-recovery.R test calls it once at
// suite start and then fits with `cell_coupling = "test_separable_bernoulli"`.
//
// This file is the in-package analogue of the
// `tulpa_register_cell_coupling` C callable a consumer DLL would use; it
// stays under src/ rather than inst/include/ because no other package
// should depend on the test spec.

#include "cell_coupling_registry.h"
#include "test_cell_coupling_separable_bernoulli.h"

#include <Rcpp.h>
#include <memory>

// [[Rcpp::export]]
void cpp_register_test_separable_bernoulli_coupling() {
    tulpa::register_cell_coupling(
        "test_separable_bernoulli",
        std::make_shared<tulpa::TestSeparableBernoulliCoupling>()
    );
}
