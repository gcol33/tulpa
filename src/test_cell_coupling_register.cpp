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
#include "test_cell_coupling_bivariate_gaussian.h"

#include <Rcpp.h>
#include <memory>

// [[Rcpp::export]]
void cpp_register_test_separable_bernoulli_coupling() {
    tulpa::register_cell_coupling(
        "test_separable_bernoulli",
        std::make_shared<tulpa::TestSeparableBernoulliCoupling>()
    );
}

// [[Rcpp::export]]
void cpp_register_test_bivariate_gaussian_coupling(double lam00,
                                                   double lam11,
                                                   double lam01) {
    tulpa::register_cell_coupling(
        "test_bivariate_gaussian",
        std::make_shared<tulpa::TestBivariateGaussianCoupling>(
            lam00, lam11, lam01)
    );
}
