// Registers the test CellCouplingSpec subclasses in the loaded tulpa's
// registry under the names the coupling tests fit with
// (`cell_coupling = "test_separable_bernoulli"`, ...). Registration goes
// through the `tulpa_register_cell_coupling` C callable, the door a consumer
// package's R_init_<pkg> uses, so a front-door fit finds the spec in the
// registry it reads.

#include "cell_coupling_separable_bernoulli.h"
#include "cell_coupling_bivariate_gaussian.h"
#include "cell_coupling_occupancy_mixture.h"
#include "cell_coupling_weighted_gaussian.h"

#include <Rcpp.h>
#include <R_ext/Rdynload.h>
#include <memory>
#include <string>

namespace {

void register_coupling(const std::string& name,
                       std::shared_ptr<tulpa::CellCouplingSpec> spec) {
    static tulpa::RegisterCellCouplingFn fn = nullptr;
    if (fn == nullptr) {
        fn = reinterpret_cast<tulpa::RegisterCellCouplingFn>(
            R_GetCCallable("tulpa", "tulpa_register_cell_coupling"));
    }
    fn(name.c_str(), std::move(spec));
}

}  // namespace

// [[Rcpp::export]]
void cpp_register_test_separable_bernoulli_coupling() {
    register_coupling(
        "test_separable_bernoulli",
        std::make_shared<tulpa::TestSeparableBernoulliCoupling>()
    );
}

// [[Rcpp::export]]
void cpp_register_test_bivariate_gaussian_coupling(double lam00,
                                                   double lam11,
                                                   double lam01) {
    register_coupling(
        "test_bivariate_gaussian",
        std::make_shared<tulpa::TestBivariateGaussianCoupling>(
            lam00, lam11, lam01)
    );
}

// [[Rcpp::export]]
void cpp_register_test_occupancy_mixture_coupling() {
    register_coupling(
        "test_occupancy_mixture",
        std::make_shared<tulpa::TestOccupancyMixtureCoupling>()
    );
}

// `arm_ids` selects which arms the spec couples, so a fixture can couple two
// at once -- the arrangement that makes the batched dispersion table's
// [arm x species] layout arbitrable.
// [[Rcpp::export]]
void cpp_register_test_weighted_gaussian_coupling(Rcpp::IntegerVector arm_ids) {
    register_coupling(
        "test_weighted_gaussian",
        std::make_shared<tulpa::TestWeightedGaussianCoupling>(
            std::vector<int>(arm_ids.begin(), arm_ids.end()))
    );
}
