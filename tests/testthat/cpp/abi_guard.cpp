#include <Rcpp.h>
#include <tulpa/types.h>

#define TULPA_MODEL_DATA_H
namespace tulpa {
struct ModelData;
constexpr int TULPA_ABI_VERSION = -1;
}
#include <tulpa/nuts_api.h>

// [[Rcpp::export(rng = false)]]
bool cpp_test_abi_guard_mismatch() {
    return tulpa::get_nuts_fn() != nullptr;
}
