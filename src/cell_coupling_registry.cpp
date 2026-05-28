// cell_coupling_registry.cpp
// Process-global registry of compiled `CellCouplingSpec` implementations
// and the `tulpa_register_cell_coupling` registered C callable that user
// DLLs use to insert specs at load time.
//
// Wired from src/tulpa_init.cpp::tulpa_register_callables() so the existing
// R_init_tulpa hook picks it up at DLL load. Mirrors tgmrf_registry.cpp.

#include "cell_coupling_registry.h"
#include "cell_coupling_separable.h"

#include <Rcpp.h>
#include <R_ext/Rdynload.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tulpa {

namespace {

// Singleton container. Single definition in this TU so the registry is
// unambiguous across every translation unit linked into tulpa.so.
struct RegistryImpl {
    std::mutex mtx;
    std::unordered_map<std::string,
                       std::shared_ptr<CellCouplingSpec>> specs;
};

RegistryImpl& registry() {
    static RegistryImpl r;
    return r;
}

// Pre-register the canonical separable default under the name "separable"
// the first time the registry is touched. Function-local static guards
// against the registration racing a concurrent first lookup.
void ensure_defaults_registered() {
    static const bool inited = []() {
        auto& r = registry();
        std::lock_guard<std::mutex> lock(r.mtx);
        r.specs.emplace(std::string("separable"),
                        make_separable_cell_coupling());
        return true;
    }();
    (void)inited;
}

} // namespace

void register_cell_coupling(const std::string& name,
                            std::shared_ptr<CellCouplingSpec> spec) {
    ensure_defaults_registered();
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.mtx);
    r.specs[name] = std::move(spec);
}

std::shared_ptr<CellCouplingSpec> lookup_cell_coupling(const std::string& name) {
    ensure_defaults_registered();
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.mtx);
    auto it = r.specs.find(name);
    if (it == r.specs.end()) return std::shared_ptr<CellCouplingSpec>();
    return it->second;
}

std::size_t cell_coupling_registry_size() {
    ensure_defaults_registered();
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.mtx);
    return r.specs.size();
}

bool cell_coupling_registry_has(const std::string& name) {
    ensure_defaults_registered();
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.mtx);
    return r.specs.find(name) != r.specs.end();
}

} // namespace tulpa

// ============================================================================
// Registered C callable wrapper. Signature matches
// tulpa::RegisterCellCouplingFn from <tulpa/cell_coupling.h>; that is the
// type consumer DLLs cast their `R_GetCCallable("tulpa", "tulpa_register_
// cell_coupling")` result to.
// ============================================================================

static void tulpa_register_cell_coupling_impl(
    const char* name,
    std::shared_ptr<tulpa::CellCouplingSpec> spec
) {
    if (name == nullptr || !spec) return;
    tulpa::register_cell_coupling(std::string(name), std::move(spec));
}

void tulpa_register_cell_coupling_callables(DllInfo* dll) {
    R_RegisterCCallable(
        "tulpa", "tulpa_register_cell_coupling",
        (DL_FUNC)&tulpa_register_cell_coupling_impl
    );
}

// ============================================================================
// Rcpp-export helpers for R-side validation of the `cell_coupling = "<name>"`
// argument and for smoke tests of the registry.
// ============================================================================

// [[Rcpp::export]]
bool cpp_cell_coupling_registry_has(std::string name) {
    return tulpa::cell_coupling_registry_has(name);
}

// [[Rcpp::export]]
int cpp_cell_coupling_registry_size() {
    return static_cast<int>(tulpa::cell_coupling_registry_size());
}
