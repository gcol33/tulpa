// cell_coupling_registry.h
// Process-global registry of compiled `CellCouplingSpec` implementations
// for the joint nested-Laplace path.
//
// Consumer packages (e.g. tulpaObs) compile a `CellCouplingSpec` subclass
// in their own src/, register it under a string name, and reference it
// from R via `cell_coupling = "<name>"` on the joint-fit call. The
// registry is the lookup table the joint driver consults to resolve the
// name to a shared_ptr the inner Newton drives.
//
// The separable default (`make_separable_cell_coupling()`) is pre-
// registered under `"separable"` the first time the registry is touched,
// so a fit with no explicit `cell_coupling` argument resolves to the
// arm-separable per-obs path.
//
// ============================================================================
// Header-only singleton
// ============================================================================
//
// The registry is a Meyers' singleton (`registry()` returns a reference to
// a function-local static `RegistryImpl`). Header-only is the right shape
// while no `R_RegisterCCallable` shim is exposed: the only writers and
// readers live in the same DLL (tulpa itself), so a single TU's static
// storage is unambiguous and there is no need for an additional `.cpp` /
// `R_init_tulpa` hook.
//
// When consumer-package registration over `R_GetCCallable` is added, the
// singleton splits into a `.cpp` (mirroring `src/tgmrf_registry.cpp`)
// owning the one cross-DLL definition and the
// `R_RegisterCCallable("tulpa", "tulpa_register_cell_coupling", ...)`
// wrapper. The header-side API (`register_cell_coupling`,
// `lookup_cell_coupling`) stays the same across that split.
//
// ============================================================================
// Consumer-side registration pattern
// ============================================================================
//
// Mirrors `<tulpa/tgmrf.h>`: a consumer package, from `R_init_<pkg>` (or
// a static initializer), retrieves a function pointer to the registered
// callable and forwards its compiled spec:
//
//   auto fp = (tulpa::RegisterCellCouplingFn) R_GetCCallable(
//       "tulpa", "tulpa_register_cell_coupling");
//   if (fp) fp("occu_cover_lognormal",
//             std::make_shared<OccuCoverLognormalCoupling>(per_cell_meta));
//
// The R-side joint-fit call then passes `cell_coupling = "occu_cover_lognormal"`
// and the joint driver resolves the spec via
// `lookup_cell_coupling("occu_cover_lognormal")`.

#ifndef TULPA_CELL_COUPLING_REGISTRY_H
#define TULPA_CELL_COUPLING_REGISTRY_H

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "tulpa/cell_coupling.h"
#include "cell_coupling_separable.h"

namespace tulpa {

// ----------------------------------------------------------------------------
// Registered-callable signature for the upcoming
// `R_RegisterCCallable("tulpa", "tulpa_register_cell_coupling", ...)` shim.
// Declared here so consumer headers can match the type when looking up the
// callable via `R_GetCCallable`. The wrapper is added in the integration
// commit (see `src/cell_coupling_registry.cpp` in that commit); the
// signature is fixed now to lock the consumer-side ABI.
// ----------------------------------------------------------------------------
using RegisterCellCouplingFn = void (*)(const char* name,
                                        std::shared_ptr<CellCouplingSpec> spec);

namespace cell_coupling_detail {

struct RegistryImpl {
    std::mutex mtx;
    std::unordered_map<std::string,
                       std::shared_ptr<CellCouplingSpec>> specs;
};

inline RegistryImpl& registry() {
    static RegistryImpl r;
    return r;
}

// Pre-register the canonical separable default under the name "separable".
// Done once, lazily, the first time the registry is touched -- using a
// function-local static guard so the registration cannot race with a
// concurrent first lookup.
inline void ensure_defaults_registered() {
    static const bool inited = []() {
        auto& r = registry();
        std::lock_guard<std::mutex> lock(r.mtx);
        r.specs.emplace(std::string("separable"),
                        make_separable_cell_coupling());
        return true;
    }();
    (void)inited;
}

} // namespace cell_coupling_detail

// ----------------------------------------------------------------------------
// Register `spec` under `name`. Replaces any previous entry under that name
// (matches `tgmrf_registry::register_spec` semantics: last-writer-wins, so a
// consumer can override a default by registering under the same name -- the
// integration commit will likely add a "name already taken" warning at the
// registered-callable shim layer).
//
// Thread-safety: serialized via the registry mutex. Safe to call from any
// thread, but the production call site is `R_init_<pkg>` on the main R
// thread, so contention is essentially nil.
// ----------------------------------------------------------------------------
inline void register_cell_coupling(const std::string& name,
                                   std::shared_ptr<CellCouplingSpec> spec) {
    cell_coupling_detail::ensure_defaults_registered();
    auto& r = cell_coupling_detail::registry();
    std::lock_guard<std::mutex> lock(r.mtx);
    r.specs[name] = std::move(spec);
}

// ----------------------------------------------------------------------------
// Look up the spec registered under `name`. Returns an empty shared_ptr if
// no such spec exists (the joint driver's resolver will surface this as a
// user-facing R error pointing at the available names).
//
// The returned shared_ptr keeps the spec (and any backing storage it
// captured at construction time) alive across the caller's use of it, even
// if the registry is mutated concurrently.
// ----------------------------------------------------------------------------
inline std::shared_ptr<CellCouplingSpec>
lookup_cell_coupling(const std::string& name) {
    cell_coupling_detail::ensure_defaults_registered();
    auto& r = cell_coupling_detail::registry();
    std::lock_guard<std::mutex> lock(r.mtx);
    auto it = r.specs.find(name);
    if (it == r.specs.end()) return std::shared_ptr<CellCouplingSpec>();
    return it->second;
}

// ----------------------------------------------------------------------------
// Diagnostics helper -- returns the current number of registered specs
// (always >= 1 because the separable default is auto-registered on first
// touch). Used by the integration commit's R-side smoke test and by the
// upcoming `cpp_cell_coupling_registry_size()` Rcpp export.
// ----------------------------------------------------------------------------
inline std::size_t cell_coupling_registry_size() {
    cell_coupling_detail::ensure_defaults_registered();
    auto& r = cell_coupling_detail::registry();
    std::lock_guard<std::mutex> lock(r.mtx);
    return r.specs.size();
}

} // namespace tulpa

#endif // TULPA_CELL_COUPLING_REGISTRY_H
