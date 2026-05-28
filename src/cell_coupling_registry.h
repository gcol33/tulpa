// cell_coupling_registry.h
// Process-global registry of compiled `CellCouplingSpec` implementations
// for the joint nested-Laplace path.
//
// Consumer packages (e.g. tulpaObs) compile a `CellCouplingSpec` subclass
// in their own src/, register it under a string name from `R_init_<pkg>`
// via the `tulpa_register_cell_coupling` registered C callable (signature
// `tulpa::RegisterCellCouplingFn` in <tulpa/cell_coupling.h>), and
// reference it from R via `cell_coupling = "<name>"` on the joint-fit call.
// The registry is the lookup table the joint driver consults to resolve
// the name to a shared_ptr the inner Newton drives.
//
// The separable default (`make_separable_cell_coupling()`) is pre-registered
// under `"separable"` the first time the registry is touched, so a fit with
// no explicit `cell_coupling` argument resolves to the arm-separable per-obs
// path.
//
// ============================================================================
// Storage in .cpp, API in .h
// ============================================================================
//
// The singleton storage and mutex live in src/cell_coupling_registry.cpp so
// the one definition is shared across every translation unit linked into
// tulpa.so. The header declares only the public API (register / lookup /
// size). Mirrors src/tgmrf_registry.{h,cpp}.

#ifndef TULPA_CELL_COUPLING_REGISTRY_H
#define TULPA_CELL_COUPLING_REGISTRY_H

#include <cstddef>
#include <memory>
#include <string>

#include "tulpa/cell_coupling.h"

namespace tulpa {

// Register `spec` under `name`. Replaces any previous entry under that
// name (last-writer-wins, matching `tgmrf_backend::register_spec`).
// Thread-safe; the production call site is `R_init_<pkg>` on the main R
// thread, so contention is essentially nil.
void register_cell_coupling(const std::string& name,
                            std::shared_ptr<CellCouplingSpec> spec);

// Look up the spec registered under `name`. Returns an empty shared_ptr
// if no such spec exists; the joint driver's resolver surfaces that as
// a user-facing R error pointing at the available names.
//
// The returned shared_ptr keeps the spec (and any backing storage it
// captured at construction) alive across the caller's use of it, even
// if the registry is mutated concurrently.
std::shared_ptr<CellCouplingSpec> lookup_cell_coupling(const std::string& name);

// Diagnostics helper: returns the current number of registered specs
// (always >= 1 because the separable default is auto-registered on first
// touch). Used by R-side smoke tests and by `cpp_cell_coupling_registry_size`.
std::size_t cell_coupling_registry_size();

// True iff `name` resolves to a registered spec. Cheap O(1) check used by
// the R-side argument validation on `tulpa_nested_laplace_joint(cell_coupling = ...)`.
bool cell_coupling_registry_has(const std::string& name);

} // namespace tulpa

#endif // TULPA_CELL_COUPLING_REGISTRY_H
