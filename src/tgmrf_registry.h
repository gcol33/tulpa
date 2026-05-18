// tgmrf_registry.h
// Process-global registry of user-compiled tgmrf specs.
//
// Specs are inserted by user DLLs at load time via the
// `tulpa_register_tgmrf` registered C callable (see src/tgmrf_registry.cpp
// and inst/include/tulpa/tgmrf.h).
//
// The inference layers (nested_laplace_multi) look up a spec by id when
// the R-side tgmrf block carries `backend == "cpp"`. Lookup is read-only
// after registration and does not lock; insertion takes a mutex so
// multiple user DLLs loaded concurrently do not race.
//
// This header is internal to tulpa. The exported C callable signatures
// live in inst/include/tulpa/tgmrf.h.

#ifndef TULPA_SRC_TGMRF_REGISTRY_H
#define TULPA_SRC_TGMRF_REGISTRY_H

#include <string>
#include "tulpa/tgmrf.h"

namespace tulpa {
namespace tgmrf_backend {

// Insert or overwrite a spec. Thread-safe.
void register_spec(const std::string& id, const TgmrfSpec& spec);

// Look up a spec by id. Returns nullptr if no such id is registered.
// Read-only after registration; no lock taken.
const TgmrfSpec* lookup_spec(const std::string& id);

// Count of registered specs (for diagnostics / tests).
int registry_size();

} // namespace tgmrf_backend
} // namespace tulpa

#endif // TULPA_SRC_TGMRF_REGISTRY_H
