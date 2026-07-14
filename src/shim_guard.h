// shim_guard.h
// Exception guard for the extern "C" C-callable shim boundary. A C++
// exception must not unwind through the C ABI into a consumer package
// compiled elsewhere; the guard converts it to an R error at the boundary.
//
// The conversion is Rf_error, i.e. a longjmp: destructors of the CALLER's
// in-flight C++ frames do not run. Consumer packages should invoke the
// shims from frames without non-trivial locals (or wrap the call site and
// re-enter R state via Rcpp's boundary macros).

#ifndef TULPA_SHIM_GUARD_H
#define TULPA_SHIM_GUARD_H

#include <exception>
#include <R_ext/Error.h>

#define TULPA_SHIM_GUARD_BEGIN try {
#define TULPA_SHIM_GUARD_END(fn_name)                                   \
    } catch (const std::exception& e) {                                 \
        Rf_error("%s: %s", fn_name, e.what());                          \
    } catch (...) {                                                     \
        Rf_error("%s: unknown C++ exception", fn_name);                 \
    }

#endif // TULPA_SHIM_GUARD_H
