// test_ccallable_registry.cpp
// Does a name the exported headers advertise actually resolve?
//
// `inst/include/tulpa/*.h` reach the engine through R_GetCCallable, and an
// unregistered name is a hard R error at the FIRST call -- so a LinkingTo
// package compiles cleanly against the header and dies at runtime
// (gcol33/tulpa#688, joint_nested_laplace_api.h, which advertised
// "tulpa_nested_laplace_joint_bym2" and was registered nowhere). The test that
// reads this scrapes every name out of the installed headers and asks here, so
// a header added without its registration fails in tulpa's own suite rather
// than in a consumer's.

#include <Rcpp.h>
#include <R_ext/Rdynload.h>

#include <string>

// R_GetCCallable errors (a longjmp) when the name is not registered, which the
// caller catches with tryCatch. Nothing with a destructor is alive across it.
// [[Rcpp::export]]
bool cpp_test_ccallable_resolves(std::string name) {
    DL_FUNC fn = R_GetCCallable("tulpa", name.c_str());
    return fn != nullptr;
}
