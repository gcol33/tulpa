// nested_likelihood.h
// Model-supplied likelihood for the nested-Laplace outer-grid driver.
//
// tulpa's built-in families reach the nested grid through builtin_family_spec
// (src/laplace_builtin_family_spec.h). A model package (tulpaObs occupancy,
// tulpaGlmm custom families) instead supplies its own single-process
// LikelihoodSpec. Because the nested grid is driven from R
// (tulpa_nested_laplace), the spec must cross the R boundary: the model package
// builds this bundle in its own C++, wraps it in an
// Rcpp::XPtr<NestedLikelihood>, and passes it as the `likelihood` argument.
// tulpa derefs the XPtr, reads `spec` + `response_data`, and routes the inner
// solve's per-observation score / Fisher weight / log-likelihood through the
// spec instead of the family enum. This is the nested analogue of the
// conditional-Laplace spec path (laplace_spec_api.h), adapted for the
// R-initiated call.
//
// Ownership. The model package allocates one NestedLikelihood on the heap and
// returns Rcpp::XPtr<NestedLikelihood>(p, /*finalize=*/true). The XPtr
// finalizer runs ~NestedLikelihood at garbage collection, dropping
// `keepalive`. The model package parks ALL backing storage there -- the
// LikelihoodSpec object `spec` points at and the per-observation response
// arrays the spec callbacks read through `response_data` -- so both pointers
// stay valid for exactly the lifetime of the bundle. `response_data` is opaque
// to tulpa; only the spec callbacks know its layout.
//
// Contract. The spec MUST be single-process (n_processes == 1): the nested
// driver fits one linear predictor per observation, so eta_weights_fn sees
// scalar grad_eta[0] / neg_hess_eta[0] and ll_double sees eta[0]. As with the
// built-in families, eta_weights_fn must return the expected (Fisher)
// information in neg_hess_eta[0] (not the AD-observed Hessian) so the Newton
// Hessian stays positive-definite on non-canonical links.

#ifndef TULPA_NESTED_LIKELIHOOD_H
#define TULPA_NESTED_LIKELIHOOD_H

#include "likelihood.h"
#include <memory>

namespace tulpa {

struct NestedLikelihood {
    // Single-process likelihood; eta_weights_fn + ll_double must be non-null.
    // Points into `keepalive`.
    const LikelihoodSpec* spec = nullptr;
    // Opaque per-observation response (e.g. {y, det_prob} for occupancy),
    // passed to the spec callbacks as model_response_data. Non-const to match
    // ModelData::model_response_data (the model package owns it; the spec
    // callbacks receive it as `const void*`). Points into `keepalive`.
    void* response_data = nullptr;
    // Owns the spec object and the response storage so both outlive the fit.
    std::shared_ptr<void> keepalive;
};

} // namespace tulpa

#endif // TULPA_NESTED_LIKELIHOOD_H
