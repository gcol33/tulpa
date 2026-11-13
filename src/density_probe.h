#ifndef TULPA_DENSITY_PROBE_H
#define TULPA_DENSITY_PROBE_H

#include "autodiff_utils.h"

template<class Fn>
inline Rcpp::NumericVector density_callback_terms(double eta, double log_phi, const Fn& fn) {
    const auto e = fn(fwd::Dual(eta, 1), fwd::Dual(log_phi, 0));
    const auto p = fn(fwd::Dual(eta, 0), fwd::Dual(log_phi, 1));
    tulpa::arena::ArenaScope scope;
    tulpa::arena::Var ae(eta), ap(log_phi);
    auto a = fn(ae, ap);
    a.backward();
    return Rcpp::NumericVector::create(e.val, e.grad, p.grad,
        a.val(), ae.adj(), ap.adj());
}

#endif
