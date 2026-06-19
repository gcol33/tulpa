// multinomial_logit_export.cpp
// Internal Rcpp entry exposing the baseline-category multinomial logit kernel
// (gcol33/tulpaObs#106) for unit / FD-gradient tests. Not user-facing; the family
// backs tulpaObs's categorical positive arm (occu_categorical / a categorical
// hurdle). `eta` is length K-1 (the non-baseline class predictors); `cls` is the
// observed class, 1-based in 1..K.

#include "multinomial_logit.h"
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::List cpp_multinomial_logit_terms(Rcpp::NumericVector eta, int cls) {
    const int Km1 = eta.size();
    Rcpp::NumericVector grad(Km1);
    Rcpp::NumericMatrix neg_hess(Km1, Km1);
    const double ll = tulpa::multinomial_logit_ll(eta.begin(), Km1, cls);
    std::vector<double> nh((size_t)Km1 * Km1);
    tulpa::multinomial_logit_grad_hess(eta.begin(), Km1, cls, grad.begin(),
                                       nh.data());
    for (int j = 0; j < Km1; j++)
        for (int l = 0; l < Km1; l++)
            neg_hess(j, l) = nh[(size_t)j * Km1 + l];   // row-major -> matrix
    return Rcpp::List::create(
        Rcpp::Named("ll")       = ll,
        Rcpp::Named("grad")     = grad,
        Rcpp::Named("neg_hess") = neg_hess);
}
