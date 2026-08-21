// multinomial_logit_export.cpp
// Internal Rcpp entry exposing the baseline-category multinomial logit kernel
// for unit / FD-gradient tests. Not user-facing; the family
// backs tulpaObs's categorical positive arm (occu_categorical / a categorical
// hurdle). `eta` is length K-1 (the non-baseline class predictors); `cls` is the
// observed class, 1-based in 1..K.

#include "multinomial_logit.h"
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::List cpp_multinomial_logit_terms(Rcpp::NumericVector eta, int cls) {
    const int Km1 = eta.size();
    if (Km1 < 1) {
        Rcpp::stop("cpp_multinomial_logit_terms: eta must hold the K-1 "
                   "non-baseline predictors, so it needs at least one entry.");
    }
    if (!tulpa::multinomial_class_valid(Km1, cls)) {
        Rcpp::stop("cpp_multinomial_logit_terms: cls is the observed class, "
                   "1-based in 1..%d (got %d).", Km1 + 1, cls);
    }
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
