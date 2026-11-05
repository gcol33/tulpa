#include <RcppEigen.h>
#include "laplace_newton.h"

// [[Rcpp::export]]
Rcpp::List cpp_subspace_density_work(int n_iter) {
    int value_calls = 0;
    int weight_calls = 0;
    auto log_lik = [&](const Rcpp::NumericVector& eta,
                       bool need_weights = true) {
        if (need_weights) ++weight_calls;
        else ++value_calls;
        return -0.5 * (eta[0] * eta[0] + eta[1] * eta[1]);
    };
    auto compute_eta = [](const Rcpp::NumericVector& x,
                           Rcpp::NumericVector& eta) {
        std::copy(x.begin(), x.end(), eta.begin());
    };
    auto scatter = [](const Rcpp::NumericVector& x,
                       const Rcpp::NumericVector&,
                       tulpa::DenseVec& grad, tulpa::DenseMat& H) {
        for (int j = 0; j < 2; ++j) {
            grad[j] = -x[j];
            H[j][j] = 1.0;
        }
    };
    auto center = [](Rcpp::NumericVector&) {};
    auto prior = [](const Rcpp::NumericVector&,
                     const Rcpp::NumericVector&) { return 0.0; };
    tulpa::NewtonScratch scratch;
    scratch.allocate(2, 2);
    tulpa::SubspaceDebiasOptions opts;
    opts.idx = {0};
    opts.n_warmup = 10;
    opts.n_iter = n_iter;
    const auto fit = tulpa::laplace_newton_solve_ll(
        2, 2, 50, 1e-10, compute_eta, scatter, center, prior, log_lik,
        scratch, std::vector<double>{0.0, 0.0}, nullptr, false, nullptr,
        -1, nullptr, false, nullptr, nullptr, &opts);
    return Rcpp::List::create(
        Rcpp::Named("value_calls") = value_calls,
        Rcpp::Named("weight_calls") = weight_calls,
        Rcpp::Named("converged") = fit.converged,
        Rcpp::Named("n_kept") = fit.debias_n_kept);
}
