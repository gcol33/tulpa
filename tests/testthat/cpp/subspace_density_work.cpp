#include <RcppEigen.h>
#include "laplace_newton.h"
#include "nested_laplace_checkpoint.h"

// [[Rcpp::export]]
bool cpp_subspace_checkpoint_roundtrip() {
    tulpa::LaplaceResult a, b;
    a.debias_idx = {0, 1};
    a.debias_draws = {0.1, 0.2, 0.3, 0.4};
    a.debias_n_kept = 2;
    a.debias_importance = {0.2, 20.0, 256.0, 0.1, 4096.0, 0.3, 0.7};
    a.debias_importance_log_weights = {0.2, 0.3, 0.4};
    const auto buffer = tulpa::ckpt_serialize(a);
    tulpa::CkptReader reader(buffer.data(), buffer.size());
    return tulpa::ckpt_deserialize(reader, b) && buffer == tulpa::ckpt_serialize(b);
}

// [[Rcpp::export]]
double cpp_subspace_log_accept(double log_target_new, double log_target_old,
                               double residual_quad_new, double residual_quad_old) {
    return tulpa::subspace_log_accept(log_target_new, log_target_old,
                                      residual_quad_new, residual_quad_old);
}

// [[Rcpp::export]]
double cpp_subspace_conditional_scale() {
    tulpa::DenseCholeskyScratch chol;
    chol.ensure(2);
    chol.L = {1.0, 0.0, 0.0, 10.0};
    tulpa::SparseCholeskySolver sparse;
    tulpa::SubspaceDebiasOptions opts;
    opts.idx = {0};
    opts.exact = true;
    opts.n_warmup = 2000;
    opts.n_iter = 100;
    Rcpp::NumericVector x(2);
    const auto log_joint = [](const Rcpp::NumericVector& value) {
        return -0.5 * (value[0] * value[0] + value[1] * value[1]);
    };
    const auto out = tulpa::compute_subspace_debias(
        2, std::vector<double>{0.0, 0.0}, chol, sparse, false,
        log_joint, x, opts);
    return out.scale;
}

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
