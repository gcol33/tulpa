#include <RcppEigen.h>
#include "hmc_multiscale_temporal_grad.h"

namespace {
template<class T>
T multiscale_prior_value(const std::string& structure, const std::vector<T>& p) {
    const int n = static_cast<int>(p.size()) - 2;
    std::vector<T> phi(p.begin(), p.begin() + n);
    const T sigma2 = tulpa::math::safe_exp(p[n]);
    const T rho = T(2) * tulpa::math::inv_logit(p[n + 1]) - T(1);
    T value;
    if (structure == "rw1" || structure == "cyclic")
        value = tulpa_temporal::rw1_log_lik(phi, sigma2, structure == "cyclic", true);
    else if (structure == "rw2")
        value = tulpa_temporal::rw2_log_lik(phi, sigma2, false, true);
    else if (structure == "ar1")
        value = tulpa_temporal::ar1_log_lik(phi, sigma2, rho);
    else if (structure == "iid")
        value = tulpa_temporal::iid_log_lik(phi, sigma2);
    else Rcpp::stop("Unknown multiscale structure");
    return value + tulpa_temporal::log_prior_sigma2_temporal_pc(sigma2, 1, .01) + p[n];
}
}

// [[Rcpp::export]]
Rcpp::List cpp_test_multiscale_prior(std::string structure, Rcpp::NumericVector phi,
                                   double log_sigma2, double logit_rho) {
    const int n = phi.size();
    std::vector<double> p(phi.begin(), phi.end());
    p.push_back(log_sigma2);
    p.push_back(logit_rho);
    Rcpp::NumericMatrix gradient(n + 2, 3);
    Rcpp::colnames(gradient) = Rcpp::CharacterVector::create("handcoded", "forward", "arena");
    Rcpp::NumericVector value(3);
    value[0] = multiscale_prior_value(structure, p);
    const double sigma2 = std::exp(log_sigma2);
    const double rho = 2 * tulpa::math::inv_logit(logit_rho) - 1;
    using namespace tulpa_temporal_grad;
    if (structure == "rw1" || structure == "cyclic") {
        if (structure == "cyclic") {
            rw1_cyclic_grad_phi(phi.begin(), n, sigma2, &gradient(0, 0), true);
            gradient(n, 0) = rw1_cyclic_grad_log_sigma2(phi.begin(), n, sigma2, true);
        } else {
            rw1_grad_phi(phi.begin(), n, sigma2, &gradient(0, 0), true);
            gradient(n, 0) = rw1_grad_log_sigma2(phi.begin(), n, sigma2, true);
        }
    } else if (structure == "rw2") {
        rw2_grad_phi(phi.begin(), n, sigma2, &gradient(0, 0), true);
        gradient(n, 0) = rw2_grad_log_sigma2(phi.begin(), n, sigma2, true);
    } else if (structure == "ar1") {
        ar1_grad_phi(phi.begin(), n, sigma2, rho, &gradient(0, 0));
        gradient(n, 0) = ar1_grad_log_sigma2(phi.begin(), n, sigma2, rho);
        gradient(n + 1, 0) = ar1_grad_logit_rho(phi.begin(), n, sigma2, rho);
    } else if (structure == "iid") {
        iid_grad_phi(phi.begin(), n, sigma2, &gradient(0, 0));
        gradient(n, 0) = iid_grad_log_sigma2(phi.begin(), n, sigma2);
    }
    gradient(n, 0) += tulpa::log_prior_log_sigma2_pc(fwd::Dual(log_sigma2,1),1,.01).grad;
    for (int j = 0; j < n + 2; ++j) {
        std::vector<fwd::Dual> dual;
        for (int k = 0; k < n + 2; ++k) dual.emplace_back(p[k], j == k ? 1 : 0);
        auto result = multiscale_prior_value(structure, dual);
        value[1] = result.val;
        gradient(j, 1) = result.grad;
    }
    tulpa::arena::ArenaScope scope;
    std::vector<tulpa::arena::Var> arena;
    for (double x : p) arena.emplace_back(x);
    auto result = multiscale_prior_value(structure, arena);
    result.backward();
    value[2] = result.val();
    for (int j = 0; j < n + 2; ++j) gradient(j, 2) = arena[j].adj();
    return Rcpp::List::create(Rcpp::_["value"] = value, Rcpp::_["gradient"] = gradient);
}
