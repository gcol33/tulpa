// nested_laplace_test_occupancy.cpp
// Reference scaled-Bernoulli occupancy LikelihoodSpec for the nested-Laplace
// spec path. This is the in-package equivalence net for the spec-driven
// likelihood path AND the template a model package (tulpaObs) copies into its own
// C++ -- the same role cpp_laplace_spec_test_family plays for the conditional
// path.
//
// tulpa's family enum carries no occupancy `det_prob` hook: the marginalized
// single-season occupancy likelihood arrives as a model LikelihoodSpec, a
// detection indicator y_i in {0, 1} with mean
// mu_i = q_i * sigma(eta_i), where sigma(eta_i) is the occupancy probability
// and q_i in [0, 1] is the per-site probability of at least one detection given
// occupancy (the latent occupancy state integrated out analytically). This
// file builds that spec + its {y, q} response, wraps them in a
// tulpa::NestedLikelihood, and hands R an XPtr for tulpa_nested_laplace(
// likelihood = ).
//
// The Fisher working weight q*sigma*(1-sigma)^2/(1-q*sigma) is the expected
// information of the scaled (non-canonical) link: it stays positive so the
// Newton Hessian is positive-definite, and at the mode it is the marginal
// occupancy curvature, so the per-row predictive variance is calibrated with no
// rescaling. q_i = 0 (an unvisited site) contributes zero score and zero
// information -- it drops from the likelihood while keeping its latent value
// (the held-out case). q_i = 1 reduces the family to a plain logit Bernoulli.

#include "tulpa/likelihood.h"
#include "tulpa/nested_likelihood.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace {

using tulpa::LikelihoodSpec;
using tulpa::ModelData;
using tulpa::NestedLikelihood;
using tulpa::ParamLayout;

// Per-observation response the occupancy spec reads through model_response_data.
struct OccupancyResponse {
    std::vector<double> y;  // [N] detection indicator in {0, 1}
    std::vector<double> q;  // [N] per-site P(>=1 detection | occupied), in [0, 1]
};

// mu = q * sigma(eta) is held inside (eps, 1 - eps) so that log(mu) and
// log(1 - mu) stay finite. Same magnitude as tulpa::math::PROB_EPS: at 1e-15
// the log is -34.5, small enough to dominate any competing term and large
// enough to stay well inside the double range.
inline constexpr double OCC_MU_EPS = 1e-15;

// Floor on the (1 - mu) denominator of the score and the expected information.
// Looser than OCC_MU_EPS because it divides rather than being logged: at mu
// exactly 1 - OCC_MU_EPS the quotient would be 1e15, which swamps the Newton
// step, so the working weight is capped at 1e12 instead.
inline constexpr double OCC_DENOM_FLOOR = 1e-12;

inline double occ_sigma(double eta) {
    if (eta > 0) return 1.0 / (1.0 + std::exp(-eta));
    double e = std::exp(eta);
    return e / (1.0 + e);
}

// LikelihoodFn<double>: per-obs log p(y_i | eta_i) of the scaled Bernoulli.
double occ_ll_double(
    int i, const double* eta, const double& /*logit_zi*/,
    const double& /*logit_oi*/, const std::vector<double>& /*params*/,
    const ModelData& /*data*/, const ParamLayout& /*layout*/,
    const void* model_data
) {
    const auto* r = static_cast<const OccupancyResponse*>(model_data);
    const double q = r->q[i];
    if (q <= 0.0) return 0.0;
    double mu = q * occ_sigma(eta[0]);
    mu = std::max(std::min(mu, 1.0 - OCC_MU_EPS), OCC_MU_EPS);
    return r->y[i] ? std::log(mu) : std::log(1.0 - mu);
}

// EtaWeightsFn: per-obs eta-space score + expected (Fisher) information.
//   grad     = (y - mu) (1 - sigma) / (1 - mu),     mu = q sigma
//   neg_hess = q sigma (1 - sigma)^2 / (1 - mu)      [expected information]
void occ_eta_weights(
    int i, const double* eta, double /*logit_zi*/, double /*logit_oi*/,
    const std::vector<double>& /*params*/, const ModelData& /*data*/,
    const ParamLayout& /*layout*/, const void* model_data,
    double* grad_eta, double* neg_hess_eta
) {
    const auto* r = static_cast<const OccupancyResponse*>(model_data);
    const double q = r->q[i];
    if (q <= 0.0) { grad_eta[0] = 0.0; neg_hess_eta[0] = 0.0; return; }
    const double s     = occ_sigma(eta[0]);
    const double mu    = q * s;
    const double denom = std::max(1.0 - mu, OCC_DENOM_FLOOR);  // 1 - q sigma
    grad_eta[0]     = (r->y[i] - mu) * (1.0 - s) / denom;
    neg_hess_eta[0] = q * s * (1.0 - s) * (1.0 - s) / denom;
}

// Owns the spec object + response so both outlive the XPtr (parked in
// NestedLikelihood::keepalive).
struct OccupancyBundle {
    LikelihoodSpec    spec;
    OccupancyResponse resp;
};

} // namespace

// Build a model-supplied occupancy likelihood for tulpa_nested_laplace(
// likelihood = ). Returns an external pointer to a tulpa::NestedLikelihood that
// owns its spec + {y, det_prob} response; the XPtr finalizer frees both at GC.
// [[Rcpp::export]]
SEXP cpp_nested_laplace_test_occupancy_likelihood(Rcpp::NumericVector y,
                                                  Rcpp::NumericVector det_prob) {
    if (det_prob.size() != y.size()) {
        Rcpp::stop("cpp_nested_laplace_test_occupancy_likelihood: "
                   "det_prob and y must have equal length.");
    }
    auto bundle = std::make_shared<OccupancyBundle>();
    bundle->resp.y.assign(y.begin(), y.end());
    bundle->resp.q.assign(det_prob.begin(), det_prob.end());
    bundle->spec.n_processes    = 1;
    bundle->spec.name           = "occupancy_scaled_bernoulli";
    bundle->spec.ll_double      = &occ_ll_double;
    bundle->spec.eta_weights_fn = &occ_eta_weights;
    bundle->spec.n_extra_params = 0;

    auto* lk = new NestedLikelihood;
    lk->spec          = &bundle->spec;
    lk->response_data = &bundle->resp;
    lk->keepalive     = bundle;   // shared_ptr<OccupancyBundle> -> shared_ptr<void>

    return Rcpp::XPtr<NestedLikelihood>(lk, true);
}
