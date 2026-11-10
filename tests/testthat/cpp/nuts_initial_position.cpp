#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <vector>

#include "hmc_sampler.h"
#include "tulpa/likelihood.h"
#include "tulpa/nuts_api.h"

namespace {

struct InitialTarget {
    int fault;
    bool origin_only;
    double bad_value;
};

double initial_likelihood(int i, const double*, const double&, const double&,
                          const std::vector<double>& params,
                          const tulpa::ModelData&, const tulpa::ParamLayout&,
                          const void* response) {
    const auto& target = *static_cast<const InitialTarget*>(response);
    bool at_origin = true;
    double largest = 0.0;
    for (double value : params) {
        at_origin &= value == 0.0;
        largest = std::max(largest, std::abs(value));
    }
    const bool invalid =
        (target.fault == 1 && (!target.origin_only || !at_origin)) ||
        (target.fault == 4 && !at_origin) ||
        (target.fault == 5 && largest > 1e-4) ||
        (target.fault == 7 && largest > 0.0 && largest < 1e-4);
    if (invalid) return target.bad_value;
    return -0.5 * params[i] * params[i];
}

void initial_gradient(const std::vector<double>& params,
                      const tulpa::ModelData& data, const tulpa::ParamLayout&,
                      std::vector<double>& gradient, double* lp) {
    const auto& target = *static_cast<const InitialTarget*>(data.model_response_data);
    gradient.resize(params.size());
    double value = 0.0;
    bool at_origin = true;
    for (size_t j = 0; j < params.size(); ++j) {
        gradient[j] = -params[j] * (target.fault >= 6 ? 2.0 : 1.0);
        value -= 0.5 * params[j] * params[j];
        at_origin &= params[j] == 0.0;
    }
    if (target.fault == 2 && (!target.origin_only || !at_origin))
        gradient.back() = target.bad_value;
    if (target.fault == 3 && (!target.origin_only || !at_origin))
        value = target.bad_value;
    if (lp) *lp = value;
}

} // namespace

// [[Rcpp::export]]
Rcpp::List cpp_test_nuts_initial_position(
    Rcpp::NumericMatrix init, int fault = 0, bool origin_only = false,
    bool numerical = false, int L = 0, bool walnuts = false,
    int n_warmup = 0, double epsilon = 0.0, bool direct = false,
    double bad_value = NA_REAL) {
    InitialTarget target{fault, origin_only, bad_value};
    tulpa::LikelihoodSpec spec;
    spec.n_processes = 1;
    spec.n_extra_params = init.ncol();
    spec.ll_double = initial_likelihood;
    if (!numerical) spec.gradient_fn = initial_gradient;
    tulpa::ModelData data;
    data.N = init.ncol();
    data.n_processes = 1;
    data.processes.resize(1);
    data.processes[0].p = 0;
    data.sharing.init(1);
    data.likelihood_spec = &spec;
    data.model_response_data = &target;
    const auto layout = tulpa_hmc::compute_param_layout(data);
    const int chains = init.nrow();
    const int p = init.ncol();
    if (chains < 1 || p < 1) Rcpp::stop("init must contain a position");
    std::vector<double> positions(static_cast<size_t>(chains) * p);
    for (int c = 0; c < chains; ++c)
        for (int j = 0; j < p; ++j) positions[c * p + j] = init(c, j);
    if (direct) {
        tulpa_hmc::GradientModeFitScope scope;
        return Rcpp::List::create(Rcpp::Named("verified") =
            tulpa_hmc::verify_gradient_runtime(positions, data, layout));
    }
    tulpa::SamplerControl control;
    control.L = L;
    control.walnuts = walnuts;
    std::vector<tulpa::NUTSResult> results(chains);
    if (chains == 1) {
        tulpa::get_nuts_fn()(&data, &layout, positions.data(), p,
            n_warmup + 20, n_warmup, 6, 0.8, 17, 0, &control,
            nullptr, epsilon, results.data());
    } else {
        std::vector<double> steps(chains, epsilon);
        tulpa::get_nuts_chains_fn()(&data, &layout, positions.data(), p, chains,
            n_warmup + 20, n_warmup, 6, 0.8, 17, 0, &control,
            nullptr, steps.data(), results.data());
    }
    Rcpp::List draws(chains), log_prob(chains);
    Rcpp::NumericVector steps(chains);
    Rcpp::CharacterVector samplers(chains);
    for (int c = 0; c < chains; ++c) {
        auto& result = results[c];
        Rcpp::NumericMatrix samples(result.n_sample, p);
        for (int i = 0; i < result.n_sample; ++i)
            for (int j = 0; j < p; ++j)
                samples(i, j) = result.samples[i * p + j];
        draws[c] = samples;
        log_prob[c] = Rcpp::NumericVector(result.log_prob,
                                         result.log_prob + result.n_sample);
        steps[c] = result.epsilon;
        samplers[c] = result.sampler;
        result.free_buffers();
    }
    return Rcpp::List::create(Rcpp::Named("draws") = draws,
        Rcpp::Named("log_prob") = log_prob, Rcpp::Named("epsilon") = steps,
        Rcpp::Named("sampler") = samplers);
}
