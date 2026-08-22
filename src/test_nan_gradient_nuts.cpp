// test_nan_gradient_nuts.cpp
// Test-only entry point: a NUTS chain whose gradient is non-finite where its
// log-posterior is not.
//
// The leaf divergence test used to inspect the position vector and the
// log-posterior and never the momentum. The default leapfrog op sequence ends
// in a Kick, so a non-finite gradient at the endpoint leaves p non-finite while
// q -- last written by the preceding Drift -- and log_prob are both finite. The
// leaf then passed as valid, the Hamiltonian it computed was NaN, the leaf's
// multinomial weight H0 - H_new was NaN, and because NaN > delta_max is false
// the iteration was reported as an ordinary acceptance with no divergence: a
// stuck draw, invisible in the output.
//
// Reproducing that needs a gradient source the model controls independently of
// its density, which LikelihoodSpec::gradient_fn is -- a model package's
// hand-coded full gradient, taking the GradientFn signature and writing the
// fused log-posterior itself. The target is a standard normal, so the clean arm
// is an ordinary well-behaved chain and the two arms differ in one planted
// entry.

#include <Rcpp.h>
#include <cmath>
#include <limits>
#include <vector>

#include "hmc_sampler.h"
#include "log_post_impl.h"
#include "tulpa/likelihood.h"
#include "tulpa/autodiff_arena.h"

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

namespace {

// Whether the hand-coded gradient plants a NaN. A file-scope flag rather than
// model_data because the gradient hook is a plain function pointer and the two
// arms are run one after the other from a single-threaded probe.
bool g_plant_nan = false;

struct NormalData { int K = 0; };

// log N(z_i | 0, 1) per "observation", carried on the extra parameters. Only
// ever evaluated for the runtime gradient check and the initial log-posterior;
// the sampler's own gradient comes from the hook below.
template <typename T>
T normal_likelihood(int i, const T* eta, const T& logit_zi, const T& logit_oi,
                    const std::vector<T>& params, const ModelData& data,
                    const ParamLayout& layout, const void* model_data) {
    (void)eta; (void)logit_zi; (void)logit_oi; (void)data; (void)model_data;
    const T z = params[layout.extra_offset + i];
    return T(-0.5) * z * z;
}

// The hand-coded full gradient. The log-posterior it writes is always finite;
// the gradient's first entry is NaN when the flag is set.
void normal_gradient(const std::vector<double>& params, const ModelData& data,
                     const ParamLayout& layout, std::vector<double>& grad,
                     double* log_post_out) {
    (void)data;
    const int n = static_cast<int>(params.size());
    grad.assign(n, 0.0);
    double lp = 0.0;
    const auto* nd = static_cast<const NormalData*>(nullptr);
    (void)nd;
    for (int j = layout.extra_offset; j < n; j++) {
        grad[j] = -params[j];
        lp += -0.5 * params[j] * params[j];
    }
    if (g_plant_nan) {
        grad[layout.extra_offset] = std::numeric_limits<double>::quiet_NaN();
    }
    if (log_post_out) *log_post_out = lp;
}

void build_normal_model(int K, NormalData& nd, tulpa::LikelihoodSpec& spec,
                        ModelData& data, ParamLayout& layout) {
    nd.K = K;

    spec.name = "nan_grad_probe";
    spec.n_processes = 1;
    spec.ll_double = normal_likelihood<double>;
    spec.ll_arena = normal_likelihood<tulpa::arena::Var>;
    spec.gradient_fn = &normal_gradient;
    spec.n_extra_params = K;

    data.N = K;
    data.n_processes = 1;
    data.sigma_beta = 10.0;

    tulpa_hmc::ProcessData proc;
    proc.p = 0;
    data.processes.push_back(proc);
    data.model_response_data = &nd;
    data.likelihood_spec = &spec;
    data.sharing.init(1);

    data.zi_type = tulpa::ZIType::NONE;
    data.p_zi = 0;
    data.p_oi = 0;
    data.zi_prior_sd = 1.0;
    data.oi_prior_sd = 1.0;

    layout = tulpa_hmc::compute_param_layout(data);
}

}  // namespace

namespace tulpa_hmc {
    HMCResultCpp run_hmc_chain_cpp(
        const std::vector<double>& q_init,
        const ModelData& data,
        const ParamLayout& layout,
        int n_iter, int n_warmup, int L, int chain_id,
        unsigned int seed, bool verbose, int max_treedepth,
        MassMatrixType metric_type, double adapt_delta, int riemannian,
        const std::vector<double>& inv_metric_init);
}

// [[Rcpp::export]]
Rcpp::List cpp_test_nan_gradient_nuts(bool plant_nan, int K = 3,
                                       int n_iter = 40, int n_warmup = 20,
                                       int seed = 1) {
    if (K < 1) Rcpp::stop("K must be >= 1");

    NormalData nd;
    tulpa::LikelihoodSpec spec;
    ModelData data;
    ParamLayout layout;
    build_normal_model(K, nd, spec, data, layout);

    std::vector<double> init(layout.total_params, 0.3);
    std::vector<double> inv_metric_vec;

    g_plant_nan = plant_nan;
    tulpa_hmc::HMCResultCpp result = tulpa_hmc::run_hmc_chain_cpp(
        init, data, layout, n_iter, n_warmup, /*L=*/0, /*chain_id=*/1,
        static_cast<unsigned int>(seed), /*verbose=*/false,
        /*max_treedepth=*/6, tulpa::MassMatrixType::DIAG,
        /*adapt_delta=*/0.8, /*riemannian=*/0, inv_metric_vec);
    g_plant_nan = false;

    const int n_sample = result.n_sample;
    const int n_params = layout.total_params;
    Rcpp::NumericMatrix draws(n_sample, n_params);
    int n_div = 0, n_moved = 0;
    for (int s = 0; s < n_sample; s++) {
        const double* row = result.sample_row(s);
        for (int j = 0; j < n_params; j++) draws(s, j) = row[j];
        n_div += result.divergent[s];
        if (s > 0) {
            const double* prev = result.sample_row(s - 1);
            for (int j = 0; j < n_params; j++) {
                if (row[j] != prev[j]) { n_moved++; break; }
            }
        }
    }

    return Rcpp::List::create(
        Rcpp::_["draws"]       = draws,
        Rcpp::_["n_divergent"] = n_div,
        Rcpp::_["n_samples"]   = n_sample,
        Rcpp::_["n_moved"]     = n_moved
    );
}

// The two predicates the leaf reads, driven directly. `leapfrog_state_nonfinite`
// is what carries the momentum into the state scan; `hamiltonian_divergent` is
// what routes a NaN energy difference to the divergent branch, which the plain
// `H_new - H0 > delta_max` comparison does not.
// [[Rcpp::export]]
Rcpp::List cpp_test_divergence_predicates(double log_prob,
                                           Rcpp::NumericVector q,
                                           Rcpp::NumericVector p,
                                           double H0, double H_new,
                                           double delta_max = 1000.0) {
    if (q.size() != p.size()) {
        Rcpp::stop("q and p must have the same length; got %d and %d.",
                   static_cast<int>(q.size()), static_cast<int>(p.size()));
    }
    const int n = static_cast<int>(q.size());
    std::vector<double> qv(q.begin(), q.end());
    std::vector<double> pv(p.begin(), p.end());
    return Rcpp::List::create(
        Rcpp::_["state_nonfinite"] =
            tulpa_hmc::leapfrog_state_nonfinite(log_prob, qv.data(), pv.data(), n),
        Rcpp::_["hamiltonian_divergent"] =
            tulpa_hmc::hamiltonian_divergent(H0, H_new, delta_max)
    );
}
