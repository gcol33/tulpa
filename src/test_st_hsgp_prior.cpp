// test_st_hsgp_prior.cpp
// The HSGP-ST interaction prior, driven directly.
//
// Nothing in tulpa sets ModelData::has_spatiotemporal -- spatiotemporal() errors
// at the R door and the whole ST sampler path is a consumer-package
// configuration -- so this fixture fills the ModelData a consumer would hand the
// engine, exactly as src/test_st_iv_fixture.cpp does for the Kronecker Type-IV
// branch. It is the only way an R test reaches the spectral branch.
//
// It evaluates the SHIPPED density (tulpa::priors::compute_st_prior), not a copy
// of it: the fixture builds the ModelData and the layout and nothing else.

#include <Rcpp.h>

#include <string>
#include <vector>

#include "hmc_sampler.h"
#include "tulpa/likelihood.h"
#include "tulpa/autodiff_arena.h"
#include "tulpa_priors_st.h"

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

namespace {

struct StHsgpData { int n = 0; };

// A flat likelihood: the fixture scores the PRIOR, so the observation arm is
// present only because compute_param_layout needs a well-formed model.
template <typename T>
T st_hsgp_likelihood(int i, const T* eta, const T& logit_zi, const T& logit_oi,
                     const std::vector<T>& params, const ModelData& data,
                     const ParamLayout& layout, const void* model_data) {
    (void)i; (void)eta; (void)logit_zi; (void)logit_oi;
    (void)params; (void)data; (void)layout; (void)model_data;
    return T(0.0);
}

void build_st_hsgp_model(
    int M, int T_st,
    const std::string& temporal,
    bool temporal_cyclic,
    const std::vector<double>& eigenvalues,
    StHsgpData& sd,
    tulpa::LikelihoodSpec& spec,
    ModelData& data,
    ParamLayout& layout
) {
    if (M < 1 || T_st < 3) Rcpp::stop("need M >= 1 and T >= 3");
    if ((int)eigenvalues.size() != M) {
        Rcpp::stop("eigenvalues must have m_total = %d entries", M);
    }

    sd.n = 1;

    spec.name = "st_hsgp_prior_probe";
    spec.n_processes = 1;
    spec.ll_double = st_hsgp_likelihood<double>;
    spec.ll_arena = st_hsgp_likelihood<tulpa::arena::Var>;

    data.N = 1;
    data.n_processes = 1;
    data.sigma_beta = 10.0;

    tulpa::ProcessData proc;
    proc.p = 1;
    proc.X_flat.assign(1, 1.0);
    data.processes.push_back(proc);
    data.model_response_data = &sd;
    data.likelihood_spec = &spec;
    data.sharing.init(1);

    data.zi_type = tulpa::ZIType::NONE;
    data.p_zi = 0;
    data.p_oi = 0;
    data.zi_prior_sd = 1.0;
    data.oi_prior_sd = 1.0;

    data.has_spatiotemporal = true;
    data.st_parameterization = 0;
    data.st_is_hsgp = true;
    data.st_sigma2_prior_U = 1.0;
    data.st_sigma2_prior_alpha = 0.01;
    data.st_hsgp_sigma2_prior_U = 1.0;
    data.st_hsgp_sigma2_prior_alpha = 0.01;
    data.st_hsgp_data.m_total = M;
    data.st_hsgp_data.eigenvalues = eigenvalues;

    auto& st = data.spatiotemporal_data;
    st.type = tulpa::STType::TYPE_IV;
    st.shared = true;
    st.n_spatial = M;
    st.n_times = T_st;
    st.n_params = M * T_st;
    st.temporal_type = (temporal == "rw2") ? tulpa::TemporalType::RW2
                                           : tulpa::TemporalType::RW1;
    st.temporal_cyclic = temporal_cyclic;

    layout = tulpa_hmc::compute_param_layout(data);
}

}  // namespace

// The interaction prior at one parameter vector.
//
// `delta` is the M x T_st interaction, basis-major (the layout the branch reads:
// st_delta[j * T_st + t]). The hyperparameters are given on their sampled log
// scales, so a caller states exactly what the density sees.
// [[Rcpp::export]]
double cpp_test_st_hsgp_log_prior(
    Rcpp::NumericVector delta,
    Rcpp::NumericVector eigenvalues,
    int T,
    double log_tau_st,
    double log_sigma2_hsgp,
    double log_lengthscale_hsgp,
    std::string temporal = "rw2",
    bool temporal_cyclic = false
) {
    const int M = eigenvalues.size();
    StHsgpData sd;
    tulpa::LikelihoodSpec spec;
    ModelData data;
    ParamLayout layout;
    build_st_hsgp_model(M, T, temporal, temporal_cyclic,
                        std::vector<double>(eigenvalues.begin(), eigenvalues.end()),
                        sd, spec, data, layout);

    if ((int)delta.size() != M * T) {
        Rcpp::stop("delta has %d entries; the layout holds %d",
                   (int)delta.size(), M * T);
    }

    std::vector<double> params(layout.total_params, 0.0);
    params[layout.log_tau_st_idx] = log_tau_st;
    params[layout.log_sigma2_st_hsgp_idx] = log_sigma2_hsgp;
    params[layout.log_lengthscale_st_hsgp_idx] = log_lengthscale_hsgp;
    for (int k = 0; k < M * T; k++) {
        params[layout.st_delta_start + k] = delta[k];
    }

    std::vector<double> st_delta;
    return tulpa::priors::compute_st_prior(params, data, layout, st_delta);
}

// The precision the trend pin holds a per-basis ramp coefficient at, read from
// the engine's own helper so the test does not restate the constant.
// [[Rcpp::export]]
double cpp_test_st_trend_precision(int T) {
    return tulpa_st::st_trend_precision(T);
}
