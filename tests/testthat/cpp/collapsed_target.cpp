#include <Rcpp.h>
#include <tulpa/nuts_api.h>
#include <tulpa/likelihood.h>
#include <type_traits>

template<class T> T collapsed_target_ll(int, const T* eta, const T&, const T&,
    const std::vector<T>&, const tulpa::ModelData&, const tulpa::ParamLayout&, const void*) {
    if constexpr (std::is_same_v<T, tulpa::arena::Var>) tulpa::arena::current_arena() = eta[0].arena_;
    return -.5 * eta[0] * eta[0];
}

template<class T> void collapsed_target_weights(int, const T* eta,
    tulpa::EtaWeightScalar<T>, tulpa::EtaWeightScalar<T>, const std::vector<T>&,
    const tulpa::ModelData&, const tulpa::ParamLayout&, const void*, T* score, T* weights) {
    if constexpr (std::is_same_v<T, tulpa::arena::Var>) tulpa::arena::current_arena() = eta[0].arena_;
    score[0] = -eta[0]; weights[0] = T(1);
}

// [[Rcpp::export]]
Rcpp::List cpp_collapsed_target(bool declared, int n_chains = 1, bool collapsed = true,
    bool weights = true) {
    tulpa::LikelihoodSpec spec;
    spec.n_processes = 1;
    spec.ll_double = collapsed_target_ll<double>;
    spec.ll_arena = collapsed_target_ll<tulpa::arena::Var>;
    spec.eta_weights_fn = collapsed_target_weights<double>;
    spec.eta_weights_arena_fn = collapsed_target_weights<tulpa::arena::Var>;
    spec.gaussian_spatial_marginal = declared;
    if (!weights) spec.eta_weights_arena_fn = nullptr;
    tulpa::ModelData data;
    data.N = 4; data.n_processes = 1; data.processes.resize(1); data.sharing.init(1);
    data.processes[0].p = 1; data.processes[0].X_flat.assign(4, 1.0);
    data.likelihood_spec = &spec;
    data.spatial_type = tulpa::SpatialType::ICAR;
    data.icar_collapsed = collapsed;
    data.spatial_component_centering = true;
    data.spatial_group = {1,2,1,2};
    data.set_spatial_adjacency(2, {0,1,2}, {1,0}, {1,1});
    auto layout = tulpa::compute_layout(data);
    std::vector<double> init(n_chains * layout.total_params, 0.0);
    std::vector<tulpa::NUTSResult> results(n_chains, tulpa::NUTSResult{});
    tulpa::OmpCapScope threads(1);
    if (n_chains == 1)
        tulpa::get_nuts_fn()(&data, &layout, init.data(), layout.total_params,
            30, 15, 3, .9, 1177, 0, nullptr, nullptr, 0.0, results.data());
    else
        tulpa::get_nuts_chains_fn()(&data, &layout, init.data(), layout.total_params,
            n_chains, 30, 15, 3, .9, 1177, 0, nullptr, nullptr, nullptr, results.data());
    Rcpp::List out(n_chains);
    for (int c=0; c<n_chains; ++c) {
        const auto& r = results[c];
        out[c] = Rcpp::List::create(Rcpp::Named("target") = tulpa::sampling_target_name(r.target),
            Rcpp::Named("inference_tier") = r.inference_tier,
            Rcpp::Named("n_fields") = r.n_spatial_fields,
            Rcpp::Named("n_samples") = r.n_sample);
        results[c].free_buffers();
    }
    return out;
}

// [[Rcpp::export]]
Rcpp::List cpp_laplace_target_metadata() {
    const auto target = tulpa::SamplingTarget::LAPLACE_SPATIAL_MARGINAL;
    return Rcpp::List::create(Rcpp::Named("target") = tulpa::sampling_target_name(target),
        Rcpp::Named("inference_tier") = tulpa::sampling_target_tier(target));
}
