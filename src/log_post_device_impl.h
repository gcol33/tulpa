#ifndef TULPA_LOG_POST_DEVICE_IMPL_H
#define TULPA_LOG_POST_DEVICE_IMPL_H

template<class T> static void collect_device_observation_inputs(
    const ModelData& data, const ParamLayout& layout,
    const GenericLogPostState<T>& state, const std::vector<T>& params,
    const std::vector<int>& parameter_indices, std::vector<T>& inputs,
    GenericLogPostState<device::Projection>* symbolic,
    std::vector<device::Projection>& spatial) {
    const auto append = [&](const T& value) {
        const int index = static_cast<int>(inputs.size()); inputs.push_back(value);
        return symbolic ? device::Projection::leaf(index) : device::Projection(0);
    };
    for (int k=0; k<data.n_processes; ++k)
        for (int j=0; j<data.processes[k].p; ++j) append(state.beta[k][j]);
    for (const auto& value : state.beta_zi) append(value);
    if (state.phi_spatial && state.spatial_projected.empty()) {
        if (symbolic) spatial.reserve(data.n_spatial_units);
        for (int s=0; s<data.n_spatial_units; ++s) {
            auto p=append(generic_spatial_unit_effect(s,data,layout,state));
            if (symbolic) spatial.push_back(std::move(p));
        }
        if (symbolic) symbolic->phi_spatial=spatial.data();
    }
    if (symbolic) {
        symbolic->re_term_offsets=state.re_term_offsets;
        symbolic->phi_temporal_center_all=state.phi_temporal_center_all;
    }
    if (state.phi_temporal_center_all) {
        auto p=append(state.phi_temporal_mean);
        if (symbolic) symbolic->phi_temporal_mean=std::move(p);
    }
#define TULPA_OBSERVATION_VECTOR(member) \
    if (symbolic) symbolic->member.reserve(state.member.size()); \
    for (const auto& value : state.member) { \
        auto p=append(value); if (symbolic) symbolic->member.push_back(std::move(p)); \
    }
    TULPA_OBSERVATION_VECTOR(re_vals)
    TULPA_OBSERVATION_VECTOR(spatial_projected)
    TULPA_OBSERVATION_VECTOR(gp_w)
    TULPA_OBSERVATION_VECTOR(ms_gp_effect)
    TULPA_OBSERVATION_VECTOR(hsgp_f)
    TULPA_OBSERVATION_VECTOR(spde_w)
    TULPA_OBSERVATION_VECTOR(phi_temporal)
    TULPA_OBSERVATION_VECTOR(temporal_obs_eta)
    TULPA_OBSERVATION_VECTOR(ms_temporal_eta)
    TULPA_OBSERVATION_VECTOR(svc_eta)
    TULPA_OBSERVATION_VECTOR(tvc_eta)
    TULPA_OBSERVATION_VECTOR(latent_eta)
    TULPA_OBSERVATION_VECTOR(st_delta)
#undef TULPA_OBSERVATION_VECTOR
    if (symbolic) symbolic->latent_process_eta.resize(state.latent_process_eta.size());
    for (std::size_t k=0; k<state.latent_process_eta.size(); ++k)
        for (const auto& value : state.latent_process_eta[k]) {
            auto p=append(value);
            if (symbolic) symbolic->latent_process_eta[k].push_back(std::move(p));
        }
    for (int index : parameter_indices) {
        if (index<0 || index>=static_cast<int>(params.size()))
            throw std::invalid_argument("device likelihood parameter index outside parameter vector");
        append(params[index]);
    }
}

template<class T> static bool compute_device_likelihood_sum(
    const std::vector<T>& params, const ModelData& data, const ParamLayout& layout,
    const GenericLogPostState<T>& state, T& value) {
    if constexpr (!std::is_same_v<T,double> && !std::is_same_v<T,arena::Var>) return false;
    else {
        auto policy=device::current_policy();
        device::Status status;
        DeviceLikelihood kernel;
        if (!device::select_likelihood(data,layout,kernel,*policy,status)) {
            if (!status) { policy->fail(status); throw std::runtime_error(status.message); }
            return false;
        }
        const auto& spec=*static_cast<const LikelihoodSpec*>(data.likelihood_spec);
        std::vector<T> inputs;
        auto workspace=std::static_pointer_cast<device::LikelihoodWorkspace>(policy->workspace(
            &data,&layout,"observation",[&]() {
                GenericLogPostState<device::Projection> symbolic;
                std::vector<device::Projection> spatial;
                collect_device_observation_inputs(data,layout,state,params,kernel.parameter_indices,inputs,&symbolic,spatial);
                ParamLayout physical_layout=layout;
                physical_layout.is_bym2=false;
                const int model_np=data.n_processes+(layout.has_zi ? 1 : 0);
                const int np=model_np+static_cast<int>(kernel.parameter_indices.size());
                if (np>MAX_PROCESSES) throw std::invalid_argument("device likelihood has too many score channels");
                device::ObservationPlan plan; plan.n=data.N; plan.np=np;
                plan.width=static_cast<int>(inputs.size());
                int beta=0;
                for (int k=0; k<np; ++k) {
                    const int p=k<data.n_processes ? data.processes[k].p : k<model_np ? data.p_zi : 0;
                    const auto& x=k<data.n_processes ? data.processes[k].X_flat : data.X_zi_flat;
                    plan.add_design(x.data(),p,beta); beta+=p;
                }
                std::vector<device::Projection> row(static_cast<std::size_t>(data.N)*data.n_processes);
                for (int i=0; i<data.N; ++i) {
                    device::Projection eta[MAX_PROCESSES];
                    add_generic_named_effects(i,eta,data,physical_layout,symbolic);
                    add_generic_st_effect(i,eta,data,physical_layout,symbolic);
                    for (int k=0; k<data.n_processes; ++k) row[static_cast<std::size_t>(k)*data.N+i]=std::move(eta[k]);
                }
                for (int k=0; k<np; ++k) {
                    const std::vector<double>* offset=k<data.n_processes ? &data.processes[k].offset : nullptr;
                    if (offset && !offset->empty() && offset->size()!=static_cast<std::size_t>(data.N))
                        throw std::invalid_argument("device observation offset length must equal N");
                    for (int i=0; i<data.N; ++i) {
                        const auto e=k<data.n_processes ? std::move(row[static_cast<std::size_t>(k)*data.N+i])
                            : k<model_np ? device::Projection(0) : device::Projection::leaf(plan.width-(np-k));
                        plan.append_row(e,offset && !offset->empty() ? (*offset)[i] : 0);
                    }
                }
                if (!spec.device_fn(data,layout,data.model_response_data,true,kernel))
                    throw std::logic_error("device likelihood declined its initialized descriptor");
                return std::make_shared<device::LikelihoodWorkspace>(std::move(plan),std::move(kernel));
            }));
        if (inputs.empty()) {
            std::vector<device::Projection> spatial;
            collect_device_observation_inputs(data,layout,state,params,workspace->kernel().parameter_indices,inputs,nullptr,spatial);
        }
        spec.device_fn(data,layout,data.model_response_data,false,workspace->kernel());
        std::vector<double> coefficients; coefficients.reserve(inputs.size());
        for (const auto& input : inputs) coefficients.push_back(math::precision_value(input));
        device::LikelihoodResult result;
        status=workspace->evaluate(coefficients,1,true,false,result);
        if (!status) { policy->fail(status); throw std::runtime_error(status.message); }
        policy->record("likelihood",true);
        if constexpr (std::is_same_v<T,double>) value=result.value[0];
        else {
            std::vector<int32_t> indices; indices.reserve(inputs.size());
            for (const auto& input : inputs) indices.push_back(input.idx_);
            std::vector<int32_t> output;
            auto* ar=inputs.front().arena_;
            ar->add_custom_backward(indices,{result.value[0]},
                [gradient=std::move(result.gradient)](const double*,int n,const double*,int,
                    const double* adj,double* out) {
                    for (int j=0; j<n; ++j) out[j]=adj[0]*gradient[j];
                },output);
            value.arena_=ar; value.idx_=output[0];
        }
        return true;
    }
}
#endif
