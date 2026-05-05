// ============================================================================
// LikelihoodSpec-driven Laplace shim. Routes the per-observation log-lik
// and IRLS weights through data->likelihood_spec instead of the family enum.
// First cut: n_processes == 1 with at most one iid RE term.
// ============================================================================

namespace tulpa {

void laplace_mode_spec_dense_impl(
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& params_inout,
    const std::vector<int>& re_group_1based,
    int max_iter, double tol, int n_threads,
    int* n_iter_out,
    int* converged_out,
    double* log_det_Q_out,
    double* log_marginal_out
);

} // namespace tulpa

extern "C" void tulpa_laplace_spec_dense_impl(
    const tulpa::ModelData* data,
    const tulpa::ParamLayout* layout,
    double* params_inout,
    int n_params,
    const int* re_group,
    int n_re_group,
    int max_iter,
    double tol,
    int n_threads,
    tulpa::LaplaceShimResult* result_out
) {
    if (data == nullptr || layout == nullptr || params_inout == nullptr ||
        result_out == nullptr) {
        Rf_error("tulpa_laplace_spec_dense: null pointer in call");
        return;
    }
    if (n_params != layout->total_params) {
        Rf_error("tulpa_laplace_spec_dense: n_params (%d) != layout.total_params (%d)",
                 n_params, layout->total_params);
        return;
    }

    std::vector<double> params(params_inout, params_inout + n_params);
    std::vector<int> re_idx;
    if (re_group != nullptr && n_re_group > 0) {
        re_idx.assign(re_group, re_group + n_re_group);
    }

    int n_iter = 0;
    int converged = 0;
    double log_det_Q = 0.0;
    double log_marginal = 0.0;

    tulpa::laplace_mode_spec_dense_impl(
        *data, *layout, params, re_idx,
        max_iter, tol, n_threads,
        &n_iter, &converged, &log_det_Q, &log_marginal
    );

    // Write the moved latent slots back to the caller's params buffer so a
    // warm-start round-trip works without a separate scratch allocation.
    for (int j = 0; j < n_params; j++) params_inout[j] = params[j];

    // Concatenate every process's beta block in order, then every RE term
    // in term order. Caller's view of the result mode mirrors
    // SpecLatentLayout in src/laplace_spec.cpp:
    //   [beta_0 | beta_1 | ... | beta_{np-1} | re_term_0 | re_term_1 | ...]
    // Single-term case is bit-identical to the previous concatenation
    // because re_start_multi[0] / re_end_multi[0] alias re_start / re_end.
    const int np = (int)layout->process_beta_start.size();
    int p_total = 0;
    for (int k = 0; k < np; k++) {
        if (k < (int)layout->process_beta_count.size()) {
            p_total += layout->process_beta_count[k];
        }
    }
    int n_re_total = 0;
    if (layout->has_re) {
        if (!layout->re_start_multi.empty()) {
            for (size_t t = 0; t < layout->re_start_multi.size(); t++) {
                int rs = layout->re_start_multi[t];
                int re = layout->re_end_multi[t];
                if (rs >= 0 && re >= rs) n_re_total += (re - rs);
            }
        } else if (layout->re_start >= 0) {
            n_re_total = layout->re_end - layout->re_start;
        }
    }
    int n_x = p_total + n_re_total;

    result_out->n_x = n_x;
    result_out->mode = (n_x > 0) ? new double[n_x] : new double[1];
    int out_idx = 0;
    for (int k = 0; k < np; k++) {
        int beta_start = layout->process_beta_start[k];
        int p_k = (k < (int)layout->process_beta_count.size())
                  ? layout->process_beta_count[k] : 0;
        for (int j = 0; j < p_k; j++) {
            result_out->mode[out_idx++] = params[beta_start + j];
        }
    }
    if (layout->has_re) {
        if (!layout->re_start_multi.empty()) {
            for (size_t t = 0; t < layout->re_start_multi.size(); t++) {
                int rs = layout->re_start_multi[t];
                int re = layout->re_end_multi[t];
                if (rs < 0 || re < rs) continue;
                for (int j = rs; j < re; j++) {
                    result_out->mode[out_idx++] = params[j];
                }
            }
        } else if (layout->re_start >= 0) {
            for (int g = layout->re_start; g < layout->re_end; g++) {
                result_out->mode[out_idx++] = params[g];
            }
        }
    }
    result_out->log_det_Q    = log_det_Q;
    result_out->log_marginal = log_marginal;
    result_out->n_iter       = n_iter;
    result_out->converged    = converged;
}
