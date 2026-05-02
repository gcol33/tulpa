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

    int beta_start = layout->process_beta_start.empty()
                     ? 0 : layout->process_beta_start[0];
    int p          = layout->process_beta_count.empty()
                     ? 0 : layout->process_beta_count[0];
    int n_re       = (layout->has_re && layout->re_start >= 0)
                     ? (layout->re_end - layout->re_start) : 0;
    int n_x        = p + n_re;

    result_out->n_x = n_x;
    result_out->mode = (n_x > 0) ? new double[n_x] : new double[1];
    for (int j = 0; j < p; j++) {
        result_out->mode[j] = params[beta_start + j];
    }
    if (n_re > 0) {
        for (int g = 0; g < n_re; g++) {
            result_out->mode[p + g] = params[layout->re_start + g];
        }
    }
    result_out->log_det_Q    = log_det_Q;
    result_out->log_marginal = log_marginal;
    result_out->n_iter       = n_iter;
    result_out->converged    = converged;
}
