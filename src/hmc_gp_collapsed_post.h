
// =========================================================================
// High-level wrappers for compute_log_post integration
// These encapsulate all collapsed GP logic so the main log_post
// function only needs a single call instead of 60+ lines of inline code.
// =========================================================================

struct CollapsedGPLogPostResult {
    double log_post_contribution = 0.0;  // NNGP prior + Laplace correction
    // After calling, ws.w_star contains the mode values.
    // Caller should copy ws.w_star to gp_w buffer for observation loop.
};

// Compute collapsed GP contribution to log_post:
//   1. Find w* via Newton
//   2. Evaluate NNGP prior at w*: -0.5*w*^T Q w* + 0.5*log|Q|
//   3. Add Laplace correction: -0.5*log|W+Q|
//   4. ws.w_star is populated for use in observation loop
inline CollapsedGPLogPostResult collapsed_gp_log_post_contribution(
    const double* beta_num, const double* beta_denom,
    double sigma2_gp, double phi_gp,
    double phi_num, double phi_denom,
    const ModelData& data,
    CollapsedGPWorkspace& ws) {

    CollapsedGPLogPostResult res;
    int N_gp = data.gp_data.n_obs;

    // Find mode w*
    collapsed_gp_find_mode(beta_num, beta_denom, sigma2_gp, phi_gp,
                           phi_num, phi_denom, data, ws);

    // NNGP prior at w*: -0.5 * w*^T Q w* + 0.5 * log|Q|
    std::vector<double> Qw(N_gp);
    nngp_precision_matvec(ws.w_star.data(), Qw.data(), ws);
    double wQw = 0.0;
    for (int i = 0; i < N_gp; i++) wQw += ws.w_star[i] * Qw[i];

    double log_det_Q = 0.0;
    for (int i = 0; i < N_gp; i++) log_det_Q -= std::log(ws.d_cond[i]);

    res.log_post_contribution = -0.5 * wQw + 0.5 * log_det_Q;

    // Laplace correction
    res.log_post_contribution += ws.laplace_log_det;

    return res;
}

// Store collapsed GP mode values (w*) into result buffer
inline void collapsed_gp_store_sample(
    int sample_idx,
    const CollapsedGPWorkspace& ws,
    std::vector<double>& gp_w_star_flat,
    int N_gp) {
    std::memcpy(&gp_w_star_flat[sample_idx * N_gp],
                ws.w_star.data(), N_gp * sizeof(double));
}
