  // Populate the obs-loop context (only meaningful when populate_obs_state).
  // Pointers reference either `params`, `data`, thread_local workspaces, or
  // buffers in `state` that outlive this call.
  if (populate_obs_state) {
    ctx.beta_num        = beta_num;
    ctx.beta_denom      = beta_denom;
    ctx.re              = re;
    ctx.re_nc_flat      = re_nc_flat.empty() ? nullptr : re_nc_flat.data();
    ctx.phi_spatial     = phi_spatial;
    ctx.theta_bym2      = theta_bym2;
    ctx.sigma_s_bym2    = sigma_s_bym2;
    ctx.sigma_u_bym2    = sigma_u_bym2;
    ctx.phi_temporal    = phi_temporal;
    ctx.gp_w            = gp_w;
    ctx.gp_local        = gp_local;
    ctx.gp_regional     = gp_regional;
    ctx.msgp_hsgp_f_local    = &msgp_hsgp_f_local;
    ctx.msgp_hsgp_f_regional = &msgp_hsgp_f_regional;
    ctx.hsgp_f          = &hsgp_f;
    ctx.trend           = trend;
    ctx.seasonal        = seasonal;
    ctx.short_term      = short_term;
    ctx.latent_sigma    = &latent_sigma;
    ctx.latent_factors  = &latent_factors_vec;
    ctx.st_delta        = st_delta;
    ctx.tvc_eta         = &tvc_eta;
    ctx.svc_eta         = svc_eta_ptr;
    ctx.beta_zi         = beta_zi;
    ctx.beta_oi         = beta_oi;
    ctx.phi_num         = phi_num;
    ctx.phi_denom       = phi_denom;
  }

  return log_post;
