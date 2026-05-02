    // Store sample (flat row-major storage, single memcpy)
    if (!is_warmup) {
      // NC GP: transform z -> w for stored samples (keep q as z for sampling)
      if (data.gp_parameterization == 1 && data.has_gp && layout.is_gp) {
          double sigma2_store = std::exp(q[layout.log_sigma2_gp_idx]);
          double phi_store = std::exp(q[layout.log_phi_gp_idx]);
          static thread_local tulpa_gp::NNGPNCWorkspace nc_ws_store;
          tulpa_gp::nngp_nc_forward(&q[layout.gp_w_start], sigma2_store, phi_store,
                                     data.gp_data, nc_ws_store);
          // Copy q, replace z with w
          std::memcpy(result.sample_row(sample_idx), q.data(),
                      n_params * sizeof(double));
          double* row = result.sample_row(sample_idx);
          int N_gp = data.gp_data.n_obs;
          for (int i = 0; i < N_gp; i++) {
              row[layout.gp_w_start + i] = nc_ws_store.w[i];
          }
      } else {
          std::memcpy(result.sample_row(sample_idx), q.data(),
                      n_params * sizeof(double));
      }
      result.log_prob[sample_idx] = log_prob_current;
      result.accept_prob[sample_idx] = alpha;
      result.n_leapfrog[sample_idx] = iter_n_leapfrog;
      result.divergent[sample_idx] = divergent ? 1 : 0;
      result.treedepth[sample_idx] = iter_treedepth;

      // Collapsed: store mode values from the last gradient evaluation
      if (data.gp_collapsed && data.has_gp && result.n_gp_collapsed > 0) {
          collapsed_gp_store_sample(sample_idx, collapsed_gp_ws,
              result.gp_w_star_flat, result.n_gp_collapsed);
      }
      if ((data.icar_collapsed || data.bym2_collapsed) && result.n_icar_collapsed > 0) {
          collapsed_icar_store_sample(sample_idx, data, collapsed_icar_ws,
              result.icar_phi_star_flat, result.bym2_theta_star_flat,
              result.n_icar_collapsed);
      }

      sample_idx++;
    } else {
      warmup_total_leapfrog += iter_n_leapfrog;  // TEMP: diagnostic
    }

