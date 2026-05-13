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

      // Collapsed-spatial mode storage was deleted in Phase D
      // (gcol33/tulpa#15) along with the icar_collapsed / gp_collapsed
      // kernels. Downstream packages on the generic LikelihoodSpec
      // path never set those flags.

      sample_idx++;
    } else {
      warmup_total_leapfrog += iter_n_leapfrog;  // TEMP: diagnostic
    }

