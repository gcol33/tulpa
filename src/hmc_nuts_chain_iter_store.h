    // Store sample (flat row-major storage, single memcpy baseline; NC
    // blocks below overwrite their own z slice with the reconstructed field
    // w so stored draws carry w, not z -- keeping q itself as z for
    // sampling). The scales come off q through safe_exp, the same bounded map
    // the log-posterior's own transform uses (nngp_nc_term_apply.cpp), so the
    // stored field is the same function of q as the field the likelihood saw
    // and a large draw gives a bounded reconstruction rather than Inf * 0. GP / SVC / multiscale-GP NC blocks are independent (a model
    // can carry more than one structured term at once), so each is applied
    // unconditionally to `row` rather than as mutually exclusive branches.
    if (!is_warmup) {
      std::memcpy(result.sample_row(sample_idx), q.data(),
                  n_params * sizeof(double));
      double* row = result.sample_row(sample_idx);

      // NC GP: transform z -> w for stored samples (keep q as z for sampling)
      if (data.gp_parameterization == 1 && data.has_gp && layout.is_gp) {
          double sigma2_store = tulpa::math::safe_exp(q[layout.log_sigma2_gp_idx]);
          double phi_store = tulpa::math::safe_exp(q[layout.log_phi_gp_idx]);
          // POD-pointer TLS (constant init, no thread-atexit destructor): a
          // lazily-initialized thread_local object here corrupts the heap
          // under the mingw toolchain when chains run in parallel. The
          // workspace intentionally leaks per thread.
          static thread_local tulpa_gp::NNGPNCWorkspace* nc_ws_store_p = nullptr;
          if (!nc_ws_store_p) nc_ws_store_p = new tulpa_gp::NNGPNCWorkspace();
          tulpa_gp::NNGPNCWorkspace& nc_ws_store = *nc_ws_store_p;
          tulpa_gp::nngp_nc_forward(&q[layout.gp_w_start], sigma2_store, phi_store,
                                     tulpa_gp::make_gp_nc_view(data.gp_data), nc_ws_store);
          int N_gp = data.gp_data.n_obs;
          for (int i = 0; i < N_gp; i++) {
              row[layout.gp_w_start + i] = nc_ws_store.w[i];
          }
      }

      // SVC (NNGP path only -- HSGP SVC carries basis coefficients, not a
      // field, and no z block here). Under the non-centered parameterization
      // the stored slice is the reconstructed w rather than the sampled z;
      // under the centered one it is already w. Either way it is stored
      // CENTERED, matching what the log-post fed into svc_eta -- the level is
      // identified by centering, so an uncentered draw would report a
      // different field than the one the likelihood saw.
      if (data.has_svc && layout.has_svc && !data.svc_is_hsgp) {
          static thread_local tulpa_gp::NNGPNCWorkspace* svc_ws_store_p = nullptr;
          if (!svc_ws_store_p) svc_ws_store_p = new tulpa_gp::NNGPNCWorkspace();
          tulpa_gp::NNGPNCWorkspace& svc_ws_store = *svc_ws_store_p;

          const auto& sv = data.svc_data;
          const bool nc = (data.svc_parameterization == 1);
          const tulpa_gp::NNGPNCView svc_view = tulpa_gp::make_svc_nc_view(sv);

          int N_svc = sv.n_obs;
          for (int j = 0; j < sv.n_svc; j++) {
              int w0 = layout.svc_w_start + j * N_svc;
              if (nc) {
                  double sigma2_store = tulpa::math::safe_exp(q[layout.log_sigma2_svc_start + j]);
                  double phi_store = tulpa::math::safe_exp(q[layout.log_phi_svc_start + j]);
                  tulpa_gp::nngp_nc_forward(&q[w0], sigma2_store, phi_store,
                                             svc_view, svc_ws_store);
                  for (int i = 0; i < N_svc; i++) row[w0 + i] = svc_ws_store.w[i];
              }
              (void)tulpa::s2z_centre_component(row, w0, N_svc);
          }
      }

      // NC multiscale GP: same transform, once per scale.
      if (data.msgp_parameterization == 1 && data.has_multiscale_gp &&
          layout.is_multiscale_gp && !data.msgp_is_hsgp) {
          static thread_local tulpa_gp::NNGPNCWorkspace* msgp_ws_store_p = nullptr;
          if (!msgp_ws_store_p) msgp_ws_store_p = new tulpa_gp::NNGPNCWorkspace();
          tulpa_gp::NNGPNCWorkspace& msgp_ws_store = *msgp_ws_store_p;

          const auto& ms = data.multiscale_gp_data;
          int N_local = ms.n_obs;
          double sigma2_local_store = tulpa::math::safe_exp(q[layout.log_sigma2_gp_local_idx]);
          double phi_local_store = tulpa::math::safe_exp(q[layout.log_phi_gp_local_idx]);
          tulpa_gp::nngp_nc_forward(&q[layout.gp_local_start], sigma2_local_store,
                                     phi_local_store, tulpa_gp::make_msgp_nc_view_local(ms),
                                     msgp_ws_store);
          for (int i = 0; i < N_local; i++) {
              row[layout.gp_local_start + i] = msgp_ws_store.w[i];
          }

          int N_regional = ms.n_obs;
          double sigma2_regional_store = tulpa::math::safe_exp(q[layout.log_sigma2_gp_regional_idx]);
          double phi_regional_store = tulpa::math::safe_exp(q[layout.log_phi_gp_regional_idx]);
          tulpa_gp::nngp_nc_forward(&q[layout.gp_regional_start], sigma2_regional_store,
                                     phi_regional_store, tulpa_gp::make_msgp_nc_view_regional(ms),
                                     msgp_ws_store);
          for (int i = 0; i < N_regional; i++) {
              row[layout.gp_regional_start + i] = msgp_ws_store.w[i];
          }
      }

      result.log_prob[sample_idx] = log_prob_current;
      result.accept_prob[sample_idx] = alpha;
      result.n_leapfrog[sample_idx] = iter_n_leapfrog;
      result.divergent[sample_idx] = divergent ? 1 : 0;
      result.treedepth[sample_idx] = iter_treedepth;


      sample_idx++;
    } else {
      warmup_total_leapfrog += iter_n_leapfrog;  // summed over warmup (verbose)
    }

