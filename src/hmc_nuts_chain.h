// hmc_nuts_chain.h
// Fragment of hmc_nuts_sampler.cpp. Included from the umbrella
// translation unit inside namespace tulpa_hmc; do NOT add a
// namespace wrapper here; do not list this file in the package SRCS —
// it is not a standalone translation unit.
// Single-chain NUTS driver: run_hmc_chain_cpp + run_hmc_chain wrapper.
#ifndef TULPA_HMC_NUTS_CHAIN_H
#define TULPA_HMC_NUTS_CHAIN_H


// =====================================================================
// Run single HMC chain
// =====================================================================

// Pure C++ version - safe for OpenMP parallel regions
HMCResultCpp run_hmc_chain_cpp(
    const std::vector<double>& q_init,
    const ModelData& data,
    const ParamLayout& layout,
    int n_iter,
    int n_warmup,
    int L,
    int chain_id,
    unsigned int seed,
    bool verbose,
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    int riemannian
) {
  int n_params = q_init.size();
  int n_sample = n_iter - n_warmup;
  bool use_nuts = (L == 0);

  HMCResultCpp result;
  result.n_params_stored = n_params;
  result.samples_flat.resize(static_cast<size_t>(n_sample) * n_params);
  result.log_prob.resize(n_sample);
  result.accept_prob.resize(n_sample);
  result.n_leapfrog.resize(n_sample, L);
  result.divergent.resize(n_sample, 0);
  result.treedepth.resize(n_sample, 0);
  result.n_warmup = n_warmup;
  result.n_sample = n_sample;
  result.chain_id = chain_id;
  result.n_max_treedepth = 0;

  // Collapsed GP: allocate w* storage
  if (data.gp_collapsed && data.has_gp) {
      result.n_gp_collapsed = data.gp_data.n_obs;
      result.gp_w_star_flat.resize(static_cast<size_t>(n_sample) * data.gp_data.n_obs, 0.0);
  }

  // Collapsed ICAR/BYM2: allocate phi*/theta* storage
  if (data.icar_collapsed || data.bym2_collapsed) {
      int S = data.n_spatial_units;
      result.n_icar_collapsed = S;
      result.icar_phi_star_flat.resize(static_cast<size_t>(n_sample) * S, 0.0);
      if (data.bym2_collapsed) {
          result.bym2_theta_star_flat.resize(static_cast<size_t>(n_sample) * S, 0.0);
      }
  }

  std::mt19937 rng(seed + chain_id * 12345);
  std::normal_distribution<double> normal(0.0, 1.0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  // Reset VecGradWorkspace cache for new model fit
  reset_grad_workspace_cache();

  std::vector<double> q = q_init;

  // For NUTS: fuse initial log_post + gradient into single O(N) pass
  std::vector<double> current_grad(n_params);
  double log_prob_current;
  if (use_nuts) {
    compute_gradient(q, data, layout, current_grad, &log_prob_current);
  } else {
    // Use same log-post function as the active gradient mode
    const bool is_generic = data.n_processes > 0 && data.likelihood_spec != nullptr;
    if (!is_generic &&
        (g_gradient_mode == GradientMode::AUTODIFF_ARENA ||
        g_gradient_mode == GradientMode::AUTODIFF_FWD ||
        g_gradient_mode == GradientMode::AUTODIFF_TAPE)) {
      log_prob_current = tulpa::compute_log_post_impl(q, data, layout);
    } else {
      log_prob_current = compute_log_post(q, data, layout);
    }
  }

  double epsilon = find_reasonable_epsilon(q, data, layout, rng);

  // Compute target_boost for challenging model combinations
  // MSGP and GP with temporal are particularly challenging
  double target_boost = 0.0;
  if (data.has_multiscale_gp) {
    target_boost += 0.10;  // MSGP models need higher target acceptance
    if (layout.has_temporal) {
      target_boost += 0.05;  // MSGP + temporal is even more challenging
    }
  } else if (data.spatial_type == SpatialType::GP) {
    target_boost += 0.05;  // GP models moderately challenging
    if (layout.has_temporal) {
      target_boost += 0.05;  // GP + temporal combination
    }
  }
  DualAveraging da(epsilon, n_params, target_boost);

  // For NUTS: model-adaptive target acceptance
  // Store in nuts_target_accept for reuse at mass window boundaries (avoids bug
  // where da.target_accept was reset to 0.80 at each window reset).
  double nuts_target_accept = 0.80;
  if (use_nuts) {
    if (adapt_delta > 0) {
      // User override
      nuts_target_accept = adapt_delta;
    } else {
      // Auto-select based on model complexity
      nuts_target_accept = 0.80;  // Stan default base

      // BYM2: high correlation between ICAR phi + unstructured theta
      if (data.spatial_type == SpatialType::BYM2) {
        nuts_target_accept = 0.90;
      }
      // ICAR: correlated spatial params need slightly higher target
      else if (data.spatial_type == SpatialType::ICAR) {
        nuts_target_accept = 0.85;
      }

      // Correlated random slopes add funnel geometry
      if (data.has_re_correlated_slopes) {
        nuts_target_accept = std::max(nuts_target_accept, 0.90);
      }

      // Temporal GP NC: z ~ N(0,1) decorrelates parameters, lower target OK
      // Benchmarked: 0.70 gives 20% fewer LF steps and 50% less seed variance
      if (layout.is_temporal_gp && nuts_target_accept > 0.70) {
        nuts_target_accept = 0.70;
      }

      nuts_target_accept = std::min(0.99, nuts_target_accept);
    }
    da.target_accept = nuts_target_accept;
  }

  // current_grad already computed above (fused with log_prob for NUTS)

  // Select and initialize mass matrix (AUTO resolution, block detection, sparse GMRF)
  DenseMassMatrix mass;
  MassMatrixConfig mm_config = select_and_init_mass_matrix(mass, data, layout, n_params, metric_type, verbose);
  MassMatrixType effective_metric = mm_config.effective_metric;
  bool auto_selected_diag = mm_config.auto_selected_diag;
  std::vector<std::pair<int,int>> block_specs = std::move(mm_config.block_specs);

  // Warm-start mass matrix diagonal from model structure
  warm_start_mass_matrix(mass, data, layout, n_params, verbose);

  // Recompute epsilon with warm-start mass (if informed)
  // This gives the dual averaging a better starting point when mass is pre-set
  if (mass.type != MassMatrixType::DIAG && !mass.inv_mass_diag.empty()) {
    epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
  }

  WelfordStats mass_stats(n_params);              // Always track diagonal
  WelfordCovStats cov_stats(n_params);            // Only used when dense
  bool use_mass_matrix = false;

  // L-BFGS mass matrix adaptation (warmup-only)
  // Uses L-BFGS to learn curvature during warmup, then switches to standard HMC
  bool use_lbfgs = data.has_multiscale_gp &&
                   data.multiscale_gp_data.sampler == tulpa_gp::MSGPSampler::LBFGS;
  tulpa_gp::LBFGSState lbfgs_state;
  std::vector<double> q_prev, grad_prev;
  bool lbfgs_initialized = false;
  bool lbfgs_warmup_done = false;  // After warmup, use standard HMC
  if (use_lbfgs) {
    lbfgs_state = tulpa_gp::LBFGSState(10, n_params);
    q_prev.resize(n_params);
    grad_prev.resize(n_params);
  }

  // Stan-style expanding warmup windows for mass matrix adaptation
  // Phase 1: [0, init_buffer) - step size adaptation only
  // Phase 2: [init_buffer, n_warmup - term_buffer) - mass matrix adaptation
  //   Windows double in size: 25, 50, 100, 200, ...
  //   Last window extends to fill remaining space
  // Phase 3: [n_warmup - term_buffer, n_warmup) - final step size tuning
  // Models with structured warm-start (ICAR degree, BYM2 scale, HSGP) already
  // have reasonable mass, so we can start adaptation earlier (init_buffer=25).
  // This saves ~50 iterations of deep-tree warmup. Temporal_gp warm-start is
  // trivial (identity-like) so it still needs the full 75 iterations.
  bool has_structured_warmstart = layout.is_hsgp || layout.is_bym2 ||
    (layout.has_spatial && !layout.is_bym2 && data.spatial_type == SpatialType::ICAR && !data.adj_row_ptr.empty());
  int init_buffer = has_structured_warmstart ? 25 : 75;
  int term_buffer = 50;
  // For high-dimensional models (p>80), a 25-sample first mass window gives
  // very noisy variance estimates (25 samples / 108 params = 0.23 samples/param).
  // Skip the tiny first window by using a larger init_window (=50), so the first
  // mass update has ~50 samples. This trades one less mass update for better quality.
  int init_window = (n_params > 80) ? 50 : 25;

  // Dense mass models: balance final step size tuning vs warmup budget.
  // Models with p>100 need sufficient mass adaptation windows (warmup is fixed),
  // so keep term_buffer moderate. Previous 75 was too aggressive ? used 30%
  // of warmup for final tuning, leaving fewer samples for mass adaptation.
  if (effective_metric == MassMatrixType::DENSE && n_params > 100) {
    term_buffer = 60;  // Reduced from 75 ? saves 15 iterations for mass adaptation
  }

  // Note: For p~24, first mass window (25 samples < 29 needed) fails,
  // but this is fine ? better to wait for more samples than set a poor
  // mass estimate early. The second window (100+ samples) gives good mass.

  // Adjust for short warmup
  if (n_warmup < init_buffer + term_buffer + init_window) {
    init_buffer = std::max(1, n_warmup / 5);
    term_buffer = std::max(1, n_warmup / 10);
    init_window = std::max(1, n_warmup - init_buffer - term_buffer);
  }

  // Compute mass adaptation window endpoints
  std::vector<int> mass_window_ends;
  {
    int adapt_end = n_warmup - term_buffer;
    if (adapt_end <= init_buffer) {
      // No room for mass adaptation windows
      mass_window_ends.push_back(std::max(1, adapt_end));
    } else {
      int next_end = init_buffer + init_window;
      int win_size = init_window;
      while (next_end < adapt_end) {
        int next_win = 2 * win_size;
        if (next_end + next_win > adapt_end) {
          // Extend current window to fill remaining space
          mass_window_ends.push_back(adapt_end);
          break;
        }
        mass_window_ends.push_back(next_end);
        win_size = next_win;
        next_end += win_size;
      }
      if (mass_window_ends.empty() || mass_window_ends.back() < adapt_end) {
        mass_window_ends.push_back(adapt_end);
      }
    }
  }
  int next_window_idx = 0;

  // Pre-allocate NUTS workspace (zero-allocation tree building)
  NUTSWorkspace nuts_ws;
  std::vector<double> _nuts_p;              // Momentum sampling buffer
  std::vector<double> _nuts_q_proposal;     // Persistent proposal (survives tree resets)
  std::vector<double> _nuts_grad_proposal;  // Persistent proposal gradient
  if (use_nuts) {
    nuts_ws.init(n_params, max_treedepth);
    nuts_ws.gradient_fn = resolve_gradient_fn(g_gradient_mode, data, layout);
    _nuts_p.resize(n_params);
    _nuts_q_proposal.resize(n_params);
    _nuts_grad_proposal.resize(n_params);
  }

  int sample_idx = 0;
  int n_accept = 0;
  int n_divergent = 0;
  // Adaptive NUTS?fixed-L switching: monitor early sampling for max treedepth
  int nuts_probe_window = std::min(20, n_sample);  // Check first 20 sampling iterations
  int nuts_probe_maxd = 0;  // Count of maxd hits in probe window
  bool nuts_probing = use_nuts && (L == 0);  // Only probe when using NUTS by default

  // SoftAbs divergence retry: compute local Hessian-based metric on divergent
  // trajectories and retry. Only active for BYM2/ICAR + dense mass (auto) or
  // when explicitly forced on.
  bool use_softabs_retry = false;
  if (riemannian == 1) {
    use_softabs_retry = true;
  } else if (riemannian == -1) {
    // Auto: enable for BYM2/ICAR with dense mass
    use_softabs_retry = (mass.type == MassMatrixType::DENSE &&
                         (data.spatial_type == SpatialType::BYM2 ||
                          data.spatial_type == SpatialType::ICAR));
  }
  // Disable if not using NUTS (SoftAbs retry only makes sense with NUTS)
  if (!use_nuts) use_softabs_retry = false;
  int softabs_retries = 0;
  int softabs_successes = 0;
  constexpr int SOFTABS_MAX_RETRIES = 3;  // Up to 3 retry attempts per divergence

  // Persistent SoftAbs metric (improvement #2): once computed, reuse for
  // all subsequent trajectories. Initialized at warmup?sampling transition
  // (improvement #4) or on first divergence, whichever comes first.
  bool softabs_metric_active = false;
  DenseMassMatrix softabs_persistent_mass;
  double softabs_persistent_eps = 0.0;
  if (use_softabs_retry) {
    softabs_persistent_mass.init(n_params, MassMatrixType::DENSE);
  }

  int warmup_total_leapfrog = 0;  // TEMP: diagnostic counter
  // Note: warmup divergences are normal for DIAG models and resolve via dual
  // averaging ? only final epsilon matters (checked at warmup end).

  if (verbose && layout.is_temporal_gp) {
    REprintf("  [MASS-WINDOWS] n_warmup=%d, windows=[", n_warmup);
    for (size_t w = 0; w < mass_window_ends.size(); w++) {
      REprintf("%d%s", mass_window_ends[w], w + 1 < mass_window_ends.size() ? "," : "");
    }
    REprintf("]\n");
  }
  for (int iter = 0; iter < n_iter; iter++) {
    bool is_warmup = (iter < n_warmup);
    // Check if we've reached a mass adaptation window boundary
    if (is_warmup && next_window_idx < (int)mass_window_ends.size() &&
        iter == mass_window_ends[next_window_idx]) {
      bool dense_covariance_set = false;  // Track if DENSE covariance (not just diagonal) succeeded this window
      // Dense mass matrix: try full covariance first
      // OAS shrinkage guarantees PD even when n < p, so we can lower the
      // threshold from n_params+5.  For large p the original threshold is
      // unreachable during warmup (e.g. p=159, need 164 but only get 125).
      // New threshold: min(p+5, max(50, p/2))  ? for p=159 this is 79.
      int dense_threshold = std::min(n_params + 5,
                                     std::max(50, n_params / 2));
      if (mass.type == MassMatrixType::DENSE && cov_stats.n >= dense_threshold) {
        auto cov = cov_stats.covariance();
        if (mass.update_from_covariance(cov.data(), cov_stats.n)) {
          use_mass_matrix = true;
          dense_covariance_set = true;
          if (verbose) {
            REprintf("  [DENSE] Window %d (iter %d): dense mass SET (n=%d, p=%d, OAS shrinkage=%.3f)\n",
                     next_window_idx, iter, cov_stats.n, n_params,
                     cov_stats.shrinkage_intensity);
          }
        } else {
          // Cholesky failed ? mass auto-degraded to DIAG, use diagonal stats
          if (verbose) {
            REprintf("  [DENSE] Window %d (iter %d): Cholesky FAILED (cov_stats.n=%d, p=%d)\n",
                     next_window_idx, iter, cov_stats.n, n_params);
          }
          if (mass_stats.n >= 10) {
            mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
            use_mass_matrix = true;
          }
        }
      } else if (mass.type == MassMatrixType::DENSE) {
        // Not enough samples for dense yet ? use diagonal as interim
        if (verbose) {
          REprintf("  [DENSE] Window %d (iter %d): not enough samples (cov_stats.n=%d, need=%d)\n",
                   next_window_idx, iter, cov_stats.n, dense_threshold);
        }
        if (mass_stats.n >= 10) {
          mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
          use_mass_matrix = true;
        }
      } else if (mass.type == MassMatrixType::BLOCK_DIAG) {
        // Block-diagonal: set diagonal for all params, then adapt block covariances
        if (mass_stats.n >= 10) {
          mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
          use_mass_matrix = true;
        }
        int n_adapted = 0;
        for (auto& blk : mass.blocks) {
          if (blk.update_from_welford()) {
            n_adapted++;
          }
        }
        if (verbose && n_adapted > 0) {
          REprintf("  [BLOCK_DIAG] Window %d (iter %d): %d/%d blocks adapted (n=%d)\n",
                   next_window_idx, iter, n_adapted, (int)mass.blocks.size(), mass_stats.n);
        }
        // Reset block Welford accumulators for next window
        for (auto& blk : mass.blocks) {
          blk.reset_welford();
        }
      } else if (mass_stats.n >= 10) {
        // Diagonal path
        mass.set_diagonal(mass_stats.inv_mass(), mass_stats.sqrt_mass());
        use_mass_matrix = true;
      }

      // Temporal GP NC: z ~ N(0,1) by construction ? optimal diag mass ? 1.0.
      // With limited warmup samples, noisy variance estimates for 20 z params
      // create unbalanced mass ? small epsilon. Fix z entries to 1.0 so the
      // step size is driven by the hyperparameters (beta, sigma2, phi) only.
      if (verbose && layout.is_temporal_gp) {
        REprintf("  [Z-DEBUG] Window %d (iter %d): use_mass=%d, tgp=%d, nc=%d, ts=%d, te=%d, mass_n=%d\n",
                 next_window_idx, iter, (int)use_mass_matrix,
                 (int)layout.is_temporal_gp, data.temporal_gp_parameterization,
                 layout.temporal_start, layout.temporal_end, mass_stats.n);
      }
      if (use_mass_matrix && layout.is_temporal_gp &&
          data.temporal_gp_parameterization == 1 &&
          layout.temporal_start >= 0 && layout.temporal_end > layout.temporal_start) {
        if (verbose) {
          REprintf("  [Z-FREEZE] Window %d: z mass before=[", next_window_idx);
          for (int j = layout.temporal_start; j < std::min(layout.temporal_end, layout.temporal_start + 5); j++) {
            REprintf("%.3f%s", mass.inv_mass_diag[j], j < layout.temporal_start + 4 ? "," : "");
          }
          REprintf("...], hyper=[");
          // Print beta and hyperparams
          for (int j = 0; j < std::min(4, layout.temporal_start); j++) {
            REprintf("%.3f%s", mass.inv_mass_diag[j], j < 3 ? "," : "");
          }
          REprintf("], sigma2=%.3f, phi=%.3f\n",
                   layout.log_sigma2_temporal_gp_idx >= 0 ? mass.inv_mass_diag[layout.log_sigma2_temporal_gp_idx] : -1.0,
                   layout.logit_phi_temporal_gp_idx >= 0 ? mass.inv_mass_diag[layout.logit_phi_temporal_gp_idx] : -1.0);
        }
        for (int j = layout.temporal_start; j < layout.temporal_end; j++) {
          mass.inv_mass_diag[j] = 1.0;
          mass.sqrt_mass_diag[j] = 1.0;
        }
      }

      mass_stats.reset();
      // For dense: only reset cov_stats when full covariance was successfully
      // computed THIS window. Otherwise keep accumulating across windows until
      // we have enough samples. This prevents the chicken-and-egg problem
      // where short windows never collect enough.
      // NOTE: We use dense_covariance_set (not mass.adapted) because
      // set_diagonal() also sets adapted=true, which would incorrectly
      // trigger a reset when we're still building up covariance samples.
      if (mass.type != MassMatrixType::DENSE || dense_covariance_set) {
        cov_stats.reset();
      }
      // Re-initialize step size with current mass matrix (A3)
      // Use dense-aware version when dense mass is adapted, so the step size
      // is calibrated for the rotated phase space (not just the diagonal).
      if (use_mass_matrix && mass.type == MassMatrixType::DENSE && mass.adapted) {
        epsilon = find_reasonable_epsilon_dense(q, data, layout, rng, mass);
      } else if (use_mass_matrix) {
        epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
      } else {
        epsilon = find_reasonable_epsilon(q, data, layout, rng);
      }
      da = DualAveraging(epsilon, n_params, target_boost);
      if (use_nuts) da.target_accept = nuts_target_accept;  // Preserve model-adaptive target

      next_window_idx++;
    }

    // L-BFGS: transition from L-BFGS to standard HMC at end of warmup
    // Extract diagonal mass matrix from learned curvature
    if (use_lbfgs && !lbfgs_warmup_done && iter == n_warmup - 1 && lbfgs_initialized) {
      // Use gamma from L-BFGS as uniform scaling for mass matrix
      // gamma = (s^T y) / (y^T y) approximates average inverse Hessian scaling
      double gamma = lbfgs_state.gamma;
      if (gamma > 0.01 && gamma < 100.0) {
        // Set inv_mass = gamma * I (larger gamma = larger variance = larger step in that direction)
        std::vector<double> inv_m(n_params, gamma);
        std::vector<double> sqrt_m(n_params, 1.0 / std::sqrt(gamma));
        mass.set_diagonal(inv_m, sqrt_m);
        use_mass_matrix = true;
      }
      lbfgs_warmup_done = true;
    }

    // =========================================================================
    // NUTS or fixed-trajectory HMC
    // =========================================================================
    double alpha = 0.0;
    bool divergent = false;
    int iter_n_leapfrog = L;
    int iter_treedepth = 0;

    if (use_nuts && !(use_lbfgs && !lbfgs_warmup_done)) {
      // -----------------------------------------------------------------
      // NUTS: No-U-Turn Sampler (optimized zero-allocation path)
      // -----------------------------------------------------------------

      auto& p = _nuts_p;

      // Step size jitter (improvement #5): ?20% random noise per trajectory
      // Prevents systematic step-size resonances that cause divergences.
      // Only during post-warmup sampling ? warmup needs stable epsilon for adaptation.
      double eps_iter = epsilon;
      if (!is_warmup) {
        double jitter = 1.0 + 0.2 * (2.0 * unif(rng) - 1.0);  // U[0.8, 1.2]
        eps_iter = epsilon * jitter;
      }

      // Sample momentum p ~ N(0, M) where M = C^{-1}
      mass.sample_momentum(p.data(), rng);

      // Initial Hamiltonian (pointer-based, no vector overhead)
      double H0 = nuts_compute_hamiltonian_fast(
        log_prob_current, p.data(), mass, n_params
      );
      double delta_max = 1000.0;

      // Load current state into workspace persistent slots
      nuts_ws.load_node(NUTSWorkspace::NODE_LEFT_SLOT,
                        q.data(), p.data(), current_grad.data(), log_prob_current);
      nuts_ws.load_node(NUTSWorkspace::NODE_RIGHT_SLOT,
                        q.data(), p.data(), current_grad.data(), log_prob_current);

      // Initialize persistent proposal buffers (pre-allocated, no per-iter malloc)
      auto& q_proposal_data = _nuts_q_proposal;
      auto& grad_proposal_data = _nuts_grad_proposal;
      std::memcpy(q_proposal_data.data(), q.data(), n_params * sizeof(double));
      std::memcpy(grad_proposal_data.data(), current_grad.data(), n_params * sizeof(double));
      double log_prob_proposal = log_prob_current;
      double sum_log_weight = 0.0;  // Relative weights: log(exp(H0 - H0)) = 0

      int total_leapfrog = 0;
      double sum_accept_prob = 0.0;
      divergent = false;

      // Generalized U-turn tracking at top level (Stan-style)
      // rho = total momentum sum. rho_bck/rho_fwd = halves for 3-juncture checks.
      // At each iteration the entire old trajectory becomes one half,
      // the new subtree becomes the other half (Stan's approach).
      // Uses pre-allocated workspace vectors (no per-iteration heap allocation).
      auto& rho = nuts_ws.iter_rho;
      std::memcpy(rho.data(), p.data(), n_params * sizeof(double));
      auto& rho_bck = nuts_ws.iter_rho_bck;
      auto& rho_fwd = nuts_ws.iter_rho_fwd;
      std::fill(rho_bck.begin(), rho_bck.end(), 0.0);
      std::fill(rho_fwd.begin(), rho_fwd.end(), 0.0);

      // p_sharp = M^{-1} * p at initial point ? full mass for correct U-turn geometry
      auto& p_sharp_init = nuts_ws.iter_p_sharp_init;
      mass.inv_mass_times_p(p.data(), p_sharp_init.data());

      // Boundary momenta: _end = far endpoint, _beg = origin-facing boundary
      // Stan naming: bck_end=bck_bck, bck_beg=bck_fwd, fwd_beg=fwd_bck, fwd_end=fwd_fwd
      auto& p_fwd_beg = nuts_ws.iter_p_fwd_beg;
      auto& p_fwd_end = nuts_ws.iter_p_fwd_end;
      auto& p_bck_beg = nuts_ws.iter_p_bck_beg;
      auto& p_bck_end = nuts_ws.iter_p_bck_end;
      std::memcpy(p_fwd_beg.data(), p.data(), n_params * sizeof(double));
      std::memcpy(p_fwd_end.data(), p.data(), n_params * sizeof(double));
      std::memcpy(p_bck_beg.data(), p.data(), n_params * sizeof(double));
      std::memcpy(p_bck_end.data(), p.data(), n_params * sizeof(double));
      auto& p_sharp_fwd_beg = nuts_ws.iter_p_sharp_fwd_beg;
      auto& p_sharp_fwd_end = nuts_ws.iter_p_sharp_fwd_end;
      auto& p_sharp_bck_beg = nuts_ws.iter_p_sharp_bck_beg;
      auto& p_sharp_bck_end = nuts_ws.iter_p_sharp_bck_end;
      std::memcpy(p_sharp_fwd_beg.data(), p_sharp_init.data(), n_params * sizeof(double));
      std::memcpy(p_sharp_fwd_end.data(), p_sharp_init.data(), n_params * sizeof(double));
      std::memcpy(p_sharp_bck_beg.data(), p_sharp_init.data(), n_params * sizeof(double));
      std::memcpy(p_sharp_bck_end.data(), p_sharp_init.data(), n_params * sizeof(double));

      // Build tree until U-turn or max depth
      for (int j = 0; j < max_treedepth; j++) {
        std::uniform_int_distribution<int> dir_dist(0, 1);
        int direction = 2 * dir_dist(rng) - 1;

        nuts_ws.reset_tree();

        int start_slot = nuts_ws.alloc_slot();
        if (start_slot < 0) break;
        if (direction == 1) {
          nuts_ws.copy_node(start_slot, NUTSWorkspace::NODE_RIGHT_SLOT);
        } else {
          nuts_ws.copy_node(start_slot, NUTSWorkspace::NODE_LEFT_SLOT);
        }

        // Stan: relabel halves before building subtree
        // Entire old trajectory becomes one half; new subtree is the other
        if (direction == 1) {
          // Extending forward: old trajectory ? backward half
          std::memcpy(rho_bck.data(), rho.data(), n_params * sizeof(double));
          std::memcpy(p_bck_beg.data(), p_fwd_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_bck_beg.data(), p_sharp_fwd_end.data(), n_params * sizeof(double));
        } else {
          // Extending backward: old trajectory ? forward half
          std::memcpy(rho_fwd.data(), rho.data(), n_params * sizeof(double));
          std::memcpy(p_fwd_beg.data(), p_bck_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_fwd_beg.data(), p_sharp_bck_end.data(), n_params * sizeof(double));
        }

        TreeStats subtree = build_tree_fast(
          nuts_ws, start_slot, direction, j,
          eps_iter, mass, H0, delta_max,
          data, layout, rng
        );

        total_leapfrog += subtree.n_leapfrog;
        sum_accept_prob += subtree.sum_accept_prob;

        if (subtree.divergent) {
          divergent = true;
        }

        if (!subtree.stop) {
          // Multinomial acceptance
          double log_sum_weight_subtree = subtree.sum_log_weight;
          double new_sum_log_weight = nuts_log_sum_exp(sum_log_weight, log_sum_weight_subtree);

          double accept_prob_subtree;
          if (log_sum_weight_subtree > new_sum_log_weight) {
            accept_prob_subtree = 1.0;
          } else {
            accept_prob_subtree = std::exp(log_sum_weight_subtree - new_sum_log_weight);
          }
          if (!std::isfinite(accept_prob_subtree)) accept_prob_subtree = 0.0;

          std::uniform_real_distribution<double> unif01(0.0, 1.0);
          if (unif01(rng) < accept_prob_subtree) {
            std::memcpy(q_proposal_data.data(), nuts_ws.q_at(subtree.proposal_slot),
                        n_params * sizeof(double));
            std::memcpy(grad_proposal_data.data(), nuts_ws.grad_at(subtree.proposal_slot),
                        n_params * sizeof(double));
            log_prob_proposal = subtree.log_prob_proposal;
          }

          sum_log_weight = new_sum_log_weight;
        }

        // Update direction endpoints and rho half from subtree
        // Use memcpy instead of std::move to preserve pre-allocated buffers
        if (direction == 1) {
          nuts_ws.copy_node(NUTSWorkspace::NODE_RIGHT_SLOT, subtree.right_slot);
          std::memcpy(rho_fwd.data(), subtree.rho.data(), n_params * sizeof(double));
          std::memcpy(p_fwd_beg.data(), subtree.p_beg.data(), n_params * sizeof(double));
          std::memcpy(p_fwd_end.data(), subtree.p_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_fwd_beg.data(), subtree.p_sharp_beg.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_fwd_end.data(), subtree.p_sharp_end.data(), n_params * sizeof(double));
        } else {
          nuts_ws.copy_node(NUTSWorkspace::NODE_LEFT_SLOT, subtree.left_slot);
          std::memcpy(rho_bck.data(), subtree.rho.data(), n_params * sizeof(double));
          std::memcpy(p_bck_beg.data(), subtree.p_beg.data(), n_params * sizeof(double));
          std::memcpy(p_bck_end.data(), subtree.p_end.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_bck_beg.data(), subtree.p_sharp_beg.data(), n_params * sizeof(double));
          std::memcpy(p_sharp_bck_end.data(), subtree.p_sharp_end.data(), n_params * sizeof(double));
        }

        // Combine rho = rho_bck + rho_fwd
        for (int i = 0; i < n_params; i++) {
          rho[i] = rho_bck[i] + rho_fwd[i];
        }

        iter_treedepth = j + 1;

        // Generalized U-turn check at top level (3 junctures)
        if (subtree.stop) break;

        // Check 1: Full trajectory ? far endpoints vs total rho
        bool persist = compute_criterion(p_sharp_bck_end.data(), p_sharp_fwd_end.data(),
                                         rho.data(), n_params);

        // Check 2: Backward half + seam from forward (rho = rho_bck + p_fwd_beg)
        auto& rho_seam = nuts_ws.iter_rho_seam;
        for (int i = 0; i < n_params; i++) {
          rho_seam[i] = rho_bck[i] + p_fwd_beg[i];
        }
        persist &= compute_criterion(p_sharp_bck_end.data(), p_sharp_fwd_beg.data(),
                                      rho_seam.data(), n_params);

        // Check 3: Seam from backward + forward half (rho = rho_fwd + p_bck_beg)
        for (int i = 0; i < n_params; i++) {
          rho_seam[i] = rho_fwd[i] + p_bck_beg[i];
        }
        persist &= compute_criterion(p_sharp_bck_beg.data(), p_sharp_fwd_end.data(),
                                      rho_seam.data(), n_params);

        if (!persist) break;
      }

      // SoftAbs divergence retry (improvements #1, #2): if trajectory diverged,
      // compute local Hessian-based metric and retry up to SOFTABS_MAX_RETRIES
      // times, halving step size each attempt. On first successful metric
      // computation, persist it for all subsequent trajectories.
      if (divergent && !is_warmup && use_softabs_retry) {
        softabs_retries++;

        // Compute fresh Hessian at current position (p+1 gradient evals)
        std::vector<double> hessian_buf;
        compute_hessian_finite_diff(q, data, layout, hessian_buf);
        for (auto& v : hessian_buf) v = -v;  // Negate: -H = curvature

        std::vector<double> G_inv_buf, L_G_inv_buf;
        bool metric_ok = compute_softabs_metric(
          hessian_buf, n_params, 1.0, G_inv_buf, L_G_inv_buf
        );

        if (metric_ok) {
          // Update persistent SoftAbs metric for retry use only (improvement #2).
          // Do NOT override main mass/epsilon ? warmup-adapted values work better
          // for general trajectories. SoftAbs is rescue-only.
          softabs_persistent_mass.set_from_metric(G_inv_buf, L_G_inv_buf);
          double eps_base = find_reasonable_epsilon_dense(
            q, data, layout, rng, softabs_persistent_mass);
          softabs_persistent_eps = eps_base;
          softabs_metric_active = true;

          // Multiple retry attempts (improvement #1): try up to 3 times
          // with halving step size each attempt
          for (int retry_attempt = 0; retry_attempt < SOFTABS_MAX_RETRIES; retry_attempt++) {
            double eps_retry = eps_base * std::pow(0.5, retry_attempt);

            // Sample new momentum and re-run NUTS trajectory
            softabs_persistent_mass.sample_momentum(p.data(), rng);
            double H0_retry = nuts_compute_hamiltonian_fast(
              log_prob_current, p.data(), softabs_persistent_mass, n_params
            );

            // Load current state into workspace
            nuts_ws.load_node(NUTSWorkspace::NODE_LEFT_SLOT,
                              q.data(), p.data(), current_grad.data(), log_prob_current);
            nuts_ws.load_node(NUTSWorkspace::NODE_RIGHT_SLOT,
                              q.data(), p.data(), current_grad.data(), log_prob_current);

            std::memcpy(q_proposal_data.data(), q.data(), n_params * sizeof(double));
            std::memcpy(grad_proposal_data.data(), current_grad.data(), n_params * sizeof(double));
            log_prob_proposal = log_prob_current;
            sum_log_weight = 0.0;
            total_leapfrog = 0;
            sum_accept_prob = 0.0;
            bool retry_divergent = false;

            // Full NUTS tree with SoftAbs metric + 3-juncture U-turn
            std::memcpy(rho.data(), p.data(), n_params * sizeof(double));
            std::fill(rho_bck.begin(), rho_bck.end(), 0.0);
            std::fill(rho_fwd.begin(), rho_fwd.end(), 0.0);
            softabs_persistent_mass.inv_mass_times_p(p.data(), p_sharp_init.data());
            std::copy(p.begin(), p.end(), p_fwd_beg.begin());
            std::copy(p.begin(), p.end(), p_fwd_end.begin());
            std::copy(p.begin(), p.end(), p_bck_beg.begin());
            std::copy(p.begin(), p.end(), p_bck_end.begin());
            std::copy(p_sharp_init.begin(), p_sharp_init.end(), p_sharp_fwd_beg.begin());
            std::copy(p_sharp_init.begin(), p_sharp_init.end(), p_sharp_fwd_end.begin());
            std::copy(p_sharp_init.begin(), p_sharp_init.end(), p_sharp_bck_beg.begin());
            std::copy(p_sharp_init.begin(), p_sharp_init.end(), p_sharp_bck_end.begin());

            int retry_treedepth = 0;
            for (int j = 0; j < max_treedepth; j++) {
              std::uniform_int_distribution<int> dir_dist(0, 1);
              int direction = 2 * dir_dist(rng) - 1;

              nuts_ws.reset_tree();
              int start_slot = nuts_ws.alloc_slot();
              if (start_slot < 0) break;
              if (direction == 1) {
                nuts_ws.copy_node(start_slot, NUTSWorkspace::NODE_RIGHT_SLOT);
              } else {
                nuts_ws.copy_node(start_slot, NUTSWorkspace::NODE_LEFT_SLOT);
              }

              if (direction == 1) {
                std::memcpy(rho_bck.data(), rho.data(), n_params * sizeof(double));
                std::memcpy(p_bck_beg.data(), p_fwd_end.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_bck_beg.data(), p_sharp_fwd_end.data(), n_params * sizeof(double));
              } else {
                std::memcpy(rho_fwd.data(), rho.data(), n_params * sizeof(double));
                std::memcpy(p_fwd_beg.data(), p_bck_end.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_fwd_beg.data(), p_sharp_bck_end.data(), n_params * sizeof(double));
              }

              TreeStats subtree = build_tree_fast(
                nuts_ws, start_slot, direction, j,
                eps_retry, softabs_persistent_mass, H0_retry, 1000.0,
                data, layout, rng
              );

              total_leapfrog += subtree.n_leapfrog;
              sum_accept_prob += subtree.sum_accept_prob;
              if (subtree.divergent) retry_divergent = true;

              if (!subtree.stop) {
                double log_sum_weight_subtree = subtree.sum_log_weight;
                double new_sum_log_weight = nuts_log_sum_exp(sum_log_weight, log_sum_weight_subtree);
                double accept_prob_subtree;
                if (log_sum_weight_subtree > new_sum_log_weight) {
                  accept_prob_subtree = 1.0;
                } else {
                  accept_prob_subtree = std::exp(log_sum_weight_subtree - new_sum_log_weight);
                }
                if (!std::isfinite(accept_prob_subtree)) accept_prob_subtree = 0.0;

                std::uniform_real_distribution<double> unif01(0.0, 1.0);
                if (unif01(rng) < accept_prob_subtree) {
                  std::memcpy(q_proposal_data.data(), nuts_ws.q_at(subtree.proposal_slot),
                              n_params * sizeof(double));
                  std::memcpy(grad_proposal_data.data(), nuts_ws.grad_at(subtree.proposal_slot),
                              n_params * sizeof(double));
                  log_prob_proposal = subtree.log_prob_proposal;
                }
                sum_log_weight = new_sum_log_weight;
              }

              if (direction == 1) {
                nuts_ws.copy_node(NUTSWorkspace::NODE_RIGHT_SLOT, subtree.right_slot);
                std::memcpy(rho_fwd.data(), subtree.rho.data(), n_params * sizeof(double));
                std::memcpy(p_fwd_beg.data(), subtree.p_beg.data(), n_params * sizeof(double));
                std::memcpy(p_fwd_end.data(), subtree.p_end.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_fwd_beg.data(), subtree.p_sharp_beg.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_fwd_end.data(), subtree.p_sharp_end.data(), n_params * sizeof(double));
              } else {
                nuts_ws.copy_node(NUTSWorkspace::NODE_LEFT_SLOT, subtree.left_slot);
                std::memcpy(rho_bck.data(), subtree.rho.data(), n_params * sizeof(double));
                std::memcpy(p_bck_beg.data(), subtree.p_beg.data(), n_params * sizeof(double));
                std::memcpy(p_bck_end.data(), subtree.p_end.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_bck_beg.data(), subtree.p_sharp_beg.data(), n_params * sizeof(double));
                std::memcpy(p_sharp_bck_end.data(), subtree.p_sharp_end.data(), n_params * sizeof(double));
              }

              for (int i = 0; i < n_params; i++) {
                rho[i] = rho_bck[i] + rho_fwd[i];
              }
              retry_treedepth = j + 1;

              if (subtree.stop) break;

              bool persist = compute_criterion(p_sharp_bck_end.data(), p_sharp_fwd_end.data(),
                                               rho.data(), n_params);
              auto& rho_seam_retry = nuts_ws.iter_rho_seam;
              for (int i = 0; i < n_params; i++) {
                rho_seam_retry[i] = rho_bck[i] + p_fwd_beg[i];
              }
              persist &= compute_criterion(p_sharp_bck_end.data(), p_sharp_fwd_beg.data(),
                                            rho_seam_retry.data(), n_params);
              for (int i = 0; i < n_params; i++) {
                rho_seam_retry[i] = rho_fwd[i] + p_bck_beg[i];
              }
              persist &= compute_criterion(p_sharp_bck_beg.data(), p_sharp_fwd_end.data(),
                                            rho_seam_retry.data(), n_params);
              if (!persist) break;
            }

            // If retry succeeded (no divergence), accept and stop retrying
            if (!retry_divergent) {
              divergent = false;
              iter_treedepth = retry_treedepth;
              softabs_successes++;
              alpha = (total_leapfrog > 0) ? (sum_accept_prob / total_leapfrog) : 0.0;
              iter_n_leapfrog = total_leapfrog;
              break;  // Success ? stop retry loop
            }
            // Otherwise: try again with halved step size (next iteration)
          }  // end retry_attempt loop

          // If all retries failed, update stats from last attempt
          if (divergent) {
            alpha = (total_leapfrog > 0) ? (sum_accept_prob / total_leapfrog) : 0.0;
            iter_n_leapfrog = total_leapfrog;
          }
        }
        // else: metric computation failed, keep original divergent result
      }

      // Accept proposal: copy from persistent proposal buffers (memcpy, no alloc)
      std::memcpy(q.data(), q_proposal_data.data(), n_params * sizeof(double));
      std::memcpy(current_grad.data(), grad_proposal_data.data(), n_params * sizeof(double));
      log_prob_current = log_prob_proposal;
      n_accept++;

      // Average acceptance statistic for dual averaging
      alpha = (total_leapfrog > 0) ? (sum_accept_prob / total_leapfrog) : 0.0;
      iter_n_leapfrog = total_leapfrog;

      if (divergent) n_divergent++;
      if (iter_treedepth >= max_treedepth) result.n_max_treedepth++;

      // Adaptation during warmup
      if (is_warmup) {
        epsilon = da.update(alpha);

        // Early detection of catastrophic dense mass during terminal buffer.
        // Normal epsilon with dense mass is 0.1-0.5. If it exceeds 2.0, the mass
        // matrix eigenvalues are pathological. Fall back to DIAG immediately so
        // the remaining terminal buffer iterations (~48) properly adapt epsilon.
        // inv_mass_diag is always kept in sync with the dense diagonal (line 62).
        if (iter >= n_warmup - term_buffer && iter < n_warmup - 1 &&
            mass.type == MassMatrixType::DENSE && mass.adapted && epsilon > 2.0) {
          if (verbose) {
            REprintf("  [DENSE] WARNING at iter %d: epsilon=%.4f (catastrophic). "
                     "Falling back to DIAG mass.\n", iter, epsilon);
          }
          mass.type = MassMatrixType::DIAG;
          // inv_mass_diag already populated from dense diagonal (update_from_covariance)
          epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
          da = DualAveraging(epsilon, n_params, target_boost);
          if (use_nuts) da.target_accept = nuts_target_accept;
        }

        // DIAG?DENSE recovery is checked at warmup end (after da.final_epsilon)
        // rather than during warmup ? warmup divergences are normal for DIAG models
        // and resolve via dual averaging. Only catastrophic final epsilon matters.

        if (iter >= init_buffer && iter < n_warmup - term_buffer) {
          mass_stats.update(q);
          if (mass.type == MassMatrixType::DENSE) {
            cov_stats.update(q);
          }
          if (mass.type == MassMatrixType::BLOCK_DIAG) {
            for (auto& blk : mass.blocks) {
              blk.welford_update(q.data());
            }
          }
        }
        if (iter == n_warmup - 1) {
          epsilon = da.final_epsilon();

          // DIAG?BLOCK_DIAG?DENSE recovery at warmup end: if AUTO selected DIAG but the
          // final adapted epsilon is catastrophic (>2.0), DIAG can't capture the
          // posterior geometry. Try BLOCK_DIAG first if block_specs available,
          // otherwise fall back to DENSE with identity mass.
          if (auto_selected_diag && mass.type == MassMatrixType::DIAG &&
              epsilon > 2.0) {
            if (!block_specs.empty()) {
              // Try BLOCK_DIAG recovery first (cheaper than full DENSE)
              if (verbose) {
                REprintf("  [DIAG->BLOCK_DIAG] Warmup end: final epsilon=%.4f (catastrophic). "
                         "Switching to BLOCK_DIAG (adapted=false).\n", epsilon);
              }
              mass.init_block_diag(n_params, block_specs);
              effective_metric = MassMatrixType::BLOCK_DIAG;
              auto_selected_diag = false;
              epsilon = find_reasonable_epsilon(q, data, layout, rng);
              da = DualAveraging(epsilon, n_params, target_boost);
              if (use_nuts) da.target_accept = nuts_target_accept;
              epsilon = da.final_epsilon();
            } else if (n_params <= DENSE_MAX_PARAMS) {
              if (verbose) {
                REprintf("  [DIAG->DENSE] Warmup end: final epsilon=%.4f (catastrophic). "
                         "Switching to DENSE identity mass.\n", epsilon);
              }
              mass.init(n_params, MassMatrixType::DENSE);
              effective_metric = MassMatrixType::DENSE;
              auto_selected_diag = false;
              epsilon = find_reasonable_epsilon(q, data, layout, rng);
              da = DualAveraging(epsilon, n_params, target_boost);
              if (use_nuts) da.target_accept = nuts_target_accept;
              epsilon = da.final_epsilon();
            }
          }

          // BLOCK_DIAG?DIAG fallback: if epsilon still catastrophic after BLOCK_DIAG
          if (epsilon > 2.0 && mass.type == MassMatrixType::BLOCK_DIAG) {
            if (verbose) {
              REprintf("  [BLOCK_DIAG->DIAG] WARNING: epsilon=%.4f still catastrophic. "
                       "Falling back to DIAG.\n", epsilon);
            }
            mass.init(n_params, MassMatrixType::DIAG);
            effective_metric = MassMatrixType::DIAG;
            epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
            da = DualAveraging(epsilon, n_params, target_boost);
            if (use_nuts) da.target_accept = nuts_target_accept;
            epsilon = da.final_epsilon();
          }

          // Final safety net: if epsilon is still > 1.0 with dense mass after
          // the full terminal buffer, fall back to DIAG. This catches cases
          // where the catastrophe develops slowly.
          if (epsilon > 1.0 && mass.type == MassMatrixType::DENSE && mass.adapted) {
            if (verbose) {
              REprintf("  [DENSE] WARNING: epsilon=%.4f after warmup (catastrophic). "
                       "Falling back to DIAG mass.\n", epsilon);
            }
            mass.type = MassMatrixType::DIAG;
            epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
            da = DualAveraging(epsilon, n_params, target_boost);
            if (use_nuts) da.target_accept = nuts_target_accept;
            epsilon = da.final_epsilon();
          }

          // Precision-informed diagonal mass for ST_IV at warmup end.
          // Build Q_post = tau*(Q_s?Q_t) + diag(h_lik), factorize, extract diag(Q^{-1}).
          if (mass.sparse_gmrf.active && !mass.sparse_gmrf.factorized) {
            int st_S = data.spatiotemporal_data.n_spatial;
            int st_T = data.spatiotemporal_data.n_times;
            int ST = st_S * st_T;
            double tau_st = std::exp(q[layout.log_tau_st_idx]);

            // Compute likelihood Hessian diagonal for each ST cell
            std::vector<double> h_lik(ST, 0.0);
            for (int i = 0; i < data.N; i++) {
              if (data.spatiotemporal_data.st_flat[i] <= 0) continue;
              int k = data.spatiotemporal_data.st_flat[i] - 1;
              double eta_i = 0.0;
              for (int p2 = 0; p2 < data.legacy.p_num; p2++)
                eta_i += data.legacy.X_num_flat[static_cast<size_t>(i) * data.legacy.p_num + p2] * q[layout.legacy.beta_num_start + p2];
              if (layout.has_re && data.re_group[i] > 0)
                eta_i += re_value_for_eta(&q[layout.re_start], data.re_group[i] - 1,
                                           std::exp(q[layout.log_sigma_re_idx]), data.re_parameterization);
              if (layout.has_spatial && data.spatial_group[i] > 0)
                eta_i += q[layout.spatial_start + data.spatial_group[i] - 1];
              if (layout.has_temporal && !data.temporal_time_idx.empty() &&
                  i < (int)data.temporal_time_idx.size() && data.temporal_time_idx[i] > 0) {
                int t_idx = data.temporal_time_idx[i] - 1;
                int g_idx = data.temporal_group_idx[i] - 1;
                int t_base = g_idx * data.n_times + t_idx;
                if (t_base >= 0 && t_base < (int)q.size()) eta_i += q[layout.temporal_start + t_base];
              }
              eta_i += q[layout.st_delta_start + k];
              double mu_i = std::exp(eta_i);
              double h_i = 0.0;
              if (data.legacy.model_type == ModelType::POISSON_GAMMA) {
                h_i = mu_i;
              } else if (data.legacy.model_type == ModelType::BINOMIAL) {
                double p_i = 1.0 / (1.0 + std::exp(-eta_i));
                h_i = data.legacy.y_denom[i] * p_i * (1.0 - p_i);
              } else if (data.legacy.model_type == ModelType::NEGBIN_NEGBIN ||
                         data.legacy.model_type == ModelType::NEGBIN_GAMMA) {
                double phi = std::exp(q[layout.legacy.log_phi_num_idx]);
                h_i = mu_i / (1.0 + mu_i / phi);
              } else {
                h_i = mu_i;
              }
              if (data.spatiotemporal_data.shared) h_i *= 2.0;
              h_lik[k] += std::max(h_i, 1e-6);
            }

            mass.sparse_gmrf.build_and_factorize(
                data.spatiotemporal_data.adj_row_ptr,
                data.spatiotemporal_data.adj_col_idx,
                data.spatiotemporal_data.temporal_type,
                data.spatiotemporal_data.temporal_cyclic,
                tau_st, h_lik.data(), 0.001
            );

            if (mass.sparse_gmrf.factorized) {
              // Extract diag(Q^{-1}) and set diagonal mass for ST params
              int n_set = 0;
              double sum_var = 0.0;
              for (int k = 0; k < ST; k++) {
                Eigen::VectorXd ek = Eigen::VectorXd::Zero(ST);
                ek[k] = 1.0;
                Eigen::VectorXd col_k = mass.sparse_gmrf.llt.solve(ek);
                double var_k = col_k[k];
                if (var_k > 1e-10 && var_k < 100.0) {
                  mass.inv_mass_diag[layout.st_delta_start + k] = var_k;
                  n_set++;
                  sum_var += var_k;
                }
              }
              // Deactivate sparse GMRF ? diagonal mass is now informed
              mass.sparse_gmrf.active = false;
              // Recompute epsilon with new mass
              epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
              da = DualAveraging(epsilon, n_params, target_boost);
              if (use_nuts) da.target_accept = nuts_target_accept;
              epsilon = da.final_epsilon();
              if (verbose) {
                REprintf("  [SPARSE_GMRF] Diagonal mass set for %d/%d ST params, avg_var=%.6f, tau=%.4f, new epsilon=%.6f\n",
                         n_set, ST, n_set > 0 ? sum_var / n_set : 0.0, tau_st, epsilon);
              }
            } else if (verbose) {
              REprintf("  [SPARSE_GMRF] WARNING: Cholesky failed, keeping adapted diagonal mass\n");
            }
          }

          if (verbose) {
            REprintf("  [METRIC] Warmup done: epsilon=%.6f, mass.type=%s, mass.adapted=%d\n",
                     epsilon, metric_name(mass.type), (int)mass.adapted);
          }
          // Proactive SoftAbs at warmup?sampling transition (improvement #4):
          // Pre-compute SoftAbs metric so it's ready for retry attempts.
          // Do NOT override main mass/epsilon ? warmup-adapted values are better
          // for general sampling. SoftAbs is only used as rescue on divergences.
          if (use_softabs_retry && !softabs_metric_active) {
            std::vector<double> hessian_warmup_end;
            compute_hessian_finite_diff(q, data, layout, hessian_warmup_end);
            for (auto& v : hessian_warmup_end) v = -v;

            std::vector<double> G_inv_init, L_G_inv_init;
            if (compute_softabs_metric(hessian_warmup_end, n_params, 1.0,
                                       G_inv_init, L_G_inv_init)) {
              softabs_persistent_mass.set_from_metric(G_inv_init, L_G_inv_init);
              softabs_persistent_eps = find_reasonable_epsilon_dense(
                q, data, layout, rng, softabs_persistent_mass);
              softabs_metric_active = true;
              // Note: main mass and epsilon are NOT overridden
              if (verbose) {
                REprintf("  [SoftAbs] Proactive metric pre-computed at warmup end: retry_eps=%.6f\n",
                         softabs_persistent_eps);
              }
            }
          }
        }
        // Print tree depth for last 10 warmup iterations
        if (verbose && iter >= n_warmup - 10) {
          REprintf("  [%s] warmup iter %d: treedepth=%d, epsilon=%.6f\n",
                   metric_name(mass.type),
                   iter, iter_treedepth, epsilon);
        }
      }

      // Adaptive NUTS probe: warn if most early iterations hit max treedepth
      // (Stan's approach: warn but keep NUTS running ? truncated NUTS picks
      // from up to 2^depth candidates, far better than HMC(L=10) with tiny epsilon)
      if (nuts_probing && !is_warmup && sample_idx < nuts_probe_window) {
        if (iter_treedepth >= max_treedepth) nuts_probe_maxd++;
        if (sample_idx == nuts_probe_window - 1) {
          nuts_probing = false;  // Probe window complete
          if (nuts_probe_maxd >= (nuts_probe_window * 8 + 9) / 10) {
            result.n_max_treedepth += 0;  // Already counted above
            if (verbose) {
              REprintf("  [NUTS] %d/%d initial sampling iterations hit max treedepth (%d). "
                       "Consider increasing max_treedepth or reparameterizing.\n",
                       nuts_probe_maxd, nuts_probe_window, max_treedepth);
            }
          }
        }
      }
    } else {
      // -----------------------------------------------------------------
      // Fixed-trajectory HMC (original code)
      // -----------------------------------------------------------------

      // Sample momentum and compute kinetic energy
      std::vector<double> p(n_params);
      double kinetic_current = 0.0;
      double H_current;

      if (use_lbfgs && lbfgs_initialized && !lbfgs_warmup_done && lbfgs_state.d == n_params) {
        // L-BFGS: Sample p ~ N(0, B) where B ? 1/gamma * I (warmup only)
        std::vector<double> sqrt_diag = lbfgs_state.get_sqrt_B_diag();
        if ((int)sqrt_diag.size() == n_params) {
          for (int i = 0; i < n_params; i++) {
            p[i] = normal(rng) * sqrt_diag[i];
          }
          kinetic_current = lbfgs_state.kinetic_energy(p);
          H_current = -log_prob_current + kinetic_current;
        } else {
          mass.sample_momentum(p.data(), rng);
          kinetic_current = mass.kinetic_energy(p.data());
          H_current = -log_prob_current + kinetic_current;
        }
      } else {
        mass.sample_momentum(p.data(), rng);
        kinetic_current = mass.kinetic_energy(p.data());
        H_current = -log_prob_current + kinetic_current;
      }

      // Leapfrog integration
      std::vector<double> q_prop = q;
      std::vector<double> p_prop = p;

      // Determine effective L for this iteration
      int L_eff = L;
      if (use_nuts && use_lbfgs && !lbfgs_warmup_done) {
        // During L-BFGS warmup with NUTS mode, use fixed L=20
        L_eff = 20;
      }

      if (use_lbfgs && lbfgs_initialized && !lbfgs_warmup_done && lbfgs_state.d == n_params) {
        // L-BFGS leapfrog
        std::vector<double> grad(n_params);
        compute_gradient(q_prop, data, layout, grad);

        for (int l = 0; l < L_eff; l++) {
          for (int i = 0; i < n_params; i++) {
            p_prop[i] += 0.5 * epsilon * grad[i];
          }
          std::vector<double> Hp(n_params);
          lbfgs_state.multiply_H(p_prop, Hp);
          for (int i = 0; i < n_params; i++) {
            q_prop[i] += epsilon * Hp[i];
            if (!std::isfinite(q_prop[i])) {
              divergent = true;
              break;
            }
          }
          if (divergent) break;
          compute_gradient(q_prop, data, layout, grad);
          for (int i = 0; i < n_params; i++) {
            p_prop[i] += 0.5 * epsilon * grad[i];
          }
          for (int i = 0; i < n_params; i++) {
            if (!std::isfinite(p_prop[i]) || std::abs(p_prop[i]) > 1e10) {
              divergent = true;
              break;
            }
          }
          if (divergent) break;
        }
      } else {
        // Standard leapfrog
        for (int l = 0; l < L_eff; l++) {
          LeapfrogResult lf;
          if (use_mass_matrix) {
            lf = leapfrog_step(q_prop, p_prop, epsilon, data, layout, mass.inv_mass_diag.data());
          } else {
            lf = leapfrog_step(q_prop, p_prop, epsilon, data, layout);
          }
          q_prop = lf.q;
          p_prop = lf.p;
          if (lf.divergent) {
            divergent = true;
            break;
          }
        }
      }

      // Compute proposed Hamiltonian (use same log-post as gradient mode)
      double log_prob_prop;
      const bool is_generic = data.n_processes > 0 && data.likelihood_spec != nullptr;
      if (!is_generic &&
          (g_gradient_mode == GradientMode::AUTODIFF_ARENA ||
          g_gradient_mode == GradientMode::AUTODIFF_FWD ||
          g_gradient_mode == GradientMode::AUTODIFF_TAPE)) {
        log_prob_prop = tulpa::compute_log_post_impl(q_prop, data, layout);
      } else {
        log_prob_prop = compute_log_post(q_prop, data, layout);
      }
      double kinetic_prop = 0.0;

      if (use_lbfgs && lbfgs_initialized && !lbfgs_warmup_done && lbfgs_state.d == n_params) {
        kinetic_prop = lbfgs_state.kinetic_energy(p_prop);
      } else {
        kinetic_prop = mass.kinetic_energy(p_prop.data());
      }
      double H_prop = -log_prob_prop + kinetic_prop;

      // Metropolis accept/reject
      alpha = std::min(1.0, std::exp(H_current - H_prop));
      if (!std::isfinite(alpha)) alpha = 0.0;

      std::uniform_real_distribution<double> unif01(0.0, 1.0);
      bool accepted = (unif01(rng) < alpha) && !divergent;
      if (accepted) {
        q = q_prop;
        log_prob_current = log_prob_prop;
        n_accept++;
        // Update cached gradient for transition to NUTS after L-BFGS warmup
        if (use_nuts) {
          compute_gradient(q, data, layout, current_grad);
        }
      }
      if (divergent) n_divergent++;

      // Adaptation during warmup
      if (is_warmup) {
        epsilon = da.update(alpha);
        // Only collect mass stats during mass adaptation phase (A5)
        if (iter >= init_buffer && iter < n_warmup - term_buffer) {
          mass_stats.update(q);
          if (mass.type == MassMatrixType::DENSE) {
            cov_stats.update(q);
          }
        }
        // On last warmup iteration, use averaged step size for sampling (A1)
        if (iter == n_warmup - 1) {
          epsilon = da.final_epsilon();
        }
      }

      // L-BFGS update: collect (s, y) pairs from accepted samples (warmup only)
      if (use_lbfgs && !lbfgs_warmup_done) {
        std::vector<double> grad_current(n_params);
        compute_gradient(q, data, layout, grad_current);

        if (!lbfgs_initialized) {
          q_prev = q;
          grad_prev = grad_current;
          lbfgs_initialized = true;
        } else if (accepted) {
          std::vector<double> s(n_params), y(n_params);
          for (int i = 0; i < n_params; i++) {
            s[i] = q[i] - q_prev[i];
            y[i] = grad_current[i] - grad_prev[i];
          }
          lbfgs_state.add_pair(s, y);
          q_prev = q;
          grad_prev = grad_current;
        }
      }

      iter_n_leapfrog = L_eff;
    }  // end fixed-trajectory HMC

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

    // Note: verbose output disabled in parallel - not thread-safe
    // Progress will be reported after parallel region
  }

  result.epsilon = da.final_epsilon();

  // Diagnostic stats - only when verbose
  if (verbose) {
    int sampling_total_lf = 0;
    for (int i = 0; i < result.n_sample; i++) sampling_total_lf += result.n_leapfrog[i];
    REprintf("  [STATS] Chain %d: metric=%s, adapted=%d, warmup_LF=%d, sampling_LF=%d, total_LF=%d, epsilon=%.6f\n",
             chain_id + 1,
             metric_name(mass.type),
             (int)mass.adapted,
             warmup_total_leapfrog, sampling_total_lf,
             warmup_total_leapfrog + sampling_total_lf, result.epsilon);
  }

  if (verbose && (softabs_retries > 0 || softabs_metric_active)) {
    REprintf("  [SoftAbs] Chain %d: metric=%s, %d divergent retried (up to %d attempts), %d resolved (%d remained)\n",
             chain_id + 1,
             softabs_metric_active ? "active" : "inactive",
             softabs_retries, SOFTABS_MAX_RETRIES, softabs_successes,
             softabs_retries - softabs_successes);
  }

  return result;
}

// R wrapper version - for single chain or non-parallel use
HMCResult run_hmc_chain(
    const std::vector<double>& q_init,
    const ModelData& data,
    const ParamLayout& layout,
    int n_iter,
    int n_warmup,
    int L,
    int chain_id,
    unsigned int seed,
    bool verbose,
    int max_treedepth,
    MassMatrixType metric_type,
    double adapt_delta,
    int riemannian
) {
  // Runtime gradient check: compare active gradient function against numerical
  if (g_gradient_mode != GradientMode::NUMERICAL) {
    bool grad_ok = verify_gradient_runtime(q_init, data, layout, 1e-4);
    if (!grad_ok) {
      g_gradient_mode = GradientMode::NUMERICAL;
      Rcpp::warning(
        "Gradient mismatch detected: active gradient function disagrees with "
        "numerical gradients (max rel diff > 1e-4). Falling back to numerical "
        "gradients (mode='N'). This is slower but correct. Please report this "
        "as a bug at https://github.com/gcol33/numdenom/issues"
      );
    }
  }

  // Run C++ version - pass verbose through for debugging
  HMCResultCpp cpp_result = run_hmc_chain_cpp(
    q_init, data, layout, n_iter, n_warmup, L, chain_id, seed, verbose, max_treedepth, metric_type, adapt_delta, riemannian
  );

  // Convert to R result
  int n_params = q_init.size();
  HMCResult result = cpp_to_r_result(cpp_result, n_params);

  if (verbose) {
    int n_div = 0;
    for (int i = 0; i < cpp_result.n_sample; i++) {
      n_div += cpp_result.divergent[i];
    }
    Rcpp::Rcerr << "Chain " << (chain_id + 1) << " complete. "
                << "Divergent: " << n_div << std::endl;
  }

  return result;
}

#endif  // TULPA_HMC_NUTS_CHAIN_H
