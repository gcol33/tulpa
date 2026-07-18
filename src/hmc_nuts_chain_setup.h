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
    // After Phase D every caller is generic LikelihoodSpec
    // and compute_log_post forwards to compute_log_post_generic_spec_double,
    // so the legacy autodiff-vs-H log-post split is no longer needed.
    log_prob_current = compute_log_post(q, data, layout);
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

  // Caller-supplied inv-mass diagonal (e.g. from a Laplace approximation).
  // Overrides any structural warm-start above. Subsequent mass adaptation
  // still runs and refines this starting point.
  // A wrong-length init is silently ignored here; this fragment runs inside
  // the parallel-chain region, where REprintf (R API) is not thread-safe.
  // The verbose branch below reports the applied metric on serial runs.
  bool caller_inv_metric = (!inv_metric_init.empty()
                            && (int)inv_metric_init.size() == n_params);
  if (verbose && !inv_metric_init.empty() && !caller_inv_metric) {
    REprintf("  [WARMSTART] inv_metric_init length %d != n_params %d, ignoring\n",
             (int)inv_metric_init.size(), n_params);
  }
  if (caller_inv_metric) {
    std::vector<double> inv_m = inv_metric_init;
    std::vector<double> sqrt_m(n_params, 1.0);
    for (int i = 0; i < n_params; i++) {
      // Same clamp as warm_start_mass_matrix: prevents singular/runaway metrics.
      inv_m[i] = std::max(1e-3, std::min(inv_m[i], 1e3));
      sqrt_m[i] = 1.0 / std::sqrt(inv_m[i]);
    }
    mass.set_diagonal(inv_m, sqrt_m);
    if (verbose) {
      REprintf("  [WARMSTART] Caller-supplied inv_metric_diag applied (n=%d)\n",
               n_params);
    }
  }

  // Recompute epsilon with warm-start mass (if informed)
  // This gives the dual averaging a better starting point when mass is pre-set
  if ((mass.type != MassMatrixType::DIAG || caller_inv_metric)
      && !mass.inv_mass_diag.empty()) {
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
    // Start from the process-global integrator selection. For an adaptive
    // selection this holds the fixed placeholder (same stage count as the
    // resolved scheme); the step-adapted coefficient is set at warmup end.
    nuts_ws.scheme = get_integrator_scheme();

    // Multiple-time-stepping selection: resolve the prior-only ("fast") gradient
    // once, alongside the full gradient, and record the inner-substep count.
    nuts_ws.mts = get_integrator_mts();
    nuts_ws.mts_m = get_mts_substeps();
    if (nuts_ws.mts) {
      nuts_ws.prior_gradient_fn = resolve_prior_gradient_fn(g_gradient_mode, data, layout);
    }
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

  int warmup_total_leapfrog = 0;  // leapfrog steps summed over warmup (verbose)
  // Warmup divergences are normal for DIAG models and resolve via dual
  // averaging; only the final epsilon matters (checked at warmup end).

  if (verbose && layout.is_temporal_gp) {
    REprintf("  [MASS-WINDOWS] n_warmup=%d, windows=[", n_warmup);
    for (size_t w = 0; w < mass_window_ends.size(); w++) {
      REprintf("%d%s", mass_window_ends[w], w + 1 < mass_window_ends.size() ? "," : "");
    }
    REprintf("]\n");
  }
