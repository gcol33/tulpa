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

        // Freeze the SoftAbs metric: compute the Hessian-based metric and its
        // step size ONCE (at the first post-warmup divergence) and reuse it for
        // every later rescue. Recomputing a position-dependent metric and
        // re-tuning epsilon at each divergent q made the rescue kernel depend on
        // the current state in a non-reversible way, biasing exactly the hard
        // region it targets. A single frozen metric is a fixed alternative
        // proposal (Riemannian in spirit, state-independent in practice).
        bool metric_ok = softabs_metric_active;
        if (!softabs_metric_active) {
          std::vector<double> hessian_buf;
          compute_hessian_finite_diff(q, data, layout, hessian_buf);
          for (auto& v : hessian_buf) v = -v;  // Negate: -H = curvature

          std::vector<double> G_inv_buf, L_G_inv_buf;
          metric_ok = compute_softabs_metric(
            hessian_buf, n_params, 1.0, G_inv_buf, L_G_inv_buf
          );
          if (metric_ok) {
            softabs_persistent_mass.set_from_metric(G_inv_buf, L_G_inv_buf);
            softabs_persistent_eps = find_reasonable_epsilon_dense(
              q, data, layout, rng, softabs_persistent_mass);
            softabs_metric_active = true;
          }
        }

        if (metric_ok) {
          double eps_base = softabs_persistent_eps;

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

          // The precision-informed diagonal mass override for ST_IV is not
          // wired for the generic LikelihoodSpec path: it would route
          // per-observation likelihood Hessians through spec->eta_weights_fn
          // (the IRLS callback already used by laplace_mode_spec_dense).
          // Until that wiring lands, deactivate the sparse GMRF block so
          // ST_IV chains fall back to the adapted DIAG mass matrix (one
          // warmup-end no-op per chain).
          if (mass.sparse_gmrf.active) {
            mass.sparse_gmrf.active = false;
            if (verbose) {
              REprintf("  [SPARSE_GMRF] ST_IV precision-informed mass override "
                       "not wired for the generic path; using adapted "
                       "diagonal mass.\n");
            }
          }

          if (verbose) {
            REprintf("  [METRIC] Warmup done: epsilon=%.6f, mass.type=%s, mass.adapted=%d\n",
                     epsilon, metric_name(mass.type), (int)mass.adapted);
          }

          // Step-adapted integrator. With the mass matrix and
          // step size settled, resolve the minimum-error multistage coefficient
          // for this chain's operating band (0, nu_max]. nu_max follows from the
          // adapted metric and the local curvature -- the
          // nested-approximation-informs-the-sampler synthesis. Warmup ran the
          // fixed placeholder (same stage count, so epsilon transfers); the
          // sampling phase walks the resolved per-chain scheme.
          if (get_integrator_adaptive() != IntegratorAdaptive::NONE) {
            double nu_max = compute_adaptive_nu_max(q, data, layout, mass, epsilon);
            if (get_integrator_adaptive() == IntegratorAdaptive::THREE_STAGE) {
              nuts_ws.scheme = simp::three_stage_adaptive(nu_max);
            } else {
              nuts_ws.scheme = simp::two_stage_adaptive(nu_max);
            }
            if (verbose) {
              REprintf("  [INTEGRATOR] Step-adapted %s: nu_max=%.4f, epsilon=%.6f\n",
                       nuts_ws.scheme.name.c_str(), nu_max, epsilon);
            }
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
