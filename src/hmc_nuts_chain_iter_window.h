    bool is_warmup = (iter < n_warmup);
    // Check if we've reached a mass adaptation window boundary
    if (is_warmup && next_window_idx < (int)mass_window_ends.size() &&
        iter == mass_window_ends[next_window_idx]) {
      bool dense_covariance_set = false;  // Track if DENSE covariance (not just diagonal) succeeded this window
      // Dense mass matrix: try full covariance first.
      // OAS shrinkage guarantees PD even when n < p, so the threshold sits
      // below n_params + 5, which for large p is unreachable inside warmup
      // (at p = 159 it needs 164 samples and the windows supply 125).
      int dense_threshold = std::min(n_params + 5,
                                     std::max(50, n_params / 2));
      // `cov_stats` is allocated only for a metric that starts DENSE, so its
      // dimension is the statement that this chain has a dense accumulator to
      // read at all.
      if (mass.type == MassMatrixType::DENSE && cov_stats.dim == n_params &&
          cov_stats.n >= dense_threshold) {
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
          // Cholesky failed -- mass auto-degraded to DIAG, use diagonal stats
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
        // Not enough samples for dense yet -- use diagonal as interim
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

      // Type-IV interaction: replace the Welford-adapted variances over the
      // st_delta block with diag(Q^-1) of that block's own posterior
      // precision, evaluated at the current position. Placed here rather than
      // once at warmup end so the step-size restart below is calibrated for
      // the metric the chain will actually run under, and so the terminal
      // buffer tunes epsilon against a metric that no longer moves.
      if (st_gmrf_mass) {
        StGmrfMassResult gmrf = st_gmrf_inv_mass(q, data, layout);
        if (gmrf.ok) {
          for (int k = 0; k < (int)gmrf.inv_mass.size(); k++) {
            const int j = layout.st_delta_start + k;
            mass.inv_mass_diag[j] = clamp_inv_mass(gmrf.inv_mass[k]);
            mass.sqrt_mass_diag[j] = 1.0 / std::sqrt(mass.inv_mass_diag[j]);
          }
          mass.adapted = true;
          use_mass_matrix = true;
          st_gmrf_applied = true;
          if (verbose) {
            REprintf("  [GMRF] Window %d (iter %d): %d block variances from "
                     "Q^-1 (clamped curvature: %d, ridge: %.3g)\n",
                     next_window_idx, iter, gmrf.n_block,
                     gmrf.n_curvature_clamped, gmrf.ridge_applied);
          }
          // The soft sum-to-zero penalty's own directions as an explicit
          // low-rank term on top of that diagonal. They are the stiffest
          // directions in the block by three to four orders of magnitude and
          // are linear combinations rather than coordinates, so this is the
          // part of the geometry no diagonal metric reaches
          //. The trend family rides the same term wherever
          // the penalty carries one, since a metric spanning less than the
          // penalty leaves a residual that grows with tau.
          // Rebuilt every window because set_diagonal() above drops the
          // previous term along with the diagonal it was built against.
          if (st_gmrf_margin_mass) {
            LowRankMassTerm term = make_margin_mass_term(
                layout.st_delta_start, gmrf.n_spatial, gmrf.n_times,
                gmrf.lambda_row, gmrf.lambda_col,
                mass.inv_mass_diag.data() + layout.st_delta_start,
                gmrf.n_block, gmrf.lambda_trend);
            const int term_rank = term.rank();
            const bool ok = mass.install_lowrank(std::move(term));
            st_gmrf_margin_applied = ok;
            if (!ok) st_gmrf_declined = "lowrank_factorize_failed";
            if (verbose) {
              REprintf("  [GMRF] Window %d (iter %d): margin term rank %d %s\n",
                       next_window_idx, iter, term_rank,
                       ok ? "installed" : "REFUSED (keeping the diagonal)");
            }
          }
        } else {
          // A window the override could not serve keeps that window's adapted
          // diagonal. The reason is recorded rather than dropped, so a fit
          // that fell back mid-warmup is not read as one that never asked.
          st_gmrf_declined = gmrf.reason;
          st_gmrf_margin_applied = false;
          if (verbose) {
            REprintf("  [GMRF] Window %d (iter %d): declined (%s)\n",
                     next_window_idx, iter, gmrf.reason);
          }
        }
      }

      // Temporal GP NC: z ~ N(0,1) by construction, so the optimal diagonal mass
      // is ~1.0. With limited warmup samples, noisy variance estimates for 20 z
      // params create unbalanced mass and a small epsilon. Fix z entries to 1.0 so
      // the step size is driven by the hyperparameters (beta, sigma2, phi) only.
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
      if (use_mass_matrix &&
          ((mass.type == MassMatrixType::DENSE && mass.adapted) ||
           mass.has_lowrank())) {
        epsilon = find_reasonable_epsilon_dense(q, data, layout, rng, mass);
      } else if (use_mass_matrix) {
        epsilon = find_reasonable_epsilon(q, data, layout, rng, mass.inv_mass_diag);
      } else {
        epsilon = find_reasonable_epsilon(q, data, layout, rng);
      }
      da = DualAveraging(epsilon, target_accept);

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
