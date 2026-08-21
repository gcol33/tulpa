    } else {
      // -----------------------------------------------------------------
      // Fixed-trajectory HMC
      // -----------------------------------------------------------------

      // One metric for the whole iteration. The momentum draw, the kinetic
      // energy and the leapfrog drift all read the same object, so the
      // trajectory is a Hamiltonian flow of the density the Metropolis ratio
      // below corrects for. Under the L-BFGS warmup metric that object is the
      // limited-memory operator: M^{-1} = H (the two-loop inverse-Hessian
      // approximation the drift applies), so the refresh draws from
      // M = H^{-1} = B, the direct BFGS matrix built from the same pairs.
      const bool lbfgs_metric = use_lbfgs && lbfgs_initialized &&
                                !lbfgs_warmup_done && lbfgs_state.d == n_params;

      std::vector<double> p(n_params);
      if (lbfgs_metric) {
        lbfgs_state.sample_momentum(p, rng);
      } else {
        mass.sample_momentum(p.data(), rng);
      }
      auto kinetic_energy_of = [&](const std::vector<double>& pv) -> double {
        return lbfgs_metric ? lbfgs_state.kinetic_energy(pv)
                            : mass.kinetic_energy(pv.data());
      };
      double kinetic_current = kinetic_energy_of(p);
      double H_current = -log_prob_current + kinetic_current;

      // Leapfrog integration
      std::vector<double> q_prop = q;
      std::vector<double> p_prop = p;

      // Determine effective L for this iteration
      int L_eff = L;
      if (use_nuts && use_lbfgs && !lbfgs_warmup_done) {
        // During L-BFGS warmup with NUTS mode, use fixed L=20
        L_eff = 20;
      }

      if (lbfgs_metric) {
        // L-BFGS leapfrog: the drift applies M^{-1} = H.
        std::vector<double> grad(n_params);
        double log_prob_step = log_prob_current;
        compute_gradient(q_prop, data, layout, grad);

        for (int l = 0; l < L_eff; l++) {
          for (int i = 0; i < n_params; i++) {
            p_prop[i] += 0.5 * epsilon * grad[i];
          }
          std::vector<double> Hp(n_params);
          lbfgs_state.multiply_H(p_prop, Hp);
          for (int i = 0; i < n_params; i++) {
            q_prop[i] += epsilon * Hp[i];
          }
          compute_gradient(q_prop, data, layout, grad, &log_prob_step);
          for (int i = 0; i < n_params; i++) {
            p_prop[i] += 0.5 * epsilon * grad[i];
          }
          if (leapfrog_state_nonfinite(log_prob_step, q_prop.data(),
                                       p_prop.data(), n_params)) {
            divergent = true;
            break;
          }
          double H_step = -log_prob_step + lbfgs_state.kinetic_energy(p_prop);
          if (hamiltonian_divergent(H_current, H_step)) {
            divergent = true;
            break;
          }
        }
      } else {
        // Standard leapfrog under the sampler's own metric.
        for (int l = 0; l < L_eff; l++) {
          LeapfrogResult lf = leapfrog_step(q_prop, p_prop, epsilon, data,
                                            layout, nullptr, &mass);
          q_prop = lf.q;
          p_prop = lf.p;
          if (lf.divergent) {
            divergent = true;
            break;
          }
          double H_step = -lf.log_prob + mass.kinetic_energy(p_prop.data());
          if (hamiltonian_divergent(H_current, H_step)) {
            divergent = true;
            break;
          }
        }
      }

      // Compute proposed Hamiltonian. After Phase D every caller is
      // generic LikelihoodSpec, so compute_log_post forwards to the
      // generic-spec evaluator.
      double log_prob_prop = compute_log_post(q_prop, data, layout);
      double kinetic_prop = kinetic_energy_of(p_prop);
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
