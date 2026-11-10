    // Store sample: the row is q in its stored form (sampler_field_store.h),
    // each non-centered block's field in place of its coordinates and every
    // level-free block centred; q itself stays the sampled coordinates.
    if (!is_warmup) {
      double* row = result.sample_row(sample_idx);
      std::vector<double> field_position;
      const double coordinate_jacobian = coordinates.forward(q, field_position);
      if (has_collapsed_spatial(layout)) {
        const int units = layout.is_gp ? result.n_gp_collapsed : result.n_icar_collapsed;
        auto& fields = layout.is_gp ? result.gp_w_star_flat : result.icar_phi_star_flat;
        reconstruct_spatial_point(field_position, data, layout, field_rng, true,
          fields.data() + static_cast<size_t>(sample_idx) * units,
          layout.is_bym2_collapsed ? result.bym2_theta_star_flat.data() + static_cast<size_t>(sample_idx) * units : nullptr);
        if (!layout.is_gp) {
          double* phi = fields.data() + static_cast<size_t>(sample_idx) * units;
          std::vector<double> centered;
          tulpa::priors::areal_likelihood_field(phi, data, layout, centered);
          std::copy(centered.begin(), centered.end(), phi);
        }
      }
      std::memcpy(row, field_position.data(), n_params * sizeof(double));
      tulpa::store_field_draw(row, field_position, data, layout);

      result.log_prob[sample_idx] = log_prob_current - coordinate_jacobian;
      result.accept_prob[sample_idx] = alpha;
      result.n_leapfrog[sample_idx] = iter_n_leapfrog;
      result.divergent[sample_idx] = divergent ? 1 : 0;
      result.treedepth[sample_idx] = iter_treedepth;
      result.energy[sample_idx] = iter_H0;


      sample_idx++;
    } else {
      warmup_total_leapfrog += iter_n_leapfrog;  // summed over warmup (verbose)
    }

