    // Store sample: the row is q in its stored form (sampler_field_store.h),
    // each non-centered block's field in place of its coordinates and every
    // level-free block centred; q itself stays the sampled coordinates.
    if (!is_warmup) {
      double* row = result.sample_row(sample_idx);
      std::memcpy(row, q.data(), n_params * sizeof(double));
      tulpa::store_field_draw(row, q, data, layout);

      result.log_prob[sample_idx] = log_prob_current;
      result.accept_prob[sample_idx] = alpha;
      result.n_leapfrog[sample_idx] = iter_n_leapfrog;
      result.divergent[sample_idx] = divergent ? 1 : 0;
      result.treedepth[sample_idx] = iter_treedepth;
      result.energy[sample_idx] = iter_H0;


      sample_idx++;
    } else {
      warmup_total_leapfrog += iter_n_leapfrog;  // summed over warmup (verbose)
    }

