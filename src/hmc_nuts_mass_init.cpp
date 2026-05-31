// hmc_nuts_mass_init.cpp
// Mass-matrix selection (AUTO / DIAG / DENSE), warm-start helpers.

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <Rcpp.h>

#include "hmc_sampler.h"

namespace tulpa_hmc {

// =====================================================================
// Mass matrix selection and initialization helpers
// =====================================================================

// DENSE_MAX_PARAMS and MassMatrixConfig now defined in hmc_sampler.h

// Select mass matrix type (AUTO resolution, block detection, DENSE override)
// and initialize the DenseMassMatrix object.
MassMatrixConfig select_and_init_mass_matrix(
    DenseMassMatrix& mass,
    const ModelData& data,
    const ParamLayout& layout,
    int n_params,
    MassMatrixType metric_type,
    bool verbose
) {
  // Mass matrix adaptation
  // Resolve AUTO metric: DIAG default with DIAG?DENSE identity recovery.
  //
  // Key insight: adapted DENSE mass (where adapted=true) incurs O(n?) per leapfrog
  // step for matvec/kinetic/p_sharp operations. For n=54 (RE models), this is 22x
  // slower per step than identity mass (adapted=false). DIAG?identity recovery gives
  // fast per-step execution while still finding correct epsilon via dual averaging.
  //
  // Strategy: Start with DIAG for most models. If DIAG fails catastrophically
  // (epsilon > 2.0 at warmup end), recover to DENSE identity (adapted=false).
  // This gives:
  //   - O(n) per-step cost (identity path)
  //   - Correct epsilon from find_reasonable_epsilon
  //   - No divergences (epsilon small enough for the geometry)
  //
  // Only genuinely complex posteriors (correlated slopes, BYM2, GP, SVC) start
  // with DENSE, where the adapted covariance actually helps sampling efficiency
  // enough to justify the O(n?) per-step cost.
  //
  // HISTORY:
  // 2026-02-27: has_re/has_temporal in needs_dense ? 1.25s PG+RE (adapted dense)
  // 2026-02-28: TVC gradient fix ? removed TVC
  // 2026-03-03: Removed has_re/has_temporal ? DIAG + recovery. PG+RE: 0.8s
  //   (identity dense, 22x faster per step). Adapted dense measured at 17.7s
  //   due to O(n?) per-step cost for n=54.
  MassMatrixType effective_metric = metric_type;
  bool auto_selected_diag = false;
  // Block specs for BLOCK_DIAG: (start_index, block_size) pairs
  std::vector<std::pair<int,int>> block_specs;

  if (effective_metric == MassMatrixType::AUTO) {
    // Build block_specs from param layout: detect pairs of correlated hyperparameters.
    // Each block captures a small dense correlation (2-4 params) at O(block?) cost,
    // avoiding full O(n?) DENSE while handling the key correlations DIAG misses.
    // NOTE: temporal_gp excluded ? DIAG is faster (2.11s vs 2.39s, 5 seeds Bin+GP_t).
    // NOTE: HSGP excluded from AUTO blocks ? DIAG is faster for HSGP-only (29k LF
    // vs 39k LF), and HSGP+temporal uses full DENSE (tested BLOCK_DIAG 2026-03-10:
    // worse performance and more divergences than DENSE).
    if (layout.is_bym2 && layout.log_sigma_bym2_idx >= 0 &&
        layout.logit_rho_bym2_idx == layout.log_sigma_bym2_idx + 1) {
      block_specs.push_back({layout.log_sigma_bym2_idx, 2});
    }
    if (layout.is_gp && layout.log_sigma2_gp_idx >= 0 &&
        layout.log_phi_gp_idx == layout.log_sigma2_gp_idx + 1) {
      block_specs.push_back({layout.log_sigma2_gp_idx, 2});
    }
    if (layout.is_multiscale_gp) {
      if (layout.log_sigma2_gp_local_idx >= 0 &&
          layout.log_phi_gp_local_idx == layout.log_sigma2_gp_local_idx + 1) {
        block_specs.push_back({layout.log_sigma2_gp_local_idx, 2});
      }
      if (layout.log_sigma2_gp_regional_idx >= 0 &&
          layout.log_phi_gp_regional_idx == layout.log_sigma2_gp_regional_idx + 1) {
        block_specs.push_back({layout.log_sigma2_gp_regional_idx, 2});
      }
    }
    // SVC layout groups all sigma2 then all phi: [sigma2_0..sigma2_{k-1},
    // phi_0..phi_{k-1}]. The correlated pair for SVC t is (sigma2_t, phi_t),
    // which is contiguous only when k == 1 (sigma2_0 immediately followed by
    // phi_0). A BLOCK_DIAG MassBlock spans a contiguous index range, so the
    // (sigma2_t, phi_t) pairs are blockable only in that single-SVC case;
    // multi-SVC models keep the DIAG/DENSE selection below.
    if (layout.has_svc && layout.log_sigma2_svc_start >= 0 &&
        layout.log_phi_svc_start >= 0) {
      int n_svc = layout.log_sigma2_svc_end - layout.log_sigma2_svc_start;
      if (n_svc == 1 && layout.log_phi_svc_start == layout.log_sigma2_svc_start + 1) {
        block_specs.push_back({layout.log_sigma2_svc_start, 2});
      }
    }
    if (layout.is_st_gp && layout.log_phi_st_space_idx >= 0 &&
        layout.log_phi_st_time_idx == layout.log_phi_st_space_idx + 1) {
      block_specs.push_back({layout.log_phi_st_space_idx, 2});
    }
    if (layout.has_multiscale_temporal) {
      // Multiscale temporal hyperparams form a natural block (3-4 params):
      // log_sigma2_trend, log_sigma2_seasonal, log_sigma2_short [, logit_rho_short]
      // These are correlated but the temporal effects themselves (phi) are not
      // strongly correlated with the hyperparams ? BLOCK_DIAG, not full DENSE.
      int ms_block_start = -1;
      int ms_block_size = 0;
      if (layout.log_sigma2_trend_idx >= 0) {
        ms_block_start = layout.log_sigma2_trend_idx;
        ms_block_size = 1;
      }
      if (layout.log_sigma2_seasonal_idx >= 0) {
        if (ms_block_start < 0) ms_block_start = layout.log_sigma2_seasonal_idx;
        ms_block_size++;
      }
      if (layout.log_sigma2_short_idx >= 0) {
        if (ms_block_start < 0) ms_block_start = layout.log_sigma2_short_idx;
        ms_block_size++;
      }
      if (layout.logit_rho_short_idx >= 0) {
        ms_block_size++;
      }
      if (ms_block_start >= 0 && ms_block_size >= 2 && ms_block_size <= 4) {
        block_specs.push_back({ms_block_start, ms_block_size});
      }
    }
    // Correlated slopes: the per-term Cholesky params form a contiguous,
    // naturally correlated block. A MassBlock holds a dense block up to 4x4
    // (its stack-allocated storage stride), so terms with 2..4 Cholesky params
    // get a dedicated block; wider correlated-slope terms keep the DIAG/DENSE
    // selection below.
    if (layout.has_re_correlated_slopes) {
      for (size_t t = 0; t < layout.chol_re_start_multi.size(); t++) {
        if (layout.re_correlated_multi[t]) {
          int chol_start = layout.chol_re_start_multi[t];
          int chol_size = layout.chol_re_end_multi[t] - chol_start;
          if (chol_size >= 2 && chol_size <= 4) {
            block_specs.push_back({chol_start, chol_size});
          }
        }
      }
    }

    // Family-specific overdispersion block-spec heuristics used to live
    // here, driven by layout.legacy.has_phi_num/denom and
    // data.legacy.model_type. Phase D (gcol33/tulpa#15) removed those
    // fields; downstream LikelihoodSpec packages ship their own
    // extra-parameter blocks via spec->n_extra_params at
    // layout.extra_offset. A future revision can re-introduce the
    // family-specific BLOCK_DIAG heuristic via a LikelihoodSpec callback
    // (e.g. spec->suggest_mass_blocks).
    bool is_icar = (data.spatial_type == SpatialType::ICAR);
    // HSGP+temporal: 36 HSGP basis coefs and 20 temporal effects have complex
    // cross-correlations that DIAG can't handle (106 div) and BLOCK_DIAG misses
    // (16 div, eps~0.006). DENSE with eigenvalue conditioning captures the geometry
    // correctly (0-1 div). Tested BLOCK_DIAG (2026-03-10): 303s/0div PG, 214s/16div NB,
    // 133s/3div Bin ? worse than DENSE (211s/3div, 176s/1div, 142s/0div).
    // Also applies to HSGP+TVC and HSGP+MS_t (same cross-correlation issue).
    // Only use DENSE when p <= 200 to avoid O(n?) per-step overhead dominating.
    bool hsgp_temporal = layout.is_hsgp && data.has_hsgp && n_params <= DENSE_MAX_PARAMS &&
                         (layout.has_temporal || layout.has_tvc || layout.has_multiscale_temporal);

    // Family + ICAR heuristics (NB+ICAR, Bin+ICAR forcing DENSE) used to
    // live here, gated on legacy ModelType. Phase D (gcol33/tulpa#15)
    // removed the family enum; the ICAR+family pairing would need to be
    // re-expressed through a LikelihoodSpec hint to come back.
    bool needs_full_dense = layout.has_latent ||  // N×K latent factors
                            hsgp_temporal ||      // HSGP+temporal cross-correlations
                            (is_icar && n_params <= DENSE_MAX_PARAMS);

    // HSGP-only (no temporal): DIAG outperforms BLOCK_DIAG (29k LF/6 div vs 39k LF/15 div).
    // HSGP-only (no temporal): DIAG outperforms BLOCK_DIAG (29k LF/6 div vs
    // 39k LF/15 div). The real correlations are between lengthscale and m^2 basis
    // coefficients, which small blocks can't capture. Block adaptation adds noise.
    // HSGP+temporal uses full DENSE (handled above in needs_full_dense).
    bool prefer_diag = layout.is_hsgp && data.has_hsgp && !layout.has_temporal;

    if (needs_full_dense) {
      effective_metric = MassMatrixType::DENSE;
      block_specs.clear();
      auto_selected_diag = false;
    } else if (!block_specs.empty() && !prefer_diag) {
      // BLOCK_DIAG: captures key correlations without full O(n?)
      effective_metric = MassMatrixType::BLOCK_DIAG;
      auto_selected_diag = false;
    } else {
      // No blocks detected, no DENSE needed ? fall back to DIAG
      effective_metric = MassMatrixType::DIAG;
      auto_selected_diag = true;
    }

    if (verbose) {
      REprintf("  [METRIC] auto -> %s (p=%d", metric_name(effective_metric), n_params);
      if (effective_metric == MassMatrixType::BLOCK_DIAG) {
        REprintf(", %d blocks:", (int)block_specs.size());
        for (const auto& bs : block_specs) {
          REprintf(" [%d,%d)", bs.first, bs.first + bs.second);
        }
      }
      if (layout.has_re) REprintf(", re");
      if (layout.has_re_correlated_slopes) REprintf(", correlated_slopes");
      if (layout.has_temporal) REprintf(", temporal");
      if (layout.is_bym2) REprintf(", bym2");
      if (layout.is_hsgp) REprintf(", hsgp");
      if (layout.is_gp) REprintf(", gp");
      if (layout.is_multiscale_gp) REprintf(", msgp");
      if (layout.is_temporal_gp) REprintf(", temporal_gp");
      if (layout.has_multiscale_temporal) REprintf(", ms_temporal");
      if (layout.has_svc) REprintf(", svc");
      if (layout.has_spatiotemporal) REprintf(", spatiotemporal");
      if (layout.has_latent) REprintf(", latent");
      if (layout.has_tvc) REprintf(", tvc");
      REprintf(")\n");
    }
  }
  // Also build block_specs when user explicitly requests BLOCK_DIAG
  if (effective_metric == MassMatrixType::BLOCK_DIAG && block_specs.empty()) {
    // User forced block_diag but AUTO didn't run ? detect blocks from layout
    if (layout.is_temporal_gp && layout.log_sigma2_temporal_gp_idx >= 0 &&
        layout.logit_phi_temporal_gp_idx == layout.log_sigma2_temporal_gp_idx + 1) {
      block_specs.push_back({layout.log_sigma2_temporal_gp_idx, 2});
    }
    if (layout.is_hsgp && layout.log_sigma2_hsgp_idx >= 0 &&
        layout.log_lengthscale_hsgp_idx == layout.log_sigma2_hsgp_idx + 1) {
      block_specs.push_back({layout.log_sigma2_hsgp_idx, 2});
    }
    if (layout.is_bym2 && layout.log_sigma_bym2_idx >= 0 &&
        layout.logit_rho_bym2_idx == layout.log_sigma_bym2_idx + 1) {
      block_specs.push_back({layout.log_sigma_bym2_idx, 2});
    }
    if (layout.is_gp && layout.log_sigma2_gp_idx >= 0 &&
        layout.log_phi_gp_idx == layout.log_sigma2_gp_idx + 1) {
      block_specs.push_back({layout.log_sigma2_gp_idx, 2});
    }
    if (layout.is_multiscale_gp) {
      if (layout.log_sigma2_gp_local_idx >= 0 &&
          layout.log_phi_gp_local_idx == layout.log_sigma2_gp_local_idx + 1) {
        block_specs.push_back({layout.log_sigma2_gp_local_idx, 2});
      }
      if (layout.log_sigma2_gp_regional_idx >= 0 &&
          layout.log_phi_gp_regional_idx == layout.log_sigma2_gp_regional_idx + 1) {
        block_specs.push_back({layout.log_sigma2_gp_regional_idx, 2});
      }
    }
    if (layout.is_st_gp && layout.log_phi_st_space_idx >= 0 &&
        layout.log_phi_st_time_idx == layout.log_phi_st_space_idx + 1) {
      block_specs.push_back({layout.log_phi_st_space_idx, 2});
    }
    // Legacy ratio overdispersion pair (log_phi_num, log_phi_denom) used
    // to add a 2x2 block here; both indices moved into the LikelihoodSpec
    // extra-parameter block in Phase D (gcol33/tulpa#15).

    // If still no blocks found, fall back to DIAG
    if (block_specs.empty()) {
      effective_metric = MassMatrixType::DIAG;
      if (verbose) {
        REprintf("  [BLOCK_DIAG] No correlated hyperparameter pairs found, falling back to DIAG\n");
      }
    }
  }

  // Auto-downgrade dense to diagonal when n_params too large
  // Dense needs O(p^2) storage and O(p^3) Cholesky; also needs n_warmup >= p samples
  if (effective_metric == MassMatrixType::DENSE && n_params > DENSE_MAX_PARAMS) {
    if (verbose) {
      REprintf("  [DENSE] n_params=%d > %d: auto-downgrading to diagonal\n",
               n_params, DENSE_MAX_PARAMS);
    }
    effective_metric = MassMatrixType::DIAG;
  }

  try {
    if (effective_metric == MassMatrixType::BLOCK_DIAG && !block_specs.empty()) {
      mass.init_block_diag(n_params, block_specs);
    } else {
      mass.init(n_params, effective_metric);
    }
  } catch (const std::bad_alloc&) {
    if (effective_metric == MassMatrixType::DENSE) {
      if (verbose) {
        REprintf("  [DENSE] Allocation failed for p=%d, falling back to diagonal\n", n_params);
      }
      effective_metric = MassMatrixType::DIAG;
      mass.init(n_params, effective_metric);
    } else {
      throw;
    }
  }

  // Initialize sparse GMRF block for ST_IV spatiotemporal interaction.
  // Uses sparse Cholesky of posterior precision Q = tau*(Q_s?Q_t) + diag(H_lik).
  // At warmup end, extracts diag(Q^{-1}) to set precision-informed diagonal mass.
  // NOTE: Factorization happens later (after warmup discovers tau and H_lik).
  bool use_sparse_gmrf_mass = true;
  if (use_sparse_gmrf_mass && data.has_spatiotemporal && data.spatiotemporal_data.type == STType::TYPE_IV) {
    int st_S = data.spatiotemporal_data.n_spatial;
    int st_T = data.spatiotemporal_data.n_times;
    mass.sparse_gmrf.init(layout.st_delta_start, st_S, st_T);
    if (verbose) {
      REprintf("  [SPARSE_GMRF] ST_IV block initialized: %dx%d=%d params at offset %d\n",
               st_S, st_T, st_S * st_T, layout.st_delta_start);
    }
  }

  return {effective_metric, auto_selected_diag, std::move(block_specs)};
}

// Warm-start mass matrix diagonal from model structure.
// Sets informed diagonal entries for parameter groups with known posterior scale,
// giving the step size tuner a reasonable starting point even before warmup
// samples are collected. This is critical for HSGP (m^2 basis coefficients),
// BYM2 (spatial + IID), and correlated slopes (z ~ N(0,1)) models where the
// identity mass causes excessively small epsilon -> deep NUTS trees.
void warm_start_mass_matrix(
    DenseMassMatrix& mass,
    const ModelData& data,
    const ParamLayout& layout,
    int n_params,
    bool verbose
) {
  std::vector<double> inv_m(n_params, 1.0);
  std::vector<double> sqrt_m(n_params, 1.0);
  bool any_informed = false;

  // HSGP basis coefficients: beta_j ~ N(0, 1) ? posterior variance ? 1
  // Hyperparameters: log_sigma2 ~ prior with moderate variance,
  //                  log_lengthscale ~ LogNormal(0,1) ? variance ? 1
  if (layout.is_hsgp) {
    for (int j = layout.hsgp_beta_start; j < layout.hsgp_beta_end; j++) {
      inv_m[j] = 1.0;  // N(0,1) prior ? unit scale
    }
    inv_m[layout.log_sigma2_hsgp_idx] = 1.0;
    inv_m[layout.log_lengthscale_hsgp_idx] = 1.0;
    any_informed = true;
  }

  // ICAR: phi[s] precision ? degree (number of neighbors)
  // Higher degree ? smaller variance ? tighter mass
  if (layout.has_spatial && !layout.is_bym2 &&
      data.spatial_type == SpatialType::ICAR && !data.adj_row_ptr.empty()) {
    for (int s = 0; s < (layout.spatial_end - layout.spatial_start); s++) {
      int degree = data.adj_row_ptr[s + 1] - data.adj_row_ptr[s];
      // ICAR precision diagonal ? degree; variance ? 1/degree
      double var_est = 1.0 / std::max(1.0, (double)degree);
      inv_m[layout.spatial_start + s] = var_est;
    }
    any_informed = true;
  }

  // BYM2: spatial phi ~ ICAR (eigenvalue-scaled), theta ~ N(0, I)
  // Riebler parameterization: phi[s] ? scale_factor variance
  if (layout.is_bym2) {
    double sf = std::max(data.bym2_scale_factor, 0.1);
    for (int s = layout.spatial_start; s < layout.spatial_end; s++) {
      inv_m[s] = sf * sf;  // ICAR variance ~ scale_factor^2
    }
    for (int s = layout.theta_bym2_start; s < layout.theta_bym2_end; s++) {
      inv_m[s] = 1.0;  // IID: N(0,1)
    }
    inv_m[layout.log_sigma_bym2_idx] = 1.0;
    inv_m[layout.logit_rho_bym2_idx] = 4.0;  // logit scale: wider
    any_informed = true;
  }

  // Correlated slopes: z ~ N(0, 1) (non-centered), Cholesky raw ~ tanh
  if (layout.has_re_correlated_slopes) {
    // RE slopes z values
    for (int j = layout.re_start; j < layout.re_end; j++) {
      inv_m[j] = 1.0;  // z ~ N(0,1)
    }
    any_informed = true;
  }

  // Temporal effects: RW1/RW2 have known precision structure
  if (layout.has_temporal) {
    // Temporal effects: moderate scale (tau-dependent, start at 1.0)
    for (int j = layout.temporal_start; j < layout.temporal_end; j++) {
      inv_m[j] = 1.0;
    }
    // AR1 rho: logit scale variance ? 4
    if (layout.logit_rho_ar1_idx >= 0) {
      inv_m[layout.logit_rho_ar1_idx] = 4.0;
    }
    any_informed = true;
  }

  // Non-centered RE: z ~ N(0, 1) ? unit scale
  if (layout.has_re && data.re_parameterization == 1) {  // 1 = non-centered
    for (int j = layout.re_start; j < layout.re_end; j++) {
      inv_m[j] = 1.0;
    }
    any_informed = true;
  }

  // GP/SVC/TVC hyperparameters: moderate scale
  if (layout.is_gp) {
    if (layout.log_sigma2_gp_idx >= 0) inv_m[layout.log_sigma2_gp_idx] = 1.0;
    if (layout.log_phi_gp_idx >= 0) inv_m[layout.log_phi_gp_idx] = 1.0;
    any_informed = true;
  }

  // Temporal GP: NC z ~ N(0,1), logit_phi has wider posterior scale
  if (layout.is_temporal_gp) {
    for (int j = layout.temporal_start; j < layout.temporal_end; j++) {
      inv_m[j] = 1.0;  // z ~ N(0,1) for NC
    }
    if (layout.log_sigma2_temporal_gp_idx >= 0)
      inv_m[layout.log_sigma2_temporal_gp_idx] = 1.0;
    if (layout.logit_phi_temporal_gp_idx >= 0)
      inv_m[layout.logit_phi_temporal_gp_idx] = 4.0;  // logit scale: wider
    any_informed = true;
  }

  if (any_informed) {
    // Compute sqrt_mass from inv_mass
    for (int i = 0; i < n_params; i++) {
      inv_m[i] = std::max(1e-3, std::min(inv_m[i], 1e3));
      sqrt_m[i] = 1.0 / std::sqrt(inv_m[i]);
    }
    mass.set_diagonal(inv_m, sqrt_m);
    if (verbose) {
      REprintf("  [WARMSTART] Initialized mass matrix from model structure\n");
    }
  }
}

}  // namespace tulpa_hmc
