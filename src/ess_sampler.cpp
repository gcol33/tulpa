// ess_sampler.cpp
// Definitions of tulpa_ess::run_ess_sampler and of compute_log_post_double,
// its closure target (both declared in ess_sampler.h). ESS is reached from downstream
// packages via the C-callable shim `tulpa_run_ess_sampler`
// (tulpa_shims_vi_ess.h) and the generic-layout LikelihoodSpec path.

#include <Rcpp.h>
#include <RcppEigen.h>
#include "hmc_sampler.h"
#include "ess_sampler.h"
#include "log_post_impl.h"

namespace tulpa_ess {

double compute_log_post_double(
    const std::vector<double>& params,
    const tulpa_hmc::ModelData& data,
    const tulpa_hmc::ParamLayout& layout
) {
    return tulpa::compute_log_post_impl<double>(params, data, layout);
}


ESSResult run_ess_sampler(
    const std::vector<double>& init_params,
    const ModelData& data,
    const ParamLayout& layout,
    const ESSConfig& config
) {
    ESSResult result;
    result.success = true;

    int n_params = init_params.size();
    // Number of stored draws: post-warmup iters store when
    // (iter - n_warmup) % thin == 0, which fires ceil(post / thin) times.
    const int thin = std::max(1, config.n_thin);
    const int n_post = config.n_iter - config.n_warmup;
    const int n_save = n_post > 0 ? (n_post + thin - 1) / thin : 0;

    result.samples.resize(n_save, n_params);
    result.log_lik.resize(n_save);
    result.n_slice_evals = 0;
    result.n_slice_exhausted = 0;
    result.n_degenerate_scale = 0;

    // Build Gaussian references for ESS blocks
    std::vector<GaussianReference> gaussian_references = build_gaussian_references(data, layout, n_params);

    // Get non-Gaussian parameters for RWMH
    std::vector<int> non_gaussian = get_non_gaussian_params(layout, n_params);
    complete_ess_partition(gaussian_references, non_gaussian, n_params);

    // Random-effect terms, and the one-dimensional slice moves the sweep
    // adds for them: each log-SD in the other parameterization (when
    // joint_sigma_re), and each fixed effect with its between-group part.
    const std::vector<EssReTerm> re_terms = ess_re_terms(data, layout);
    const std::vector<EssFixedShift> shifts =
        ess_fixed_shifts(data, layout, re_terms);
    struct SliceMove {
        int kind;          // 0 / 2: log-SD (term, coef) in the other / the
                           // block's own parameterization; 3: a scalar
                           // term's log-SD partially non-centered;
                           // 1: shifts[term]
        int term, coef;
        int tracked;       // the coordinate whose spread sets the width
        double width, sum = 0.0, sumsq = 0.0;
        EssPncp pn;        // kind 3: the path, read off the likelihood
    };
    std::vector<SliceMove> slice_moves;
    if (config.joint_sigma_re) {
        for (int t = 0; t < (int)re_terms.size(); t++)
            for (int c = 0; c < re_terms[t].v.n_coefs; c++)
                for (int kind : {2, 0, 3}) {
                    if (kind == 3 && tulpa::priors::re_term_correlated(
                            re_terms[t].v.chol_start, re_terms[t].v.n_coefs))
                        continue;
                    slice_moves.push_back({kind, t, c,
                                           re_terms[t].v.log_sigma_idx[c],
                                           config.joint_sigma_proposal_sd});
                }
    }
    // Sweeps at which warmup re-reads the partially non-centered paths.
    auto pncp_read_now = [&](int iter) {
        const int w = config.n_warmup;
        for (int at : {w / 10, w / 4, w / 2, w - w / 4, w - 1})
            if (iter == at) return true;
        return false;
    };
    for (int k = 0; k < (int)shifts.size(); k++)
        slice_moves.push_back({1, k, 0,
                               layout.process_beta_start[0] + shifts[k].j,
                               config.joint_sigma_proposal_sd});
    int slice_n = 0;

    // Adaptive proposals for the RWMH parameters. The slice widths are set
    // during warmup to three SDs of the tracked coordinate's own draws so far;
    // tracking the steps taken instead feeds back -- a narrow width takes
    // short steps, which narrow it further, until stepping out no longer
    // reaches the edge of the slice.
    const int n_rwmh = (int)non_gaussian.size();
    AdaptiveProposal adaptive(n_rwmh, config.adapt_interval);

    EllipseFit ellipse_fit(gaussian_references, config.n_warmup);

    // Current state
    std::vector<double> params = init_params;

    // Compute initial log-posterior
    double current_log_post = compute_log_post_double(params, data, layout);

    if (!std::isfinite(current_log_post)) {
        result.success = false;
        result.error_msg = "Initial log-posterior is not finite";
        return result;
    }

    // Progress
    if (config.verbose) {
        Rcpp::Rcout << "Running ESS sampler...\n";
        Rcpp::Rcout << "  Parameters: " << n_params << "\n";
        Rcpp::Rcout << "  Gaussian blocks: " << gaussian_references.size() << "\n";
        Rcpp::Rcout << "  Non-Gaussian params: " << non_gaussian.size() << "\n";
        Rcpp::Rcout << "  Iterations: " << config.n_iter << " (warmup: " << config.n_warmup << ")\n";
    }

    int save_idx = 0;
    int total_ess_evals = 0;
    int total_ess_steps = 0;
    int n_slice_exhausted = 0;
    int n_degenerate_scale = 0;

    for (int iter = 0; iter < config.n_iter; iter++) {
        // Check for user interrupt
        if (iter % 100 == 0) {
            Rcpp::checkUserInterrupt();
        }

        // ----------------------------------------------------------------
        // ESS updates for Gaussian-prior blocks
        // ----------------------------------------------------------------
        for (auto& prior : gaussian_references) {
            // Refresh the ellipse scale from the sampled log-SD when bound to a
            // parameter (the RE block tracks sigma_re = exp(log_sigma_re)).
            if (prior.scale_param_idx >= 0) {
                prior.scale = std::exp(params[prior.scale_param_idx]);
            }
            if (!prior.effect_scale_idx.empty() && prior.center_b.size() > 0) {
                const Eigen::VectorXd unit = prior.effect_units(params);
                if (unit.allFinite() && unit.minCoeff() > 0.0) {
                    prior.center = prior.center_b.cwiseQuotient(unit);
                    prior.sd = prior.sd_b.cwiseQuotient(unit);
                }
            }
            const double blk_prec = 1.0 / (prior.scale * prior.scale);

            // A sufficiently negative sampled log-SD underflows the scale to
            // exactly zero, making blk_prec infinite and the slice target below
            // -Inf + Inf = NaN at every angle. Leave the block where it is for
            // this sweep and count it; the hyperparameter moves that follow can
            // still carry the scale back into range.
            if (!prior.fitted() && (!(prior.scale > 0.0) ||
                                    !std::isfinite(prior.scale) ||
                                    !std::isfinite(blk_prec))) {
                n_degenerate_scale++;
                continue;
            }

            // Extract current values for this block
            int block_size = prior.param_indices.size();
            Eigen::VectorXd f(block_size);
            for (int i = 0; i < block_size; i++) {
                f(i) = params[prior.param_indices[i]];
            }

            // ESS slice target: the ellipse (nu ~ N(0, scale^2 I), or the
            // fitted N(center, diag(sd^2))) already carries a Gaussian factor on
            // this block, so the slice threshold must be the full log-posterior
            // with exactly that factor removed (log_ellipse). ESS then targets
            // N(f; ellipse) * exp(lp - log_ellipse(f)) = exp(lp), the block
            // conditional itself, for ANY ellipse: the correction's contract is
            // with the ellipse (both read `prior`), not with whatever the
            // templated priors contribute for this block. The ellipse decides
            // how closely it tracks the block's conditional, i.e. how often a
            // proposal clears the slice, not what is sampled. Passing the full
            // posterior uncorrected is what double-counts the block prior and
            // over-shrinks; the normalizer is constant in f and cancels in the
            // slice.
            auto log_lik_fn = [&](const Eigen::VectorXd& f_new) -> double {
                std::vector<double> params_temp = params;
                for (int i = 0; i < block_size; i++) {
                    params_temp[prior.param_indices[i]] = f_new(i);
                }
                double lp = compute_log_post_double(params_temp, data, layout);
                return lp - prior.log_ellipse(f_new);  // remove the ellipse
            };

            // Perform ESS step. The slice level must be this block's target at
            // the current f, evaluated with the SAME log_lik_fn (block prior
            // removed) and the current params (which already include earlier
            // blocks' updates this sweep), so compute it directly.
            int n_evals = 0;
            bool exhausted = false;
            double cur_block_loglik = log_lik_fn(f);
            Eigen::VectorXd f_new = ess_step(f, prior, log_lik_fn,
                                             cur_block_loglik, n_evals, exhausted);
            if (exhausted) n_slice_exhausted++;

            // Update params
            for (int i = 0; i < block_size; i++) {
                params[prior.param_indices[i]] = f_new(i);
            }

            total_ess_evals += n_evals;
            total_ess_steps++;
        }

        // Update log-posterior after ESS updates
        current_log_post = compute_log_post_double(params, data, layout);

        // ----------------------------------------------------------------
        // RWMH updates for non-Gaussian parameters
        // ----------------------------------------------------------------
        for (size_t i = 0; i < non_gaussian.size(); i++) {
            int idx = non_gaussian[i];
            bool accepted = false;
            current_log_post = rwmh_step(
                params, idx, adaptive.proposal_sds[i],
                [&](const std::vector<double>& p) { return compute_log_post_double(p, data, layout); },
                current_log_post, accepted
            );
            adaptive.record(i, accepted);
        }

        // ----------------------------------------------------------------
        // Random-effect moves (see above): the slice moves, then the exact
        // fixed-effect / group-level translation.
        // ----------------------------------------------------------------
        auto log_post_of = [&](const std::vector<double>& p) {
            return compute_log_post_double(p, data, layout);
        };
        for (auto& mv : slice_moves) {
            if (mv.kind == 3) {
                if (mv.pn.ready)
                    current_log_post = ess_re_pncp_move(
                        params, re_terms[mv.term], mv.coef, mv.pn, mv.width,
                        log_post_of, current_log_post);
            } else if (mv.kind != 1) {
                double delta = 0.0;
                current_log_post = ess_re_scale_move(
                    params, re_terms[mv.term], mv.coef, mv.kind == 2, mv.width,
                    log_post_of, current_log_post, delta);
            } else {
                current_log_post = ess_fixed_shift_move(
                    params, shifts[mv.term], re_terms, layout, mv.width,
                    log_post_of, current_log_post);
            }
        }
        for (const auto& term : re_terms) {
            bool accepted = false;
            current_log_post = ess_translate_alias(
                params, term, data, layout, log_post_of, current_log_post,
                accepted);
        }
        if (iter < config.n_warmup && config.adapt_during_warmup &&
            pncp_read_now(iter)) {
            for (auto& mv : slice_moves)
                if (mv.kind == 3)
                    ess_pncp_read(params, re_terms[mv.term], mv.coef,
                                  log_post_of, mv.pn);
        }
        if (iter < config.n_warmup && config.adapt_during_warmup &&
            !slice_moves.empty()) {
            slice_n++;
            for (auto& mv : slice_moves) {
                const double v = params[mv.tracked];
                mv.sum += v;
                mv.sumsq += v * v;
                if (slice_n >= 20 && iter % config.adapt_interval == 0) {
                    const double m = mv.sum / slice_n;
                    const double var = mv.sumsq / slice_n - m * m;
                    if (var > 0.0 && std::isfinite(var))
                        mv.width = std::max(3.0 * std::sqrt(var), 1e-2);
                }
            }
        }

        // Adapt RWMH proposals during warmup, and fit the block ellipses.
        if (iter < config.n_warmup && config.adapt_during_warmup) {
            adaptive.adapt(iter);
            ellipse_fit.observe(iter, params, gaussian_references);
        }

        // ----------------------------------------------------------------
        // Store sample
        // ----------------------------------------------------------------
        if (iter >= config.n_warmup && (iter - config.n_warmup) % thin == 0) {
            for (int j = 0; j < n_params; j++) {
                result.samples(save_idx, j) = params[j];
            }
            result.log_lik[save_idx] = current_log_post;
            save_idx++;
        }

        // Progress
        if (config.verbose && (iter + 1) % config.print_every == 0) {
            Rcpp::Rcout << "Iter " << (iter + 1) << "/" << config.n_iter;
            if (iter < config.n_warmup) {
                Rcpp::Rcout << " (warmup)";
            }
            Rcpp::Rcout << " log_post = " << current_log_post << "\n";
        }
    }

    result.n_slice_evals = total_ess_evals;
    result.avg_slice_evals = total_ess_steps > 0 ?
        static_cast<double>(total_ess_evals) / total_ess_steps : 0.0;
    result.n_slice_exhausted = n_slice_exhausted;
    result.n_degenerate_scale = n_degenerate_scale;

    if (config.verbose) {
        Rcpp::Rcout << "ESS complete. Avg slice evals per step: "
                    << result.avg_slice_evals << "\n";
    }

    // Reported once for the whole run: a per-sweep warning from inside the
    // sampler is both noise and, under options(warn = 2), a longjmp through the
    // sampler loop's own C++ frames.
    if (n_slice_exhausted > 0 || n_degenerate_scale > 0) {
        char msg[320];
        std::snprintf(msg, sizeof(msg),
            "ESS: %d of %d block updates returned unchanged (the slice level "
            "was not orderable, or the bracket collapsed without an "
            "acceptance), and %d were skipped on a prior scale that underflowed "
            "to zero. Those blocks did not move.",
            n_slice_exhausted, total_ess_steps, n_degenerate_scale);
        Rcpp::warning(std::string(msg));
    }

    return result;
}

} // namespace tulpa_ess
