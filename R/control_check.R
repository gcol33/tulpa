# Control-list validation: typo protection for the `control = list()` surface.
#
# Every front-door fitter carries its perf / numerical / tuning knobs in a
# single `control` list (design principle 6), which without validation makes a
# misspelled knob a silent no-op (`control$adaptve_grid` fits the default).
# Each fitter validates its control NAMES against the canonical whitelist of
# what it (and every helper it forwards `control` to) actually reads;
# `tulpa()`'s set is the union over the backends it can dispatch.

# Subset a validated front-door control list to the keys an inner fitter
# accepts, so wholesale forwarding does not carry front-door-only knobs
# (grid shape, backend selection) into the inner fitter's narrower check.
#' @keywords internal
.control_subset <- function(control, allowed) {
  if (is.null(control) || length(control) == 0L) return(list())
  control[intersect(names(control), allowed)]
}

#' @keywords internal
.check_control <- function(control, allowed, where) {
  if (is.null(control)) return(invisible(NULL))
  if (!is.list(control)) {
    stop(sprintf("`control` must be a list in %s().", where), call. = FALSE)
  }
  if (length(control) == 0L) return(invisible(NULL))
  nm <- names(control)
  if (is.null(nm) || any(!nzchar(nm))) {
    stop(sprintf("every `control` entry must be named in %s().", where),
         call. = FALSE)
  }
  unknown <- setdiff(nm, allowed)
  if (length(unknown)) {
    stop(sprintf(
      "Unknown control knob(s) for %s(): %s.\nAllowed: %s.",
      where, paste(sQuote(unknown, q = FALSE), collapse = ", "),
      paste(sort(allowed), collapse = ", ")), call. = FALSE)
  }
  invisible(NULL)
}

# Canonical per-fitter control keys. A fitter that forwards `control`
# wholesale to another fitter unions that fitter's set.
.CONTROL_KEYS <- local({
  progress <- c("progress", "progress.every", "progress.file",
                "progress.throttle")
  keys <- list(
    nested_laplace = c("max_iter", "tol", "n_threads", "x_init",
                       "keep_grid_hessians", "diagnose_k", "k_samples",
                       "checkpoint", progress),
    nested_laplace_joint = c(
      "max_iter", "tol", "n_threads", "n_threads_outer", "n_threads_scatter",
      "x_init", "verbose", "hessian", "store_Q", "force_sparse",
      "inner_refresh", "integration", "local_ccd", "tile_warm",
      "prune", "prune_tol",
      "adaptive_grid", "adaptive_grid_cutoff", "adaptive_grid_edge_thresh",
      "adaptive_grid_max_frac", "adaptive_grid_max_passes",
      "adaptive_grid_min_cells", "adaptive_grid_stride",
      "var_of_means_consistency", "var_of_means_tolerance",
      "diagnose_k", "k_samples", "k_threads", "k_quality", "k_refine",
      "k_max_rounds", "k_bootstrap", "k_tail_points", "k_conf_bands",
      "checkpoint", progress),
    spde = c("method", "n_grid", "max_iter", "tol", "n_threads",
             "diagnose_k", "k_samples", "checkpoint"),
    re_cov_nested = c("integration", "n_per_axis", "span", "n_draws", "seed",
                      "max_iter", "tol", "n_threads", "diagnose_k",
                      "k_samples", "checkpoint"),
    re_cov_gibbs = c("n_iter", "warmup", "thin", "seed", "max_iter", "tol",
                     "n_threads"),
    ep = c("max_sweeps", "tol", "damping", "n_quad", "n_draws", "seed"),
    gaussian = c("iter", "warmup", "step_size", "n_leapfrog", "seed"),
    gibbs = c("n_iter", "warmup", "thin", "seed", "verbose", "n_threads"),
    nuts_beta = c("n_iter", "n_warmup", "max_treedepth", "adapt_delta",
                  "seed", "verbose"),
    nuts_spde = c("n_iter", "n_warmup", "max_treedepth", "adapt_delta",
                  "seed", "verbose", "mass_matrix", "noncenter"),
    multinomial = c("max_iter", "tol", "n_draws", "seed"),
    ordinal = c("max_iter", "n_draws", "seed"),
    sample_glmm = c("n_iter", "warmup", "n_chains", "seed", "verbose",
                    "epsilon", "L", "adapt_delta", "max_treedepth",
                    "n_draws", "alpha", "batch_size", "ess_threshold",
                    "n_particles", "n_mcmc_steps", "mclmc_adjusted",
                    "vi_variant", "vi_mc_samples", "vi_max_iter")
  )
  # tulpa() dispatches across the nested / spde / re_cov / gibbs / agq /
  # sampler backends and forwards `control` wholesale on the nested and
  # modeldata routes, so its surface is the union plus its own
  # backend-selection and conditioning knobs.
  # tulpa()'s control surface is the union of the backends it dispatches, minus
  # the statistical hyperpriors -- those now ride the `re_prior` / `beta_prior`
  # signature arguments (design principle 6: statistical args in the signature,
  # tuning knobs in control). The re_cov_nested / re_cov_gibbs sets still list
  # prior_sigma / eta / prior_df / prior_scale for their DIRECT callers; the
  # union drops them so tulpa(control = list(prior_sigma = )) errors and points
  # the user to re_prior.
  .tulpa_hyperprior_keys <- c("prior_sigma", "eta", "prior_df", "prior_scale",
                              "sigma_re_scale", "prior_sigma_scale")
  keys$tulpa <- sort(unique(setdiff(c(
    keys$nested_laplace, keys$nested_laplace_joint, keys$spde,
    keys$re_cov_nested, keys$re_cov_gibbs,
    keys$sample_glmm, keys$ep, keys$nuts_spde,
    c("re_cov", "n_quad", "sigma_init", "beta_init",
      "sigma_eps", "scale", "method")
  ), .tulpa_hyperprior_keys)))
  keys
})

# Valid keys for the `re_prior = list()` statistical argument on tulpa(): the
# random-effect / variance-component hyperpriors that used to hide in control.
#' @keywords internal
.RE_PRIOR_KEYS <- c("prior_sigma", "eta", "prior_df", "prior_scale",
                    "prior_sigma_scale", "sigma_re_scale")
