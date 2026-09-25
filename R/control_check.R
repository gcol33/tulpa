# Control-list validation: typo protection for the `control = list()` surface.
#
# Every front-door fitter carries its perf / numerical / tuning knobs in a
# single `control` list (design principle 6), which without validation makes a
# misspelled knob a silent no-op (`control$adaptve_grid` fits the default).
# Each fitter validates its control NAMES against the canonical whitelist of
# what it (and every helper it forwards `control` to) actually reads;
# `tulpa()`'s set is the union over the backends it can dispatch.

# Knobs that mean the same thing under two spellings across fitters: the NUTS
# drivers (`nuts_beta`, `nuts_spde`) took `n_warmup` while the sampler / Gibbs
# fitters took `warmup`. `tulpa()`'s control surface is the union of the
# backends it dispatches, so BOTH spellings pass its check -- and the subset
# below then silently dropped whichever the chosen backend did not list. A knob
# accepted at the front door and discarded on the way in is the exact silent
# no-op this file exists to prevent, so the spellings are canonicalized here,
# once, rather than aliased per fitter.
#
# Names the target fitter already accepts are never rewritten, so a fitter that
# genuinely reads `n_warmup` keeps receiving it. The pair is SYMMETRIC: a
# fitter wanting either spelling can be reached from a caller using the other
# (gcol33/tulpa#767 -- a one-directional alias silently dropped `warmup` on
# `nuts_spde` / `nuts_beta`, which list `n_warmup` only).
#' @keywords internal
.CONTROL_ALIASES <- c(n_warmup = "warmup")

# The one check on a sampler's run length, applied by every sampling fitter
# once its `n_iter` / `warmup` defaults are resolved (gcol33/tulpa#872). Only
# `mala()` and `imh_laplace()` used to refuse `warmup >= n_iter`; everywhere
# else it reached the kernel, which returned a fit with no draws (coef() read
# zeros under NA names, summary() failed on missing row names), NaN (the
# Polya-Gamma Gibbs), or aborted on a negative vector size. The natural
# `n_iter = 300, n_warmup = 300` hit it, because `n_iter` does not mean the
# same thing everywhere:
#
#   * "total" -- `n_iter` counts every iteration, warmup included, so the run
#     keeps `n_iter - warmup` draws: the NUTS / HMC, ESS, SGHMC, SGLD, Gibbs,
#     MALA and IMH kernels. Needs `0 <= warmup < n_iter` and `n_iter >= 2`.
#   * "post"  -- `n_iter` counts only the kept iterations and warmup runs on
#     top: `tulpa_re_cov_gibbs()` and MCLMC. Any `warmup >= 0` is valid; the
#     run needs `n_iter >= 1` (MCLMC's own kernel then asks for 2).
#
# The message states the convention so a caller who meant the other one sees
# which it is.
#' @keywords internal
.check_run_length <- function(n_iter, warmup, where,
                              counts = c("total", "post"),
                              n_iter_name = "n_iter",
                              warmup_name = "warmup") {
  counts <- match.arg(counts)
  is_count <- function(v) {
    is.numeric(v) && length(v) == 1L && !is.na(v) && is.finite(v) &&
      v == round(v)
  }
  if (!is_count(n_iter) || !is_count(warmup)) {
    stop(sprintf("%s(): `%s` and `%s` must each be a single whole number.",
                 where, n_iter_name, warmup_name), call. = FALSE)
  }
  if (identical(counts, "total")) {
    if (n_iter < 2 || warmup < 0 || warmup >= n_iter) {
      stop(sprintf(paste0(
        "Need 0 <= %s < %s and %s >= 2 in %s(); got %s = %d, %s = %d. ",
        "Here `%s` counts every iteration, warmup included, so the run keeps ",
        "%s - %s draws: for 1000 kept draws after 1000 warmup, pass ",
        "%s = 2000, %s = 1000."),
        warmup_name, n_iter_name, n_iter_name, where,
        n_iter_name, as.integer(n_iter), warmup_name, as.integer(warmup),
        n_iter_name, n_iter_name, warmup_name, n_iter_name, warmup_name),
        call. = FALSE)
    }
  } else if (n_iter < 1 || warmup < 0) {
    stop(sprintf(paste0(
      "Need %s >= 1 and %s >= 0 in %s(); got %s = %d, %s = %d. Here `%s` ",
      "counts the kept iterations only, and warmup runs on top of them."),
      n_iter_name, warmup_name, where, n_iter_name, as.integer(n_iter),
      warmup_name, as.integer(warmup), n_iter_name), call. = FALSE)
  }
  invisible(NULL)
}

# Subset a validated front-door control list to the keys an inner fitter
# accepts, so wholesale forwarding does not carry front-door-only knobs
# (grid shape, backend selection) into the inner fitter's narrower check.
#' @keywords internal
.control_subset <- function(control, allowed) {
  if (is.null(control) || length(control) == 0L) return(list())

  nm <- names(control)
  for (from in names(.CONTROL_ALIASES)) {
    to <- .CONTROL_ALIASES[[from]]
    # Rewrite only when the target fitter wants the other spelling, and only
    # when it has not been given explicitly -- an explicit value wins over one
    # arriving under an alias. Tried in both directions, since either spelling
    # may be the one a given fitter reads.
    if (from %in% nm && !(from %in% allowed) &&
        to %in% allowed && !(to %in% nm)) {
      nm[nm == from] <- to
    } else if (to %in% nm && !(to %in% allowed) &&
               from %in% allowed && !(from %in% nm)) {
      nm[nm == to] <- from
    }
  }
  names(control) <- nm

  control[intersect(names(control), allowed)]
}

#' Validate a `control = list()` surface against its canonical key set
#'
#' Front-door fitters across the `tulpa*` ecosystem carry their perf /
#' numerical / tuning knobs in a single `control` list (design principle 6).
#' Without a name check a misspelled knob is a silent no-op --
#' `control$adaptve_grid` fits the default and reports nothing. This validates
#' the names a caller supplied against the whitelist of what the target fitter
#' actually reads, and errors listing the allowed set.
#'
#' Consumer packages (`tulpaRatio`, `tulpaObs`) call this with their own key
#' registry rather than reimplementing the check.
#'
#' @param control The `control` list to validate. `NULL` and empty lists pass.
#' @param allowed Character vector of accepted knob names.
#' @param where Name of the calling fitter, used in the error message.
#'
#' @return `invisible(NULL)`, called for the side effect of erroring on an
#'   unknown or unnamed knob.
#'
#' @examples
#' tulpa_check_control(list(max_iter = 50), c("max_iter", "tol"), "my_fit")
#' try(tulpa_check_control(list(max_itr = 50), c("max_iter", "tol"), "my_fit"))
#'
#' @export
tulpa_check_control <- function(control, allowed, where) {
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
  # `control$key` reads the first of two same-named entries, so a repeated knob
  # set a value the fit never saw (#896).
  dup <- unique(nm[duplicated(nm)])
  if (length(dup)) {
    stop(sprintf(
      "control knob(s) given more than once in %s(): %s. Set each once.",
      where, paste(sQuote(dup, q = FALSE), collapse = ", ")), call. = FALSE)
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
                       "k_tail_points",
                       "diagnose_skew", "skew_idx", "skew_correct",
                       "auto_recenter", "subspace_debias", "cila",
                       "max_grid_cells", "within_cell",
                       "prune", "prune_tol", "prune_log_gap", "screen_iters",
                       "fitted_var",
                       "checkpoint", progress),
    nested_laplace_joint = c(
      "max_iter", "tol", "n_threads", "n_threads_outer", "n_threads_scatter",
      "x_init", "verbose", "hessian", "store_Q", "keep_grid_hessians",
      "force_sparse", "inner_factorization",
      "inner_refresh", "integration", "local_ccd", "tile_warm",
      "ccd_budget", "ccd_budget_floor",
      "prune", "prune_tol", "prune_log_gap", "screen_iters", "fitted_var",
      "adaptive_grid", "adaptive_grid_cutoff", "adaptive_grid_edge_thresh",
      "adaptive_grid_max_frac", "adaptive_grid_max_passes",
      "adaptive_grid_min_cells", "adaptive_grid_stride", "axis_refine",
      "var_of_means_consistency", "var_of_means_min_ess",
      "copy_atom_mass", "copy_slab",
      "diagnose_k", "k_samples", "k_threads", "k_quality", "k_refine",
      "k_max_rounds", "k_bootstrap", "k_tail_points", "k_conf_bands",
      "diagnose_skew", "skew_idx", "skew_correct", "auto_recenter",
      "recenter_pilot",
      "subspace_debias", "cila", "max_grid_cells", "within_cell",
      "checkpoint", progress),
    # `tulpa_hyper_grid()`'s own surface: the refinement / consistency passes
    # it drives and nothing else, since the inner solve is the caller's own
    # callback and takes no knobs from here.
    hyper_grid = c("adaptive_grid", "adaptive_grid_edge_thresh",
                   "adaptive_grid_max_passes", "var_of_means_consistency",
                   "var_of_means_min_ess"),
    st_nested = c("n_grid_spatial", "n_grid_temporal", "n_grid_rho",
                  "tau_lower", "tau_upper", "rho_lower", "rho_upper",
                  "rho_spatial", "sigma_lower", "sigma_upper",
                  "max_iter", "tol", "n_threads",
                  "auto_recenter", "within_cell"),
    spde = c("method", "n_grid", "max_iter", "tol", "n_threads",
             "diagnose_k", "k_samples", "k_tail_points", "checkpoint",
             "mode_find",
             # Reachable since the SPDE grid entry went through the shared
             # entry bundle (gcol33/tulpa#699); it hand-rolled its driver call
             # and hardcoded prune_tol = 0 before that.
             "prune", "prune_tol", "prune_log_gap", "screen_iters",
             "fitted_var", "subspace_debias", "cila"),
    re_cov_nested = c("integration", "n_per_axis", "span", "n_draws", "seed",
                      "max_iter", "tol", "n_threads", "diagnose_k",
                      "k_samples", "k_tail_points", "checkpoint",
                      "outer_maxit", "subspace_debias"),
    re_cov_gibbs = c("n_iter", "warmup", "thin", "seed", "max_iter", "tol",
                     "n_threads"),
    # EB stops at the maximizer, so beyond the inner-solve knobs and the outer
    # iteration budget it takes nothing: no integration design, no node count,
    # no draw synthesis. `marginal` requests the hyperparameter-uncertainty
    # correction -- a formal argument of tulpa_eb(), and a control knob here so
    # the tulpa() front door can reach it. The two marginal_* knobs
    # tune the stencil behind it and are inert without it.
    eb = c("max_iter", "tol", "n_threads", "outer_maxit", "outer_reltol",
           "sigma_init", "marginal", "marginal_step", "marginal_richardson"),
    ep = c("max_sweeps", "tol", "damping", "n_quad", "n_draws", "seed"),
    gaussian = c("iter", "warmup", "max_treedepth", "adapt_delta", "seed"),
    gibbs = c("n_iter", "warmup", "thin", "seed", "verbose", "n_threads"),
    nuts_beta = c("n_iter", "n_warmup", "max_treedepth", "adapt_delta",
                  "seed", "verbose"),
    nuts_spde = c("n_iter", "n_warmup", "max_treedepth", "adapt_delta",
                  "seed", "verbose", "mass_matrix", "noncenter"),
    multinomial = c("max_iter", "tol", "n_draws", "seed"),
    ordinal = c("max_iter", "n_draws", "seed"),
    sample_glmm = c("n_iter", "warmup", "n_chains", "seed", "verbose",
                    "epsilon", "L", "adapt_delta", "max_treedepth",
                    "mass_matrix",
                    "n_draws", "alpha", "batch_size", "ess_threshold",
                    "n_particles", "n_mcmc_steps", "mclmc_adjusted",
                    "vi_variant", "vi_mc_samples", "vi_max_iter",
                    "vi_max_grad_norm",
                    # The VI stopping rule. `vi_max_iter` is a ceiling the run
                    # rarely reaches; these three are what decide where it
                    # actually stops (gcol33/tulpa#821).
                    "vi_tol_grad", "vi_tol_rel_elbo", "vi_patience",
                    # Elliptical-slice kernel. Note `ess_threshold` above is
                    # SMC's resampling threshold, not one of these -- the two
                    # unrelated meanings of "ess" are why these carry the
                    # prefix.
                    "ess_adapt_during_warmup",
                    "ess_adapt_interval", "ess_joint_sigma_re",
                    "ess_joint_proposal_sd",
                    # Per-chain checkpoint/resume (NUTS/HMC only; gcol33/tulpa#808).
                    # Same `list(path =, resume =)` shape the nested-Laplace
                    # fitters take, parsed by the same `.nl_checkpoint_args()`.
                    "checkpoint"),
    # The three R log-posterior samplers (gcol33/tulpa#770): each takes its
    # tuning knobs as plain formals, not a `control` list, so tulpa() is the
    # only place that can validate what it forwards. `mala()` / `imh_laplace()`
    # both have a `thin` formal; `pathfinder()` has none.
    mala        = c("n_iter", "warmup", "epsilon", "thin", "seed", "verbose"),
    imh_laplace = c("n_iter", "warmup", "scale", "thin", "seed", "verbose"),
    pathfinder  = c("n_draws", "max_iter", "tol", "seed", "verbose"),
    # agq_fit() is a marginal-likelihood maximizer: no sampler knobs
    # (n_iter / warmup / seed / n_chains / thin) do anything there.
    agq = c("n_quad", "beta_init", "sigma_init", "max_iter", "tol", "verbose"),
    # tulpa_laplace() at a fixed hyperparameter: one inner Newton solve, whose
    # numerical knobs are plain formals. No grid, no draws, no sampler, so
    # nothing else a tulpa() union key names is read there (gcol33/tulpa#870).
    laplace = c("max_iter", "tol", "n_threads")
  )
  # tulpa() dispatches across the nested / spde / re_cov / gibbs / agq /
  # sampler backends and forwards `control` wholesale on the nested and
  # modeldata routes, so its surface is the union plus its own
  # backend-selection and conditioning knobs.
  # tulpa()'s control surface is the union of the backends it dispatches, minus
  # the statistical hyperpriors -- those ride the `re_prior` / `beta_prior`
  # signature arguments (design principle 6: statistical args in the signature,
  # tuning knobs in control), so `tulpa(control = list(prior_sigma = ))` errors
  # and points the user at `re_prior`.
  #
  # The subtraction is defensive rather than active: `prior_sigma` / `eta` /
  # `prior_df` / `prior_scale` are SIGNATURE arguments of the re_cov fitters,
  # not control knobs, so no key set in the union carries them today and the
  # setdiff removes nothing. It stays so that adding one to a backend's set
  # cannot re-open a door on tulpa()'s. (The comment used to say those sets
  # "still list" them, which sent the next reader looking for key sets that do
  # not exist -- gcol33/tulpa#708.)
  .tulpa_hyperprior_keys <- c("prior_sigma", "eta", "prior_df", "prior_scale",
                              "sigma_re_scale", "prior_sigma_scale")
  keys$tulpa <- sort(unique(setdiff(c(
    keys$nested_laplace, keys$nested_laplace_joint, keys$spde,
    keys$re_cov_nested, keys$re_cov_gibbs, keys$eb,
    keys$sample_glmm, keys$ep, keys$nuts_spde,
    keys$mala, keys$imh_laplace, keys$pathfinder, keys$agq, keys$laplace,
    c("re_cov", "n_quad", "sigma_init", "beta_init",
      "scale", "method")
  ), .tulpa_hyperprior_keys)))
  keys
})

# Drop the entries a caller did not set, so a `do.call` onto a fitter takes
# that fitter's OWN formal default rather than one restated at the dispatch
# site. Restating them put the same default in two files, where a bump on one
# side is invisible from the other -- the drift gcol33/tulpa#632 measured on
# `k_samples`, in the shape gcol33/tulpa#676 found it.
#
# NULL is the "unset" marker throughout `control`, so an argument a fitter
# genuinely takes as NULL (an absent init) is unaffected: it means the same
# thing either way.
#' @keywords internal
.drop_null <- function(x) x[!vapply(x, is.null, logical(1))]

# The RE-covariance integrator a call asks for, validated once.
#
# Read at two sites with two different defaults, so the value is resolved here
# rather than `match.arg`-ed at each (gcol33/tulpa#668). Validation is
# unconditional: a typo errors even where the caller's model would not have been
# redirected, instead of being accepted and ignored.
#' @keywords internal
.re_cov_method <- function(control, default) {
  match.arg(control$re_cov %||% default, c("nested", "gibbs", "aghq"))
}

# Valid keys for the `re_prior = list()` statistical argument on tulpa(): the
# random-effect / variance-component hyperpriors that used to hide in control.
#' @keywords internal
# `hyperprior` was documented in ?tulpa and read by the front door
# (R/tulpa.R, the re_cov_nested / eb branches) and omitted here, so
# tulpa_check_control() -- which runs first -- rejected the documented key
# (gcol33/tulpa#667).
.RE_PRIOR_KEYS <- c("prior_sigma", "eta", "prior_df", "prior_scale",
                    "prior_sigma_scale", "sigma_re_scale")
