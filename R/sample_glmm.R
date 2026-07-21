# sample_glmm.R
# ------------------------------------------------------------------------------
# Generic R front-door fitter for the model-agnostic sampler kernels that drive
# a tulpa ModelData: NUTS and ESS / SGHMC / SGLD / MCLMC / SMC
# / VI. All seven share one C++ entry (cpp_tulpa_sample_glmm),
# which builds the ModelData once through the built-in-family spec scaffold and
# dispatches to the requested kernel. This R wrapper assembles that call from a
# design-matrix bundle and returns a `tulpa_fit` carrying the backend's draws.
#
# Scope: fixed-effect GLMs over the built-in family set (binomial / poisson /
# gaussian / neg_binomial_2 / ...). Random-effect terms route to the conditional
# R-closure logpost backends (mode = "mala" / "pathfinder" / "imh_laplace"),
# which condition on sigma_re; see tulpa() and `build_glmm_logpost()`.
# ------------------------------------------------------------------------------

#' Fit a fixed-effect GLM with a model-agnostic sampler kernel
#'
#' @description
#' Drives one of tulpa's ModelData sampler kernels -- NUTS (`"hmc"`), elliptical
#' slice sampling (`"ess"`), SGHMC (`"sghmc"`), SGLD (`"sgld"`), MCLMC
#' (`"mclmc"`), sequential Monte Carlo (`"smc"`), or variational inference
#' (`"vi"`) -- on a fixed-effect GLM. The model (design + per-observation
#' likelihood) is built once through the same built-in-family scaffold the
#' single-point Laplace fit uses, so no likelihood / link logic is duplicated.
#'
#' @param y Response vector.
#' @param n_trials Binomial denominators (or `NULL` -> all 1).
#' @param X Fixed-effect design matrix (`nrow(X) == length(y)`).
#' @param family Character family name (see [family_names()]).
#' @param backend One of `"hmc"`, `"ess"`, `"sghmc"`, `"sgld"`, `"mclmc"`,
#'   `"smc"`, `"vi"`.
#' @param phi Dispersion/precision passed to the family (held fixed). The
#'   kernel parameterization: for `gaussian` / `lognormal` this is the residual
#'   SD (the [tulpa()] front door passes `sqrt(phi)`, its `phi` being the
#'   variance); for `t` the scale.
#' @param phi2 Optional second dispersion: the Student-t degrees of freedom
#'   (`family = "t"`; default 4 when `NULL`).
#' @param offset Optional fixed additive term on the linear predictor
#'   (`eta = offset + X beta`), length `length(y)`; `NULL` -> no offset.
#' @param fixed_names Optional fixed-effect names for the draw columns.
#' @param re_spec Optional random-effect spec: a list with `idx` (list of
#'   per-term 1-based group-index vectors), `ngroups`, `ncoefs`, `correlated`
#'   (per-term), and `Z` (per-term RE design or `NULL`). `NULL` -> no RE.
#' @param spatial_spec Optional areal spatial spec: a list with `type`
#'   (`"icar"`/`"bym2"`), `spatial_idx`, `n_spatial_units`, `adj_row_ptr`,
#'   `adj_col_idx`, `n_neighbors`, and `scale_factor` (BYM2). `NULL` -> none.
#' @param temporal_spec Optional temporal spec: a list with `type`
#'   (`"rw1"`/`"rw2"`/`"ar1"`), `time_idx`, `n_times`, `n_groups`, `group_idx`,
#'   and `cyclic`. `NULL` -> none.
#' @param sigma_re_scale Half-Cauchy scale for the RE / BYM2 standard-deviation
#'   hyperprior (sampled jointly with the latent effects).
#' @param warm_start Optional `list(init, inv_metric_diag)` seeding the NUTS
#'   kernel: `init` an `n_chains x total_params` matrix of initial positions,
#'   one row per chain, and `inv_metric_diag` a positive vector of length
#'   `total_params` used as the starting inverse mass (warmup adaptation still
#'   runs from it). Build it with `.build_warm_start()` against
#'   `cpp_tulpa_glmm_layout()` rather than by hand -- the entries are positional
#'   and the layout owns the positions. Only the NUTS/HMC kernel takes one;
#'   any other backend errors rather than sampling from the default start.
#' @param control List of kernel tuning knobs (`n_iter`, `warmup`, `seed`,
#'   `sigma_beta`, `n_chains`, `max_treedepth`, `adapt_delta`, `epsilon`, `L`,
#'   `batch_size`, `alpha`, `n_particles`, `n_mcmc_steps`, `ess_threshold`,
#'   `vi_variant`, `vi_mc_samples`, `vi_max_iter`, `n_draws`, `verbose`).
#'
#' @return A `tulpa_fit` with `draws`, `means`, `param_names`, the kernel's
#'   diagnostics, and (for `"hmc"`) `chain_id` / `n_chains` so chain diagnostics
#'   apply.
#' @keywords internal
tulpa_sample_glmm <- function(y, n_trials, X, family, backend, phi = 1.0,
                              phi2 = NULL,
                              offset = NULL, fixed_names = NULL,
                              re_spec = NULL, spatial_spec = NULL,
                              temporal_spec = NULL, svc_spec = NULL,
                              tvc_spec = NULL, zi_spec = NULL,
                              sigma_re_scale = 2.5,
                              sigma_beta = 10.0,
                              warm_start = NULL,
                              control = list()) {
  .check_control(control, .CONTROL_KEYS$sample_glmm, "tulpa_sample_glmm")
  .family_or_stop(family)
  if (!is.null(phi2)) .phi2_or_stop(family, phi2)
  X <- as.matrix(X)
  N <- length(y)
  if (!is.null(offset)) {
    offset <- as.numeric(offset)
    if (length(offset) != N) {
      stop(sprintf("length(offset) (%d) must equal length(y) (%d).",
                   length(offset), N), call. = FALSE)
    }
  }
  n_iter  <- control$n_iter %||% 2000L
  warmup  <- control$warmup %||% (n_iter %/% 2L)

  res <- cpp_tulpa_sample_glmm(
    y          = as.numeric(y),
    n_trials   = if (is.null(n_trials)) rep(1L, N) else as.integer(n_trials),
    X          = X,
    family     = family,
    backend    = backend,
    phi        = as.numeric(phi),
    sigma_beta = as.numeric(sigma_beta),
    n_iter     = as.integer(n_iter),
    n_warmup   = as.integer(warmup),
    seed       = as.integer(control$seed %||% sample.int(.Machine$integer.max, 1L)),
    verbose    = isTRUE(control$verbose),
    n_chains   = as.integer(control$n_chains %||% 4L),
    max_treedepth = as.integer(control$max_treedepth %||% 10L),
    adapt_delta   = as.numeric(control$adapt_delta %||% 0.8),
    epsilon       = as.numeric(control$epsilon %||% 0.0),
    L             = as.integer(control$L %||% 10L),
    batch_size    = as.integer(control$batch_size %||% 0L),
    alpha         = as.numeric(control$alpha %||% 0.1),
    mclmc_adjusted = as.integer(control$mclmc_adjusted %||% 0L),
    n_particles   = as.integer(control$n_particles %||% 1000L),
    n_mcmc_steps  = as.integer(control$n_mcmc_steps %||% 5L),
    ess_threshold = as.numeric(control$ess_threshold %||% 0.5),
    vi_variant    = as.integer(control$vi_variant %||% 3L),
    vi_mc_samples = as.integer(control$vi_mc_samples %||% 10L),
    vi_max_iter   = as.integer(control$vi_max_iter %||% 10000L),
    vi_n_draws    = as.integer(control$n_draws %||% 2000L),
    offset_nullable = offset,
    re_spec       = re_spec,
    spatial_spec  = spatial_spec,
    temporal_spec = temporal_spec,
    sigma_re_scale = as.numeric(sigma_re_scale),
    fixed_names   = fixed_names %||% colnames(X),
    phi2          = phi2 %||% NA_real_,
    svc_spec      = svc_spec,
    tvc_spec      = tvc_spec,
    zi_spec       = zi_spec,
    init_nullable = warm_start$init,
    inv_metric_diag_nullable = warm_start$inv_metric_diag
  )

  # The C++ kernel names every column of the full parameter vector (fixed effects
  # + latent effects + variance-component hyperparameters) via the ParamLayout,
  # so the draws / means carry their own names. Fall back to fixed-effect names
  # only when the draw matrix is empty (no columns named).
  nm <- colnames(res$draws)
  if (is.null(nm)) {
    nm <- fixed_names %||% colnames(X) %||% paste0("beta", seq_len(ncol(X)))
  }
  res$param_names <- nm
  class(res) <- c("tulpa_sample_fit", "tulpa_fit")
  res
}
