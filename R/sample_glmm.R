# sample_glmm.R
# ------------------------------------------------------------------------------
# Generic R front-door fitter for the model-agnostic sampler kernels that drive
# a tulpa ModelData: NUTS (gcol33/tulpa#55) and ESS / SGHMC / SGLD / MCLMC / SMC
# / VI (gcol33/tulpa#54). All seven share one C++ entry (cpp_tulpa_sample_glmm),
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
#' @param phi Dispersion/precision passed to the family (held fixed).
#' @param offset Optional fixed additive term on the linear predictor
#'   (`eta = offset + X beta`), length `length(y)`; `NULL` -> no offset.
#' @param fixed_names Optional fixed-effect names for the draw columns.
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
                              offset = NULL,
                              fixed_names = NULL, control = list()) {
  if (is.null(.FAMILY_OPS[[family]])) {
    stop(sprintf("Unknown family '%s'. Supported: %s.",
                 family, paste(family_names(), collapse = ", ")), call. = FALSE)
  }
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
    sigma_beta = control$sigma_beta %||% 10.0,
    n_iter     = as.integer(n_iter),
    n_warmup   = as.integer(warmup),
    seed       = as.integer(control$seed %||% 42L),
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
    offset        = offset
  )

  nm <- fixed_names %||% colnames(X) %||% paste0("beta", seq_len(ncol(X)))
  res$param_names <- nm
  if (!is.null(res$means)) names(res$means) <- nm
  if (!is.null(res$draws) && ncol(res$draws) == length(nm)) {
    colnames(res$draws) <- nm
  }
  class(res) <- c("tulpa_sample_fit", "tulpa_fit")
  res
}
