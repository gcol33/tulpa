# ============================================================================
# Generic EM + Laplace engine.
#
# Model packages (tulpaObs, numdenom, ...) provide callbacks for:
#   - e_step:        compute latent variable posterior weights
#   - m_step_encode: build per-submodel data blocks from weights
#
# Each block is fit independently via tulpa_laplace(); the engine reads
# `family` and `offset` per block so heterogeneous-family mixtures
# (e.g. binomial + poisson hurdle) work without engine-side changes.
# ============================================================================

# Allowed per-submodel families. Extend this list when tulpa_laplace() gains
# new family support; the validator routes off the same set.
.tulpa_em_allowed_families <- c(
  "binomial", "poisson", "gaussian", "negbin", "neg_binomial_2",
  "gamma", "beta"
)


# ----------------------------------------------------------------------------
# Submodel block validator.
#
# Per gcol33/tulpa#3, each entry returned by m_step_encode MUST be a list
# with these fields:
#   - y        numeric                       (required)
#   - X        matrix, nrow == length(y)     (required)
#   - family   character scalar              (required, one of allowed)
#   - n_trials numeric or NULL (default 1)   (optional; absent treated as NULL)
#   - offset   numeric or NULL               (optional; absent treated as NULL)
# Optional:
#   - phi      numeric scalar (dispersion)
#   - re_list, spatial, weights              (passthrough to tulpa_laplace)
#
# Errors are deliberately specific so model packages can debug without
# stepping into the engine.
# ----------------------------------------------------------------------------
.validate_submodel_block <- function(block, idx) {
  prefix <- sprintf("m_step_encode() block %s", idx)

  if (!is.list(block)) {
    stop(sprintf("%s: must be a list, got %s", prefix, class(block)[1]),
         call. = FALSE)
  }

  required <- c("y", "X", "family")
  missing_fields <- setdiff(required, names(block))
  if (length(missing_fields) > 0) {
    stop(sprintf("%s: missing required field(s): %s",
                 prefix, paste(missing_fields, collapse = ", ")),
         call. = FALSE)
  }

  if (!is.numeric(block$y) && !is.integer(block$y)) {
    stop(sprintf("%s: `y` must be numeric, got %s",
                 prefix, class(block$y)[1]),
         call. = FALSE)
  }

  if (!is.matrix(block$X)) {
    stop(sprintf("%s: `X` must be a matrix, got %s",
                 prefix, class(block$X)[1]),
         call. = FALSE)
  }
  if (nrow(block$X) != length(block$y)) {
    stop(sprintf("%s: nrow(X) (%d) != length(y) (%d)",
                 prefix, nrow(block$X), length(block$y)),
         call. = FALSE)
  }

  if (!is.character(block$family) || length(block$family) != 1L) {
    stop(sprintf("%s: `family` must be a character scalar", prefix),
         call. = FALSE)
  }
  if (!block$family %in% .tulpa_em_allowed_families) {
    stop(sprintf("%s: `family` must be one of %s, got '%s'",
                 prefix,
                 paste(shQuote(.tulpa_em_allowed_families), collapse = ", "),
                 block$family),
         call. = FALSE)
  }

  if (!is.null(block$n_trials) &&
      !(is.numeric(block$n_trials) || is.integer(block$n_trials))) {
    stop(sprintf("%s: `n_trials` must be numeric or NULL", prefix),
         call. = FALSE)
  }

  if (!is.null(block$offset) && !is.numeric(block$offset)) {
    stop(sprintf("%s: `offset` must be numeric or NULL", prefix),
         call. = FALSE)
  }
  if (!is.null(block$offset) && length(block$offset) != length(block$y)) {
    stop(sprintf("%s: length(offset) (%d) != length(y) (%d)",
                 prefix, length(block$offset), length(block$y)),
         call. = FALSE)
  }

  if (!is.null(block$phi) &&
      (!is.numeric(block$phi) || length(block$phi) != 1L)) {
    stop(sprintf("%s: `phi` must be a numeric scalar or NULL", prefix),
         call. = FALSE)
  }

  # Optional `prior` field. When present, the block is fit via
  # tulpa_nested_laplace() with hyperparameter integration instead of
  # plain tulpa_laplace(). Must be a single-block prior list (has a
  # `type` field) or a multi-block list (list of single-block lists).
  if (!is.null(block$prior)) {
    if (!is.list(block$prior) || length(block$prior) == 0L) {
      stop(sprintf("%s: `prior` must be a non-empty list", prefix),
           call. = FALSE)
    }
    is_single <- !is.null(block$prior$type)
    is_multi  <- !is_single &&
      all(vapply(block$prior,
                 function(p) is.list(p) && !is.null(p$type),
                 logical(1)))
    if (!is_single && !is_multi) {
      stop(sprintf(
        "%s: `prior` must be a single-block list (with `type`) or a list-of-blocks (each with `type`)",
        prefix), call. = FALSE)
    }
  }

  invisible(TRUE)
}


# ----------------------------------------------------------------------------
# Fit one validated block via tulpa_laplace(), forwarding family / offset
# / phi / re_list / spatial / weights from the block. Single source of truth
# for "block -> tulpa_laplace() call" so MI/Gibbs (when implemented) can
# share it.
# ----------------------------------------------------------------------------
.fit_block_via_laplace <- function(block, n_threads, return_hessian = TRUE,
                                   beta_prior = NULL) {
  n_trials <- if (is.null(block$n_trials)) {
    rep(1L, length(block$y))
  } else {
    block$n_trials
  }

  # Per-block `beta_prior` overrides the EM-level default. The default is
  # applied only to non-spatial blocks (the spatial solvers carry their own
  # fixed-effect prior); an explicit per-block `beta_prior` on a spatial block
  # is forwarded and errors in tulpa_laplace().
  bp <- block$beta_prior
  if (is.null(bp) && is.null(block$spatial)) bp <- beta_prior

  args <- list(
    y         = block$y,
    n_trials = n_trials,
    X         = block$X,
    re_list   = if (is.null(block$re_list)) list() else block$re_list,
    family    = block$family,
    spatial   = block$spatial,
    weights   = block$weights,
    offset    = block$offset,
    n_threads = as.integer(n_threads),
    return_hessian = isTRUE(return_hessian),
    beta_prior = bp
  )
  if (!is.null(block$phi)) args$phi <- block$phi

  do.call(tulpa_laplace, args)
}


# ----------------------------------------------------------------------------
# Fit one validated block via tulpa_nested_laplace(). Used when the block
# carries a `prior` field (single-block list or list-of-blocks). The mode
# vector returned in `$mode` is the latent vector at the grid-weighted
# posterior mean of theta, so .fits_to_param_vec()'s convergence check works
# unchanged.
#
# `weights` / `offset` / multi-term `re_list` are not currently supported by
# tulpa_nested_laplace(); fail loudly if a block sets them. The expected
# pattern from callers is to absorb obs-level weighting into the response
# (e.g. pseudo-binomial encoding) before handing the block to the engine.
# ----------------------------------------------------------------------------
.fit_block_via_nested_laplace <- function(block, n_threads) {
  if (!is.null(block$weights)) {
    stop("`weights` on a nested-Laplace block is not supported. ",
         "Encode the weights into the response (e.g. pseudo-binomial) ",
         "or route the block through plain Laplace.", call. = FALSE)
  }
  if (!is.null(block$offset)) {
    stop("`offset` on a nested-Laplace block is not supported. ",
         "Bake the offset into a covariate column or route the block ",
         "through plain Laplace.", call. = FALSE)
  }
  if (!is.null(block$spatial)) {
    stop("Set `prior` (not `spatial`) on a nested-Laplace block. ",
         "The latent structure is passed via the `prior` field; the ",
         "`spatial` field is for single-Laplace blocks only.", call. = FALSE)
  }

  n_trials <- if (is.null(block$n_trials)) {
    rep(1L, length(block$y))
  } else {
    block$n_trials
  }

  re_idx <- 0L
  n_re_groups <- 0L
  sigma_re <- 1.0
  if (!is.null(block$re_list) && length(block$re_list) > 0L) {
    if (length(block$re_list) > 1L) {
      stop("Multi-term `re_list` on a nested-Laplace block is not yet ",
           "supported. Use a single RE term or fold additional RE ",
           "structure into the multi-block `prior`.", call. = FALSE)
    }
    re_idx      <- as.integer(block$re_list[[1]]$idx)
    n_re_groups <- as.integer(block$re_list[[1]]$n_groups)
    sigma_re    <- as.numeric(block$re_list[[1]]$sigma)
  }

  args <- list(
    y           = block$y,
    n_trials    = n_trials,
    X           = block$X,
    prior       = block$prior,
    re_idx      = re_idx,
    n_re_groups = n_re_groups,
    sigma_re    = sigma_re,
    family      = block$family,
    n_threads   = as.integer(n_threads)
  )
  if (!is.null(block$phi)) args$phi <- block$phi
  # Per-site detection probability for the marginalized `occupancy` family
  # (mu = det_prob * sigma(eta)); the engine then fits the true marginal state
  # likelihood, so the converged Hessian is the calibrated marginal curvature
  # and fitted_eta_var needs no rescaling (see tulpa_nested_laplace()).
  if (!is.null(block$det_prob)) {
    args$det_prob <- as.numeric(block$det_prob)
  }

  fit <- do.call(tulpa_nested_laplace, args)

  # Project the multi-grid result onto a single mode vector so EM-side
  # convergence + downstream `extract_beta()` callers see the same shape
  # as a plain Laplace fit. The fixed-effects block is the first p entries
  # of each per-grid mode; we take the weighted mean across grid points.
  p <- ncol(block$X)
  if (!is.null(fit$modes) && is.matrix(fit$modes)) {
    w <- fit$weights
    if (is.null(w)) w <- rep(1 / nrow(fit$modes), nrow(fit$modes))
    fit$mode <- as.numeric(crossprod(fit$modes, w))
  } else if (is.null(fit$mode)) {
    # Fallback: no per-grid modes stored. Use theta-mean entries from
    # whatever the driver gave us, padded to length p with zeros.
    fit$mode <- numeric(p)
  }

  fit
}


# ----------------------------------------------------------------------------
# Dispatch one block to the right engine based on whether `prior` is set.
# Single point of truth for "block -> fit" so all EM-driven code paths
# (main loop, MI/Gibbs correction) share the same routing.
# ----------------------------------------------------------------------------
.fit_em_block <- function(block, n_threads, return_hessian = TRUE,
                          beta_prior = NULL) {
  if (!is.null(block$prior)) {
    # Nested-Laplace blocks carry their own prior spec; `beta_prior` (a plain
    # fixed-effect penalty) does not apply and is ignored here.
    .fit_block_via_nested_laplace(block, n_threads = n_threads)
  } else {
    .fit_block_via_laplace(block, n_threads = n_threads,
                           return_hessian = return_hessian,
                           beta_prior = beta_prior)
  }
}


# ----------------------------------------------------------------------------
# Track parameter state across iterations for convergence. We summarize each
# submodel fit by its mode vector; the engine compares max relative change
# across submodels.
# ----------------------------------------------------------------------------
.fits_to_param_vec <- function(fits) {
  parts <- lapply(fits, function(f) {
    if (is.null(f) || is.null(f$mode)) return(numeric(0))
    as.numeric(f$mode)
  })
  unlist(parts, use.names = FALSE)
}

.max_rel_change <- function(prev, curr) {
  if (length(prev) == 0 || length(curr) == 0 ||
      length(prev) != length(curr)) return(Inf)
  denom <- pmax(abs(prev), 1e-8)
  max(abs(curr - prev) / denom, na.rm = TRUE)
}


#' Fit a latent-variable model via EM + Laplace approximation
#'
#' Generic EM engine. Model packages provide callbacks for the E-step
#' (latent variable posterior) and the M-step encoding (how to assemble
#' submodel data from weights). Each M-step submodel block is fit
#' independently via [tulpa_laplace()]; the engine reads `family` and
#' `offset` per block so heterogeneous-family mixtures (e.g. a binomial
#' zero submodel + poisson positive submodel for a hurdle model) work
#' without engine changes. See `gcol33/tulpa#3`.
#'
#' @param e_step Callback: `function(fits, ...) -> list(weights = numeric, ...)`.
#'   Called once per EM iteration with the current per-submodel
#'   [tulpa_laplace()] fits. Must return a list whose `weights` element
#'   is the latent-variable posterior used by the next M-step.
#' @param m_step_encode Callback: `function(weights, ...) -> list of blocks`.
#'   Each block is itself a list with the following fields:
#'   * `y` (numeric, required) — response.
#'   * `X` (matrix, required) — fixed-effects design with `nrow(X) == length(y)`.
#'   * `family` (character scalar, required) — one of `"binomial"`,
#'     `"poisson"`, `"gaussian"`, `"negbin"` / `"neg_binomial_2"`, `"gamma"`,
#'     `"beta"`. Forwarded to [tulpa_laplace()] for that block.
#'   * `n_trials` (numeric or `NULL`, optional; absent or `NULL` defaults to 1) —
#'     binomial trial counts.
#'   * `offset` (numeric or `NULL`, optional) — observation-level offset,
#'     length-matched to `y` when non-`NULL`.
#'   * `phi` (numeric scalar, optional) — dispersion forwarded to
#'     [tulpa_laplace()] (used by `negbin`, `gamma`).
#'   * `re_list`, `spatial`, `weights` (optional) — forwarded as-is.
#' @param spatial Optional default spatial spec. Currently unused by the
#'   engine; pass a `spatial` field on each block instead. Reserved so the
#'   API matches the published contract in CLAUDE.md.
#' @param re_list Optional default RE list. Same status as `spatial`.
#' @param max_iter Maximum EM iterations.
#' @param tol Convergence tolerance on max relative parameter change.
#' @param damping EM damping factor in `[0, 1)`. With `damping = d`, the
#'   parameter update is `(1 - d) * new + d * prev`, so `d = 0` is no
#'   damping. The same factor is applied to weights between iterations.
#' @param correction Post-EM correction. `"none"` returns the EM point
#'   estimate only. `"mi"` draws `n_imputations` independent hard `z` from
#'   the converged posterior weights P(z|y, theta_hat), refits each block
#'   on the hard z, and pools via [rubins_pool()]. `"gibbs"` runs a
#'   warm-started `z|theta -> theta|z` Markov chain of length `n_gibbs`
#'   starting from the EM fits, also pooled via [rubins_pool()]. `"auto"`
#'   resolves to `"none"`.
#' @param n_imputations Number of MI draws (default `20L`). Used when
#'   `correction = "mi"`.
#' @param n_gibbs Length of the Gibbs chain (default `10L`). Used when
#'   `correction = "gibbs"`.
#' @param draw_z Optional function `function(weights) -> hard_z` that
#'   turns the E-step's continuous weights into a hard latent draw. Used
#'   only by `correction %in% c("mi", "gibbs")`. The default treats
#'   `weights` as a numeric vector of Bernoulli probabilities and draws
#'   per-observation. Multi-class / matrix-valued latent structures must
#'   supply their own callback.
#' @param m_step_extra Optional `function(fits, weights, ...) -> fits`. Fired
#'   once per M-step in every phase (EM iterations, MI draws, Gibbs steps).
#'   Receives the freshly assembled list of [tulpa_laplace()] results (`fits`),
#'   the continuous E-step weights P(z|y, theta) (NOT the hard z draw used by
#'   MI/Gibbs to encode the block), and any extra arguments forwarded through
#'   `...`. Returns a list with the same length and names as the input,
#'   possibly with mutated dispersion / shape / precision fields (e.g.
#'   `fits[[k]]$phi`). Use this to update non-eta parameters that fall out of
#'   the Laplace M-step (NB overdispersion, Gamma shape, Beta precision,
#'   Gaussian sigma). When `NULL` (default), behavior is unchanged. See
#'   `gcol33/tulpa#4`.
#' @param beta_prior Optional Gaussian prior on the fixed effects, applied to
#'   every block fit via [tulpa_laplace()] (i.e. blocks without a `prior`
#'   field). `NULL` (default) keeps the weak built-in prior. Otherwise a list
#'   with `sd` (required) and optional `mean`; see [tulpa_laplace()]. The same
#'   prior flows into the MI / Gibbs correction refits, so penalized
#'   corrections come for free. A block may override the default by setting its
#'   own `beta_prior` field in `m_step_encode` (e.g. different priors for the
#'   occupancy and detection submodels). Use scalar `mean` / `sd` here when
#'   blocks differ in width; per-coefficient vectors belong on the block.
#' @param verbose Print per-iteration progress.
#' @param ... Forwarded to `e_step`, `m_step_encode`, and `m_step_extra`.
#'
#' @return A list with:
#' \itemize{
#'   \item `fits` — named list of [tulpa_laplace()] results, one per block.
#'   \item `weights` — final E-step weights.
#'   \item `n_iter` — number of EM iterations actually run.
#'   \item `converged` — logical.
#'   \item `history` — `data.frame(iter, delta)` of max relative parameter
#'     change per iteration.
#'   \item `correction` — the resolved correction mode (`"none"`, `"mi"`,
#'     or `"gibbs"`).
#'   \item `pooled` — present when `correction %in% c("mi", "gibbs")`.
#'     Named list of pooled per-submodel summaries from [rubins_pool()].
#'   \item `draws` — present when `correction %in% c("mi", "gibbs")`.
#'     List of per-draw fits with `beta` / `se` attached.
#' }
#'
#' @export
tulpa_em_laplace <- function(e_step, m_step_encode,
                              spatial = NULL, re_list = list(),
                              max_iter = 50L, tol = 1e-4, damping = 0.3,
                              correction = c("auto", "mi", "gibbs", "none"),
                              n_imputations = 20L, n_gibbs = 10L,
                              draw_z = NULL,
                              m_step_extra = NULL,
                              beta_prior = NULL,
                              verbose = TRUE, ...) {
  correction <- match.arg(correction)
  if (correction == "auto") correction <- "none"

  if (!is.null(m_step_extra) && !is.function(m_step_extra)) {
    stop("`m_step_extra` must be NULL or a function(fits, weights, ...).",
         call. = FALSE)
  }
  if (!is.null(draw_z) && !is.function(draw_z)) {
    stop("`draw_z` must be NULL or a function(weights).", call. = FALSE)
  }
  if (correction == "mi") {
    n_imputations <- as.integer(n_imputations)
    if (length(n_imputations) != 1L || is.na(n_imputations) ||
        n_imputations < 2L) {
      stop("`n_imputations` must be an integer >= 2 ",
           "(Rubin's rules need at least 2 draws to estimate ",
           "between-imputation variance).", call. = FALSE)
    }
  }
  if (correction == "gibbs") {
    n_gibbs <- as.integer(n_gibbs)
    if (length(n_gibbs) != 1L || is.na(n_gibbs) || n_gibbs < 2L) {
      stop("`n_gibbs` must be an integer >= 2 ",
           "(Rubin's rules need at least 2 draws to estimate ",
           "between-imputation variance).", call. = FALSE)
    }
  }
  draw_z_fn <- if (is.null(draw_z)) .draw_z_default else draw_z

  if (!is.function(e_step)) {
    stop("`e_step` must be a function", call. = FALSE)
  }
  if (!is.function(m_step_encode)) {
    stop("`m_step_encode` must be a function", call. = FALSE)
  }
  damping <- as.numeric(damping)
  if (length(damping) != 1L || is.na(damping) ||
      damping < 0 || damping >= 1) {
    stop("`damping` must be a numeric scalar in [0, 1)", call. = FALSE)
  }

  fits <- list()
  weights <- NULL
  prev_params <- numeric(0)
  history <- data.frame(iter = integer(0), delta = double(0))
  converged <- FALSE
  iter <- 0L
  delta <- Inf

  for (iter in seq_len(max_iter)) {
    # ---- E-step ----
    e_result <- e_step(fits, ...)
    if (!is.list(e_result) || is.null(e_result$weights)) {
      stop("`e_step` must return a list with a `weights` element",
           call. = FALSE)
    }
    weights_new <- e_result$weights

    # Damped weight update: w <- (1 - d) * w_new + d * w_prev.
    # First iteration just uses w_new (no prev to mix with).
    weights <- if (is.null(weights)) {
      weights_new
    } else {
      (1 - damping) * weights_new + damping * weights
    }

    # ---- M-step ----
    blocks <- m_step_encode(weights, ...)
    if (!is.list(blocks) || length(blocks) == 0) {
      stop("`m_step_encode` must return a non-empty list of blocks",
           call. = FALSE)
    }

    block_names <- names(blocks)
    if (is.null(block_names)) {
      block_names <- paste0("submodel_", seq_along(blocks))
    }

    new_fits <- vector("list", length(blocks))
    names(new_fits) <- block_names

    for (k in seq_along(blocks)) {
      .validate_submodel_block(blocks[[k]],
                               idx = if (nzchar(block_names[k])) {
                                 sprintf("'%s' (#%d)", block_names[k], k)
                               } else {
                                 sprintf("#%d", k)
                               })
      # Damped parameter update is applied via the weights; the M-step itself
      # is a fresh fit on the damped weights, so we don't need to mix beta
      # vectors directly. A block with `$prior` set routes through
      # tulpa_nested_laplace(); otherwise tulpa_laplace().
      new_fits[[k]] <- .fit_em_block(blocks[[k]], n_threads = 1L,
                                     beta_prior = beta_prior)
    }

    fits <- new_fits

    # ---- Optional non-eta parameter update (NB phi, Gamma shape, ...) ----
    # Fired between the M-step and the next E-step; engine is model-agnostic.
    if (!is.null(m_step_extra)) {
      fits <- apply_m_step_extra(m_step_extra, fits, weights, ...)
    }

    # ---- Convergence ----
    curr_params <- .fits_to_param_vec(fits)
    delta <- .max_rel_change(prev_params, curr_params)
    prev_params <- curr_params

    history <- rbind(history,
                     data.frame(iter = iter, delta = delta))

    if (verbose) {
      cat(sprintf("  EM iter %d: delta = %.6g\n", iter, delta))
    }

    if (is.finite(delta) && delta < tol) {
      converged <- TRUE
      if (verbose) {
        cat(sprintf("  EM converged after %d iterations\n", iter))
      }
      break
    }
  }

  result <- list(
    fits       = fits,
    weights    = weights,
    n_iter     = iter,
    converged  = converged,
    history    = history,
    correction = correction
  )

  if (correction == "mi") {
    if (verbose) {
      cat(sprintf("  Running MI correction with %d imputations\n",
                  n_imputations))
    }
    mi <- .mi_correction(
      weights        = weights,
      m_step_encode  = m_step_encode,
      draw_z         = draw_z_fn,
      n_imputations  = n_imputations,
      m_step_extra   = m_step_extra,
      beta_prior     = beta_prior,
      verbose        = verbose,
      ...
    )
    result$pooled <- mi$pooled
    result$draws  <- mi$draws
  } else if (correction == "gibbs") {
    if (verbose) {
      cat(sprintf("  Running Gibbs correction with %d steps\n", n_gibbs))
    }
    gibbs <- .gibbs_correction(
      initial_fits   = fits,
      e_step         = e_step,
      m_step_encode  = m_step_encode,
      draw_z         = draw_z_fn,
      n_gibbs        = n_gibbs,
      m_step_extra   = m_step_extra,
      beta_prior     = beta_prior,
      verbose        = verbose,
      ...
    )
    result$pooled <- gibbs$pooled
    result$draws  <- gibbs$draws
  }

  result
}


# ============================================================================
# Optional non-eta parameter update hook (gcol33/tulpa#4).
#
# The callback receives the freshly assembled M-step fits, the current E-step
# weights, and any extra arguments forwarded through `...`. It must return a
# list with the same length and per-element names as the input. Per-element
# shape is otherwise opaque to the engine — the callback owns the update rule.
# ============================================================================
apply_m_step_extra <- function(m_step_extra, fits, weights, ...) {
  updated <- m_step_extra(fits = fits, weights = weights, ...)

  if (!is.list(updated)) {
    stop("`m_step_extra` must return a list, not ",
         paste(class(updated), collapse = "/"), ".", call. = FALSE)
  }
  if (length(updated) != length(fits)) {
    stop(sprintf(
      "`m_step_extra` returned %d submodel(s); expected %d (one per M-step fit).",
      length(updated), length(fits)
    ), call. = FALSE)
  }
  if (!is.null(names(fits))) {
    if (is.null(names(updated)) || !identical(names(updated), names(fits))) {
      got <- if (is.null(names(updated))) "<unnamed>" else paste(names(updated), collapse = ", ")
      stop("`m_step_extra` must preserve submodel names. Expected: ",
           paste(names(fits), collapse = ", "),
           "; got: ", got, ".", call. = FALSE)
    }
  }
  updated
}


# ============================================================================
# Rubin's rules: pool K imputation draws. Used by the MI and Gibbs corrections
# (.mi_correction / .gibbs_correction) and exported for downstream callers that
# pool their own draws.
# ============================================================================

#' Pool multiple imputation draws via Rubin's rules
#'
#' @description
#' Pools per-imputation block fits via Rubin's rules. When every draw also
#' carries a per-coefficient skewness vector `$gamma`, the third cumulant
#' is pooled by the law of total cumulants
#' \deqn{\kappa_3 = E[\kappa_3(X | k)] + 3\,\mathrm{Cov}(\mu_k, \sigma_k^2) + \kappa_3(\mu_k),}
#' giving a pooled skewness `$gamma` alongside the usual `$mean` / `$se`.
#' If any draw is missing `$gamma`, the third-cumulant path is skipped.
#'
#' @param draws List of K draws. Each draw is a named list of submodel
#'   results, each with `beta` (numeric vector), `se` (numeric vector), and
#'   optionally `gamma` (numeric vector, same length as `beta`).
#' @return A named list of pooled submodel summaries, each with `mean`,
#'   `se`, `V_within`, `V_between`, `V_total`, `K` (number of draws that
#'   contributed), and when applicable `gamma` (pooled skewness) and
#'   `kappa3` (pooled third cumulant).
#' @export
rubins_pool <- function(draws) {
  K <- length(draws)
  if (K == 0L) return(list())
  if (K < 2L) {
    stop("rubins_pool() requires at least 2 draws to estimate ",
         "between-imputation variance; got K = ", K, ".", call. = FALSE)
  }

  submodel_names <- names(draws[[1]])
  result <- list()

  for (nm in submodel_names) {
    betas <- lapply(draws, function(d) {
      if (is.null(d[[nm]])) return(NULL)
      d[[nm]]$beta
    })
    betas <- betas[!vapply(betas, is.null, logical(1))]
    if (length(betas) < 2L) next

    ses <- lapply(draws, function(d) {
      if (is.null(d[[nm]])) return(NULL)
      d[[nm]]$se
    })
    ses <- ses[!vapply(ses, is.null, logical(1))]

    K_actual <- length(betas)
    beta_mat <- do.call(rbind, betas)        # K x p
    se_mat   <- do.call(rbind, ses)          # K x p

    pooled_mean <- colMeans(beta_mat)
    V_within   <- colMeans(se_mat^2, na.rm = TRUE)
    V_between  <- apply(beta_mat, 2, var)
    V_total    <- V_within + (1 + 1 / K_actual) * V_between

    pooled <- list(
      mean      = pooled_mean,
      se        = sqrt(V_total),
      V_within  = V_within,
      V_between = V_between,
      V_total   = V_total,
      K         = K_actual
    )

    # Third-cumulant pooling. Engaged only when every draw supplies a
    # per-coefficient `gamma` vector matching `beta` in length.
    gammas <- lapply(draws, function(d) {
      if (is.null(d[[nm]])) return(NULL)
      d[[nm]]$gamma
    })
    have_gamma <- vapply(gammas, function(g) {
      !is.null(g) && length(g) == ncol(beta_mat) && all(is.finite(g))
    }, logical(1))
    if (length(have_gamma) == K_actual && all(have_gamma)) {
      gamma_mat <- do.call(rbind, gammas)    # K x p
      sigma_mat <- se_mat                    # K x p (sd, not variance)
      mu_centered <- sweep(beta_mat, 2L, pooled_mean, FUN = "-")
      kappa3 <- colMeans(sigma_mat^3 * gamma_mat) +
        3 * colMeans(mu_centered * sigma_mat^2) +
        colMeans(mu_centered^3)
      sigma_pooled <- sqrt(V_total)
      # Guard against division when V_total underflows.
      pooled$gamma  <- ifelse(sigma_pooled > 0, kappa3 / sigma_pooled^3, NA_real_)
      pooled$kappa3 <- kappa3
    }

    result[[nm]] <- pooled
  }

  result
}
