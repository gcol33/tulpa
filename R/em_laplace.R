# ============================================================================
# Generic EM + Laplace engine.
#
# Model packages (tulpaOcc, numdenom, ...) provide callbacks for:
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

  invisible(TRUE)
}


# ----------------------------------------------------------------------------
# Fit one validated block via tulpa_laplace(), forwarding family / offset
# / phi / re_list / spatial / weights from the block. Single source of truth
# for "block -> tulpa_laplace() call" so MI/Gibbs (when implemented) can
# share it.
# ----------------------------------------------------------------------------
.fit_block_via_laplace <- function(block, n_threads, return_hessian = TRUE) {
  n_trials <- if (is.null(block$n_trials)) {
    rep(1L, length(block$y))
  } else {
    block$n_trials
  }

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
    return_hessian = isTRUE(return_hessian)
  )
  if (!is.null(block$phi)) args$phi <- block$phi

  do.call(tulpa_laplace, args)
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
#' @param correction Post-EM correction. Only `"none"` is implemented today;
#'   `"mi"` and `"gibbs"` raise an error pointing at this issue. `"auto"`
#'   resolves to `"none"`.
#' @param n_imputations Reserved for MI correction.
#' @param n_gibbs Reserved for Gibbs correction.
#' @param m_step_extra Optional `function(fits, weights, ...) -> fits`. Fired
#'   once per EM iteration, between the M-step and the next E-step. Receives
#'   the freshly assembled list of [tulpa_laplace()] results (`fits`), the
#'   current E-step weights, and any extra arguments forwarded through `...`.
#'   Returns a list with the same length and names as the input, possibly
#'   with mutated dispersion / shape / precision fields (e.g. `fits[[k]]$phi`).
#'   Use this to update non-eta parameters that fall out of the Laplace
#'   M-step (NB overdispersion, Gamma shape, Beta precision, Gaussian sigma).
#'   When `NULL` (default), behavior is unchanged. See `gcol33/tulpa#4`.
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
#' }
#'
#' @export
tulpa_em_laplace <- function(e_step, m_step_encode,
                              spatial = NULL, re_list = list(),
                              max_iter = 50L, tol = 1e-4, damping = 0.3,
                              correction = c("auto", "mi", "gibbs", "none"),
                              n_imputations = 20L, n_gibbs = 10L,
                              m_step_extra = NULL,
                              verbose = TRUE, ...) {
  correction <- match.arg(correction)
  if (correction == "auto") correction <- "none"

  if (!is.null(m_step_extra) && !is.function(m_step_extra)) {
    stop("`m_step_extra` must be NULL or a function(fits, weights, ...).",
         call. = FALSE)
  }

  # TODO(gcol33/tulpa): implement MI and Gibbs corrections.
  # MI: draw hard z from weights, refit blocks unweighted, pool via
  # rubins_pool(). Gibbs: warm-started Markov chain z|theta -> theta|z.
  if (correction %in% c("mi", "gibbs")) {
    stop(sprintf("correction = '%s' is not yet implemented", correction),
         call. = FALSE)
  }

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
      # is a fresh Laplace fit on the damped weights, so we don't need to
      # mix beta vectors directly.
      new_fits[[k]] <- .fit_block_via_laplace(blocks[[k]], n_threads = 1L)
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

  list(
    fits      = fits,
    weights   = weights,
    n_iter    = iter,
    converged = converged,
    history   = history
  )
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
# Rubin's rules: pool K imputation draws.
#
# Kept as a generic helper for downstream use even though MI/Gibbs corrections
# are not yet wired into tulpa_em_laplace().
# ============================================================================

#' Pool multiple imputation draws via Rubin's rules
#'
#' @param draws List of K draws. Each draw is a named list of submodel
#'   results, each with `beta` (numeric vector) and `se` (numeric vector).
#' @return A named list of pooled submodel summaries, each with `mean`,
#'   `se`, `V_within`, `V_between`, `V_total`, and `K` (number of draws
#'   that contributed).
#' @export
rubins_pool <- function(draws) {
  K <- length(draws)
  if (K == 0L) return(list())

  submodel_names <- names(draws[[1]])
  result <- list()

  for (nm in submodel_names) {
    betas <- lapply(draws, function(d) {
      if (is.null(d[[nm]])) return(NULL)
      d[[nm]]$beta
    })
    betas <- betas[!vapply(betas, is.null, logical(1))]
    if (length(betas) == 0L) next

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

    result[[nm]] <- list(
      mean      = pooled_mean,
      se        = sqrt(V_total),
      V_within  = V_within,
      V_between = V_between,
      V_total   = V_total,
      K         = K_actual
    )
  }

  result
}
