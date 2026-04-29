# ============================================================================
# Generic EM + Laplace engine with MI/Gibbs correction
#
# Model packages (tulpaOcc, numdenom, ...) provide callbacks for:
#   - e_step: compute latent variable posterior weights
#   - m_step_encode: build submodel data from weights
#   - m_step_submodels: list of submodel specs (formula, design matrix, etc.)
# ============================================================================

#' Fit a latent-variable model via EM + Laplace approximation
#'
#' Generic EM engine. Model packages provide callbacks for the E-step
#' (latent variable posterior) and M-step encoding (how to build submodel
#' data from weights). Each M-step submodel is fit via [tulpa_laplace()].
#'
#' After EM convergence, optional MI or Gibbs correction removes attenuation
#' bias from the soft E-step encoding.
#'
#' @param e_step Function: `function(fits, ...) -> list(weights = numeric, ...)`.
#'   Takes current submodel fits, returns posterior weights for latent variables.
#' @param m_step_encode Function: `function(weights, ...) -> list of submodel specs`.
#'   Each spec is a list with `y`, `n_trials`, `X`, and optional `re_list`, `spatial`,
#'   `obs_weights`. Returns one spec per submodel.
#' @param m_step_extra Optional `function(fits, weights, ...) -> fits`.
#'   Fired once per EM iteration, between the M-step and the next E-step.
#'   Receives the freshly assembled list of `tulpa_laplace()` results
#'   (`fits`), the current E-step weights, and any extra arguments that
#'   the caller routes through `...` (e.g. a `data` bundle). Returns a
#'   list with the same length and names as the input, possibly with
#'   mutated dispersion / shape / precision fields
#'   (e.g. `fits[[k]]$phi`). Use this to update non-`eta` parameters that
#'   fall out of the Laplace M-step (NB overdispersion, Gamma shape, Beta
#'   precision, Gaussian residual SD, ...). When `NULL` (default),
#'   behaviour is unchanged from the no-callback case. The engine
#'   validates that the returned object has the same shape as the input
#'   and raises a clear error otherwise.
#' @param init List with initial parameter estimates for the E-step.
#'   Model-specific; passed as `fits` to the first E-step call.
#' @param z_draw Function: `function(weights, ...) -> integer vector`.
#'   Draws hard latent variable values from weights. Used by MI/Gibbs correction.
#' @param hard_encode Function: `function(z, ...) -> list of submodel specs`.
#'   Like `m_step_encode` but uses hard z draws (unweighted). Used by MI/Gibbs.
#' @param family Character: `"binomial"` (default), `"poisson"`, etc.
#' @param max_iter Maximum EM iterations (default 50).
#' @param tol Convergence tolerance (default 1e-4).
#' @param damping Initial EM damping factor 0-1 (default 0.3).
#' @param correction Post-EM correction: `"auto"`, `"mi"`, `"gibbs"`, or `"none"`.
#' @param n_imputations Number of MI imputations per round (default 20).
#' @param n_rounds Number of MI rounds (default 3). Early rounds use K_early draws
#'   to update z_hat; final round uses full n_imputations.
#' @param K_early Imputations per early MI round (default 5).
#' @param n_gibbs Total Gibbs iterations (default 10).
#' @param n_burn Gibbs burn-in (default n_gibbs/3).
#' @param n_threads Number of threads for Laplace (default 1).
#' @param verbose Print progress (default TRUE).
#' @param ... Passed to e_step, m_step_encode, z_draw, hard_encode.
#'
#' @return A list with:
#'   - `fits`: list of submodel Laplace fits (modes, SEs)
#'   - `weights`: final E-step weights
#'   - `convergence`: list with `converged`, `n_iter`, `history`
#'   - `correction`: MI/Gibbs pooled results (if correction != "none")
#'
#' @export
tulpa_em_laplace <- function(e_step, m_step_encode, init,
                              z_draw = NULL, hard_encode = NULL,
                              m_step_extra = NULL,
                              family = "binomial",
                              max_iter = 50L, tol = 1e-4, damping = 0.3,
                              correction = c("auto", "mi", "gibbs", "none"),
                              n_imputations = 20L, n_rounds = 3L, K_early = 5L,
                              n_gibbs = 10L, n_burn = NULL,
                              n_threads = 1L, verbose = TRUE, ...) {
  correction <- match.arg(correction)
  if (is.null(n_burn)) n_burn <- max(3L, n_gibbs %/% 3L)
  if (!is.null(m_step_extra) && !is.function(m_step_extra)) {
    stop("`m_step_extra` must be NULL or a function(fits, weights, data, ...).",
         call. = FALSE)
  }

  # ---- Phase 1: EM iteration ----
  fits <- init
  weights <- NULL
  prev_delta <- Inf
  damp_current <- damping
  history <- data.frame(iter = integer(), delta = double())

  for (iter in seq_len(max_iter)) {
    # E-step
    e_result <- e_step(fits, ...)
    weights_new <- e_result$weights

    # Damped weight update
    if (!is.null(weights)) {
      weights <- damp_current * weights + (1 - damp_current) * weights_new
    } else {
      weights <- weights_new
    }

    # M-step: encode and fit each submodel
    submodel_specs <- m_step_encode(weights, ...)
    fits_new <- vector("list", length(submodel_specs))
    names(fits_new) <- names(submodel_specs)

    for (k in seq_along(submodel_specs)) {
      spec <- submodel_specs[[k]]
      if (is.null(spec) || length(spec$y) == 0) next

      fits_new[[k]] <- tulpa_laplace(
        y = as.integer(spec$y),
        n_trials = as.integer(spec$n_trials),
        X = spec$X,
        re_list = spec$re_list %||% list(),
        family = family,
        spatial = spec$spatial,
        max_iter = 100L, tol = 1e-6,
        n_threads = n_threads,
        return_hessian = TRUE
      )
    }

    # Optional non-eta parameter update (NB phi, Gamma shape, Beta precision,
    # Gaussian sigma, ...). Fired once per EM iteration, between the M-step
    # and the next E-step. Engine is model-agnostic; the callback owns the
    # update rule and returns a list of the same shape as `fits_new`.
    if (!is.null(m_step_extra)) {
      fits_new <- apply_m_step_extra(m_step_extra, fits_new, weights, ...)
    }

    # Convergence: max change in weights
    fits_old <- fits
    fits <- fits_new

    delta <- if (!is.null(e_result$delta) && !is.na(e_result$delta)) {
      e_result$delta
    } else {
      max(abs(weights_new - weights), na.rm = TRUE)
    }

    # Adaptive damping
    if (!is.na(delta) && is.finite(prev_delta)) {
      if (delta > prev_delta * 1.05) {
        damp_current <- min(0.9, damp_current + 0.15)
      } else if (delta < prev_delta * 0.5) {
        damp_current <- max(0.1, damp_current - 0.1)
      }
    }
    prev_delta <- delta

    history <- rbind(history, data.frame(iter = iter, delta = delta))
    if (verbose) cat(sprintf("  EM iter %d: delta = %.6f (damp = %.2f)\n",
                             iter, delta, damp_current))

    if (delta < tol) {
      if (verbose) cat(sprintf("  EM converged after %d iterations\n", iter))
      break
    }
  }

  em_result <- list(
    fits = fits,
    weights = weights,
    convergence = list(
      converged = delta < tol,
      n_iter = iter,
      history = history
    )
  )

  # ---- Phase 2: MI/Gibbs correction ----
  if (correction == "auto") {
    correction <- "mi"  # Default; model packages can override
  }

  if (correction == "none" || is.null(z_draw) || is.null(hard_encode)) {
    em_result$correction <- NULL
    return(em_result)
  }

  if (correction == "mi") {
    em_result$correction <- mi_correct(
      weights, z_draw, hard_encode, e_step, family,
      n_imputations, n_rounds, K_early, n_threads, verbose, ...
    )
  } else if (correction == "gibbs") {
    em_result$correction <- gibbs_correct(
      weights, z_draw, hard_encode, e_step, family,
      n_gibbs, n_burn, n_threads, verbose, ...
    )
  }

  em_result
}


# ============================================================================
# MI correction: multi-round independent imputations + Rubin's pooling
#
# Round 1..n_rounds-1: K_early imputations, update z_hat between rounds
# Final round: K imputations, full Rubin's pooling
# ============================================================================
mi_correct <- function(weights, z_draw, hard_encode, e_step, family,
                       K, n_rounds, K_early, n_threads, verbose, ...) {
  if (verbose) cat(sprintf("  MI correction: %d rounds (K_early=%d, K_final=%d)\n",
                           n_rounds, K_early, K))

  current_weights <- weights

  for (round in seq_len(n_rounds)) {
    K_round <- if (round < n_rounds) K_early else K
    all_draws <- vector("list", K_round)

    # Collect fitted psi/p across imputations to update z_hat
    all_fits <- vector("list", K_round)

    for (k in seq_len(K_round)) {
      z <- z_draw(current_weights, ...)
      specs <- hard_encode(z, ...)

      draw_k <- fit_submodels(specs, family, n_threads)
      all_draws[[k]] <- draw_k
      all_fits[[k]] <- draw_k
    }

    if (verbose) cat(sprintf("    Round %d: %d imputations done\n", round, K_round))

    # Update z_hat from pooled fits (for next round)
    if (round < n_rounds) {
      pooled <- rubins_pool(all_draws)
      # Use pooled betas to recompute E-step weights
      current_weights <- e_step(pooled, ...)$weights
    }
  }

  rubins_pool(all_draws)
}

# Fit all submodels for one MI/Gibbs draw
fit_submodels <- function(specs, family, n_threads) {
  draw <- vector("list", length(specs))
  names(draw) <- names(specs)
  for (j in seq_along(specs)) {
    spec <- specs[[j]]
    if (is.null(spec) || length(spec$y) == 0) next
    fit <- tulpa_laplace(
      y = as.integer(spec$y),
      n_trials = as.integer(spec$n_trials),
      X = spec$X,
      re_list = spec$re_list %||% list(),
      family = family,
      spatial = spec$spatial,
      max_iter = 100L, tol = 1e-6,
      n_threads = n_threads,
      return_hessian = TRUE
    )
    n_beta <- ncol(spec$X)
    beta <- fit$mode[seq_len(n_beta)]
    se <- rep(NA_real_, n_beta)
    if (!is.null(fit$H_beta)) {
      v <- tryCatch(diag(solve(fit$H_beta[seq_len(n_beta), seq_len(n_beta)])),
                    error = function(e) rep(NA_real_, n_beta))
      se <- sqrt(pmax(v, 0))
    }
    draw[[j]] <- list(beta = beta, se = se)
  }
  draw
}


# ============================================================================
# Gibbs correction: Markov chain z → fit → z → fit → ...
# ============================================================================
gibbs_correct <- function(weights, z_draw, hard_encode, e_step, family,
                          n_iter, n_burn, n_threads, verbose, ...) {
  if (verbose) cat(sprintf("  Gibbs correction: up to %d iter (%d burn-in)\n", n_iter, n_burn))

  current_weights <- weights
  post_burn <- list()
  prev_mean <- NULL

  for (g in seq_len(n_iter)) {
    z <- z_draw(current_weights, ...)
    specs <- hard_encode(z, ...)
    fits_g <- fit_submodels(specs, family, n_threads)

    current_weights <- e_step(fits_g, ...)$weights

    if (g > n_burn) {
      post_burn[[length(post_burn) + 1]] <- fits_g

      # Early stopping: check if running mean has stabilized
      if (length(post_burn) >= 3) {
        pooled <- rubins_pool(post_burn)
        curr_mean <- unlist(lapply(pooled, function(p) p$mean))
        if (!is.null(prev_mean) && length(curr_mean) == length(prev_mean)) {
          max_shift <- max(abs(curr_mean - prev_mean), na.rm = TRUE)
          if (max_shift < 0.01) {
            if (verbose) cat(sprintf("    Gibbs converged at iter %d (shift=%.4f)\n", g, max_shift))
            break
          }
        }
        prev_mean <- curr_mean
      }
    }

    if (verbose) cat(sprintf("    Gibbs iter %d/%d\n", g, n_iter))
  }

  if (length(post_burn) == 0) {
    warning("No post-burn-in Gibbs draws. Increase n_gibbs or decrease n_burn.")
    return(NULL)
  }

  rubins_pool(post_burn)
}


# ============================================================================
# m_step_extra plumbing: invoke the callback and validate its return shape
#
# The callback receives the freshly assembled M-step fits, the current E-step
# weights, and the user data bundle (forwarded through `...` as `data`,
# matching the callback contract documented on `tulpa_em_laplace`). It must
# return a list with the same length and per-element names as the input.
# Per-element shape is otherwise opaque to the engine — the callback owns
# the update rule.
# ============================================================================
apply_m_step_extra <- function(m_step_extra, fits, weights, ...) {
  # Forward `...` verbatim. If the caller routes a `data` bundle through
  # `...`, the callback receives it as a named argument; otherwise `data`
  # is simply absent. This keeps the engine model-agnostic.
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
      stop("`m_step_extra` must preserve submodel names. Expected: ",
           paste(names(fits), collapse = ", "),
           "; got: ",
           paste(names(updated) %||% "<unnamed>", collapse = ", "),
           ".", call. = FALSE)
    }
  }
  updated
}


# ============================================================================
# Rubin's rules: pool K draws
# ============================================================================

#' Pool multiple imputation draws via Rubin's rules
#'
#' @param draws List of K draws. Each draw is a list of submodel results,
#'   each with `beta` (numeric vector) and `se` (numeric vector).
#' @return A list of pooled submodel summaries, each with `mean`, `se`,
#'   `V_within`, `V_between`, `V_total`.
#' @export
rubins_pool <- function(draws) {
  K <- length(draws)
  submodel_names <- names(draws[[1]])
  result <- list()

  for (nm in submodel_names) {
    # Collect betas and SEs across K draws
    betas <- lapply(draws, function(d) {
      if (is.null(d[[nm]])) return(NULL)
      d[[nm]]$beta
    })
    betas <- betas[!vapply(betas, is.null, logical(1))]
    if (length(betas) == 0) next

    ses <- lapply(draws, function(d) {
      if (is.null(d[[nm]])) return(NULL)
      d[[nm]]$se
    })
    ses <- ses[!vapply(ses, is.null, logical(1))]

    K_actual <- length(betas)
    p <- length(betas[[1]])

    beta_mat <- do.call(rbind, betas)  # K x p
    se_mat <- do.call(rbind, ses)      # K x p

    # Rubin's rules
    pooled_mean <- colMeans(beta_mat)
    V_within <- colMeans(se_mat^2, na.rm = TRUE)
    V_between <- apply(beta_mat, 2, var)
    V_total <- V_within + (1 + 1 / K_actual) * V_between
    pooled_se <- sqrt(V_total)

    result[[nm]] <- list(
      mean = pooled_mean,
      se = pooled_se,
      V_within = V_within,
      V_between = V_between,
      V_total = V_total,
      K = K_actual
    )
  }

  result
}
