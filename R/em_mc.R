# ============================================================================
# Generic Monte-Carlo EM engine.
#
# Sibling of `tulpa_em_laplace`. Same `m_step_encode` callback contract;
# the E-step is replaced by a *sampler* that returns n_mc weight draws
# per iteration. The M-step refits each draw, then results are pooled
# via Rubin's rules so uncertainty reflects both within-imputation
# Laplace variance and between-imputation Monte-Carlo variance.
#
# Use when:
#   * The latent variable distribution is multimodal or far from
#     Gaussian (Laplace E-step is biased).
#   * The conditional expectation in the E-step has no closed form and
#     can only be approximated by Monte Carlo.
#
# Use `tulpa_em_laplace` instead when the E-step has a closed form
# (e.g., conditional mean of a known mixture distribution).
# ============================================================================


#' Generic Monte-Carlo EM driver
#'
#' @description
#' Same M-step plumbing as [tulpa_em_laplace()]: every iteration calls
#' `m_step_encode(weights, ...)` to assemble per-submodel blocks and
#' fits each block via [tulpa_laplace()]. The difference is the E-step:
#' instead of computing closed-form weights, an `e_step_sample`
#' callback returns `n_mc` weight draws per iteration. Each draw is
#' run through the M-step independently and the resulting parameter
#' estimates are pooled via [rubins_pool()].
#'
#' Convergence criterion is the max relative change in *pooled* M-step
#' parameter estimates between iterations. To increase Monte-Carlo
#' accuracy as iterations progress (Booth-Hobert ascent-based MCEM),
#' set `n_mc_growth > 1`.
#'
#' @section Tier:
#' Inherits the tier of the inner M-step (Laplace ⇒ Tier 2). The
#' Monte-Carlo E-step *itself* is exact in the limit `n_mc -> infinity`,
#' so as a *full pipeline* MCEM is asymptotically Tier 1 if
#' `e_step_sample` is exact.
#'
#' @param e_step_sample Function `function(fits, n_mc, ...) -> list`.
#'   Must return a list of length `n_mc`, each element a weights
#'   object of the same shape `m_step_encode` consumes (typically a
#'   numeric vector of length `n_obs`, or a matrix `n_obs × K` for
#'   K-class latent variables). On the first iteration `fits` is
#'   `list()` — the callback should return draws from the prior.
#' @param m_step_encode Function `function(weights, ...) -> list of blocks`.
#'   Identical to the contract used by [tulpa_em_laplace()]: each
#'   block is a list with required fields `y`, `X`, `family`, and
#'   optional `n_trials`, `offset`, `phi`, `re_list`, `spatial`,
#'   `weights`. See `?tulpa_em_laplace` for the full spec.
#' @param n_mc Initial number of Monte-Carlo draws per iteration
#'   (default `10L`).
#' @param n_mc_growth Multiplicative growth of `n_mc` per iteration
#'   (default `1.0` = constant). Set `> 1` for ascent-based MCEM
#'   that ramps up MC accuracy near convergence.
#' @param n_mc_max Cap on `n_mc` (default `200L`).
#' @param max_iter Maximum EM iterations (default `30L`).
#' @param tol Convergence tolerance on max relative change in pooled
#'   parameter estimates (default `1e-3`). Looser than `tulpa_em_laplace`
#'   default because Monte-Carlo noise floors the achievable precision.
#' @param verbose Print per-iteration progress (default `TRUE`).
#' @param ... Forwarded to `e_step_sample` and `m_step_encode`.
#'
#' @return A list with:
#'   * `pooled` — named list of pooled per-submodel summaries
#'     (`mean`, `se`, `V_within`, `V_between`, `V_total`); see
#'     [rubins_pool()].
#'   * `fits` — list of per-MC-draw fit lists from the *final*
#'     iteration, each indexed by submodel name.
#'   * `n_iter` — iterations actually run.
#'   * `n_mc_final` — `n_mc` value used in the last iteration.
#'   * `converged` — logical.
#'   * `history` — `data.frame(iter, n_mc, delta)`.
#'
#' @references
#' Wei, G. C. G., & Tanner, M. A. (1990). A Monte Carlo implementation
#' of the EM algorithm and the poor man's data augmentation algorithms.
#' *Journal of the American Statistical Association*, 85(411), 699–704.
#'
#' Booth, J. G., & Hobert, J. P. (1999). Maximizing generalized linear
#' mixed model likelihoods with an automated Monte Carlo EM algorithm.
#' *JRSS B*, 61(1), 265–285.
#'
#' @seealso [tulpa_em_laplace()] for the closed-form-weights variant,
#'   [rubins_pool()] for the pooling rule.
#'
#' @export
tulpa_em_mc <- function(e_step_sample, m_step_encode,
                        n_mc = 10L,
                        n_mc_growth = 1.0,
                        n_mc_max = 200L,
                        max_iter = 30L,
                        tol = 1e-3,
                        verbose = TRUE,
                        ...) {
  if (!is.function(e_step_sample)) {
    stop("`e_step_sample` must be a function(fits, n_mc, ...).",
         call. = FALSE)
  }
  if (!is.function(m_step_encode)) {
    stop("`m_step_encode` must be a function(weights, ...).",
         call. = FALSE)
  }
  n_mc <- as.integer(n_mc)
  if (n_mc < 1L) stop("`n_mc` must be >= 1.", call. = FALSE)
  if (n_mc_growth < 1) {
    stop("`n_mc_growth` must be >= 1 (use 1 for constant n_mc).",
         call. = FALSE)
  }
  n_mc_max <- as.integer(n_mc_max)
  if (n_mc_max < n_mc) {
    stop("`n_mc_max` must be >= `n_mc`.", call. = FALSE)
  }

  fits_per_draw <- list()
  pooled <- list()
  prev_pooled_means <- numeric(0)
  history <- data.frame(iter = integer(0L), n_mc = integer(0L),
                        delta = double(0L))
  converged <- FALSE
  iter <- 0L

  for (iter in seq_len(max_iter)) {
    n_mc_curr <- min(n_mc_max, as.integer(round(n_mc)))

    # ---- E-step: sample n_mc_curr weights ----
    weight_draws <- e_step_sample(fits_per_draw, n_mc_curr, ...)
    if (!is.list(weight_draws) || length(weight_draws) != n_mc_curr) {
      stop(sprintf(
        "`e_step_sample` must return a list of length n_mc (%d), got %s.",
        n_mc_curr,
        if (is.list(weight_draws)) length(weight_draws)
        else paste(class(weight_draws), collapse = "/")
      ), call. = FALSE)
    }

    # ---- M-step: fit each draw independently ----
    fits_per_draw <- vector("list", n_mc_curr)
    for (s in seq_len(n_mc_curr)) {
      blocks <- m_step_encode(weight_draws[[s]], ...)
      if (!is.list(blocks) || length(blocks) == 0L) {
        stop("`m_step_encode` must return a non-empty list of blocks.",
             call. = FALSE)
      }
      block_names <- names(blocks)
      if (is.null(block_names)) {
        block_names <- paste0("submodel_", seq_along(blocks))
      }
      fits_s <- vector("list", length(blocks))
      names(fits_s) <- block_names
      for (k in seq_along(blocks)) {
        .validate_submodel_block(
          blocks[[k]],
          idx = if (nzchar(block_names[k])) {
            sprintf("'%s' (#%d, draw %d)", block_names[k], k, s)
          } else {
            sprintf("#%d (draw %d)", k, s)
          })
        fit <- .fit_block_via_laplace(blocks[[k]], n_threads = 1L)
        # Build the (beta, se) pair Rubin's pool consumes. tulpa_laplace
        # returns mode = c(beta, RE_values...); pooling is over the
        # fixed-effects block only (RE values are nuisance and aren't
        # comparable across MC draws because the dummy RE indexing
        # depends on the encoded block, not the latent structure).
        n_fixed <- ncol(blocks[[k]]$X)
        beta <- if (is.null(fit$mode) || length(fit$mode) < n_fixed) {
          numeric(0)
        } else {
          fit$mode[seq_len(n_fixed)]
        }
        se <- if (!is.null(fit$H_beta) && length(beta) == n_fixed) {
          tryCatch(sqrt(diag(solve(fit$H_beta)))[seq_len(n_fixed)],
                   error = function(e) rep(NA_real_, n_fixed))
        } else {
          rep(NA_real_, length(beta))
        }
        fit$beta <- beta
        fit$se <- se
        fits_s[[k]] <- fit
      }
      fits_per_draw[[s]] <- fits_s
    }

    # ---- Pool via Rubin's rules ----
    pooled <- rubins_pool(fits_per_draw)

    # ---- Convergence on pooled means ----
    curr_pooled_means <- unlist(lapply(pooled, function(p) p$mean),
                                use.names = FALSE)
    delta <- .max_rel_change(prev_pooled_means, curr_pooled_means)
    prev_pooled_means <- curr_pooled_means

    history <- rbind(history,
                     data.frame(iter = iter, n_mc = n_mc_curr,
                                delta = delta))
    if (verbose) {
      cat(sprintf("  MCEM iter %d (n_mc = %d): delta = %.6g\n",
                  iter, n_mc_curr, delta))
    }
    if (is.finite(delta) && delta < tol) {
      converged <- TRUE
      if (verbose) cat(sprintf("  MCEM converged after %d iterations\n",
                               iter))
      break
    }

    # ---- Grow n_mc for next iter (Booth-Hobert) ----
    n_mc <- min(n_mc_max, n_mc * n_mc_growth)
  }

  list(
    pooled = pooled,
    fits = fits_per_draw,
    n_iter = iter,
    n_mc_final = as.integer(round(n_mc)),
    converged = converged,
    history = history
  )
}
