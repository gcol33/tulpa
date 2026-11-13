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
#' @section Convergence and the Monte Carlo sample size:
#' An MCEM update carries Monte Carlo error, so a rule on the raw change
#' between iterations stalls at the noise floor of the draws however long it
#' runs, and a change relative to `|theta|` asks for unbounded precision on a
#' coefficient near zero. The rule here follows Booth and Hobert (1999):
#'
#' * **Step against noise.** The Monte Carlo standard error of the pooled
#'   update is `mcse_j = sqrt(V_between_j / n_mc)`. When the previous
#'   estimate lies inside the `1 - alpha` confidence ellipsoid of the update,
#'   `sum_j (theta_j - theta_prev_j)^2 / mcse_j^2 < qchisq(1 - alpha, P)`,
#'   the EM step is swamped by Monte Carlo error and the next iteration
#'   draws `n_mc * n_mc_growth` (capped at `n_mc_max`). A step clearly
#'   larger than the noise keeps the draw count.
#' * **Stopping on the posterior scale.** The change is measured against
#'   the pooled standard error `se_j` of each coefficient,
#'   `delta = max_j |theta_j - theta_prev_j| / se_j`, and the fit stops once
#'   `delta < tol` on `n_consecutive` consecutive iterations. A step of
#'   `tol` posterior SDs is small whatever the coefficient's magnitude.
#'
#' The noise in `delta` is about `sqrt(2) mcse_j / se_j`, so `tol` is also
#' the precision the draws have to reach; `history$mcse` reports
#' `max_j mcse_j / se_j` per iteration. A run that ends with `n_mc` at
#' `n_mc_max` and `delta` still at that noise level warns: the cap, not the
#' EM, is what stopped it.
#'
#' Each M-step fits every draw on its own and averages the per-draw
#' estimates. The fixed point is therefore the mean of the complete-data
#' estimators under `p(z | y, theta)`, which agrees with the EM fixed point
#' (the maximizer of the averaged complete-data log-likelihood) to first
#' order in `1/n`.
#'
#' This is an engine block, not a front door: model packages call it
#' programmatically, so its tuning knobs (`max_iter`, `tol`, `n_mc`)
#' sit in the signature rather than in a `control` list.
#'
#' @section Tier:
#' Tier 2 (structured): complete-data Laplace fits pooled over latent draws.
#' Increasing `n_mc` reduces Monte Carlo error in the E-step. The fitted
#' modes, Gaussian conditional approximations and Rubin pooling remain the
#' inferential approximation. An exact E-step sampler alone supplies no
#' posterior correction for those approximations.
#'
#' With `correction = "gibbs"`, the Rubin fit initializes a data-augmentation
#' chain. Each sweep calls `e_step_sample(fits, 1L, ...)` for one exact latent
#' draw and applies true-density MH to the full complete-data parameter vector
#' with a Laplace t proposal. `fits[[k]]$mode` and `$beta` then contain the
#' current sampled parameter values. The target is `p(theta, Z | y)`, conditional
#' on fixed block dispersion and random-effect covariance, and the result is
#' Tier 1. See [tulpa_em_laplace()] for the density contract, supported blocks,
#' parameter layout, computational cost and convergence requirements.
#'
#' @param e_step_sample Function `function(fits, n_mc, ...) -> list`.
#'   Must return a list of length `n_mc`, each element a weights
#'   object of the same shape `m_step_encode` consumes (typically a
#'   numeric vector of length `n_obs`, or a matrix `n_obs x K` for
#'   K-class latent variables). `fits` has the shape the `e_step` of
#'   [tulpa_em_laplace()] receives: a named list with one entry per
#'   submodel, each carrying `mode` (the mean of the per-draw mode vectors of
#'   the previous M-step), `beta` and `se` (the pooled fixed effects and their
#'   standard errors). On the first iteration `fits` is `list()` -- the
#'   callback should return draws from the prior or a starting guess.
#' @param m_step_encode Function `function(weights, ...) -> list of blocks`.
#'   Identical to the contract used by [tulpa_em_laplace()]: each
#'   block is a list with required fields `y`, `X`, `family`, and
#'   optional `n_trials`, `offset`, `phi`, `re_list`, `spatial`,
#'   `weights`. See `?tulpa_em_laplace` for the full spec. A submodel has to
#'   keep one latent layout (the length of its mode vector) across draws.
#' @param n_mc Initial number of Monte-Carlo draws per iteration
#'   (default `10L`).
#' @param n_mc_growth Factor by which `n_mc` grows after an iteration whose
#'   step is swamped by Monte Carlo error (default `4 / 3`, Booth and
#'   Hobert's `1 + 1/r` with `r` set to 3). `1` keeps `n_mc` constant.
#' @param n_mc_max Cap on `n_mc` (default `200L`).
#' @param max_iter Maximum EM iterations (default `30L`).
#' @param tol Convergence tolerance on the largest change between iterations
#'   in units of each coefficient's pooled standard error (default `0.1`).
#' @param alpha Level of the confidence ellipsoid that decides whether a step
#'   is swamped by Monte Carlo error (default `0.25`, Booth and Hobert's
#'   choice).
#' @param n_consecutive Number of consecutive iterations `delta < tol` has to
#'   hold for (default `3L`).
#' @param verbose Print per-iteration progress (default `TRUE`).
#' @param correction `"none"` (default) returns the Tier 2 Rubin estimator.
#'   `"gibbs"` adds true-density MH-within-Gibbs, returning paired parameter
#'   and latent draws from the joint posterior.
#' @inheritParams tulpa_em_laplace
#' @param ... Forwarded to `e_step_sample` and `m_step_encode`.
#'
#' @return A list with:
#'   * `pooled` -- named list of pooled per-submodel summaries
#'     (`mean`, `se`, `V_within`, `V_between`, `V_total`); see
#'     [rubins_pool()].
#'   * `fits` -- the point estimate in the shape `e_step_sample` receives:
#'     per submodel, `mode`, `beta`, `se`.
#'   * `draws` -- with `correction = "none"`, per-MC-draw fit lists from the
#'     final iteration, indexed by submodel. With `"gibbs"`, a matrix of full
#'     parameter draws paired by row with `latent_draws`; `log_prob` stores
#'     the corresponding complete-data log density and `final_fits` the last
#'     sampled state in E-step form.
#'   * `em_draws`, `pooled_target` -- with `"gibbs"`, the original final
#'     MCEM refits and the Tier 2 target of the retained `pooled` summaries.
#'   * `n_iter` -- iterations actually run.
#'   * `n_mc_final` -- `n_mc` used in the last iteration.
#'   * `converged` -- logical.
#'   * `inference_mode`, `inference_tier`, `target` -- `"structured"`, `2L`,
#'     and `"Rubin-pooled complete-data Laplace estimates"` for `"none"`.
#'     Gibbs reports `"exact"`, `1L`, and
#'     `"joint posterior p(theta, Z | y), conditional on fixed block hyperparameters"`.
#'   * `accept_prob`, `mean_accept`, `draws_kind` -- with `"gibbs"`, conditional
#'     MH acceptance indicators on retained sweeps, their mean, and `"chain"`.
#'   * `means`, `sds`, `vcov`, `se` -- with `"gibbs"`, posterior means, SDs,
#'     full parameter covariance and per-block fixed-effect SDs from the chain.
#'   * `history` -- `data.frame(iter, n_mc, delta, mcse, swamped)`: `delta`
#'     the standardized step, `mcse` the largest Monte Carlo standard error
#'     as a fraction of the pooled standard error, `swamped` whether the step
#'     lay inside the Monte Carlo confidence ellipsoid.
#'
#' @references
#' Wei, G. C. G., & Tanner, M. A. (1990). A Monte Carlo implementation
#' of the EM algorithm and the poor man's data augmentation algorithms.
#' *Journal of the American Statistical Association*, 85(411), 699-704.
#' \doi{10.1080/01621459.1990.10474930}.
#'
#' Tanner, M. A., and Wong, W. H. (1987). The calculation of posterior
#' distributions by data augmentation. *JASA*, 82(398), 528-540.
#' \doi{10.1080/01621459.1987.10478458}.
#'
#' Robert, C. P., and Casella, G. (1999). *Monte Carlo Statistical Methods*.
#' Springer, sections 7.3.2-7.3.3. \doi{10.1007/978-1-4757-3071-5}.
#'
#' Andrieu, C., and Roberts, G. O. (2009). The pseudo-marginal approach for
#' efficient Monte Carlo computations. *Annals of Statistics*, 37(2), 697-725.
#' \doi{10.1214/07-AOS574}.
#'
#' Booth, J. G., & Hobert, J. P. (1999). Maximizing generalized linear
#' mixed model likelihoods with an automated Monte Carlo EM algorithm.
#' *JRSS B*, 61(1), 265-285.
#'
#' Caffo, B. S., Jank, W., & Jones, G. L. (2005). Ascent-based Monte Carlo
#' expectation-maximization. *JRSS B*, 67(2), 235-251.
#'
#' @seealso [tulpa_em_laplace()] for the closed-form-weights variant,
#'   [rubins_pool()] for the pooling rule.
#'
#' @examples
#' \donttest{
#' # Two-component Gaussian mixture with unit variances: the latent class z is
#' # sampled in the E-step, and each class mean is an intercept-only Gaussian
#' # block in the M-step.
#' set.seed(1)
#' z <- rbinom(100, 1, 0.4)
#' y <- ifelse(z == 0, rnorm(100, -1), rnorm(100, 2))
#'
#' e_step_sample <- function(fits, n_mc, ...) {
#'   p1 <- if (length(fits) == 0L) {
#'     ifelse(y > median(y), 0.8, 0.2)
#'   } else {
#'     l0 <- dnorm(y, fits$comp0$mode[1], log = TRUE) + log(0.6)
#'     l1 <- dnorm(y, fits$comp1$mode[1], log = TRUE) + log(0.4)
#'     1 / (1 + exp(l0 - l1))
#'   }
#'   lapply(seq_len(n_mc), function(s) rbinom(length(y), 1, p1))
#' }
#' m_step_encode <- function(weights, ...) {
#'   list(comp0 = list(y = y[weights == 0], X = matrix(1, sum(weights == 0), 1),
#'                     family = "gaussian"),
#'        comp1 = list(y = y[weights == 1], X = matrix(1, sum(weights == 1), 1),
#'                     family = "gaussian"))
#' }
#'
#' fit <- tulpa_em_mc(e_step_sample, m_step_encode, n_mc = 5L, max_iter = 8L,
#'                    verbose = FALSE)
#' sapply(fit$pooled, function(p) p$mean)
#' fit$history
#' }
#' @export
tulpa_em_mc <- function(e_step_sample, m_step_encode,
                        n_mc = 10L,
                        n_mc_growth = 4 / 3,
                        n_mc_max = 200L,
                        max_iter = 30L,
                        tol = 0.1,
                        alpha = 0.25,
                        n_consecutive = 3L,
                        verbose = TRUE,
                        correction = c("none", "gibbs"),
                        n_gibbs = 2000L,
                        log_complete = NULL,
                        correction_control = list(),
                        beta_prior = NULL,
                        ...) {
  correction <- match.arg(correction)
  if (correction == "gibbs") {
    .em_gibbs_control(n_gibbs, correction_control)
    .em_check_exact_callbacks(log_complete)
  } else if (!is.null(log_complete) || length(correction_control)) {
    stop("`log_complete` and `correction_control` require correction = 'gibbs'.",
         call. = FALSE)
  }
  if (!is.function(e_step_sample)) {
    stop("`e_step_sample` must be a function(fits, n_mc, ...).",
         call. = FALSE)
  }
  if (!is.function(m_step_encode)) {
    stop("`m_step_encode` must be a function(weights, ...).",
         call. = FALSE)
  }
  n_mc <- as.integer(n_mc)
  if (n_mc < 2L) {
    stop("`n_mc` must be >= 2 (the Monte Carlo error of an update is read ",
         "off the spread of its draws).", call. = FALSE)
  }
  if (n_mc_growth < 1) {
    stop("`n_mc_growth` must be >= 1 (use 1 for constant n_mc).",
         call. = FALSE)
  }
  n_mc_max <- as.integer(n_mc_max)
  if (n_mc_max < n_mc) {
    stop("`n_mc_max` must be >= `n_mc`.", call. = FALSE)
  }
  if (!is.numeric(tol) || length(tol) != 1L || !is.finite(tol) || tol <= 0) {
    stop("`tol` must be a positive number.", call. = FALSE)
  }
  if (!is.numeric(alpha) || length(alpha) != 1L || !(alpha > 0 && alpha < 1)) {
    stop("`alpha` must be a number in (0, 1).", call. = FALSE)
  }
  n_consecutive <- as.integer(n_consecutive)
  if (length(n_consecutive) != 1L || is.na(n_consecutive) ||
      n_consecutive < 1L) {
    stop("`n_consecutive` must be an integer >= 1.", call. = FALSE)
  }

  fits <- list()
  fits_per_draw <- list()
  pooled <- list()
  prev_mean <- numeric(0)
  history <- data.frame(iter = integer(0L), n_mc = integer(0L),
                        delta = double(0L), mcse = double(0L),
                        swamped = logical(0L))
  converged <- FALSE
  iter <- 0L
  n_mc_curr <- n_mc
  n_small <- 0L
  n_mc_next <- n_mc

  for (iter in seq_len(max_iter)) {
    n_mc_curr <- min(n_mc_max, as.integer(ceiling(n_mc_next)))

    # ---- E-step: sample n_mc_curr weights ----
    weight_draws <- e_step_sample(fits, n_mc_curr, ...)
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
      fits_per_draw[[s]] <- .fit_one_draw(
        weight_draws[[s]], m_step_encode,
        label_prefix = sprintf("MCEM draw %d ", s), beta_prior = beta_prior, ...)
    }

    # ---- Pool via Rubin's rules ----
    pooled <- rubins_pool(fits_per_draw)
    fits <- .em_mc_point_fits(fits_per_draw, pooled)

    # ---- Step against Monte Carlo noise and against the posterior SD ----
    step <- .em_mc_step(prev_mean, pooled, n_mc_curr, alpha)
    prev_mean <- step$mean

    history <- rbind(history,
                     data.frame(iter = iter, n_mc = n_mc_curr,
                                delta = step$delta, mcse = step$mcse,
                                swamped = step$swamped))
    if (verbose) {
      message(sprintf(
        "  MCEM iter %d (n_mc = %d): delta = %.4g SE, mcse = %.4g SE%s",
        iter, n_mc_curr, step$delta, step$mcse,
        if (isTRUE(step$swamped)) ", swamped" else ""))
    }

    n_small <- if (is.finite(step$delta) && step$delta < tol) n_small + 1L
               else 0L
    if (n_small >= n_consecutive) {
      converged <- TRUE
      if (verbose) message(sprintf("  MCEM converged after %d iterations",
                                   iter))
      break
    }

    # ---- Grow n_mc when the step is swamped (Booth-Hobert) ----
    if (isTRUE(step$swamped)) n_mc_next <- min(n_mc_max, n_mc_curr * n_mc_growth)
    else n_mc_next <- n_mc_curr
  }

  if (!converged && n_mc_curr >= n_mc_max && nrow(history) &&
      isTRUE(history$swamped[nrow(history)])) {
    warning(sprintf(paste0(
      "tulpa_em_mc() did not converge: the last step (%.3g SE) lies inside ",
      "the Monte Carlo error of its update at n_mc = n_mc_max = %d ",
      "(mcse %.3g SE against tol = %.3g). Raise `n_mc_max` or `tol`."),
      history$delta[nrow(history)], n_mc_max,
      history$mcse[nrow(history)], tol), call. = FALSE)
  }

  result <- list(
    pooled = pooled,
    fits = fits,
    draws = fits_per_draw,
    n_iter = iter,
    n_mc_final = as.integer(n_mc_curr),
    converged = converged,
    inference_mode = "structured",
    inference_tier = 2L,
    target = "Rubin-pooled complete-data Laplace estimates",
    correction = correction,
    history = history
  )
  if (correction == "gibbs") {
    e_step <- function(fits, ...) {
      draw <- e_step_sample(fits, 1L, ...)
      if (!is.list(draw) || length(draw) != 1L) {
        stop("`e_step_sample` must return a list of length 1 in Gibbs correction.",
             call. = FALSE)
      }
      list(weights = draw[[1L]])
    }
    result$em_draws <- result$draws
    result$pooled_target <- result$target
    gibbs <- .gibbs_correction(fits, e_step, m_step_encode, identity,
                                n_gibbs = n_gibbs, verbose = verbose,
                                beta_prior = beta_prior,
                                log_complete = log_complete,
                                correction_control = correction_control, ...)
    result[names(gibbs)] <- gibbs
  }
  result
}


# The point estimate an MCEM iteration hands its E-step, in the shape
# tulpa_em_laplace() hands its own: per submodel, the mean of the per-draw
# mode vectors (the mean of the complete-data estimators) with the pooled
# fixed effects and standard errors beside it.
.em_mc_point_fits <- function(fits_per_draw, pooled) {
  nms <- names(fits_per_draw[[1L]])
  out <- lapply(stats::setNames(nms, nms), function(nm) {
    modes <- lapply(fits_per_draw, function(f) as.numeric(f[[nm]]$mode))
    len <- unique(lengths(modes))
    if (length(len) != 1L) {
      stop(sprintf(paste0(
        "tulpa_em_mc(): submodel '%s' returned mode vectors of different ",
        "lengths across draws (%s); `m_step_encode` has to keep one latent ",
        "layout per submodel."), nm, paste(sort(len), collapse = ", ")),
        call. = FALSE)
    }
    list(mode = colMeans(do.call(rbind, modes)),
         beta = pooled[[nm]]$mean,
         se   = pooled[[nm]]$se)
  })
  out
}


# One MCEM step read against its Monte Carlo error. `mcse_j` is the standard
# error of the pooled mean over the n_mc independent draws. The step is
# swamped when the previous estimate lies inside the 1 - alpha ellipsoid of
# the update (Booth & Hobert 1999); `delta` is the step in units of each
# coefficient's pooled posterior SE (their SE-scaled stopping rule).
.em_mc_step <- function(prev_mean, pooled, n_mc, alpha) {
  mean <- unlist(lapply(pooled, `[[`, "mean"), use.names = FALSE)
  se   <- unlist(lapply(pooled, `[[`, "se"), use.names = FALSE)
  vb   <- unlist(lapply(pooled, `[[`, "V_between"), use.names = FALSE)
  mcse <- sqrt(vb / n_mc)
  mcse_rel <- suppressWarnings(max(mcse / se, na.rm = TRUE))
  if (!is.finite(mcse_rel)) mcse_rel <- NA_real_
  if (!length(prev_mean) || length(prev_mean) != length(mean)) {
    return(list(mean = mean, delta = Inf, mcse = mcse_rel, swamped = NA))
  }
  d <- mean - prev_mean
  delta <- max(abs(d) / se)
  if (!is.finite(delta)) delta <- Inf
  # A coordinate with no Monte Carlo spread contributes nothing when it did
  # not move and makes the step unmistakable when it did.
  z2 <- ifelse(mcse > 0, d^2 / mcse^2, ifelse(d == 0, 0, Inf))
  swamped <- sum(z2) < stats::qchisq(1 - alpha, df = length(d))
  list(mean = mean, delta = delta, mcse = mcse_rel, swamped = swamped)
}
