# =============================================================================
# criteria.R -- the generic single-model goodness-of-fit layer. One pointwise
# log-likelihood matrix [n_draws x n_obs] in, the full INLA/`loo` currency out:
# WAIC, DIC, CPO/LPML, PSIS-LOO, and (separately) PIT. This is the one place the
# engine derives lppd / p_waic / per-observation predictive densities, so model
# packages (tulpaObs, tulpaRatio) build their `*_waic` / `*_cpo` on top of it
# instead of re-deriving the arithmetic. The LOO surface reuses the native
# `tulpa_psis()` Pareto-smoothing, so the CPO and PSIS-LOO numbers share one
# computation (CPO_i = exp(elpd_loo_i), LPML = sum_i log CPO_i = elpd_loo).
#
# At consumer scale the [S x N] matrix can be several GB (an EVA cover-hurdle
# fit is [200 x 1.16M]); `tulpa_loglik()` wraps a column-block generator so the
# accumulators stream over observation blocks and never materialize the whole
# matrix. The per-observation reductions are exact regardless of block size.
# =============================================================================

#' Streaming pointwise log-likelihood
#'
#' Wrap a pointwise log-likelihood for [tulpa_criteria()] without materializing
#' the whole `[n_draws x n_obs]` matrix. A plain matrix is wrapped directly; a
#' block generator (a function of an integer column vector returning the
#' `[n_draws x length(cols)]` submatrix) lets the criteria accumulators stream
#' over observation blocks, so an EVA-scale `[200 x 1.16M]` log-likelihood is
#' consumed a few thousand columns at a time.
#'
#' @param x Either a numeric `[n_draws x n_obs]` matrix, an existing
#'   `tulpa_loglik`, or a function `f(cols)` returning the
#'   `[n_draws x length(cols)]` submatrix for the integer column indices `cols`.
#' @param n_obs,n_draws Required when `x` is a generator function; the column
#'   and row counts of the implied matrix.
#' @return A `tulpa_loglik` object: a list with `get(cols)`, `n_obs`,
#'   `n_draws`, and `materialized`.
#' @seealso [tulpa_criteria()]
#' @export
tulpa_loglik <- function(x, n_obs = NULL, n_draws = NULL) {
  if (inherits(x, "tulpa_loglik")) return(x)
  if (is.function(x)) {
    if (is.null(n_obs) || is.null(n_draws)) {
      stop("`n_obs` and `n_draws` are required when `x` is a column-block ",
           "generator.", call. = FALSE)
    }
    return(structure(
      list(get = x, n_obs = as.integer(n_obs), n_draws = as.integer(n_draws),
           materialized = FALSE),
      class = "tulpa_loglik"
    ))
  }
  x <- as.matrix(x)
  if (!is.numeric(x)) {
    stop("`x` must be a numeric matrix, a tulpa_loglik, or a generator ",
         "function.", call. = FALSE)
  }
  structure(
    list(get = function(cols) x[, cols, drop = FALSE],
         n_obs = ncol(x), n_draws = nrow(x), materialized = TRUE),
    class = "tulpa_loglik"
  )
}

# Column-wise log-sum-exp of an [S x m] block: returns length-m vector of
# log(sum_s exp(B[s, j])), numerically stabilized by the column max.
.criteria_col_lse <- function(B) {
  m <- apply(B, 2L, max)
  m[!is.finite(m)] <- 0
  m + log(colSums(exp(B - rep(m, each = nrow(B)))))
}

# Column variances of an [S x m] block (unbiased, divisor S - 1), vectorized.
.criteria_col_var <- function(B) {
  S <- nrow(B)
  cs <- colSums(B)
  (colSums(B * B) - cs * cs / S) / (S - 1)
}

#' Model criteria from a pointwise log-likelihood
#'
#' Turn an `[n_draws x n_obs]` pointwise log-likelihood into the standard
#' Bayesian goodness-of-fit currency: WAIC, DIC, conditional predictive
#' ordinates (CPO) and their sum LPML, PSIS-LOO, all from one matrix. PSIS-LOO
#' reuses the native [tulpa_psis()] smoothing, so the CPO and LOO numbers are
#' the same computation (`CPO_i = exp(elpd_loo_i)`, `LPML = elpd_loo`). The
#' input may be a matrix or a streaming [tulpa_loglik()] so EVA-scale fits are
#' processed in observation blocks.
#'
#' @param log_lik An `[n_draws x n_obs]` numeric matrix of pointwise
#'   log-likelihoods, or a [tulpa_loglik()] streaming wrapper.
#' @param criteria Which criteria to compute. Any of `"waic"`, `"loo"`,
#'   `"cpo"`, `"lpml"`, `"dic"`. `"loo"`, `"cpo"`, and `"lpml"` share the single
#'   PSIS pass; `"dic"` additionally needs `loglik_at_mean`.
#' @param group Optional length-`n_obs` grouping (an integer / factor /
#'   character vector). The LOO unit is **one column of `log_lik`**: with
#'   `group = NULL` (the default) every column is its own fold (leave-one-row-out,
#'   e.g. per plot / per visit) and the result is byte-identical to the
#'   ungrouped call. When supplied, the per-draw pointwise log-likelihoods are
#'   summed within group to a `[n_draws x n_groups]` matrix **before** PSIS, so
#'   each fold is a whole group (leave-one-group-out cross-validation, LOGO-CV).
#'   Use it to switch the estimand from per-row to per-group LOO -- e.g. on a
#'   cell-compressed hierarchical fit, leave out a whole cell rather than one of
#'   its rows. WAIC's variance term, `lppd`, `elpd_loo`, `cpo` and `pareto_k`
#'   are all computed on the grouped matrix. DIC is a plug-in deviance over all
#'   observations and is unaffected by `group`.
#' @param loglik_at_mean Optional length-`n_obs` vector of pointwise
#'   log-likelihoods evaluated at the posterior mean of the parameters, supplied
#'   by the caller (the model package knows the parameterization). Required for
#'   DIC's plug-in deviance; without it the DIC fields are `NA`.
#' @param chunk_size Number of observation columns to process per block. The
#'   default streams the whole matrix at once when materialized, else picks a
#'   block sized to a few million entries.
#' @param pointwise If `TRUE`, also return the per-observation vectors
#'   (`elpd_waic`, `p_waic`, `elpd_loo`, `pareto_k`, `cpo`) for plotting /
#'   stacking.
#' @return A `tulpa_criteria` object: a list with the requested scalar scores
#'   (each estimate paired with its standard error where defined),
#'   `n_draws` / `n_obs`, the PSIS `pareto_k` summary, and -- when
#'   `pointwise = TRUE` -- a `pointwise` data frame.
#' @details
#' `p_waic` is the well-known positively-biased variance estimator at low draw
#' counts; the result records `n_draws` and the count of observations with
#' `p_waic_i > 0.4` (the `loo` heuristic for an unreliable WAIC), and the
#' PSIS-LOO `elpd_loo` is the more stable figure to report when that count is
#' non-trivial.
#'
#' The LOO unit is whatever **one column of `log_lik`** holds. If the consumer
#' built the matrix with one column per row (plot / visit), the default is
#' per-row LOO; if a column already carries a whole group's compressed
#' likelihood, leaving it out drops that group and `pareto_k` can blow up by
#' construction. The `group` argument makes the unit explicit: supply it to
#' aggregate columns into folds and report leave-one-group-out CV (LOGO-CV)
#' instead, a different and deliberate estimand.
#' @references
#' Vehtari, Gelman & Gabry (2017). Practical Bayesian model evaluation using
#' leave-one-out cross-validation and WAIC. \emph{Statistics and Computing}
#' 27(5):1413-1432. Watanabe (2010). Spiegelhalter et al. (2002). Geisser &
#' Eddy (1979).
#' @seealso [tulpa_psis()] for the smoothing core, [tulpa_pit()] for the
#' probability-integral-transform companion, [compare_models()] for
#'   model comparison.
#' @examples
#' # A draws x observations log-likelihood matrix (here built directly;
#' # in practice extracted from a fitted model's posterior draws).
#' set.seed(1)
#' y  <- rnorm(40)
#' mu <- matrix(rnorm(200 * 40, sd = 0.2), 200, 40)
#' ll <- dnorm(matrix(y, 200, 40, byrow = TRUE), mean = mu, log = TRUE)
#' tulpa_criteria(ll)
#' tulpa_criteria(ll, criteria = "waic", pointwise = TRUE)$pointwise[1:3, ]
#' @export
tulpa_criteria <- function(log_lik,
                           criteria = c("waic", "loo", "cpo", "lpml", "dic"),
                           loglik_at_mean = NULL,
                           group = NULL,
                           chunk_size = NULL,
                           pointwise = FALSE) {
  criteria <- match.arg(criteria, several.ok = TRUE)
  ll <- tulpa_loglik(log_lik)
  S <- ll$n_draws
  N <- ll$n_obs
  if (S < 2L) {
    stop("Need at least 2 draws for model criteria; got ", S, ".",
         call. = FALSE)
  }

  want_waic <- "waic" %in% criteria
  want_loo  <- any(c("loo", "cpo", "lpml") %in% criteria)
  want_dic  <- "dic" %in% criteria
  # lppd is shared by WAIC (lppd - p_waic) and LOO (p_loo = lppd - elpd_loo).
  need_lppd <- want_waic || want_loo

  if (is.null(chunk_size)) {
    chunk_size <- if (ll$materialized) N else max(1L, floor(4e6 / S))
  }
  chunk_size <- as.integer(min(max(1L, chunk_size), N))

  # Optional leave-one-group-out folding (LOGO-CV). A column of `log_lik` is the
  # LOO unit; `group` collapses columns into folds by summing the per-draw
  # pointwise log-likelihoods within group (the group's joint conditional
  # log-likelihood given each draw) to a [S x n_groups] matrix BEFORE PSIS. The
  # grouping pass streams over the (possibly EVA-scale) input once and never
  # materializes it; the grouped matrix is bounded by n_groups << n_obs, so the
  # downstream reductions run on it directly. The reduction code below is the
  # single source of truth -- it operates on `red` / `n_fold`, which is the raw
  # input ungrouped and the small grouped matrix otherwise.
  glabels <- NULL
  if (is.null(group)) {
    red    <- ll
    n_fold <- N
  } else {
    if (length(group) != N) {
      stop("`group` must have length n_obs = ", N, "; got ", length(group),
           ".", call. = FALSE)
    }
    gf      <- factor(group)
    g_id    <- as.integer(gf)
    glabels <- levels(gf)
    n_fold  <- nlevels(gf)
    if (n_fold < 1L) stop("`group` has no usable levels.", call. = FALSE)
    G <- matrix(0, S, n_fold)
    for (st in seq.int(1L, N, by = chunk_size)) {
      cols <- st:min(st + chunk_size - 1L, N)
      B    <- ll$get(cols)                  # [S x length(cols)]
      gc   <- g_id[cols]
      for (jj in seq_along(cols)) G[, gc[jj]] <- G[, gc[jj]] + B[, jj]
    }
    red    <- tulpa_loglik(G)
    chunk_size <- n_fold
  }

  lppd_i  <- if (need_lppd) numeric(n_fold) else NULL
  pwaic_i <- if (want_waic) numeric(n_fold) else NULL
  eloo_i  <- if (want_loo)  numeric(n_fold) else NULL
  pk_i    <- if (want_loo)  rep(NA_real_, n_fold) else NULL
  draw_ll_sum <- if (want_dic) numeric(S) else NULL

  starts <- seq.int(1L, n_fold, by = chunk_size)
  for (st in starts) {
    cols <- st:min(st + chunk_size - 1L, n_fold)
    B <- red$get(cols)                      # [S x length(cols)]
    if (need_lppd) lppd_i[cols] <- .criteria_col_lse(B) - log(S)
    if (want_waic) pwaic_i[cols] <- .criteria_col_var(B)
    if (want_dic)  draw_ll_sum <- draw_ll_sum + rowSums(B)
    if (want_loo) {
      for (jj in seq_along(cols)) {
        col <- B[, jj]
        ps  <- tulpa_psis(-col)             # IS weights w_s ~ 1 / p(y | theta_s)
        lw  <- ps$log_weights               # normalized, log-sum-exp == 0
        eloo_i[cols[jj]] <- if (length(lw)) .tulpa_logsumexp(lw + col) else
          (.tulpa_logsumexp(col) - log(S))
        pk_i[cols[jj]] <- ps$pareto_k
      }
    }
  }

  out <- list(n_draws = S, n_obs = N,
              criteria = criteria, has_loglik_at_mean = !is.null(loglik_at_mean))
  if (!is.null(group)) out$n_groups <- n_fold

  if (want_waic) {
    elpd_waic_i <- lppd_i - pwaic_i
    out$lppd <- sum(lppd_i)
    out$p_waic <- sum(pwaic_i)
    out$elpd_waic <- sum(elpd_waic_i)
    out$waic <- -2 * out$elpd_waic
    out$se_elpd_waic <- sqrt(n_fold * stats::var(elpd_waic_i))
    out$se_p_waic <- sqrt(n_fold * stats::var(pwaic_i))
    out$se_waic <- 2 * out$se_elpd_waic
    out$n_high_p_waic <- sum(pwaic_i > 0.4)
  }

  if (want_loo) {
    out$elpd_loo <- sum(eloo_i)
    out$p_loo <- sum(lppd_i) - out$elpd_loo
    out$looic <- -2 * out$elpd_loo
    out$se_elpd_loo <- sqrt(n_fold * stats::var(eloo_i))
    out$se_looic <- 2 * out$se_elpd_loo
    out$pareto_k <- pk_i
    out$n_high_k <- sum(pk_i >= .nl_diag("k_usable"), na.rm = TRUE)
    if ("cpo" %in% criteria || "lpml" %in% criteria) {
      out$lpml <- out$elpd_loo               # sum_i log CPO_i
    }
  }

  if (want_dic) {
    out$dbar <- -2 * mean(draw_ll_sum)       # posterior-mean deviance
    if (!is.null(loglik_at_mean)) {
      if (length(loglik_at_mean) != N) {
        stop("`loglik_at_mean` must have length n_obs = ", N, "; got ",
             length(loglik_at_mean), ".", call. = FALSE)
      }
      out$dhat <- -2 * sum(loglik_at_mean)   # deviance at the posterior mean
      out$p_dic <- out$dbar - out$dhat
      out$dic <- out$dbar + out$p_dic
    } else {
      out$dhat <- NA_real_
      out$p_dic <- NA_real_
      out$dic <- NA_real_
    }
  }

  if (isTRUE(pointwise)) {
    pw <- if (is.null(group)) data.frame(obs = seq_len(n_fold))
          else data.frame(group = glabels)
    if (need_lppd) pw$lppd <- lppd_i
    if (want_waic) {
      pw$elpd_waic <- lppd_i - pwaic_i
      pw$p_waic <- pwaic_i
    }
    if (want_loo) {
      pw$elpd_loo <- eloo_i
      pw$p_loo <- lppd_i - eloo_i
      pw$pareto_k <- pk_i
      pw$cpo <- exp(eloo_i)
    }
    out$pointwise <- pw
  }

  class(out) <- "tulpa_criteria"
  out
}

#' @export
print.tulpa_criteria <- function(x, digits = 1, ...) {
  if (is.null(x$n_groups)) {
    cat(sprintf("tulpa model criteria  (%d draws x %d observations)\n",
                x$n_draws, x$n_obs))
  } else {
    cat(sprintf(paste0("tulpa model criteria  (%d draws x %d observations",
                       " in %d leave-one-group-out folds)\n"),
                x$n_draws, x$n_obs, x$n_groups))
  }
  fmt <- function(est, se) {
    if (is.null(est) || is.na(est)) return("        NA")
    if (is.null(se) || is.na(se)) return(sprintf("%10.*f", digits, est))
    sprintf("%10.*f  (SE %.*f)", digits, est, digits, se)
  }
  if (!is.null(x$waic)) {
    cat(sprintf("  WAIC      %s\n", fmt(x$waic, x$se_waic)))
    cat(sprintf("  elpd_waic %s\n", fmt(x$elpd_waic, x$se_elpd_waic)))
    cat(sprintf("  p_waic    %s\n", fmt(x$p_waic, x$se_p_waic)))
  }
  if (!is.null(x$elpd_loo)) {
    cat(sprintf("  LOOIC     %s\n", fmt(x$looic, x$se_looic)))
    cat(sprintf("  elpd_loo  %s\n", fmt(x$elpd_loo, x$se_elpd_loo)))
    cat(sprintf("  p_loo     %s\n", fmt(x$p_loo, NULL)))
  }
  if (!is.null(x$lpml)) cat(sprintf("  LPML      %s\n", fmt(x$lpml, NULL)))
  if (!is.null(x$dic)) {
    cat(sprintf("  DIC       %s\n", fmt(x$dic, NULL)))
    cat(sprintf("  p_DIC     %s\n", fmt(x$p_dic, NULL)))
  }
  if (!is.null(x$n_high_k) && x$n_high_k > 0L) {
    cat(sprintf("  %d obs with Pareto k >= %s (PSIS-LOO unreliable there)\n",
                format(.nl_diag("k_usable")), x$n_high_k))
  }
  if (!is.null(x$n_high_p_waic) && x$n_high_p_waic > 0L) {
    cat(sprintf("  %d obs with p_waic > 0.4 (WAIC biased; prefer elpd_loo)\n",
                x$n_high_p_waic))
  }
  invisible(x)
}

#' Probability integral transform from a predictive CDF
#'
#' The generic, family-agnostic half of a PIT residual check: the model package
#' supplies the posterior-predictive CDF evaluated at each observation (a
#' `[n_draws x n_obs]` matrix, or a draw-averaged `[n_obs]` vector), and this
#' returns the PIT value per observation. For a discrete or mixed response
#' (a hurdle has a point mass at zero) supply the left limit `cdf_lower`
#' (`P(Y < y)`); the randomized PIT then draws one uniform per observation and
#' interpolates `F(y^-) + U (F(y) - F(y^-))`, which is uniform under a correct
#' model. With `cdf_lower = NULL` the response is treated as continuous and the
#' PIT is the draw-averaged CDF.
#'
#' Supplying `log_lik` switches to the **leave-one-out** PIT (as in INLA's
#' `cpo$pit` or `loo::psis_loo()`'s LOO-PIT): each observation's CDF limits are
#' averaged over draws with PSIS leave-one-out weights (from that
#' observation's own pointwise log-likelihood) instead of equal weights, so
#' the PIT does not use the observation to predict itself. A column whose
#' importance ratio is not all finite falls back to the equal-weight average
#' for that observation.
#'
#' @param cdf Posterior-predictive CDF at the observed value, `P(Y <= y)`. A
#'   `[n_draws x n_obs]` matrix (averaged over draws here) or an `[n_obs]`
#'   vector.
#' @param cdf_lower Optional left-limit CDF `P(Y < y)`, same shape as `cdf`, for
#'   the randomized PIT of a discrete / mixed response.
#' @param jitter If `TRUE` (default) and `cdf_lower` is `NULL`, add a tiny
#'   uniform jitter to break ties from a discretized CDF; ignored when
#'   `cdf_lower` is supplied (the interpolation already randomizes).
#' @param log_lik Optional `[n_draws x n_obs]` matrix of the pointwise
#'   log-likelihood at each draw. When supplied, the PIT is the
#'   **leave-one-out** PIT: `cdf` / `cdf_lower` are reweighted by that
#'   observation's PSIS leave-one-out weights instead of being column-averaged.
#' @param tail_points Optional override for the PSIS tail size used by the
#'   leave-one-out weighting (see [tulpa_psis()]); `NULL` uses the automatic
#'   rule. Ignored unless `log_lik` is supplied.
#' @param n_threads Number of threads for the leave-one-out weighting (one
#'   observation per thread). Ignored unless `log_lik` is supplied.
#' @return Numeric vector of length `n_obs` of PIT values in `[0, 1]`.
#' @seealso [tulpa_criteria()], [tulpa_psis()]
#' @export
tulpa_pit <- function(cdf, cdf_lower = NULL, jitter = TRUE, log_lik = NULL,
                       tail_points = NULL, n_threads = 1L) {
  # A vector CDF is one draw; treat it as a 1-row matrix so the kernel's
  # column-mean recovers it. The randomization (runif) runs in cpp_tulpa_pit in
  # the same index order, so results are unchanged under a fixed seed.
  as_mat <- function(z) if (is.matrix(z)) z else matrix(as.numeric(z), 1L)
  cdfm <- as_mat(cdf)

  if (!is.null(log_lik)) {
    llm <- as_mat(log_lik)
    if (nrow(llm) != nrow(cdfm)) {
      stop("`log_lik` and `cdf` must have the same number of draws (rows).",
           call. = FALSE)
    }
    if (ncol(llm) != ncol(cdfm)) {
      stop("`log_lik` and `cdf` imply different numbers of observations.",
           call. = FALSE)
    }
    clm <- if (is.null(cdf_lower)) cdfm else as_mat(cdf_lower)
    if (ncol(clm) != ncol(cdfm)) {
      stop("`cdf_lower` and `cdf` imply different numbers of observations.",
           call. = FALSE)
    }
    tail_len <- .psis_tail_len(nrow(llm), tail_points)
    pit <- cpp_psis_loo_pit(llm, clm, cdfm, as.integer(tail_len),
                             as.integer(n_threads))
    if (is.null(cdf_lower) && isTRUE(jitter)) {
      pit <- pmin(1, pmax(0, pit + stats::runif(length(pit), 0, 1e-6)))
    }
    return(pit)
  }

  if (!is.null(cdf_lower)) {
    clm <- as_mat(cdf_lower)
    if (ncol(clm) != ncol(cdfm)) {
      stop("`cdf_lower` and `cdf` imply different numbers of observations.",
           call. = FALSE)
    }
    cpp_tulpa_pit(cdfm, clm, TRUE, isTRUE(jitter))
  } else {
    cpp_tulpa_pit(cdfm, matrix(0, 0L, 0L), FALSE, isTRUE(jitter))
  }
}


# ==============================================================================
# One-verb doors onto the criteria layer
# ==============================================================================

#' DIC, CPO, WAIC and PSIS-LOO on a fit
#'
#' Generic front doors onto the two criteria [tulpa_criteria()] computes that
#' the \pkg{loo} package owns no generic for. WAIC and PSIS-LOO have theirs
#' (`loo::waic()`, `loo::loo()`), so a model package registers methods on
#' those rather than on new names that would mask them.
#'
#' `pointwise_loglik()` is the one door onto the `[n_draws x n_obs]` matrix
#' itself, the input every criterion above is computed from. [compare_models()]
#' and [model_average()] call it (through an internal wrapper) rather than
#' assuming the engine's own fit layout, so a model package that registers
#' `pointwise_loglik.<its fit class>()` -- alongside its own `waic()` / `loo()`
#' / `dic()` / `cpo()` methods -- reaches model comparison and averaging too.
#'
#' The default methods take a draws x observations pointwise log-likelihood
#' matrix, the same input [tulpa_criteria()] takes. A model package registers a
#' method taking its own fit object, builds the matrix from the posterior, and
#' delegates here.
#'
#' The `tulpa_fit` methods build that matrix from the fit itself, from the same
#' source `compare_models(criterion = "waic")` reads: a `log_lik` the backend
#' stored with its draws, else the family density evaluated at posterior draws
#' of the in-sample linear predictor (sampler draws, the outer-grid mixture of a
#' nested-Laplace fit, or Gaussian draws at the fixed-effect mode and
#' covariance). The draws are pinned to a fixed internal seed, so repeated
#' calls return the same numbers and leave the session RNG untouched.
#' `dic()` plugs in the posterior mean of the linear predictor, so its
#' `p_dic` counts effective parameters in the linear predictor's
#' parameterization; where the fit stored its `log_lik` and carries no linear
#' predictor draws, the DIC fields are `NA` and `dbar` is still reported.
#' A fit with no pointwise log-likelihood (no stored response, a family that is
#' not one built-in family name, or a linear predictor the fit cannot
#' reproduce) is refused with an error naming the fit's class and the reason.
#'
#' `loo::waic()` and `loo::loo()` dispatch to the `tulpa_fit` methods once
#' \pkg{loo} is loaded, and return \pkg{loo}'s own `waic` / `psis_loo` objects,
#' so [loo::loo_compare()] reads them. On an MCMC chain fit `loo()` passes
#' relative effective sample sizes computed over the fit's chains; on an i.i.d.
#' or approximation fit the draws are independent and `r_eff` is 1.
#'
#' @param object,x A pointwise log-likelihood matrix (draws x observations), or
#'   a fitted model object a method is registered for, such as a `tulpa_fit`.
#' @param loglik_at_mean Length-`n_obs` vector of pointwise log-likelihoods at
#'   the posterior mean of the parameters. Required for DIC's plug-in deviance;
#'   without it the DIC fields are `NA`.
#' @param ndraws Number of posterior draws the matrix is evaluated at. Defaults
#'   to all stored draws, or 400 on the draw-free Laplace tier; a smaller
#'   number subsamples the stored ones. Read by the `tulpa_fit` method of
#'   `pointwise_loglik()`, which is the door the criteria below reach the
#'   matrix through.
#' @param ... For `dic()` and `cpo()`, passed to [tulpa_criteria()] (e.g.
#'   `group`, `chunk_size`). For `waic()` and `loo()`, passed to \pkg{loo}'s
#'   matrix methods (e.g. `cores`, `save_psis`).
#' @return `dic()` and `cpo()` return a `tulpa_criteria` object. `waic()`
#'   returns a \pkg{loo} `waic` object and `loo()` a \pkg{loo} `psis_loo`
#'   object.
#' @seealso [tulpa_criteria()] for every criterion at once and for what the LOO
#'   unit means; [compare_models()] to rank several fits.
#' @examples
#' set.seed(1)
#' y  <- rnorm(40)
#' mu <- matrix(rnorm(200 * 40, sd = 0.2), 200, 40)
#' ll <- dnorm(matrix(y, 200, 40, byrow = TRUE), mean = mu, log = TRUE)
#' cpo(ll)
#' \donttest{
#' d <- data.frame(x = rnorm(120))
#' d$y <- rpois(120, exp(0.4 + 0.6 * d$x))
#' fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
#' dic(fit)
#' cpo(fit)$lpml
#' if (requireNamespace("loo", quietly = TRUE)) loo::waic(fit)
#' }
#' @name criteria_doors
NULL

#' @rdname criteria_doors
#' @export
pointwise_loglik <- function(object, ...) {
  UseMethod("pointwise_loglik")
}

#' @rdname criteria_doors
#' @export
pointwise_loglik.default <- function(object, ...) {
  stop(sprintf(paste0("pointwise_loglik(): no method for an object of ",
                      "class %s; a model package registers ",
                      "pointwise_loglik.<its fit class>()."),
               paste(class(object), collapse = "/")), call. = FALSE)
}

#' @rdname criteria_doors
#' @export
pointwise_loglik.tulpa_fit <- function(object, ndraws = NULL, ...) {
  .tulpa_pointwise_loglik(object, ndraws = ndraws,
                          caller = "pointwise_loglik()")
}

#' @rdname criteria_doors
#' @export
dic <- function(object, ...) {
  UseMethod("dic")
}

#' @rdname criteria_doors
#' @export
dic.default <- function(object, loglik_at_mean = NULL, ...) {
  .criteria_matrix_or_stop(object, "dic()")
  tulpa_criteria(object, criteria = "dic", loglik_at_mean = loglik_at_mean, ...)
}

#' @rdname criteria_doors
#' @export
dic.tulpa_fit <- function(object, ...) {
  parts <- .tulpa_loglik_parts(object, caller = "dic()")
  dic.default(parts$loglik, loglik_at_mean = parts$loglik_at_mean, ...)
}

#' @rdname criteria_doors
#' @export
cpo <- function(object, ...) {
  UseMethod("cpo")
}

#' @rdname criteria_doors
#' @export
cpo.default <- function(object, ...) {
  .criteria_matrix_or_stop(object, "cpo()")
  tulpa_criteria(object, criteria = c("loo", "cpo", "lpml"),
                 pointwise = TRUE, ...)
}

#' @rdname criteria_doors
#' @export
cpo.tulpa_fit <- function(object, ...) {
  cpo.default(.tulpa_pointwise_loglik(object, caller = "cpo()"), ...)
}

#' @rdname criteria_doors
#' @exportS3Method loo::waic
waic.tulpa_fit <- function(x, ...) {
  loo::waic(.tulpa_pointwise_loglik(x, caller = "waic()"), ...)
}

#' @rdname criteria_doors
#' @exportS3Method loo::loo
loo.tulpa_fit <- function(x, ...) {
  ll <- .tulpa_pointwise_loglik(x, caller = "loo()")
  loo::loo(ll, r_eff = .tulpa_loglik_r_eff(x, ll), ...)
}

# The default doors take a pointwise log-likelihood; anything else reaching them
# is an object no method is registered for, and is named as such.
.criteria_matrix_or_stop <- function(object, caller) {
  if (inherits(object, "tulpa_loglik") || is.function(object) ||
      is.numeric(tryCatch(as.matrix(object), error = function(e) NULL))) {
    return(invisible(NULL))
  }
  stop(sprintf(paste0("%s: `object` must be a numeric draws x observations ",
                      "log-likelihood matrix, a tulpa_loglik, a column-block ",
                      "generator, or a fit with a registered method; got an ",
                      "object of class %s."),
               caller, paste(class(object), collapse = "/")), call. = FALSE)
}

# The pointwise log-likelihood of a fit, [n_draws x n_obs], with the
# log-likelihood at the posterior-mean predictors DIC reads (NULL where the
# backend stored its `log_lik` directly). A stored `log_lik` wins, then the
# two-process `log_lik_num` + `log_lik_denom`; a categorical fit evaluates the
# log probability of each observed class at its stored parameter draws;
# otherwise the built-in family density is evaluated at the fit's in-sample
# linear-predictor draws. The draws are pinned to a fixed seed so a criteria
# read is repeatable and RNG-neutral. A fit with no pointwise log-likelihood is
# refused, naming its class and the reason.
.tulpa_loglik_parts <- function(object, ndraws = NULL,
                                caller = "pointwise log-likelihood") {
  refuse <- function(why) {
    backend <- if (is.list(object)) object[["backend"]]
    stop(sprintf("%s: the fit of class %s%s has no pointwise log-likelihood: %s.",
                 caller, paste(class(object), collapse = "/"),
                 if (is.null(backend)) "" else sprintf(" (backend '%s')", backend),
                 why), call. = FALSE)
  }
  if (!is.list(object)) refuse("it is not a fitted-model object")

  draws <- object[["draws"]]
  if (is.list(draws) && !is.data.frame(draws)) {
    proc <- Filter(Negate(is.null), draws[c("log_lik_num", "log_lik_denom")])
    ll <- draws[["log_lik"]] %||% (if (length(proc)) Reduce(`+`, proc))
    if (!is.null(ll)) return(list(loglik = as.matrix(ll), loglik_at_mean = NULL))
  }

  # The response and the family, resolved the one way the predictive readers
  # resolve them: flat on a `tulpa()` fit, on the arm of a one-arm joint fit
  # (gcol33/tulpa#850). A fit with no single response process resolves neither
  # and falls through to the refusals below, which name what is missing.
  proc   <- tryCatch(.tulpa_response_process(object, caller),
                     error = function(e) NULL)
  y      <- proc$y      %||% object[["y"]]
  family <- proc$family %||% object[["family"]]
  if (is.null(y)) refuse("it stores no response `$y`")
  if (inherits(object, "tulpa_categorical")) {
    # The class probabilities are a function of the parameter vector through
    # linear predictors (and, ordinal, the cutpoints), so the posterior-mean
    # predictors are those of the posterior-mean parameter vector.
    theta <- object[["draws"]]
    if (!is.null(ndraws) && ndraws < nrow(theta)) {
      .preserve_seed_in_frame()
      set.seed(285603L)
      theta <- theta[sample.int(nrow(theta), ndraws), , drop = FALSE]
    }
    return(list(
      loglik = .categorical_loglik(object, theta),
      loglik_at_mean = as.numeric(
        .categorical_loglik(object, matrix(colMeans(theta), 1L)))))
  }
  if (!is.character(family) || length(family) != 1L) {
    refuse(paste0("its family is not a single built-in family name, so there ",
                  "is no engine density to evaluate at the draws"))
  }
  eta <- tryCatch(.tulpa_eta_draws(object, ndraws = ndraws,
                                   synth_seed = 285603L),
                  error = identity)
  if (inherits(eta, "error")) {
    refuse(paste0("its linear-predictor draws could not be formed (",
                  conditionMessage(eta), ")"))
  }
  if (ncol(eta) != length(y)) {
    refuse(sprintf(paste0("its linear predictor has %d columns for %d ",
                          "responses"), ncol(eta), length(y)))
  }
  ll <- tryCatch(.tulpa_eta_loglik(object, eta), error = identity)
  if (inherits(ll, "error")) {
    refuse(paste0("the '", family, "' density could not be evaluated (",
                  conditionMessage(ll), ")"))
  }
  zi <- attr(eta, "logit_zi")
  at_mean <- .tulpa_eta_loglik(
    object, matrix(colMeans(eta), 1L),
    logit_zi = if (!is.null(zi)) matrix(colMeans(zi), 1L))
  list(loglik = ll, loglik_at_mean = as.numeric(at_mean))
}

# The [n_draws x n_obs] matrix alone, for the criteria readers that do not
# need the linear predictor.
.tulpa_pointwise_loglik <- function(object, ndraws = NULL,
                                    caller = "pointwise log-likelihood") {
  .tulpa_loglik_parts(object, ndraws = ndraws, caller = caller)$loglik
}

# The fit's response log-density at each row of a [n_rows x n_obs] linear
# predictor, against the stored response, trials and dispersion. On a
# zero-inflated fit `logit_zi` is the structural-zero logit drawn with `eta`
# (the "logit_zi" attribute .tulpa_eta_draws() attaches), and the density is the
# mixture's.
.tulpa_eta_loglik <- function(object, eta, logit_zi = attr(eta, "logit_zi")) {
  S <- nrow(eta)
  n <- ncol(eta)
  proc <- .tulpa_response_process(object, "pointwise log-likelihood")
  Y  <- matrix(as.numeric(proc$y), S, n, byrow = TRUE)
  NT <- matrix(as.numeric(proc$n_trials %||% 1), S, n, byrow = TRUE)
  # The same per-replicate dispersion posterior_predict() samples at, so a fit
  # that integrated a dispersion axis scores its draws under the value each one
  # was drawn at rather than under one scalar (gcol33/tulpa#825). A fit with no
  # such axis gets its scalar back and the density is evaluated unchanged.
  phi <- .tulpa_phi_draws(object, proc, attr(eta, "cells"), S)
  PHI <- if (length(unique(phi)) == 1L) phi[1L] else matrix(phi, S, n)
  ll <- .response_loglik(eta, logit_zi, Y, proc$family, n_trials = NT,
                         phi = PHI, phi2 = proc$phi2)
  dim(ll) <- dim(eta)
  ll
}

# Relative effective sample sizes for PSIS-LOO. Rows of the pointwise matrix
# are the fit's own draws in stored order whenever the fit carries draws, so on
# an MCMC chain fit they are read per chain; independent draws have r_eff = 1.
.tulpa_loglik_r_eff <- function(fit, ll) {
  if (!.tulpa_is_chain(fit) || is.null(.fit_draws(fit))) return(1)
  loo::relative_eff(exp(ll), chain_id = .tulpa_chain_id(fit, nrow(ll)))
}
