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
    out$n_high_k <- sum(pk_i >= 0.7, na.rm = TRUE)
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
    cat(sprintf("  %d obs with Pareto k >= 0.7 (PSIS-LOO unreliable there)\n",
                x$n_high_k))
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
#' @param cdf Posterior-predictive CDF at the observed value, `P(Y <= y)`. A
#'   `[n_draws x n_obs]` matrix (averaged over draws here) or an `[n_obs]`
#'   vector.
#' @param cdf_lower Optional left-limit CDF `P(Y < y)`, same shape as `cdf`, for
#'   the randomized PIT of a discrete / mixed response.
#' @param jitter If `TRUE` (default) and `cdf_lower` is `NULL`, add a tiny
#'   uniform jitter to break ties from a discretized CDF; ignored when
#'   `cdf_lower` is supplied (the interpolation already randomizes).
#' @return Numeric vector of length `n_obs` of PIT values in `[0, 1]`.
#' @seealso [tulpa_criteria()]
#' @export
tulpa_pit <- function(cdf, cdf_lower = NULL, jitter = TRUE) {
  # A vector CDF is one draw; treat it as a 1-row matrix so the kernel's
  # column-mean recovers it. The randomization (runif) runs in cpp_tulpa_pit in
  # the same index order, so results are unchanged under a fixed seed.
  as_mat <- function(z) if (is.matrix(z)) z else matrix(as.numeric(z), 1L)
  cdfm <- as_mat(cdf)
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
