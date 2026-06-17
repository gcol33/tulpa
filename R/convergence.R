# =============================================================================
# convergence.R -- MCMC convergence diagnostics (generic, model-agnostic).
#
# Rank-normalized split-Rhat, folded split-Rhat, bulk / tail / mean / sd /
# quantile effective sample size, and Monte Carlo standard errors, following
# Vehtari, Gelman, Simpson, Carpenter & Burkner (2021), "Rank-normalization,
# folding, and localization: an improved Rhat for assessing convergence of
# MCMC", Bayesian Analysis 16(2):667-718. Native implementation -- no
# posterior / coda dependency.
#
# The scalar estimators here reproduce posterior::rhat / ess_* / mcse_* to
# machine precision on a fixed draws array (see tests/testthat/test-convergence.R).
#
# `mcmc_diagnostics()` is the single source of truth for Rhat / ESS / MCSE in
# the tulpa ecosystem: the plotting layer (plot_rhat, plot_ess,
# diagnostic_summary, check_diagnostics) and downstream model packages
# (tulpaObs, tulpaRatio) all call it. These statistics are location/scale-
# invariant per parameter, so they may be computed on engine-scale draws.
# =============================================================================

# --- Scalar estimators on a prepared [N_iter x M_chain] matrix --------------

# Autocovariance at lags 0..n-1 for one chain, via FFT (biased estimator,
# divisor n so element 1 is the population variance).
.tulpa_autocov <- function(x) {
  n <- length(x)
  x <- x - mean(x)
  nfft <- 2^ceiling(log2(2 * n))                 # zero-pad to avoid wraparound
  f <- stats::fft(c(x, rep(0, nfft - n)))
  acov <- Re(stats::fft(f * Conj(f), inverse = TRUE)) / nfft
  acov[seq_len(n)] / n
}

# Split each chain in half: [N x M] -> [floor(N/2) x 2M], so a single long
# chain still yields a within/between comparison.
.tulpa_split_chains <- function(sims) {
  n <- nrow(sims)
  if (n < 2L) return(sims)
  half <- n %/% 2L
  cbind(sims[seq_len(half), , drop = FALSE],
        sims[(n - half + 1L):n, , drop = FALSE])
}

# Rank-normalize the pooled draws (average ranks for ties, then the normal
# quantile of the Blom-transformed fractional rank, c = 3/8 so the divisor is
# S + 1/4 -- matches posterior::z_scale).
.tulpa_rank_normalize <- function(sims) {
  r <- rank(c(sims), ties.method = "average")
  s <- length(r)
  z <- stats::qnorm((r - 3 / 8) / (s + 1 / 4))
  matrix(z, nrow = nrow(sims), ncol = ncol(sims))
}

# Fold draws about their pooled median (detects scale / tail divergence).
.tulpa_fold <- function(sims) abs(sims - stats::median(c(sims)))

# Classic Gelman-Rubin potential scale reduction on a prepared (split,
# rank-normalized) [N x M] matrix.
.tulpa_rhat_basic <- function(sims) {
  n <- nrow(sims); m <- ncol(sims)
  if (n < 2L || m < 2L) return(NA_real_)
  chain_mean <- colMeans(sims)
  chain_var  <- apply(sims, 2L, stats::var)
  W <- mean(chain_var)
  if (!is.finite(W) || W <= 0) return(NA_real_)
  B <- n * stats::var(chain_mean)                # var_between
  sqrt((B / W + n - 1) / n)
}

# Bulk Rhat: rank-normalized split-Rhat.
.tulpa_rhat_bulk <- function(sims) {
  .tulpa_rhat_basic(.tulpa_rank_normalize(.tulpa_split_chains(sims)))
}

# Folded Rhat: rank-normalized split-Rhat of the folded draws.
.tulpa_rhat_fold <- function(sims) {
  .tulpa_rhat_basic(.tulpa_rank_normalize(.tulpa_split_chains(.tulpa_fold(sims))))
}

# Improved Rhat: the larger of the bulk and folded values (Vehtari et al. 2021).
.tulpa_rhat <- function(sims) {
  rb <- .tulpa_rhat_bulk(sims)
  rf <- .tulpa_rhat_fold(sims)
  if (is.na(rb) && is.na(rf)) return(NA_real_)
  max(rb, rf, na.rm = TRUE)
}

# Effective sample size for a [N x M] matrix, via the multi-chain
# autocorrelation estimator and Geyer's initial monotone-sequence truncation
# (the estimator used by Stan and posterior::ess_rfun).
.tulpa_ess <- function(sims) {
  if (is.null(dim(sims))) sims <- matrix(sims, ncol = 1L)
  n <- nrow(sims); m <- ncol(sims)
  if (n < 4L) return(NA_real_)
  acov <- vapply(seq_len(m), function(j) .tulpa_autocov(sims[, j]), numeric(n))
  chain_mean <- colMeans(sims)
  mean_var <- mean(acov[1L, ]) * n / (n - 1)
  if (!is.finite(mean_var) || mean_var <= 0) return(NA_real_)
  var_plus <- mean_var * (n - 1) / n
  if (m > 1L) var_plus <- var_plus + stats::var(chain_mean)
  if (!is.finite(var_plus) || var_plus <= 0) return(NA_real_)

  rho_hat_t <- numeric(n)
  rho_hat_t[1L] <- 1                                   # lag 0
  rho_hat_odd <- 1 - (mean_var - mean(acov[2L, ])) / var_plus
  rho_hat_t[2L] <- rho_hat_odd
  rho_hat_even <- 1
  t <- 0L
  while (t < n - 5L && is.finite(rho_hat_even + rho_hat_odd) &&
         (rho_hat_even + rho_hat_odd) > 0) {
    t <- t + 2L
    rho_hat_even <- 1 - (mean_var - mean(acov[t + 1L, ])) / var_plus
    rho_hat_odd  <- 1 - (mean_var - mean(acov[t + 2L, ])) / var_plus
    if ((rho_hat_even + rho_hat_odd) >= 0) {
      rho_hat_t[t + 1L] <- rho_hat_even
      rho_hat_t[t + 2L] <- rho_hat_odd
    }
  }
  max_t <- t
  if (rho_hat_even > 0) rho_hat_t[max_t + 1L] <- rho_hat_even
  # initial monotone sequence: pair sums must be non-increasing
  k <- 0L
  while (k <= max_t - 4L) {
    k <- k + 2L
    if (rho_hat_t[k + 1L] + rho_hat_t[k + 2L] >
        rho_hat_t[k - 1L] + rho_hat_t[k]) {
      rho_hat_t[k + 1L] <- (rho_hat_t[k - 1L] + rho_hat_t[k]) / 2
      rho_hat_t[k + 2L] <- rho_hat_t[k + 1L]
    }
  }
  tau_hat <- -1 + 2 * sum(rho_hat_t[seq_len(max_t)]) + rho_hat_t[max_t + 1L]
  ess <- n * m
  tau_bound <- 1 / log10(ess)                          # cap unstable estimates
  if (!is.finite(tau_hat) || tau_hat < tau_bound) tau_hat <- tau_bound
  ess / tau_hat
}

# Bulk ESS: ESS of the rank-normalized split draws.
.tulpa_ess_bulk <- function(sims) {
  .tulpa_ess(.tulpa_rank_normalize(.tulpa_split_chains(sims)))
}

# ESS for a quantile probability: ESS of the split tail-indicator draws.
.tulpa_ess_quantile <- function(sims, prob) {
  if (prob == 1) {
    len <- length(sims)
    prob <- (len - 0.5) / len
  }
  q <- stats::quantile(c(sims), prob, names = FALSE, na.rm = TRUE)
  ind <- matrix(as.numeric(sims <= q), nrow(sims))
  .tulpa_ess(.tulpa_split_chains(ind))
}

# Tail ESS: the smaller of the 5% and 95% tail-quantile ESS values.
.tulpa_ess_tail <- function(sims) {
  min(.tulpa_ess_quantile(sims, 0.05), .tulpa_ess_quantile(sims, 0.95))
}

# ESS for the mean: ESS of the split (non-rank-normalized) draws.
.tulpa_ess_mean <- function(sims) {
  .tulpa_ess(.tulpa_split_chains(sims))
}

# ESS for the standard deviation: ESS of the split squared centred draws.
.tulpa_ess_sd <- function(sims) {
  sc <- sims - mean(c(sims))
  .tulpa_ess(.tulpa_split_chains(sc^2))
}

# Monte Carlo standard error of the posterior mean.
.tulpa_mcse_mean <- function(sims) {
  stats::sd(c(sims)) / sqrt(.tulpa_ess_mean(sims))
}

# Monte Carlo standard error of the posterior standard deviation.
.tulpa_mcse_sd <- function(sims) {
  sc <- sims - mean(c(sims))
  ess <- .tulpa_ess(.tulpa_split_chains(sc^2))
  e_var <- mean(sc^2)
  var_var <- (mean(sc^4) - e_var^2) / ess
  sqrt(var_var / e_var / 4)
}

# Monte Carlo standard error of a posterior quantile (Vehtari et al. 2021,
# beta-quantile interval around the tail-indicator ESS).
.tulpa_mcse_quantile <- function(sims, prob) {
  ess <- .tulpa_ess_quantile(sims, prob)
  p <- c(0.1586553, 0.8413447)                         # +- 1 sd of N(0, 1)
  a <- stats::qbeta(p, ess * prob + 1, ess * (1 - prob) + 1)
  ssims <- sort(c(sims))
  s <- length(ssims)
  th1 <- ssims[max(floor(a[1L] * s), 1L)]
  th2 <- ssims[min(ceiling(a[2L] * s), s)]
  (th2 - th1) / 2
}

# --- Measure registry -------------------------------------------------------
# Each measure maps a prepared [N_iter x M_chain] matrix (and quantile probs)
# to a named numeric vector. Adding a column is O(1): register one entry.
# `probs`-driven measures expand to one column per probability.

.tulpa_diag_measures <- list(
  rhat      = function(sims, probs) c(rhat = .tulpa_rhat(sims)),
  rhat_bulk = function(sims, probs) c(rhat_bulk = .tulpa_rhat_bulk(sims)),
  rhat_fold = function(sims, probs) c(rhat_fold = .tulpa_rhat_fold(sims)),
  ess_bulk  = function(sims, probs) c(ess_bulk = .tulpa_ess_bulk(sims)),
  ess_tail  = function(sims, probs) c(ess_tail = .tulpa_ess_tail(sims)),
  ess_mean  = function(sims, probs) c(ess_mean = .tulpa_ess_mean(sims)),
  ess_sd    = function(sims, probs) c(ess_sd = .tulpa_ess_sd(sims)),
  mcse_mean = function(sims, probs) c(mcse_mean = .tulpa_mcse_mean(sims)),
  mcse_sd   = function(sims, probs) c(mcse_sd = .tulpa_mcse_sd(sims)),
  ess_quantile = function(sims, probs)
    stats::setNames(vapply(probs, function(p) .tulpa_ess_quantile(sims, p),
                           numeric(1)), paste0("ess_q", probs * 100)),
  mcse_quantile = function(sims, probs)
    stats::setNames(vapply(probs, function(p) .tulpa_mcse_quantile(sims, p),
                           numeric(1)), paste0("mcse_q", probs * 100))
)

# Column names a measure set produces, in order (deterministic given probs).
.tulpa_measure_cols <- function(measures, probs) {
  unlist(lapply(measures, function(mn) {
    if (mn == "ess_quantile")  return(paste0("ess_q",  probs * 100))
    if (mn == "mcse_quantile") return(paste0("mcse_q", probs * 100))
    mn
  }), use.names = FALSE)
}

# Evaluate a measure set on one parameter's draws -> named numeric vector.
.tulpa_measure_eval <- function(sims, measures, probs) {
  unlist(lapply(measures, function(mn) .tulpa_diag_measures[[mn]](sims, probs)))
}

# --- Draws provenance gate --------------------------------------------------
# A fit's `$draws` may hold an autocorrelated MCMC chain or an i.i.d. sample
# from a deterministic approximation (a nested-Laplace / VI / SMC fit
# mixture-samples or resamples its posterior). Chain diagnostics (Rhat,
# autocorrelation-ESS) are only meaningful for the former: on i.i.d. draws
# split-Rhat sits at ~1 and ESS ~ n_draws by construction, which reads as a
# clean convergence result while saying nothing about approximation bias. The
# kind is carried explicitly on the fit (`$draws_kind`, stamped by
# `tulpa_dispatch`) or derived from the backend's registry `emits` property.

# Posterior representation kind: "chain", "iid", "point", or NA when unknown.
# A model-package wrapper (e.g. a tobs_fit) may leave its own `draws_kind`
# unset while carrying a nested-Laplace `joint_fit` that stamps the kind; that
# inner tag is read when the top-level tag and backend give no answer.
.tulpa_draws_kind <- function(fit) {
  k <- fit$draws_kind
  if (!is.null(k)) return(k)
  b <- fit$backend
  if (!is.null(b) && !is.null(BACKEND_REGISTRY[[b]])) {
    return(BACKEND_REGISTRY[[b]]$emits %||% NA_character_)
  }
  jk <- fit$joint_fit$draws_kind
  if (!is.null(jk)) return(jk)
  NA_character_
}

# TRUE when MCMC chain diagnostics describe the fit. Unknown provenance counts
# as a chain, so a fit that predates the `emits` tag is never silently refused.
.tulpa_is_chain <- function(fit) {
  k <- .tulpa_draws_kind(fit)
  is.na(k) || identical(k, "chain")
}

# Shared explanation for why chain diagnostics are withheld on a non-chain fit.
.tulpa_non_chain_msg <- function(fit) {
  k <- .tulpa_draws_kind(fit)
  b <- fit$backend %||% "this backend"
  if (identical(k, "point")) {
    return(sprintf("Backend '%s' returns a point summary (mode + covariance), not a posterior sample.", b))
  }
  sprintf(paste0(
    "Backend '%s' returns i.i.d. / resampled draws, not an MCMC chain: Rhat is ",
    "vacuous and ESS = n_draws by construction, so they do not diagnose ",
    "approximation accuracy. Assess this fit with the approximation/coverage ",
    "diagnostics instead (the IMH/Gibbs debias step, PIT residuals, recovery ",
    "checks)."), b)
}

#' Posterior parameter sample from a fit
#'
#' Returns a fit's posterior draws for summary purposes (quantiles, derived
#' quantities, density plots), regardless of how they were produced -- an MCMC
#' chain, a nested-Laplace node mixture, or a variational sample all answer
#' here. For the chain-only view used by convergence diagnostics, see
#' [mcmc_draws()].
#'
#' @param fit A `tulpa_fit` (or subclass) carrying posterior `$draws`.
#' @return The posterior draws matrix/array, or `NULL` if the fit carries none.
#' @seealso [mcmc_draws()], [mcmc_diagnostics()]
#' @export
posterior_sample <- function(fit) {
  fit$draws %||% fit$samples
}

#' MCMC chain draws from a fit
#'
#' Returns a fit's posterior draws only when they form a genuine MCMC chain
#' (`$draws_kind == "chain"`, or an untagged legacy fit); for an i.i.d. /
#' approximation fit (nested Laplace, VI, SMC, ...) it returns `NULL`, because
#' chain diagnostics do not apply. This is the accessor [mcmc_diagnostics()]
#' gates on. For the provenance-agnostic posterior sample used by summaries,
#' see [posterior_sample()].
#'
#' @param fit A `tulpa_fit` (or subclass) carrying posterior `$draws`.
#' @return The chain draws matrix/array, or `NULL` for a non-chain fit.
#' @seealso [posterior_sample()], [mcmc_diagnostics()]
#' @export
mcmc_draws <- function(fit) {
  if (!.tulpa_is_chain(fit)) return(NULL)
  fit$draws %||% fit$samples
}

# --- Draws extraction -------------------------------------------------------

# Split a fit's posterior draws into a list of per-chain [n_draws x n_par]
# matrices. Handles a 3D [iter, chain, param] array, a pooled matrix with a
# `chain_id` row map or an `n_chains` count (contiguous chain-major blocks),
# and the single-chain fallback.
.tulpa_chain_list <- function(fit) {
  draws <- fit$draws
  if (is.null(draws)) draws <- fit$samples
  if (is.null(draws)) return(NULL)

  if (length(dim(draws)) == 3L) {
    pn <- dimnames(draws)[[3L]]
    return(lapply(seq_len(dim(draws)[2L]), function(m) {
      mm <- draws[, m, , drop = TRUE]
      if (is.null(dim(mm))) mm <- matrix(mm, ncol = dim(draws)[3L])
      colnames(mm) <- pn
      mm
    }))
  }

  draws <- as.matrix(draws)
  cid <- fit$chain_id
  nch <- fit$n_chains %||% 1L
  if (!is.null(cid) && length(cid) == nrow(draws)) {
    ks <- sort(unique(cid))
    lapply(ks, function(k) draws[cid == k, , drop = FALSE])
  } else if (nch > 1L && nrow(draws) %% nch == 0L) {
    per <- nrow(draws) %/% nch
    lapply(seq_len(nch), function(k)
      draws[((k - 1L) * per + 1L):(k * per), , drop = FALSE])
  } else {
    list(draws)
  }
}

#' Posterior draws as a 3D array
#'
#' Assembles a fitted model's posterior draws into an `[iteration, chain,
#' parameter]` array -- the layout expected by `bayesplot` and analogous to
#' `posterior::as_draws_array()`. Multiple chains are recognised from a 3D
#' draws array, a `$chain_id` row map, or an `$n_chains` count over chain-major
#' rows; a single pooled chain yields a one-chain array.
#'
#' @param fit A `tulpa_fit` (or subclass) carrying posterior `$draws`.
#' @return A numeric array with dimensions `[n_iter, n_chain, n_param]` and the
#'   parameter names on the third dimension, or `NULL` if the fit carries no
#'   draws. Chains of unequal length are truncated to the shortest.
#' @seealso [mcmc_diagnostics()]
#' @export
tulpa_draws_array <- function(fit) {
  chain_list <- .tulpa_chain_list(fit)
  if (is.null(chain_list)) return(NULL)
  n_iter <- min(vapply(chain_list, nrow, integer(1)))
  p  <- ncol(chain_list[[1L]])
  nm <- colnames(chain_list[[1L]])
  if (is.null(nm)) nm <- paste0("param", seq_len(p))
  arr <- array(
    NA_real_, dim = c(n_iter, length(chain_list), p),
    dimnames = list(NULL, NULL, nm)
  )
  for (m in seq_along(chain_list)) {
    arr[, m, ] <- chain_list[[m]][seq_len(n_iter), , drop = FALSE]
  }
  arr
}

# bayesplot-shaped accessor: list(draws = <3D array>).
get_draws_array <- function(fit) list(draws = tulpa_draws_array(fit))

#' MCMC convergence diagnostics
#'
#' Per-parameter convergence and accuracy diagnostics computed natively from a
#' fitted model's posterior draws: improved Rhat (the maximum of rank-normalized
#' split-Rhat and folded split-Rhat), bulk / tail / mean / sd / quantile
#' effective sample size, and Monte Carlo standard errors. Split-Rhat is defined
#' for a single chain (the chain is split in two), so a result is produced for
#' any number of chains. The estimators follow Vehtari et al. (2021) and
#' reproduce the corresponding `posterior` functions.
#'
#' For a fit whose draws are not an MCMC chain -- an i.i.d. / approximation fit
#' such as nested Laplace, VI, or SMC, where the between-chain Rhat is vacuous
#' and ESS equals the draw count by construction -- this dispatches to
#' [laplace_diagnostics()], returning the PSIS approximation-reliability table
#' (the "did the approximation work" diagnostic) rather than chain mixing.
#' Provenance is read from `$draws_kind` (or the backend's registry `emits`
#' property). Pass a `point` fit (a mode + covariance, no sample) and the
#' function returns `NULL` with a message.
#'
#' @param fit A `tulpa_fit` (or subclass) carrying posterior `$draws`. Multiple
#'   chains are recognised from a 3D `[iter, chain, param]` draws array, a
#'   `$chain_id` row map, or an `$n_chains` count over chain-major rows.
#' @param pars Optional character vector of parameter names to restrict to.
#' @param measures Character vector selecting which diagnostics to compute, in
#'   output-column order. Available: `"rhat"`, `"rhat_bulk"`, `"rhat_fold"`,
#'   `"ess_bulk"`, `"ess_tail"`, `"ess_mean"`, `"ess_sd"`, `"mcse_mean"`,
#'   `"mcse_sd"`, `"ess_quantile"`, `"mcse_quantile"`. Defaults to the core set
#'   `c("rhat", "ess_bulk", "ess_tail")`.
#' @param probs Numeric probabilities for the quantile-based measures
#'   (`"ess_quantile"`, `"mcse_quantile"`); each expands to one column named
#'   e.g. `ess_q5`, `ess_q95`. Default `c(0.05, 0.95)`.
#' @return A data frame with a `parameter` column followed by one column per
#'   requested measure. Entries are `NA` for parameters that are constant or
#'   have too few draws.
#' @references Vehtari, Gelman, Simpson, Carpenter & Burkner (2021).
#'   Rank-normalization, folding, and localization: an improved Rhat for
#'   assessing convergence of MCMC. \emph{Bayesian Analysis} 16(2):667-718.
#' @seealso [tulpa_draws_array()], [plot_rhat()], [plot_ess()],
#'   [diagnostic_summary()], [check_diagnostics()]
#' @export
mcmc_diagnostics <- function(fit, pars = NULL,
                             measures = c("rhat", "ess_bulk", "ess_tail"),
                             probs = c(0.05, 0.95)) {
  measures <- match.arg(measures, names(.tulpa_diag_measures), several.ok = TRUE)
  if (!.tulpa_is_chain(fit)) {
    # A point fit (mode + covariance) carries no sample to diagnose; an i.i.d.
    # approximation fit dispatches to the PSIS reliability table.
    if (identical(.tulpa_draws_kind(fit), "point")) {
      message(.tulpa_non_chain_msg(fit))
      return(NULL)
    }
    return(laplace_diagnostics(fit, pars = pars))
  }
  chain_list <- .tulpa_chain_list(fit)
  if (is.null(chain_list) || nrow(chain_list[[1L]]) < 4L) return(NULL)

  p  <- ncol(chain_list[[1L]])
  nm <- colnames(chain_list[[1L]])
  if (is.null(nm)) nm <- paste0("param", seq_len(p))
  keep <- if (is.null(pars)) seq_len(p) else which(nm %in% pars)
  if (length(keep) == 0L) return(NULL)

  n     <- nrow(chain_list[[1L]])
  cols  <- .tulpa_measure_cols(measures, probs)
  out   <- matrix(NA_real_, nrow = length(keep), ncol = length(cols),
                  dimnames = list(NULL, cols))
  for (i in seq_along(keep)) {
    j <- keep[i]
    sims <- vapply(chain_list, function(cd) cd[, j], numeric(n))
    if (is.null(dim(sims))) sims <- matrix(sims, ncol = length(chain_list))
    if (stats::sd(c(sims)) == 0) next                  # constant: leave NA
    vals <- tryCatch(.tulpa_measure_eval(sims, measures, probs),
                     error = function(e) NULL)
    if (!is.null(vals)) out[i, names(vals)] <- vals
  }
  data.frame(parameter = nm[keep], out, check.names = FALSE,
             stringsAsFactors = FALSE, row.names = NULL)
}

#' Select the "main" model parameters for diagnostic display
#'
#' Drops per-element latent-field entries (names ending in a bracketed index,
#' e.g. `u[12]`, `w[3]`) so plots and summaries focus on scalar coefficients
#' and hyperparameters rather than thousands of latent values.
#'
#' @param param_names Character vector of parameter names.
#' @return The subset of `param_names` that are not bracketed-index entries
#'   (or the full vector if that would be empty).
#' @export
select_main_params <- function(param_names) {
  keep <- !grepl("\\[[0-9]+\\]$", param_names)
  if (!any(keep)) return(param_names)
  param_names[keep]
}
