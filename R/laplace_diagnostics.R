# =============================================================================
# laplace_diagnostics.R -- approximation-reliability diagnostics for
# deterministic (i.i.d.-draw) nested-Laplace fits.
#
# Gelman-Rubin Rhat is a between-chain statistic and does not apply to the
# i.i.d. draws a nested-Laplace fit produces from its grid-mixture posterior
# (mcmc_diagnostics() refuses such fits). The reliability question for this
# engine class is not chain mixing but whether the deterministic approximation
# q -- the grid mixture sum_k w_k N(mode_k, V_k) the draws are sampled from --
# is a faithful stand-in for the exact posterior pi. That is an
# importance-sampling question, scored by the Pareto-smoothed importance
# sampling (PSIS) shape diagnostic k-hat of Vehtari, Simpson, Gelman, Yao &
# Gabry (2024) and the variational-reliability framing of Yao, Vehtari, Simpson
# & Gelman (2018): k-hat < 0.5 good, 0.5-0.7 usable, >= 0.7 unreliable.
#
# `laplace_diagnostics()` (and `mcmc_diagnostics()` dispatched onto an i.i.d.
# fit) returns a per-parameter table -- posterior mean / sd plus the
# rank-normalized split-Rhat and bulk / tail effective sample size of the draws
# (Vehtari et al. 2021), labelled as i.i.d.-draw Monte-Carlo diagnostics, NOT
# chain mixing: they sit at ~1.00 / ~S by construction and document only that
# the reported summaries are not Monte-Carlo-limited. The reliability headline
# -- the PSIS k-hat and the grid quadrature effective sample size -- is attached
# as attributes and as a `summary` row.
# =============================================================================

# Interpretation band for a PSIS / IS Pareto-k-hat (Vehtari et al. 2024).
.tulpa_khat_band <- function(k) {
  if (!is.finite(k)) return(NA_character_)
  if (k < 0.5)  return("good")
  if (k < 0.7)  return("ok")
  "unreliable"
}

# Outer-grid (hyperparameter) quadrature reliability of a nested-Laplace fit:
# the normalized integration weights `w_k` summarise how the marginal
# hyperparameter posterior is spread over the grid. A grid that collapses onto a
# single cell carries no integrated hyperparameter uncertainty (the outer
# integration degenerates to a point); a grid that spreads its weight broadly
# integrates that uncertainty. Returned: the quadrature effective sample size
# `ess_grid = 1 / sum(w_k^2)`, its share of the cell count `rel_ess_grid`, the
# largest single weight, and the number of grid cells. NULL when the fit
# carries no outer-grid weights.
.tulpa_grid_reliability <- function(fit) {
  jf <- if (!is.null(fit$joint_fit)) fit$joint_fit else fit
  w <- jf$weights
  if (is.null(w)) return(NULL)
  w <- w[is.finite(w) & w > 0]
  if (length(w) == 0L) return(NULL)
  w <- w / sum(w)
  n_grid <- length(w)
  ess <- 1 / sum(w^2)
  list(ess_grid = ess, n_grid = n_grid,
       rel_ess_grid = ess / n_grid, max_weight = max(w))
}

# PSIS / importance-sampling reliability of a nested-Laplace fit, read from the
# fields the outer integrator already attaches: `pareto_k` (the GPD tail-shape
# k-hat of the importance ratio log p_target(theta) - log q_proposal(theta) over
# the hyperparameter grid, computed against the EXACT inner-Laplace marginal at
# fit time) and `pareto_k_is_ess` (the importance-sampling effective sample size
# on the PSIS-smoothed weights). `is_ess` is on the n_samples scale the outer
# diagnostic drew (`pareto_k_is_ess` / n is the relative IS efficiency). A fit
# that never ran the diagnostic, or whose proposal degenerated, leaves these NA.
.tulpa_psis_reliability <- function(fit) {
  jf <- if (!is.null(fit$joint_fit)) fit$joint_fit else fit
  list(pareto_k = jf$pareto_k %||% NA_real_,
       pareto_k_is_ess = jf$pareto_k_is_ess %||% NA_real_,
       pareto_k_scope = jf$pareto_k_scope %||% NA_character_)
}

# Per-parameter posterior summary + i.i.d.-draw Monte-Carlo diagnostics on a
# [n_draws x n_par] draws matrix. Returns a data frame with `parameter`, `mean`,
# `sd`, `ess_bulk`, `ess_tail`, `rhat`. The split-Rhat / ESS estimators are the
# convergence.R ones (Vehtari et al. 2021); on i.i.d. draws split-Rhat sits at
# ~1 and ESS ~ n_draws, so these certify the summaries are not MC-limited rather
# than diagnosing chain mixing.
.tulpa_iid_param_table <- function(draws, pars = NULL) {
  draws <- as.matrix(draws)
  nm <- colnames(draws)
  if (is.null(nm)) nm <- paste0("param", seq_len(ncol(draws)))
  keep <- if (is.null(pars)) seq_along(nm) else which(nm %in% pars)
  if (length(keep) == 0L) return(NULL)
  n <- nrow(draws)
  out <- data.frame(parameter = nm[keep],
                    mean = NA_real_, sd = NA_real_,
                    ess_bulk = NA_real_, ess_tail = NA_real_,
                    rhat = NA_real_,
                    stringsAsFactors = FALSE, row.names = NULL)
  for (i in seq_along(keep)) {
    x <- draws[, keep[i]]
    out$mean[i] <- mean(x)
    out$sd[i]   <- stats::sd(x)
    if (n >= 4L && is.finite(out$sd[i]) && out$sd[i] > 0) {
      sims <- matrix(x, ncol = 1L)
      out$ess_bulk[i] <- tryCatch(.tulpa_ess_bulk(sims), error = function(e) NA_real_)
      out$ess_tail[i] <- tryCatch(.tulpa_ess_tail(sims), error = function(e) NA_real_)
      out$rhat[i]     <- tryCatch(.tulpa_rhat(sims),     error = function(e) NA_real_)
    }
  }
  out
}

#' Approximation-reliability diagnostics for a deterministic nested-Laplace fit
#'
#' @description
#' Per-parameter reliability diagnostics for a fit whose posterior draws are
#' i.i.d. samples from a deterministic approximation (the nested-Laplace
#' grid-mixture posterior `sum_k w_k N(mode_k, V_k)`), where the between-chain
#' Gelman-Rubin Rhat that [mcmc_diagnostics()] reports does not apply. This is
#' the accessor that plays Rhat's role for the deterministic engine: it answers
#' "did the approximation work", not "did the chains mix".
#'
#' The headline is a Pareto-smoothed importance-sampling (PSIS) reliability
#' diagnostic for the approximation. The nested integrator scores its
#' hyperparameter grid against the exact inner-Laplace marginal posterior with a
#' generalized-Pareto fit to the upper tail of the importance ratios
#' `log p_target(theta) - log q_proposal(theta)` (see [tulpa_psis()]); the
#' resulting tail-shape `pareto_k` is the "did it work" number -- `k-hat < 0.5`
#' good, `0.5-0.7` usable, `>= 0.7` unreliable (Vehtari et al. 2024; Yao et al.
#' 2018). It is computed at fit time and read back here; a fit that did not run
#' the diagnostic, or whose grid proposal degenerated, reports it as `NA` and is
#' assessed on the grid quadrature reliability instead.
#'
#' The grid quadrature reliability -- the effective sample size
#' `ess_grid = 1 / sum(w_k^2)` of the outer integration weights and the largest
#' single cell weight -- is always computed from the stored grid: a grid that
#' collapses onto one cell (`ess_grid` near 1) integrates no hyperparameter
#' uncertainty, while a spread grid does.
#'
#' Each parameter row also carries the rank-normalized split-Rhat and bulk /
#' tail effective sample size of the draws (Vehtari et al. 2021). On i.i.d.
#' draws these sit at `~1.00` and `~n_draws` by construction; they are reported,
#' clearly as i.i.d.-draw Monte-Carlo diagnostics and not chain mixing, to
#' document that the reported posterior summaries are not Monte-Carlo-limited.
#'
#' @section Scope:
#' The PSIS `pareto_k` diagnoses the OUTER (hyperparameter) integration: whether
#' the Gaussian-proposal-over-grid approximation of the marginal hyperparameter
#' posterior `p(theta | data)` can be importance-corrected to the exact inner
#' marginal. This is the dominant approximation in nested Laplace and the one
#' with an exactly evaluable target. A full latent-space PSIS against the exact
#' joint posterior `pi(x)` is not computed: the latent prior marginal
#' `p(x) = integral p(x | theta) p(theta) dtheta` has no closed form, and for
#' the marginalized-occupancy / cover-hurdle likelihoods the exact joint density
#' is evaluable only inside the C++ kernel, so a stored fit cannot reconstruct
#' it. The grid quadrature reliability is the complementary stored-fit number.
#'
#' @param fit A `tulpa_fit` (or subclass, e.g. a `tobs_fit`) whose draws are an
#'   i.i.d. approximation sample (`$draws_kind == "iid"`).
#' @param pars Optional character vector of parameter names to restrict to.
#' @return A data frame with one row per parameter -- `parameter`, `mean`, `sd`,
#'   `ess_bulk`, `ess_tail`, `rhat` -- carrying attributes:
#'   \describe{
#'     \item{`pareto_k`}{the outer PSIS reliability k-hat (`NA` if not computed).}
#'     \item{`pareto_k_band`}{`"good"` / `"ok"` / `"unreliable"` / `NA`.}
#'     \item{`pareto_k_is_ess`}{importance-sampling ESS on the smoothed weights.}
#'     \item{`ess_grid`, `n_grid`, `rel_ess_grid`, `max_weight`}{grid quadrature
#'       reliability.}
#'     \item{`scope`}{the diagnostic's scope string.}
#'   }
#'   and a trailing `summary` attribute (a one-row data frame of the headline
#'   numbers) for printing.
#' @references
#' Vehtari, Simpson, Gelman, Yao & Gabry (2024). Pareto smoothed importance
#'   sampling. \emph{JMLR} 25(72):1-58.
#'
#' Yao, Vehtari, Simpson & Gelman (2018). Yes, but did it work?: Evaluating
#'   variational inference. \emph{ICML}, PMLR 80:5581-5590.
#'
#' Vehtari, Gelman, Simpson, Carpenter & Burkner (2021). Rank-normalization,
#'   folding, and localization: an improved Rhat for assessing convergence of
#'   MCMC. \emph{Bayesian Analysis} 16(2):667-718.
#' @seealso [mcmc_diagnostics()] (the chain counterpart, which dispatches here
#'   for i.i.d. fits), [tulpa_psis()].
#' @examples
#' set.seed(1)
#' n <- 200L; x <- rnorm(n)
#' y <- rbinom(n, 1, plogis(-0.2 + 0.6 * x))
#' fit <- tulpa(y ~ x, data.frame(y = y, x = x), family = "binomial",
#'              mode = "laplace")
#' laplace_diagnostics(fit)
#' @export
laplace_diagnostics <- function(fit, pars = NULL) {
  draws <- fit$draws %||% fit$samples
  if (is.null(draws)) {
    message("laplace_diagnostics(): the fit carries no posterior draws.")
    return(NULL)
  }
  tab <- .tulpa_iid_param_table(draws, pars = pars)
  if (is.null(tab)) return(NULL)

  grid <- .tulpa_grid_reliability(fit)
  psis <- .tulpa_psis_reliability(fit)
  k    <- psis$pareto_k

  attr(tab, "pareto_k")        <- k
  attr(tab, "pareto_k_band")   <- .tulpa_khat_band(k)
  attr(tab, "pareto_k_is_ess") <- psis$pareto_k_is_ess
  attr(tab, "scope")           <- psis$pareto_k_scope
  if (!is.null(grid)) {
    attr(tab, "ess_grid")     <- grid$ess_grid
    attr(tab, "n_grid")       <- grid$n_grid
    attr(tab, "rel_ess_grid") <- grid$rel_ess_grid
    attr(tab, "max_weight")   <- grid$max_weight
  }

  summary_row <- data.frame(
    pareto_k      = k,
    pareto_k_band = .tulpa_khat_band(k),
    ess_grid      = if (is.null(grid)) NA_real_ else grid$ess_grid,
    n_grid        = if (is.null(grid)) NA_integer_ else grid$n_grid,
    max_weight    = if (is.null(grid)) NA_real_ else grid$max_weight,
    n_draws       = nrow(as.matrix(draws)),
    stringsAsFactors = FALSE, row.names = NULL
  )
  attr(tab, "summary") <- summary_row
  class(tab) <- c("laplace_diagnostics", class(tab))
  tab
}

#' @export
print.laplace_diagnostics <- function(x, ...) {
  s <- attr(x, "summary")
  k <- attr(x, "pareto_k")
  band <- attr(x, "pareto_k_band")
  cat("Nested-Laplace approximation reliability (i.i.d. draws)\n")
  if (is.finite(k)) {
    cat(sprintf("  PSIS pareto_k = %.3f (%s); IS-ESS = %.1f\n",
                k, band, attr(x, "pareto_k_is_ess")))
  } else {
    cat("  PSIS pareto_k = NA (outer diagnostic not run or proposal degenerate)\n")
  }
  if (!is.null(attr(x, "ess_grid"))) {
    cat(sprintf("  grid quadrature ESS = %.2f of %d cells (max weight %.3f)\n",
                attr(x, "ess_grid"), attr(x, "n_grid"), attr(x, "max_weight")))
  }
  cat(sprintf("  %d parameters, %d draws; per-parameter rhat / ESS below are\n",
              nrow(x), if (is.null(s)) NA_integer_ else s$n_draws))
  cat("  i.i.d.-draw Monte-Carlo diagnostics (not chain mixing).\n\n")
  print(as.data.frame(x), ...)
  invisible(x)
}
