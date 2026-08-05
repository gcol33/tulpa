# =============================================================================
# laplace_diagnostics.R -- approximation-reliability diagnostics for
# deterministic (i.i.d.-draw) nested-Laplace fits.
#
# Gelman-Rubin Rhat is a between-chain statistic and does not apply to the
# i.i.d. draws a nested-Laplace fit produces from its grid-mixture posterior
# (`diagnostics()` routes such fits here instead). The reliability question for this
# engine class is not chain mixing but whether the deterministic approximation
# q -- the grid mixture sum_k w_k N(mode_k, V_k) the draws are sampled from --
# is a faithful stand-in for the exact posterior pi. That is an
# importance-sampling question, scored by the Pareto-smoothed importance
# sampling (PSIS) shape diagnostic k-hat of Vehtari, Simpson, Gelman, Yao &
# Gabry (2024) and the variational-reliability framing of Yao, Vehtari, Simpson
# & Gelman (2018): k-hat < 0.5 good, 0.5-0.7 usable, >= 0.7 unreliable.
#
# `diagnostics()` on an i.i.d. fit returns a per-parameter table -- posterior
# mean / sd plus the
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

# =============================================================================
# Inner-Laplace skewness diagnostic (gcol33/tulpa#272) -- the layer pareto_k
# above does NOT cover: whether the Gaussian (inner) Laplace approximation to
# the latent-field conditional posterior pi(x_i | theta, y) is itself a good
# fit, as opposed to whether the OUTER hyperparameter grid integrates that
# (fixed) inner approximation correctly.
#
# `gamma_3` (src/inner_laplace_skew.h, Rue, Martino & Chopin 2009 Sec 3.2.3
# eq. 21's cubic term) is the leading-order Edgeworth estimate of the third
# STANDARDIZED CUMULANT (skewness) of pi(x_i | theta, y) relative to the
# Gaussian inner Laplace: log f(z) = -z^2/2 + (gamma_3/6)(z^3 - 3z) + ...,
# so |gamma_3| directly measures how large that cubic correction is. NaN means
# "not computable there" (a coupled multi-process likelihood, or a family
# without a registered third derivative -- see build_spec_curvature3_fn) and
# is never conflated with 0 ("no skew").
# =============================================================================

# Band a single |gamma_3| magnitude. Thresholds follow the common
# skewness-magnitude convention (|skew| < 0.5 approximately symmetric, 0.5-1
# moderately skewed, > 1 highly skewed; e.g. Bulmer 1979, "Principles of
# Statistics") applied to gamma_3 as a skewness estimate -- Rue, Martino &
# Chopin (2009) motivate the cubic correction but do not themselves give a
# numeric cutoff for when it is "large", so this is a general skewness
# heuristic being applied here, not a threshold specific to that paper.
.tulpa_gamma3_band <- function(g) {
  if (!is.finite(g)) return(NA_character_)
  ag <- abs(g)
  if (ag < 0.5) return("good")
  if (ag < 1.0) return("ok")
  "unreliable"
}

# Aggregate a per-latent-index gamma_3 vector (as returned by the C++ kernel's
# `inner_skew` field) into a per-fit inner-reliability summary. `n_dropped` is
# the (index, observation) contribution count the kernel reported as
# non-computable (inner_skew_dropped) -- surfaced so a partially-scored fit
# (some latents NaN, others finite) is visible rather than silently averaged
# away. Returns NULL when `gamma3` is NULL/empty/all-NaN (nothing computed).
.tulpa_inner_skew_summary <- function(gamma3, n_dropped = 0L) {
  if (is.null(gamma3) || length(gamma3) == 0L) return(NULL)
  finite <- gamma3[is.finite(gamma3)]
  n_scored <- length(finite)
  if (n_scored == 0L) {
    return(list(max_abs_gamma3 = NA_real_, band = NA_character_,
               n_scored = 0L, n_probed = length(gamma3),
               n_dropped = as.integer(n_dropped),
               share_moderate = NA_real_, share_unreliable = NA_real_))
  }
  ag <- abs(finite)
  list(
    max_abs_gamma3   = max(ag),
    band             = .tulpa_gamma3_band(max(ag)),
    n_scored         = n_scored,
    n_probed         = length(gamma3),
    n_dropped        = as.integer(n_dropped),
    share_moderate   = mean(ag >= 0.5),
    share_unreliable = mean(ag >= 1.0)
  )
}

# Combine the outer (grid-quadrature / PSIS) band with the inner (Laplace
# skewness) band into a single whole-fit verdict: reliable only when BOTH
# layers are reliable; otherwise "scoped", naming which layer(s) degrade, so
# the caller never reads a high outer k-hat (say) as "the whole fit is
# broken" when the inner layer is fine, or vice versa -- the framing #272
# was filed to fix (42/78 occu_cover species read as "broken" on outer k-hat
# alone when their point estimates, governed by the inner layer, were fine).
.tulpa_combined_reliability <- function(outer_band, inner_band) {
  bad_outer <- identical(outer_band, "unreliable")
  bad_inner <- identical(inner_band, "unreliable")
  ok_outer  <- identical(outer_band, "ok")
  ok_inner  <- identical(inner_band, "ok")
  if (is.na(outer_band) && is.na(inner_band)) {
    return("not computed (neither layer's diagnostic ran)")
  }
  if (bad_outer && bad_inner) {
    return("unreliable (both outer integration and inner Laplace flagged)")
  }
  if (bad_outer) return("scoped: outer (hyperparameter) integration flagged")
  if (bad_inner) return("scoped: inner (latent-field) Laplace flagged")
  if (ok_outer && ok_inner) return("usable (both layers borderline)")
  if (ok_outer) return("scoped: outer integration borderline")
  if (ok_inner) return("scoped: inner Laplace borderline")
  "reliable (both layers good)"
}

# Re-dispatch the SAME kernel `res` came from at a length-1 grid pinned to the
# fitted MAP cell (`which.max(res$weights)`), with `compute_skew = TRUE`, and
# attach `res$inner_skew` / `res$inner_skew_idx` / `res$inner_skew_dropped`.
# One extra deterministic Newton solve (no importance sampling, unlike the
# outer Pareto-k diagnostic) -- mirrors the "re-dispatch through the SAME
# driver, RNG-restored where relevant" convention of .nl_attach_pareto_k /
# .nested_outer_pareto_k, but gamma_3 is a point evaluation so there is no RNG
# to restore.
#
# `probe_idx` (1-based latent indices) defaults to the p fixed-effects
# coefficients (positions 1:p_fixed in the [beta | re | blocks] latent
# layout every kernel shares) -- cheap (p is almost always small) and answers
# the most common question ("is my reported beta CI trustworthy"); pass
# `control$skew_idx` to probe additional latent indices (e.g. specific
# spatial units), since scoring the FULL latent field costs one extra linear
# solve per requested index and is not on by default for large fields.
#
# `dispatch_kind` is "single" (.nl_dispatch, one `prior` block) or "multi"
# (.nl_dispatch_multi, `prior` a list of blocks) -- the two shapes
# tulpa_nested_laplace() itself dispatches on. Declines (leaves res
# untouched) for any other dispatch_kind; the joint driver
# (tulpa_nested_laplace_joint()) attaches its own inner-skew field via
# .nlj_inner_skew_at_theta() in nested_laplace_joint.R, which re-dispatches
# cpp_nested_laplace_joint_multi directly using the SAME theta_grid MAP-row
# trick (no `dispatch_kind` string of its own to add here).
.nl_inner_skew_at_theta <- function(res, prior, cargs, dispatch_kind, type,
                                    likelihood, p_fixed, skew_idx = NULL,
                                    compute = TRUE) {
  res$inner_skew         <- NULL
  res$inner_skew_idx     <- integer(0)
  res$inner_skew_dropped <- 0L
  if (!compute) return(res)

  w <- res$weights
  if (is.null(w) || !any(is.finite(w))) return(res)
  map_idx <- which.max(w)

  probe_idx <- if (is.null(skew_idx)) seq_len(max(as.integer(p_fixed %||% 0L), 0L))
              else as.integer(skew_idx)
  if (length(probe_idx) == 0L) return(res)

  cargs_no_ckpt <- utils::modifyList(cargs, list(checkpoint_path = ""))

  out <- tryCatch({
    if (dispatch_kind == "multi") {
      tg <- res$theta_grid
      if (is.null(tg)) return(NULL)
      tg1 <- matrix(as.numeric(tg[map_idx, ]), nrow = 1L)
      .nl_dispatch_multi(cargs_no_ckpt, prior, likelihood = likelihood,
                         theta_grid_override = tg1,
                         compute_skew = TRUE, skew_idx = probe_idx)
    } else if (dispatch_kind == "single") {
      blocks <- if (is.list(prior) && is.null(prior$type)) prior else list(prior)
      if (length(blocks) != 1L) return(NULL)   # defensive; single-block only
      blk <- blocks[[1L]]
      type <- type %||% tolower(blk$type)
      spec <- .NL_REGISTRY[[type]]
      if (!is.null(spec) && is.function(spec$defaults)) {
        blk <- tryCatch(spec$defaults(blk, cargs), error = function(e) blk)
      }
      # Narrow EVERY *_grid field (icar has one, bym2/car_proper/nngp/hsgp
      # have two paired axes) to its MAP-row value -- unlike
      # .nl_attach_pareto_k this does not require exactly one axis, since
      # there is no importance-sampling proposal to fit, just a point re-solve.
      gfs <- grep("_grid$", names(blk), value = TRUE)
      gfs <- gfs[vapply(gfs, function(f) is.numeric(blk[[f]]) && length(blk[[f]]) == length(w),
                        logical(1))]
      if (length(gfs) == 0L) return(NULL)
      blk2 <- blk
      for (f in gfs) blk2[[f]] <- blk[[f]][map_idx]
      cargs2 <- utils::modifyList(cargs_no_ckpt,
                                  list(compute_skew = TRUE, skew_idx = probe_idx))
      .nl_dispatch(type, cargs2, blk2)
    } else {
      NULL
    }
  }, error = function(e) NULL)

  if (is.null(out) || is.null(out$inner_skew)) return(res)
  res$inner_skew         <- as.numeric(out$inner_skew)
  res$inner_skew_idx     <- as.integer(out$inner_skew_idx)
  res$inner_skew_dropped <- as.integer(out$inner_skew_dropped %||% 0L)
  res
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

# Inner-Laplace skewness reliability of a nested-Laplace fit, read from the
# `inner_skew` / `inner_skew_idx` / `inner_skew_dropped` fields
# .nl_inner_skew_at_theta() attaches at fit time (see nested_laplace.R). NULL
# when the fit never ran the diagnostic (control$diagnose_skew = FALSE, or a
# backend that does not yet populate it).
.tulpa_inner_skew_reliability <- function(fit) {
  jf <- if (!is.null(fit$joint_fit)) fit$joint_fit else fit
  .tulpa_inner_skew_summary(jf$inner_skew, jf$inner_skew_dropped %||% 0L)
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

# Approximation-reliability table for an i.i.d. deterministic fit. The
# provenance gate lives in `diagnostics()`; this builds the table and attaches
# the PSIS / grid-quadrature headline as attributes. Documented user-side under
# `?laplace_diagnostics`.
.tulpa_approx_diag_table <- function(fit, pars = NULL) {
  draws <- .fit_draws(fit)
  if (is.null(draws)) {
    message("diagnostics(): the fit carries no posterior draws.")
    return(NULL)
  }
  tab <- .tulpa_iid_param_table(draws, pars = pars)
  if (is.null(tab)) return(NULL)

  grid  <- .tulpa_grid_reliability(fit)
  psis  <- .tulpa_psis_reliability(fit)
  inner <- .tulpa_inner_skew_reliability(fit)
  k          <- psis$pareto_k
  outer_band <- .tulpa_khat_band(k)
  inner_band <- if (is.null(inner)) NA_character_ else inner$band

  attr(tab, "pareto_k")        <- k
  attr(tab, "pareto_k_band")   <- outer_band
  attr(tab, "pareto_k_is_ess") <- psis$pareto_k_is_ess
  attr(tab, "scope")           <- psis$pareto_k_scope
  if (!is.null(grid)) {
    attr(tab, "ess_grid")     <- grid$ess_grid
    attr(tab, "n_grid")       <- grid$n_grid
    attr(tab, "rel_ess_grid") <- grid$rel_ess_grid
    attr(tab, "max_weight")   <- grid$max_weight
  }
  if (!is.null(inner)) {
    attr(tab, "inner_skew_max")    <- inner$max_abs_gamma3
    attr(tab, "inner_skew_band")   <- inner$band
    attr(tab, "inner_skew_scored") <- inner$n_scored
    attr(tab, "inner_skew_probed") <- inner$n_probed
  }
  attr(tab, "reliability") <- .tulpa_combined_reliability(outer_band, inner_band)

  summary_row <- data.frame(
    pareto_k        = k,
    pareto_k_band   = outer_band,
    ess_grid        = if (is.null(grid)) NA_real_ else grid$ess_grid,
    n_grid          = if (is.null(grid)) NA_integer_ else grid$n_grid,
    max_weight      = if (is.null(grid)) NA_real_ else grid$max_weight,
    inner_skew_max  = if (is.null(inner)) NA_real_ else inner$max_abs_gamma3,
    inner_skew_band = inner_band,
    reliability     = .tulpa_combined_reliability(outer_band, inner_band),
    n_draws         = nrow(as.matrix(draws)),
    stringsAsFactors = FALSE, row.names = NULL
  )
  attr(tab, "summary") <- summary_row
  class(tab) <- c("laplace_diagnostics", class(tab))
  tab
}

#' Approximation-reliability diagnostics for a deterministic nested-Laplace fit
#'
#' @description
#' `r lifecycle::badge("deprecated")`
#'
#' Use [diagnostics()], which returns this table for any fit whose draws are an
#' i.i.d. approximation sample. The sections below document that table; they
#' remain the reference for its columns and attributes.
#'
#' Per-parameter reliability diagnostics for a fit whose posterior draws are
#' i.i.d. samples from a deterministic approximation (the nested-Laplace
#' grid-mixture posterior `sum_k w_k N(mode_k, V_k)`), where the between-chain
#' Gelman-Rubin Rhat that [diagnostics()] reports for a chain fit does not apply. This is
#' the accessor that plays Rhat's role for the deterministic engine: it answers
#' "did the approximation work", not "did the chains mix".
#'
#' The headline is a Pareto-smoothed importance-sampling (PSIS) reliability
#' diagnostic for the OUTER hyperparameter-grid integration. The nested
#' integrator scores its hyperparameter grid against the exact inner-Laplace
#' marginal posterior with a generalized-Pareto fit to the upper tail of the
#' importance ratios `log p_target(theta) - log q_proposal(theta)`
#' (see [tulpa_psis()]); the resulting tail-shape `pareto_k` is the "did the
#' outer integration work" number -- `k-hat < 0.5` good, `0.5-0.7` usable,
#' `>= 0.7` unreliable (Vehtari et al. 2024; Yao et al. 2018). It is computed at
#' fit time and read back here; a fit that did not run the diagnostic, or whose
#' grid proposal degenerated, reports it as `NA` and is assessed on the grid
#' quadrature reliability instead.
#'
#' `pareto_k` scores the outer integration only. A high `pareto_k` on a fit
#' whose grid quadrature is healthy (`ess_grid` well above 1, largest cell
#' weight modest) flags outer-integration (CI-width) calibration in the
#' right-skewed hyperparameter tail and does not by itself invalidate the
#' point estimates, which the grid quadrature governs.
#'
#' The grid quadrature reliability -- the effective sample size
#' `ess_grid = 1 / sum(w_k^2)` of the outer integration weights and the largest
#' single cell weight -- is always computed from the stored grid: a grid that
#' collapses onto one cell (`ess_grid` near 1) integrates no hyperparameter
#' uncertainty, while a spread grid does.
#'
#' A SEPARATE layer -- the inner Gaussian Laplace approximation to the
#' latent-field conditional posterior `pi(x | theta, y)`, which `pareto_k`
#' does not cover -- is scored by `inner_skew`: the leading-order Edgeworth
#' skewness estimate `gamma_3` (Rue, Martino & Chopin 2009 Sec 3.2.3) at the
#' fitted MAP grid cell, computed when `control$diagnose_skew = TRUE` (the
#' default) on the fitting call. Reading a high `pareto_k` alone as "the fit
#' is broken" conflates the two layers: an occu_cover batch flagged 42/78
#' species "unreliable" on outer k-hat alone when their point estimates,
#' governed by the healthy inner layer, were fine (gcol33/tulpa#272) -- the
#' `reliability` attribute is the combined verdict that names which layer
#' degrades, if either does.
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
#' `inner_skew` diagnoses the INNER (latent-field) Laplace: whether the
#' Gaussian approximation to `pi(x_i | theta, y)` is itself a good fit, at
#' each scored latent index `i`. `gamma_3` is exact for a gaussian-family
#' coefficient (the log-likelihood is exactly quadratic in eta) and
#' declines to `NA` -- never a silently-wrong `0` ("perfectly Gaussian") --
#' whenever no per-observation third-derivative oracle is available: a
#' coupled multi-process likelihood (e.g. tulpaObs's `occu_cover`, whose
#' arms combine non-separably through a `CellCouplingSpec`) or a family with
#' no registered third derivative. Only the requested latent indices are
#' scored (every arm's fixed-effects coefficients by default -- see
#' `control$skew_idx`), since each index costs one extra linear solve; the
#' full latent field is not scored by default on a large spatial field.
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
#'     \item{`scope`}{the outer diagnostic's scope string.}
#'     \item{`inner_skew_max`}{the largest `|gamma_3|` among the scored latent
#'       indices (`NA` if `control$diagnose_skew = FALSE` or nothing scored).}
#'     \item{`inner_skew_band`}{`"good"` / `"ok"` / `"unreliable"` / `NA`, banded
#'       on `inner_skew_max` by the general skewness-magnitude convention
#'       (Bulmer 1979) -- not a Rue-Martino-Chopin-specific cutoff.}
#'     \item{`inner_skew_scored`, `inner_skew_probed`}{how many of the probed
#'       latent indices returned a finite `gamma_3` vs how many were probed.}
#'     \item{`reliability`}{the combined whole-fit verdict: `"reliable"` only
#'       when both layers are good; otherwise names which layer is scoped or
#'       flags both as unreliable.}
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
#'
#' Rue, Martino & Chopin (2009). Approximate Bayesian inference for latent
#'   Gaussian models by using integrated nested Laplace approximations.
#'   \emph{JRSS-B} 71(2):319-392.
#' @seealso [diagnostics()] (the front door, which returns this table for
#'   i.i.d. fits), [tulpa_psis()].
#' @examples
#' set.seed(1)
#' n <- 200L; x <- rnorm(n)
#' y <- rbinom(n, 1, plogis(-0.2 + 0.6 * x))
#' # `mode = "laplace"` returns a mode + covariance and carries no draws; a
#' # sampled deterministic backend is what this table describes.
#' fit <- tulpa(y ~ x, data.frame(y = y, x = x), family = "binomial",
#'              mode = "smc")
#' diagnostics(fit)
#' @export
laplace_diagnostics <- function(fit, pars = NULL) {
  lifecycle::deprecate_warn("0.0.95", "laplace_diagnostics()", "diagnostics()")
  .tulpa_approx_diag_table(fit, pars = pars)
}

#' @export
print.laplace_diagnostics <- function(x, ...) {
  s <- attr(x, "summary")
  k <- attr(x, "pareto_k")
  band <- attr(x, "pareto_k_band")
  has_inner <- !is.null(attr(x, "inner_skew_band"))
  if (has_inner) {
    cat("Nested-Laplace WHOLE-FIT reliability (i.i.d. draws)\n")
    cat("  two layers: the outer hyperparameter-grid integration, and the",
        "inner Gaussian Laplace on the latent field\n")
  } else {
    cat("Nested-Laplace OUTER-integration reliability (i.i.d. draws)\n")
    cat("  scope: the outer hyperparameter-grid integration; the latent-field",
        "Laplace is a separate, unscored layer (control$diagnose_skew = FALSE)\n")
  }
  if (is.finite(k)) {
    cat(sprintf("  outer PSIS pareto_k = %.3f (%s); IS-ESS = %.1f\n",
                k, band, attr(x, "pareto_k_is_ess")))
  } else {
    cat("  outer PSIS pareto_k = NA (outer diagnostic not run or proposal degenerate)\n")
  }
  if (!is.null(attr(x, "ess_grid"))) {
    cat(sprintf("  outer grid quadrature ESS = %.2f of %d cells (max weight %.3f)\n",
                attr(x, "ess_grid"), attr(x, "n_grid"), attr(x, "max_weight")))
  }
  if (has_inner) {
    ib <- attr(x, "inner_skew_band")
    im <- attr(x, "inner_skew_max")
    if (is.finite(im)) {
      cat(sprintf("  inner Laplace max |gamma_3| = %.3f (%s), scored %d/%d latents\n",
                  im, ib, attr(x, "inner_skew_scored"), attr(x, "inner_skew_probed")))
    } else {
      cat("  inner Laplace gamma_3 = NA (not computable for this likelihood)\n")
    }
    cat(sprintf("  whole-fit verdict: %s\n", attr(x, "reliability")))
  }
  cat(sprintf("  %d parameters, %d draws; per-parameter rhat / ESS below are\n",
              nrow(x), if (is.null(s)) NA_integer_ else s$n_draws))
  cat("  i.i.d.-draw Monte-Carlo diagnostics (not chain mixing).\n\n")
  print(as.data.frame(x), ...)
  invisible(x)
}
