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
# Inner-Laplace skewness diagnostic -- the layer pareto_k
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
# "not computable there" (a likelihood that ships no way to reach a third
# derivative -- see build_spec_curvature3_oracle) and is never conflated with 0
# ("no skew"). A likelihood whose unit reads several linear predictors at once --
# a zero-inflation mixture, a cell-coupled `occu_cover` -- is scored through the
# widened contraction of the third-derivative tensor
# (src/curvature3_contract.h) rather than declined.
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
  if (ag < .nl_diag("gamma3_ok")) return("good")
  if (ag < .nl_diag("gamma3_unreliable")) return("ok")
  "unreliable"
}

# --- inner-skew decline reasons ---------------------------
#
# `gamma_3` is careful never to return a silently-wrong 0 for a non-computable
# skewness -- every decline is NaN. But NaN says only "not computable",
# and the reasons are not interchangeable: a fit whose coupled arms have no
# scorable path at all leaves the outer k-hat as the only reliability number,
# while a failed finite difference is specific to one fit and a disabled knob is
# not a problem at all. Without the reason an unscorable fit printed as
# `control$diagnose_skew = FALSE`, attributing an impossibility to a setting the
# user likely left at its default `TRUE`.
#
# `res$inner_skew_declined` carries it, from this closed vocabulary (the first
# four come from C++ -- see `InnerSkewOutcome::declined` and
# `build_spec_curvature3_oracle` -- the rest are decided R-side):
#
#   coupled_arm            a joint fit whose coupled arms could be scored neither
#                          by a per-observation sum (the coupling spec took them
#                          over) nor by the cell tensor contraction. STRUCTURAL:
#                          nothing about this fit's coupling is readable, so the
#                          outer k-hat is the only reliability number it has.
#                          `inner_skew_arms_declined` names them (1-based), so a
#                          PARTIALLY scored joint fit is visible too.
#   curvature3_unavailable no registered third derivative and no `eta_weights_fn`
#                          to finite-difference.
#   no_finite_contribution an oracle existed but nothing finite reached any
#                          probed index (an unidentified or degenerate mode).
#   no_probe_indices       nothing was asked for (`p_fixed = 0` and no
#                          `control$skew_idx`).
#   not_requested          `control$diagnose_skew = FALSE`.
#   backend_unsupported    this backend does not populate gamma_3.
#   solve_failed           the probe re-solve errored or returned no field.
#   not_converged          the solve returned a point that is not a mode.
#                          gamma_3 is a cubic expansion ABOUT the mode and the
#                          inner k-hat an importance ratio against the Gaussian
#                          AT it, so neither exists there. This is what a fit
#                          whose inner Newton stalled reports, instead of
#                          `backend_unsupported` -- the backend computes gamma_3
#                          on every converged solve of the same model. The
#                          kernel settles it where it can see it; the probe
#                          re-solve settles it otherwise.
#   pd_eigen_clamp         sparse joint only: the fit ran under
#                          `control$hessian = "psd"`, whose inner step
#                          eigen-solves a densified Hessian and leaves no CHOLMOD
#                          factor for either probe to solve against.
#   s2z_rank1_factor       sparse joint only: the field carries sum-to-zero
#                          rank-1 pins the stored Hessian does not hold, so the
#                          live factor is of a different matrix than the one the
#                          solve stepped with.
#   factor_unavailable     sparse joint only: the solve left no live factor at
#                          all.
#
# Coupling several processes in ONE likelihood is no longer a reason of its own:
# a multi-process `LikelihoodSpec` (a ZI mixture) is scored by the
# per-observation tensor contraction, so the former `"coupled_likelihood"` has no
# producer left and is gone from the vocabulary.
#
# Structural reasons are the ones a reader must act on differently: for those
# fits the outer k-hat is the only reliability number available.
.INNER_SKEW_STRUCTURAL <- c("coupled_arm")

# =============================================================================
# Inner-Laplace importance k-hat -- the second score on the
# SAME inner layer, and the one that survives where the cubic term declines.
#
# `gamma_3` above expands the joint log density along the Gaussian
# conditional-mean curve at a probed index and reads its cubic coefficient, so
# it needs a per-observation third derivative and has none for a coupled
# multi-process likelihood. The inner k-hat walks the same curve and simply
# EVALUATES the joint density along it (src/inner_laplace_is.h), treating the
# inner Gaussian as an importance proposal for the exact conditional posterior:
# no likelihood derivative anywhere, so a fully coupled fit still gets an
# inner-layer number.
#
# The engine returns the raw material -- the standardized proposal draws and the
# joint log density at them, per probed index -- and the Pareto fit is the
# shared `.nested_is_pareto_k()` core, one dimension at a time. Reported on the
# same band convention as the outer k-hat (`.tulpa_khat_band`), and declined
# with a reason from the same closed vocabulary (`.K_DECLINE_REASONS`).
# =============================================================================

# Reset the inner-k fields and record why nothing was computed, from the shared
# outer-k decline vocabulary.
.inner_k_decline <- function(res, reason, detail = NULL) {
    res$inner_pareto_k          <- NULL
    res$inner_pareto_k_is_ess   <- NULL
    res$inner_pareto_k_rel_ess  <- NULL
    res$inner_pareto_k_declined <- .k_decline_label(.k_decline(reason, detail))
    res
}

# A skew decline that also settles the inner k-hat: the two diagnostics share a
# probe dispatch, so whatever stopped one before it ran stopped both. Mapped
# onto the k vocabulary rather than carried across verbatim, so a reader never
# meets a skew-specific reason on a k-hat field.
.INNER_SKEW_TO_K_DECLINE <- list(
    not_requested       = list("not_requested", NULL),
    no_probe_indices    = list("not_applicable", "no probed latent index"),
    backend_unsupported = list("not_applicable",
                               "this backend does not compute the inner importance curve"),
    solve_failed        = list("degenerate_proposal", "the probe re-solve failed"),
    not_converged       = list("not_converged", "the inner solve did not converge"),
    pd_eigen_clamp      = list("not_applicable",
                               "the PSD inner step leaves no factor to propose from"),
    s2z_rank1_factor    = list("not_applicable",
                               "the live factor omits the field's sum-to-zero pins"),
    factor_unavailable  = list("not_applicable", "the solve left no live factor")
)

.inner_k_decline_from_skew <- function(res, reason) {
    map <- .INNER_SKEW_TO_K_DECLINE[[reason]]
    if (is.null(map)) map <- list("not_applicable", reason)
    .inner_k_decline(res, map[[1L]], map[[2L]])
}

# One-line user-facing reading of an inner-k decline, for `print` /
# `diagnostic_summary()`. Same reason codes as the outer k-hat, read in the
# inner layer's terms.
.inner_k_decline_note <- function(label) {
    if (is.null(label) || !length(label) || is.na(label)) return(NULL)
    reason <- sub(":.*$", "", label)
    detail <- if (grepl(":", label, fixed = TRUE)) sub("^[^:]*:\\s*", "", label) else NULL
    with_detail <- function(txt) {
        if (is.null(detail)) txt else paste0(txt, " (", detail, ")")
    }
    switch(reason,
        not_requested = "the inner diagnostic was not requested (control$diagnose_skew = FALSE)",
        not_applicable = with_detail("this fit exposes no inner importance curve to score"),
        draws_too_few = paste("too few finite joint-density evaluations along the probed",
                              "curve to fit the GPD tail"),
        degenerate_proposal = with_detail(paste(
            "the inner Gaussian could not be used as a proposal (the probed",
            "conditional variance was not positive)")),
        internal_inconsistency = with_detail(paste(
            "an internal bookkeeping mismatch stopped the diagnostic;",
            "this indicates an engine bug -- please report it")),
        not_converged = with_detail(paste(
            "the inner Newton solve did not reach a mode, so there is no",
            "Gaussian at a mode to score as a proposal")),
        NULL)
}

# Per-probed-index inner k-hat from the engine's importance curve. `z` are the
# standardized N(0, 1) proposal draws, `lj` the [length(z) x n_probe] matrix of
# joint log densities along each index's conditional-mean curve at those draws.
# The proposal is N(0, 1) by construction on that curve, so the whole importance
# problem is one-dimensional per index and the shared core scores it with the
# draws it was already evaluated at (`Z = z`, no radius cap: at d = 1 there is
# no grid hull to bound the cost against, and every draw is one cheap density
# evaluation that has already been paid).
.inner_k_from_curve <- function(z, lj) {
    S <- length(z)
    P <- ncol(lj)
    k      <- rep(NA_real_, P)
    is_ess <- rep(NA_real_, P)
    n_eval <- rep(NA_real_, P)
    reasons <- character(0)
    Z1 <- matrix(z, ncol = 1L)
    for (j in seq_len(P)) {
        col <- lj[, j]
        target <- function(U) if (nrow(U) == S) col else rep(NA_real_, nrow(U))
        out <- .nested_is_pareto_k(0, matrix(1, 1L, 1L), target,
                                   n_samples = S, radius_cap = Inf, Z = Z1)
        k[j]      <- out$pareto_k
        is_ess[j] <- out$is_ess
        n_eval[j] <- out$n_eval
        if (length(out$declined)) reasons <- c(reasons, out$declined)
    }
    declined <- if (any(is.finite(k))) NA_character_
                else if (length(reasons)) reasons[1L]
                else "degenerate_proposal"
    # Realized importance efficiency per index -- the size of the correction the
    # scale-free shape index describes. `.tulpa_inner_k_summary()` bands the
    # shape only where this says there is something to correct.
    list(pareto_k = k, is_ess = is_ess, rel_ess = is_ess / n_eval,
         declined = declined)
}

# Copy the kernel's inner importance curve onto a fit as a per-index k-hat. One
# attach point for all three drivers, reached through `.inner_skew_attach()`.
.inner_k_attach <- function(res, out) {
    lj <- out$inner_is_log_joint
    z  <- out$inner_is_z
    if (is.null(lj) || is.null(z) || !length(z) || !is.matrix(lj)) {
        return(.inner_k_decline(res, "not_applicable",
                                "this backend does not compute the inner importance curve"))
    }
    kernel_reason <- as.character(out$inner_is_declined %||% "")
    if (length(kernel_reason) && nzchar(kernel_reason[1L])) {
        return(.inner_k_decline_from_skew(res, kernel_reason[1L]))
    }
    if (nrow(lj) != length(z)) {
        return(.inner_k_decline(res, "internal_inconsistency",
                                "importance curve rows do not match the draws"))
    }
    got <- .inner_k_from_curve(as.numeric(z), lj)
    if (is.na(got$declined)) {
        res$inner_pareto_k          <- got$pareto_k
        res$inner_pareto_k_is_ess   <- got$is_ess
        res$inner_pareto_k_rel_ess  <- got$rel_ess
        res$inner_pareto_k_declined <- NA_character_
        return(res)
    }
    .inner_k_decline(res, got$declined)
}

# Reset the inner-skew fields and record why nothing was computed. Used by every
# R-side early return, so a decline is never an absent field. The inner k-hat
# rides the same probe dispatch, so it is settled here too.
.inner_skew_decline <- function(res, reason) {
    res$inner_skew                 <- NULL
    res$inner_skew_gamma1          <- NULL
    res$inner_skew_gamma1_declined <- reason
    res$inner_skew_idx             <- integer(0)
    res$inner_skew_dropped         <- 0L
    res$inner_skew_declined        <- reason
    res$inner_skew_arms_declined   <- integer(0)
    .inner_k_decline_from_skew(res, reason)
}

# Copy the kernel's inner-skew output (including its own decline reason) onto a
# fit. One attach point for all three drivers.
.inner_skew_attach <- function(res, out) {
    res$inner_skew         <- as.numeric(out$inner_skew)
    g1 <- out$inner_skew_gamma1
    res$inner_skew_gamma1 <- if (is.null(g1) || !length(g1)) NULL else as.numeric(g1)
    g1d <- as.character(out$inner_skew_gamma1_declined %||% "")
    res$inner_skew_gamma1_declined <-
        if (!length(g1d) || !nzchar(g1d[1L])) NA_character_ else g1d[1L]
    res$inner_skew_idx     <- as.integer(out$inner_skew_idx)
    res$inner_skew_dropped <- as.integer(out$inner_skew_dropped %||% 0L)
    d <- as.character(out$inner_skew_declined %||% "")
    res$inner_skew_declined <- if (!length(d) || !nzchar(d[1L])) NA_character_
                               else d[1L]
    arms <- out$inner_skew_arms_declined
    res$inner_skew_arms_declined <- if (is.null(arms)) integer(0) else as.integer(arms)
    .inner_k_attach(res, out)
}

# Attach the inner-layer output of a probe RE-SOLVE, or decline with the reason
# that solve itself establishes. The three drivers all re-dispatch their own
# kernel at a length-1 grid pinned to the fitted MAP cell, so they all meet the
# same two failure shapes and settle them here rather than each writing its own
# tail.
#
# Convergence is read FIRST because it is upstream of everything else the output
# can be missing: a solve that stopped short of a mode has no point for gamma_3
# to expand about and no Gaussian at a mode for the inner k-hat to treat as a
# proposal, whatever the backend supports. Reporting `backend_unsupported` there
# names a capability the backend has -- it computes gamma_3 on every converged
# solve of the same model -- and sends a reader to the wrong layer.
.inner_skew_attach_probe <- function(res, out) {
    if (is.null(out)) return(.inner_skew_decline(res, "backend_unsupported"))
    conv <- out$converged
    if (!is.null(conv) && length(conv) &&
        !any(as.logical(conv) %in% TRUE)) {
        return(.inner_skew_decline(res, "not_converged"))
    }
    if (is.null(out$inner_skew)) {
        return(.inner_skew_decline(res, "backend_unsupported"))
    }
    .inner_skew_attach(res, out)
}

# One-line user-facing reading of an inner-skew decline, for `print` /
# `diagnostic_summary()`. `arms` are the 1-based joint arms with no oracle.
.inner_skew_decline_note <- function(reason, arms = integer(0)) {
    if (is.null(reason) || !length(reason) || is.na(reason)) return(NULL)
    arm_txt <- if (length(arms)) paste0(" (arms ", paste(arms, collapse = ", "), ")") else ""
    switch(reason,
        not_requested = "the inner diagnostic was not requested (control$diagnose_skew = FALSE)",
        no_probe_indices = "no latent indices were probed; pass control$skew_idx",
        coupled_arm = paste0("every scorable arm is coupled through the cell-coupling spec",
                             arm_txt, ", so gamma_3 has no separable per-observation sum to",
                             " read, and no cell third-derivative tensor could be built for",
                             " this fit either; the inner layer is unscorable here"),
        curvature3_unavailable = paste("this likelihood registers no third derivative and",
                                       "exposes no eta-weight callback to finite-difference"),
        no_finite_contribution = paste("an oracle was available but no probed index",
                                       "accumulated a finite contribution"),
        no_oracle = "no per-observation third-derivative oracle was available",
        eta_var_budget = paste("the latent field is larger than the location",
                               "term's solve budget, so gamma_1 was not computed",
                               "(gamma_3 and the importance k-hat are unaffected)"),
        eta_var_solve_failed = paste("the marginal variance of the linear",
                                     "predictor could not be solved for, so",
                                     "gamma_1 was not computed"),
        multi_eta_unit = paste("this likelihood's unit reads several linear",
                               "predictors at once, so the location term's",
                               "contraction widens past what the third-derivative",
                               "oracle exposes"),
        backend_unsupported = "this backend does not compute the inner-Laplace skewness",
        solve_failed = "the probe re-solve failed",
        not_converged = paste("the inner Newton solve did not reach a mode, so",
                              "there is no point for the cubic expansion to",
                              "expand about"),
        pd_eigen_clamp = paste("this fit runs the PSD inner step, which",
                               "eigen-solves a densified Hessian and leaves no",
                               "sparse factor for the probe to solve against;",
                               "control$hessian = \"lm\" restores it"),
        s2z_rank1_factor = paste("the field carries sum-to-zero rank-1 pins that",
                                 "the stored Hessian does not hold, so the live",
                                 "factor is of a different matrix than the solve",
                                 "stepped with"),
        factor_unavailable = "the inner solve left no live factor to probe",
        NULL)
}

# Is the inner layer unscorable BY CONSTRUCTION for this model, as opposed to
# merely unscored? The distinction the combined verdict needs.
.inner_skew_is_structural <- function(reason) {
    !is.null(reason) && length(reason) && !is.na(reason) &&
        reason %in% .INNER_SKEW_STRUCTURAL
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
    share_moderate   = mean(ag >= .nl_diag("gamma3_ok")),
    share_unreliable = mean(ag >= .nl_diag("gamma3_unreliable"))
  )
}

# Aggregate the per-probed-index inner k-hat vector into a per-fit summary.
# Mirrors `.tulpa_inner_skew_summary`: the WORST index governs, since one badly
# approximated coefficient is what a reader has to act on.
#
# The band is read off the MATERIAL indices only -- those whose realized
# importance efficiency `rel_ess` falls below `.nl_diag("inner_k_material_ess")`.
# A Pareto shape index is scale-free, so an index whose weights are uniform (the
# inner Gaussian already reproduces the conditional posterior over the sampled
# region) still returns a shape, fitted to the residual wiggle; banding that
# would flag healthy fits. See the settings note for the measured values. The
# raw shape is reported regardless, so nothing is hidden; `weights_uniform`
# records that no index carried a correction worth describing. Returns NULL when
# nothing was computed.
.tulpa_inner_k_summary <- function(k, is_ess = NULL, rel_ess = NULL) {
  if (is.null(k) || length(k) == 0L) return(NULL)
  ok <- is.finite(k)
  n_scored <- sum(ok)
  if (n_scored == 0L) {
    return(list(max_pareto_k = NA_real_, band = NA_character_,
                min_is_ess = NA_real_, min_rel_ess = NA_real_,
                weights_uniform = NA, n_material = 0L,
                n_scored = 0L, n_probed = length(k)))
  }
  ess <- if (is.null(is_ess)) numeric(0) else is_ess[is.finite(is_ess)]
  rel <- if (is.null(rel_ess)) rep(NA_real_, length(k)) else rel_ess
  material <- ok & is.finite(rel) & rel < .nl_diag("inner_k_material_ess")
  # A fit whose backend reported no efficiency at all cannot be gated, so every
  # scored index counts as material rather than being silently waved through.
  if (!any(is.finite(rel))) material <- ok
  list(
    max_pareto_k    = max(k[ok]),
    band            = if (any(material)) .tulpa_khat_band(max(k[material])) else "good",
    min_is_ess      = if (length(ess)) min(ess) else NA_real_,
    min_rel_ess     = if (any(is.finite(rel))) min(rel[is.finite(rel)]) else NA_real_,
    weights_uniform = !any(material),
    n_material      = sum(material),
    n_scored        = n_scored,
    n_probed        = length(k)
  )
}

# Inner-Laplace importance reliability of a fit, read from the fields
# `.inner_k_attach()` stores at fit time. NULL when the fit carries neither a
# k-hat nor a reason (a backend predating the diagnostic).
.tulpa_inner_k_reliability <- function(fit) {
  jf <- if (!is.null(fit$joint_fit)) fit$joint_fit else fit
  # `[[` throughout: on a DECLINED fit the only field carrying the
  # `inner_pareto_k` prefix is `inner_pareto_k_declined`, and `$` would
  # partial-match the k-hat to that reason string -- a decline read back as a
  # computed value.
  s <- .tulpa_inner_k_summary(jf[["inner_pareto_k"]],
                              jf[["inner_pareto_k_is_ess"]],
                              jf[["inner_pareto_k_rel_ess"]])
  reason <- jf[["inner_pareto_k_declined"]] %||% NA_character_
  if (is.null(s)) {
    if (is.na(reason[1L])) return(NULL)
    s <- list(max_pareto_k = NA_real_, band = NA_character_,
              min_is_ess = NA_real_, min_rel_ess = NA_real_,
              weights_uniform = NA, n_material = 0L,
              n_scored = 0L, n_probed = 0L)
  }
  c(s, list(declined = reason[1L]))
}

# One layer's band collapsed to a state: "bad" (unreliable), "ok" (borderline),
# "na" (never assessed for this fit/backend), or "good" (anything else).
.tulpa_layer_state <- function(band) {
  if (is.na(band)) return("na")
  if (identical(band, "unreliable")) return("bad")
  if (identical(band, "ok")) return("ok")
  "good"
}

# Combine the outer (grid-quadrature / PSIS) band with the inner (Laplace
# skewness) band into a single whole-fit verdict: reliable only when BOTH
# layers are reliable; otherwise "scoped", naming which layer(s) degrade, so
# the caller never reads a high outer k-hat (say) as "the whole fit is
# broken" when the inner layer is fine, or vice versa -- the framing this
# layering exists to fix (42/78 occu_cover species read as "broken" on outer k-hat
# alone when their point estimates, governed by the inner layer, were fine).
#
# A layer can also be "na" -- never assessed (gamma_3 not wired for a coupled
# likelihood; a multi-block/multi-axis outer grid that declines to a guessed
# support transform) -- which must never collapse into the SAME string as
# "assessed and good", or a batch consumer reading the verdict off many fits
# cannot tell "outer bad, inner genuinely fine" from "outer bad, inner never
# checked". Every combination naming an "na" layer says so
# explicitly ("... not assessed"), so `grepl("not assessed", reliability)`
# reliably separates the two.
# An "na" layer is further split by WHY it was not assessed: a layer that
# CANNOT be assessed for this model or family -- a coupled
# multi-process likelihood has no per-observation term gamma_3 can score, and a
# car_proper `rho_car` axis has no support the outer k-hat may guess -- will
# never become assessable, so the verdict says so rather than implying a rerun
# with the right knob would fill it in. The `"not assessed"` wording is kept in
# every such verdict (a documented `grepl("not assessed", ...)` contract),
# with the permanence as a qualifier on top.
#
# The INNER layer carries two scores, not one: the cubic
# term gamma_3 and the importance k-hat, both read off the same probed
# conditional-mean curve through the same joint density. `.tulpa_inner_layer()`
# resolves them into the one band this verdict is built on -- the worse of the
# two where both computed, the one that did where only one did. That is what
# lets a fully coupled fit, whose cubic term can never be computed, still get an
# inner verdict rather than "not assessed".
.tulpa_inner_layer <- function(inner_band, inner_declined = NA_character_,
                               inner_k_band = NA_character_,
                               inner_k_declined = NA_character_) {
  s_skew <- .tulpa_layer_state(inner_band)
  s_k    <- .tulpa_layer_state(inner_k_band)
  if (s_skew == "na" && s_k == "na") {
    # Neither score exists. Keep the cubic term's reason: it is the one that
    # separates a structurally unscorable model class from an unset knob.
    return(list(band = NA_character_, declined = inner_declined))
  }
  if (s_skew == "na") return(list(band = inner_k_band, declined = NA_character_))
  if (s_k == "na")    return(list(band = inner_band,   declined = NA_character_))
  rank <- c(good = 0L, ok = 1L, bad = 2L)
  worse <- if (rank[[s_k]] > rank[[s_skew]]) inner_k_band else inner_band
  list(band = worse, declined = NA_character_)
}

.tulpa_combined_reliability <- function(outer_band, inner_band,
                                        inner_declined = NA_character_,
                                        outer_declined = NA_character_,
                                        inner_k_band = NA_character_,
                                        inner_k_declined = NA_character_) {
  layer <- .tulpa_inner_layer(inner_band, inner_declined, inner_k_band,
                              inner_k_declined)
  inner_band     <- layer$band
  inner_declined <- layer$declined

  outer_state <- .tulpa_layer_state(outer_band)
  inner_state <- .tulpa_layer_state(inner_band)

  inner_na <- if (.inner_skew_is_structural(inner_declined))
    "inner Laplace not assessed (unscorable for this model class)"
    else "inner Laplace not assessed"
  outer_na <- if (.k_decline_is_permanent(outer_declined))
    "outer integration not assessed (unscorable for this family)"
    else "outer integration not assessed"

  if (outer_state == "na" && inner_state == "na") {
    return(paste0("not computed (", outer_na, "; ", inner_na, ")"))
  }
  if (outer_state == "bad" && inner_state == "bad") {
    return("unreliable (both outer integration and inner Laplace flagged)")
  }
  if (outer_state == "bad" && inner_state == "na") {
    return(paste0("scoped: outer (hyperparameter) integration flagged; ", inner_na))
  }
  if (outer_state == "bad") return("scoped: outer (hyperparameter) integration flagged")
  if (inner_state == "bad" && outer_state == "na") {
    return(paste0("scoped: inner (latent-field) Laplace flagged; ", outer_na))
  }
  if (inner_state == "bad") return("scoped: inner (latent-field) Laplace flagged")
  if (outer_state == "ok" && inner_state == "ok") return("usable (both layers borderline)")
  if (outer_state == "ok" && inner_state == "na") {
    return(paste0("scoped: outer integration borderline; ", inner_na))
  }
  if (outer_state == "ok") return("scoped: outer integration borderline")
  if (inner_state == "ok" && outer_state == "na") {
    return(paste0("scoped: inner Laplace borderline; ", outer_na))
  }
  if (inner_state == "ok") return("scoped: inner Laplace borderline")
  if (outer_state == "na") return(paste0(outer_na, "; inner Laplace good"))
  if (inner_state == "na") return(paste0("outer integration good; ", inner_na))
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
  res <- .inner_skew_decline(res, "not_requested")
  if (!compute) return(res)

  w <- res$weights
  if (is.null(w) || !any(is.finite(w))) {
    return(.inner_skew_decline(res, "backend_unsupported"))
  }
  map_idx <- which.max(w)

  probe_idx <- if (is.null(skew_idx)) seq_len(max(as.integer(p_fixed %||% 0L), 0L))
              else as.integer(skew_idx)
  if (length(probe_idx) == 0L) {
    return(.inner_skew_decline(res, "no_probe_indices"))
  }
  res <- .inner_skew_decline(res, "solve_failed")

  cargs_no_ckpt <- utils::modifyList(cargs, list(checkpoint_path = ""))

  # The probe is its own function so its `return(NULL)` guards return from IT.
  # A `return()` inside a tryCatch() expression evaluates in the ENCLOSING
  # function's frame, so written inline those guards returned NULL as the whole
  # fit instead of skipping the probe.
  probe <- function() {
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
  }
  out <- tryCatch(probe(), error = function(e) NULL)

  .inner_skew_attach_probe(res, out)
}

# =============================================================================
# Inner-Laplace skew CORRECTION -- the consumer of gamma_3.
#
# Everything above SCORES the inner Gaussian. Rue, Martino & Chopin (2009)
# Sec 3.2.3 compute the same terms in order to correct the marginal, and
# without a consumer the inner layer is nested approximation with no debias.
# `.nl_skew_marginal()` is that consumer: given each coordinate's Gaussian
# marginal (mu_i, sigma_i), its gamma_3 and its gamma_1, it reports
# Cornish-Fisher quantiles about the centre gamma_1 + gamma_3 / 2 wherever the
# band says the expansion is in its regime, and Gaussian ones everywhere else.
#
# WHICH CORRECTION, AND WHY NOT THE PAPER'S. RMC fit a skew normal to their
# eq. (22) under three constraints: mean gamma^(1), variance 1, and third
# log-density derivative at the mode gamma^(3). A skew normal saturates
# (attainable |skewness| < ~0.995, shape parameter diverging as that bound is
# approached) inside the very band this correction is gated to; Cornish-Fisher
# is the quantile-side inverse of the same Edgeworth series gamma_3 is the
# leading term of, is linear in gamma_3 so it does not saturate, and returns
# quantiles directly, which is what the summaries need. The derivation and the
# monotonicity condition are in src/cornish_fisher.h.
#
# THE CENTRE, AND WHERE IT DIFFERS FROM THEIRS. eq. (22) does not have mean
# zero: expanding it gives E[z] = gamma_1 + gamma_3 / 2, so the reshaped variate
# is placed there (src/cornish_fisher.h). RMC constrain their
# skew normal to mean gamma^(1) instead, setting aside the contribution the
# cubic term makes to the mean. The centre used here is the one their own
# expansion implies, measured against exact quadrature rather than adopted.
# A coordinate with no gamma_1 declines the whole correction: reading it as zero
# would assert a value for the location term rather than an absence of one.
#
# gamma_3 is also a LOWER BOUND on the true skewness, not a two-sided estimate.
# test-inner-skew.R pins the ratio against exact quadrature: 0.4 to 0.7 on the
# coupled fixture's occupancy coordinate, above 0.8 on its detection coordinate
# and on the separable rare-event binomials. A marginal corrected from it
# therefore moves PART of the way, and an index banded `good` may truly be `ok`.
# Both are properties of the input, and both show up in the measured numbers in
# test-inner-skew-correction.R rather than being argued away. The scale is not
# corrected either: eq. (22) leaves it at sigma_i to the order kept.
# =============================================================================

# Skew-corrected marginal quantiles for a block of coordinates.
#
# `mu` / `sigma` are the Gaussian marginal centre and scale, `gamma3` the
# per-coordinate cubic term and `gamma1` the location term (NaN where either is
# not computable, which declines that coordinate), `probs` the requested
# probabilities. Returns the [length(mu) x length(probs)] quantile matrix and a
# per-coordinate logical saying whether the row is the corrected or the Gaussian
# quantile -- a declined coordinate is still reported, from the Gaussian, so the
# table is always complete and always says how each row was produced.
#
# `enabled = FALSE` short-circuits to the Gaussian quantiles with every flag
# FALSE, so a caller that leaves the correction off gets bit-for-bit the numbers
# it got before this existed.
.nl_skew_marginal <- function(mu, sigma, gamma3, gamma1, probs, enabled = TRUE,
                              max_abs_gamma3 = .nl_diag("gamma3_unreliable"),
                              max_abs_centre = .nl_diag("centre_unreliable")) {
  mu    <- as.numeric(mu)
  sigma <- as.numeric(sigma)
  n     <- length(mu)
  z     <- stats::qnorm(as.numeric(probs))
  # A short or absent term leaves the tail of the block unscored, which is
  # NA ("not computable") and never 0 ("no skew" / "no shift"). gamma_1 is
  # required, so a block reaching here without one reports Gaussian quantiles.
  pad <- function(v) {
    out <- rep(NA_real_, n)
    if (isTRUE(enabled) && length(v)) {
      m <- min(n, length(v))
      out[seq_len(m)] <- as.numeric(v)[seq_len(m)]
    }
    out
  }
  out <- cpp_cornish_fisher_quantile(mu, sigma, pad(gamma3), pad(gamma1), z,
                                     as.numeric(max_abs_gamma3),
                                     as.numeric(max_abs_centre))
  list(q = out$q, applied = as.logical(out$applied))
}

# Map a per-probed-index vector onto the p fixed-effect coefficients through the
# `inner_skew_idx` correspondence. The probe defaults to the p fixed-effects
# latent indices, so the default fit gives values[j] for coefficient j; a fit
# whose `control$skew_idx` probed something else is read through the index and
# leaves any unprobed coefficient at `fill` (never 0). A fit carrying values but
# no usable index is read positionally, which is what the default probe means.
.nl_probe_to_fixed <- function(values, idx, p, fill = NA_real_) {
  p <- max(as.integer(p), 0L)
  out <- rep(fill, p)
  if (p == 0L || is.null(values) || !length(values)) return(out)
  if (is.null(idx) || length(idx) != length(values)) {
    m <- min(p, length(values))
    out[seq_len(m)] <- values[seq_len(m)]
    return(out)
  }
  idx <- as.integer(idx)
  keep <- which(idx >= 1L & idx <= p)
  out[idx[keep]] <- values[keep]
  out
}

# gamma_3 per FIXED EFFECT, from the per-probed-index vector the kernel attached.
.nl_skew_by_fixed <- function(fit, p) {
  g <- fit[["inner_skew"]]
  if (is.null(g) || !length(g) || p <= 0L) return(rep(NA_real_, max(p, 0L)))
  .nl_probe_to_fixed(as.numeric(g), fit[["inner_skew_idx"]], p)
}

# gamma_1 per FIXED EFFECT, read through the same probe
# correspondence. A fit whose kernel declined the location term carries none, so
# every coefficient is NA -- "not computable", which the correction reads as a
# decline rather than as a zero shift.
.nl_gamma1_by_fixed <- function(fit, p) {
  g <- fit[["inner_skew_gamma1"]]
  if (is.null(g) || !length(g) || p <= 0L) return(rep(NA_real_, max(p, 0L)))
  .nl_probe_to_fixed(as.numeric(g), fit[["inner_skew_idx"]], p)
}

# The COMBINED inner band per fixed effect: the worse of gamma_3's band and the
# importance k-hat's, resolved per index by `.subspace_bands()` -- the same
# resolution `.tulpa_inner_layer()` performs for the whole-fit verdict, kept per
# index because that is what a per-coefficient gate needs. Also returns the
# k-hat itself, so a declined coefficient can report the number that declined it.
.nl_inner_bands_by_fixed <- function(fit, p) {
  b <- .subspace_bands(fit)
  list(pareto_k = .nl_probe_to_fixed(b$pareto_k, b$idx, p),
       rel_ess  = .nl_probe_to_fixed(b$rel_ess, b$idx, p),
       band     = .nl_probe_to_fixed(b$band, b$idx, p, NA_character_))
}

# Record the skew correction's inputs and its per-coefficient eligibility on the
# fit, so `summary()` / `confint()` are readers rather than deciders and a
# stored fit says which coefficients its intervals were corrected on. One attach
# point for every nested driver, called right after the gamma_3 attach.
#
# `enabled` is the `control$skew_correct` knob. Eligibility is the band gate;
# the monotonicity condition also depends on the requested interval level, so the
# final per-level decision is `.nl_skew_marginal()`'s and is reported as an
# attribute on the summary it produced.
#
# THE GATE IS THE COMBINED INNER BAND, NOT gamma_3 ALONE. The
# inner layer carries two scores of the same Gaussian approximation -- the cubic
# term and the importance k-hat -- and a coefficient the k-hat calls
# unreliable is one whose inner Gaussian misfits the conditional posterior in
# ways a leading-order cubic term does not describe. Measured on the rare-event
# binomial-logit fixture of `test-inner-skew-correction.R`, gamma_3 alone admits
# every replicate while the combined band is `unreliable` on 38% of them, driven
# by the k-hat; those are the same fits the subspace debias selects for exact
# sampling, so reading the two scores differently in the two places had one fit
# judged reliable enough to correct and unreliable enough to resample.
#
# Every coefficient carries `reason` from a closed vocabulary, so a coefficient
# reported from the Gaussian says which score declined it rather than going
# silently uncorrected:
#
#   eligible               the correction applies, subject to the per-level
#                          monotonicity check in `.nl_skew_marginal()`
#   not_enabled            `control$skew_correct` is FALSE
#   gamma3_not_computable  no cubic term at this coefficient (NaN, never 0)
#   gamma3_unreliable      |gamma_3| at or past the band cutoff
#   inner_k_unreliable     the importance k-hat flags this coordinate while
#                          gamma_3 does not
#   gamma1_not_computable  no location term at this coefficient (NaN, never 0);
#                          `inner_skew_gamma1_declined` on the fit says why the
#                          kernel could not produce one
#   centre_unreliable      |gamma_1 + gamma_3 / 2| at or past the CENTRE band.
#                          The correction relocates the
#                          marginal by that many standard errors, so banding
#                          |gamma_3| alone bounds only the reshaping. The
#                          shipped cutoff is `Inf`, so this
#                          reason is reachable only at a caller-supplied finite
#                          `max_abs_centre`; it is retained, along with its
#                          place in the precedence below, because the cutoff is
#                          a setting and restoring it must restore the record
.SKEW_CORRECT_REASONS <- c("eligible", "not_enabled", "gamma3_not_computable",
                           "gamma3_unreliable", "inner_k_unreliable",
                           "gamma1_not_computable", "centre_unreliable")

# `max_abs_centre` is the CENTRE band, defaulted to the setting the quantile
# path reads. It is an argument for the same reason `.nl_skew_marginal()`
# carries one: the shipped cutoff is `Inf`, and a caller that wants the decline
# path has to drive the shipped predicate at a finite cutoff rather than write
# a second rule. The SHAPE band is not an argument -- `band` below is the
# reported gamma_3 band and reads the setting, so a caller-supplied shape cutoff
# would classify a coefficient the two disagreed about as `centre_unreliable`.
.nl_skew_correction_attach <- function(res, p_fixed, enabled,
                                       max_abs_centre = .nl_diag("centre_unreliable")) {
  p <- max(as.integer(p_fixed %||% 0L), 0L)
  g <- .nl_skew_by_fixed(res, p)
  g1 <- .nl_gamma1_by_fixed(res, p)
  band <- vapply(g, .tulpa_gamma3_band, character(1))
  inner <- .nl_inner_bands_by_fixed(res, p)
  # WHETHER the bands admit a coefficient is the quantile path's own predicate,
  # read here rather than re-derived; only WHY is classified below, and the
  # centre is the cause left once the two terms' own causes are named.
  cf <- cpp_cornish_fisher_bands(g, g1,
                                 as.numeric(.nl_diag("gamma3_unreliable")),
                                 as.numeric(max_abs_centre))

  reason <- rep("eligible", p)
  if (!isTRUE(enabled)) {
    reason <- rep("not_enabled", p)
  } else if (p > 0L) {
    reason[!is.finite(g)] <- "gamma3_not_computable"
    reason[reason == "eligible" & !is.finite(g1)] <- "gamma1_not_computable"
    reason[reason == "eligible" & !is.na(band) & band == "unreliable"] <-
      "gamma3_unreliable"
    reason[reason == "eligible" & !is.na(inner$band) &
             inner$band == "unreliable"] <- "inner_k_unreliable"
    reason[reason == "eligible" & !cf$in_band] <- "centre_unreliable"
  }

  res$skew_correction <- list(
    enabled       = isTRUE(enabled),
    gamma3        = g,
    gamma1        = g1,
    centre        = as.numeric(cf$centre),
    gamma1_declined = res[["inner_skew_gamma1_declined"]] %||% NA_character_,
    band          = band,
    pareto_k      = inner$pareto_k,
    band_combined = inner$band,
    eligible      = reason == "eligible",
    reason        = reason,
    reliability   = .nl_fit_reliability(res)
  )
  res
}

# The whole-fit combined verdict at attach time, from the summaries the
# diagnostics already stored on the fit. `diagnostics()` recomputes the same
# string from the same reader functions; recording it here is what lets a stored
# fit say the verdict its per-coefficient eligibility was decided under.
.nl_fit_reliability <- function(res) {
  psis    <- .tulpa_psis_reliability(res)
  inner   <- .tulpa_inner_skew_reliability(res)
  inner_k <- .tulpa_inner_k_reliability(res)
  .tulpa_combined_reliability(
    .tulpa_khat_band(psis$pareto_k),
    if (is.null(inner)) NA_character_ else inner$band,
    if (is.null(inner)) NA_character_ else inner$declined,
    psis$pareto_k_declined,
    if (is.null(inner_k)) NA_character_ else inner_k$band,
    if (is.null(inner_k)) NA_character_ else inner_k$declined)
}

# The correction's state on a fit, defaulted for a fit that predates it or a
# backend that does not attach one.
.nl_skew_correction <- function(object, p) {
  sc <- object[["skew_correction"]]
  if (is.null(sc)) {
    return(list(enabled = FALSE, gamma3 = rep(NA_real_, p),
                gamma1 = rep(NA_real_, p),
                centre = rep(NA_real_, p),
                gamma1_declined = NA_character_,
                band = rep(NA_character_, p),
                pareto_k = rep(NA_real_, p),
                band_combined = rep(NA_character_, p),
                eligible = rep(FALSE, p),
                reason = rep("not_enabled", p),
                reliability = NA_character_))
  }
  sc
}

# gamma_3 masked to the coefficients the gate admits: NA elsewhere, which
# `.nl_skew_marginal()` reads as "not computable here" and reports from the
# Gaussian. The eligibility record is therefore what the quantile path consumes,
# not a parallel note beside it.
.nl_skew_gamma3_eligible <- function(sc) .nl_skew_term_eligible(sc, "gamma3")

# The same mask on the location term. Both are read by `.nl_fixed_interval()`,
# and a coefficient masked in either declines: the correction is the centre AND
# the reshaping, and half of it is not a correction.
.nl_skew_gamma1_eligible <- function(sc) .nl_skew_term_eligible(sc, "gamma1")

.nl_skew_term_eligible <- function(sc, field) {
  g <- sc[[field]]
  el <- sc$eligible
  if (is.null(g)) return(NULL)
  if (is.null(el) || length(el) != length(g)) return(g)
  g[!el] <- NA_real_
  g
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
  k  <- jf$pareto_k %||% NA_real_
  # WHY it is NA, when it is. A backend with no outer
  # hyperparameter grid at all (SMC, VI, ...) records no reason of its own,
  # because it never reached the diagnostic -- name that case here rather than
  # leave the reader with a bare NA.
  declined <- jf$pareto_k_declined %||% NA_character_
  if (is.na(declined) && !is.finite(k) && is.null(jf$weights)) {
    declined <- .k_decline_label(
      .k_decline("not_applicable", "this backend has no outer hyperparameter grid"))
  }
  list(pareto_k = k,
       pareto_k_is_ess = jf$pareto_k_is_ess %||% NA_real_,
       pareto_k_scope = jf$pareto_k_scope %||% NA_character_,
       pareto_k_declined = declined)
}

# Outer-integration REGIME of a nested-Laplace fit, read
# from the fields `.joint_attach_pareto_k_regime()` stores at fit time.
#
# `pareto_k` alone cannot separate two very different situations, and a bare
# threshold on it bins them together: a SPREAD grid whose integration is
# genuinely tail-misfit, versus a COLLAPSED grid where the outer integration
# degenerated to a point evaluation at the modal hyperparameter and the k-hat is
# scoring a mode-Gaussian's stand-in for the hyperparameter marginal instead.
# The collapse is itself benign or actionable depending on where the dominant
# cell sits -- interior (the grid bracketed the mode; only integrated
# hyperparameter uncertainty is missing) or against a grid boundary (the grid
# may be too narrow, so widen that axis and refit).
#
# `outer_skew_max` is the largest |skewness| of the hyperparameter marginal in
# the proposal's whitened coordinate, computed only on a fit whose k-hat
# triggered the skew-normal rescue pass -- so `NA` means "the Gaussian proposal
# already fit", not "symmetric and unchecked". NULL when the fit stores no
# regime at all (an older fit, or a backend that does not populate it).
.tulpa_outer_regime <- function(fit) {
  jf <- if (!is.null(fit$joint_fit)) fit$joint_fit else fit
  rg <- jf$pareto_k_regime
  if (is.null(rg)) return(NULL)
  sk <- jf$pareto_k_outer_skew
  sk <- if (is.null(sk) || !any(is.finite(sk))) NA_real_ else max(abs(sk[is.finite(sk)]))
  list(regime         = as.character(rg),
       edge_axes      = jf$pareto_k_grid_edge_axes  %||% character(0),
       edge_sides     = jf$pareto_k_grid_edge_sides %||% character(0),
       outer_skew_max = sk)
}

# One-line reading of an outer regime, for `print` and `diagnostic_summary()`.
# Returns NULL for a spread grid (nothing to explain) or an unknown regime.
.tulpa_outer_regime_note <- function(rg) {
  if (is.null(rg) || is.na(rg$regime)) return(NULL)
  if (identical(rg$regime, "collapsed_interior")) {
    return(paste("outer grid collapsed onto an interior mode: hyperparameter",
                 "uncertainty is not integrated (empirical Bayes at the mode);",
                 "pareto_k here scores the mode-Gaussian's fit to the",
                 "hyperparameter marginal, not the point estimates"))
  }
  if (identical(rg$regime, "collapsed_edge")) {
    ax <- if (length(rg$edge_axes))
      paste(sprintf("%s (%s)", rg$edge_axes, rg$edge_sides), collapse = ", ")
      else "an axis"
    return(paste0("outer grid collapsed against a boundary node on ", ax,
                  ": the grid may be too narrow -- widen that axis and refit ",
                  "to confirm the mode is bracketed"))
  }
  NULL
}

# Outer-grid PLACEMENT of a nested-Laplace fit: which
# default axes do not contain their own posterior mode, and what the engine did
# about it.
#
# The neighbour above reads the grid REGIME, a joint property of the whole
# tensor. This reads the per-axis one: an axis whose own marginal is maximal at
# an endpoint has its mode at or beyond that endpoint, and the span is
# integrating a tail at any spacing. The two disagree exactly where a crossed
# grid hides one railed axis behind another axis's spread.
#
# `declined` is why no axis was moved. It exists for every registry fit,
# including one whose family the rescue covers no axis of -- that case used to
# return before stamping anything and was indistinguishable from a fit the
# rescue never applied to. NULL when the fit records neither, so an older fit
# and a backend that does not populate them report nothing rather than "none".
.tulpa_grid_placement <- function(fit) {
  jf <- if (!is.null(fit$joint_fit)) fit$joint_fit else fit
  railed <- jf$outer_grid_railed_axes
  placed <- jf$outer_grid_placement
  if (is.null(railed) && is.null(placed)) return(NULL)
  cl <- jf$outer_grid_recenter_sd_clamp
  list(placement = as.character(placed %||% NA_character_),
       railed    = as.character(railed %||% character(0)),
       moved     = as.character(jf$outer_grid_recenter_axes %||% character(0)),
       clamped   = if (is.null(cl)) character(0) else
                     stats::setNames(as.character(cl), names(cl)),
       declined  = as.character(jf$outer_grid_recenter_declined %||%
                                  NA_character_))
}

# One-line reading of a placement. NULL when every axis brackets its own mode
# and nothing had to move -- the common case, which needs no sentence.
.tulpa_grid_placement_note <- function(pl) {
  if (is.null(pl)) return(NULL)
  if (identical(pl$placement, "auto_recentered")) {
    ax <- if (length(pl$moved)) paste(pl$moved, collapse = ", ") else "an axis"
    msg <- paste0("outer grid re-centred on ", ax,
                  ": the default span did not contain that axis's posterior ",
                  "mode, so the reported grid is not the default one")
    # A bound the mode SD hit is not a spread the stencil measured, and the
    # interval is read off the span it produced.
    cl  <- pl$clamped %||% character(0)
    hit <- if (!length(cl) || is.null(names(cl))) character(0) else
      names(cl)[cl %in% c("ceiling", "floor")]
    if (length(hit)) {
      msg <- paste0(msg, "; the mode SD hit its bound on ",
                    paste(hit, collapse = ", "), ", so that axis was laid from ",
                    "a substituted spread rather than the curvature the ",
                    "stencil measured")
    }
    return(msg)
  }
  if (!length(pl$railed)) return(NULL)
  ax <- paste(pl$railed, collapse = ", ")
  why <- if (is.na(pl$declined)) "" else
    paste0(" (not moved: ", pl$declined, ")")
  paste0("outer grid axis maximal at its own boundary on ", ax, why,
         ": the span does not contain that axis's mode, so its marginal is a ",
         "truncated tail at any spacing -- widen or pin that axis and refit")
}

# What the reported per-axis hyperparameter intervals were read off, and how much
# of the support underneath them is a quadrature design rather than posterior
# mass. NULL for a fit that does not record it.
#
# `within_cell` is the second half of the same question: the
# node-set kind says what the integrator LEFT, the within-cell construction says
# how each cell's mass was spread inside its own box when the grid was read
# back. A fit carries one per axis, so a grid whose partition could not be built
# on one axis says which axis fell back and why.
.tulpa_interval_read <- function(fit) {
  jf <- if (!is.null(fit$joint_fit)) fit$joint_fit else fit
  read <- jf$theta_interval_read
  if (is.null(read) || is.na(read)) return(NULL)
  wc <- jf$theta_within_cell
  list(read = as.character(read),
       design_mass = as.numeric(jf$theta_interval_design_mass %||% NA_real_),
       within_cell = as.character(wc %||% character(0)),
       within_cell_requested =
         as.character(jf$within_cell_requested %||% NA_character_),
       within_cell_declined =
         as.character(jf$theta_within_cell_declined %||% character(0)),
       within_cell_axes = names(wc) %||% character(0),
       # One layer further down: the COORDINATE each axis's
       # outer cell edges were mirrored in, and why a declared support's own
       # mirror did not produce them. A declined axis reports the conservative
       # edge -- its extreme coordinate -- rather than a guessed one outside the
       # support the caller named.
       edge_coord = as.character(jf$theta_cell_edge_coord %||% character(0)),
       edge_declined =
         as.character(jf$theta_cell_edge_declined %||% character(0)))
}

# One-line reading of that provenance: a mixed support, or a within-cell
# construction other than the shipped one (including one that was asked for and
# fell back). NULL for the ordinary case -- a homogeneous support read with the
# default construction, which `integration` already names.
.tulpa_interval_read_note <- function(ir) {
  if (is.null(ir)) return(NULL)
  out <- character(0)
  if (identical(ir$read, "mixed")) {
    share <- if (is.finite(ir$design_mass))
      sprintf("%.1f%% of the integration weight", 100 * ir$design_mass) else
      "part of the integration weight"
    out <- c(out, paste0(
      "hyperparameter intervals read off a MIXED support: ", share,
      " sits on locally CCD-refined nodes, which are a moment rule rather ",
      "than posterior mass, so on that part the weighted quantile is bounded ",
      "by the refined cells' own grid neighbourhoods -- widen the base grid ",
      "(or turn local_ccd off) to move weight back onto mass-weighted cells"))
  }
  ax   <- ir$within_cell_axes
  ed   <- ir$edge_declined
  # The vocabulary splits in two, and the two say different things about the
  # bound. `.NL_EDGE_FALLBACK` are the reasons whose edges ARE the extreme
  # coordinates -- a declared support's mirror that left it
  # and a mirror that left the double range -- so the bound
  # is conservative. The other two set a DECLARATION aside, after which the
  # guess ran and its mirror stood, so the bound is a guessed edge and not a
  # conservative one. Reporting the second pair as running to the extreme
  # coordinate would say the opposite of what happened.
  nmof <- function(i) if (length(ax) >= max(i)) ax[i] else as.character(i)
  fbx <- which(!is.na(ed) & ed %in% .NL_EDGE_FALLBACK)
  if (length(fbx)) {
    out <- c(out, paste0(
      "the outer cell edge of ", paste(nmof(fbx), collapse = ", "),
      " could not be mirrored usably (",
      paste(unique(ed[fbx]), collapse = ", "), "), so its interval runs to the ",
      "extreme grid coordinate instead of half a spacing past it -- the ",
      "reported bound is conservative on that side"))
  }
  gsx <- which(!is.na(ed) & nzchar(ed) & !(ed %in% .NL_EDGE_FALLBACK))
  if (length(gsx)) {
    out <- c(out, paste0(
      "the support declared for ", paste(nmof(gsx), collapse = ", "),
      " could not be used (", paste(unique(ed[gsx]), collapse = ", "),
      "), so its outer cell edge was mirrored in a coordinate guessed from the ",
      "values instead"))
  }
  # The note fires on a fit read with something OTHER than the engine's own
  # default, whichever that is. Naming the default here as a
  # literal is what made it say "rather than the default 'chord'" on a fit that
  # had just been read with the default.
  req <- ir$within_cell_requested
  dflt <- .nl_diag("within_cell")
  if (!length(req) || is.na(req) || identical(req, dflt)) {
    return(if (length(out)) out else NULL)
  }
  used <- ir$within_cell
  fell <- which(!is.na(used) & used != req)
  out <- c(out, paste0(
    "hyperparameter intervals read with the '", req, "' within-cell ",
    "construction rather than the default '", dflt, "': ",
    if (identical(req, "chord"))
      paste0("each cell's mass placed at its own coordinate rather than spread ",
             "over its box, which is the wider read of the two")
    else
      paste0("the same cell masses spread over the cells' own boxes, so an ",
             "endpoint is resolved to within one box"),
    " -- see `outer_grid_h_over_sd` for how wide a box is on each axis"))
  if (length(fell)) {
    nm <- if (length(ax) >= max(fell)) ax[fell] else as.character(fell)
    why <- stats::na.omit(unique(ir$within_cell_declined[fell]))
    out <- c(out, paste0(
      "'", req, "' declined on ", paste(nm, collapse = ", "), " (",
      paste(why, collapse = ", "), "): those axes report the '",
      as.character(used[fell][1L]), "' read instead"))
  }
  out
}

# The RESOLUTION of a fit's outer grid: per axis, the cell
# width and the posterior SD in that axis's own coordinate, and their ratio.
#
# The neighbours above read WHAT the interval was built from. This reads how
# finely the thing it was built from divides the posterior, which is what bounds
# how much of the reported number is the grid rather than the posterior. NULL
# for a fit that records none -- an older fit, or a backend whose summary does
# not go through `.nl_posterior_moments()`.
.tulpa_grid_resolution <- function(fit) {
  jf <- if (!is.null(fit$joint_fit)) fit$joint_fit else fit
  r <- jf$outer_grid_h_over_sd
  if (is.null(r) || !length(r)) return(NULL)
  ok <- is.finite(r)
  nm <- names(r) %||% as.character(seq_along(r))
  dec <- jf$outer_grid_resolution_declined
  if (is.null(dec) || length(dec) != length(r)) {
    dec <- setNames(rep(NA_character_, length(r)), nm)
  }
  railed <- jf$outer_grid_railed_axes %||% character(0)
  # `resolved` is a claim about the WHOLE grid, so an axis that could not be
  # scored withholds it rather than being skipped over. A fit
  # where nothing scored still reports -- what it reports is that nothing could
  # be, which is the case a NULL return used to make indistinguishable from a
  # fit that carries no resolution at all.
  list(h_over_sd = r,
       h         = jf$outer_grid_cell_width %||% rep(NA_real_, length(r)),
       sd        = jf$outer_grid_axis_sd    %||% rep(NA_real_, length(r)),
       max       = if (any(ok)) max(r[ok]) else NA_real_,
       coarsest  = if (any(ok)) nm[ok][which.max(r[ok])] else NA_character_,
       declined  = dec,
       unscored  = nm[!ok],
       railed    = railed,
       n_scored  = sum(ok),
       n_axes    = length(r),
       resolved  = all(ok) && all(r[ok] <= .nl_diag("grid_resolved")))
}

# Reading of a resolution. NULL when every axis scored AND every one is at or
# below `grid_resolved`, where the cells are narrower than the posterior they
# discretize and the within-cell construction stops mattering.
#
# An UNSCORED axis is reported first and separately. Reporting
# only the coarsest SCORED axis pointed the reader at a healthy axis while the
# one whose grid did not contain its own mode went unmentioned, and told them to
# add nodes to the wrong one.
.tulpa_grid_resolution_note <- function(rs) {
  if (is.null(rs) || isTRUE(rs$resolved)) return(NULL)
  out <- character(0)
  if (length(rs$unscored)) {
    why <- rs$declined[rs$unscored]
    lab <- paste0(rs$unscored, " (",
                  ifelse(is.na(why), "reason not recorded", why), ")")
    out <- c(out, paste0(
      "outer grid resolution could not be scored on ", paste(lab, collapse = ", "),
      ": ", rs$n_scored, " of ", rs$n_axes, " axes carry a cell-width / ",
      "posterior-SD ratio, so the grid is not established as resolved on the ",
      "rest -- an axis reading `mode_at_edge` is one whose nodes do not contain ",
      "its own posterior mode, which no spacing statement can be made about"))
  }
  if (length(rs$railed)) {
    out <- c(out, paste0(
      "outer grid does not contain its own posterior mode on ",
      paste(rs$railed, collapse = ", "),
      ": that axis's extreme node carries the modal mass, so its reported ",
      "bound is an extrapolation off the end of the design -- widen the axis ",
      "rather than adding nodes inside it"))
  }
  if (!is.na(rs$max) && rs$max > .nl_diag("grid_resolved")) {
    out <- c(out, paste0(
      "outer grid coarser than its own posterior on ", rs$coarsest,
      " (cell width / posterior SD = ", sprintf("%.2f", rs$max),
      ", both in that axis's own coordinate): the reported interval's ",
      "endpoints are resolved to within one cell, so part of their width ",
      "and their realized coverage are properties of where the grid fell ",
      "rather than of the posterior -- add nodes on that axis to reduce it"))
  }
  if (!length(out)) return(NULL)
  out
}

# The grid axes a fit's own resolved path could not read.
# The nested-Laplace front doors record them on `$axis_fields_dropped`; every
# reader goes through here, so the record has ONE reading and a fit with nothing
# dropped is silent in `diagnostic_summary()`, `print()` and `summary()` alike.
# NULL for such a fit, which is the ordinary case.
.tulpa_axis_dropped <- function(fit) {
  rec <- fit$axis_fields_dropped
  if (is.null(rec) && !is.null(fit$joint_fit)) rec <- fit$joint_fit$axis_fields_dropped
  if (!is.data.frame(rec) || !nrow(rec)) return(NULL)
  rec
}

# One sentence per dropped axis: the field, the block that carried it, and the
# axis the resolved path integrated instead. The path is named with the same
# label the refusal uses (`.NL_AXIS_PATH_LABEL`), so the two halves --
# the pin that errors and the default that is dropped -- read alike.
.tulpa_axis_dropped_note <- function(rec) {
  if (is.null(rec) || !nrow(rec)) return(NULL)
  vapply(seq_len(nrow(rec)), function(i) {
    blk <- if (is.na(rec$block[i])) "the prior block" else
      paste0("prior block ", rec$block[i])
    reads <- trimws(strsplit(rec$integrates[i], ",", fixed = TRUE)[[1L]])
    sprintf(paste("Supplied `%s` on %s '%s' was not used: it is not an axis %s",
                  "reads, and it was an engine default rather than a pin, so it",
                  "was dropped. That path integrates %s."),
            rec$field[i], blk, rec$type[i],
            .NL_AXIS_PATH_LABEL[[rec$path[i]]] %||% rec$path[i],
            paste0("`", reads, "`", collapse = ", "))
  }, character(1))
}

# The compact form a fit's own `print()` carries: which axes went unused, and
# where the full reading is.
.tulpa_axis_dropped_line <- function(rec) {
  if (is.null(rec) || !nrow(rec)) return(NULL)
  paste0("  unused axis fields: ",
         paste(unique(sprintf("%s (%s)", rec$field, rec$type)), collapse = ", "),
         "\n    dropped as engine defaults this path does not read;",
         " see diagnostic_summary()")
}

# Inner-Laplace skewness reliability of a nested-Laplace fit, read from the
# `inner_skew` / `inner_skew_idx` / `inner_skew_dropped` fields
# .nl_inner_skew_at_theta() attaches at fit time (see nested_laplace.R). NULL
# when the fit never ran the diagnostic (control$diagnose_skew = FALSE, or a
# backend that does not yet populate it).
# How many outer-grid cells reported the reported log-determinant from the
# PD-enforced factor instead of the pinned sum-to-zero matrix. On that path the
# escalated matrix is H + lambda I rather than B = H + sum_k coef_k 1_k 1_k', so
# a cell that fell back is weighted against its neighbours on a different
# quantity. NULL where the fit carries no such vector -- every tier that does
# not take the sum-to-zero route, and every fit produced before the flag
# existed.
.tulpa_s2z_fallback_cells <- function(fit) {
  jf <- if (!is.null(fit$joint_fit)) fit$joint_fit else fit
  v <- jf$s2z_log_det_fallback
  if (is.null(v) || !length(v)) return(NULL)
  n <- sum(as.logical(v), na.rm = TRUE)
  if (n == 0L) return(NULL)
  list(n = as.integer(n), n_grid = length(v))
}

.tulpa_inner_skew_reliability <- function(fit) {
  jf <- if (!is.null(fit$joint_fit)) fit$joint_fit else fit
  s <- .tulpa_inner_skew_summary(jf$inner_skew, jf$inner_skew_dropped %||% 0L)
  reason <- jf$inner_skew_declined %||% NA_character_
  arms   <- as.integer(jf$inner_skew_arms_declined %||% integer(0))
  # A fit that computed NOTHING still reports WHY, so an
  # unscorable model class is distinguishable from a disabled knob.
  if (is.null(s)) {
    if (is.na(reason[1L])) return(NULL)
    s <- list(max_abs_gamma3 = NA_real_, band = NA_character_,
              n_scored = 0L, n_probed = 0L, n_dropped = 0L,
              share_moderate = NA_real_, share_unreliable = NA_real_)
  }
  c(s, list(declined = reason[1L], arms_declined = arms))
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

# The zero-row shape of the table above, for a fit whose reliability band is
# available but whose per-parameter body is not. Written from the same column
# set so the two cannot drift.
.tulpa_iid_param_table_empty <- function() {
  .tulpa_iid_param_table(matrix(0.0, 1L, 1L,
                                dimnames = list(NULL, "x")))[0L, , drop = FALSE]
}

# Approximation-reliability table for an i.i.d. deterministic fit. The
# provenance gate lives in `diagnostics()`; this builds the table and attaches
# the PSIS / grid-quadrature headline as attributes. Documented user-side under
# `?laplace_diagnostics`.
#
# DRAWS ARE NEEDED FOR THE BODY, NOT THE HEADLINE. Every
# reliability quantity below -- the outer PSIS k-hat and its regime, the grid
# quadrature ESS, the inner-Laplace gamma_3 and importance k-hat, and the
# combined verdict -- is read off `fit` at fit time and needs no posterior
# sample. Only the per-parameter Monte-Carlo columns and `n_draws` do. So a fit
# with no draws gets the whole band with an empty body, rather than silence
# about a k-hat that has already cleared the escalation threshold; the body's
# absence is recorded in `param_table_declined` and printed.
#
# A fit with NEITHER draws NOR any band quantity -- a plain Laplace / EB fit,
# which has no outer grid to score -- is the one case that still returns NULL:
# there is nothing to report at either layer, and an all-"not assessed" table
# would be a report about the absence of a report.
.tulpa_approx_diag_table <- function(fit, pars = NULL) {
  grid  <- .tulpa_grid_reliability(fit)
  psis  <- .tulpa_psis_reliability(fit)
  inner <- .tulpa_inner_skew_reliability(fit)
  inner_k <- .tulpa_inner_k_reliability(fit)
  regime <- .tulpa_outer_regime(fit)
  placement <- .tulpa_grid_placement(fit)
  iread <- .tulpa_interval_read(fit)
  resolution <- .tulpa_grid_resolution(fit)
  k          <- psis$pareto_k

  draws <- .fit_draws(fit)
  tab <- if (is.null(draws)) NULL else .tulpa_iid_param_table(draws, pars = pars)
  param_declined <- NA_character_
  if (is.null(draws)) {
    has_band <- !is.null(grid) || !is.null(inner) || !is.null(inner_k) ||
      !is.null(regime) || is.finite(k)
    if (!has_band) {
      message(.tulpa_no_draws_note(fit, "diagnostics"),
              " No approximation-reliability quantity was computed for it ",
              "either, so there is nothing to report.")
      return(NULL)
    }
    param_declined <- paste(
      "the fit carries no posterior draws, so the per-parameter mean / sd /",
      "ESS / rhat columns are empty; tulpa_posterior_draws(fit) samples the",
      "retained outer-grid mixture if a sample is wanted")
    tab <- .tulpa_iid_param_table_empty()
  } else if (is.null(tab)) {
    # Draws are present and `pars` matched nothing -- that is a bad selector,
    # not an absent representation, and stays an empty answer.
    return(NULL)
  }

  outer_band <- .tulpa_khat_band(k)
  inner_band <- if (is.null(inner)) NA_character_ else inner$band
  inner_k_band <- if (is.null(inner_k)) NA_character_ else inner_k$band

  attr(tab, "pareto_k")        <- k
  attr(tab, "pareto_k_band")   <- outer_band
  attr(tab, "pareto_k_is_ess") <- psis$pareto_k_is_ess
  attr(tab, "scope")           <- psis$pareto_k_scope
  if (!is.na(psis$pareto_k_declined)) {
    attr(tab, "pareto_k_declined")      <- psis$pareto_k_declined
    attr(tab, "pareto_k_declined_note") <- .k_decline_note(psis$pareto_k_declined)
  }
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
    if (!is.na(inner$declined)) {
      attr(tab, "inner_skew_declined")      <- inner$declined
      attr(tab, "inner_skew_arms_declined") <- inner$arms_declined
      attr(tab, "inner_skew_declined_note") <-
        .inner_skew_decline_note(inner$declined, inner$arms_declined)
    }
  }
  if (!is.null(regime)) {
    attr(tab, "outer_regime")    <- regime$regime
    attr(tab, "grid_edge_axes")  <- regime$edge_axes
    attr(tab, "grid_edge_sides") <- regime$edge_sides
    attr(tab, "outer_skew_max")  <- regime$outer_skew_max
    attr(tab, "outer_regime_note") <- .tulpa_outer_regime_note(regime)
  }
  if (!is.null(placement)) {
    attr(tab, "grid_placement")        <- placement$placement
    attr(tab, "grid_railed_axes")      <- placement$railed
    attr(tab, "grid_recentred_axes")   <- placement$moved
    attr(tab, "grid_placement_declined") <- placement$declined
    attr(tab, "grid_placement_note")   <- .tulpa_grid_placement_note(placement)
  }
  # What the reported per-axis hyperparameter intervals were read off, and how
  # finely the node set they came from divides the posterior. Both are
  # properties of the OUTER read rather than of the inner solve,
  # and neither had a consumer before: `.tulpa_interval_read_note()` was written
  # for the mixed support and never surfaced anywhere.
  if (!is.null(iread)) {
    attr(tab, "interval_read")        <- iread$read
    attr(tab, "interval_design_mass") <- iread$design_mass
    attr(tab, "within_cell")           <- iread$within_cell
    attr(tab, "within_cell_requested") <- iread$within_cell_requested
    attr(tab, "within_cell_declined")  <- iread$within_cell_declined
    attr(tab, "interval_read_note")   <- .tulpa_interval_read_note(iread)
  }
  if (!is.null(resolution)) {
    attr(tab, "grid_h_over_sd")      <- resolution$h_over_sd
    attr(tab, "grid_h_over_sd_max")  <- resolution$max
    attr(tab, "grid_coarsest_axis")  <- resolution$coarsest
    attr(tab, "grid_resolved")       <- resolution$resolved
    attr(tab, "grid_resolution_note") <- .tulpa_grid_resolution_note(resolution)
  }
  if (!is.null(inner_k)) {
    attr(tab, "inner_pareto_k")        <- inner_k$max_pareto_k
    attr(tab, "inner_pareto_k_band")   <- inner_k$band
    attr(tab, "inner_pareto_k_is_ess")  <- inner_k$min_is_ess
    attr(tab, "inner_pareto_k_rel_ess") <- inner_k$min_rel_ess
    attr(tab, "inner_pareto_k_scored") <- inner_k$n_scored
    attr(tab, "inner_pareto_k_probed") <- inner_k$n_probed
    attr(tab, "inner_pareto_k_uniform") <- inner_k$weights_uniform
    if (!is.na(inner_k$declined)) {
      attr(tab, "inner_pareto_k_declined")      <- inner_k$declined
      attr(tab, "inner_pareto_k_declined_note") <-
        .inner_k_decline_note(inner_k$declined)
    }
  }
  inner_declined <- if (is.null(inner)) NA_character_ else inner$declined
  inner_k_declined <- if (is.null(inner_k)) NA_character_ else inner_k$declined
  reliability <- .tulpa_combined_reliability(outer_band, inner_band,
                                            inner_declined,
                                            psis$pareto_k_declined,
                                            inner_k_band,
                                            inner_k_declined)
  attr(tab, "reliability") <- reliability

  summary_row <- data.frame(
    pareto_k        = k,
    pareto_k_band   = outer_band,
    outer_regime    = if (is.null(regime)) NA_character_ else regime$regime,
    outer_skew_max  = if (is.null(regime)) NA_real_ else regime$outer_skew_max,
    ess_grid        = if (is.null(grid)) NA_real_ else grid$ess_grid,
    n_grid          = if (is.null(grid)) NA_integer_ else grid$n_grid,
    max_weight      = if (is.null(grid)) NA_real_ else grid$max_weight,
    inner_skew_max  = if (is.null(inner)) NA_real_ else inner$max_abs_gamma3,
    inner_skew_band = if (is.null(inner)) NA_character_ else inner$band,
    inner_pareto_k      = if (is.null(inner_k)) NA_real_ else inner_k$max_pareto_k,
    inner_pareto_k_band = inner_k_band,
    pareto_k_declined   = psis$pareto_k_declined,
    inner_skew_declined = if (is.null(inner)) NA_character_ else inner$declined,
    inner_pareto_k_declined = inner_k_declined,
    reliability     = reliability,
    n_draws         = if (is.null(draws)) NA_integer_ else nrow(as.matrix(draws)),
    stringsAsFactors = FALSE, row.names = NULL
  )
  if (!is.na(param_declined)) {
    attr(tab, "param_table_declined") <- param_declined
  }
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
#' `outer_regime` qualifies what a high `pareto_k` means, and is the reason a
#' bare threshold on `pareto_k` is not a reliability verdict. A sharp
#' hyperparameter posterior collapses the grid onto ~1 cell (`ess_grid` near
#' 1); the outer integration has then degenerated to a point evaluation at the
#' modal hyperparameter, so `pareto_k` is scoring how well a Gaussian at that
#' mode stands in for the hyperparameter marginal, not how well a grid
#' integrated it. Where the dominant cell is INTERIOR to the grid the collapse
#' is benign -- the grid bracketed the mode, the estimate is empirical Bayes
#' there, and only integrated hyperparameter uncertainty is missing. Where it
#' sits at a grid BOUNDARY the grid may simply be too narrow: `grid_edge_axes`
#' / `grid_edge_sides` name the axes to widen. On a fit whose `pareto_k`
#' cleared the good band, the outer diagnostic also fits a skew-normal proposal
#' and reports the marginal's estimated skewness as `outer_skew_max`, so an
#' inflated k-hat that was purely the symmetric proposal's mismatch with a
#' skewed variance-component marginal is both corrected and explained. A
#' skew-normal has Gaussian tails, so this can never mask a genuinely
#' heavy-tailed target.
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
#' governed by the healthy inner layer, were fine -- the
#' `reliability` attribute is the combined verdict that names which layer
#' degrades, if either does.
#'
#' Each parameter row also carries the rank-normalized split-Rhat and bulk /
#' tail effective sample size of the draws (Vehtari et al. 2021). On i.i.d.
#' draws these sit at `~1.00` and `~n_draws` by construction; they are reported,
#' clearly as i.i.d.-draw Monte-Carlo diagnostics and not chain mixing, to
#' document that the reported posterior summaries are not Monte-Carlo-limited.
#'
#' A posterior sample is what those per-parameter rows are computed from, and
#' nothing else here needs one: every reliability quantity above is read off the
#' fit. So a fit that carries no draws -- a default single-block nested-Laplace
#' fit, whose posterior is the retained outer-grid mixture rather than a sample
#' -- reports the full band with an empty per-parameter body and `n_draws = NA`,
#' and records why in the `param_table_declined` attribute.
#' [tulpa_posterior_draws()] samples that mixture where the rows are wanted.
#' The one case that still returns `NULL` is a fit with neither draws nor any
#' reliability quantity, such as a plain Laplace fit with no outer grid to
#' score.
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
#' The inner layer carries a SECOND score, `inner_pareto_k`, which needs no
#' likelihood derivative at all and therefore answers where `inner_skew`
#' declines. The inner Gaussian at the fitted hyperparameter is an importance
#' proposal for the exact conditional posterior, and the joint density is the
#' target, so PSIS on that ratio scores the inner approximation directly. It is
#' computed on the same probed indices along the same conditional-mean curve,
#' one dimension per index, since importance sampling degrades with dimension
#' and a k-hat over the whole latent field would report `n_x` rather than the
#' approximation. A Pareto shape index is scale-free, so it is banded only on
#' indices whose realized importance efficiency shows a correction worth
#' describing; `inner_pareto_k_uniform` records that none did, which is what a
#' well-approximated inner layer looks like.
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
#'     \item{`pareto_k_declined`, `pareto_k_declined_note`}{when `pareto_k` is
#'       `NA`, WHY: `"not_requested"`, `"not_applicable"`,
#'       `"unguessable_axis"` (naming the axis -- a permanent limitation of that
#'       family, so read `ess_grid` instead), `"draws_too_few"`,
#'       `"grid_too_small"`, `"no_varying_axis"`, `"degenerate_proposal"`, or
#'       `"internal_inconsistency"` (an engine bug worth reporting), plus a
#'       one-line reading of it.}
#'     \item{`pareto_k_is_ess`}{importance-sampling ESS on the smoothed weights.}
#'     \item{`ess_grid`, `n_grid`, `rel_ess_grid`, `max_weight`}{grid quadrature
#'       reliability.}
#'     \item{`outer_regime`}{`"spread"` / `"collapsed_interior"` /
#'       `"collapsed_edge"` -- whether the outer grid integrated hyperparameter
#'       uncertainty at all, and if not whether its dominant cell is interior
#'       (empirical Bayes at the mode: point estimates sound, hyperparameter
#'       uncertainty not integrated) or against a grid boundary (widen it).}
#'     \item{`grid_edge_axes`, `grid_edge_sides`}{for an edge collapse, the axes
#'       the dominant cell sits against and on which side.}
#'     \item{`outer_skew_max`}{largest estimated |skewness| of the hyperparameter
#'       marginal, computed only when the k-hat triggered the skew-normal
#'       proposal rescue (`NA` means the Gaussian proposal already fit, not
#'       "symmetric and unchecked").}
#'     \item{`outer_regime_note`}{a one-line reading of a collapsed regime, or
#'       absent on a spread grid.}
#'     \item{`grid_railed_axes`}{outer axes whose OWN marginal is maximal at one
#'       of their own endpoints, as `axis:side` -- the span does not contain that
#'       axis's posterior mode, so its marginal is a truncated tail at any
#'       spacing. Reported whether or not the engine was allowed to, able to, or
#'       built to move the axis.}
#'     \item{`grid_placement`, `grid_recentred_axes`,
#'       `grid_placement_declined`, `grid_placement_note`}{whether the outer grid
#'       was re-centred, on which axes, and -- when it was not -- why.}
#'     \item{`scope`}{the outer diagnostic's scope string.}
#'     \item{`inner_skew_max`}{the largest `|gamma_3|` among the scored latent
#'       indices (`NA` if `control$diagnose_skew = FALSE` or nothing scored).}
#'     \item{`inner_skew_band`}{`"good"` / `"ok"` / `"unreliable"` / `NA`, banded
#'       on `inner_skew_max` by the general skewness-magnitude convention
#'       (Bulmer 1979) -- not a Rue-Martino-Chopin-specific cutoff.}
#'     \item{`inner_skew_scored`, `inner_skew_probed`}{how many of the probed
#'       latent indices returned a finite `gamma_3` vs how many were probed.}
#'     \item{`inner_skew_declined`, `inner_skew_arms_declined`,
#'       `inner_skew_declined_note`}{when nothing was scored, WHY:
#'       `"coupled_arm"` (STRUCTURAL -- the coupled arms have neither a
#'       per-observation sum nor a cell third-derivative tensor to read, so the
#'       outer k-hat is the only reliability number this fit has),
#'       `"curvature3_unavailable"`, `"no_finite_contribution"`,
#'       `"no_probe_indices"`, `"not_requested"`, `"backend_unsupported"`,
#'       `"solve_failed"`, or `"not_converged"` (the probe re-solve stopped
#'       short of a mode, so neither inner score has a point to read); the arms
#'       (1-based) a joint fit had no oracle for, which is also set on a
#'       PARTIALLY scored fit; and a one-line reading.} \item{`inner_pareto_k`,
#'       `inner_pareto_k_band`}{the inner-Laplace importance k-hat over the
#'       probed subspace, and its band on the same convention as the outer
#'       k-hat. Available wherever a mode was found, including a fit `gamma_3`
#'       cannot score.} \item{`inner_pareto_k_rel_ess`,
#'       `inner_pareto_k_is_ess`}{the smallest realized importance efficiency
#'       and effective sample size across the probed indices -- how much
#'       correcting the inner Gaussian actually needs, which is what makes the
#'       scale-free shape above readable.}
#'       \item{`inner_pareto_k_uniform`}{`TRUE` when no probed index carried a
#'       material correction, i.e. the inner Gaussian reproduces the
#'       conditional posterior over the sampled region.}
#'       \item{`inner_pareto_k_scored`, `inner_pareto_k_probed`}{how many
#'       probed indices returned a finite k-hat vs how many were probed.}
#'       \item{`inner_pareto_k_declined`, `inner_pareto_k_declined_note`}{when
#'       it is `NA`, WHY, from the same closed vocabulary the outer k-hat
#'       uses.} \item{`reliability`}{the combined whole-fit verdict:
#'       `"reliable"` only when both layers are good; otherwise names which
#'       layer is scoped or flags both as unreliable. The inner layer enters
#'       through the worse of its two scores, so a fit whose cubic term
#'       declined is still assessed.} } and a trailing `summary` attribute (a
#'       one-row data frame of the headline numbers) for printing.
#'
#' @references
#' Vehtari, Simpson, Gelman, Yao & Gabry (2024). Pareto smoothed
#'   importance sampling. \emph{JMLR} 25(72):1-58.
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
#' \donttest{
#' set.seed(1)
#' n <- 200L; x <- rnorm(n)
#' y <- rbinom(n, 1, plogis(-0.2 + 0.6 * x))
#' # `mode = "laplace"` returns a mode + covariance and carries no draws; a
#' # sampled deterministic backend is what this table describes.
#' fit <- tulpa(y ~ x, data.frame(y = y, x = x), family = "binomial",
#'              mode = "smc")
#' diagnostics(fit)
#' }
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
  inner_note <- attr(x, "inner_skew_declined_note")
  has_skew <- !is.null(attr(x, "inner_skew_band")) &&
    is.finite(attr(x, "inner_skew_max") %||% NA_real_)
  has_inner_k <- !is.null(attr(x, "inner_pareto_k_band")) &&
    is.finite(attr(x, "inner_pareto_k") %||% NA_real_)
  has_inner <- has_skew || has_inner_k
  if (!has_skew && has_inner_k) {
    # The cubic term declined but the importance k-hat did not, so the inner
    # layer IS assessed -- say which score carries it rather than reprinting the
    # cubic term's decline as though nothing were known.
    inner_note <- NULL
  }
  if (has_inner) {
    cat("Nested-Laplace WHOLE-FIT reliability (i.i.d. draws)\n")
    cat("  two layers: the outer hyperparameter-grid integration, and the",
        "inner Gaussian Laplace on the latent field\n")
  } else {
    # The inner layer is unscored -- say WHY. Attributing a
    # structural impossibility to `control$diagnose_skew` sent readers looking
    # for a knob they never touched.
    cat("Nested-Laplace OUTER-integration reliability (i.i.d. draws)\n")
    cat("  scope: the outer hyperparameter-grid integration; the latent-field\n",
        "  Laplace is a separate, unscored layer",
        if (!is.null(inner_note)) paste0(":\n    ", inner_note, "\n") else ".\n",
        sep = "")
  }
  if (is.finite(k)) {
    cat(sprintf("  outer PSIS pareto_k = %.3f (%s); IS-ESS = %.1f\n",
                k, band, attr(x, "pareto_k_is_ess")))
  } else {
    # Every decline path says which one it was instead of
    # the old "not run or proposal degenerate" disjunction.
    knote <- attr(x, "pareto_k_declined_note")
    cat("  outer PSIS pareto_k = NA",
        if (!is.null(knote)) paste0(":\n    ", knote, "\n")
        else " (no reason recorded by this backend)\n",
        sep = "")
  }
  if (!is.null(attr(x, "ess_grid"))) {
    cat(sprintf("  outer grid quadrature ESS = %.2f of %d cells (max weight %.3f)\n",
                attr(x, "ess_grid"), attr(x, "n_grid"), attr(x, "max_weight")))
  }
  osk <- attr(x, "outer_skew_max")
  if (!is.null(osk) && is.finite(osk)) {
    cat(sprintf("  outer hyperparameter marginal max |skewness| = %.2f\n", osk))
  }
  note <- attr(x, "outer_regime_note")
  if (!is.null(note)) cat("  note: ", note, "\n", sep = "")
  pnote <- attr(x, "grid_placement_note")
  if (!is.null(pnote)) cat("  note: ", pnote, "\n", sep = "")
  for (rnote in c(attr(x, "interval_read_note"),
                  attr(x, "grid_resolution_note"))) {
    cat("  note: ", rnote, "\n", sep = "")
  }
  if (has_skew) {
    ib <- attr(x, "inner_skew_band")
    im <- attr(x, "inner_skew_max")
    cat(sprintf("  inner Laplace max |gamma_3| = %.3f (%s), scored %d/%d latents\n",
                im, ib, attr(x, "inner_skew_scored"), attr(x, "inner_skew_probed")))
    # A partly-scored joint fit names the arms that had no oracle at all.
    arms <- attr(x, "inner_skew_arms_declined")
    if (!is.null(arms) && length(arms)) {
      cat(sprintf("  inner Laplace unscored on arm(s) %s\n",
                  paste(arms, collapse = ", ")))
    }
  } else if (has_inner_k && !is.null(attr(x, "inner_skew_declined_note"))) {
    cat("  inner Laplace |gamma_3| = NA:\n    ",
        attr(x, "inner_skew_declined_note"), "\n", sep = "")
  }
  if (has_inner_k) {
    cat(sprintf(
      paste0("  inner Laplace importance pareto_k = %.3f (%s), min IS efficiency",
             " %.4f, scored %d/%d latents\n"),
      attr(x, "inner_pareto_k"), attr(x, "inner_pareto_k_band"),
      attr(x, "inner_pareto_k_rel_ess") %||% NA_real_,
      attr(x, "inner_pareto_k_scored"), attr(x, "inner_pareto_k_probed")))
    if (isTRUE(attr(x, "inner_pareto_k_uniform"))) {
      cat("    the importance weights are uniform on every probed index: the",
          "inner\n    Gaussian reproduces the conditional posterior over the",
          "sampled region,\n    so the shape above describes no correction and",
          "is not banded\n")
    }
  } else {
    knote <- attr(x, "inner_pareto_k_declined_note")
    if (!is.null(knote)) {
      cat("  inner Laplace importance pareto_k = NA:\n    ", knote, "\n", sep = "")
    }
  }
  if (!is.null(attr(x, "reliability"))) {
    cat(sprintf("  whole-fit verdict: %s\n", attr(x, "reliability")))
  }
  # The reliability band above scores THIS fit's internal approximation; the SBC
  # result scores whether the backend's posterior is calibrated over many
  # simulated data sets. They disagree in both directions, so the two are
  # printed together rather than one standing in for the other.
  sbc_rep <- attr(x, "sbc_report")
  if (!is.null(sbc_rep)) {
    cat(sprintf("  calibration (SBC, %s): %s\n",
                attr(x, "sbc_experiment"), attr(x, "sbc_verdict")))
  }
  pdecl <- attr(x, "param_table_declined")
  if (!is.null(pdecl)) {
    cat("  per-parameter columns: none.\n    ", pdecl, "\n", sep = "")
    return(invisible(x))
  }
  cat(sprintf("  %d parameters, %d draws; per-parameter rhat / ESS below are\n",
              nrow(x), if (is.null(s)) NA_integer_ else s$n_draws))
  cat("  i.i.d.-draw Monte-Carlo diagnostics (not chain mixing).\n\n")
  print(as.data.frame(x), ...)
  invisible(x)
}
