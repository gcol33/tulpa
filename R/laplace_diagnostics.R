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
  if (ag < .nl_diag("gamma3_ok")) return("good")
  if (ag < .nl_diag("gamma3_unreliable")) return("ok")
  "unreliable"
}

# --- inner-skew decline reasons (gcol33/tulpa#296) ---------------------------
#
# `gamma_3` is careful never to return a silently-wrong 0 for a non-computable
# skewness -- every decline is NaN (#272). But NaN says only "not computable",
# and the reasons are not interchangeable: a coupled multi-process likelihood
# (a ZI mixture, tulpaObs's `occu_cover`) can NEVER be scored by this formula,
# while a failed finite difference is specific to one fit and a disabled knob is
# not a problem at all. Without the reason a structurally unscorable model
# printed as `control$diagnose_skew = FALSE`, attributing an impossibility to a
# setting the user likely left at its default `TRUE`.
#
# `res$inner_skew_declined` carries it, from this closed vocabulary (the first
# five come from C++ -- see `InnerSkewOutcome::declined` and
# `build_spec_curvature3_fn` -- the rest are decided R-side):
#
#   coupled_likelihood     a LikelihoodSpec with n_processes != 1. STRUCTURAL:
#                          this model class will never be scorable.
#   coupled_arm            a joint fit whose scorable arms all declined because
#                          the coupling spec excluded them. Also structural.
#                          `inner_skew_arms_declined` names them (1-based), so a
#                          PARTIALLY scored joint fit is visible too.
#   curvature3_unavailable no registered third derivative / no `eta_weights_fn`
#                          to finite-difference.
#   no_finite_contribution an oracle existed but nothing finite reached any
#                          probed index (an unidentified or degenerate mode).
#   no_probe_indices       nothing was asked for (`p_fixed = 0` and no
#                          `control$skew_idx`).
#   not_requested          `control$diagnose_skew = FALSE`.
#   backend_unsupported    this backend does not populate gamma_3.
#   solve_failed           the probe re-solve errored or returned no field.
#
# Structural reasons are the ones a reader must act on differently: for those
# models the outer k-hat is the only reliability number available, permanently.
.INNER_SKEW_STRUCTURAL <- c("coupled_likelihood", "coupled_arm")

# =============================================================================
# Inner-Laplace importance k-hat (gcol33/tulpa#303) -- the second score on the
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
    solve_failed        = list("degenerate_proposal", "the probe re-solve failed")
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
    # The shared core stashes its log-ratios when `tulpa.kdiag.capture` holds an
    # environment. That aperture belongs to the OUTER diagnostic's validation
    # harness, so the inner pass runs with it closed rather than overwriting
    # what the outer pass put there.
    cap <- getOption("tulpa.kdiag.capture", NULL)
    if (!is.null(cap)) {
        options(tulpa.kdiag.capture = NULL)
        on.exit(options(tulpa.kdiag.capture = cap), add = TRUE)
    }
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
    res$inner_skew               <- NULL
    res$inner_skew_idx           <- integer(0)
    res$inner_skew_dropped       <- 0L
    res$inner_skew_declined      <- reason
    res$inner_skew_arms_declined <- integer(0)
    .inner_k_decline_from_skew(res, reason)
}

# Copy the kernel's inner-skew output (including its own decline reason) onto a
# fit. One attach point for all three drivers.
.inner_skew_attach <- function(res, out) {
    res$inner_skew         <- as.numeric(out$inner_skew)
    res$inner_skew_idx     <- as.integer(out$inner_skew_idx)
    res$inner_skew_dropped <- as.integer(out$inner_skew_dropped %||% 0L)
    d <- as.character(out$inner_skew_declined %||% "")
    res$inner_skew_declined <- if (!length(d) || !nzchar(d[1L])) NA_character_
                               else d[1L]
    arms <- out$inner_skew_arms_declined
    res$inner_skew_arms_declined <- if (is.null(arms)) integer(0) else as.integer(arms)
    .inner_k_attach(res, out)
}

# One-line user-facing reading of an inner-skew decline, for `print` /
# `diagnostic_summary()`. `arms` are the 1-based joint arms with no oracle.
.inner_skew_decline_note <- function(reason, arms = integer(0)) {
    if (is.null(reason) || !length(reason) || is.na(reason)) return(NULL)
    arm_txt <- if (length(arms)) paste0(" (arms ", paste(arms, collapse = ", "), ")") else ""
    switch(reason,
        not_requested = "the inner diagnostic was not requested (control$diagnose_skew = FALSE)",
        no_probe_indices = "no latent indices were probed; pass control$skew_idx",
        coupled_likelihood = paste("this likelihood couples several processes, so it has no",
                                   "single per-observation term gamma_3 can score; the inner",
                                   "layer is unscorable for this model class, permanently"),
        coupled_arm = paste0("every scorable arm is coupled through the cell-coupling spec",
                             arm_txt, ", so gamma_3 has no separable per-observation sum to",
                             " read; unscorable for this model class, permanently"),
        curvature3_unavailable = paste("this likelihood registers no third derivative and",
                                       "exposes no eta-weight callback to finite-difference"),
        no_finite_contribution = paste("an oracle was available but no probed index",
                                       "accumulated a finite contribution"),
        no_oracle = "no per-observation third-derivative oracle was available",
        backend_unsupported = "this backend does not compute the inner-Laplace skewness",
        solve_failed = "the probe re-solve failed",
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
# broken" when the inner layer is fine, or vice versa -- the framing #272
# was filed to fix (42/78 occu_cover species read as "broken" on outer k-hat
# alone when their point estimates, governed by the inner layer, were fine).
#
# A layer can also be "na" -- never assessed (gamma_3 not wired for a coupled
# likelihood; a multi-block/multi-axis outer grid that declines to a guessed
# support transform) -- which must never collapse into the SAME string as
# "assessed and good", or a batch consumer reading the verdict off many fits
# cannot tell "outer bad, inner genuinely fine" from "outer bad, inner never
# checked" (gcol33/tulpa#274). Every combination naming an "na" layer says so
# explicitly ("... not assessed"), so `grepl("not assessed", reliability)`
# reliably separates the two.
# An "na" layer is further split by WHY it was not assessed (gcol33/tulpa#295,
# #296): a layer that CANNOT be assessed for this model or family -- a coupled
# multi-process likelihood has no per-observation term gamma_3 can score, and a
# car_proper `rho_car` axis has no support the outer k-hat may guess -- will
# never become assessable, so the verdict says so rather than implying a rerun
# with the right knob would fill it in. The `"not assessed"` wording is kept in
# every such verdict (a documented `grepl("not assessed", ...)` contract from
# gcol33/tulpa#274), with the permanence as a qualifier on top.
#
# The INNER layer carries two scores, not one (gcol33/tulpa#303): the cubic
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
  # fit instead of skipping the probe (gcol33/tulpa#298).
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

  if (is.null(out) || is.null(out$inner_skew)) {
    return(.inner_skew_decline(res, "backend_unsupported"))
  }
  .inner_skew_attach(res, out)
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
  # WHY it is NA, when it is (gcol33/tulpa#295). A backend with no outer
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

# Outer-integration REGIME of a nested-Laplace fit (gcol33/tulpa#276), read
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

# Inner-Laplace skewness reliability of a nested-Laplace fit, read from the
# `inner_skew` / `inner_skew_idx` / `inner_skew_dropped` fields
# .nl_inner_skew_at_theta() attaches at fit time (see nested_laplace.R). NULL
# when the fit never ran the diagnostic (control$diagnose_skew = FALSE, or a
# backend that does not yet populate it).
.tulpa_inner_skew_reliability <- function(fit) {
  jf <- if (!is.null(fit$joint_fit)) fit$joint_fit else fit
  s <- .tulpa_inner_skew_summary(jf$inner_skew, jf$inner_skew_dropped %||% 0L)
  reason <- jf$inner_skew_declined %||% NA_character_
  arms   <- as.integer(jf$inner_skew_arms_declined %||% integer(0))
  # A fit that computed NOTHING still reports WHY (gcol33/tulpa#296), so an
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
  inner_k <- .tulpa_inner_k_reliability(fit)
  regime <- .tulpa_outer_regime(fit)
  k          <- psis$pareto_k
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
#' `outer_regime` qualifies what a high `pareto_k` means, and is the reason a
#' bare threshold on `pareto_k` is not a reliability verdict
#' (gcol33/tulpa#276). A sharp hyperparameter posterior collapses the grid onto
#' ~1 cell (`ess_grid` near 1); the outer integration has then degenerated to a
#' point evaluation at the modal hyperparameter, so `pareto_k` is scoring how
#' well a Gaussian at that mode stands in for the hyperparameter marginal, not
#' how well a grid integrated it. Where the dominant cell is INTERIOR to the
#' grid the collapse is benign -- the grid bracketed the mode, the estimate is
#' empirical Bayes there, and only integrated hyperparameter uncertainty is
#' missing. Where it sits at a grid BOUNDARY the grid may simply be too narrow:
#' `grid_edge_axes` / `grid_edge_sides` name the axes to widen. On a fit whose
#' `pareto_k` cleared the good band, the outer diagnostic also fits a
#' skew-normal proposal and reports the marginal's estimated skewness as
#' `outer_skew_max`, so an inflated k-hat that was purely the symmetric
#' proposal's mismatch with a skewed variance-component marginal is both
#' corrected and explained. A skew-normal has Gaussian tails, so this can never
#' mask a genuinely heavy-tailed target.
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
#'       `NA`, WHY (gcol33/tulpa#295): `"not_requested"`, `"not_applicable"`,
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
#'       (benign) or against a grid boundary (widen it).}
#'     \item{`grid_edge_axes`, `grid_edge_sides`}{for an edge collapse, the axes
#'       the dominant cell sits against and on which side.}
#'     \item{`outer_skew_max`}{largest estimated |skewness| of the hyperparameter
#'       marginal, computed only when the k-hat triggered the skew-normal
#'       proposal rescue (`NA` means the Gaussian proposal already fit, not
#'       "symmetric and unchecked").}
#'     \item{`outer_regime_note`}{a one-line reading of a collapsed regime, or
#'       absent on a spread grid.}
#'     \item{`scope`}{the outer diagnostic's scope string.}
#'     \item{`inner_skew_max`}{the largest `|gamma_3|` among the scored latent
#'       indices (`NA` if `control$diagnose_skew = FALSE` or nothing scored).}
#'     \item{`inner_skew_band`}{`"good"` / `"ok"` / `"unreliable"` / `NA`, banded
#'       on `inner_skew_max` by the general skewness-magnitude convention
#'       (Bulmer 1979) -- not a Rue-Martino-Chopin-specific cutoff.}
#'     \item{`inner_skew_scored`, `inner_skew_probed`}{how many of the probed
#'       latent indices returned a finite `gamma_3` vs how many were probed.}
#'     \item{`inner_skew_declined`, `inner_skew_arms_declined`,
#'       `inner_skew_declined_note`}{when nothing was scored, WHY
#'       (gcol33/tulpa#296): `"coupled_likelihood"` / `"coupled_arm"` (both
#'       STRUCTURAL -- the formula has no per-observation term to read for this
#'       model class, so the outer k-hat is the only reliability number
#'       available, permanently), `"curvature3_unavailable"`,
#'       `"no_finite_contribution"`, `"no_probe_indices"`, `"not_requested"`,
#'       `"backend_unsupported"`, or `"solve_failed"`; the arms (1-based) a
#'       joint fit had no oracle for, which is also set on a PARTIALLY scored
#'       fit; and a one-line reading.}
#'     \item{`inner_pareto_k`, `inner_pareto_k_band`}{the inner-Laplace
#'       importance k-hat over the probed subspace, and its band on the same
#'       convention as the outer k-hat. Available wherever a mode was found,
#'       including a coupled likelihood `gamma_3` cannot score.}
#'     \item{`inner_pareto_k_rel_ess`, `inner_pareto_k_is_ess`}{the smallest
#'       realized importance efficiency and effective sample size across the
#'       probed indices -- how much correcting the inner Gaussian actually
#'       needs, which is what makes the scale-free shape above readable.}
#'     \item{`inner_pareto_k_uniform`}{`TRUE` when no probed index carried a
#'       material correction, i.e. the inner Gaussian reproduces the conditional
#'       posterior over the sampled region.}
#'     \item{`inner_pareto_k_scored`, `inner_pareto_k_probed`}{how many probed
#'       indices returned a finite k-hat vs how many were probed.}
#'     \item{`inner_pareto_k_declined`, `inner_pareto_k_declined_note`}{when it
#'       is `NA`, WHY, from the same closed vocabulary the outer k-hat uses.}
#'     \item{`reliability`}{the combined whole-fit verdict: `"reliable"` only
#'       when both layers are good; otherwise names which layer is scoped or
#'       flags both as unreliable. The inner layer enters through the worse of
#'       its two scores, so a fit whose cubic term declined is still assessed.}
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
    # The inner layer is unscored -- say WHY (gcol33/tulpa#296). Attributing a
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
    # Every decline path says which one it was (gcol33/tulpa#295) instead of
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
  cat(sprintf("  %d parameters, %d draws; per-parameter rhat / ESS below are\n",
              nrow(x), if (is.null(s)) NA_integer_ else s$n_draws))
  cat("  i.i.d.-draw Monte-Carlo diagnostics (not chain mixing).\n\n")
  print(as.data.frame(x), ...)
  invisible(x)
}
