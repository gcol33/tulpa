# Outer Pareto-k-hat: the proposal candidates, and the choice between them.
#
# Every backend that reports an outer k-hat is asking the same question -- how
# well does a tractable proposal cover the hyperparameter posterior the
# integrator actually weights -- and the answer depends on which proposal
# families are offered, not on which backend asked. Scoring a single Gaussian
# where four candidates were available reports the WORST of them: measured over
# 165 configurations, the raw grid-moment Gaussian reads `unreliable` on 53
# where the full dispatch reads it on 8, median k-hat 1.159 -> 0.736 -> 0.259
# across the three layers (gcol33/tulpa#630, dev_notes/issue630/RESULTS630.md).
# So the candidate layer lives here, on a backend-agnostic contract, and each
# backend supplies a spec rather than a scorer.
#
# `.k_cand_spec()` is that contract, and everything it carries is already
# restricted to the subspace being scored -- a backend that holds some axes
# fixed does the restriction when it builds the spec, so no scorer indexes an
# axis set:
#
#   lt(U)             the outer target, an S x p matrix of unconstrained
#                     coordinates to S unnormalized log-densities, in whatever
#                     coordinate the integrator weights (any change-of-variables
#                     Jacobian is the spec builder's job, so the diagnostic
#                     always scores the density the fit actually integrated)
#   u_hat, Su         the Gaussian proposal, p and p x p
#   u_grid, w         the integration nodes and their weights, K x p and K.
#                     NULL where the backend has no node set -- a mode-Hessian
#                     proposal, say -- which withholds the grid mixture and the
#                     radius cap rather than inventing either
#   proposal_source   where (u_hat, Su) came from; the grid mixture is offered
#                     only on "grid_moment", the one source whose covariance IS
#                     the node spread
#   names             axis labels for the reported `outer_skew`
#
# The four candidates, cheapest first, each gated on the previous one still
# missing the good band: the grid-moment Gaussian, that Gaussian after
# moment-matching refinement, the grid mixture, the skew-normal rescue. The
# minimum is kept, so adding a candidate can never make a fit read worse.

# Build the candidate contract. `Su` is coerced to a matrix so a one-axis spec
# behaves like any other, and `u_grid` to a matrix with the same column count,
# which is what lets every scorer below be written without a rank-1 special
# case. Returns NULL for an unusable proposal (no axis, or a shape mismatch),
# which `.k_dispatch()` turns into a decline.
.k_cand_spec <- function(lt, u_hat, Su, u_grid = NULL, w = NULL,
                         proposal_source = "grid_moment", names = NULL) {
    u_hat <- as.numeric(u_hat)
    p     <- length(u_hat)
    if (!p) return(NULL)
    Su <- matrix(as.numeric(Su), p, p)
    Su <- (Su + t(Su)) / 2
    if (!is.null(u_grid)) {
        u_grid <- matrix(as.numeric(u_grid), ncol = p)
        if (!nrow(u_grid) || is.null(w) || length(w) != nrow(u_grid)) {
            u_grid <- NULL
            w      <- NULL
        }
    } else {
        w <- NULL
    }
    list(lt = lt, u_hat = u_hat, Su = Su, u_grid = u_grid, w = w,
         proposal_source = proposal_source,
         names = if (is.null(names)) NULL else as.character(names))
}

# The full candidate dispatch plus the reporting fields every backend needs from
# it: the point k-hat, the IS-ESS, the raw finite log-ratios the bootstrap
# re-fits on, which proposal produced them, and the target's whitened skewness
# where the rescue estimated it. `scope` names what the ratios belong to for the
# `.kdiag_capture()` aperture, which is written ONCE here -- after the choice --
# from the SELECTED proposal, so an external recomputation reproduces the number
# the backend reports. Returns the `declined` label instead of a k-hat when no
# candidate could be scored.
.k_dispatch_report <- function(spec, n_samples, tail_points = NULL,
                               scope = NULL) {
    na_out <- function(x) list(pareto_k = NA_real_, is_ess = NA_real_,
                               n_eval = 0L, lr = NULL,
                               proposal_source = NA_character_,
                               first_pass_k = NA_real_, outer_skew = NULL,
                               declined = .k_reason_of(x))
    if (is.null(spec)) return(na_out(.k_decline("degenerate_proposal")))
    out <- .k_dispatch(spec, n_samples, tail_points = tail_points)
    if (.k_is_decline(out)) return(na_out(out))
    lr <- out$best$lr
    .kdiag_capture(lr, tail_points = tail_points,
                   scope = if (is.null(scope)) NULL
                           else paste0(scope, " (", out$source, ")"))
    list(pareto_k = out$best$pareto_k, is_ess = out$best$is_ess,
         n_eval = length(lr), lr = lr, proposal_source = out$source,
         first_pass_k = out$first_pass_k %||% NA_real_,
         outer_skew = out$outer_skew, declined = NA_character_)
}

# Score the outer Pareto-k over a chosen set of varying axes `vary`, holding the
# rest fixed at `u_hat`. The shared scorer behind both the joint k (all
# genuinely-varying axes) and the per-arm k (one arm's axes).
# `prep` is the `.joint_pareto_prepare` summary, `refit_log_marginal` the inner
# re-solve closure (round-trips an `S x d` user-facing theta matrix through the
# SAME joint kernel the integrator used, returning the per-cell log-marginal WITH
# the baked hyperprior). The proposal's normalizing constant is common to every
# draw and drops under PSIS, so only the quadratic enters (handled inside
# `.nested_is_pareto_k`). Does NOT manage the RNG -- the driver saves / restores
# it once around all scoring so the fit's draws are bit-for-bit unchanged.
#
# Importance-sampling k-hat with moment-matching refinement (after Paananen,
# Piironen, Burkner & Vehtari 2021, Stat. Comput. 31:16). The
# initial proposal is the integration-node covariance (or the mode-Hessian / CCD
# curvature). When the grid is sharply concentrated that covariance is estimated
# from few effective cells and can mis-scale the proposal -- too wide scatters
# draws to extreme hyperparameters where the inner Laplace log-marginal inflates,
# too narrow leaves the target tail uncovered -- so the k-hat reads unreliable
# even though the fit is fine. Re-estimate the proposal from the PSIS-smoothed
# importance-weighted moments of its own draws and re-score, keeping the
# lowest-k-hat proposal; the smoothed weights bound any single draw's influence,
# so a sharp posterior is matched in a couple of passes. Iteration stops once the
# k-hat reaches the usable band (<= 0.7); a proposal that already fits pays a
# single pass. The per-pass radius cap is recomputed for the
# current proposal so the grid-coverage envelope follows it. Returns
# list(pareto_k, is_ess, refined) or NULL (an empty or rank-deficient
# finite importance draw).
.k_score_gaussian <- function(spec, n_samples, tail_points = NULL) {
    Su_v <- spec$Su
    if (!nrow(Su_v)) return(NULL)
    L_v  <- tryCatch(t(chol(Su_v)), error = function(e) NULL)
    if (is.null(L_v)) return(NULL)
    u_hat_v <- spec$u_hat

    lt <- spec$lt

    u_grid_v <- spec$u_grid
    prop_u  <- u_hat_v
    prop_L  <- L_v
    best    <- NULL
    gm_full <- NULL                      # first-pass (grid-moment) result
    refined <- FALSE
    prev_k  <- Inf                       # k-hat of the proposal each pass refines from
    for (iter in seq_len(.K_DIAG_MM_MAX)) {
        rc <- if (is.null(u_grid_v)) Inf
              else .nested_grid_radius_cap(u_grid_v, prop_u, prop_L)
        kd <- tryCatch(.nested_is_pareto_k(prop_u, prop_L, lt, n_samples,
                                           radius_cap = rc, return_draws = TRUE,
                                           tail_points = tail_points),
                       error = function(e) NULL)
        if (is.null(kd) || !is.finite(kd$pareto_k)) break
        cand <- list(pareto_k = kd$pareto_k, is_ess = kd$is_ess, refined = refined,
                     U = kd$U, log_weights = kd$log_weights, lr = kd$lr,
                     prop_u = prop_u, prop_L = prop_L)
        # The first pass is the canonical grid-moment Gaussian -- its draws span the
        # actual posterior spread (not the moment-matching-widened proposal, which
        # scatters seed-dependently). The dispatcher reads it for the grid-coverage
        # check and the escaped-grid guard, so the decision is stable across RNG
        # seeds.
        if (iter == 1L) gm_full <- cand
        # `prop_u` / `prop_L` are the proposal that PRODUCED this `kd`, so the stored
        # proposal stays consistent with `best$pareto_k` and the bootstrap re-fits
        # the GPD on this same converged proposal's ratios.
        if (is.null(best) || kd$pareto_k < best$pareto_k) best <- cand
        if (kd$pareto_k <= .nl_diag("k_usable") || iter == .K_DIAG_MM_MAX) break
        # Stop refining once a pass no longer improves on the proposal it was
        # estimated from: moment matching has converged (or started to drift on the
        # seed-dependent widening), so further passes only spend budget without
        # lowering k. This keeps the generous .K_DIAG_MM_MAX a safe backstop -- a
        # stubborn k that is still falling keeps refining; a plateaued one stops.
        if (kd$pareto_k >= prev_k) break
        prev_k <- kd$pareto_k
        wts <- exp(kd$log_weights); sw <- sum(wts)
        if (!is.finite(sw) || sw <= 0 || nrow(kd$U) < length(prop_u) + 1L) break
        wts   <- wts / sw
        mu_w  <- as.numeric(crossprod(wts, kd$U))
        cen   <- sweep(kd$U, 2L, mu_w)
        Sig_w <- crossprod(cen * wts, cen); Sig_w <- (Sig_w + t(Sig_w)) / 2
        Lw    <- tryCatch(t(chol(Sig_w)), error = function(e) NULL)
        if (is.null(Lw)) break
        prop_u <- mu_w; prop_L <- Lw; refined <- TRUE
    }
    if (!is.null(best)) {
        best$gm    <- gm_full
        best$gm_U  <- if (!is.null(gm_full)) gm_full$U else NULL
        best$gm_lw <- if (!is.null(gm_full)) gm_full$log_weights else NULL
    }
    best
}

# Grid-mixture (basin) outer Pareto-k scorer. Scores the
# diagnostic against the proposal the engine ACTUALLY samples hyperparameters
# from: a defensive mixture q(u) = sum_k w_k N(u_k, diag(s^2)) of local Gaussian
# bumps at the integration-grid cells `u_k` (already on the spec's own
# subspace), mixed by the
# grid weights `w_k`. This covers a skewed or multi-node hyperparameter posterior
# by construction (the single grid-moment Gaussian cannot), so the importance
# weights stay bounded and the k-hat reflects the grid representation's fidelity
# rather than a Gaussian's mismatch to a skewed marginal. Per-axis bump SD is
# `bw * (largest adjacent grid gap on that axis)`, falling back to the
# grid-weighted SD where an axis keeps a single distinct value. Draws `n_samples`
# points -- the resolved `control$k_samples`, carried internally and reported as
# `diagnose_draws`: the mixture's tail-shape k-hat needs a long enough tail to be
# stable, so a tighter estimate takes a larger `control$k_samples`. The draws
# stay near the grid cells, so
# no inner Newton stalls. Returns
# list(pareto_k, is_ess, refined = FALSE, lr, s, lo, hi) -- `lr` the raw finite
# importance log-ratios (for the k bootstrap), `s` the per-axis bump SD and
# `lo` / `hi` the kept-cell node range, which the dispatcher uses to
# bound the mixture's coverage -- or NULL to fall back to the single-Gaussian
# scorer (fewer than two distinct weighted cells -- not actually spread -- or too
# few finite importance draws). Like .k_score_gaussian it does NOT manage the
# RNG (the driver saves / restores it once around all scoring).
.k_score_mixture <- function(spec, n_samples, tail_points = NULL) {
    if (!nrow(spec$Su)) return(NULL)
    w <- spec$w
    if (is.null(w) || is.null(spec$u_grid)) return(NULL)
    if (!any(is.finite(w)) || sum(w) <= 0) return(NULL)
    cu   <- spec$u_grid
    keep <- is.finite(w) & w >= .K_DIAG_MIX_FLOOR * max(w)
    cu   <- cu[keep, , drop = FALSE]
    wk   <- w[keep]
    if (nrow(cu) < 2L) return(NULL)

    # Collapse cells that coincide on the spec's axes (a pinned axis dropped here
    # can repeat a varying-axis point across its values): sum their weights so
    # each mixture component is a distinct varying-subspace cell.
    key <- apply(round(cu, 10L), 1L, paste, collapse = "\r")
    if (anyDuplicated(key)) {
        first <- !duplicated(key)
        wk    <- as.numeric(tapply(wk, factor(key, levels = key[first]), sum))
        cu    <- cu[first, , drop = FALSE]
    }
    K <- nrow(cu); p <- ncol(cu)
    if (K < 2L) return(NULL)
    wk <- wk / sum(wk)

    bw   <- as.numeric(getOption("tulpa.kdiag.mix_bw", .K_DIAG_MIX_BW))
    sdax <- sqrt(pmax(diag(spec$Su), 0))
    # Bump SD from the FULL grid's node spacing on each axis (the grid's actual
    # resolution), NOT the floor-surviving cells: when the posterior weight
    # concentrates on one node of an axis only that cell survives the floor, and a
    # gap from the kept cells alone would collapse to a degenerate near-zero width
    # (a spike the single-Gaussian's near-mode draws then read as "uncovered").
    full_v <- spec$u_grid
    gap  <- vapply(seq_len(p), function(j) {
        u <- sort(unique(full_v[, j])); if (length(u) < 2L) NA_real_ else max(diff(u))
    }, numeric(1))
    s   <- bw * gap
    bad <- !is.finite(s) | s <= 0
    s[bad] <- (bw * sdax)[bad]
    s[!is.finite(s) | s <= 0] <- 1e-3

    n_mix <- as.integer(n_samples)
    lt   <- spec$lt
    comp <- sample.int(K, n_mix, replace = TRUE, prob = wk)
    U_v  <- cu[comp, , drop = FALSE] +
            sweep(matrix(stats::rnorm(n_mix * p), n_mix, p), 2L, s, `*`)

    # log q(u) up to the per-component-common diagonal normalizer (the SAME `s`
    # for every component, so it cancels under PSIS): logsumexp over components of
    # log w_k - 0.5 sum_j ((u_j - u_kj) / s_j)^2.
    tcu  <- t(cu)
    logq <- vapply(seq_len(n_mix), function(i) {
        q <- log(wk) - 0.5 * colSums(((tcu - U_v[i, ]) / s)^2)
        m <- max(q); if (!is.finite(m)) return(-Inf)
        m + log(sum(exp(q - m)))
    }, numeric(1))

    ltv <- lt(U_v)
    if (length(ltv) != n_mix) return(NULL)
    lr  <- ltv - logq
    fin <- is.finite(lr)
    if (sum(fin) < .PSIS_MIN_EVAL) return(NULL)
    ps <- tulpa_psis(lr[fin], tail_points = tail_points)
    if (!is.finite(ps$pareto_k)) return(NULL)
    list(pareto_k = ps$pareto_k, is_ess = ps$is_ess, refined = FALSE,
         lr = lr[fin], s = s, lo = apply(cu, 2L, min), hi = apply(cu, 2L, max))
}

# Fraction of the single-Gaussian importance weight that falls OUTSIDE the
# mixture's coverage hull. `U` are the proposal's importance
# draws on the varying subspace, `log_weights` their PSIS-smoothed (normalized)
# log weights, `lo` / `hi` the per-axis coverage bounds (the kept-cell node range
# expanded by a few bump SDs -- the mixture's actual reach). A draw is outside if
# any axis falls beyond [lo, hi]. The WEIGHT (not count) outside measures how much
# target mass sits beyond what the mixture covers: small => the grid covers the
# posterior (a high single-Gaussian k-hat is then a within-grid shape mismatch the
# mixture can fix); large => the grid is too narrow and the single Gaussian's
# k-hat must stand. Returns NA when draws / weights are unavailable.
.k_hull_weight <- function(U, log_weights, lo, hi) {
    if (is.null(U) || is.null(log_weights) || !is.matrix(U) || nrow(U) == 0L ||
        length(log_weights) != nrow(U)) return(NA_real_)
    outside <- logical(nrow(U))
    for (j in seq_len(ncol(U))) outside <- outside | (U[, j] < lo[j]) | (U[, j] > hi[j])
    w <- exp(log_weights - max(log_weights))
    sw <- sum(w)
    if (!is.finite(sw) || sw <= 0) return(NA_real_)
    sum(w[outside]) / sw
}

# Per-axis centred moments of the outer target in the WHITENED coordinate of the
# Gaussian proposal that produced the draws.
#
# `U` are the proposal's importance draws and `log_weights` their PSIS-smoothed
# log weights, so the weighted moments estimate the TARGET's moments (importance
# correction), not the proposal's. Whitening by the proposal's own
# `(u_c, L_v)` decorrelates first, so a per-axis skewness is a statement about
# the target's shape along the directions the proposal already aligned to,
# rather than about an arbitrary raw parameterization -- which is what lets the
# skew proposal be an independent PRODUCT of univariate skew-normals.
#
# PSIS smoothing is what makes the third moment usable: it caps any single
# draw's weight, so the estimate cannot be set by one runaway ratio. This is the
# same weighted-moment machinery `.k_score_gaussian`'s moment-matching loop
# already applies to the first two moments, extended by one order -- and it
# costs NO extra inner solves, unlike differencing the target at the mode.
#
# Returns list(mu, sd, skew) in whitened coordinates (one per spec axis), or
# NULL when the weights are degenerate.
.k_wtd_moments <- function(U, log_weights, u_c, L_v) {
    if (is.null(U) || !is.matrix(U) || nrow(U) < 3L ||
        is.null(log_weights) || length(log_weights) != nrow(U)) return(NULL)
    Z <- tryCatch(t(forwardsolve(L_v, t(sweep(U, 2L, u_c)))),
                  error = function(e) NULL)
    if (is.null(Z) || any(!is.finite(Z))) return(NULL)
    w <- exp(log_weights - max(log_weights))
    sw <- sum(w)
    if (!is.finite(sw) || sw <= 0) return(NULL)
    w  <- w / sw
    mu <- as.numeric(crossprod(w, Z))
    cen <- sweep(Z, 2L, mu)
    v  <- as.numeric(crossprod(w, cen^2))
    if (any(!is.finite(v)) || any(v <= 0)) return(NULL)
    sd <- sqrt(v)
    g  <- as.numeric(crossprod(w, cen^3)) / sd^3
    if (any(!is.finite(mu)) || any(!is.finite(g))) return(NULL)
    # Effective sample size of the importance weights: what the third moment is
    # actually estimated from, and hence the n its standard error is read at.
    list(mu = mu, sd = sd, skew = g, n_eff = 1 / sum(w^2))
}

# Score the outer Pareto-k against a SKEW-NORMAL proposal:
# the product of one univariate skew-normal per spec axis, in the whitened
# coordinate of
# the Gaussian proposal `(u_c, L_v)`, each matched to `mom`'s corresponding
# whitened mean / sd / skewness. Draws are mapped back as
# `U = u_c + L_v z`; the whitening Jacobian |det L_v| is common to every draw
# and drops under PSIS self-normalization, so the proposal density is evaluated
# in `z` directly.
#
# Because a skew-normal has GAUSSIAN tails on BOTH sides, this can only absorb
# ASYMMETRY. A genuinely heavy-tailed target keeps its high k-hat here, which is
# the property that makes the rescue safe: it cannot launder a real tail problem
# into a clean verdict, only remove a mismatch the proposal itself created.
#
# Returns list(pareto_k, is_ess, refined = FALSE, lr) or NULL (degenerate
# proposal, too few finite importance draws). Does NOT manage the RNG (the
# driver saves / restores it once around all scoring).
.k_score_skew <- function(spec, n_samples, u_c, L_v, mom,
                          tail_points = NULL) {
    dp <- .sn_prop_from_moments(mu = mom$mu, sd = mom$sd, skew = mom$skew)
    if (is.null(dp)) return(NULL)

    lt  <- spec$lt
    n_s <- as.integer(n_samples)
    Z   <- .sn_prop_rand(n_s, dp)
    U   <- sweep(Z %*% t(L_v), 2L, u_c, `+`)

    ltv <- lt(U)
    if (length(ltv) != n_s) return(NULL)
    lr  <- ltv - .sn_prop_logpdf(Z, dp)
    fin <- is.finite(lr)
    if (sum(fin) < .PSIS_MIN_EVAL) return(NULL)
    ps <- tulpa_psis(lr[fin], tail_points = tail_points)
    if (!is.finite(ps$pareto_k)) return(NULL)
    list(pareto_k = ps$pareto_k, is_ess = ps$is_ess, refined = FALSE,
         lr = lr[fin])
}

# Skew-normal rescue pass. Runs after the Gaussian / mixture
# dispatch has chosen a proposal, and only when that choice is still above the
# good band -- so a fit whose Gaussian proposal already fits pays nothing.
#
# The target's whitened skewness is estimated from the best SINGLE-Gaussian
# pass's own draws and PSIS weights (`g`, always carrying a centre and scale,
# unlike the mixture), so the estimate is free and is automatically located and
# scaled where the target actually is. It is REPORTED whether or not the rescue
# is adopted: on a collapsed grid it is the EXPLANATION for a high k-hat, which
# is what a downstream bare-k threshold is missing.
#
# Adopts the skew-normal only when it gives a strictly lower k-hat -- the same
# rule the mixture rescue follows. A proposal that covers the target better
# reports the target's true tail behaviour; one that covers worse is discarded.
# Returns the (possibly replaced) `list(best, source)` with `outer_skew`
# attached.
.k_skew_rescue <- function(chosen, g, spec, n_samples, tail_points = NULL) {
    if (is.null(chosen) || is.null(g) || is.null(g$prop_u) || is.null(g$prop_L)) {
        return(chosen)
    }
    k_now <- chosen$best$pareto_k %||% NA_real_
    if (!is.finite(k_now) || k_now <= .K_DIAG_GOOD) return(chosen)

    mom <- .k_wtd_moments(g$U, g$log_weights, g$prop_u, g$prop_L)
    if (is.null(mom)) return(chosen)
    chosen$outer_skew <- stats::setNames(mom$skew, spec$names)
    # Engage only on skewness that is BOTH materially large and distinguishable
    # from zero at this many effective draws -- see .K_DIAG_SKEW_Z.
    gate <- max(.K_DIAG_SKEW_MIN, .K_DIAG_SKEW_Z * .skew_se(mom$n_eff))
    if (max(abs(mom$skew)) < gate) {
        return(chosen)                     # symmetric enough: nothing to correct
    }

    sk <- tryCatch(.k_score_skew(spec, n_samples, g$prop_u, g$prop_L, mom,
                                 tail_points = tail_points),
                   error = function(e) NULL)
    if (is.null(sk) || !is.finite(sk$pareto_k) || sk$pareto_k >= k_now) return(chosen)
    list(best = sk, source = "skew_normal", outer_skew = chosen$outer_skew)
}

# Dispatch the outer Pareto-k scoring to the right proposal.
# Always scores the single-Gaussian proposal (the grid-moment / mode-Hessian / CCD
# Gaussian with moment-matching refinement), whose tails extend beyond the grid so
# it detects a target heavier than the grid represents (the grid-width signal). On
# a spread-tensor `grid_moment` grid it ALSO scores the grid-mixture proposal --
# the faithful representation of what the engine samples (a local bump at each grid
# cell, mixed by the grid weights), which covers a skewed / multi-node posterior
# the single Gaussian underweights WITHIN the grid. The mixture covers only the
# grid (its bumps reach a few SDs past the outer nodes), so it is adopted only when
# (1) the single Gaussian's importance weight is essentially all inside that
# coverage hull -- the grid covers the posterior, so a high single-Gaussian k-hat
# is a within-grid shape mismatch -- AND (2) the mixture gives a lower k-hat.
# Otherwise the single Gaussian stands: a target tail beyond the grid keeps its
# (higher) k-hat so the grid-width deficiency is flagged, and a near-collapsed grid
# (where the mixture's few bumps cover worse) keeps the moment-matched Gaussian. A
# true delta collapse and a supplied CCD proposal are not `grid_moment`, so only
# the single Gaussian is scored. Returns list(best, source) or NULL. The
# SYMMETRIC half of the dispatch; `.k_dispatch` runs the
# skew-normal rescue on top of whatever this chooses.
.k_score_symmetric <- function(spec, g, n_samples, tail_points = NULL) {
    g_src <- if (!is.null(g) && isTRUE(g$refined)) "moment_matched"
             else spec$proposal_source
    g_out <- if (is.null(g)) NULL else list(best = g, source = g_src)

    if (!identical(spec$proposal_source, "grid_moment")) return(g_out)
    gm     <- g$gm %||% g
    gm_k   <- if (is.list(gm)) gm$pareto_k %||% NA_real_ else NA_real_
    gm_out <- if (is.list(gm)) list(best = gm, source = "grid_moment") else g_out

    # The skip is judged on the GRID-MOMENT k (what the engine samples), not the
    # moment-matching-widened single Gaussian: when the
    # grid-moment proposal is already good the verdict cannot improve, so skip the
    # mixture's `diagnose_draws`-draw evaluation. The rescue still runs for any
    # grid-moment k in the ok / unreliable band.
    if (is.finite(gm_k) && gm_k < .K_DIAG_GOOD) return(g_out)

    mix <- .k_score_mixture(spec, n_samples, tail_points = tail_points)
    if (is.null(mix)) return(gm_out)
    if (is.null(g))   return(list(best = mix, source = "grid_mixture"))

    # The mixture's coverage hull: the kept-cell node range expanded by a few bump
    # SDs. The check reads the first-pass GRID-MOMENT draws (`gm_U`), whose spread
    # is the posterior's, so the decision does not depend on how moment matching
    # happened to widen the single Gaussian on a given seed.
    gm_U  <- g$gm_U %||% g$U
    gm_lw <- g$gm_lw %||% g$log_weights
    out_frac <- .k_hull_weight(
        gm_U, gm_lw,
        mix$lo - .K_DIAG_HULL_PAD * mix$s, mix$hi + .K_DIAG_HULL_PAD * mix$s)
    covered  <- is.finite(out_frac) && out_frac <= .K_DIAG_HULL_TOL

    # Compare the mixture to the GRID-MOMENT single Gaussian (`gm_k`), not the
    # moment-matching-refined one: a refined k that dropped below
    # the mixture only by widening the single Gaussian past the grid is not a
    # faithful within-grid reading, so it must not win. Adopt the mixture (the
    # faithful within-grid proposal) when the grid covers the posterior AND the
    # mixture IMPROVES on the grid-moment proposal. A mixture that does NOT improve
    # on it is degenerate -- a near-collapsed grid where the few
    # bumps cover worse than the moment-matched Gaussian -- so the moment-matched
    # single Gaussian stands. For a target heavier / wider than the grid the
    # mixture is confined to the grid and reads unreliable, correctly flagging the
    # grid-width deficiency the escaped single Gaussian masked.
    improves <- is.finite(mix$pareto_k) && is.finite(gm_k) && mix$pareto_k < gm_k
    if (covered && improves) list(best = mix, source = "grid_mixture") else g_out
}

# Full outer Pareto-k proposal dispatch: the symmetric choice above (grid-moment
# / mode-Hessian / moment-matched Gaussian, or the grid mixture where the grid
# is spread and covers the posterior), then the skew-normal rescue
# on top of it. The rescue is last because it is the most
# expensive and the least often needed: it fires only when the symmetric choice
# is still above the good band, which is exactly the collapsed-grid regime where
# the mixture cannot help. Returns list(best, source[, outer_skew]), or a
# `.k_decline()` separating "the grid pins every axis, so
# there is no direction to sample along" from "the proposal could not be built
# or its GPD shape came back non-finite". Shared by the joint k and the per-arm k.
.k_dispatch <- function(spec, n_samples, tail_points = NULL) {
    if (is.null(spec) || !nrow(spec$Su)) return(.k_decline("no_varying_axis"))
    # Below the GPD-fit floor no candidate can return a number, and the mixture
    # would sample and evaluate the (expensive) target before finding that out,
    # so the floor is read once here rather than in each scorer.
    if (as.integer(n_samples) < .PSIS_MIN_EVAL) return(.k_decline("draws_too_few"))
    g <- .k_score_gaussian(spec, n_samples, tail_points = tail_points)
    chosen <- .k_score_symmetric(spec, g, n_samples, tail_points = tail_points)
    out <- .k_skew_rescue(chosen, g, spec, n_samples, tail_points = tail_points)
    if (is.null(out)) return(.k_decline("degenerate_proposal"))
    # The FIRST pass's k-hat -- the proposal exactly as the backend placed it,
    # before any candidate refined or replaced it. Reported alongside the chosen
    # k because the two answer different questions: the chosen one is how
    # reliable the integration is, the first pass is whether the placement
    # itself was any good. A large gap says the nodes are badly scaled around
    # the posterior even though the verdict is fine (gcol33/tulpa#630 measured
    # 15-49 falling to 0.3-0.8 on a small-group binary RE-covariance fit); no
    # gap says the placement was already right.
    gm <- (g$gm %||% g)
    out$first_pass_k <- if (is.list(gm)) gm$pareto_k %||% NA_real_ else NA_real_
    out
}